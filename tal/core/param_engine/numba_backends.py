from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows
from tal.utils.numba_support import njit_kernel, require_numba

_METHOD_NEAREST = 0
_METHOD_LINEAR = 1
_STATUS_OK = 0
_STATUS_MONOTONIC = 1
_STATUS_DUPLICATE = 2

_MAP_MONOTONIC_ERROR = "build_param_map: parameter coordinate must be monotonic non-decreasing on valid domain."
_BOUNDS_MONOTONIC_ERROR = (
    "build_param_bounds_map: parameter coordinate must be monotonic non-decreasing on valid domain."
)
_DUPLICATE_BRACKET_ERROR = (
    "build_param_map: duplicate parameter bracket encountered for linear interpolation."
)
_HELPERS_JITTED = False


@lru_cache(maxsize=1)
def _compiled_map_block():
    numba = require_numba("build_param_map")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _map_block_impl)


@lru_cache(maxsize=1)
def _compiled_bounds_block():
    numba = require_numba("build_param_bounds_map")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _bounds_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _HELPERS_JITTED
    global _fill_source, _map_linear_interior, _map_linear_row, _map_nearest_row
    global _search_left, _search_right, _write_bounds_edge, _write_bounds_empty
    global _write_bounds_span, _write_constant, _write_duplicate
    if _HELPERS_JITTED:
        return
    _fill_source = njit_kernel(numba, _fill_source)
    _search_left = njit_kernel(numba, _search_left)
    _search_right = njit_kernel(numba, _search_right)
    _write_constant = njit_kernel(numba, _write_constant)
    _write_duplicate = njit_kernel(numba, _write_duplicate)
    _map_nearest_row = njit_kernel(numba, _map_nearest_row)
    _map_linear_interior = njit_kernel(numba, _map_linear_interior)
    _map_linear_row = njit_kernel(numba, _map_linear_row)
    _write_bounds_empty = njit_kernel(numba, _write_bounds_empty)
    _write_bounds_edge = njit_kernel(numba, _write_bounds_edge)
    _write_bounds_span = njit_kernel(numba, _write_bounds_span)
    _HELPERS_JITTED = True


def _method_code(method: str) -> int:
    if method == "nearest":
        return _METHOD_NEAREST
    if method == "linear":
        return _METHOD_LINEAR
    raise ValueError(f"build_param_map: method must be 'nearest' or 'linear', got {method!r}.")


def _broadcast_map_blocks(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (param_block, valid_block, query_block),
        (
            BlockInputSpec("param", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("query", 1, np.float64),
        ),
        output_core_shape=(),
        owner="build_param_map",
    )
    param_rows, valid_rows, query_rows = prepared.row_arrays
    seq_size = int(param_rows.shape[-1])
    query_size = int(query_rows.shape[-1])
    if int(valid_rows.shape[-1]) != seq_size:
        raise ValueError("build_param_map: valid block trailing dimension must match param block.")
    return param_rows, valid_rows, query_rows, prepared.outer_shape + (query_size,)


def _broadcast_bounds_blocks(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    start_block: np.ndarray,
    stop_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (param_block, valid_block, start_block, stop_block),
        (
            BlockInputSpec("param", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("start", 0, np.float64),
            BlockInputSpec("stop", 0, np.float64),
        ),
        output_core_shape=(),
        owner="build_param_bounds_map",
    )
    param_rows, valid_rows, start_rows, stop_rows = prepared.row_arrays
    seq_size = int(param_rows.shape[-1])
    if int(valid_rows.shape[-1]) != seq_size:
        raise ValueError("build_param_bounds_map: valid block trailing dimension must match param block.")
    return param_rows, valid_rows, start_rows, stop_rows, prepared.outer_shape


def _raise_map_status(status: int) -> None:
    if status == _STATUS_MONOTONIC:
        raise ValueError(_MAP_MONOTONIC_ERROR)
    if status == _STATUS_DUPLICATE:
        raise ValueError(_DUPLICATE_BRACKET_ERROR)


def _raise_bounds_status(status: int) -> None:
    if status == _STATUS_MONOTONIC:
        raise ValueError(_BOUNDS_MONOTONIC_ERROR)


def map_block_numba(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require_numba("build_param_map")
    method_code = _method_code(method)
    param_rows, valid_rows, query_rows, output_shape = _broadcast_map_blocks(param_block, valid_block, query_block)
    i0, i1, alpha, valid, status = _compiled_map_block()(param_rows, valid_rows, query_rows, method_code, int(dup_code))
    _raise_map_status(int(status))
    return i0.reshape(output_shape), i1.reshape(output_shape), alpha.reshape(output_shape), valid.reshape(output_shape)


def bounds_block_numba(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    start_block: np.ndarray,
    stop_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    require_numba("build_param_bounds_map")
    param_rows, valid_rows, start_rows, stop_rows, output_shape = _broadcast_bounds_blocks(
        param_block,
        valid_block,
        start_block,
        stop_block,
    )
    i0, i1, status = _compiled_bounds_block()(param_rows, valid_rows, start_rows, stop_rows)
    _raise_bounds_status(int(status))
    return i0.reshape(output_shape), i1.reshape(output_shape)


def _fill_source(param_row, valid_row, src_idx, src_vals):
    count = 0
    previous = 0.0
    have_previous = False
    for idx in range(param_row.shape[0]):
        value = param_row[idx]
        if valid_row[idx] and np.isfinite(value):
            if have_previous and value < previous:
                return count, _STATUS_MONOTONIC
            src_idx[count] = idx
            src_vals[count] = value
            previous = value
            have_previous = True
            count += 1
    return count, _STATUS_OK


def _search_left(values, count, query):
    lo = 0
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < query:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _search_right(values, count, query):
    lo = 0
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] <= query:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _write_constant(row, col, pick, src_idx, i0, i1, valid):
    source_index = src_idx[pick]
    i0[row, col] = source_index
    i1[row, col] = source_index
    valid[row, col] = True


def _write_duplicate(row, col, dup_code, left_pick, right_pick, src_idx, i0, i1, valid):
    if dup_code == 3:
        return _STATUS_DUPLICATE
    if dup_code == 0:
        return _STATUS_OK
    pick = left_pick
    if dup_code != 1:
        pick = right_pick
    _write_constant(row, col, pick, src_idx, i0, i1, valid)
    return _STATUS_OK


def _map_nearest_row(row, src_idx, src_vals, count, query_row, i0, i1, valid):
    for col in range(query_row.shape[0]):
        query = query_row[col]
        if not np.isfinite(query):
            continue
        right = _search_left(src_vals, count, query)
        pick = 0
        if right >= count:
            pick = count - 1
        elif right > 0:
            left = right - 1
            if abs(query - src_vals[left]) <= abs(src_vals[right] - query):
                pick = left
            else:
                pick = right
        _write_constant(row, col, pick, src_idx, i0, i1, valid)


def _map_linear_interior(row, col, query, lo, dup_code, src_idx, src_vals, i0, i1, alpha, valid):
    left = lo - 1
    right = lo
    t0 = src_vals[left]
    t1 = src_vals[right]
    if t1 == t0:
        return _write_duplicate(row, col, dup_code, left, right, src_idx, i0, i1, valid)
    i0[row, col] = src_idx[left]
    i1[row, col] = src_idx[right]
    alpha[row, col] = (query - t0) / (t1 - t0)
    valid[row, col] = True
    return _STATUS_OK


def _map_linear_row(row, src_idx, src_vals, count, query_row, dup_code, i0, i1, alpha, valid):
    for col in range(query_row.shape[0]):
        query = query_row[col]
        if not np.isfinite(query):
            continue
        lo = _search_left(src_vals, count, query)
        hi = _search_right(src_vals, count, query)
        if hi - lo > 1:
            status = _write_duplicate(row, col, dup_code, lo, hi - 1, src_idx, i0, i1, valid)
        elif (lo == 0 and query == src_vals[0]) or (lo >= count and query == src_vals[count - 1]):
            _write_constant(row, col, 0 if lo == 0 else count - 1, src_idx, i0, i1, valid)
            status = _STATUS_OK
        elif lo > 0 and lo < count:
            status = _map_linear_interior(row, col, query, lo, dup_code, src_idx, src_vals, i0, i1, alpha, valid)
        else:
            status = _STATUS_OK
        if status != _STATUS_OK:
            return status
    return _STATUS_OK


def _map_block_impl(param, valid_in, query, method_code, dup_code):
    rows = param.shape[0]
    seq_size = param.shape[1]
    query_size = query.shape[1]
    i0 = np.zeros((rows, query_size), dtype=np.int64)
    i1 = np.zeros((rows, query_size), dtype=np.int64)
    alpha = np.zeros((rows, query_size), dtype=np.float64)
    valid = np.zeros((rows, query_size), dtype=np.bool_)
    src_idx = np.empty(seq_size, dtype=np.int64)
    src_vals = np.empty(seq_size, dtype=np.float64)
    for row in range(rows):
        count, status = _fill_source(param[row], valid_in[row], src_idx, src_vals)
        if status != _STATUS_OK:
            return i0, i1, alpha, valid, status
        if count == 0:
            continue
        if method_code == _METHOD_NEAREST:
            _map_nearest_row(row, src_idx, src_vals, count, query[row], i0, i1, valid)
        else:
            status = _map_linear_row(row, src_idx, src_vals, count, query[row], dup_code, i0, i1, alpha, valid)
            if status != _STATUS_OK:
                return i0, i1, alpha, valid, status
    return i0, i1, alpha, valid, _STATUS_OK


def _write_bounds_empty(row, i0, i1):
    i0[row] = 0
    i1[row] = 0


def _write_bounds_edge(row, lo, count, row_len, src_idx, i0, i1):
    edge = row_len
    if lo < count:
        edge = src_idx[lo]
    if edge < 0:
        edge = 0
    if edge > row_len:
        edge = row_len
    i0[row] = edge
    i1[row] = edge


def _write_bounds_span(row, lo, hi, count, row_len, src_idx, i0, i1):
    left = lo
    right = hi - 1
    if left >= count:
        left = count - 1
    if right >= count:
        right = count - 1
    start = src_idx[left]
    stop = src_idx[right] + 1
    i0[row] = start if start < row_len else row_len
    i1[row] = stop if stop < row_len else row_len


def _bounds_block_impl(param, valid_in, start, stop):
    rows = param.shape[0]
    seq_size = param.shape[1]
    i0 = np.zeros(rows, dtype=np.int64)
    i1 = np.zeros(rows, dtype=np.int64)
    src_idx = np.empty(seq_size, dtype=np.int64)
    src_vals = np.empty(seq_size, dtype=np.float64)
    for row in range(rows):
        count, status = _fill_source(param[row], valid_in[row], src_idx, src_vals)
        if status != _STATUS_OK:
            return i0, i1, status
        if count == 0 or np.isnan(start[row]) or np.isnan(stop[row]):
            _write_bounds_empty(row, i0, i1)
            continue
        lo = _search_left(src_vals, count, start[row])
        hi = _search_right(src_vals, count, stop[row])
        if hi <= lo:
            _write_bounds_edge(row, lo, count, seq_size, src_idx, i0, i1)
        else:
            _write_bounds_span(row, lo, hi, count, seq_size, src_idx, i0, i1)
    return i0, i1, _STATUS_OK


__all__ = ["bounds_block_numba", "map_block_numba"]
