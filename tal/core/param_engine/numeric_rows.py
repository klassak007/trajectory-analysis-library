"""Exact row kernels for ordered real numeric parameter domains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ordered_dtypes import is_float64_exact_integer, is_integral_dtype, is_ordered_real_numeric_dtype

_DUPLICATE_BRACKET_ERROR = (
    "build_param_map: duplicate parameter bracket encountered for linear interpolation."
)
_UNSAFE_MIXED_ERROR = (
    "cannot be represented exactly as float64; use matching integer parameter/query "
    "dtypes or rescale the parameter domain"
)


@dataclass(frozen=True)
class _SourceRow:
    indexes: np.ndarray
    values: tuple[object, ...]
    integral: bool
    row_len: int


@dataclass
class _MapBuffers:
    i0: np.ndarray
    i1: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


def _empty_map_result(size: int) -> _MapBuffers:
    return _MapBuffers(
        i0=np.zeros(size, dtype="int64"),
        i1=np.zeros(size, dtype="int64"),
        alpha=np.zeros(size, dtype="float64"),
        valid=np.zeros(size, dtype=bool),
    )


def _scalar(value: object) -> object:
    return np.asarray(value).reshape(()).item()


def _is_finite(value: object) -> bool:
    return bool(np.isfinite(value))


def _source_row(param_row: np.ndarray, valid_row: np.ndarray, *, owner: str) -> _SourceRow:
    param = np.asarray(param_row)
    if not is_ordered_real_numeric_dtype(param.dtype):
        raise ValueError(f"{owner}: param coordinate must have an ordered real numeric dtype.")
    mask = np.asarray(valid_row, dtype=bool)
    if param.dtype.kind == "f":
        mask = mask & np.isfinite(param)
    indexes = np.flatnonzero(mask).astype("int64", copy=False)
    integral = is_integral_dtype(param.dtype)
    values = tuple(int(param[index]) if integral else _scalar(param[index]) for index in indexes)
    if any(values[index] < values[index - 1] for index in range(1, len(values))):
        raise ValueError(f"{owner}: parameter coordinate must be monotonic non-decreasing on valid domain.")
    return _SourceRow(indexes=indexes, values=values, integral=integral, row_len=int(param.shape[0]))


def _require_float64_safe_integral_source(source: _SourceRow, *, owner: str) -> None:
    for value in source.values:
        if is_float64_exact_integer(value):
            continue
        raise ValueError(
            f"{owner}: integer parameter value {int(value)!r} {_UNSAFE_MIXED_ERROR}."
        )


def _operand_value(
    value: object,
    *,
    kind: str,
    source_integral: bool,
    owner: str,
    allow_infinite: bool = False,
) -> object | None:
    if kind == "f":
        out = _scalar(value)
        if bool(np.isnan(out)):
            return None
        return out if allow_infinite or _is_finite(out) else None
    out = int(value)
    if source_integral:
        return out
    if not is_float64_exact_integer(out):
        raise ValueError(f"{owner}: integer operand {out!r} {_UNSAFE_MIXED_ERROR}.")
    return float(out)


def _bisect_left(values: tuple[object, ...], target: object) -> int:
    low, high = 0, len(values)
    while low < high:
        mid = (low + high) // 2
        if values[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


def _bisect_right(values: tuple[object, ...], target: object) -> int:
    low, high = 0, len(values)
    while low < high:
        mid = (low + high) // 2
        if target < values[mid]:
            high = mid
        else:
            low = mid + 1
    return low


def _float_work_arrays(*values: np.ndarray) -> tuple[np.ndarray, ...]:
    dtype = np.result_type(*(np.asarray(value).dtype for value in values))
    work_dtype = np.dtype("float64") if dtype.itemsize <= 8 else dtype
    return tuple(np.asarray(value, dtype=work_dtype) for value in values)


def _assign_float_picks(
    buffers: _MapBuffers,
    row_indexes: np.ndarray,
    mask: np.ndarray,
    source_picks: np.ndarray,
    source_indexes: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    output_indexes = row_indexes[mask]
    picked = source_indexes[source_picks[mask]].astype("int64", copy=False)
    buffers.i0[output_indexes] = picked
    buffers.i1[output_indexes] = picked
    buffers.valid[output_indexes] = True


def _apply_float_duplicate(
    buffers: _MapBuffers,
    *,
    row_indexes: np.ndarray,
    mask: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    source_indexes: np.ndarray,
    dup_code: int,
) -> None:
    if not np.any(mask):
        return
    if dup_code == 3:
        raise ValueError(_DUPLICATE_BRACKET_ERROR)
    if dup_code in {1, 2}:
        picks = left if dup_code == 1 else right
        _assign_float_picks(buffers, row_indexes, mask, picks, source_indexes)


def _float_nearest_row(
    source_indexes: np.ndarray,
    source: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    buffers = _empty_map_result(int(query.size))
    finite = np.isfinite(query)
    if source.size == 0 or not np.any(finite):
        return buffers.i0, buffers.i1, buffers.alpha, buffers.valid
    query_finite = query[finite]
    right = np.searchsorted(source, query_finite, side="left")
    right_pick = np.clip(right, 0, source.size - 1)
    left_pick = np.clip(right - 1, 0, source.size - 1)
    interior = (right > 0) & (right < source.size)
    left_distance = np.abs(query_finite[interior] - source[left_pick[interior]])
    right_distance = np.abs(source[right_pick[interior]] - query_finite[interior])
    right_pick[interior] = np.where(left_distance <= right_distance, left_pick[interior], right_pick[interior])
    picks = source_indexes[right_pick].astype("int64", copy=False)
    output_indexes = np.flatnonzero(finite)
    buffers.i0[output_indexes] = picks
    buffers.i1[output_indexes] = picks
    buffers.valid[output_indexes] = True
    return buffers.i0, buffers.i1, buffers.alpha, buffers.valid


def _apply_float_interior(
    buffers: _MapBuffers,
    *,
    query: np.ndarray,
    low: np.ndarray,
    duplicate: np.ndarray,
    output_indexes: np.ndarray,
    source: np.ndarray,
    source_indexes: np.ndarray,
    dup_code: int,
) -> None:
    interior = (~duplicate) & (low > 0) & (low < source.size)
    if not np.any(interior):
        return
    left, right = low[interior] - 1, low[interior]
    start, stop = source[left], source[right]
    inner_indexes = output_indexes[interior]
    degenerate = stop == start
    _apply_float_duplicate(
        buffers,
        row_indexes=inner_indexes,
        mask=degenerate,
        left=left,
        right=right,
        source_indexes=source_indexes,
        dup_code=dup_code,
    )
    interpolate = ~degenerate
    selected = inner_indexes[interpolate]
    left_selected, right_selected = left[interpolate], right[interpolate]
    buffers.i0[selected] = source_indexes[left_selected]
    buffers.i1[selected] = source_indexes[right_selected]
    buffers.alpha[selected] = (query[interior][interpolate] - start[interpolate]) / (
        stop[interpolate] - start[interpolate]
    )
    buffers.valid[selected] = True


def _float_linear_row(
    source_indexes: np.ndarray,
    source: np.ndarray,
    query: np.ndarray,
    *,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    buffers = _empty_map_result(int(query.size))
    finite = np.isfinite(query)
    if source.size == 0 or not np.any(finite):
        return buffers.i0, buffers.i1, buffers.alpha, buffers.valid
    query_finite = query[finite]
    low = np.searchsorted(source, query_finite, side="left")
    high = np.searchsorted(source, query_finite, side="right")
    output_indexes = np.flatnonzero(finite)
    duplicate = (high - low) > 1
    _apply_float_duplicate(
        buffers,
        row_indexes=output_indexes,
        mask=duplicate,
        left=low,
        right=high - 1,
        source_indexes=source_indexes,
        dup_code=dup_code,
    )
    endpoint = (~duplicate) & (
        ((low == 0) & (query_finite == source[0]))
        | ((low >= source.size) & (query_finite == source[-1]))
    )
    _assign_float_picks(buffers, output_indexes, endpoint, np.where(low == 0, 0, source.size - 1), source_indexes)
    _apply_float_interior(
        buffers,
        query=query_finite,
        low=low,
        duplicate=duplicate,
        output_indexes=output_indexes,
        source=source,
        source_indexes=source_indexes,
        dup_code=dup_code,
    )
    return buffers.i0, buffers.i1, buffers.alpha, buffers.valid


def _float_map_row(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    query_row: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_values, query = _float_work_arrays(param_row, query_row)
    valid = np.asarray(valid_row, dtype=bool) & np.isfinite(source_values)
    source_indexes = np.flatnonzero(valid).astype("int64", copy=False)
    source = source_values[source_indexes]
    if source.size >= 2 and np.any(np.diff(source) < 0):
        raise ValueError("build_param_map: parameter coordinate must be monotonic non-decreasing on valid domain.")
    if method == "nearest":
        return _float_nearest_row(source_indexes, source, query)
    return _float_linear_row(source_indexes, source, query, dup_code=dup_code)


def _assign_constant(buffers: _MapBuffers, *, out_index: int, source: _SourceRow, pick: int) -> None:
    sample = int(source.indexes[pick])
    buffers.i0[out_index] = sample
    buffers.i1[out_index] = sample
    buffers.valid[out_index] = True


def _apply_duplicate(
    buffers: _MapBuffers,
    *,
    out_index: int,
    source: _SourceRow,
    low: int,
    high: int,
    dup_code: int,
) -> bool:
    if high - low <= 1:
        return False
    if dup_code == 3:
        raise ValueError(_DUPLICATE_BRACKET_ERROR)
    if dup_code in {1, 2}:
        _assign_constant(buffers, out_index=out_index, source=source, pick=low if dup_code == 1 else high - 1)
    return True


def _apply_linear_value(
    buffers: _MapBuffers,
    *,
    out_index: int,
    source: _SourceRow,
    query: object,
    low: int,
    dup_code: int,
) -> None:
    high = _bisect_right(source.values, query)
    if _apply_duplicate(buffers, out_index=out_index, source=source, low=low, high=high, dup_code=dup_code):
        return
    if (low == 0 and query == source.values[0]) or (low == len(source.values) and query == source.values[-1]):
        _assign_constant(buffers, out_index=out_index, source=source, pick=0 if low == 0 else len(source.values) - 1)
        return
    if low <= 0 or low >= len(source.values):
        return
    left, right = low - 1, low
    start, stop = source.values[left], source.values[right]
    if stop == start:
        _apply_duplicate(buffers, out_index=out_index, source=source, low=left, high=right + 1, dup_code=dup_code)
        return
    buffers.i0[out_index] = int(source.indexes[left])
    buffers.i1[out_index] = int(source.indexes[right])
    buffers.alpha[out_index] = float((query - start) / (stop - start))
    buffers.valid[out_index] = True


def _apply_nearest_value(
    buffers: _MapBuffers,
    *,
    out_index: int,
    source: _SourceRow,
    query: object,
    low: int,
) -> None:
    if low <= 0:
        pick = 0
    elif low >= len(source.values):
        pick = len(source.values) - 1
    else:
        left = low - 1
        pick = left if query - source.values[left] <= source.values[low] - query else low
    _assign_constant(buffers, out_index=out_index, source=source, pick=pick)


def numeric_map_row(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    query_row: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.asarray(param_row).dtype.kind == "f" and np.asarray(query_row).dtype.kind == "f":
        return _float_map_row(param_row, valid_row, query_row, method=method, dup_code=dup_code)
    source = _source_row(param_row, valid_row, owner="build_param_map")
    query = np.asarray(query_row)
    buffers = _empty_map_result(int(query.size))
    if not source.values:
        return buffers.i0, buffers.i1, buffers.alpha, buffers.valid
    float_source_checked = False
    for out_index, raw in enumerate(query):
        value = _operand_value(raw, kind=query.dtype.kind, source_integral=source.integral, owner="build_param_map")
        if value is None:
            continue
        if source.integral and query.dtype.kind == "f" and not float_source_checked:
            _require_float64_safe_integral_source(source, owner="build_param_map")
            float_source_checked = True
        low = _bisect_left(source.values, value)
        if method == "nearest":
            _apply_nearest_value(buffers, out_index=out_index, source=source, query=value, low=low)
        else:
            _apply_linear_value(
                buffers,
                out_index=out_index,
                source=source,
                query=value,
                low=low,
                dup_code=dup_code,
            )
    return buffers.i0, buffers.i1, buffers.alpha, buffers.valid


def _float_bounds_row(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_values, start_value, stop_value = _float_work_arrays(param_row, start, stop)
    valid = np.asarray(valid_row, dtype=bool) & np.isfinite(source_values)
    source_indexes = np.flatnonzero(valid).astype("int64", copy=False)
    source = source_values[source_indexes]
    if source.size >= 2 and np.any(np.diff(source) < 0):
        raise ValueError(
            "build_param_bounds_map: parameter coordinate must be monotonic non-decreasing on valid domain."
        )
    if source.size == 0 or np.isnan(start_value).any() or np.isnan(stop_value).any():
        return np.asarray(0, dtype="int64"), np.asarray(0, dtype="int64")
    low = int(np.searchsorted(source, start_value.reshape(()).item(), side="left"))
    high = int(np.searchsorted(source, stop_value.reshape(()).item(), side="right"))
    if high <= low:
        edge = int(source_values.size) if low >= source_indexes.size else int(source_indexes[low])
        edge = max(0, min(edge, int(source_values.size)))
        return np.asarray(edge, dtype="int64"), np.asarray(edge, dtype="int64")
    i0 = int(source_indexes[min(low, source_indexes.size - 1)])
    i1 = int(source_indexes[min(high - 1, source_indexes.size - 1)] + 1)
    return (
        np.asarray(min(i0, source_values.size), dtype="int64"),
        np.asarray(min(i1, source_values.size), dtype="int64"),
    )


def numeric_bounds_row(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if all(np.asarray(value).dtype.kind == "f" for value in (param_row, start, stop)):
        return _float_bounds_row(param_row, valid_row, start, stop)
    source = _source_row(param_row, valid_row, owner="build_param_bounds_map")
    if not source.values:
        return np.asarray(0, dtype="int64"), np.asarray(0, dtype="int64")
    start_arr, stop_arr = np.asarray(start), np.asarray(stop)
    start_value = _operand_value(
        start_arr.reshape(()),
        kind=start_arr.dtype.kind,
        source_integral=source.integral,
        owner="build_param_bounds_map",
        allow_infinite=True,
    )
    stop_value = _operand_value(
        stop_arr.reshape(()),
        kind=stop_arr.dtype.kind,
        source_integral=source.integral,
        owner="build_param_bounds_map",
        allow_infinite=True,
    )
    if start_value is None or stop_value is None:
        return np.asarray(0, dtype="int64"), np.asarray(0, dtype="int64")
    finite_float_bound = any(
        value.dtype.kind == "f" and bool(np.isfinite(value).all())
        for value in (start_arr, stop_arr)
    )
    if source.integral and finite_float_bound:
        _require_float64_safe_integral_source(source, owner="build_param_bounds_map")
    low = _bisect_left(source.values, start_value)
    high = _bisect_right(source.values, stop_value)
    if high <= low:
        edge = source.row_len if low >= len(source.indexes) else int(source.indexes[low])
        edge = max(0, min(edge, source.row_len))
        return np.asarray(edge, dtype="int64"), np.asarray(edge, dtype="int64")
    i0 = int(source.indexes[min(low, len(source.indexes) - 1)])
    i1 = int(source.indexes[min(high - 1, len(source.indexes) - 1)] + 1)
    return np.asarray(min(i0, source.row_len), dtype="int64"), np.asarray(min(i1, source.row_len), dtype="int64")


__all__ = ["numeric_bounds_row", "numeric_map_row"]
