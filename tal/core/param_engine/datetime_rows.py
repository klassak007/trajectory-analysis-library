from __future__ import annotations

import numpy as np

_DUPLICATE_BRACKET_ERROR = (
    "build_param_map: duplicate parameter bracket encountered for linear interpolation."
)
NAT_INT = np.datetime64("NaT", "ns").view("int64")
DATETIME_OPEN_START = np.datetime64("1677-09-21T00:12:43.145224193", "ns")
DATETIME_OPEN_STOP = np.datetime64("2262-04-11T23:47:16.854775807", "ns")
_DATETIME_OPEN_START_INT = DATETIME_OPEN_START.view("int64")
_DATETIME_OPEN_STOP_INT = DATETIME_OPEN_STOP.view("int64")


def _empty_map_result(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i0 = np.zeros(k, dtype="int64")
    i1 = np.zeros(k, dtype="int64")
    alpha = np.zeros(k, dtype="float64")
    valid = np.zeros(k, dtype=bool)
    return i0, i1, alpha, valid


def _datetime_ns(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").view("int64")


def _local_ns(values: np.ndarray, *, anchor: int, owner: str) -> np.ndarray:
    out = np.empty(values.size, dtype=np.int64)
    lo = int(np.iinfo(np.int64).min)
    hi = int(np.iinfo(np.int64).max)
    for idx, value in enumerate(values):
        delta = int(value) - anchor
        if delta < lo or delta > hi:
            raise ValueError(f"{owner}: datetime64 values span more than int64 nanoseconds from row anchor.")
        out[idx] = delta
    return out


def _datetime_source_row(param_row: np.ndarray, valid_row: np.ndarray, *, owner: str) -> tuple[np.ndarray, np.ndarray, int]:
    src_abs = _datetime_ns(param_row)
    valid = np.asarray(valid_row, dtype=bool) & (src_abs != NAT_INT)
    src_idx = np.flatnonzero(valid).astype("int64", copy=False)
    src_vals_abs = src_abs[src_idx]
    if src_vals_abs.size >= 2 and np.any(np.diff(src_vals_abs) < 0):
        raise ValueError(f"{owner}: parameter coordinate must be monotonic non-decreasing on valid domain.")
    anchor = int(src_vals_abs[0]) if src_vals_abs.size else 0
    return src_idx, _local_ns(src_vals_abs, anchor=anchor, owner=owner), anchor


def _datetime_query_row(query_row: np.ndarray, *, anchor: int) -> tuple[np.ndarray, np.ndarray]:
    query_abs = _datetime_ns(query_row)
    valid = query_abs != NAT_INT
    query_local = np.zeros(query_abs.shape, dtype=np.int64)
    if np.any(valid):
        query_local[valid] = _local_ns(query_abs[valid], anchor=anchor, owner="build_param_map")
    return query_local, valid


def _nearest_datetime_row(
    src_idx: np.ndarray,
    src: np.ndarray,
    query: np.ndarray,
    query_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i0, i1, alpha, valid = _empty_map_result(int(query.size))
    if src.size == 0 or not np.any(query_valid):
        return i0, i1, alpha, valid
    for out_idx in np.flatnonzero(query_valid):
        pick = _nearest_pick(src, int(query[out_idx]))
        sample = int(src_idx[pick])
        i0[out_idx] = sample
        i1[out_idx] = sample
        valid[out_idx] = True
    return i0, i1, alpha, valid


def _nearest_pick(src: np.ndarray, q: int) -> int:
    right = int(np.searchsorted(src, q, side="left"))
    if right <= 0:
        return 0
    if right >= src.size:
        return src.size - 1
    left = right - 1
    return left if q - int(src[left]) <= int(src[right]) - q else right


def _assign_constant(
    out_idx: int,
    pick: int,
    src_idx: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    valid: np.ndarray,
) -> None:
    sample = int(src_idx[pick])
    i0[out_idx] = sample
    i1[out_idx] = sample
    valid[out_idx] = True


def _try_duplicate(
    dup_code: int,
    out_idx: int,
    lo: int,
    hi: int,
    src_idx: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    valid: np.ndarray,
) -> bool:
    if hi - lo <= 1:
        return False
    if dup_code == 3:
        raise ValueError(_DUPLICATE_BRACKET_ERROR)
    if dup_code == 0:
        return True
    _assign_constant(out_idx, lo if dup_code == 1 else hi - 1, src_idx, i0, i1, valid)
    return True


def _try_endpoint(
    out_idx: int,
    q: int,
    lo: int,
    src: np.ndarray,
    src_idx: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    valid: np.ndarray,
) -> bool:
    if (lo == 0 and q == int(src[0])) or (lo >= src.size and q == int(src[-1])):
        _assign_constant(out_idx, 0 if lo == 0 else src.size - 1, src_idx, i0, i1, valid)
        return True
    return False


def _try_interior(
    dup_code: int,
    out_idx: int,
    q: int,
    lo: int,
    src: np.ndarray,
    src_idx: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
) -> None:
    if lo <= 0 or lo >= src.size:
        return
    left = lo - 1
    right = lo
    t0 = int(src[left])
    t1 = int(src[right])
    if t1 == t0:
        _try_duplicate(dup_code, out_idx, left, right + 1, src_idx, i0, i1, valid)
        return
    i0[out_idx] = int(src_idx[left])
    i1[out_idx] = int(src_idx[right])
    alpha[out_idx] = float(q - t0) / float(t1 - t0)
    valid[out_idx] = True


def _linear_datetime_row(
    src_idx: np.ndarray,
    src: np.ndarray,
    query: np.ndarray,
    query_valid: np.ndarray,
    *,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i0, i1, alpha, valid = _empty_map_result(int(query.size))
    if src.size == 0 or not np.any(query_valid):
        return i0, i1, alpha, valid
    for out_idx in np.flatnonzero(query_valid):
        q = int(query[out_idx])
        lo = int(np.searchsorted(src, q, side="left"))
        hi = int(np.searchsorted(src, q, side="right"))
        if _try_duplicate(dup_code, out_idx, lo, hi, src_idx, i0, i1, valid):
            continue
        if _try_endpoint(out_idx, q, lo, src, src_idx, i0, i1, valid):
            continue
        _try_interior(dup_code, out_idx, q, lo, src, src_idx, i0, i1, alpha, valid)
    return i0, i1, alpha, valid


def datetime_map_row(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    query_row: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_idx, src, anchor = _datetime_source_row(param_row, valid_row, owner="build_param_map")
    if src.size == 0:
        return _empty_map_result(int(np.asarray(query_row).size))
    query, query_valid = _datetime_query_row(query_row, anchor=anchor)
    if method == "nearest":
        return _nearest_datetime_row(src_idx, src, query, query_valid)
    return _linear_datetime_row(src_idx, src, query, query_valid, dup_code=dup_code)


def _datetime_bound_index(
    src_local: np.ndarray,
    value: int,
    *,
    anchor: int,
    side: str,
    owner: str,
) -> int:
    local = _local_ns(np.asarray([value], dtype=np.int64), anchor=anchor, owner=owner)[0]
    return int(np.searchsorted(src_local, local, side=side))


def datetime_bounds_row(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src_idx, src_local, anchor = _datetime_source_row(param_row, valid_row, owner="build_param_bounds_map")
    row_len = int(np.asarray(param_row).shape[0])
    if src_local.size == 0:
        return np.asarray(0, dtype="int64"), np.asarray(0, dtype="int64")
    start_ns = int(_datetime_ns(np.asarray(start).reshape(1))[0])
    stop_ns = int(_datetime_ns(np.asarray(stop).reshape(1))[0])
    if start_ns == NAT_INT or stop_ns == NAT_INT:
        return np.asarray(0, dtype="int64"), np.asarray(0, dtype="int64")
    lo = 0 if start_ns == _DATETIME_OPEN_START_INT else _datetime_bound_index(
        src_local,
        start_ns,
        anchor=anchor,
        side="left",
        owner="build_param_bounds_map",
    )
    hi = src_local.size if stop_ns == _DATETIME_OPEN_STOP_INT else _datetime_bound_index(
        src_local,
        stop_ns,
        anchor=anchor,
        side="right",
        owner="build_param_bounds_map",
    )
    if hi <= lo:
        edge = row_len if lo >= src_idx.size else int(src_idx[lo])
        edge = max(0, min(edge, row_len))
        return np.asarray(edge, dtype="int64"), np.asarray(edge, dtype="int64")
    i0 = int(src_idx[min(lo, src_idx.size - 1)])
    i1 = int(src_idx[min(hi - 1, src_idx.size - 1)] + 1)
    return np.asarray(min(i0, row_len), dtype="int64"), np.asarray(min(i1, row_len), dtype="int64")


__all__ = [
    "DATETIME_OPEN_START",
    "DATETIME_OPEN_STOP",
    "NAT_INT",
    "datetime_bounds_row",
    "datetime_map_row",
]
