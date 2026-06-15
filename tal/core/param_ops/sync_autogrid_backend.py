from __future__ import annotations

"""Row-local backend helpers for synchronized auto-grid synthesis."""

from collections.abc import Sequence
from typing import Literal

import numpy as np

_NAT_INT = np.datetime64("NaT", "ns").view("int64")


def _dedup_sorted(values: np.ndarray, *, tol: float) -> np.ndarray:
    """Collapse sorted values within tolerance using last-kept semantics."""
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return arr
    out = np.empty(arr.size, dtype="float64")
    out[0] = arr[0]
    count = 1
    for value in arr[1:]:
        if abs(float(value) - out[count - 1]) > tol:
            out[count] = float(value)
            count += 1
    return out[:count]


def _dedup_sorted_int(values: np.ndarray, *, tol: int) -> np.ndarray:
    arr = np.asarray(values, dtype="int64")
    if arr.size == 0:
        return arr
    out = np.empty(arr.size, dtype="int64")
    out[0] = arr[0]
    count = 1
    for value in arr[1:]:
        if abs(int(value) - int(out[count - 1])) > tol:
            out[count] = int(value)
            count += 1
    return out[:count]


def _merge_union(a: np.ndarray, b: np.ndarray, *, tol: float) -> np.ndarray:
    a2 = _dedup_sorted(a, tol=tol)
    b2 = _dedup_sorted(b, tol=tol)
    if a2.size == 0:
        return b2
    if b2.size == 0:
        return a2
    pos = np.searchsorted(a2, b2, side="left")
    near_right = (pos < a2.size) & (np.abs(a2[np.minimum(pos, a2.size - 1)] - b2) <= tol)
    near_left = (pos > 0) & (np.abs(b2 - a2[pos - 1]) <= tol)
    b_keep = b2[~(near_left | near_right)]
    if b_keep.size == 0:
        return a2
    out = np.concatenate([a2, b_keep], axis=0)
    out.sort(kind="stable")
    return _dedup_sorted(out, tol=tol)


def _has_near_value(sorted_values: np.ndarray, value: int, *, tol: int) -> bool:
    pos = int(np.searchsorted(sorted_values, value, side="left"))
    if pos < sorted_values.size and abs(int(sorted_values[pos]) - value) <= tol:
        return True
    return pos > 0 and abs(value - int(sorted_values[pos - 1])) <= tol


def _merge_union_int(a: np.ndarray, b: np.ndarray, *, tol: int) -> np.ndarray:
    a2 = _dedup_sorted_int(a, tol=tol)
    b2 = _dedup_sorted_int(b, tol=tol)
    if a2.size == 0:
        return b2
    if b2.size == 0:
        return a2
    keep = [value for value in b2 if not _has_near_value(a2, int(value), tol=tol)]
    if not keep:
        return a2
    out = np.concatenate([a2, np.asarray(keep, dtype="int64")], axis=0)
    out.sort(kind="stable")
    return _dedup_sorted_int(out, tol=tol)


def _merge_intersection(a: np.ndarray, b: np.ndarray, *, tol: float) -> np.ndarray:
    a2 = _dedup_sorted(a, tol=tol)
    b2 = _dedup_sorted(b, tol=tol)
    if a2.size == 0 or b2.size == 0:
        return np.asarray([], dtype="float64")
    pos = np.searchsorted(b2, a2, side="left")
    near_right = (pos < b2.size) & (np.abs(b2[np.minimum(pos, b2.size - 1)] - a2) <= tol)
    near_left = (pos > 0) & (np.abs(a2 - b2[pos - 1]) <= tol)
    out = a2[near_left | near_right]
    if out.size == 0:
        return out
    return _dedup_sorted(out, tol=tol)


def _merge_intersection_int(a: np.ndarray, b: np.ndarray, *, tol: int) -> np.ndarray:
    a2 = _dedup_sorted_int(a, tol=tol)
    b2 = _dedup_sorted_int(b, tol=tol)
    if a2.size == 0 or b2.size == 0:
        return np.asarray([], dtype="int64")
    out = np.asarray([value for value in a2 if _has_near_value(b2, int(value), tol=tol)], dtype="int64")
    if out.size == 0:
        return out
    return _dedup_sorted_int(out, tol=tol)


def _merge_many(rows: Sequence[np.ndarray], *, mode: Literal["outer", "inner"], tol: float) -> np.ndarray:
    out = np.asarray(rows[0], dtype="float64")
    for row in rows[1:]:
        out = _merge_union(out, row, tol=tol) if mode == "outer" else _merge_intersection(out, row, tol=tol)
    return out


def _merge_many_int(rows: Sequence[np.ndarray], *, mode: Literal["outer", "inner"], tol: int) -> np.ndarray:
    out = np.asarray(rows[0], dtype="int64")
    for row in rows[1:]:
        out = _merge_union_int(out, row, tol=tol) if mode == "outer" else _merge_intersection_int(out, row, tol=tol)
    return out


def _finite_row(param: np.ndarray, valid: np.ndarray, *, owner: str) -> np.ndarray:
    p = np.asarray(param, dtype="float64")
    m = np.asarray(valid, dtype=bool) & np.isfinite(p)
    row = p[np.flatnonzero(m)]
    if row.size >= 2 and np.any(np.diff(row) < 0):
        raise ValueError(f"{owner}: parameter coordinate must be monotonic non-decreasing.")
    return _dedup_sorted(row, tol=0.0)


def _finite_datetime_row(param: np.ndarray, valid: np.ndarray, *, owner: str) -> np.ndarray:
    p = np.asarray(param, dtype="int64")
    m = np.asarray(valid, dtype=bool) & (p != _NAT_INT)
    row = p[np.flatnonzero(m)]
    if row.size >= 2 and np.any(np.diff(row) < 0):
        raise ValueError(f"{owner}: parameter coordinate must be monotonic non-decreasing.")
    return _dedup_sorted_int(row, tol=0)


def _rows_join_domain(rows: Sequence[np.ndarray], *, tol: float) -> np.ndarray:
    if any(row.size == 0 for row in rows):
        return np.asarray([], dtype="float64")
    t0 = max(float(row[0]) for row in rows)
    t1 = min(float(row[-1]) for row in rows)
    if t0 > t1:
        return np.asarray([], dtype="float64")
    union = _merge_many(rows, mode="outer", tol=tol)
    return union[(union >= t0) & (union <= t1)]


def _rows_join_domain_int(rows: Sequence[np.ndarray], *, tol: int) -> np.ndarray:
    if any(row.size == 0 for row in rows):
        return np.asarray([], dtype="int64")
    t0 = max(int(row[0]) for row in rows)
    t1 = min(int(row[-1]) for row in rows)
    if t0 > t1:
        return np.asarray([], dtype="int64")
    union = _merge_many_int(rows, mode="outer", tol=tol)
    return union[(union >= t0) & (union <= t1)]


def _rows_join_exact(rows: Sequence[np.ndarray], *, tol: float, owner: str) -> np.ndarray:
    base = rows[0]
    for row in rows[1:]:
        if base.shape != row.shape or not np.all(np.abs(base - row) <= tol):
            raise ValueError(f"{owner}: join='exact' requires equal parameter grids within tolerance.")
    return base


def _rows_join_exact_int(rows: Sequence[np.ndarray], *, tol: int, owner: str) -> np.ndarray:
    base = rows[0]
    for row in rows[1:]:
        if base.shape != row.shape:
            raise ValueError(f"{owner}: join='exact' requires equal parameter grids within tolerance.")
        for left, right in zip(base, row, strict=True):
            if abs(int(left) - int(right)) > tol:
                raise ValueError(f"{owner}: join='exact' requires equal parameter grids within tolerance.")
    return base


def _rows_join(
    rows: Sequence[np.ndarray],
    *,
    join: Literal["outer", "inner", "domain", "exact"],
    tol: float,
    owner: str,
) -> np.ndarray:
    if join == "outer":
        return _merge_many(rows, mode="outer", tol=tol)
    if join == "inner":
        return _merge_many(rows, mode="inner", tol=tol)
    if join == "domain":
        return _rows_join_domain(rows, tol=tol)
    if join == "exact":
        return _rows_join_exact(rows, tol=tol, owner=owner)
    raise ValueError(f"{owner}: unsupported join {join!r}.")


def _rows_join_int(
    rows: Sequence[np.ndarray],
    *,
    join: Literal["outer", "inner", "domain", "exact"],
    tol: int,
    owner: str,
) -> np.ndarray:
    if join == "outer":
        return _merge_many_int(rows, mode="outer", tol=tol)
    if join == "inner":
        return _merge_many_int(rows, mode="inner", tol=tol)
    if join == "domain":
        return _rows_join_domain_int(rows, tol=tol)
    if join == "exact":
        return _rows_join_exact_int(rows, tol=tol, owner=owner)
    raise ValueError(f"{owner}: unsupported join {join!r}.")


def _validate_row_count(
    param_rows: Sequence[np.ndarray],
    valid_rows: Sequence[np.ndarray],
    *,
    owner: str,
) -> int:
    counts = {arr.shape[0] for arr in param_rows}
    counts.update(arr.shape[0] for arr in valid_rows)
    if len(counts) != 1:
        raise ValueError(f"{owner}: auto-grid join inputs must share batch row counts.")
    return int(next(iter(counts)))


def _pad_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    width = max((row.size for row in rows), default=0)
    out = np.full((len(rows), width), np.nan, dtype="float64")
    for row_index, row in enumerate(rows):
        if row.size:
            out[row_index, : row.size] = row
    return out


def _pad_datetime_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    width = max((row.size for row in rows), default=0)
    out = np.full((len(rows), width), _NAT_INT, dtype="int64")
    for row_index, row in enumerate(rows):
        if row.size:
            out[row_index, : row.size] = row
    return out


def join_rows_batched(
    param_rows: Sequence[np.ndarray],
    valid_rows: Sequence[np.ndarray],
    *,
    join: Literal["outer", "inner", "domain", "exact"],
    tol: float,
    owner: str,
) -> np.ndarray:
    """Join synthesized auto-grid rows batch-locally with deterministic NaN tail padding.

    Parameters
    ----------
    param_rows : Sequence[np.ndarray]
        Parameter-domain input used for temporal evaluation/alignment.
    valid_rows : Sequence[np.ndarray]
        Validity/mask payload used by this operation.
    join : Literal['outer', 'inner', 'domain', 'exact'], optional
        Policy selector controlling alignment/join behavior.
    tol : float, optional
        Numeric tolerance used for matching/alignment logic.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    np.ndarray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    row_count = _validate_row_count(param_rows, valid_rows, owner=owner)
    rows_out: list[np.ndarray] = []
    for row_index in range(row_count):
        finite_rows = [
            _finite_row(param[row_index], valid[row_index], owner=owner)
            for param, valid in zip(param_rows, valid_rows, strict=True)
        ]
        rows_out.append(_rows_join(finite_rows, join=join, tol=tol, owner=owner))
    return _pad_rows(rows_out)


def join_datetime_rows_batched(
    param_rows: Sequence[np.ndarray],
    valid_rows: Sequence[np.ndarray],
    *,
    join: Literal["outer", "inner", "domain", "exact"],
    tol: int,
    owner: str,
) -> np.ndarray:
    row_count = _validate_row_count(param_rows, valid_rows, owner=owner)
    rows_out: list[np.ndarray] = []
    for row_index in range(row_count):
        finite_rows = [
            _finite_datetime_row(param[row_index], valid[row_index], owner=owner)
            for param, valid in zip(param_rows, valid_rows, strict=True)
        ]
        rows_out.append(_rows_join_int(finite_rows, join=join, tol=tol, owner=owner))
    return _pad_datetime_rows(rows_out)


__all__ = ["join_datetime_rows_batched", "join_rows_batched"]
