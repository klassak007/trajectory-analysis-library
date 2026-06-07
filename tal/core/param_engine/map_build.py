from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .backends import (
    PARAM_BOUNDS_BACKEND_NUMBA,
    PARAM_BOUNDS_BACKEND_NUMPY_BLOCK,
    PARAM_MAP_BACKEND_NUMBA,
    PARAM_MAP_BACKEND_NUMPY_BLOCK,
    bounds_block_backend,
    map_block_backend,
)
from .types import ParamBoundsMap, ParamMap, ParamMapOptions
from tal.utils.numba_support import _numba_available

_DUPLICATE_CODES = {"invalid": 0, "left": 1, "right": 2, "raise": 3}
_DUPLICATE_BRACKET_ERROR = (
    "build_param_map: duplicate parameter bracket encountered for linear interpolation."
)


@dataclass
class _LinearOutputBuffers:
    i0: np.ndarray
    i1: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


def _empty_map_result(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i0 = np.zeros(k, dtype="int64")
    i1 = np.zeros(k, dtype="int64")
    alpha = np.zeros(k, dtype="float64")
    valid = np.zeros(k, dtype=bool)
    return i0, i1, alpha, valid


def _nearest_row(
    src_idx: np.ndarray,
    src: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map query values to nearest valid sample indices for one row."""
    i0, i1, alpha, valid = _empty_map_result(int(query.size))
    if src.size == 0:
        return i0, i1, alpha, valid
    q = np.asarray(query, dtype="float64")
    finite = np.isfinite(q)
    if not np.any(finite):
        return i0, i1, alpha, valid

    qf = q[finite]
    right = np.searchsorted(src, qf, side="left")
    right_pick = np.clip(right, 0, src.size - 1)
    left_pick = np.clip(right - 1, 0, src.size - 1)
    interior = (right > 0) & (right < src.size)
    if np.any(interior):
        left_dist = np.abs(qf[interior] - src[left_pick[interior]])
        right_dist = np.abs(src[right_pick[interior]] - qf[interior])
        choose_left = left_dist <= right_dist
        right_pick[interior] = np.where(choose_left, left_pick[interior], right_pick[interior])
    picks = src_idx[right_pick].astype("int64", copy=False)
    finite_idx = np.flatnonzero(finite)
    i0[finite_idx] = picks
    i1[finite_idx] = picks
    valid[finite_idx] = True
    return i0, i1, alpha, valid


def _assign_constant_picks(
    row_idx: np.ndarray,
    mask: np.ndarray,
    src_pick: np.ndarray,
    src_idx: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    valid: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    out_idx = row_idx[mask]
    picked = src_idx[src_pick[mask]].astype("int64", copy=False)
    i0[out_idx] = picked
    i1[out_idx] = picked
    valid[out_idx] = True


def _apply_duplicate_policy(
    dup_code: int,
    row_idx: np.ndarray,
    mask: np.ndarray,
    left_pick: np.ndarray,
    right_pick: np.ndarray,
    src_idx: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    valid: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    if dup_code == 3:
        raise ValueError(_DUPLICATE_BRACKET_ERROR)
    if dup_code == 0:
        return
    src_pick = left_pick if dup_code == 1 else right_pick
    _assign_constant_picks(row_idx, mask, src_pick, src_idx, i0, i1, valid)


def _apply_linear_interior(
    dup_code: int,
    qf: np.ndarray,
    lo: np.ndarray,
    dup: np.ndarray,
    finite_idx: np.ndarray,
    src: np.ndarray,
    src_idx: np.ndarray,
    out: _LinearOutputBuffers,
) -> None:
    interior = (~dup) & (lo > 0) & (lo < src.size)
    if not np.any(interior):
        return
    left = lo[interior] - 1
    right = lo[interior]
    t0 = src[left]
    t1 = src[right]
    inner_idx = finite_idx[interior]
    degenerate = t1 == t0
    _apply_duplicate_policy(dup_code, inner_idx, degenerate, left, right, src_idx, out.i0, out.i1, out.valid)
    interp = ~degenerate
    if not np.any(interp):
        return
    interp_idx = inner_idx[interp]
    left_idx = left[interp]
    right_idx = right[interp]
    out.i0[interp_idx] = src_idx[left_idx].astype("int64", copy=False)
    out.i1[interp_idx] = src_idx[right_idx].astype("int64", copy=False)
    out.alpha[interp_idx] = (qf[interior][interp] - t0[interp]) / (t1[interp] - t0[interp])
    out.valid[interp_idx] = True


def _linear_row(
    src_idx: np.ndarray,
    src: np.ndarray,
    query: np.ndarray,
    *,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map query values to interpolation brackets for one row."""
    i0, i1, alpha, valid = _empty_map_result(int(query.size))
    if src.size == 0:
        return i0, i1, alpha, valid
    q = np.asarray(query, dtype="float64")
    finite = np.isfinite(q)
    if not np.any(finite):
        return i0, i1, alpha, valid

    qf = q[finite]
    lo = np.searchsorted(src, qf, side="left")
    hi = np.searchsorted(src, qf, side="right")
    finite_idx = np.flatnonzero(finite)

    dup = (hi - lo) > 1
    _apply_duplicate_policy(dup_code, finite_idx, dup, lo, hi - 1, src_idx, i0, i1, valid)

    endpoint = (~dup) & (((lo == 0) & (qf == src[0])) | ((lo >= src.size) & (qf == src[-1])))
    _assign_constant_picks(finite_idx, endpoint, np.where(lo == 0, 0, src.size - 1), src_idx, i0, i1, valid)
    _apply_linear_interior(
        dup_code,
        qf,
        lo,
        dup,
        finite_idx,
        src,
        src_idx,
        _LinearOutputBuffers(i0=i0, i1=i1, alpha=alpha, valid=valid),
    )
    return i0, i1, alpha, valid


def _map_row(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    query_row: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(param_row, dtype="float64")
    valid = np.asarray(valid_row, dtype=bool) & np.isfinite(src)
    src_idx = np.flatnonzero(valid).astype("int64", copy=False)
    src_vals = src[src_idx]
    if src_vals.size >= 2 and np.any(np.diff(src_vals) < 0):
        raise ValueError("build_param_map: parameter coordinate must be monotonic non-decreasing on valid domain.")
    query = np.asarray(query_row, dtype="float64")
    if method == "nearest":
        return _nearest_row(src_idx, src_vals, query)
    return _linear_row(src_idx, src_vals, query, dup_code=dup_code)


def _validate_numeric_param_dtype(*, param: xr.DataArray, owner: str) -> None:
    if np.issubdtype(np.dtype(param.dtype), np.number):
        return
    raise ValueError(
        f"{owner}: param coordinate must have numeric dtype, got {param.dtype!r}. "
        "Use a numeric param_coord before interpolation."
    )


def _validate_map_dims(
    *,
    param: xr.DataArray,
    query: xr.DataArray,
    sequence_dim: str,
    query_dim: str,
) -> None:
    if sequence_dim not in param.dims:
        raise ValueError(f"build_param_map: param is missing sequence_dim {sequence_dim!r}.")
    if query_dim not in query.dims:
        raise ValueError(f"build_param_map: query is missing query_dim {query_dim!r}.")
    if query_dim == sequence_dim:
        raise ValueError(
            "build_param_map: query_dim must differ from sequence_dim; "
            f"got {query_dim!r}. Choose a distinct query axis name."
        )


def _validate_bound_target_dims(
    *,
    target: xr.DataArray,
    name: str,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
) -> None:
    dims = tuple(target.dims)
    if sequence_dim in dims:
        raise ValueError(
            f"build_param_bounds_map: {name} dims {dims!r} cannot include sequence_dim {sequence_dim!r}."
        )
    if dims == () or dims == batch_dims:
        return
    raise ValueError(
        f"build_param_bounds_map: {name} dims {dims!r} must be scalar () or exactly batch dims {batch_dims!r}."
    )


def _validate_bounds_dims(
    *,
    param: xr.DataArray,
    start: xr.DataArray,
    stop: xr.DataArray,
    sequence_dim: str,
) -> None:
    if sequence_dim not in param.dims:
        raise ValueError(f"build_param_bounds_map: param is missing sequence_dim {sequence_dim!r}.")
    batch_dims = tuple(dim for dim in param.dims if dim != sequence_dim)
    _validate_bound_target_dims(
        target=start,
        name="start",
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
    )
    _validate_bound_target_dims(
        target=stop,
        name="stop",
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
    )


def _bounds_row(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(param_row, dtype="float64")
    valid = np.asarray(valid_row, dtype=bool) & np.isfinite(src)
    src_idx = np.flatnonzero(valid).astype("int64", copy=False)
    src_vals = src[src_idx]
    if src_vals.size >= 2 and np.any(np.diff(src_vals) < 0):
        raise ValueError(
            "build_param_bounds_map: parameter coordinate must be monotonic non-decreasing on valid domain."
        )
    row_len = int(src.shape[0])
    if src_vals.size == 0:
        return np.asarray(0, dtype="int64"), np.asarray(0, dtype="int64")
    s0 = float(np.asarray(start, dtype="float64"))
    s1 = float(np.asarray(stop, dtype="float64"))
    if np.isnan(s0) or np.isnan(s1):
        return np.asarray(0, dtype="int64"), np.asarray(0, dtype="int64")
    lo = int(np.searchsorted(src_vals, s0, side="left"))
    hi = int(np.searchsorted(src_vals, s1, side="right"))
    if hi <= lo:
        edge = row_len if lo >= src_idx.size else int(src_idx[lo])
        edge = max(0, min(edge, row_len))
        return np.asarray(edge, dtype="int64"), np.asarray(edge, dtype="int64")
    i0 = int(src_idx[min(lo, src_idx.size - 1)])
    i1 = int(src_idx[min(hi - 1, src_idx.size - 1)] + 1)
    return np.asarray(min(i0, row_len), dtype="int64"), np.asarray(min(i1, row_len), dtype="int64")


def _prepare_map_inputs(
    *,
    param: xr.DataArray,
    query: xr.DataArray,
    sequence_dim: str,
    query_dim: str,
    valid_mask: xr.DataArray | None,
    options: ParamMapOptions | None,
) -> tuple[ParamMapOptions, xr.DataArray, xr.DataArray, xr.DataArray]:
    _validate_map_dims(param=param, query=query, sequence_dim=sequence_dim, query_dim=query_dim)
    _validate_numeric_param_dtype(param=param, owner="build_param_map")
    opts = options or ParamMapOptions()
    if opts.method not in ("nearest", "linear"):
        raise ValueError(f"build_param_map: method must be 'nearest' or 'linear', got {opts.method!r}.")
    if opts.duplicate_policy not in _DUPLICATE_CODES:
        raise ValueError(f"build_param_map: invalid duplicate policy {opts.duplicate_policy!r}.")
    mask = valid_mask if valid_mask is not None else xr.apply_ufunc(np.isfinite, param, dask="allowed")
    aligned = xr.align(param.astype("float64"), mask.astype(bool), query.astype("float64"), join="exact")
    return opts, aligned[0], aligned[1], aligned[2]


def _select_map_normal_backend() -> str:
    if _numba_available():
        return PARAM_MAP_BACKEND_NUMBA
    return PARAM_MAP_BACKEND_NUMPY_BLOCK


def _select_bounds_normal_backend() -> str:
    if _numba_available():
        return PARAM_BOUNDS_BACKEND_NUMBA
    return PARAM_BOUNDS_BACKEND_NUMPY_BLOCK


def _apply_param_map_block(
    *,
    param_da: xr.DataArray,
    mask_da: xr.DataArray,
    query_da: xr.DataArray,
    sequence_dim: str,
    query_dim: str,
    opts: ParamMapOptions,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    backend = _select_map_normal_backend()
    return xr.apply_ufunc(
        map_block_backend,
        param_da,
        mask_da,
        query_da,
        kwargs={
            "method": opts.method,
            "dup_code": _DUPLICATE_CODES[opts.duplicate_policy],
            "backend": backend,
        },
        input_core_dims=[[sequence_dim], [sequence_dim], [query_dim]],
        output_core_dims=[[query_dim], [query_dim], [query_dim], [query_dim]],
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int64, np.int64, np.float64, bool],
    )


def build_param_map(
    *,
    param: xr.DataArray,
    query: xr.DataArray,
    sequence_dim: str,
    query_dim: str,
    valid_mask: xr.DataArray | None = None,
    options: ParamMapOptions | None = None,
) -> ParamMap:
    """Build mapping from parameter values to sample indices.

    Parameters
    ----------
    param : xr.DataArray, optional
        Parameter-domain input used for temporal evaluation/alignment.
    query : xr.DataArray, optional
        Query coordinate/grid used for parameter evaluation.
    sequence_dim : str, optional
        Optional override for the sequence dimension used by temporal semantics.
    query_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    valid_mask : xr.DataArray | None, optional
        Validity/mask payload used by this operation.
    options : ParamMapOptions | None, optional
        Options controlling policy and execution behavior.

    Returns
    -------
    ParamMap
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    opts, param_da, mask_da, query_da = _prepare_map_inputs(
        param=param,
        query=query,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        valid_mask=valid_mask,
        options=options,
    )
    i0, i1, alpha, valid = _apply_param_map_block(
        param_da=param_da,
        mask_da=mask_da,
        query_da=query_da,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        opts=opts,
    )
    return ParamMap(i0=i0, i1=i1, alpha=alpha, valid=valid, query_dim=query_dim)


def _prepare_bounds_inputs(
    *,
    param: xr.DataArray,
    start: xr.DataArray | float,
    stop: xr.DataArray | float,
    sequence_dim: str,
    valid_mask: xr.DataArray | None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    start_da = start if isinstance(start, xr.DataArray) else xr.DataArray(np.asarray(start, dtype="float64"))
    stop_da = stop if isinstance(stop, xr.DataArray) else xr.DataArray(np.asarray(stop, dtype="float64"))
    _validate_bounds_dims(param=param, start=start_da, stop=stop_da, sequence_dim=sequence_dim)
    _validate_numeric_param_dtype(param=param, owner="build_param_bounds_map")
    mask = valid_mask if valid_mask is not None else xr.apply_ufunc(np.isfinite, param, dask="allowed")
    aligned = xr.align(
        param.astype("float64"),
        mask.astype(bool),
        start_da.astype("float64"),
        stop_da.astype("float64"),
        join="exact",
    )
    return aligned[0], aligned[1], aligned[2], aligned[3]


def _apply_param_bounds_block(
    *,
    param_da: xr.DataArray,
    mask_da: xr.DataArray,
    start_da: xr.DataArray,
    stop_da: xr.DataArray,
    sequence_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    backend = _select_bounds_normal_backend()
    return xr.apply_ufunc(
        bounds_block_backend,
        param_da,
        mask_da,
        start_da,
        stop_da,
        kwargs={"backend": backend},
        input_core_dims=[[sequence_dim], [sequence_dim], [], []],
        output_core_dims=[[], []],
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int64, np.int64],
    )


def build_param_bounds_map(
    *,
    param: xr.DataArray,
    start: xr.DataArray | float,
    stop: xr.DataArray | float,
    sequence_dim: str,
    valid_mask: xr.DataArray | None = None,
) -> ParamBoundsMap:
    """Build searchsorted bounds for per-row param slice selection.

    Parameters
    ----------
    param : xr.DataArray, optional
        Parameter-domain input used for temporal evaluation/alignment.
    start : xr.DataArray | float, optional
        Numeric boundary/range parameter for this operation.
    stop : xr.DataArray | float, optional
        Numeric boundary/range parameter for this operation.
    sequence_dim : str, optional
        Optional override for the sequence dimension used by temporal semantics.
    valid_mask : xr.DataArray | None, optional
        Validity/mask payload used by this operation.

    Returns
    -------
    ParamBoundsMap
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    param_da, mask_da, start_da, stop_da = _prepare_bounds_inputs(
        param=param,
        start=start,
        stop=stop,
        sequence_dim=sequence_dim,
        valid_mask=valid_mask,
    )
    i0, i1 = _apply_param_bounds_block(
        param_da=param_da,
        mask_da=mask_da,
        start_da=start_da,
        stop_da=stop_da,
        sequence_dim=sequence_dim,
    )
    return ParamBoundsMap(i0=i0, i1=i1)


__all__ = ["build_param_map", "build_param_bounds_map"]
