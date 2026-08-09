from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..ordered_dtypes import is_ordered_real_numeric_dtype
from .backends import (
    PARAM_BOUNDS_BACKEND_NUMBA,
    PARAM_BOUNDS_BACKEND_NUMPY_BLOCK,
    PARAM_MAP_BACKEND_NUMBA,
    PARAM_MAP_BACKEND_NUMPY_BLOCK,
    bounds_block_backend,
    map_block_backend,
)
from .datetime_rows import DATETIME_OPEN_START, DATETIME_OPEN_STOP
from .types import ParamBoundsMap, ParamMap, ParamMapOptions
from tal.utils.numba_support import _numba_available

_DUPLICATE_CODES = {"invalid": 0, "left": 1, "right": 2, "raise": 3}

def _validate_numeric_param_dtype(*, param: xr.DataArray, owner: str) -> None:
    if is_ordered_real_numeric_dtype(param.dtype):
        return
    raise ValueError(
        f"{owner}: param coordinate must have an ordered real numeric dtype, got {param.dtype!r}. "
        "Use an integer or floating-point param_coord before interpolation."
    )


def _validate_numeric_operand_dtype(*, value: xr.DataArray, field: str, owner: str) -> None:
    if is_ordered_real_numeric_dtype(value.dtype):
        return
    raise ValueError(
        f"{owner}: {field} must have an ordered real numeric dtype, got {value.dtype!r}."
    )


def _validate_datetime_param_dtype(*, param: xr.DataArray, owner: str) -> None:
    if np.issubdtype(np.dtype(param.dtype), np.datetime64):
        return
    raise ValueError(
        f"{owner}: param coordinate must have datetime64 dtype, got {param.dtype!r}. "
        "Use a datetime64 param_coord before datetime64 interpolation."
    )


def _validate_param_kind(param_kind: str, *, owner: str) -> None:
    if param_kind in {"numeric", "datetime64"}:
        return
    raise ValueError(f"{owner}: param_kind must be 'numeric' or 'datetime64', got {param_kind!r}.")


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


def _prepare_map_inputs(
    *,
    param: xr.DataArray,
    query: xr.DataArray,
    sequence_dim: str,
    query_dim: str,
    valid_mask: xr.DataArray | None,
    options: ParamMapOptions | None,
    param_kind: str,
) -> tuple[ParamMapOptions, xr.DataArray, xr.DataArray, xr.DataArray]:
    _validate_param_kind(param_kind, owner="build_param_map")
    _validate_map_dims(param=param, query=query, sequence_dim=sequence_dim, query_dim=query_dim)
    opts = options or ParamMapOptions()
    if opts.method not in ("nearest", "linear"):
        raise ValueError(f"build_param_map: method must be 'nearest' or 'linear', got {opts.method!r}.")
    if opts.duplicate_policy not in _DUPLICATE_CODES:
        raise ValueError(f"build_param_map: invalid duplicate policy {opts.duplicate_policy!r}.")
    if param_kind == "numeric":
        _validate_numeric_param_dtype(param=param, owner="build_param_map")
        _validate_numeric_operand_dtype(value=query, field="query", owner="build_param_map")
        mask = valid_mask if valid_mask is not None else xr.apply_ufunc(np.isfinite, param, dask="allowed")
        aligned = xr.align(param, mask.astype(bool), query, join="exact")
        return opts, aligned[0], aligned[1], aligned[2]
    _validate_datetime_param_dtype(param=param, owner="build_param_map")
    mask = valid_mask if valid_mask is not None else param.notnull()
    aligned = xr.align(param.astype("datetime64[ns]"), mask.astype(bool), query.astype("datetime64[ns]"), join="exact")
    return opts, aligned[0], aligned[1], aligned[2]


def _float_numba_compatible(*values: xr.DataArray) -> bool:
    return all(np.dtype(value.dtype).kind == "f" and np.dtype(value.dtype).itemsize in {4, 8} for value in values)


def _select_map_normal_backend(*, param: xr.DataArray, query: xr.DataArray) -> str:
    if _float_numba_compatible(param, query) and _numba_available():
        return PARAM_MAP_BACKEND_NUMBA
    return PARAM_MAP_BACKEND_NUMPY_BLOCK


def _select_bounds_normal_backend(
    *,
    param: xr.DataArray,
    start: xr.DataArray,
    stop: xr.DataArray,
) -> str:
    if _float_numba_compatible(param, start, stop) and _numba_available():
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
    param_kind: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    if param_kind == "datetime64":
        from .numpy_backends import datetime_map_block_numpy

        return xr.apply_ufunc(
            datetime_map_block_numpy,
            param_da,
            mask_da,
            query_da,
            kwargs={
                "method": opts.method,
                "dup_code": _DUPLICATE_CODES[opts.duplicate_policy],
            },
            input_core_dims=[[sequence_dim], [sequence_dim], [query_dim]],
            output_core_dims=[[query_dim], [query_dim], [query_dim], [query_dim]],
            vectorize=False,
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[np.int64, np.int64, np.float64, bool],
        )
    backend = _select_map_normal_backend(param=param_da, query=query_da)
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
    param_kind: str = "numeric",
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
    param_kind : {'numeric', 'datetime64'}, optional
        Parameter coordinate kind used for query and map coercion.

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
        param_kind=param_kind,
    )
    i0, i1, alpha, valid = _apply_param_map_block(
        param_da=param_da,
        mask_da=mask_da,
        query_da=query_da,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        opts=opts,
        param_kind=param_kind,
    )
    return ParamMap(i0=i0, i1=i1, alpha=alpha, valid=valid, query_dim=query_dim)


def _prepare_bounds_inputs(
    *,
    param: xr.DataArray,
    start: xr.DataArray | float | object,
    stop: xr.DataArray | float | object,
    sequence_dim: str,
    valid_mask: xr.DataArray | None,
    param_kind: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    _validate_param_kind(param_kind, owner="build_param_bounds_map")
    start_da = _coerce_bound_value(start, bound="start", param_kind=param_kind)
    stop_da = _coerce_bound_value(stop, bound="stop", param_kind=param_kind)
    _validate_bounds_dims(param=param, start=start_da, stop=stop_da, sequence_dim=sequence_dim)
    if param_kind == "numeric":
        _validate_numeric_param_dtype(param=param, owner="build_param_bounds_map")
        _validate_numeric_operand_dtype(value=start_da, field="slice.start", owner="build_param_bounds_map")
        _validate_numeric_operand_dtype(value=stop_da, field="slice.stop", owner="build_param_bounds_map")
        mask = valid_mask if valid_mask is not None else xr.apply_ufunc(np.isfinite, param, dask="allowed")
        aligned = xr.align(
            param,
            mask.astype(bool),
            start_da,
            stop_da,
            join="exact",
        )
        return aligned[0], aligned[1], aligned[2], aligned[3]
    _validate_datetime_param_dtype(param=param, owner="build_param_bounds_map")
    mask = valid_mask if valid_mask is not None else param.notnull()
    aligned = xr.align(
        param.astype("datetime64[ns]"),
        mask.astype(bool),
        start_da.astype("datetime64[ns]"),
        stop_da.astype("datetime64[ns]"),
        join="exact",
    )
    return aligned[0], aligned[1], aligned[2], aligned[3]


def _coerce_bound_value(value: object, *, bound: str, param_kind: str) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        return _coerce_bound_dataarray(value, bound=bound, param_kind=param_kind)
    if param_kind == "numeric":
        fill = -np.inf if bound == "start" and value is None else np.inf if value is None else value
        try:
            out = xr.DataArray(np.asarray(fill))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"build_param_bounds_map: slice.{bound} must be numeric.") from exc
        _validate_numeric_operand_dtype(value=out, field=f"slice.{bound}", owner="build_param_bounds_map")
        return out
    fill = DATETIME_OPEN_START if bound == "start" and value is None else DATETIME_OPEN_STOP if value is None else value
    if np.issubdtype(np.asarray(fill).dtype, np.number):
        raise ValueError(f"build_param_bounds_map: {bound} must be datetime-like, got numeric dtype.")
    try:
        if isinstance(fill, np.datetime64):
            arr = np.asarray(fill, dtype="datetime64[ns]")
        else:
            arr = np.asarray(pd.to_datetime(fill), dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"build_param_bounds_map: {bound} must be datetime-like for datetime64 param coordinates."
        ) from exc
    return xr.DataArray(arr)


def _coerce_bound_dataarray(value: xr.DataArray, *, bound: str, param_kind: str) -> xr.DataArray:
    if param_kind == "numeric":
        _validate_numeric_operand_dtype(value=value, field=f"slice.{bound}", owner="build_param_bounds_map")
        return value
    if np.issubdtype(np.dtype(value.dtype), np.number):
        raise ValueError(f"build_param_bounds_map: {bound} must be datetime-like, got numeric dtype.")
    return value.astype("datetime64[ns]")


def _apply_param_bounds_block(
    *,
    param_da: xr.DataArray,
    mask_da: xr.DataArray,
    start_da: xr.DataArray,
    stop_da: xr.DataArray,
    sequence_dim: str,
    param_kind: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    if param_kind == "datetime64":
        from .numpy_backends import datetime_bounds_block_numpy

        return xr.apply_ufunc(
            datetime_bounds_block_numpy,
            param_da,
            mask_da,
            start_da,
            stop_da,
            input_core_dims=[[sequence_dim], [sequence_dim], [], []],
            output_core_dims=[[], []],
            vectorize=False,
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[np.int64, np.int64],
        )
    backend = _select_bounds_normal_backend(param=param_da, start=start_da, stop=stop_da)
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
    start: xr.DataArray | float | object,
    stop: xr.DataArray | float | object,
    sequence_dim: str,
    valid_mask: xr.DataArray | None = None,
    param_kind: str = "numeric",
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
    param_kind : {'numeric', 'datetime64'}, optional
        Parameter coordinate kind used for bound coercion.

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
        param_kind=param_kind,
    )
    i0, i1 = _apply_param_bounds_block(
        param_da=param_da,
        mask_da=mask_da,
        start_da=start_da,
        stop_da=stop_da,
        sequence_dim=sequence_dim,
        param_kind=param_kind,
    )
    return ParamBoundsMap(i0=i0, i1=i1)


__all__ = ["build_param_map", "build_param_bounds_map"]
