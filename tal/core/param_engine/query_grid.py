from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from tal.utils.xarray_namespace import (
    dataarray_namespace_names,
    rename_dims_collision_safe,
    unique_temp_dim,
)

from ..orchestration.indexing import require_unique_lane_indexes
from ..ordered_dtypes import (
    is_float64_exact_integer,
    is_integral_dtype,
    is_ordered_real_numeric_dtype,
)
from .types import QueryGrid


def _assert_query_dim_namespace_safe(
    query: xr.DataArray,
    *,
    query_dim: str,
    owner: str,
) -> None:
    if query_dim in query.coords and query_dim not in query.dims:
        raise ValueError(
            f"{owner}: query input has scalar coordinate {query_dim!r} that collides with query_dim. "
            "Drop or rename that coordinate before param query operations."
        )


def _stack_to_query_dim(
    query: xr.DataArray,
    *,
    query_dim: str,
    dims: tuple[str, ...],
) -> xr.DataArray:
    if len(dims) == 1:
        dim = dims[0]
        return query if dim == query_dim else query.rename({dim: query_dim})
    names = set(dataarray_namespace_names(query))
    temp_dim = unique_temp_dim(f"{query_dim}__stack", taken_dims=tuple(sorted(names)))
    names.add(temp_dim)
    stacked = query.stack({temp_dim: list(dims)})
    if query_dim in stacked.coords:
        level_name = unique_temp_dim(
            f"{query_dim}__level",
            taken_dims=dataarray_namespace_names(stacked),
        )
        stacked = stacked.rename({query_dim: level_name})
    return rename_dims_collision_safe(
        stacked,
        mapping={temp_dim: query_dim},
        temp_prefix=f"{query_dim}__tmp__",
    )


def _as_query_dataarray(
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    query_dim: str,
    param_kind: str,
) -> tuple[xr.DataArray, tuple[str, ...] | None]:
    owner = "normalize_query_grid"
    if isinstance(query, xr.DataArray):
        _assert_query_dim_namespace_safe(
            query,
            query_dim=query_dim,
            owner=owner,
        )
        q = _coerce_query_dataarray(query, owner=owner, param_kind=param_kind)
        if q.ndim == 0:
            return q.expand_dims({query_dim: [0]}), None
        if q.ndim == 1:
            return q, None
        return q, None
    _reject_raw_lazy_query(query, owner=owner)
    arr = _coerce_query_array(query, owner=owner, param_kind=param_kind)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        raise ValueError(
            "normalize_query_grid: unlabeled numpy query with ndim > 1 is ambiguous. "
            "Pass an xr.DataArray with explicit dims."
        )
    return xr.DataArray(arr, dims=[query_dim]), None


def _coerce_query_dataarray(query: xr.DataArray, *, owner: str, param_kind: str) -> xr.DataArray:
    if param_kind == "numeric":
        if is_ordered_real_numeric_dtype(query.dtype):
            return query
        raise ValueError(f"{owner}: query values must be numeric with an ordered real dtype.")
    if param_kind == "datetime64":
        if np.issubdtype(np.dtype(query.dtype), np.number):
            raise ValueError(f"{owner}: datetime64 param queries must be datetime-like, got numeric dtype.")
        try:
            return query.astype("datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{owner}: query values must be datetime-like (coercible to datetime64[ns]).") from exc
    raise ValueError(f"{owner}: param_kind must be 'numeric' or 'datetime64', got {param_kind!r}.")


def _reject_raw_lazy_query(value: object, *, owner: str) -> None:
    if getattr(value, "chunks", None) is not None or hasattr(value, "__dask_graph__"):
        raise ValueError(
            f"{owner}: raw lazy query arrays are not supported; wrap labeled query data in xr.DataArray "
            "or materialize explicitly."
        )


def _validate_raw_numeric_promotion(query: object, out: np.ndarray, *, owner: str) -> None:
    if out.dtype.kind != "f" or isinstance(query, np.ndarray) or np.isscalar(query):
        return
    for value in query:
        scalar = np.asarray(value)
        if scalar.ndim == 0 and is_integral_dtype(scalar.dtype) and not is_float64_exact_integer(scalar.item()):
            raise ValueError(
                f"{owner}: mixed numeric query would convert integer value {int(scalar.item())!r} "
                "lossily to float64. Pass a homogeneous integer array or rescale the parameter domain."
            )


def _coerce_query_array(
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    owner: str,
    param_kind: str,
) -> np.ndarray:
    if param_kind == "numeric":
        try:
            out = np.asarray(query)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{owner}: query values must be numeric with an ordered real dtype.") from exc
        if not is_ordered_real_numeric_dtype(out.dtype):
            raise ValueError(f"{owner}: query values must be numeric with an ordered real dtype.")
        _validate_raw_numeric_promotion(query, out, owner=owner)
        return out
    if param_kind == "datetime64":
        probe = np.asarray(query)
        if np.issubdtype(np.dtype(probe.dtype), np.number):
            raise ValueError(f"{owner}: datetime64 param queries must be datetime-like, got numeric dtype.")
        try:
            return np.asarray(pd.to_datetime(query), dtype="datetime64[ns]")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{owner}: query values must be datetime-like (coercible to datetime64[ns]).") from exc
    raise ValueError(f"{owner}: param_kind must be 'numeric' or 'datetime64', got {param_kind!r}.")


def _normalize_unbatched_query(
    query: xr.DataArray,
    *,
    query_dim: str,
) -> tuple[xr.DataArray, tuple[str, ...] | None]:
    if query.ndim == 0:
        return query.expand_dims({query_dim: [0]}), None
    dims = tuple(query.dims)
    return _stack_to_query_dim(query, query_dim=query_dim, dims=dims), (dims if len(dims) > 1 else None)


def _check_partial_batch_dims(
    query: xr.DataArray,
    *,
    batch_dims: tuple[str, ...],
) -> tuple[str, ...]:
    present = tuple(dim for dim in batch_dims if dim in query.dims)
    if not present or len(present) == len(batch_dims):
        return present
    missing = [dim for dim in batch_dims if dim not in query.dims]
    raise ValueError(
        "normalize_query_grid: query includes a partial batch topology; "
        f"present={list(present)!r}, missing={missing!r}. "
        "Provide all batch_dims or none."
    )


def _assert_unique_axis_labels(
    da: xr.DataArray,
    *,
    dim: str,
    owner: str,
) -> None:
    if dim not in da.dims:
        return
    require_unique_lane_indexes(da, lane_dim=dim, owner=owner)


def _validate_query_axis_labels(
    query: xr.DataArray,
    *,
    query_dim: str,
    batch_dims: tuple[str, ...],
    owner: str,
) -> None:
    for dim in (query_dim, *batch_dims):
        _assert_unique_axis_labels(query, dim=dim, owner=owner)


def _validate_batch_indexer_topology(
    indexer: xr.DataArray,
    *,
    dim: str,
    owner: str,
) -> None:
    if tuple(indexer.dims) != (dim,):
        raise ValueError(
            f"{owner}: batch_coords[{dim!r}] must be a 1-D DataArray indexed by {dim!r}."
        )
    _assert_unique_axis_labels(indexer, dim=dim, owner=owner)


def _normalize_batched_query(
    query: xr.DataArray,
    *,
    query_dim: str,
    batch_dims: tuple[str, ...],
) -> tuple[xr.DataArray, tuple[str, ...] | None]:
    if not batch_dims:
        return _normalize_unbatched_query(query, query_dim=query_dim)
    present = _check_partial_batch_dims(query, batch_dims=batch_dims)
    if not present:
        return _normalize_unbatched_query(query, query_dim=query_dim)
    qdims = [dim for dim in query.dims if dim not in batch_dims]
    if not qdims:
        out = query.expand_dims({query_dim: [0]})
        return out.transpose(*batch_dims, query_dim), None
    out = _stack_to_query_dim(query, query_dim=query_dim, dims=tuple(qdims))
    stacked_dims = tuple(qdims) if len(qdims) > 1 else None
    return out.transpose(*batch_dims, query_dim), stacked_dims


def _reindex_batch_dim(
    query: xr.DataArray,
    *,
    dim: str,
    batch_coords: Mapping[str, xr.DataArray],
    param_kind: str,
) -> xr.DataArray:
    if dim not in query.dims or dim not in batch_coords:
        return query
    indexer = batch_coords[dim]
    if not isinstance(indexer, xr.DataArray):
        raise ValueError(f"normalize_query_grid: batch_coords[{dim!r}] must be an xr.DataArray.")
    _validate_batch_indexer_topology(
        indexer,
        dim=dim,
        owner="normalize_query_grid",
    )
    source_xindex = query.xindexes.get(dim)
    target_xindex = indexer.xindexes.get(dim)
    if (
        source_xindex is not None
        and target_xindex is not None
        and source_xindex.equals(target_xindex)
    ):
        return query
    try:
        source_index = query.get_index(dim)
        target_index = indexer.get_index(dim)
    except TypeError as exc:
        raise ValueError(
            f"normalize_query_grid: batch index along {dim!r} cannot be reindexed."
        ) from exc
    if source_index.equals(target_index):
        return query
    if is_integral_dtype(query.dtype) and not bool(target_index.isin(source_index).all()):
        raise ValueError(
            "normalize_query_grid: integral query batch reindex would introduce missing labels "
            f"along {dim!r} and require a lossy NaN upcast. Provide every target batch label "
            "or use an ordered floating-point query dtype."
        )
    fill = np.datetime64("NaT", "ns") if param_kind == "datetime64" else np.nan
    return query.reindex({dim: indexer}, fill_value=fill)


def _apply_batch_coords(
    query: xr.DataArray,
    *,
    batch_dims: tuple[str, ...],
    batch_coords: Mapping[str, xr.DataArray] | None,
    param_kind: str,
) -> xr.DataArray:
    if not batch_coords:
        return query
    out = query
    for dim in batch_dims:
        out = _reindex_batch_dim(out, dim=dim, batch_coords=batch_coords, param_kind=param_kind)
    return out


def normalize_query_grid(
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    query_dim: str = "query",
    batch_dims: Sequence[str] = (),
    batch_coords: Mapping[str, xr.DataArray] | None = None,
    param_kind: str = "numeric",
    enforce_order: bool = True,
) -> QueryGrid:
    """Normalize query input to scalar/1d/batched query grid.

    Parameters
    ----------
    query : xr.DataArray | np.ndarray | Sequence[float] | float
        Query coordinate/grid used for parameter evaluation.
    query_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    batch_dims : Sequence[str], optional
        Optional override for batch dimensions used by temporal semantics.
    batch_coords : Mapping[str, xr.DataArray] | None, optional
        Coordinate name/value used by this operation.
    param_kind : {'numeric', 'datetime64'}, optional
        Parameter coordinate kind used to normalize query values.
    enforce_order : bool, optional
        Behavior flag/policy controlling boundary semantics.

    Returns
    -------
    QueryGrid
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    batch_tuple = tuple(str(dim) for dim in batch_dims)
    if query_dim in batch_tuple:
        raise ValueError(
            "normalize_query_grid: query_dim "
            f"{query_dim!r} collides with batch_dims {list(batch_tuple)!r}."
        )
    base, stacked = _as_query_dataarray(query, query_dim=query_dim, param_kind=param_kind)
    q, batch_stacked = _normalize_batched_query(base, query_dim=query_dim, batch_dims=batch_tuple)
    stacked_dims = batch_stacked or stacked
    _validate_query_axis_labels(
        q,
        query_dim=query_dim,
        batch_dims=batch_tuple,
        owner="normalize_query_grid",
    )
    q = _apply_batch_coords(q, batch_dims=batch_tuple, batch_coords=batch_coords, param_kind=param_kind)
    if enforce_order and query_dim in q.dims:
        lead = [dim for dim in q.dims if dim != query_dim]
        q = q.transpose(*lead, query_dim)
    return QueryGrid(values=q, query_dim=query_dim, stacked_dims=stacked_dims)


__all__ = ["normalize_query_grid"]
