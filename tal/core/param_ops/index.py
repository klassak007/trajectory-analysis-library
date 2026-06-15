from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from ..param_engine import ParamMapOptions, build_param_map, normalize_query_grid
from .guards import assert_query_dim_safe, assert_reserved_metadata_safe
from .options import validate_select_options
from .types import ParamIndexResult, ParamRuntimeContext, ParamSelectOptions


def _is_scalar_query(
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    batch_dims: tuple[str, ...],
) -> bool:
    if isinstance(query, xr.DataArray):
        if query.ndim == 0:
            return True
        if batch_dims and tuple(query.dims) == batch_dims:
            return True
        return False
    return np.asarray(query).ndim == 0


def build_index_result(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamSelectOptions,
) -> ParamIndexResult:
    validate_select_options(opts, owner="param index")
    assert_reserved_metadata_safe(
        context.ds,
        reserved=("valid", "sample_index"),
        param_name=context.spec.name,
        owner="param index",
    )
    assert_query_dim_safe(
        context.ds,
        sequence_dim=context.sequence_dim,
        query_dim=opts.query_dim,
        owner="param index",
    )
    grid = normalize_query_grid(
        query,
        query_dim=opts.query_dim,
        batch_dims=context.batch_dims,
        batch_coords=context.batch_coords,
        param_kind=context.param_kind,
    )
    pmap = build_param_map(
        param=context.spec.coord,
        query=grid.values,
        sequence_dim=context.sequence_dim,
        query_dim=grid.query_dim,
        valid_mask=context.valid_mask,
        options=ParamMapOptions(method="nearest"),
        param_kind=context.param_kind,
    )
    idx = pmap.i0.where(pmap.valid, other=np.int64(-1)).astype("int64")
    return ParamIndexResult(
        index=idx,
        valid=pmap.valid,
        grid=grid,
        scalar_query=_is_scalar_query(query, batch_dims=context.batch_dims),
    )


def _finalize_index_shape(result: ParamIndexResult) -> xr.DataArray:
    if result.grid.stacked_dims is not None and result.grid.query_dim in result.index.dims:
        return result.index.unstack(result.grid.query_dim)
    if result.scalar_query and result.grid.query_dim in result.index.dims:
        return result.index.isel({result.grid.query_dim: 0}, drop=True)
    return result.index


def build_index(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamSelectOptions,
) -> xr.DataArray:
    """Build nearest sample indices for a query grid.

    Parameters
    ----------
    context : ParamRuntimeContext
        Resolved runtime context/payload used by this orchestration boundary.
    query : xr.DataArray | np.ndarray | Sequence[float] | float, optional
        Query coordinate/grid used for parameter evaluation.
    opts : ParamSelectOptions, optional
        Optional options controlling policy and numeric behavior for this operation.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    result = build_index_result(context, query=query, opts=opts)
    return _finalize_index_shape(result)


__all__ = ["build_index", "build_index_result"]
