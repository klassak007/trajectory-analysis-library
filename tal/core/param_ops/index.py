from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import xarray as xr

from ..orchestration.indexing import require_compatible_shared_batch_index_types
from ..param_engine import ParamMapOptions
from ..param_engine.query_output_verify import verify_query_output_plan
from ..param_engine.query_topology import (
    QueryOutputPlan,
    preflight_query_output_namespace,
    restore_query_topology,
)
from .guards import (
    assert_query_dim_safe,
    assert_reserved_metadata_safe,
    reserved_coord_is_owned,
)
from .options import validate_select_options
from .runtime_prepare import prepare_runtime_param_evaluation
from .types import ParamIndexResult, ParamRuntimeContext, ParamSelectOptions


def _is_scalar_query(
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    batch_dims: tuple[str, ...],
) -> bool:
    if isinstance(query, xr.DataArray):
        if query.ndim == 0:
            return True
        return bool(batch_dims) and tuple(query.dims) == batch_dims
    return np.asarray(query).ndim == 0


def _index_query_without_consumed_metadata(
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    size_coord: xr.DataArray | None,
) -> tuple[xr.DataArray | np.ndarray | Sequence[float] | float, tuple[str, ...]]:
    if not isinstance(query, xr.DataArray):
        return query, ()
    size_name = str(size_coord.name) if size_coord is not None else None
    inherited_sizes = tuple(
        name for name in query.coords
        if isinstance(name, str)
        and name not in query.dims
        and name not in ("valid", "sample_index")
        and (
            reserved_coord_is_owned(query, name=name)
            or (name == size_name and query.coords[name].dims == size_coord.dims)
        )
    )
    remove = list(inherited_sizes)
    for name in ("valid", "sample_index"):
        if name not in query.coords:
            continue
        if not reserved_coord_is_owned(query, name=name):
            raise ValueError(f"param index: query coordinate {name!r} is reserved for TAL runtime metadata.")
        remove.append(name)
    return (query.drop_vars(remove) if remove else query), inherited_sizes


def _index_source_for_preflight(context: ParamRuntimeContext) -> xr.DataArray:
    source = context.spec.coord
    names = [context.sequence_size_coord] if context.sequence_size_coord in source.coords else []
    names.extend(
        name for name in ("valid", "sample_index")
        if name in source.coords and reserved_coord_is_owned(context.ds, name=name)
    )
    return source.drop_vars(names) if names else source


def _preflight_index_request(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamSelectOptions,
    namespace_preflight: bool,
) -> tuple[xr.DataArray | np.ndarray | Sequence[float] | float, tuple[str, ...], QueryOutputPlan]:
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
    size_coord = context.ds.coords.get(context.sequence_size_coord, None)
    cleaned, inherited_sizes = _index_query_without_consumed_metadata(
        query,
        size_coord=size_coord,
    )
    if isinstance(cleaned, xr.DataArray):
        require_compatible_shared_batch_index_types(
            context.ds, cleaned, batch_dims=context.batch_dims, owner="param index",
        )
    consumed_names = ("valid", "sample_index", *inherited_sizes)
    output_plan = preflight_query_output_namespace(
        _index_source_for_preflight(context),
        cleaned if namespace_preflight else None,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        owner="param index",
        intent="index",
        consumed_names=consumed_names,
    )
    return cleaned, consumed_names, output_plan


def _without_index_output_metadata(
    index: xr.DataArray,
    *,
    consumed_names: tuple[str, ...],
) -> xr.DataArray:
    consumed = tuple(name for name in consumed_names if name in index.coords)
    return index.drop_vars(consumed) if consumed else index


def build_index_result(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamSelectOptions,
    namespace_preflight: bool = True,
) -> ParamIndexResult:
    query, consumed_names, output_plan = _preflight_index_request(
        context, query=query, opts=opts, namespace_preflight=namespace_preflight,
    )
    evaluation = prepare_runtime_param_evaluation(
        context,
        query=query,
        options=ParamMapOptions(method="nearest"),
        param_kind=context.param_kind,
        query_dim=opts.query_dim,
    )
    grid = evaluation.grid
    pmap = evaluation.param_map
    idx = _without_index_output_metadata(
        pmap.i0.where(pmap.valid, other=np.int64(-1)).astype("int64"),
        consumed_names=consumed_names,
    )
    return ParamIndexResult(
        index=idx,
        valid=pmap.valid,
        grid=grid,
        query_topology=evaluation.query_topology,
        output_plan=replace(output_plan, topology=evaluation.query_topology),
        scalar_query=_is_scalar_query(query, batch_dims=context.batch_dims),
    )


def _finalize_index_shape(result: ParamIndexResult) -> xr.DataArray:
    if result.grid.stacked_dims is not None and result.grid.query_dim in result.index.dims:
        out = restore_query_topology(
            result.index,
            plan=result.query_topology,
            owner="param index",
        )  # type: ignore[return-value]
    elif result.scalar_query and result.grid.query_dim in result.index.dims:
        out = result.index.isel({result.grid.query_dim: 0}, drop=True)
    else:
        out = result.index
    verify_query_output_plan(out, plan=result.output_plan, topology=result.query_topology)
    return out


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
