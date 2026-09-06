from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr

from ..orchestration.context import (
    DatasetContext,
    DatasetContextOptions,
    resolve_dataset_context,
)
from ..orchestration.finalize import finalize_like
from ..orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from ..validity_mask import resolve_validated_structural_mask_base
from .dims import resolve_active_reduce_dims, resolve_reduce_dims
from .finalize_policy import resolve_reducer_finalize_source
from .kernel import reduce_dataarray
from .types import (
    DimLike,
    ReducerOp,
    ResolvedReducerRequest,
    WeightInput,
    require_supported_op,
)
from .vars import select_eligible_var_names
from .weights import preflight_weight_structure, require_no_unsupported_weights

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


@dataclass(frozen=True)
class _ReducerExecutionPlan:
    reduce_dims: tuple[str, ...]
    active_reduce_dims: tuple[str, ...]
    sequence_dim: str | None
    sequence_size_coord: str | None
    reducer: ReducerOp
    skipna: bool
    ddof: int
    weights: WeightInput
    base_mask: xr.DataArray | None
    mask_dim: str | None


def _component_dims(source: AnalysisObject) -> tuple[str, ...]:
    resolver = getattr(source, "_required_component_dims_for_reduce", None)
    if resolver is None:
        return ()
    out = resolver()
    return tuple(out) if out is not None else ()


def resolve_reducer_dims_for_source(
    source: AnalysisObject,
    ds: xr.Dataset,
    *,
    dim: DimLike,
    owner: str,
) -> tuple[str, ...]:
    """Resolve reducer dimensions with the source's component restrictions."""
    return resolve_reduce_dims(
        ds,
        dim=dim,
        component_dims=_component_dims(source),
        owner=owner,
    )


def resolve_reducer_request(
    source: AnalysisObject,
    ds: xr.Dataset,
    *,
    op: str,
    dim: DimLike,
    weights: WeightInput,
    owner: str,
) -> ResolvedReducerRequest:
    """Resolve reducer metadata without realizing payloads or weights."""
    reducer = require_supported_op(op, owner=owner)
    require_no_unsupported_weights(weights=weights, op=reducer, owner=owner)
    reduce_dims = resolve_reducer_dims_for_source(source, ds, dim=dim, owner=owner)
    names = tuple(select_eligible_var_names(ds, op=reducer, owner=owner))
    active_reduce_dims = resolve_active_reduce_dims(
        ds,
        names=names,
        reduce_dims=reduce_dims,
    )
    preflight_weight_structure(
        ds,
        names=names,
        reduce_dims=reduce_dims,
        active_dims=active_reduce_dims,
        weights=weights,
        owner=owner,
    )
    return ResolvedReducerRequest(
        reducer=reducer,
        reduce_dims=reduce_dims,
        eligible_names=names,
        active_reduce_dims=active_reduce_dims,
    )


def _reduced_schema_spec(
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, ...],
    param_coord: str | None,
    sequence_size_coord: str | None,
    reduce_dims: tuple[str, ...],
) -> CoreSchemaFinalizeSpec:
    reduced = set(reduce_dims)
    sequence_out = sequence_dim if sequence_dim not in reduced else None
    batch_out = tuple(dim for dim in batch_dims if dim not in reduced)
    core_out = tuple(dim for dim in core_dims if dim not in reduced)
    return CoreSchemaFinalizeSpec(
        sequence_dim=sequence_out,
        batch_dims=batch_out,
        core_dims=core_out,
        param_name=param_coord if sequence_out is not None else None,
        size_name=sequence_size_coord if sequence_out is not None else None,
    )


def _finalize_reduced_output(
    *,
    context,
    reduce_dims: tuple[str, ...],
    source: AnalysisObject,
    out: xr.Dataset,
    reducer: str,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    finalize_source = resolve_reducer_finalize_source(source, op=reducer, owner=owner)
    if not context.roles_declared:
        return finalize_like(finalize_source, out, validate=validate, owner=owner)
    spec = _reduced_schema_spec(
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        core_dims=context.core_dims,
        param_coord=context.param_coord,
        sequence_size_coord=context.sequence_size_coord,
        reduce_dims=reduce_dims,
    )
    optional_sources = tuple(coord for _, coord in context.ds.coords.items())
    return finalize_with_schema(
        finalize_source,
        out,
        spec=spec,
        validate=validate,
        owner=owner,
        optional_sources=optional_sources,
    )


def _needs_sequence_mask(
    ds: xr.Dataset,
    *,
    names: tuple[str, ...],
    plan: _ReducerExecutionPlan,
) -> bool:
    if plan.sequence_dim is None:
        return False
    return any(
        plan.sequence_dim in ds[name].dims
        and any(dim in ds[name].dims for dim in plan.active_reduce_dims)
        for name in names
    )


def _reduce_named_data_vars(
    *,
    ds: xr.Dataset,
    names: tuple[str, ...],
    plan: _ReducerExecutionPlan,
    owner: str,
) -> dict[str, xr.DataArray]:
    reduced: dict[str, xr.DataArray] = {}
    base_mask = plan.base_mask
    mask_dim = plan.mask_dim
    if base_mask is None and _needs_sequence_mask(ds, names=names, plan=plan):
        base_mask = resolve_validated_structural_mask_base(
            ds,
            sequence_dim=plan.sequence_dim,
            sequence_size_coord=plan.sequence_size_coord,
        )
        mask_dim = plan.sequence_dim
    for name in names:
        data = ds[name]
        var_dims = tuple(
            dim_name
            for dim_name in plan.active_reduce_dims
            if dim_name in data.dims
        )
        mask = None
        if base_mask is not None and mask_dim is not None and mask_dim in data.dims:
            mask = base_mask
        reduced[name] = reduce_dataarray(
            data,
            op=plan.reducer,
            reduce_dims=var_dims,
            skipna=plan.skipna,
            ddof=plan.ddof,
            weights=plan.weights,
            mask=mask,
            owner=f"{owner}.{name}",
        )
    return reduced


def assemble_reduced_dataset(
    ds: xr.Dataset,
    reduced: dict[str, xr.DataArray],
    *,
    reduce_dims: tuple[str, ...],
    sequence_dim: str | None,
    param_coord: str | None,
    sequence_size_coord: str | None,
) -> xr.Dataset:
    """Attach reduced variables to the source coordinate-only topology."""
    out = ds.drop_vars(tuple(ds.data_vars)).assign(reduced)
    reduced_dims = set(reduce_dims)
    invalid_optional: set[str] = set()
    if sequence_dim is not None and sequence_dim in reduced_dims:
        invalid_optional.update(name for name in (param_coord, sequence_size_coord) if name is not None)
    drop = tuple(
        name
        for name, coord in out.coords.items()
        if name in invalid_optional or any(dim in reduced_dims for dim in coord.dims)
    )
    return out.drop_vars(drop) if drop else out


def _execute_reduction(
    source: AnalysisObject,
    *,
    context: DatasetContext,
    names: tuple[str, ...],
    plan: _ReducerExecutionPlan,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    reduced = _reduce_named_data_vars(
        ds=context.ds,
        names=names,
        plan=plan,
        owner=owner,
    )
    out = assemble_reduced_dataset(
        context.ds,
        reduced,
        reduce_dims=plan.reduce_dims,
        sequence_dim=context.sequence_dim,
        param_coord=context.param_coord,
        sequence_size_coord=context.sequence_size_coord,
    )
    return _finalize_reduced_output(
        context=context,
        reduce_dims=plan.reduce_dims,
        source=source,
        out=out,
        reducer=plan.reducer,
        validate=validate,
        owner=owner,
    )


def reduce_analysis_object(
    source: AnalysisObject,
    *,
    op: str,
    dim: DimLike,
    skipna: bool,
    ddof: int,
    weights: WeightInput,
    validate: bool,
    owner: str,
    base_mask: xr.DataArray | None = None,
    mask_dim: str | None = None,
) -> AnalysisObject:
    context = resolve_dataset_context(
        source,
        owner=owner,
        options=DatasetContextOptions(),
    )
    ds = context.ds
    request = resolve_reducer_request(
        source,
        ds,
        op=op,
        dim=dim,
        weights=weights,
        owner=owner,
    )
    plan = _ReducerExecutionPlan(
        reduce_dims=request.reduce_dims,
        active_reduce_dims=request.active_reduce_dims,
        sequence_dim=context.sequence_dim,
        sequence_size_coord=context.sequence_size_coord,
        reducer=request.reducer,
        skipna=skipna,
        ddof=ddof,
        weights=weights,
        base_mask=base_mask,
        mask_dim=mask_dim,
    )
    return _execute_reduction(
        source,
        context=context,
        names=request.eligible_names,
        plan=plan,
        validate=validate,
        owner=owner,
    )


__all__ = [
    "assemble_reduced_dataset",
    "reduce_analysis_object",
    "resolve_reducer_dims_for_source",
    "resolve_reducer_request",
]
