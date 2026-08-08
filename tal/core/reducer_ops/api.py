from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from ..orchestration.context import DatasetContextOptions, resolve_dataset_context
from ..orchestration.finalize import finalize_like
from ..orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from ..validity_mask import resolve_validated_structural_mask_base
from .dims import resolve_reduce_dims
from .finalize_policy import resolve_reducer_finalize_source
from .kernel import reduce_dataarray
from .types import DimLike, WeightInput, require_supported_op
from .vars import select_eligible_var_names
from .weights import require_no_unsupported_weights

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


def _component_dims(source: "AnalysisObject") -> tuple[str, ...]:
    resolver = getattr(source, "_required_component_dims_for_reduce", None)
    if resolver is None:
        return ()
    out = resolver()
    return tuple(out) if out is not None else ()


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
    source: "AnalysisObject",
    out: xr.Dataset,
    reducer: str,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
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


def _reduce_named_data_vars(
    *,
    ds: xr.Dataset,
    names: tuple[str, ...],
    reduce_dims: tuple[str, ...],
    sequence_dim: str | None,
    sequence_size_coord: str | None,
    reducer: str,
    skipna: bool,
    ddof: int,
    weights: WeightInput,
    owner: str,
) -> dict[str, xr.DataArray]:
    reduced: dict[str, xr.DataArray] = {}
    base_mask = None
    if sequence_dim is not None and any(sequence_dim in ds[name].dims for name in names):
        base_mask = resolve_validated_structural_mask_base(
            ds,
            sequence_dim=sequence_dim,
            sequence_size_coord=sequence_size_coord,
        )
    for name in names:
        data = ds[name]
        var_dims = tuple(dim_name for dim_name in reduce_dims if dim_name in data.dims)
        mask = None
        if base_mask is not None and sequence_dim is not None and sequence_dim in data.dims:
            mask = base_mask.broadcast_like(data)
        reduced[name] = reduce_dataarray(
            data,
            op=reducer,
            reduce_dims=var_dims,
            skipna=skipna,
            ddof=ddof,
            weights=weights,
            mask=mask,
            owner=f"{owner}.{name}",
        )
    return reduced


def reduce_analysis_object(
    source: "AnalysisObject",
    *,
    op: str,
    dim: DimLike,
    skipna: bool,
    ddof: int,
    weights: WeightInput,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    reducer = require_supported_op(op, owner=owner)
    require_no_unsupported_weights(weights=weights, op=reducer, owner=owner)
    context = resolve_dataset_context(
        source,
        owner=owner,
        options=DatasetContextOptions(),
    )
    ds = context.ds
    reduce_dims = resolve_reduce_dims(ds, dim=dim, component_dims=_component_dims(source), owner=owner)
    names = tuple(select_eligible_var_names(ds, op=reducer, owner=owner))
    reduced = _reduce_named_data_vars(
        ds=ds,
        names=names,
        reduce_dims=reduce_dims,
        sequence_dim=context.sequence_dim,
        sequence_size_coord=context.sequence_size_coord,
        reducer=reducer,
        skipna=skipna,
        ddof=ddof,
        weights=weights,
        owner=owner,
    )
    out = xr.Dataset(reduced)
    return _finalize_reduced_output(
        context=context,
        reduce_dims=reduce_dims,
        source=source,
        out=out,
        reducer=reducer,
        validate=validate,
        owner=owner,
    )


__all__ = ["reduce_analysis_object"]
