from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..analysis_object import AnalysisObject
from ..dataset_ownership import analysis_object_dataset
from ..orchestration.finalize import rewrap_unvalidated_like
from ..orchestration.indexing import isel_rows
from ..orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from ..reducer_ops.types import ResolvedReducerRequest, WeightInput
from ..reducer_ops.weights import slice_weights_for_rows
from ..schema_read import read_roles
from .grouped_options import (
    ResolvedGroupedReducerOptions,
    validate_group_name_collision,
)
from .grouped_types import BatchGroupedRuntimePlan
from .label_plan import object_label_vector, resolve_output_label_plan


@dataclass(frozen=True)
class BatchRowPartition:
    """One scalar group label and its positions on the primary batch lane."""

    label: object
    rows: tuple[int, ...]


def resolve_batch_row_partitions(
    plan: BatchGroupedRuntimePlan,
    *,
    options: ResolvedGroupedReducerOptions,
) -> tuple[BatchRowPartition, ...]:
    """Resolve output labels to cached row partitions without a dense mask."""
    output = resolve_output_label_plan(
        plan.group_labels,
        domain=plan.bin_domain_labels,
        include_empty=options.include_empty_groups,
    )
    return tuple(
        BatchRowPartition(
            label=label,
            rows=() if position is None else plan.global_group_rows[position],
        )
        for label, position in zip(
            output.labels,
            output.observed_positions,
            strict=True,
        )
    )


def select_batch_row_partition(
    source: AnalysisObject,
    weights: WeightInput,
    *,
    primary_dim: str,
    rows: tuple[int, ...],
    ndarray_dims: tuple[str, ...],
    group_dim: str,
) -> tuple[AnalysisObject, WeightInput]:
    """Select one payload/weight partition while preserving lazy arrays."""
    indexer = np.asarray(rows, dtype=np.int64)
    reserved = analysis_object_dataset(source).assign_coords(
        {group_dim: xr.Variable((), np.int8(0))}
    )
    selected = rewrap_unvalidated_like(
        source,
        isel_rows(
            reserved,
            dim=primary_dim,
            rows=indexer,
        ),
        owner="group.batch.partition",
    )
    selected_weights = slice_weights_for_rows(
        weights,
        dim=primary_dim,
        rows=indexer,
        ndarray_dims=ndarray_dims,
    )
    return selected, selected_weights


def _partition_result_dataset(
    output: AnalysisObject,
    *,
    group_dim: str,
) -> xr.Dataset:
    """Remove the scalar name reservation retained by a bound reducer."""
    ds = analysis_object_dataset(output)
    if group_dim not in ds.coords or ds.coords[group_dim].dims:
        return ds
    return ds.drop_vars(group_dim)


def _group_coordinate(
    partitions: tuple[BatchRowPartition, ...],
    *,
    group_dim: str,
) -> xr.DataArray:
    labels = object_label_vector(tuple(partition.label for partition in partitions))
    return xr.DataArray(labels, dims=(group_dim,), name=group_dim)


def _expected_grouped_dims(
    source: xr.DataArray,
    *,
    primary_dim: str,
    group_dim: str,
    reduce_dims: tuple[str, ...],
) -> tuple[str, ...]:
    reduced = set(reduce_dims)
    return tuple(
        group_dim if dim == primary_dim else dim
        for dim in source.dims
        if dim not in reduced or dim == primary_dim
    )


def _transpose_grouped_variables(
    candidate: xr.Dataset,
    source: xr.Dataset,
    *,
    names: tuple[str, ...],
    primary_dim: str,
    group_dim: str,
    reduce_dims: tuple[str, ...],
) -> xr.Dataset:
    updates: dict[str, xr.DataArray] = {}
    for name in names:
        expected = _expected_grouped_dims(
            source[name],
            primary_dim=primary_dim,
            group_dim=group_dim,
            reduce_dims=reduce_dims,
        )
        expected = tuple(dim for dim in expected if dim in candidate[name].dims)
        extra = tuple(dim for dim in candidate[name].dims if dim not in expected)
        updates[name] = candidate[name].transpose(*expected, *extra)
    return candidate.assign(updates)


def _concat_partition_outputs(
    outputs: tuple[AnalysisObject, ...],
    partitions: tuple[BatchRowPartition, ...],
    *,
    group_dim: str,
    dependent_names: tuple[str, ...],
) -> xr.Dataset:
    datasets = tuple(
        _partition_result_dataset(output, group_dim=group_dim)
        for output in outputs
    )
    return xr.concat(
        datasets,
        dim=_group_coordinate(partitions, group_dim=group_dim),
        data_vars=list(dependent_names),
        coords="minimal",
        compat="equals",
        join="exact",
        combine_attrs="override",
    )


def _empty_partition_output(
    prototype: AnalysisObject,
    *,
    group_dim: str,
    dependent_names: tuple[str, ...],
) -> xr.Dataset:
    source = _partition_result_dataset(prototype, group_dim=group_dim)
    group_coord = xr.DataArray(
        object_label_vector(()),
        dims=(group_dim,),
        name=group_dim,
    )
    independent = source.drop_vars(dependent_names)
    expanded = {
        name: source[name].expand_dims({group_dim: group_coord})
        for name in dependent_names
    }
    return independent.assign(expanded).assign_coords({group_dim: group_coord})


def _assemble_partition_dataset(
    partitions: tuple[BatchRowPartition, ...],
    outputs: tuple[AnalysisObject, ...],
    *,
    prototype: AnalysisObject,
    group_dim: str,
    dependent_names: tuple[str, ...],
) -> xr.Dataset:
    if outputs:
        return _concat_partition_outputs(
            outputs,
            partitions,
            group_dim=group_dim,
            dependent_names=dependent_names,
        )
    return _empty_partition_output(
        prototype,
        group_dim=group_dim,
        dependent_names=dependent_names,
    )


def _batch_output_spec(
    prototype: AnalysisObject,
    *,
    group_dim: str,
) -> CoreSchemaFinalizeSpec:
    _, _, batch_dims, core_dims = read_roles(analysis_object_dataset(prototype))
    return CoreSchemaFinalizeSpec(
        sequence_dim=None,
        batch_dims=(group_dim, *batch_dims),
        core_dims=core_dims,
        param_name=None,
        size_name=None,
    )


def _primary_dependent_names(
    plan: BatchGroupedRuntimePlan,
    request: ResolvedReducerRequest,
) -> tuple[str, ...]:
    primary_dim = plan.foundation.primary_batch_dim
    return tuple(
        name
        for name in request.eligible_names
        if primary_dim in plan.foundation.ds[name].dims
    )


def assemble_batch_partition_outputs(
    plan: BatchGroupedRuntimePlan,
    partitions: tuple[BatchRowPartition, ...],
    outputs: tuple[AnalysisObject, ...],
    *,
    prototype: AnalysisObject,
    request: ResolvedReducerRequest,
    options: ResolvedGroupedReducerOptions,
    finalize_source: AnalysisObject,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    """Assemble partition reductions and finalize batch-group topology once."""
    foundation = plan.foundation
    primary_dim = foundation.primary_batch_dim
    dependent = _primary_dependent_names(plan, request)
    prototype_ds = _partition_result_dataset(
        prototype,
        group_dim=options.group_dim,
    )
    validate_group_name_collision(
        ds=prototype_ds,
        group_dim=options.group_dim,
        owner=owner,
    )
    candidate = _assemble_partition_dataset(
        partitions,
        outputs,
        prototype=prototype,
        group_dim=options.group_dim,
        dependent_names=dependent,
    )
    candidate = _transpose_grouped_variables(
        candidate,
        foundation.ds,
        names=dependent,
        primary_dim=primary_dim,
        group_dim=options.group_dim,
        reduce_dims=request.reduce_dims,
    )
    return finalize_with_schema(
        finalize_source,
        candidate,
        spec=_batch_output_spec(
            prototype,
            group_dim=options.group_dim,
        ),
        validate=validate,
        owner=owner,
        optional_sources=tuple(foundation.ds.coords.values()),
    )


__all__ = [
    "BatchRowPartition",
    "assemble_batch_partition_outputs",
    "resolve_batch_row_partitions",
    "select_batch_row_partition",
]
