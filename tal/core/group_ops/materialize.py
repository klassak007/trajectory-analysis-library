from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ...utils.xarray_namespace import dataset_namespace_names, unique_temp_dim
from ..orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from .grouped_options import resolve_layout_names, validate_layout_name_collisions
from .grouped_types import GroupedRuntimePlan, GroupMaterializeOptions
from .label_plan import object_label_vector, resolve_output_label_plan

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


def _stack_source_rows(plan: GroupedRuntimePlan) -> xr.Dataset:
    return plan.foundation.ds.stack({plan.stacked_row_dim: list(plan.row_dims)}, create_index=False)


def _output_sequence_coord(*, sequence_dim: str, size: int) -> xr.DataArray:
    return xr.DataArray(np.arange(size, dtype=np.int64), dims=(sequence_dim,), name=sequence_dim)


def _group_labels_coord(labels: tuple[object, ...], *, group_dim: str) -> xr.DataArray:
    values = object_label_vector(labels)
    return xr.DataArray(values, dims=(group_dim,), name=group_dim)


def _dense_group_rows(
    group_rows: tuple[tuple[int, ...], ...],
    *,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    group_count = len(group_rows)
    indices = np.zeros((group_count, width), dtype=np.int64)
    valid = np.zeros((group_count, width), dtype=bool)
    for group_index, rows in enumerate(group_rows):
        limit = min(len(rows), width)
        if limit == 0:
            continue
        indices[group_index, :limit] = np.asarray(rows[:limit], dtype=np.int64)
        valid[group_index, :limit] = True
    return indices, valid


def _dense_preserve_padded_rows(
    batch_rows: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    group_count: int,
    sequence_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_count = len(batch_rows)
    indices = np.zeros((batch_count, group_count, sequence_size), dtype=np.int64)
    valid = np.zeros((batch_count, group_count, sequence_size), dtype=bool)
    for batch_index, per_group in enumerate(batch_rows):
        batch_indices, batch_valid = _dense_group_rows(
            per_group,
            width=sequence_size,
        )
        indices[batch_index] = batch_indices
        valid[batch_index] = batch_valid
    return indices, valid


def _dense_preserve_stacked_rows(
    batch_rows: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    group_labels: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged_rows = [tuple(row for rows in per_group for row in rows) for per_group in batch_rows]
    merged_labels = [tuple(label for label, rows in zip(group_labels, per_group, strict=True) for _ in rows) for per_group in batch_rows]
    width = max((len(rows) for rows in merged_rows), default=0)
    batch_count = len(batch_rows)
    indices = np.zeros((batch_count, width), dtype=np.int64)
    valid = np.zeros((batch_count, width), dtype=bool)
    labels = np.empty((batch_count, width), dtype=object)
    labels[:] = np.nan
    for batch_index, rows in enumerate(merged_rows):
        limit = len(rows)
        if limit == 0:
            continue
        indices[batch_index, :limit] = np.asarray(rows, dtype=np.int64)
        valid[batch_index, :limit] = True
        labels[batch_index, :limit] = object_label_vector(merged_labels[batch_index])
    return indices, valid, labels


def _size_coord_name(plan: GroupedRuntimePlan, ds: xr.Dataset) -> str:
    base = plan.foundation.sequence_size_coord or "group_size"
    return unique_temp_dim(base, taken_dims=dataset_namespace_names(ds))


def _finalize(
    plan: GroupedRuntimePlan,
    ds: xr.Dataset,
    *,
    source_ao: AnalysisObject,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    sequence_size_coord: str | None,
    validate: bool,
    owner: str,
):
    spec = CoreSchemaFinalizeSpec(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=plan.foundation.core_dims,
        param_name=None,
        size_name=sequence_size_coord,
    )
    return finalize_with_schema(source_ao, ds, spec=spec, validate=validate, owner=owner)


def _reshape_batch_grid(values: np.ndarray, *, batch_shape: tuple[int, ...]) -> np.ndarray:
    if batch_shape:
        return values.reshape(batch_shape + values.shape[1:])
    return values.reshape(values.shape[1:])


def _stacked_group_rows(plan: GroupedRuntimePlan) -> tuple[np.ndarray, np.ndarray]:
    rows = tuple(row for group_rows in plan.global_group_rows for row in group_rows)
    labels = tuple(label for label, group_rows in zip(plan.group_labels, plan.global_group_rows, strict=True) for _ in group_rows)
    return np.asarray(rows, dtype=np.int64), object_label_vector(labels)


def _effective_global_padded_groups(
    plan: GroupedRuntimePlan,
    *,
    include_empty_groups: bool,
) -> tuple[tuple[object, ...], tuple[tuple[int, ...], ...]]:
    output = resolve_output_label_plan(
        plan.group_labels,
        domain=plan.bin_domain_labels,
        include_empty=include_empty_groups,
    )
    rows = tuple(
        () if position is None else plan.global_group_rows[position]
        for position in output.observed_positions
    )
    return output.labels, rows


def _effective_batch_padded_groups(
    plan: GroupedRuntimePlan,
    *,
    include_empty_groups: bool,
) -> tuple[tuple[object, ...], tuple[tuple[tuple[int, ...], ...], ...]]:
    assert plan.per_batch_group_sequence_rows is not None
    output = resolve_output_label_plan(
        plan.group_labels,
        domain=plan.bin_domain_labels,
        include_empty=include_empty_groups,
    )
    batch_rows = tuple(
        tuple(
            () if position is None else per_group[position]
            for position in output.observed_positions
        )
        for per_group in plan.per_batch_group_sequence_rows
    )
    return output.labels, batch_rows


def _gather_padded_rows(
    ds: xr.Dataset,
    *,
    row_dim: str,
    member_dim: str,
    indexer: xr.DataArray,
) -> xr.Dataset:
    """Build empty target topology without indexing zero-sized dimensions."""
    if all(size != 0 for size in indexer.sizes.values()):
        return ds.isel({row_dim: indexer})
    width = int(indexer.sizes[member_dim])
    base = ds.isel({row_dim: slice(0, width)}).rename_dims({row_dim: member_dim})
    expanded = base.broadcast_like(indexer)
    data_vars = {
        name: ds[name]
        for name in ds.data_vars
        if row_dim not in ds[name].dims
    }
    coords = {
        name: ds.coords[name]
        for name in ds.coords
        if row_dim not in ds.coords[name].dims
    }
    return expanded.assign(data_vars).assign_coords(coords)


def _materialize_padded_default(
    plan: GroupedRuntimePlan,
    *,
    source_ao: AnalysisObject,
    group_dim: str,
    member_dim: str,
    include_empty_groups: bool,
    validate: bool,
    owner: str,
):
    sequence_dim = plan.foundation.sequence_dim
    labels, group_rows = _effective_global_padded_groups(plan, include_empty_groups=include_empty_groups)
    width = max((len(rows) for rows in group_rows), default=0)
    indices, valid = _dense_group_rows(group_rows, width=width)
    indexer = xr.DataArray(indices, dims=(group_dim, member_dim), coords={group_dim: _group_labels_coord(labels, group_dim=group_dim)})
    valid_mask = xr.DataArray(valid, dims=(group_dim, member_dim), coords=indexer.coords)
    gathered = _gather_padded_rows(
        _stack_source_rows(plan), row_dim=plan.stacked_row_dim,
        member_dim=member_dim, indexer=indexer,
    ).where(valid_mask)
    gathered = gathered.drop_vars(list(plan.row_dims), errors="ignore").rename({member_dim: sequence_dim})
    gathered = gathered.assign_coords({sequence_dim: _output_sequence_coord(sequence_dim=sequence_dim, size=width)})
    size_name = _size_coord_name(plan, gathered)
    grouped_size = valid_mask.rename({member_dim: sequence_dim}).sum(dim=sequence_dim).astype(np.int64)
    return _finalize(
        plan,
        gathered.assign_coords({size_name: grouped_size}),
        source_ao=source_ao,
        sequence_dim=sequence_dim,
        batch_dims=(group_dim,),
        sequence_size_coord=size_name,
        validate=validate,
        owner=owner,
    )


def _materialize_padded_preserve_batch(
    plan: GroupedRuntimePlan,
    *,
    source_ao: AnalysisObject,
    group_dim: str,
    member_dim: str,
    include_empty_groups: bool,
    validate: bool,
    owner: str,
):
    labels, batch_rows = _effective_batch_padded_groups(plan, include_empty_groups=include_empty_groups)
    sequence_dim = plan.foundation.sequence_dim
    sequence_size = int(plan.foundation.ds.sizes[sequence_dim])
    indices, valid = _dense_preserve_padded_rows(
        batch_rows,
        group_count=len(labels),
        sequence_size=sequence_size,
    )
    indices = _reshape_batch_grid(indices, batch_shape=plan.batch_shape)
    valid = _reshape_batch_grid(valid, batch_shape=plan.batch_shape)
    dims = plan.foundation.batch_dims + (group_dim, member_dim)
    coords = {dim: plan.foundation.ds.coords[dim] for dim in plan.foundation.batch_dims if dim in plan.foundation.ds.coords}
    coords[group_dim] = _group_labels_coord(labels, group_dim=group_dim)
    indexer = xr.DataArray(indices, dims=dims, coords=coords)
    valid_mask = xr.DataArray(valid, dims=dims, coords=coords)
    gathered = _gather_padded_rows(
        plan.foundation.ds, row_dim=sequence_dim, member_dim=member_dim, indexer=indexer,
    ).where(valid_mask)
    gathered = gathered.drop_vars(sequence_dim, errors="ignore").rename({member_dim: sequence_dim})
    gathered = gathered.assign_coords({sequence_dim: _output_sequence_coord(sequence_dim=sequence_dim, size=sequence_size)})
    size_name = _size_coord_name(plan, gathered)
    grouped_size = valid_mask.rename({member_dim: sequence_dim}).sum(dim=sequence_dim).astype(np.int64)
    return _finalize(
        plan,
        gathered.assign_coords({size_name: grouped_size}),
        source_ao=source_ao,
        sequence_dim=sequence_dim,
        batch_dims=plan.foundation.batch_dims + (group_dim,),
        sequence_size_coord=size_name,
        validate=validate,
        owner=owner,
    )


def _materialize_stacked_default(
    plan: GroupedRuntimePlan,
    *,
    source_ao: AnalysisObject,
    group_dim: str,
    member_dim: str,
    sequence_index_coord: str,
    validate: bool,
    owner: str,
):
    sequence_dim = plan.foundation.sequence_dim
    rows, labels = _stacked_group_rows(plan)
    indexer = xr.DataArray(rows, dims=(member_dim,))
    gathered = _stack_source_rows(plan).isel({plan.stacked_row_dim: indexer})
    seq_index = gathered.coords[sequence_dim].rename(sequence_index_coord)
    out = gathered.drop_vars(sequence_dim, errors="ignore").assign_coords({group_dim: xr.DataArray(labels, dims=(member_dim,)), sequence_index_coord: seq_index})
    return _finalize(
        plan,
        out,
        source_ao=source_ao,
        sequence_dim=member_dim,
        batch_dims=(),
        sequence_size_coord=None,
        validate=validate,
        owner=owner,
    )


def _materialize_stacked_preserve_batch(
    plan: GroupedRuntimePlan,
    *,
    source_ao: AnalysisObject,
    group_dim: str,
    member_dim: str,
    sequence_index_coord: str,
    validate: bool,
    owner: str,
):
    assert plan.per_batch_group_sequence_rows is not None
    sequence_dim = plan.foundation.sequence_dim
    indices, valid, labels = _dense_preserve_stacked_rows(plan.per_batch_group_sequence_rows, group_labels=plan.group_labels)
    indices = _reshape_batch_grid(indices, batch_shape=plan.batch_shape)
    valid = _reshape_batch_grid(valid, batch_shape=plan.batch_shape)
    labels = _reshape_batch_grid(labels, batch_shape=plan.batch_shape)
    dims = plan.foundation.batch_dims + (member_dim,)
    coords = {dim: plan.foundation.ds.coords[dim] for dim in plan.foundation.batch_dims if dim in plan.foundation.ds.coords}
    indexer = xr.DataArray(indices, dims=dims, coords=coords)
    valid_mask = xr.DataArray(valid, dims=dims, coords=coords)
    gathered = _gather_padded_rows(
        plan.foundation.ds,
        row_dim=sequence_dim,
        member_dim=member_dim,
        indexer=indexer,
    ).where(valid_mask)
    seq_index = xr.where(valid_mask, gathered.coords[sequence_dim], np.nan).rename(sequence_index_coord)
    out = gathered.drop_vars(sequence_dim, errors="ignore").assign_coords(
        {
            group_dim: xr.DataArray(labels, dims=dims, coords=coords),
            sequence_index_coord: seq_index,
        }
    )
    return _finalize(
        plan,
        out,
        source_ao=source_ao,
        sequence_dim=member_dim,
        batch_dims=plan.foundation.batch_dims,
        sequence_size_coord=None,
        validate=validate,
        owner=owner,
    )


def _materialize_padded(
    plan: GroupedRuntimePlan,
    *,
    source_ao: AnalysisObject,
    group_dim: str,
    member_dim: str,
    include_empty_groups: bool,
    validate: bool,
    owner: str,
):
    if not plan.preserve_batch:
        return _materialize_padded_default(
            plan,
            source_ao=source_ao,
            group_dim=group_dim,
            member_dim=member_dim,
            include_empty_groups=include_empty_groups,
            validate=validate,
            owner=owner,
        )
    return _materialize_padded_preserve_batch(
        plan,
        source_ao=source_ao,
        group_dim=group_dim,
        member_dim=member_dim,
        include_empty_groups=include_empty_groups,
        validate=validate,
        owner=owner,
    )


def _materialize_stacked(
    plan: GroupedRuntimePlan,
    *,
    source_ao: AnalysisObject,
    group_dim: str,
    member_dim: str,
    sequence_index_coord: str,
    validate: bool,
    owner: str,
):
    if not plan.preserve_batch:
        return _materialize_stacked_default(
            plan,
            source_ao=source_ao,
            group_dim=group_dim,
            member_dim=member_dim,
            sequence_index_coord=sequence_index_coord,
            validate=validate,
            owner=owner,
        )
    return _materialize_stacked_preserve_batch(
        plan,
        source_ao=source_ao,
        group_dim=group_dim,
        member_dim=member_dim,
        sequence_index_coord=sequence_index_coord,
        validate=validate,
        owner=owner,
    )


def materialize_grouped_view(
    plan: GroupedRuntimePlan,
    *,
    opts: GroupMaterializeOptions,
    validate: bool,
    owner: str,
    source_ao: AnalysisObject | None = None,
):
    source = plan.foundation.ao if source_ao is None else source_ao
    group_dim, member_dim, sequence_index_coord = resolve_layout_names(
        group_dim=plan.group_dim,
        member_dim=plan.member_dim,
        sequence_index_coord=plan.sequence_index_coord,
        opts=opts,
        owner=owner,
    )
    validate_layout_name_collisions(
        ds=plan.foundation.ds,
        group_dim=group_dim,
        member_dim=member_dim,
        sequence_index_coord=sequence_index_coord,
        owner=owner,
    )
    if opts.layout == "padded":
        return _materialize_padded(
            plan,
            source_ao=source,
            group_dim=group_dim,
            member_dim=member_dim,
            include_empty_groups=opts.include_empty_groups,
            validate=validate,
            owner=owner,
        )
    return _materialize_stacked(
        plan,
        source_ao=source,
        group_dim=group_dim,
        member_dim=member_dim,
        sequence_index_coord=sequence_index_coord,
        validate=validate,
        owner=owner,
    )


__all__ = ["materialize_grouped_view"]
