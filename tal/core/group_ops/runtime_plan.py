from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ...utils.xarray_namespace import dataset_namespace_names, unique_temp_dim
from ..orchestration.alignment import align_exact_for_plan
from ..orchestration.context import DatasetContextOptions, resolve_dataset_context
from ..orchestration.topology import (
    STRICT_EXACT_POLICY,
    SemanticTopology,
    TopologyOperand,
    resolve_unary_topology,
)
from ..validity_mask import resolve_validated_structural_mask_base
from .grouped_types import GroupByOptions, GroupedRuntimePlan
from .label_keys import canonical_group_label_key
from .row_dim_compat import require_row_dim_compatibility
from .types import GroupingFoundationContext, ResolvedGroupingKey


def _reference_alignment_array(context: GroupingFoundationContext, *, owner: str) -> xr.DataArray:
    if not context.keys:
        raise ValueError(f"{owner}: foundation context must include at least one grouping key.")
    dims = context.batch_dims + (context.sequence_dim,)
    probe = context.keys[0].data
    require_row_dim_compatibility(
        probe,
        row_dims=dims,
        owner=owner,
        what="grouping key[0]",
    )
    indexers = {dim: slice(0, 0) for dim in dims}
    return probe.transpose(*dims).isel(indexers).rename("__group_plan_ref__")


def _enforce_owner_reuse(context: GroupingFoundationContext, *, owner: str) -> None:
    _ = resolve_dataset_context(
        context.ao,
        owner=owner,
        options=DatasetContextOptions(require_roles=True, require_sequence_dim=True),
    )
    operand = TopologyOperand(
        index=0,
        data=_reference_alignment_array(context, owner=owner),
        semantic=SemanticTopology(
            sequence_dim=context.sequence_dim,
            batch_dims=context.batch_dims,
            core_dims=(),
        ),
        param_coord=context.param_coord,
    )
    plan = resolve_unary_topology(operand, owner=owner, what="grouped runtime plan", policy=STRICT_EXACT_POLICY)
    align_exact_for_plan(plan, owner=owner, what="grouped runtime plan")


def _python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _realize_row_values(data: xr.DataArray, *, row_dims: tuple[str, ...], owner: str, what: str) -> np.ndarray:
    require_row_dim_compatibility(
        data,
        row_dims=row_dims,
        owner=owner,
        what=what,
    )
    return np.asarray(data.transpose(*row_dims).data)


def _enforce_na_error_policy(
    key_values: tuple[np.ndarray, ...],
    *,
    keep_mask: np.ndarray,
    owner: str,
) -> None:
    for values in key_values:
        if bool((np.asarray(pd.isna(values)) & keep_mask).any()):
            raise ValueError(f"{owner}: grouping key domain contains NA values under na_key_policy='error'.")


def _combined_row_keep_mask(context: GroupingFoundationContext) -> xr.DataArray | None:
    structural = resolve_validated_structural_mask_base(
        context.ds,
        sequence_dim=context.sequence_dim,
        sequence_size_coord=context.sequence_size_coord,
    )
    na_exclusion = context.na_exclusion_mask
    if structural is None:
        return na_exclusion
    if na_exclusion is None:
        return structural
    return structural & na_exclusion


def _realize_row_keep_mask(
    context: GroupingFoundationContext,
    *,
    row_dims: tuple[str, ...],
    shape: tuple[int, ...],
    owner: str,
) -> np.ndarray:
    mask = _combined_row_keep_mask(context)
    if mask is None:
        return np.ones(shape, dtype=bool)
    return _realize_row_values(
        mask,
        row_dims=row_dims,
        owner=owner,
        what="grouping row keep mask",
    )


def _row_label_1d(values: tuple[np.ndarray, ...], row: int) -> object:
    if len(values) == 1:
        return _python_scalar(values[0][row])
    return tuple(_python_scalar(value[row]) for value in values)


def _row_label_2d(values: tuple[np.ndarray, ...], batch: int, sample: int) -> object:
    if len(values) == 1:
        return _python_scalar(values[0][batch, sample])
    return tuple(_python_scalar(value[batch, sample]) for value in values)


def _require_hashable_group_label(
    label: object,
    *,
    owner: str,
    what: str,
) -> None:
    try:
        hash(label)
    except TypeError as exc:
        raise ValueError(
            f"{owner}: {what} produced unhashable group label {label!r}; labels must be hashable."
        ) from exc


def _first_duplicate_domain_label(
    labels: tuple[object, ...],
    *,
    owner: str,
) -> tuple[object, tuple[int, ...]] | None:
    positions: dict[object, list[int]] = {}
    repr_labels: dict[object, object] = {}
    for index, label in enumerate(labels):
        _require_hashable_group_label(
            label,
            owner=owner,
            what=f"bin grouping domain_order[{index}]",
        )
        key = canonical_group_label_key(label)
        positions.setdefault(key, []).append(index)
        repr_labels.setdefault(key, label)
    for key, hits in positions.items():
        if len(hits) > 1:
            return repr_labels[key], tuple(hits)
    return None


def _ordered_group_labels(
    labels: tuple[object, ...],
    *,
    keys: tuple[ResolvedGroupingKey, ...],
    owner: str,
) -> tuple[object, ...]:
    if len(keys) != 1 or keys[0].kind != "bin":
        return labels
    domain_order = keys[0].domain_order
    if domain_order is None:
        return labels
    duplicate = _first_duplicate_domain_label(domain_order, owner=owner)
    if duplicate is not None:
        label, positions = duplicate
        raise ValueError(
            f"{owner}: bin grouping domain_order contains duplicate label {label!r} at positions {positions!r}."
        )
    present = {canonical_group_label_key(label) for label in labels}
    ordered_domain = tuple(label for label in domain_order if canonical_group_label_key(label) in present)
    ordered_set = {canonical_group_label_key(label) for label in ordered_domain}
    extras = tuple(label for label in labels if canonical_group_label_key(label) not in ordered_set)
    return ordered_domain + extras


def _bin_domain_labels(keys: tuple[ResolvedGroupingKey, ...]) -> tuple[object, ...] | None:
    if len(keys) != 1 or keys[0].kind != "bin":
        return None
    return keys[0].domain_order


def _reorder_global_rows(
    labels: tuple[object, ...],
    rows: tuple[tuple[int, ...], ...],
    *,
    ordered: tuple[object, ...],
) -> tuple[tuple[int, ...], ...]:
    if ordered == labels:
        return rows
    index = {canonical_group_label_key(label): idx for idx, label in enumerate(labels)}
    return tuple(rows[index[canonical_group_label_key(label)]] for label in ordered)


def _reorder_batch_rows(
    labels: tuple[object, ...],
    rows: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    ordered: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if ordered == labels:
        return rows
    index = {canonical_group_label_key(label): idx for idx, label in enumerate(labels)}
    return tuple(
        tuple(batch_rows[index[canonical_group_label_key(label)]] for label in ordered)
        for batch_rows in rows
    )


def _resolve_global_group_rows(
    values: tuple[np.ndarray, ...],
    *,
    keep_mask: np.ndarray,
    keys: tuple[ResolvedGroupingKey, ...],
    owner: str,
) -> tuple[tuple[object, ...], tuple[tuple[int, ...], ...]]:
    labels: list[object] = []
    groups: list[list[int]] = []
    index: dict[object, int] = {}
    for row, keep in enumerate(keep_mask):
        if not bool(keep):
            continue
        label = _row_label_1d(values, row)
        _require_hashable_group_label(
            label,
            owner=owner,
            what=f"grouping row[{row}]",
        )
        key = canonical_group_label_key(label)
        gid = index.get(key)
        if gid is None:
            gid = len(labels)
            index[key] = gid
            labels.append(label)
            groups.append([])
        groups[gid].append(row)
    frozen_labels = tuple(labels)
    ordered = _ordered_group_labels(frozen_labels, keys=keys, owner=owner)
    frozen_rows = tuple(tuple(rows) for rows in groups)
    return ordered, _reorder_global_rows(frozen_labels, frozen_rows, ordered=ordered)


def _resolve_batch_group_rows(
    values: tuple[np.ndarray, ...],
    *,
    keep_mask: np.ndarray,
    batch_count: int,
    sequence_size: int,
    keys: tuple[ResolvedGroupingKey, ...],
    owner: str,
) -> tuple[tuple[object, ...], tuple[tuple[tuple[int, ...], ...], ...]]:
    labels: list[object] = []
    index: dict[object, int] = {}
    groups: list[list[list[int]]] = [list() for _ in range(batch_count)]
    for batch in range(batch_count):
        for sample in range(sequence_size):
            if not bool(keep_mask[batch, sample]):
                continue
            label = _row_label_2d(values, batch, sample)
            _require_hashable_group_label(
                label,
                owner=owner,
                what=f"grouping row[{batch}, {sample}]",
            )
            key = canonical_group_label_key(label)
            gid = index.get(key)
            if gid is None:
                gid = len(labels)
                index[key] = gid
                labels.append(label)
                for batch_groups in groups:
                    batch_groups.append([])
            groups[batch][gid].append(sample)
    frozen_labels = tuple(labels)
    ordered = _ordered_group_labels(frozen_labels, keys=keys, owner=owner)
    frozen_rows = tuple(tuple(tuple(rows) for rows in batch_rows) for batch_rows in groups)
    return ordered, _reorder_batch_rows(frozen_labels, frozen_rows, ordered=ordered)


def _resolve_global_plan(
    context: GroupingFoundationContext,
    *,
    opts: GroupByOptions,
    row_dims: tuple[str, ...],
    stacked_dim: str,
    key_values: tuple[np.ndarray, ...],
    keep_mask: np.ndarray,
    owner: str,
) -> GroupedRuntimePlan:
    bin_domain = _bin_domain_labels(context.keys)
    labels, global_rows = _resolve_global_group_rows(
        tuple(value.reshape(-1) for value in key_values),
        keep_mask=keep_mask.reshape(-1),
        keys=context.keys,
        owner=owner,
    )
    return GroupedRuntimePlan(
        foundation=context,
        preserve_batch=False,
        group_dim=opts.group_dim,
        member_dim=opts.member_dim,
        sequence_index_coord=opts.sequence_index_coord,
        row_dims=row_dims,
        stacked_row_dim=stacked_dim,
        group_labels=labels,
        bin_domain_labels=bin_domain,
        global_group_rows=global_rows,
        per_batch_group_sequence_rows=None,
        batch_shape=tuple(int(context.ds.sizes[dim]) for dim in context.batch_dims),
    )


def _resolve_preserve_batch_plan(
    context: GroupingFoundationContext,
    *,
    opts: GroupByOptions,
    row_dims: tuple[str, ...],
    stacked_dim: str,
    key_values: tuple[np.ndarray, ...],
    keep_mask: np.ndarray,
    owner: str,
) -> GroupedRuntimePlan:
    bin_domain = _bin_domain_labels(context.keys)
    batch_shape = tuple(int(context.ds.sizes[dim]) for dim in context.batch_dims)
    sequence_size = int(context.ds.sizes[context.sequence_dim])
    batch_count = int(np.prod(batch_shape)) if batch_shape else 1
    labels, batch_rows = _resolve_batch_group_rows(
        tuple(value.reshape(batch_count, sequence_size) for value in key_values),
        keep_mask=keep_mask.reshape(batch_count, sequence_size),
        batch_count=batch_count,
        sequence_size=sequence_size,
        keys=context.keys,
        owner=owner,
    )
    return GroupedRuntimePlan(
        foundation=context,
        preserve_batch=True,
        group_dim=opts.group_dim,
        member_dim=opts.member_dim,
        sequence_index_coord=opts.sequence_index_coord,
        row_dims=row_dims,
        stacked_row_dim=stacked_dim,
        group_labels=labels,
        bin_domain_labels=bin_domain,
        global_group_rows=tuple(),
        per_batch_group_sequence_rows=batch_rows,
        batch_shape=batch_shape,
    )


def resolve_grouped_runtime_plan(
    context: GroupingFoundationContext,
    *,
    opts: GroupByOptions,
    owner: str,
) -> GroupedRuntimePlan:
    _enforce_owner_reuse(context, owner=owner)
    row_dims = context.batch_dims + (context.sequence_dim,)
    key_values = tuple(
        _realize_row_values(key.data, row_dims=row_dims, owner=owner, what=f"grouping key[{key.index}]")
        for key in context.keys
    )
    keep_mask = _realize_row_keep_mask(
        context,
        row_dims=row_dims,
        shape=key_values[0].shape,
        owner=owner,
    )
    if context.na_key_policy == "error":
        _enforce_na_error_policy(key_values, keep_mask=keep_mask, owner=owner)
    stacked_dim = unique_temp_dim("__tal_group_row__", taken_dims=dataset_namespace_names(context.ds))
    if not opts.preserve_batch:
        return _resolve_global_plan(
            context,
            opts=opts,
            row_dims=row_dims,
            stacked_dim=stacked_dim,
            key_values=key_values,
            keep_mask=keep_mask,
            owner=owner,
        )
    return _resolve_preserve_batch_plan(
        context,
        opts=opts,
        row_dims=row_dims,
        stacked_dim=stacked_dim,
        key_values=key_values,
        keep_mask=keep_mask,
        owner=owner,
    )


__all__ = ["resolve_grouped_runtime_plan"]
