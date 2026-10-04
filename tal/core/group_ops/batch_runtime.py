from __future__ import annotations

import numpy as np
import pandas as pd

from .grouped_types import BatchGroupedRuntimePlan, GroupByOptions
from .label_plan import (
    bin_domain_labels,
    enforce_na_error_policy,
    realize_grouping_values,
    replace_na_group_values,
    resolve_global_group_rows,
)
from .types import BatchGroupingFoundationContext


def _realize_keys(
    context: BatchGroupingFoundationContext,
    *,
    owner: str,
) -> tuple[np.ndarray, ...]:
    row_dims = (context.primary_batch_dim,)
    return tuple(
        realize_grouping_values(
            key.data,
            row_dims=row_dims,
            owner=owner,
            what=f"grouping key[{key.index}]",
        )
        for key in context.keys
    )


def _combined_na_mask(values: tuple[np.ndarray, ...]) -> np.ndarray:
    masks = tuple(np.asarray(pd.isna(value), dtype=bool) for value in values)
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined |= mask
    return combined


def _replace_na_values(
    values: tuple[np.ndarray, ...],
    *,
    label: object,
    owner: str,
) -> tuple[np.ndarray, ...]:
    return tuple(
        replace_na_group_values(
            value,
            active_mask=np.ones(value.shape, dtype=bool),
            label=label,
            owner=owner,
            what=f"grouping key[{index}]",
        )
        for index, value in enumerate(values)
    )


def _apply_na_policy(
    context: BatchGroupingFoundationContext,
    values: tuple[np.ndarray, ...],
    *,
    owner: str,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    combined = _combined_na_mask(values)
    keep = np.ones(combined.shape, dtype=bool)
    if context.na_key_policy == "error":
        enforce_na_error_policy(values, keep_mask=keep, owner=owner)
        return values, keep
    if context.na_key_policy == "drop":
        return values, ~combined
    assert context.na_group_label is not None
    return _replace_na_values(
        values,
        label=context.na_group_label,
        owner=owner,
    ), keep


def resolve_batch_grouped_runtime_plan(
    context: BatchGroupingFoundationContext,
    *,
    opts: GroupByOptions,
    owner: str,
) -> BatchGroupedRuntimePlan:
    """Realize batch grouping keys and construct cached row partitions."""
    values, keep = _apply_na_policy(context, _realize_keys(context, owner=owner), owner=owner)
    labels, rows = resolve_global_group_rows(
        values,
        keep_mask=keep,
        keys=context.keys,
        owner=owner,
    )
    return BatchGroupedRuntimePlan(
        foundation=context,
        group_dim=opts.group_dim,
        group_labels=labels,
        bin_domain_labels=bin_domain_labels(context.keys),
        global_group_rows=rows,
    )


__all__ = ["resolve_batch_grouped_runtime_plan"]
