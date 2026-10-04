from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .label_keys import canonical_group_label_key
from .row_dim_compat import require_row_dim_compatibility
from .types import ResolvedGroupingKey


@dataclass(frozen=True)
class GroupOutputLabelPlan:
    """Resolved output labels and their positions in the observed plan."""

    labels: tuple[object, ...]
    observed_positions: tuple[int | None, ...]


def object_label_vector(labels: Sequence[object]) -> np.ndarray:
    """Build a one-dimensional object array without expanding tuple labels."""
    values = np.empty(len(labels), dtype=object)
    for index, label in enumerate(labels):
        values[index] = label
    return values


def object_label_array(values: np.ndarray) -> np.ndarray:
    """Copy an array into object storage without coercing scalar label types."""
    out = np.empty(values.shape, dtype=object)
    for index, value in enumerate(values.flat):
        out.flat[index] = normalize_group_label_scalar(value)
    return out


def realize_grouping_values(
    data: xr.DataArray,
    *,
    row_dims: tuple[str, ...],
    owner: str,
    what: str,
) -> np.ndarray:
    """Realize only a declared grouping key or grouping-derived mask."""
    require_row_dim_compatibility(data, row_dims=row_dims, owner=owner, what=what)
    return np.asarray(data.transpose(*row_dims).data)


def enforce_na_error_policy(
    key_values: tuple[np.ndarray, ...],
    *,
    keep_mask: np.ndarray,
    owner: str,
) -> None:
    for values in key_values:
        if bool((np.asarray(pd.isna(values)) & keep_mask).any()):
            raise ValueError(
                f"{owner}: grouping key domain contains NA values under na_key_policy='error'."
            )


def normalize_group_label_scalar(value: object) -> object:
    """Return a hashable scalar without erasing NumPy temporal semantics."""
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return value
    return value.item() if isinstance(value, np.generic) else value


def _row_label(values: tuple[np.ndarray, ...], row: int) -> object:
    if len(values) == 1:
        return normalize_group_label_scalar(values[0][row])
    return tuple(normalize_group_label_scalar(value[row]) for value in values)


def require_hashable_group_label(label: object, *, owner: str, what: str) -> None:
    try:
        hash(label)
    except TypeError as exc:
        raise ValueError(
            f"{owner}: {what} produced unhashable group label {label!r}; labels must be hashable."
        ) from exc


def _label_keys_equal(left: object, right: object) -> bool:
    try:
        equal = canonical_group_label_key(left) == right
    except (TypeError, ValueError):
        return False
    return isinstance(equal, (bool, np.bool_)) and bool(equal)


def _require_no_na_label_collision(
    values: np.ndarray,
    *,
    active_mask: np.ndarray,
    na_mask: np.ndarray,
    label: object,
    owner: str,
    what: str,
) -> None:
    label_key = canonical_group_label_key(label)
    for value in values[active_mask & ~na_mask].flat:
        if _label_keys_equal(value, label_key):
            raise ValueError(
                f"{owner}: na_group_label={label!r} collides with non-NA "
                f"values in {what}."
            )


def replace_na_group_values(
    values: np.ndarray,
    *,
    active_mask: np.ndarray,
    label: object,
    owner: str,
    what: str,
) -> np.ndarray:
    """Replace active NA values while preserving every scalar label object."""
    require_hashable_group_label(label, owner=owner, what="na_group_label")
    normalized = object_label_array(values)
    na_mask = np.asarray(pd.isna(values), dtype=bool)
    active = np.asarray(active_mask, dtype=bool)
    _require_no_na_label_collision(
        normalized,
        active_mask=active,
        na_mask=na_mask,
        label=label,
        owner=owner,
        what=what,
    )
    for index in np.flatnonzero((active & na_mask).reshape(-1)):
        normalized.flat[index] = label
    return normalized


def _validate_domain_order(
    domain: tuple[object, ...],
    *,
    owner: str,
) -> None:
    positions: dict[object, list[int]] = {}
    labels: dict[object, object] = {}
    for index, label in enumerate(domain):
        require_hashable_group_label(
            label,
            owner=owner,
            what=f"bin grouping domain_order[{index}]",
        )
        key = canonical_group_label_key(label)
        positions.setdefault(key, []).append(index)
        labels.setdefault(key, label)
    for key, hits in positions.items():
        if len(hits) > 1:
            raise ValueError(
                f"{owner}: bin grouping domain_order contains duplicate label "
                f"{labels[key]!r} at positions {tuple(hits)!r}."
            )


def bin_domain_labels(keys: tuple[ResolvedGroupingKey, ...]) -> tuple[object, ...] | None:
    if len(keys) != 1 or keys[0].kind != "bin":
        return None
    return keys[0].domain_order


def order_group_labels(
    labels: tuple[object, ...],
    *,
    keys: tuple[ResolvedGroupingKey, ...],
    owner: str,
) -> tuple[object, ...]:
    """Order observed scalar labels without rebuilding their row partitions."""
    domain = bin_domain_labels(keys)
    if domain is None:
        return labels
    _validate_domain_order(domain, owner=owner)
    present = {canonical_group_label_key(label) for label in labels}
    ordered = tuple(label for label in domain if canonical_group_label_key(label) in present)
    ordered_keys = {canonical_group_label_key(label) for label in ordered}
    return ordered + tuple(
        label for label in labels if canonical_group_label_key(label) not in ordered_keys
    )


def resolve_global_group_rows(
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
        label = _row_label(values, row)
        require_hashable_group_label(label, owner=owner, what=f"grouping row[{row}]")
        key = canonical_group_label_key(label)
        group_index = index.get(key)
        if group_index is None:
            group_index = len(labels)
            index[key] = group_index
            labels.append(label)
            groups.append([])
        groups[group_index].append(row)
    frozen = tuple(labels)
    ordered = order_group_labels(frozen, keys=keys, owner=owner)
    row_lookup = {
        canonical_group_label_key(label): tuple(groups[position])
        for position, label in enumerate(frozen)
    }
    return ordered, tuple(row_lookup[canonical_group_label_key(label)] for label in ordered)


def resolve_output_label_plan(
    labels: tuple[object, ...],
    *,
    domain: tuple[object, ...] | None,
    include_empty: bool,
) -> GroupOutputLabelPlan:
    """Resolve declared and observed labels through canonical scalar identity."""
    observed = {
        canonical_group_label_key(label): position
        for position, label in enumerate(labels)
    }
    if not include_empty or domain is None:
        return GroupOutputLabelPlan(labels, tuple(range(len(labels))))
    domain_keys = {canonical_group_label_key(label) for label in domain}
    output_labels = domain + tuple(
        label for label in labels if canonical_group_label_key(label) not in domain_keys
    )
    positions = tuple(
        observed.get(canonical_group_label_key(label))
        for label in output_labels
    )
    return GroupOutputLabelPlan(output_labels, positions)


__all__ = [
    "GroupOutputLabelPlan",
    "bin_domain_labels",
    "enforce_na_error_policy",
    "normalize_group_label_scalar",
    "object_label_array",
    "object_label_vector",
    "order_group_labels",
    "realize_grouping_values",
    "replace_na_group_values",
    "require_hashable_group_label",
    "resolve_global_group_rows",
    "resolve_output_label_plan",
]
