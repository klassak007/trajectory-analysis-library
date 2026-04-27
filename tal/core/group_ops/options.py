from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .label_keys import canonical_group_label_key
from .types import GroupingBinSpec, GroupingFoundationOptions

_NA_POLICIES = frozenset({"error", "drop", "group"})


def _validate_na_policy(opts: GroupingFoundationOptions, *, owner: str) -> None:
    if opts.na_key_policy not in _NA_POLICIES:
        raise ValueError(
            f"{owner}: opts.na_key_policy must be one of {tuple(sorted(_NA_POLICIES))!r}."
        )
    if opts.na_group_label is None or opts.na_key_policy == "group":
        return
    raise ValueError(
        f"{owner}: opts.na_group_label is only valid when opts.na_key_policy='group'."
    )


def _coerce_numeric_bins(
    value: Sequence[float] | np.ndarray | xr.DataArray,
    *,
    owner: str,
) -> np.ndarray:
    if isinstance(value, xr.DataArray):
        if value.ndim != 1:
            raise ValueError(f"{owner}: bin edges must be 1-D.")
        raw = np.asarray(value.values)
    else:
        raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{owner}: bin edges must be 1-D.")
    try:
        edges = np.asarray(raw, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: bin edges must be numeric.") from exc
    if edges.size < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError(f"{owner}: bin edges must be finite, strictly increasing, and length >= 2.")
    return edges


def _first_duplicate_label(
    labels: tuple[object, ...],
) -> tuple[object, tuple[int, ...]] | None:
    positions: dict[object, list[int]] = {}
    repr_labels: dict[object, object] = {}
    for index, label in enumerate(labels):
        key = canonical_group_label_key(label)
        positions.setdefault(key, []).append(index)
        repr_labels.setdefault(key, label)
    for key, hits in positions.items():
        if len(hits) > 1:
            return repr_labels[key], tuple(hits)
    return None


def _first_unhashable_label(
    labels: tuple[object, ...],
) -> tuple[object, int] | None:
    for index, label in enumerate(labels):
        try:
            hash(label)
        except TypeError:
            return label, index
    return None


def validate_grouping_bin_spec(spec: GroupingBinSpec, *, owner: str) -> np.ndarray:
    if not isinstance(spec.source, (str, xr.DataArray)):
        raise TypeError(f"{owner}: bin spec source must be a string key or xr.DataArray.")
    edges = _coerce_numeric_bins(spec.bins, owner=owner)
    if not isinstance(spec.right, bool) or not isinstance(spec.include_lowest, bool):
        raise ValueError(f"{owner}: bin spec right/include_lowest must be bool.")
    if spec.labels is None:
        return edges
    labels = tuple(spec.labels)
    if len(labels) != int(edges.size - 1):
        raise ValueError(f"{owner}: bin spec labels length must equal len(bins)-1.")
    unhashable = _first_unhashable_label(labels)
    if unhashable is not None:
        label, index = unhashable
        raise ValueError(
            f"{owner}: bin spec labels must be hashable; "
            f"label {label!r} at position {index} is unhashable."
        )
    duplicate = _first_duplicate_label(labels)
    if duplicate is not None:
        label, positions = duplicate
        raise ValueError(
            f"{owner}: bin spec labels must be unique; duplicate label {label!r} at positions {positions!r}."
        )
    return edges


def coerce_grouping_foundation_options(
    opts: object | None,
    *,
    owner: str,
) -> GroupingFoundationOptions:
    if opts is None:
        out = GroupingFoundationOptions()
    elif isinstance(opts, GroupingFoundationOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be GroupingFoundationOptions or None.")
    _validate_na_policy(out, owner=owner)
    return out


__all__ = [
    "coerce_grouping_foundation_options",
    "validate_grouping_bin_spec",
]
