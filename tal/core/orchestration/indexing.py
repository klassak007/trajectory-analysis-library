from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr

XarrayObject = xr.Dataset | xr.DataArray


@dataclass(frozen=True)
class LaneIndexGroup:
    """One public xarray index group varying only over a semantic lane."""

    coordinate_names: tuple[Hashable, ...]
    index: xr.Index


def _varies_only_over_lane(
    coordinates: Mapping[Hashable, xr.Variable],
    *,
    lane_dim: str,
) -> bool:
    return bool(coordinates) and all(
        tuple(variable.dims) == (lane_dim,)
        for variable in coordinates.values()
    )


def lane_index_groups(
    value: XarrayObject,
    *,
    lane_dim: str,
) -> tuple[LaneIndexGroup, ...]:
    """Describe every public xarray index attached only to ``lane_dim``."""
    return tuple(
        LaneIndexGroup(tuple(coordinates), index)
        for index, coordinates in value.xindexes.group_by_index()
        if _varies_only_over_lane(coordinates, lane_dim=lane_dim)
    )


def _groups_by_coordinate_names(
    value: XarrayObject,
    *,
    lane_dim: str,
) -> dict[tuple[Hashable, ...], xr.Index]:
    return {
        group.coordinate_names: group.index
        for group in lane_index_groups(value, lane_dim=lane_dim)
    }


def require_exact_lane_indexes(
    source: XarrayObject,
    target: XarrayObject,
    *,
    lane_dim: str,
    owner: str,
    what: str,
) -> bool:
    """Require equal public xarray index topology for one semantic lane."""
    source_groups = _groups_by_coordinate_names(source, lane_dim=lane_dim)
    target_groups = _groups_by_coordinate_names(target, lane_dim=lane_dim)
    if not source_groups and not target_groups:
        return False
    if bool(source_groups) != bool(target_groups):
        raise ValueError(
            f"{owner}: {what} cannot mix indexed and unindexed {lane_dim!r} dimensions."
        )
    if source_groups.keys() != target_groups.keys():
        raise ValueError(
            f"{owner}: {what} index must exactly match dimension {lane_dim!r}; "
            "relevant coordinate topology differs."
        )
    if any(
        not index.equals(target_groups[names])
        for names, index in source_groups.items()
    ):
        raise ValueError(f"{owner}: {what} index must exactly match dimension {lane_dim!r}.")
    return True


def require_unique_lane_indexes(
    value: XarrayObject,
    *,
    lane_dim: str,
    owner: str,
) -> None:
    """Require uniqueness for every publicly projectable lane index."""
    for group in lane_index_groups(value, lane_dim=lane_dim):
        try:
            pandas_index = group.index.to_pandas_index()
        except TypeError:
            continue
        if bool(getattr(pandas_index, "is_unique", True)):
            continue
        raise ValueError(
            f"{owner}: labels along {lane_dim!r} must be unique. "
            "Provide unique coordinate labels on query-aligned axes."
        )


def _unselectable_index_coordinates(
    value: XarrayObject,
    *,
    dim: str,
    rows: np.ndarray,
) -> tuple[object, ...]:
    names: list[object] = []
    for index, coordinates in value.xindexes.group_by_index():
        lane_only = coordinates and all(
            tuple(variable.dims) == (dim,)
            for variable in coordinates.values()
        )
        if lane_only and index.isel({dim: rows}) is None:
            names.extend(coordinates)
    return tuple(names)


def isel_rows(value: XarrayObject, *, dim: str, rows: np.ndarray) -> XarrayObject:
    """Select rows, retaining the lane when unselectable coordinates are omitted.

    The only consumers are reducer partitions of payloads and weights. If
    omitted coordinates were the sole carrier of a Dataset's row dimension,
    unindexed positions retain that internal lane until reduction removes it.
    No user index or transform values are manufactured or evaluated.
    """
    names = _unselectable_index_coordinates(value, dim=dim, rows=rows)
    selectable = value.drop_vars(names) if names else value
    if names and dim not in selectable.dims:
        return selectable.assign_coords(
            xr.Coordinates({dim: xr.Variable((dim,), rows)}, indexes={})
        )
    return selectable.isel({dim: rows})


__all__ = [
    "LaneIndexGroup",
    "isel_rows",
    "lane_index_groups",
    "require_exact_lane_indexes",
    "require_unique_lane_indexes",
]
