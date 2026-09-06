from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class _LaneIndexGroup:
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


def _lane_index_groups(
    value: xr.Dataset | xr.DataArray,
    *,
    lane_dim: str,
) -> tuple[_LaneIndexGroup, ...]:
    return tuple(
        _LaneIndexGroup(tuple(coordinates), index)
        for index, coordinates in value.xindexes.group_by_index()
        if _varies_only_over_lane(coordinates, lane_dim=lane_dim)
    )


def _groups_by_coordinate_names(
    value: xr.Dataset | xr.DataArray,
    *,
    lane_dim: str,
) -> dict[tuple[Hashable, ...], xr.Index]:
    return {
        group.coordinate_names: group.index
        for group in _lane_index_groups(value, lane_dim=lane_dim)
    }


def require_exact_lane_indexes(
    source: xr.Dataset,
    key: xr.DataArray,
    *,
    lane_dim: str,
    owner: str,
    what: str,
) -> bool:
    """Require equal public xarray index topology for one semantic lane."""
    source_groups = _groups_by_coordinate_names(source, lane_dim=lane_dim)
    key_groups = _groups_by_coordinate_names(key, lane_dim=lane_dim)
    if not source_groups and not key_groups:
        return False
    if bool(source_groups) != bool(key_groups):
        raise ValueError(
            f"{owner}: {what} cannot mix indexed and unindexed primary batch dimensions."
        )
    if source_groups.keys() != key_groups.keys():
        raise ValueError(
            f"{owner}: {what} index must exactly match primary batch dimension "
            f"{lane_dim!r}; relevant coordinate topology differs."
        )
    if any(
        not index.equals(key_groups[names])
        for names, index in source_groups.items()
    ):
        raise ValueError(
            f"{owner}: {what} index must exactly match primary batch dimension {lane_dim!r}."
        )
    return True


__all__ = ["require_exact_lane_indexes"]
