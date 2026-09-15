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


@dataclass(frozen=True)
class IndexTopologySnapshot:
    """Copied public xarray index coordinates for selected dimensions."""

    coordinates: xr.Coordinates


@dataclass(frozen=True)
class ResultCoordinateSnapshot:
    """Shallow output coordinates plus copied public xarray indexes."""

    coordinates: xr.Coordinates
    indexes: tuple[IndexTopologySnapshot, ...]


def _index_is_confined_to_dims(
    coordinates: Mapping[Hashable, xr.Variable],
    *,
    dims: set[str],
) -> bool:
    coord_dims = tuple(set(variable.dims) for variable in coordinates.values())
    return bool(coord_dims) and all(item.issubset(dims) for item in coord_dims) and any(
        item & dims for item in coord_dims
    )


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


def capture_index_topology(
    value: XarrayObject,
    *,
    dims: tuple[str, ...],
) -> IndexTopologySnapshot:
    """Capture public index groups varying only over ``dims`` without evaluation."""
    allowed = set(dims)
    variables: dict[Hashable, xr.Variable] = {}
    indexes: dict[Hashable, xr.Index] = {}
    for index, coordinates in value.xindexes.group_by_index():
        if not _index_is_confined_to_dims(coordinates, dims=allowed):
            continue
        copied = index.copy(deep=False)
        rebuilt = copied.create_variables(coordinates)
        variables.update(rebuilt)
        indexes.update({name: copied for name in rebuilt})
    return IndexTopologySnapshot(xr.Coordinates(variables, indexes=indexes))


def restore_index_topology(
    value: XarrayObject,
    snapshot: IndexTopologySnapshot,
) -> XarrayObject:
    """Restore one captured public index topology onto assembled data."""
    names = tuple(snapshot.coordinates)
    if not names:
        return value
    cleared = value.drop_vars(names, errors="ignore")
    return cleared.assign_coords(snapshot.coordinates)


def without_index_topology(
    value: XarrayObject,
    *,
    dims: tuple[str, ...],
) -> XarrayObject:
    """Drop index coordinate wrappers confined to selected dimensions."""
    allowed = set(dims)
    names: list[Hashable] = []
    for _, coordinates in value.xindexes.group_by_index():
        if _index_is_confined_to_dims(coordinates, dims=allowed):
            names.extend(coordinates)
    return value.drop_vars(names, errors="ignore") if names else value


def _result_coord_is_applicable(
    name: Hashable,
    variable: xr.Variable,
    *,
    output_dims: set[str],
) -> bool:
    dims = set(variable.dims)
    if not dims.issubset(output_dims):
        return False
    return not (name in output_dims and name not in dims)


def _merge_result_coord(
    variables: dict[Hashable, xr.Variable],
    name: Hashable,
    variable: xr.Variable,
    *,
    owner: str,
) -> None:
    current = variables.get(name)
    if current is None:
        variables[name] = variable.copy(deep=False)
        return
    try:
        identical = current.identical(variable)
    except Exception as exc:
        raise ValueError(f"{owner}: coordinate {name!r} cannot be compared safely.") from exc
    if not identical:
        raise ValueError(f"{owner}: coordinate {name!r} has conflicting output values.")


def _capture_nonindex_coordinates(
    value: XarrayObject,
    variables: dict[Hashable, xr.Variable],
    *,
    output_dims: set[str],
    owner: str,
) -> None:
    indexed = set(value.xindexes)
    for name, coord in value.coords.items():
        if name in indexed or not _result_coord_is_applicable(
            name,
            coord.variable,
            output_dims=output_dims,
        ):
            continue
        _merge_result_coord(variables, name, coord.variable, owner=owner)


def capture_result_coordinates(
    *values: XarrayObject,
    output_dims: tuple[str, ...],
    owner: str,
) -> ResultCoordinateSnapshot:
    """Capture output-safe coordinates without realizing coordinate payloads."""
    allowed = set(output_dims)
    variables: dict[Hashable, xr.Variable] = {}
    indexes: list[IndexTopologySnapshot] = []
    for value in values:
        indexes.append(capture_index_topology(value, dims=output_dims))
        _capture_nonindex_coordinates(
            value,
            variables,
            output_dims=allowed,
            owner=owner,
        )
    return ResultCoordinateSnapshot(
        xr.Coordinates(variables, indexes={}),
        tuple(indexes),
    )


def restore_result_coordinates(
    value: XarrayObject,
    snapshot: ResultCoordinateSnapshot,
) -> XarrayObject:
    """Restore non-index and native-index coordinates exactly once."""
    names = tuple(snapshot.coordinates)
    out = value.drop_vars(names, errors="ignore").assign_coords(snapshot.coordinates)
    for index_snapshot in snapshot.indexes:
        out = restore_index_topology(out, index_snapshot)
    return out


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
    "IndexTopologySnapshot",
    "LaneIndexGroup",
    "ResultCoordinateSnapshot",
    "capture_index_topology",
    "capture_result_coordinates",
    "isel_rows",
    "lane_index_groups",
    "require_exact_lane_indexes",
    "require_unique_lane_indexes",
    "restore_index_topology",
    "restore_result_coordinates",
    "without_index_topology",
]
