from __future__ import annotations

import numpy as np
import xarray as xr

XarrayObject = xr.Dataset | xr.DataArray


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


__all__ = ["isel_rows"]
