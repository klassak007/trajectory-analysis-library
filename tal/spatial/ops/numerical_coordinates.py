"""Keep duplicate lazy auxiliary-coordinate checks out of spatial graph building."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core.param_ops.guards import reserved_coord_is_owned
from tal.core.schema_validate.finalize import transfer_dataarray_metadata


def _require_coordinate_agreement(values: np.ndarray, agrees: np.ndarray, *, owner: str, name: str) -> np.ndarray:
    if not bool(agrees):
        raise ValueError(f"{owner}: shared coordinate {name!r} must agree exactly.")
    return values


def _guard_values(values: xr.DataArray, agrees: xr.DataArray, *, owner: str, name: str) -> xr.DataArray:
    guarded = xr.apply_ufunc(
        _require_coordinate_agreement, values.variable, agrees.variable,
        kwargs={"owner": owner, "name": name}, dask="parallelized", output_dtypes=[values.dtype],
    )
    return transfer_dataarray_metadata(values, values.copy(data=guarded.data))


def _coordinate_agreement(coords: tuple[xr.DataArray, ...]) -> xr.DataArray:
    first = coords[0].reset_coords(drop=True)
    agrees = xr.DataArray(True)
    for coord in coords[1:]:
        other = coord.reset_coords(drop=True)
        equal = (first == other) | (first.isnull() & other.isnull())
        agrees = agrees & equal.all()
    return agrees


def _share_one_coordinate(
    arrays: tuple[xr.DataArray, ...], *, name: str, owner: str, component_agreement: bool,
) -> tuple[xr.DataArray, ...]:
    sources = tuple(array for array in arrays if name in array.coords)
    owned = tuple(array for array in sources if name == "valid" and reserved_coord_is_owned(array, name=name))
    # Generated validity is reconciled by the numerical finalizer, not compared
    # as an ordinary shared coordinate between different numerical domains.
    canonical = (owned or sources)[0].coords[name].reset_coords(drop=True)
    if component_agreement and any(array.coords[name].dims != canonical.dims for array in sources):
        raise ValueError(f"{owner}: shared coordinate {name!r} must have matching dimensions.")
    agrees = None if owned and not component_agreement else _coordinate_agreement(tuple(array.coords[name] for array in sources))
    if agrees is not None:
        canonical = _guard_values(canonical, agrees, owner=owner, name=name)
    outputs = []
    for array in arrays:
        value = array.drop_vars(name, errors="ignore")
        if set(canonical.dims) <= set(value.dims):
            value = value.assign_coords({name: canonical})
        if agrees is not None:
            value = _guard_values(value, agrees, owner=owner, name=name)
        outputs.append(value)
    return tuple(outputs)


def share_lazy_numerical_coordinates(
    *arrays: xr.DataArray, owner: str, component_agreement: bool = False,
) -> tuple[xr.DataArray, ...]:
    """Share one checked carrier for duplicate lazy auxiliaries on aligned inputs.

    Native indexes stay with the alignment owner. Ordinary eager coordinates
    retain xarray's existing merge behavior. Unknown auxiliary equality is checked
    on materialization, through both the coordinate and numerical payload.
    """
    indexed = {name for array in arrays for name in array.xindexes}
    names = dict.fromkeys(name for array in arrays for name in array.coords if name not in indexed)
    out = arrays
    for name in names:
        coords = tuple(array.coords[name] for array in arrays if name in array.coords)
        if len(coords) > 1 and any(coord.chunks is not None for coord in coords):
            out = _share_one_coordinate(out, name=name, owner=owner, component_agreement=component_agreement)
    return out
