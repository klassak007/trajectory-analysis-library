from __future__ import annotations

"""Namespace-safe xarray rename utilities.

This module owns temp-name allocation and collision-safe two-step dim renames
for DataArray/Dataset namespaces.
"""

from collections.abc import Mapping, Sequence

import xarray as xr


def dataarray_namespace_names(da: xr.DataArray) -> tuple[str, ...]:
    """Return all occupied names in a DataArray namespace.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray value processed by this operation.

    Returns
    -------
    tuple[str, ...]
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    names = {str(dim) for dim in da.dims}
    names.update(str(name) for name in da.coords)
    if da.name is not None:
        names.add(str(da.name))
    return tuple(sorted(names))


def dataset_namespace_names(ds: xr.Dataset) -> tuple[str, ...]:
    """Return all occupied names in a Dataset namespace.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.

    Returns
    -------
    tuple[str, ...]
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    names = {str(name) for name in ds.dims}
    names.update(str(name) for name in ds.coords)
    names.update(str(name) for name in ds.data_vars)
    return tuple(sorted(names))


def unique_temp_dim(base: str, *, taken_dims: Sequence[str]) -> str:
    """Allocate a temporary dimension name that does not collide with ``taken_dims``.

    Parameters
    ----------
    base : str
        Input dataset/source value processed by this operation.
    taken_dims : Sequence[str], optional
        Core-dimension labels/shape metadata used for structural operations.

    Returns
    -------
    str
        String result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    taken = {str(dim) for dim in taken_dims}
    name = str(base)
    while name in taken:
        name = f"{name}_"
    return name


def rename_dims_collision_safe(
    da: xr.DataArray,
    *,
    mapping: Mapping[str, str],
    temp_prefix: str = "__tal_dim_tmp__",
) -> xr.DataArray:
    """Rename dimensions with a collision-safe two-stage strategy.

    Parameters
    ----------
    da : xarray.DataArray
        Input array whose dimensions will be renamed.
    mapping : Mapping[str, str]
        Source-to-destination dimension mapping. Identity entries are ignored.
    temp_prefix : str, default="__tal_dim_tmp__"
        Prefix used to allocate temporary dimension names during staged renaming.

    Returns
    -------
    xarray.DataArray
        DataArray with dimensions renamed according to ``mapping``.

    Notes
    -----
    xarray requires dimension renames to be injective. For swaps like
    ``{"x": "y", "y": "x"}``, a direct rename collides. This helper first renames
    each source dim to a unique temporary dim, then applies final names.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.utils.xarray_namespace import rename_dims_collision_safe
    >>> da = xr.DataArray([[1.0]], dims=("x", "y"), coords={"x": [0], "y": [1]})
    >>> out = rename_dims_collision_safe(da, mapping={"x": "y", "y": "x"})
    >>> out.dims
    ('y', 'x')
    """
    rename_map = {src: dst for src, dst in mapping.items() if src in da.dims and src != dst}
    if not rename_map:
        return da
    taken = set(dataarray_namespace_names(da))
    temp_map: dict[str, str] = {}
    for src in rename_map:
        temp = unique_temp_dim(temp_prefix, taken_dims=tuple(taken))
        taken.add(temp)
        temp_map[src] = temp
    staged = da.rename(temp_map)
    final_map = {temp_map[src]: dst for src, dst in rename_map.items()}
    return staged.rename(final_map)


__all__ = [
    "dataarray_namespace_names",
    "dataset_namespace_names",
    "rename_dims_collision_safe",
    "unique_temp_dim",
]
