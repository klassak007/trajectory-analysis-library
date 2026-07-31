from __future__ import annotations

import numpy as np
import xarray as xr

from .. import validity_values


def _sequence_index(ds: xr.Dataset, *, sequence_dim: str) -> xr.DataArray:
    size = int(ds.sizes[sequence_dim])
    if sequence_dim in ds.coords and tuple(ds.coords[sequence_dim].dims) == (sequence_dim,):
        coord = ds.coords[sequence_dim]
    else:
        coord = xr.DataArray(np.arange(size, dtype=np.int64), dims=(sequence_dim,))
    values = np.arange(size, dtype=np.int64)
    return xr.DataArray(values, dims=(sequence_dim,), coords={sequence_dim: coord})


def _sequence_size_array(
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    sequence_size_coord: str | None,
) -> xr.DataArray | None:
    if sequence_dim is None or sequence_size_coord is None:
        return None
    if sequence_dim not in ds.dims:
        return None
    if sequence_size_coord not in ds.coords:
        return None
    size = ds.coords[sequence_size_coord]
    if sequence_dim in size.dims:
        return None
    return size


def _structural_valid_mask_base(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    size: xr.DataArray,
) -> xr.DataArray:
    idx = _sequence_index(ds, sequence_dim=sequence_dim)
    return idx < size


def resolve_structural_valid_mask_base(
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    sequence_size_coord: str | None,
    owner: str = "resolve_structural_valid_mask_base",
) -> xr.DataArray | None:
    size = _sequence_size_array(
        ds,
        sequence_dim=sequence_dim,
        sequence_size_coord=sequence_size_coord,
    )
    if size is None or sequence_dim is None or sequence_size_coord is None:
        return None
    normalized = validity_values.require_valid_sequence_size_values(
        size,
        sequence_size_coord=sequence_size_coord,
        sequence_len=int(ds.sizes[sequence_dim]),
        owner=owner,
    )
    return _structural_valid_mask_base(ds, sequence_dim=sequence_dim, size=normalized)


def _resolve_structural_valid_mask_base_validated(
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    sequence_size_coord: str | None,
) -> xr.DataArray | None:
    """Build a base mask after the dataset schema has been validated."""
    size = _sequence_size_array(
        ds,
        sequence_dim=sequence_dim,
        sequence_size_coord=sequence_size_coord,
    )
    if size is None or sequence_dim is None:
        return None
    return _structural_valid_mask_base(ds, sequence_dim=sequence_dim, size=size)


def resolve_structural_valid_mask(
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    sequence_size_coord: str | None,
    var: xr.DataArray,
    owner: str = "resolve_structural_valid_mask",
) -> xr.DataArray | None:
    if sequence_dim is None or sequence_dim not in var.dims:
        return None
    mask = resolve_structural_valid_mask_base(
        ds,
        sequence_dim=sequence_dim,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    if mask is None:
        return None
    return mask.broadcast_like(var)


def apply_structural_mask(var: xr.DataArray, *, mask: xr.DataArray | None) -> xr.DataArray:
    if mask is None:
        return var
    return var.where(mask)


def reduce_missing_on_valid_prefix(
    var: xr.DataArray,
    *,
    reduce_dims: tuple[str, ...],
    mask: xr.DataArray | None,
) -> xr.DataArray:
    missing = var.isnull()
    if mask is not None:
        missing = missing & mask
    if not reduce_dims:
        return missing
    return missing.any(dim=reduce_dims)


__all__ = [
    "apply_structural_mask",
    "reduce_missing_on_valid_prefix",
    "resolve_structural_valid_mask",
    "resolve_structural_valid_mask_base",
]
