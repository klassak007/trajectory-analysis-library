from __future__ import annotations

import xarray as xr

from ..validity_mask import resolve_structural_valid_mask_base


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
