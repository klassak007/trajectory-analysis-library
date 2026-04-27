from __future__ import annotations

import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema import set_param_coord, set_roles, set_validity

from .finalize import finalize_loaded_dataset


def _require_adapter_schema_dims(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    sequence_dim: str,
    size_name: str,
    param_name: str,
    owner: str,
) -> None:
    if batch_dim == sequence_dim:
        raise ValueError(f"{owner}: batch_dim and sequence_dim must be distinct.")
    if sequence_dim not in ds.dims:
        raise ValueError(f"{owner}: sequence_dim {sequence_dim!r} is not present in dataset dims.")
    if batch_dim not in ds.dims:
        raise ValueError(f"{owner}: batch_dim {batch_dim!r} is not present in dataset dims.")
    if size_name not in ds.coords:
        raise ValueError(f"{owner}: sequence_size_coord {size_name!r} is missing from coordinates.")
    if param_name not in ds.coords:
        raise ValueError(f"{owner}: param_coord {param_name!r} is missing from coordinates.")
    if tuple(ds.coords[size_name].dims) != (batch_dim,):
        raise ValueError(
            f"{owner}: sequence_size_coord {size_name!r} must have dims ({batch_dim!r},); "
            f"got {tuple(ds.coords[size_name].dims)!r}."
        )
    if tuple(ds.coords[param_name].dims) != (batch_dim, sequence_dim):
        raise ValueError(
            f"{owner}: param_coord {param_name!r} must have dims ({batch_dim!r}, {sequence_dim!r}); "
            f"got {tuple(ds.coords[param_name].dims)!r}."
        )


def _stamp_adapter_schema(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    sequence_dim: str,
    size_name: str,
    param_name: str,
    owner: str,
) -> xr.Dataset:
    _require_adapter_schema_dims(
        ds,
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        size_name=size_name,
        param_name=param_name,
        owner=owner,
    )
    out = set_roles(
        ds,
        sequence_dim=sequence_dim,
        batch_dims=[batch_dim],
        core_dims=[],
        validate=False,
    )
    out = set_param_coord(out, name=param_name, validate=False)
    return set_validity(out, sequence_size_coord=size_name, layout="left_packed", validate=False)


def finalize_adapter_dataset(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    sequence_dim: str,
    size_name: str,
    param_name: str,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    stamped = _stamp_adapter_schema(
        ds,
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        size_name=size_name,
        param_name=param_name,
        owner=owner,
    )
    return finalize_loaded_dataset(AnalysisObject, stamped, validate=validate, owner=owner)


__all__ = ["finalize_adapter_dataset"]
