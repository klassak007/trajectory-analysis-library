from __future__ import annotations

import xarray as xr

from tal.core import AnalysisObject, SchemaError
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema


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


def _adapter_finalize_spec(
    *,
    batch_dim: str,
    sequence_dim: str,
    size_name: str,
    param_name: str,
) -> CoreSchemaFinalizeSpec:
    return CoreSchemaFinalizeSpec(
        sequence_dim=sequence_dim,
        batch_dims=(batch_dim,),
        core_dims=(),
        param_name=param_name,
        size_name=size_name,
    )


def _owned_adapter_source(ds: xr.Dataset) -> AnalysisObject:
    # CSV/ROS kernels create this eager dataset from fresh arrays and retain no
    # caller-visible reference. External datasets must still use normal AO ingress.
    return AnalysisObject._from_unvalidated(ds)


def _finalize_owned_adapter_dataset(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    sequence_dim: str,
    size_name: str,
    param_name: str,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    _require_adapter_schema_dims(
        ds,
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        size_name=size_name,
        param_name=param_name,
        owner=owner,
    )
    spec = _adapter_finalize_spec(
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        size_name=size_name,
        param_name=param_name,
    )
    source = _owned_adapter_source(ds)
    try:
        return finalize_with_schema(
            source,
            ds,
            spec=spec,
            validate=validate,
            owner=owner,
        )
    except SchemaError as exc:
        raise ValueError(f"{owner}: failed finalizing adapter dataset schema.") from exc


__all__: list[str] = []
