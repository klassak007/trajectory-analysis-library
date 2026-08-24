from __future__ import annotations

from typing import Any

import xarray as xr

from tal.core.schema_validate import validate_schema_structure

from .finalize import finalize_loaded_dataset
from .options import AOZarrReadOptions, AOZarrWriteOptions, coerce_zarr_read_options, coerce_zarr_write_options


def _zarr_write_kwargs(opts: AOZarrWriteOptions) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if opts.mode is not None:
        out["mode"] = opts.mode
    if opts.consolidated is not None:
        out["consolidated"] = opts.consolidated
    return out


def _zarr_read_kwargs(opts: AOZarrReadOptions) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if opts.consolidated is not None:
        out["consolidated"] = opts.consolidated
    if opts.chunks is not None:
        out["chunks"] = opts.chunks
    return out


def _materialize_zarr_validity_coord(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    try:
        size_name = validate_schema_structure(ds)
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid persisted schema payload.") from exc
    if size_name is None or size_name not in ds.coords:
        return ds
    coord = ds.coords[size_name]
    if getattr(coord.data, "chunks", None) is None:
        return ds
    try:
        loaded = coord.variable.compute()
    except Exception as exc:  # pragma: no cover - backend read failure envelope.
        raise ValueError(
            f"{owner}: failed reading persisted sequence_size_coord {size_name!r}."
        ) from exc
    return ds.assign_coords({size_name: loaded})


def write_analysis_object_zarr(
    ao: "AnalysisObject",
    store: str,
    *,
    opts: AOZarrWriteOptions | None,
    owner: str,
) -> Any:
    options = coerce_zarr_write_options(opts, owner=owner)
    if not isinstance(store, str) or not store:
        raise TypeError(f"{owner}: store must be a non-empty string path.")
    return ao.unsafe_data.to_zarr(store, **_zarr_write_kwargs(options))


def read_analysis_object_zarr(
    cls: type["AnalysisObject"],
    store: str,
    *,
    opts: AOZarrReadOptions | None,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    options = coerce_zarr_read_options(opts, owner=owner)
    if not isinstance(store, str) or not store:
        raise TypeError(f"{owner}: store must be a non-empty string path.")
    try:
        ds = xr.open_zarr(store, **_zarr_read_kwargs(options))
    except Exception as exc:  # pragma: no cover - backend-specific failure envelope.
        raise ValueError(f"{owner}: failed reading zarr store {store!r}.") from exc
    ds = _materialize_zarr_validity_coord(ds, owner=owner)
    return finalize_loaded_dataset(cls, ds, validate=validate, owner=owner)


__all__ = ["read_analysis_object_zarr", "write_analysis_object_zarr"]
