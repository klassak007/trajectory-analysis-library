from __future__ import annotations

from typing import Any

import xarray as xr

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
    return finalize_loaded_dataset(cls, ds, validate=validate, owner=owner)


__all__ = ["read_analysis_object_zarr", "write_analysis_object_zarr"]
