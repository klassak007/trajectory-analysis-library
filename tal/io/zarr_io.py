from __future__ import annotations

from collections.abc import Callable
from typing import Any

import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.schema_validate import validate_schema, validate_schema_structure

from .adapter_cleanup import suppress_cleanup_during_active_error
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


def _materialize_zarr_validity_coord(
    ds: xr.Dataset,
    *,
    owner: str,
    payload_label: str,
) -> xr.Dataset:
    try:
        size_name = validate_schema_structure(ds)
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid {payload_label} schema payload.") from exc
    if size_name is None or size_name not in ds.coords:
        return ds
    variable = ds.coords[size_name].variable
    if variable._in_memory:
        return ds
    try:
        loaded = variable.compute()
    except Exception as exc:  # pragma: no cover - backend read failure envelope.
        raise ValueError(
            f"{owner}: failed reading {payload_label} sequence_size_coord {size_name!r}."
        ) from exc
    return ds.assign_coords({size_name: loaded})


def _validate_zarr_write_dataset(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    prepared = _materialize_zarr_validity_coord(
        ds,
        owner=owner,
        payload_label="AnalysisObject",
    )
    try:
        return validate_schema(prepared)
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid AnalysisObject schema payload.") from exc


def _transfer_zarr_close_ownership(
    ao: "AnalysisObject",
    *,
    source: xr.Dataset,
) -> "AnalysisObject":
    target = analysis_object_dataset(ao)
    if target is source:
        return ao
    target_close = getattr(target, "_close", None)
    if target_close is None:
        target.set_close(source.close)
        return ao
    target.set_close(_compose_zarr_close(target_close, source.close))
    return ao


def _compose_zarr_close(
    primary: Callable[[], None],
    cleanup: Callable[[], None],
) -> Callable[[], None]:
    """Compose target and backend ownership with idempotent cleanup precedence."""
    closed = False

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            primary()
        except BaseException:
            suppress_cleanup_during_active_error(cleanup)
            raise
        cleanup()

    return close


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
    ds = _validate_zarr_write_dataset(analysis_object_dataset(ao), owner=owner)
    try:
        return ds.to_zarr(store, **_zarr_write_kwargs(options))
    except Exception as exc:  # pragma: no cover - backend-specific failure envelope.
        raise ValueError(f"{owner}: failed writing zarr store {store!r}.") from exc


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
    try:
        prepared = _materialize_zarr_validity_coord(
            ds,
            owner=owner,
            payload_label="persisted",
        )
        loaded = finalize_loaded_dataset(cls, prepared, validate=validate, owner=owner)
        return _transfer_zarr_close_ownership(loaded, source=ds)
    except BaseException:
        suppress_cleanup_during_active_error(ds.close)
        raise


__all__ = ["read_analysis_object_zarr", "write_analysis_object_zarr"]
