from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from tal.core.orchestration.context import DatasetContextOptions, resolve_dataset_context
from tal.core.orchestration.lazy import is_chunked_dataarray
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.param_ops.types import ParamRuntimeContext
from tal.geo import GeodeticPosition

from .options import AstroTimeOptions, coerce_time_options

_OBSERVER_OPTIONS = DatasetContextOptions(
    require_roles=True,
    select_numeric_var=True,
    require_single_numeric_var=True,
    allowed_core_arity=(1,),
    require_semantic_dims_in_var=True,
)


@dataclass(frozen=True)
class AstroObserverContext:
    """Resolved observer location context for astro operations."""

    location: GeodeticPosition
    ds: xr.Dataset
    data: xr.DataArray | None
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    param_coord: str | None
    sequence_size_coord: str | None
    valid_mask: xr.DataArray | None


@dataclass(frozen=True)
class AstroTimeContext:
    """Resolved observation time context for astro operations."""

    coord: xr.DataArray
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    scale: str


@dataclass(frozen=True)
class AstroDirectionRuntimeContext:
    """Resolved observer/time topology for topocentric direction operations."""

    observer: AstroObserverContext
    time: AstroTimeContext
    output_sequence_dim: str | None
    output_batch_dims: tuple[str, ...]


def coerce_observer_location(value: object, *, owner: str) -> GeodeticPosition:
    """Coerce an observer location through the geo typed owner."""
    if isinstance(value, GeodeticPosition):
        return value
    return GeodeticPosition.from_lla(value)


def _resolve_observer_param_runtime(location: GeodeticPosition, param_coord: str | None) -> ParamRuntimeContext | None:
    if param_coord is None:
        return None
    return resolve_param_runtime_context(location, on=param_coord)


def resolve_observer_context(value: object, *, owner: str) -> AstroObserverContext:
    """Resolve a geodetic observer location without duplicating core owners."""
    location = coerce_observer_location(value, owner=owner)
    ctx = resolve_dataset_context(location, owner=owner, options=_OBSERVER_OPTIONS)
    runtime = _resolve_observer_param_runtime(location, ctx.param_coord)
    return AstroObserverContext(
        location=location,
        ds=runtime.ds if runtime is not None else ctx.ds,
        data=ctx.data,
        sequence_dim=runtime.sequence_dim if runtime is not None else ctx.sequence_dim,
        batch_dims=runtime.batch_dims if runtime is not None else ctx.batch_dims,
        param_coord=runtime.spec.name if runtime is not None else ctx.param_coord,
        sequence_size_coord=runtime.sequence_size_coord if runtime is not None else ctx.sequence_size_coord,
        valid_mask=runtime.valid_mask if runtime is not None else None,
    )


def _fail_if_raw_lazy_time(value: object, *, owner: str) -> None:
    has_chunks = getattr(value, "chunks", None) is not None
    graph = getattr(value, "__dask_graph__", None)
    if not has_chunks and graph is None:
        return
    raise ValueError(
        f"{owner}: raw lazy time arrays are not supported in astro A1; "
        "wrap eager datetime data in xr.DataArray or materialize explicitly."
    )


def _datetime64_array(value: object, *, owner: str) -> np.ndarray:
    _fail_if_raw_lazy_time(value, owner=owner)
    probe = np.asarray(value)
    if np.issubdtype(probe.dtype, np.number) or np.issubdtype(probe.dtype, np.bool_):
        raise ValueError(f"{owner}: time values must be datetime-like, not numeric.")
    try:
        return np.asarray(pd.to_datetime(value), dtype="datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: time values must be datetime-like.") from exc


def _time_from_array(value: object, *, owner: str) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        return value
    arr = _datetime64_array(value, owner=owner)
    if arr.ndim == 0:
        return xr.DataArray(arr)
    if arr.ndim == 1:
        return xr.DataArray(arr, dims=("astro_time",))
    raise ValueError(f"{owner}: time values must be scalar or one-dimensional.")


def _require_time_array(coord: xr.DataArray, *, owner: str, allow_numeric: bool = False) -> xr.DataArray:
    if is_chunked_dataarray(coord) and not np.issubdtype(np.dtype(coord.dtype), np.datetime64):
        raise ValueError(f"{owner}: chunked non-datetime time inputs are not supported in astro A1.")
    if np.issubdtype(np.dtype(coord.dtype), np.datetime64):
        return coord
    if allow_numeric and np.issubdtype(np.dtype(coord.dtype), np.number):
        return coord
    converted = _datetime64_array(coord.to_numpy(), owner=owner)
    return xr.DataArray(converted, dims=coord.dims, coords=coord.coords, name=coord.name)


def source_time_coord(observer: AstroObserverContext, source: str, *, owner: str) -> xr.DataArray:
    """Resolve a named time coordinate or data variable from an observer dataset."""
    if source in observer.ds.coords:
        return observer.ds.coords[source]
    if source in observer.ds.data_vars:
        return observer.ds[source]
    raise ValueError(f"{owner}: time.source {source!r} was not found in observer dataset.")


def _new_time_sequence_dim(dims: Sequence[str], observer: AstroObserverContext, *, owner: str) -> str | None:
    observer_dims = set(observer.batch_dims)
    if observer.sequence_dim is not None:
        observer_dims.add(observer.sequence_dim)
    extra = tuple(dim for dim in dims if dim not in observer_dims)
    if not extra:
        return observer.sequence_dim
    if observer.sequence_dim is None and len(extra) == 1:
        return str(extra[0])
    raise ValueError(f"{owner}: time dims {tuple(dims)!r} are incompatible with observer topology.")


def _resolve_time_topology(coord: xr.DataArray, observer: AstroObserverContext, *, owner: str) -> tuple[str | None, tuple[str, ...]]:
    sequence_dim = _new_time_sequence_dim(coord.dims, observer, owner=owner)
    illegal_batch = tuple(dim for dim in coord.dims if dim not in (*observer.batch_dims, sequence_dim))
    if illegal_batch:
        raise ValueError(f"{owner}: time dims {tuple(coord.dims)!r} would create independent expansion.")
    return sequence_dim, observer.batch_dims


def resolve_time_context(
    observer: AstroObserverContext,
    *,
    time: object | None = None,
    opts: AstroTimeOptions | None = None,
    owner: str,
) -> AstroTimeContext:
    """Resolve observation time from an explicit value or observer source."""
    time_opts = coerce_time_options(opts, owner=owner)
    if time is not None and time_opts.source is not None:
        raise ValueError(f"{owner}: pass either explicit time or time.source, not both.")
    if time_opts.source is not None:
        raw = source_time_coord(observer, time_opts.source, owner=owner)
        allow_numeric = time_opts.source == observer.param_coord
    elif time is not None:
        raw = _time_from_array(time, owner=owner)
        allow_numeric = False
    else:
        raise ValueError(f"{owner}: observation time is required in astro A1.")
    coord = _require_time_array(raw, owner=owner, allow_numeric=allow_numeric)
    sequence_dim, batch_dims = _resolve_time_topology(coord, observer, owner=owner)
    return AstroTimeContext(coord=coord, sequence_dim=sequence_dim, batch_dims=batch_dims, scale=time_opts.scale)


def resolve_direction_runtime_context(
    *,
    location: object,
    time: object | None = None,
    opts: AstroTimeOptions | None = None,
    owner: str = "astro.resolve_direction_runtime_context",
) -> AstroDirectionRuntimeContext:
    """Resolve observer and time contexts for later direction operations."""
    observer = resolve_observer_context(location, owner=owner)
    time_ctx = resolve_time_context(observer, time=time, opts=opts, owner=owner)
    return AstroDirectionRuntimeContext(
        observer=observer,
        time=time_ctx,
        output_sequence_dim=time_ctx.sequence_dim,
        output_batch_dims=time_ctx.batch_dims,
    )


__all__ = [
    "AstroDirectionRuntimeContext",
    "AstroObserverContext",
    "AstroTimeContext",
    "coerce_observer_location",
    "resolve_direction_runtime_context",
    "resolve_observer_context",
    "resolve_time_context",
    "source_time_coord",
]
