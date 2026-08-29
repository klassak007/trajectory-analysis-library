from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from tal.core.orchestration.lazy import is_chunked_dataarray
from tal.core.schema_read import read_roles

from .backends.astropy import compute_sun_altaz
from .direction import TopocentricDirection
from .finalize import finalize_topocentric_direction
from .options import AstroBackend, AstroIERSOptions, AstroTimeOptions, coerce_iers_options, coerce_time_options
from .orchestration import (
    AstroDirectionRuntimeContext,
    AstroObserverContext,
    resolve_observer_context,
    resolve_time_context,
    source_time_coord,
)

_OWNER = "astro.direction_to_sun"
_SUN_BACKENDS = frozenset({"astropy", "spice"})
_LLA_LABELS = ("lat", "lon", "alt")
_ENU_LABELS = ("east", "north", "up")
_DIRECTION_VAR = "direction"
_ALTITUDE_VAR = "altitude_deg"
_AZIMUTH_VAR = "azimuth_deg"


@dataclass(frozen=True)
class SpiceSunOptions:
    """Reserved SPICE Sun-direction options.

    Parameters
    ----------
    No fields are defined in A2. The class is intentionally zero-field until
    SPICE execution is implemented.

    Notes
    -----
    ``SpiceSunOptions`` exists so ``SunDirectionOptions.spice`` has a stable,
    runtime-evaluable annotation. Passing a non-``None`` value to
    ``direction_to_sun`` fails closed until the SPICE backend phase.

    Examples
    --------
    >>> from tal.astro.sun import SpiceSunOptions
    >>> SpiceSunOptions()
    SpiceSunOptions()
    """


@dataclass(frozen=True)
class SunDirectionOptions:
    """Options for topocentric Sun direction calculation.

    Parameters
    ----------
    backend : {'astropy', 'spice'}, optional
        Requested astronomy backend. A2 executes only ``"astropy"``.
    time : AstroTimeOptions | None, optional
        Time scale and optional observer source coordinate.
    iers : AstroIERSOptions | None, optional
        Astropy IERS download and degraded-accuracy policy.
    spice : SpiceSunOptions | None, optional
        Reserved SPICE backend options. Non-``None`` values fail closed in A2.

    Notes
    -----
    A2 requires absolute datetime-like observation time. Numeric TAL parameter
    coordinates are valid for generic TAL param operations but are not
    interpreted as Sun-observation time.

    Examples
    --------
    >>> from tal.astro import AstroIERSOptions
    >>> from tal.astro.sun import SunDirectionOptions
    >>> opts = SunDirectionOptions(iers=AstroIERSOptions(auto_download=False))
    >>> opts.backend
    'astropy'
    """

    backend: AstroBackend = "astropy"
    time: AstroTimeOptions | None = None
    iers: AstroIERSOptions | None = None
    spice: SpiceSunOptions | None = None


def _coerce_backend(value: object, *, owner: str) -> AstroBackend:
    if isinstance(value, str) and value in _SUN_BACKENDS:
        return value  # type: ignore[return-value]
    raise ValueError(f"{owner}: backend must be one of {sorted(_SUN_BACKENDS)!r}; got {value!r}.")


def _coerce_options(opts: SunDirectionOptions | None, *, owner: str) -> SunDirectionOptions:
    if opts is None:
        opts = SunDirectionOptions()
    if not isinstance(opts, SunDirectionOptions):
        raise TypeError(f"{owner}: options must be SunDirectionOptions or None.")
    backend = _coerce_backend(opts.backend, owner=owner)
    if backend == "spice":
        raise ValueError(f"{owner}: backend='spice' is reserved for a later astro phase; use backend='astropy'.")
    if opts.spice is not None:
        raise ValueError(f"{owner}: spice options are reserved for a later astro phase.")
    time = None if opts.time is None else coerce_time_options(opts.time, owner=owner)
    iers = coerce_iers_options(opts.iers, owner=owner)
    return SunDirectionOptions(backend=backend, time=time, iers=iers, spice=None)


def _fail_if_raw_lazy(value: object, *, owner: str, field: str) -> None:
    has_chunks = getattr(value, "chunks", None) is not None
    graph = getattr(value, "__dask_graph__", None)
    if not has_chunks and graph is None:
        return
    raise ValueError(f"{owner}: Dask-backed {field} is not supported in astro A2; materialize explicitly.")


def _fail_if_object_dtype(value: object, *, owner: str, field: str) -> None:
    dtype = getattr(value, "dtype", None)
    if dtype is not None and np.dtype(dtype).kind == "O":
        raise ValueError(f"{owner}: {field} must be datetime64, not object dtype.")


def _require_eager_array(coord: xr.DataArray, *, owner: str, field: str) -> None:
    if is_chunked_dataarray(coord):
        raise ValueError(f"{owner}: Dask-backed {field} is not supported in astro A2; materialize explicitly.")


def _require_datetime64(coord: xr.DataArray, *, owner: str, field: str) -> None:
    if not np.issubdtype(np.dtype(coord.dtype), np.datetime64):
        raise ValueError(f"{owner}: {field} must be absolute datetime64 for Sun direction; got {coord.dtype!r}.")


def _preflight_explicit_time(value: object, *, owner: str) -> None:
    if isinstance(value, xr.DataArray):
        _require_eager_array(value, owner=owner, field="time")
    else:
        _fail_if_raw_lazy(value, owner=owner, field="time")
    _fail_if_object_dtype(value, owner=owner, field="time")


def _preflight_time_source(observer: AstroObserverContext, opts: AstroTimeOptions, *, owner: str) -> None:
    if opts.source is None:
        return
    coord = source_time_coord(observer, opts.source, owner=owner)
    _require_eager_array(coord, owner=owner, field=f"time.source {opts.source!r}")
    _require_datetime64(coord, owner=owner, field=f"time.source {opts.source!r}")


def _resolve_sun_runtime_context(
    *,
    location: object,
    time: object | None,
    time_opts: AstroTimeOptions | None,
    owner: str,
) -> AstroDirectionRuntimeContext:
    observer = resolve_observer_context(location, owner=owner)
    normalized_time = coerce_time_options(time_opts, owner=owner)
    if time is None:
        _preflight_time_source(observer, normalized_time, owner=owner)
    else:
        _preflight_explicit_time(time, owner=owner)
    time_ctx = resolve_time_context(observer, time=time, opts=normalized_time, owner=owner)
    _require_eager_array(time_ctx.coord, owner=owner, field="time")
    _require_datetime64(time_ctx.coord, owner=owner, field="time")
    return AstroDirectionRuntimeContext(
        observer=observer,
        time=time_ctx,
        output_sequence_dim=time_ctx.sequence_dim,
        output_batch_dims=time_ctx.batch_dims,
    )


def _observer_core_dim(context: AstroDirectionRuntimeContext, *, owner: str) -> str:
    _, _, _, core_dims = read_roles(context.observer.ds)
    if len(core_dims) != 1:
        raise ValueError(f"{owner}: observer must have exactly one LLA core dimension.")
    return core_dims[0]


def _observer_components(context: AstroDirectionRuntimeContext, *, owner: str) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    data = context.observer.data
    if data is None:
        raise ValueError(f"{owner}: observer location must contain one numeric LLA variable.")
    core_dim = _observer_core_dim(context, owner=owner)
    components = tuple(data.sel({core_dim: label}, drop=True) for label in _LLA_LABELS)
    for label, component in zip(_LLA_LABELS, components, strict=True):
        _require_eager_array(component, owner=owner, field=f"observer {label}")
    return components  # type: ignore[return-value]


def _semantic_dims(context: AstroDirectionRuntimeContext) -> tuple[str, ...]:
    dims = []
    if context.output_sequence_dim is not None:
        dims.append(context.output_sequence_dim)
    dims.extend(context.output_batch_dims)
    return tuple(dims)


def _align_inputs(
    context: AstroDirectionRuntimeContext,
    *,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    lat, lon, alt = _observer_components(context, owner=owner)
    aligned = xr.align(lat, lon, alt, context.time.coord, join="exact", copy=False)
    broadcast = xr.broadcast(*aligned)
    dims = _semantic_dims(context)
    for item in broadcast:
        extra = tuple(dim for dim in item.dims if dim not in dims)
        if extra:
            raise ValueError(f"{owner}: aligned Sun direction topology has unsupported dims {extra!r}.")
    out = tuple(item.transpose(*dims) if dims else item for item in broadcast)
    return out  # type: ignore[return-value]


def _coords_for_dims(arrays: tuple[xr.DataArray, ...], dims: tuple[str, ...]) -> dict[str, xr.DataArray]:
    coords: dict[str, xr.DataArray] = {}
    for dim in dims:
        for arr in arrays:
            if dim in arr.coords and arr.coords[dim].dims == (dim,):
                coords[dim] = arr.coords[dim]
                break
    return coords


def _output_dataset(
    arrays: tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray],
    altitude_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    *,
    dims: tuple[str, ...],
) -> xr.Dataset:
    alt_rad = np.radians(altitude_deg)
    az_rad = np.radians(azimuth_deg)
    direction = np.stack(
        [np.cos(alt_rad) * np.sin(az_rad), np.cos(alt_rad) * np.cos(az_rad), np.sin(alt_rad)],
        axis=-1,
    )
    coords = _coords_for_dims(arrays, dims)
    ds = xr.Dataset(
        {
            _DIRECTION_VAR: ((*dims, "enu"), direction),
            _ALTITUDE_VAR: (dims, altitude_deg),
            _AZIMUTH_VAR: (dims, azimuth_deg),
        },
        coords={**coords, "enu": list(_ENU_LABELS)},
    )
    return ds


def direction_to_sun(
    location: object,
    *,
    time: object | None = None,
    opts: SunDirectionOptions | None = None,
    validate: bool = True,
) -> TopocentricDirection:
    """Compute the topocentric ENU direction to the Sun.

    Parameters
    ----------
    location : object
        ``GeodeticPosition`` or AO-like LLA observer location.
    time : object | None, optional
        Explicit datetime-like scalar or one-dimensional observation time.
    opts : SunDirectionOptions | None, optional
        Backend and timing policy. Key fields are ``time`` and ``iers``;
        ``backend`` selects Astropy or the reserved SPICE path.
    validate : bool, optional
        Whether to validate the finalized ``TopocentricDirection``.

    Returns
    -------
    TopocentricDirection
        Direction payload with ``direction``, ``altitude_deg``, and
        ``azimuth_deg`` variables.

    Raises
    ------
    ImportError
        If Astropy is required but not installed.
    TypeError
        If options or observer inputs have unsupported types.
    ValueError
        If time is not absolute datetime64, if SPICE is requested in A2, or if
        observer/time topology cannot be aligned.

    Notes
    -----
    A2 executes only the Astropy backend. Dask-backed observer and time arrays
    fail closed before the Astropy boundary; materialize those arrays
    explicitly before calling this function.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.astro import AstroIERSOptions
    >>> from tal.astro.sun import SunDirectionOptions, direction_to_sun
    >>> from tal.core import AnalysisObject
    >>> from tal.geo import GeodeticPosition
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset(
    ...         {"lla": (("sample", "lla_axis"), [[35.0, -106.0, 1600.0]])},
    ...         coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
    ...     ),
    ...     sequence_dim="sample",
    ...     core_dims=("lla_axis",),
    ... )
    >>> opts = SunDirectionOptions(iers=AstroIERSOptions(degraded_accuracy="ignore"))
    >>> sun = direction_to_sun(GeodeticPosition.from_lla(ao), time="2024-06-01T12:00:00", opts=opts)
    >>> sorted(sun.unsafe_data.data_vars)
    ['altitude_deg', 'azimuth_deg', 'direction']
    """
    options = _coerce_options(opts, owner=_OWNER)
    context = _resolve_sun_runtime_context(location=location, time=time, time_opts=options.time, owner=_OWNER)
    arrays = _align_inputs(context, owner=_OWNER)
    latitude, longitude, altitude, time_coord = (np.asarray(item.data) for item in arrays)
    altitude_deg, azimuth_deg = compute_sun_altaz(
        latitude_deg=latitude,
        longitude_deg=longitude,
        altitude_m=altitude,
        time=time_coord,
        scale=context.time.scale,
        iers_options=options.iers or AstroIERSOptions(),
        owner=_OWNER,
    )
    ds = _output_dataset(arrays, altitude_deg, azimuth_deg, dims=_semantic_dims(context))
    return finalize_topocentric_direction(
        context,
        ds,
        core_dim="enu",
        backend="astropy",
        time_scale=context.time.scale,
        validate=validate,
        owner=_OWNER,
    )


__all__ = ["SpiceSunOptions", "SunDirectionOptions", "direction_to_sun"]
