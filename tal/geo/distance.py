from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.combine_ops.align import align_contexts
from tal.core.combine_ops.types import AlignOptions, CombineContext
from tal.core.metadata_optional import shared_optional_name
from tal.core.orchestration.inputs import coerce_operand
from tal.core.orchestration.resolve import effective_batch_dims, effective_sequence_dim, resolve_combine_contexts
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    select_single_numeric_var,
)
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from tal.linalg import Array
from .backends import geod_inverse
from .kernels import enu_bearing, enu_distance, geodesic_final_bearing, geodesic_initial_bearing
from .metadata import options_from_geodetic_metadata
from .options import ENUOptions, GeodesicOptions, coerce_geodetic_options, coerce_geodesic_options

_LLA_LABELS = ("lat", "lon", "alt")
_XYZ_LABELS = ("x", "y", "z")


def _wrap_error(exc: TypeError | ValueError, *, owner: str) -> TypeError | ValueError:
    text = str(exc)
    if text.startswith(f"{owner}:"):
        return exc
    return type(exc)(f"{owner}: {text}")


def _component(data: xr.DataArray, *, dim: str, label: str) -> xr.DataArray:
    return data.sel({dim: label}, drop=True)


def _geodetic_operand(value: object, *, owner: str):
    from .geodetic import GeodeticPosition

    operand = coerce_operand(value, owner=owner, label="other")
    if not isinstance(operand, AnalysisObject):
        raise TypeError(f"{owner}: other operand must be AO-like.")
    return GeodeticPosition.from_lla(operand)


def _aligned_contexts(left: AnalysisObject, right: AnalysisObject, *, owner: str) -> list[CombineContext]:
    contexts = resolve_combine_contexts([left, right], require_sequence=False, owner=owner)
    return align_contexts(contexts, opts=AlignOptions())


def _payload(ctx: CombineContext, *, labels: tuple[str, str, str], owner: str) -> tuple[xr.DataArray, str]:
    if len(ctx.core_dims) != 1:
        raise ValueError(f"{owner}: expected exactly one core dimension.")
    core_dim = ctx.core_dims[0]
    actual = require_explicit_unique_dim_labels(ctx.ds, dim=core_dim, owner=owner, what="geo payload")
    require_exact_labels(actual, expected=labels, owner=owner, what="geo payload core")
    var_name = select_single_numeric_var(ctx.ds, owner=owner, what="geo payload")
    return ctx.ds[var_name], core_dim


def _shared_geodetic_options(contexts: list[CombineContext], *, owner: str):
    opts = [options_from_geodetic_metadata(ctx.ds, owner=owner) for ctx in contexts]
    left = opts[0]
    for right in opts[1:]:
        if (
            left.datum,
            left.crs,
            left.angular_unit,
            left.height_reference,
        ) != (
            right.datum,
            right.crs,
            right.angular_unit,
            right.height_reference,
        ):
            raise ValueError(f"{owner}: operands must use compatible geodetic CRS metadata.")
    return coerce_geodetic_options(left, owner=owner, validate_crs=True)


def _scalar_spec(contexts: list[CombineContext]) -> CoreSchemaFinalizeSpec:
    sequence_dim = effective_sequence_dim(contexts, owner="geo.distance", require=False)
    batch_dims = effective_batch_dims(contexts)
    return CoreSchemaFinalizeSpec(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=(),
        param_name=shared_optional_name([ctx.param_coord for ctx in contexts]) if sequence_dim else None,
        size_name=shared_optional_name([ctx.sequence_size_coord for ctx in contexts]) if sequence_dim else None,
    )


def _finalize_array(
    contexts: list[CombineContext],
    result: xr.DataArray,
    *,
    labels: tuple[str, str, str],
    var_name: str,
    units: str,
    validate: bool,
    owner: str,
) -> Array:
    data_sources = tuple(_payload(ctx, labels=labels, owner=owner)[0] for ctx in contexts)
    arr = result.rename(var_name)
    ds = arr.to_dataset(name=var_name)
    ds[var_name].attrs["units"] = units
    source = AnalysisObject._from_unvalidated(contexts[0].ds)
    finalized = finalize_with_schema(
        source,
        ds,
        spec=_scalar_spec(contexts),
        validate=validate,
        owner=owner,
        optional_sources=data_sources,
    )
    if validate:
        return Array._from_validated(finalized.unsafe_data)
    return Array._from_unvalidated(finalized.unsafe_data)


def _geodesic_inverse(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    left_core: str,
    right_core: str,
    crs: str,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    lat1 = _component(left, dim=left_core, label="lat")
    lon1 = _component(left, dim=left_core, label="lon")
    lat2 = _component(right, dim=right_core, label="lat")
    lon2 = _component(right, dim=right_core, label="lon")
    return xr.apply_ufunc(
        geod_inverse,
        lat1,
        lon1,
        lat2,
        lon2,
        input_core_dims=((), (), (), ()),
        output_core_dims=((), (), ()),
        output_dtypes=(np.float64, np.float64, np.float64),
        kwargs={"crs": crs, "owner": owner},
        dask="parallelized",
    )


def _geodesic_distance(
    contexts: list[CombineContext],
    *,
    opts: GeodesicOptions,
    owner: str,
) -> xr.DataArray:
    left, left_core = _payload(contexts[0], labels=_LLA_LABELS, owner=owner)
    right, right_core = _payload(contexts[1], labels=_LLA_LABELS, owner=owner)
    geo_opts = _shared_geodetic_options(contexts, owner=owner)
    _, _, surface = _geodesic_inverse(left, right, left_core=left_core, right_core=right_core, crs=geo_opts.crs, owner=owner)
    if not opts.include_altitude:
        return surface
    alt1 = _component(left, dim=left_core, label="alt")
    alt2 = _component(right, dim=right_core, label="alt")
    return xr.apply_ufunc(
        np.hypot,
        surface,
        alt2 - alt1,
        input_core_dims=((), ()),
        output_core_dims=[()],
        output_dtypes=[np.float64],
        dask="parallelized",
    )


def _geodesic_bearing(
    contexts: list[CombineContext],
    *,
    kind: Literal["initial", "final"],
    owner: str,
) -> xr.DataArray:
    left, left_core = _payload(contexts[0], labels=_LLA_LABELS, owner=owner)
    right, right_core = _payload(contexts[1], labels=_LLA_LABELS, owner=owner)
    geo_opts = _shared_geodetic_options(contexts, owner=owner)
    az12, az21, distance = _geodesic_inverse(
        left,
        right,
        left_core=left_core,
        right_core=right_core,
        crs=geo_opts.crs,
        owner=owner,
    )
    lat1 = _component(left, dim=left_core, label="lat")
    lat2 = _component(right, dim=right_core, label="lat")
    kernel = geodesic_initial_bearing if kind == "initial" else geodesic_final_bearing
    raw = az12 if kind == "initial" else az21
    return xr.apply_ufunc(
        kernel,
        raw,
        distance,
        lat1,
        lat2,
        input_core_dims=((), (), (), ()),
        output_core_dims=[()],
        output_dtypes=[np.float64],
        dask="parallelized",
    )


def _local_contexts(left, right, *, opts: GeodesicOptions, owner: str) -> list[CombineContext]:
    enu_opts = ENUOptions(origin=opts.local_origin)
    left_enu = left.to_enu(opts=enu_opts, validate=False)
    right_enu = right.to_enu(opts=enu_opts, validate=False)
    contexts = resolve_combine_contexts([left_enu, right_enu], require_sequence=False, owner=owner)
    return align_contexts(contexts, opts=AlignOptions())


def _local_delta(contexts: list[CombineContext], *, owner: str) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    left, left_core = _payload(contexts[0], labels=_XYZ_LABELS, owner=owner)
    right, right_core = _payload(contexts[1], labels=_XYZ_LABELS, owner=owner)
    east = _component(right, dim=right_core, label="x") - _component(left, dim=left_core, label="x")
    north = _component(right, dim=right_core, label="y") - _component(left, dim=left_core, label="y")
    up = _component(right, dim=right_core, label="z") - _component(left, dim=left_core, label="z")
    return east, north, up


def _local_distance(contexts: list[CombineContext], *, opts: GeodesicOptions, owner: str) -> xr.DataArray:
    east, north, up = _local_delta(contexts, owner=owner)
    return xr.apply_ufunc(
        enu_distance,
        east,
        north,
        up,
        input_core_dims=((), (), ()),
        output_core_dims=[()],
        output_dtypes=[np.float64],
        kwargs={"include_altitude": opts.include_altitude},
        dask="parallelized",
    )


def _local_bearing(contexts: list[CombineContext], *, owner: str) -> xr.DataArray:
    east, north, _ = _local_delta(contexts, owner=owner)
    return xr.apply_ufunc(
        enu_bearing,
        east,
        north,
        input_core_dims=((), ()),
        output_core_dims=[()],
        output_dtypes=[np.float64],
        dask="parallelized",
    )


def _run_distance(position, other: object, *, opts: GeodesicOptions, validate: bool, owner: str) -> Array:
    right = _geodetic_operand(other, owner=owner)
    if opts.method == "local_enu":
        contexts = _local_contexts(position, right, opts=opts, owner=owner)
        result = _local_distance(contexts, opts=opts, owner=owner)
    else:
        contexts = _aligned_contexts(position, right, owner=owner)
        result = _geodesic_distance(contexts, opts=opts, owner=owner)
    labels = _XYZ_LABELS if opts.method == "local_enu" else _LLA_LABELS
    return _finalize_array(contexts, result, labels=labels, var_name="distance_m", units="m", validate=validate, owner=owner)


def _run_bearing(
    position,
    other: object,
    *,
    opts: GeodesicOptions,
    validate: bool,
    owner: str,
    kind: Literal["initial", "final"],
) -> Array:
    right = _geodetic_operand(other, owner=owner)
    if opts.method == "local_enu":
        contexts = _local_contexts(position, right, opts=opts, owner=owner)
        result = _local_bearing(contexts, owner=owner)
    else:
        contexts = _aligned_contexts(position, right, owner=owner)
        result = _geodesic_bearing(contexts, kind=kind, owner=owner)
    name = "initial_bearing_deg" if kind == "initial" else "final_bearing_deg"
    labels = _XYZ_LABELS if opts.method == "local_enu" else _LLA_LABELS
    return _finalize_array(contexts, result, labels=labels, var_name=name, units="degree", validate=validate, owner=owner)


def distance_to(position, other: object, *, opts: GeodesicOptions | None = None, validate: bool = True) -> Array:
    """Compute geodetic distance from `position` to `other`."""
    owner = "geo.GeodeticPosition.distance_to"
    normalized = coerce_geodesic_options(opts, owner=owner)
    try:
        return _run_distance(position, other, opts=normalized, validate=validate, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_error(exc, owner=owner) from exc


def initial_bearing_to(position, other: object, *, opts: GeodesicOptions | None = None, validate: bool = True) -> Array:
    """Compute initial geodetic bearing from `position` to `other`."""
    owner = "geo.GeodeticPosition.initial_bearing_to"
    normalized = coerce_geodesic_options(opts, owner=owner)
    try:
        return _run_bearing(position, other, opts=normalized, validate=validate, owner=owner, kind="initial")
    except (TypeError, ValueError) as exc:
        raise _wrap_error(exc, owner=owner) from exc


def final_bearing_to(position, other: object, *, opts: GeodesicOptions | None = None, validate: bool = True) -> Array:
    """Compute final geodetic bearing from `position` to `other`."""
    owner = "geo.GeodeticPosition.final_bearing_to"
    normalized = coerce_geodesic_options(opts, owner=owner)
    try:
        return _run_bearing(position, other, opts=normalized, validate=validate, owner=owner, kind="final")
    except (TypeError, ValueError) as exc:
        raise _wrap_error(exc, owner=owner) from exc


__all__ = [
    "distance_to",
    "final_bearing_to",
    "initial_bearing_to",
]
