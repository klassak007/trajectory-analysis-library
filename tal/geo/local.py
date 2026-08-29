from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from tal.core.orchestration.context import resolve_dataset_contexts
from tal.spatial import Position
from tal.spatial.metadata import set_expressed_in, set_position_rep
from tal.utils.frame_schema import get_frames, set_frames

from .backends import transform_lla_to_ecef
from .conversion import (
    _CTX_OPTIONS,
    _XYZ_LABELS,
    _assemble,
    _component,
    _context,
    _finalize_core_schema,
    _target_dims,
    from_ecef,
    to_ecef,
)
from .kernels import ecef_delta_to_enu, enu_delta_to_ecef
from .metadata import (
    assign_origin_coordinates,
    inline_origin_metadata,
    is_cartesian_geo_system,
    options_from_enu_provenance,
    options_from_geodetic_metadata,
    read_cartesian_geo_block,
    read_cartesian_geo_block_if_present,
    read_enu_origin_metadata,
    read_geo_block,
    set_ecef_metadata,
    set_enu_metadata,
)
from .options import (
    ENUOptions,
    GeodeticOptions,
    LocalOrigin,
    coerce_enu_options,
    coerce_geodetic_options,
    coerce_local_origin,
)

if TYPE_CHECKING:
    from tal.core.orchestration.context import DatasetContext

    from .geodetic import GeodeticPosition


@dataclass(frozen=True)
class _OriginComponents:
    lat: xr.DataArray
    lon: xr.DataArray
    alt: xr.DataArray
    opts: GeodeticOptions
    metadata: dict[str, Any] | None


def _geo_opts_from_enu(opts: ENUOptions, *, owner: str, source_opts: GeodeticOptions | None = None) -> GeodeticOptions:
    base = source_opts or GeodeticOptions()
    return coerce_geodetic_options(
        replace(base, ecef_frame=opts.ecef_frame, strict_frame=opts.strict_frame),
        owner=owner,
        validate_crs=True,
    )


def _resolve_enu_options(origin: object | None, opts: ENUOptions | None, *, owner: str) -> tuple[object | None, ENUOptions]:
    normalized = coerce_enu_options(opts, owner=owner)
    effective_origin = origin if origin is not None else normalized.origin
    return effective_origin, normalized


def _semantic_dims(ctx: "DatasetContext", *, core_dim: str) -> tuple[str, ...]:
    assert ctx.data is not None
    return tuple(dim for dim in ctx.data.dims if dim != core_dim)


def _require_origin_dims(origin: _OriginComponents, *, allowed_dims: tuple[str, ...], owner: str) -> None:
    for label, arr in (("lat", origin.lat), ("lon", origin.lon), ("alt", origin.alt)):
        illegal = tuple(dim for dim in arr.dims if dim not in allowed_dims)
        if illegal:
            raise ValueError(f"{owner}: origin {label} dims {illegal!r} are not in source semantic dims {allowed_dims!r}.")


def _origin_from_local(origin: LocalOrigin, *, owner: str) -> _OriginComponents:
    local = coerce_local_origin(origin, owner=owner)
    geo_opts = coerce_geodetic_options(local.opts, owner=owner, validate_crs=True)
    return _OriginComponents(
        lat=xr.DataArray(local.lat),
        lon=xr.DataArray(local.lon),
        alt=xr.DataArray(local.alt),
        opts=geo_opts,
        metadata=inline_origin_metadata(local, owner=owner),
    )


def _origin_from_geodetic(
    source: Position,
    origin: object,
    *,
    owner: str,
) -> tuple["DatasetContext", _OriginComponents]:
    from .geodetic import GeodeticPosition

    origin_position = origin if isinstance(origin, GeodeticPosition) else GeodeticPosition.from_lla(origin)
    source_ctx, origin_ctx = resolve_dataset_contexts([source, origin_position], owner=owner, options=_CTX_OPTIONS)
    assert origin_ctx.data is not None
    origin_core = origin_ctx.core_dims[0]
    provenance_opts = options_from_geodetic_metadata(origin_ctx.ds, owner=owner)
    return source_ctx, _OriginComponents(
        lat=_component(origin_ctx.data, dim=origin_core, label="lat"),
        lon=_component(origin_ctx.data, dim=origin_core, label="lon"),
        alt=_component(origin_ctx.data, dim=origin_core, label="alt"),
        opts=coerce_geodetic_options(provenance_opts, owner=owner, validate_crs=True),
        metadata=None,
    )


def _preflight_geodetic_origin(
    source: "GeodeticPosition",
    origin: object,
    *,
    owner: str,
) -> object:
    if isinstance(origin, LocalOrigin):
        return coerce_local_origin(origin, owner=owner)

    from .geodetic import GeodeticPosition

    origin_position = origin if isinstance(origin, GeodeticPosition) else GeodeticPosition.from_lla(origin)
    source_ctx, origin_ctx = resolve_dataset_contexts([source, origin_position], owner=owner, options=_CTX_OPTIONS)
    assert source_ctx.data is not None and origin_ctx.data is not None
    source_core = source_ctx.core_dims[0]
    origin_core = origin_ctx.core_dims[0]
    origin_components = _OriginComponents(
        lat=_component(origin_ctx.data, dim=origin_core, label="lat"),
        lon=_component(origin_ctx.data, dim=origin_core, label="lon"),
        alt=_component(origin_ctx.data, dim=origin_core, label="alt"),
        opts=GeodeticOptions(),
        metadata=None,
    )
    _require_origin_dims(origin_components, allowed_dims=_semantic_dims(source_ctx, core_dim=source_core), owner=owner)
    return origin_position


def _origin_from_provenance(ctx: "DatasetContext", *, core_dim: str, owner: str) -> _OriginComponents:
    semantic_dims = _semantic_dims(ctx, core_dim=core_dim)
    origin = read_enu_origin_metadata(ctx.ds, core_dim=core_dim, semantic_dims=semantic_dims, owner=owner)
    block = read_cartesian_geo_block(ctx.ds, system="enu", owner=owner)
    geo_opts = coerce_geodetic_options(options_from_enu_provenance(block, owner=owner), owner=owner, validate_crs=True)
    if origin.get("storage") == "inline":
        return _origin_from_local(
            LocalOrigin(float(origin["lat"]), float(origin["lon"]), float(origin["alt"]), opts=geo_opts),
            owner=owner,
        )
    return _OriginComponents(
        lat=ctx.ds.coords[str(origin["lat_coord"])],
        lon=ctx.ds.coords[str(origin["lon_coord"])],
        alt=ctx.ds.coords[str(origin["alt_coord"])],
        opts=geo_opts,
        metadata=dict(origin),
    )


def _resolve_origin(
    source: Position,
    origin: object | None,
    *,
    ctx: "DatasetContext",
    core_dim: str,
    owner: str,
) -> tuple["DatasetContext", _OriginComponents]:
    if origin is None:
        return ctx, _origin_from_provenance(ctx, core_dim=core_dim, owner=owner)
    if isinstance(origin, LocalOrigin):
        return ctx, _origin_from_local(origin, owner=owner)
    return _origin_from_geodetic(source, origin, owner=owner)


def _origin_ecef(origin: _OriginComponents, *, owner: str) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    return xr.apply_ufunc(
        transform_lla_to_ecef,
        origin.lat,
        origin.lon,
        origin.alt,
        input_core_dims=((), (), ()),
        output_core_dims=((), (), ()),
        output_dtypes=(np.float64, np.float64, np.float64),
        kwargs={"crs": origin.opts.crs, "ecef_crs": origin.opts.ecef_crs, "owner": owner},
        dask="parallelized",
    )


def _ecef_to_enu_components(
    data: xr.DataArray,
    *,
    core_dim: str,
    origin: _OriginComponents,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    x0, y0, z0 = _origin_ecef(origin, owner=owner)
    dx = _component(data, dim=core_dim, label="x") - x0
    dy = _component(data, dim=core_dim, label="y") - y0
    dz = _component(data, dim=core_dim, label="z") - z0
    return xr.apply_ufunc(
        ecef_delta_to_enu,
        dx,
        dy,
        dz,
        origin.lat,
        origin.lon,
        input_core_dims=((), (), (), (), ()),
        output_core_dims=((), (), ()),
        output_dtypes=(np.float64, np.float64, np.float64),
        dask="parallelized",
    )


def _enu_to_ecef_components(
    data: xr.DataArray,
    *,
    core_dim: str,
    origin: _OriginComponents,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    x0, y0, z0 = _origin_ecef(origin, owner=owner)
    de, dn, du = xr.apply_ufunc(
        enu_delta_to_ecef,
        _component(data, dim=core_dim, label="x"),
        _component(data, dim=core_dim, label="y"),
        _component(data, dim=core_dim, label="z"),
        origin.lat,
        origin.lon,
        input_core_dims=((), (), (), (), ()),
        output_core_dims=((), (), ()),
        output_dtypes=(np.float64, np.float64, np.float64),
        dask="parallelized",
    )
    return x0 + de, y0 + dn, z0 + du


def _check_ecef_source_frame(ds: xr.Dataset, *, opts: ENUOptions, owner: str) -> None:
    parent, _ = get_frames(ds)
    if opts.ecef_frame is None or parent == opts.ecef_frame:
        return
    if opts.strict_frame:
        raise ValueError(f"{owner}: source frame parent {parent!r} is not compatible with {opts.ecef_frame!r}.")


def _apply_enu_frame_policy(ds: xr.Dataset, *, source_ds: xr.Dataset, opts: ENUOptions, owner: str) -> xr.Dataset:
    if opts.output_frame is None:
        out = set_frames(ds, parent=None, child=None, validate=False)
        return set_expressed_in(out, expressed_in=None, validate=False, owner=owner)
    _, child = get_frames(source_ds)
    out = set_frames(ds, parent=opts.output_frame, child=child, validate=False)
    return set_expressed_in(out, expressed_in=opts.output_frame, validate=False, owner=owner)


def _apply_ecef_frame_policy(ds: xr.Dataset, *, opts: ENUOptions, owner: str) -> xr.Dataset:
    out = set_frames(ds, parent=opts.ecef_frame, child=None, validate=False)
    expressed_in = opts.ecef_frame if opts.ecef_frame is not None else None
    return set_expressed_in(out, expressed_in=expressed_in, validate=False, owner=owner)


def _attach_origin_metadata(
    ds: xr.Dataset,
    *,
    origin: _OriginComponents,
    geo_opts: GeodeticOptions,
    allowed_dims: tuple[str, ...],
) -> xr.Dataset:
    if origin.metadata is not None:
        return set_enu_metadata(ds, opts=geo_opts, origin=origin.metadata, validate=False)
    for arr in (origin.lat, origin.lon, origin.alt):
        illegal = tuple(dim for dim in arr.dims if dim not in allowed_dims)
        if illegal:
            raise ValueError(f"geo.to_enu: origin coordinate dims {illegal!r} are not output semantic dims.")
    out, origin_metadata = assign_origin_coordinates(ds, lat=origin.lat, lon=origin.lon, alt=origin.alt)
    return set_enu_metadata(out, opts=geo_opts, origin=origin_metadata, validate=False)


def position_to_lla(position: Position, *, opts: GeodeticOptions | None = None, validate: bool = True):
    """Convert a Position.geo source from ECEF to LLA."""
    owner = "geo.Position.geo.to_lla"
    source = position if isinstance(position, Position) else Position(position)
    block = read_cartesian_geo_block_if_present(source.unsafe_data, system="ecef", owner=owner)
    if opts is None and block is None:
        raise ValueError(f"{owner}: ECEF geo provenance is required when opts is None.")
    return from_ecef(source, opts=opts, validate=validate)


def ecef_to_enu(
    position: Position,
    origin: object | None = None,
    *,
    opts: ENUOptions | None = None,
    validate: bool = True,
) -> Position:
    """Convert an ECEF Position to local ENU Position."""
    owner = "geo.Position.geo.to_enu"
    effective_origin, enu_opts = _resolve_enu_options(origin, opts, owner=owner)
    if effective_origin is None:
        raise ValueError(f"{owner}: origin is required for ENU conversion.")
    source = position if isinstance(position, Position) else Position(position)
    block = read_geo_block(source.unsafe_data, owner=owner)
    if block is not None and not is_cartesian_geo_system(source.unsafe_data, system="ecef", owner=owner):
        raise ValueError(f"{owner}: source Position must be canonical ECEF geo metadata.")
    if block is None and opts is None:
        raise ValueError(f"{owner}: ECEF geo provenance is required when opts is None.")
    ctx = _context(source, owner=owner)
    assert ctx.data is not None and ctx.var_name is not None
    core_dim = ctx.core_dims[0]
    ctx, origin_components = _resolve_origin(source, effective_origin, ctx=ctx, core_dim=core_dim, owner=owner)
    allowed_dims = _semantic_dims(ctx, core_dim=core_dim)
    _require_origin_dims(origin_components, allowed_dims=allowed_dims, owner=owner)
    _check_ecef_source_frame(ctx.ds, opts=enu_opts, owner=owner)
    out_core_dim = core_dim
    components = _ecef_to_enu_components(ctx.data, core_dim=core_dim, origin=origin_components, owner=owner)
    ds = _assemble(
        components,
        labels=_XYZ_LABELS,
        core_dim=out_core_dim,
        target_dims=_target_dims(ctx.data, old_core_dim=core_dim, new_core_dim=out_core_dim),
        var_name=ctx.var_name,
    )
    ds = _finalize_core_schema(ctx, ds, core_dim=out_core_dim, validate=validate, owner=owner)
    ds = set_position_rep(ds, rep="cart", validate=False, owner=owner)
    ds = _attach_origin_metadata(ds, origin=origin_components, geo_opts=origin_components.opts, allowed_dims=allowed_dims)
    ds = _apply_enu_frame_policy(ds, source_ds=ctx.ds, opts=enu_opts, owner=owner)
    if validate:
        return Position._from_validated(ds)
    return Position._from_unvalidated(ds)


def geodetic_to_enu(
    position: "GeodeticPosition",
    origin: object | None = None,
    *,
    opts: ENUOptions | None = None,
    validate: bool = True,
) -> Position:
    """Convert a GeodeticPosition to local ENU Position."""
    owner = "geo.GeodeticPosition.to_enu"
    effective_origin, enu_opts = _resolve_enu_options(origin, opts, owner=owner)
    if effective_origin is None:
        raise ValueError(f"{owner}: origin is required for ENU conversion.")
    preflight_origin = _preflight_geodetic_origin(position, effective_origin, owner=owner)
    source_opts = options_from_geodetic_metadata(position.unsafe_data, owner=owner)
    ecef_opts = _geo_opts_from_enu(enu_opts, owner=owner, source_opts=source_opts)
    ecef = to_ecef(position, opts=ecef_opts, validate=validate)
    return ecef_to_enu(ecef, origin=preflight_origin, opts=enu_opts, validate=validate)


def enu_to_ecef(
    position: Position,
    origin: object | None = None,
    *,
    opts: ENUOptions | None = None,
    validate: bool = True,
) -> Position:
    """Convert a local ENU Position back to ECEF."""
    owner = "geo.Position.geo.to_ecef"
    effective_origin, enu_opts = _resolve_enu_options(origin, opts, owner=owner)
    source = position if isinstance(position, Position) else Position(position)
    if not is_cartesian_geo_system(source.unsafe_data, system="enu", owner=owner):
        raise ValueError(f"{owner}: source Position must have canonical ENU geo metadata.")
    ctx = _context(source, owner=owner)
    assert ctx.data is not None and ctx.var_name is not None
    core_dim = ctx.core_dims[0]
    ctx, origin_components = _resolve_origin(source, effective_origin, ctx=ctx, core_dim=core_dim, owner=owner)
    allowed_dims = _semantic_dims(ctx, core_dim=core_dim)
    _require_origin_dims(origin_components, allowed_dims=allowed_dims, owner=owner)
    out_core_dim = core_dim
    components = _enu_to_ecef_components(ctx.data, core_dim=core_dim, origin=origin_components, owner=owner)
    ds = _assemble(
        components,
        labels=_XYZ_LABELS,
        core_dim=out_core_dim,
        target_dims=_target_dims(ctx.data, old_core_dim=core_dim, new_core_dim=out_core_dim),
        var_name=ctx.var_name,
    )
    ds = _finalize_core_schema(ctx, ds, core_dim=out_core_dim, validate=validate, owner=owner)
    ds = set_position_rep(ds, rep="cart", validate=False, owner=owner)
    geo_opts = replace(origin_components.opts, ecef_frame=enu_opts.ecef_frame, strict_frame=enu_opts.strict_frame)
    ds = set_ecef_metadata(ds, opts=geo_opts, validate=False)
    ds = _apply_ecef_frame_policy(ds, opts=enu_opts, owner=owner)
    if validate:
        return Position._from_validated(ds)
    return Position._from_unvalidated(ds)


__all__ = [
    "ecef_to_enu",
    "enu_to_ecef",
    "geodetic_to_enu",
    "position_to_lla",
]
