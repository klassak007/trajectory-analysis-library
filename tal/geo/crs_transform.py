from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

from tal.core.schema import merge_schema
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.spatial import Position
from tal.spatial.conversion.finalize import allocate_free_dim_name, dataset_dim_names
from tal.spatial.metadata import set_position_rep

from .backends import base_geodetic_crs, crs_has_height_axis, normalize_crs_with_class, transform_crs_xy, transform_crs_xyz
from .conversion import (
    _XYZ_LABELS,
    _component,
    _context,
    _ecef_core_dim,
    _finalize_core_schema,
    _lla_core_dim,
    _target_dims,
)
from .metadata import (
    normalize_geodetic_metadata,
    options_from_ecef_provenance,
    options_from_geodetic_metadata,
    read_geo_block,
    read_cartesian_geo_block,
    read_projected_geo_block,
    set_ecef_metadata,
    set_projected_metadata,
)
from .options import GeodeticOptions

_LLA_LABELS = ("lat", "lon", "alt")
_PROJECTED_2D_LABELS = ("easting", "northing")
_PROJECTED_3D_LABELS = ("easting", "northing", "height")


@dataclass(frozen=True)
class _Source:
    kind: Literal["geodetic", "ecef", "projected"]
    value: object
    src_crs: str
    geodetic_crs: str
    has_height: bool


def _projected_core_dim(source_ds: xr.Dataset) -> str:
    return allocate_free_dim_name(
        existing_dims=dataset_dim_names(source_ds),
        candidates=("projected", "xy"),
        base="projected",
        owner="geo.transform_crs",
        what="projected axis",
    )


def _require_dst_string(dst: object, *, owner: str) -> str:
    if isinstance(dst, str) and dst.strip():
        return dst.strip()
    raise TypeError(f"{owner}: dst must be a non-empty CRS string.")


def _source(value: object, *, owner: str) -> _Source:
    from .geodetic import GeodeticPosition
    from .projected import ProjectedPosition

    if isinstance(value, GeodeticPosition):
        opts = options_from_geodetic_metadata(value.unsafe_data, owner=owner)
        return _Source("geodetic", value, opts.crs, opts.crs, True)
    if isinstance(value, ProjectedPosition):
        block = read_projected_geo_block(value.unsafe_data, owner=owner)
        return _Source("projected", value, str(block["crs"]), str(block["geodetic_crs"]), _projected_has_height(value))
    if isinstance(value, Position):
        block = read_cartesian_geo_block(value.unsafe_data, system="ecef", owner=owner)
        opts = options_from_ecef_provenance(block, owner=owner)
        return _Source("ecef", value, opts.ecef_crs, opts.crs, True)
    try:
        ao = coerce_analysis_object_input(value, owner=owner)
    except TypeError as exc:
        raise TypeError(f"{owner}: value must be GeodeticPosition, ProjectedPosition, or ECEF Position.") from exc
    block = read_geo_block(ao.unsafe_data, owner=owner)
    if block is None:
        raise ValueError(f"{owner}: source CRS metadata is required.")
    if block.get("kind") == "geodetic_position":
        return _source(GeodeticPosition._from_unvalidated(ao.unsafe_data), owner=owner)
    if block.get("kind") == "projected_position":
        return _source(ProjectedPosition._from_unvalidated(ao.unsafe_data), owner=owner)
    return _source(Position(ao), owner=owner)


def _projected_has_height(value: object) -> bool:
    ctx = _context(value, owner="geo.transform_crs")
    assert ctx.data is not None
    return int(ctx.ds.sizes[ctx.core_dims[0]]) == 3


def _xyz_inputs(src: _Source, *, owner: str) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None, str, object]:
    ctx = _context(src.value, owner=owner)
    assert ctx.data is not None
    core_dim = ctx.core_dims[0]
    if src.kind == "geodetic":
        return (
            _component(ctx.data, dim=core_dim, label="lon"),
            _component(ctx.data, dim=core_dim, label="lat"),
            _component(ctx.data, dim=core_dim, label="alt"),
            core_dim,
            ctx,
        )
    if src.kind == "projected":
        z = _component(ctx.data, dim=core_dim, label="height") if src.has_height else None
        return (
            _component(ctx.data, dim=core_dim, label="easting"),
            _component(ctx.data, dim=core_dim, label="northing"),
            z,
            core_dim,
            ctx,
        )
    return (
        _component(ctx.data, dim=core_dim, label="x"),
        _component(ctx.data, dim=core_dim, label="y"),
        _component(ctx.data, dim=core_dim, label="z"),
        core_dim,
        ctx,
    )


def _transform_xy(x: xr.DataArray, y: xr.DataArray, *, src_crs: str, dst_crs: str, owner: str):
    return xr.apply_ufunc(
        transform_crs_xy,
        x,
        y,
        input_core_dims=((), ()),
        output_core_dims=((), ()),
        output_dtypes=(np.float64, np.float64),
        kwargs={"src_crs": src_crs, "dst_crs": dst_crs, "owner": owner},
        dask="parallelized",
    )


def _transform_xyz(
    x: xr.DataArray,
    y: xr.DataArray,
    z: xr.DataArray,
    *,
    src_crs: str,
    dst_crs: str,
    owner: str,
):
    return xr.apply_ufunc(
        transform_crs_xyz,
        x,
        y,
        z,
        input_core_dims=((), (), ()),
        output_core_dims=((), (), ()),
        output_dtypes=(np.float64, np.float64, np.float64),
        kwargs={"src_crs": src_crs, "dst_crs": dst_crs, "owner": owner},
        dask="parallelized",
    )


def _assemble(components, *, labels, core_dim: str, old_core_dim: str, var_name: str, ctx) -> xr.Dataset:
    dim = xr.IndexVariable(core_dim, list(labels))
    arr = xr.concat(list(components), dim=dim).transpose(*_target_dims(ctx.data, old_core_dim=old_core_dim, new_core_dim=core_dim))
    arr.name = var_name
    return arr.to_dataset(name=var_name)


def _to_geodetic(src: _Source, *, dst: str, validate: bool, owner: str):
    from .geodetic import GeodeticPosition

    if not src.has_height:
        raise ValueError(f"{owner}: 2D projected sources cannot be transformed to geographic output.")
    x, y, z, old_core_dim, ctx = _xyz_inputs(src, owner=owner)
    assert z is not None and ctx.data is not None and ctx.var_name is not None
    lon, lat, alt = _transform_xyz(x, y, z, src_crs=src.src_crs, dst_crs=dst, owner=owner)
    core_dim = _lla_core_dim(ctx.ds)
    ds = _assemble((lat, lon, alt), labels=_LLA_LABELS, core_dim=core_dim, old_core_dim=old_core_dim, var_name=ctx.var_name, ctx=ctx)
    ds = _finalize_core_schema(ctx, ds, core_dim=core_dim, validate=validate, owner=owner)
    ds = merge_schema(ds, {"ext": {"geo": None}}, validate=False)
    ds = normalize_geodetic_metadata(ds, opts=GeodeticOptions(crs=dst), validate=False, validate_crs=True, owner=owner)
    return GeodeticPosition._from_validated(ds) if validate else GeodeticPosition._from_unvalidated(ds)


def _to_ecef(src: _Source, *, dst: str, validate: bool, owner: str) -> Position:
    if not src.has_height:
        raise ValueError(f"{owner}: 2D projected sources cannot be transformed to ECEF output.")
    x, y, z, old_core_dim, ctx = _xyz_inputs(src, owner=owner)
    assert z is not None and ctx.data is not None and ctx.var_name is not None
    out = _transform_xyz(x, y, z, src_crs=src.src_crs, dst_crs=dst, owner=owner)
    core_dim = _ecef_core_dim(ctx.ds)
    ds = _assemble(out, labels=_XYZ_LABELS, core_dim=core_dim, old_core_dim=old_core_dim, var_name=ctx.var_name, ctx=ctx)
    ds = _finalize_core_schema(ctx, ds, core_dim=core_dim, validate=validate, owner=owner)
    ds = set_position_rep(ds, rep="cart", validate=False, owner=owner)
    ds = set_ecef_metadata(ds, opts=GeodeticOptions(crs=src.geodetic_crs, ecef_crs=dst), validate=False)
    return Position._from_validated(ds) if validate else Position._from_unvalidated(ds)


def _to_projected(src: _Source, *, dst: str, validate: bool, owner: str):
    from .projected import ProjectedPosition

    x, y, z, old_core_dim, ctx = _xyz_inputs(src, owner=owner)
    assert ctx.data is not None and ctx.var_name is not None
    if z is None:
        east, north = _transform_xy(x, y, src_crs=src.src_crs, dst_crs=dst, owner=owner)
        components = (east, north)
        labels = _PROJECTED_2D_LABELS
    else:
        east, north, height = _transform_xyz(x, y, z, src_crs=src.src_crs, dst_crs=dst, owner=owner)
        if not crs_has_height_axis(dst, owner=owner, field="dst"):
            height = z
        components = (east, north, height)
        labels = _PROJECTED_3D_LABELS
    core_dim = _projected_core_dim(ctx.ds)
    ds = _assemble(components, labels=labels, core_dim=core_dim, old_core_dim=old_core_dim, var_name=ctx.var_name, ctx=ctx)
    ds = _finalize_core_schema(ctx, ds, core_dim=core_dim, validate=validate, owner=owner)
    ds = set_projected_metadata(ds, crs=dst, geodetic_crs=base_geodetic_crs(dst, owner=owner, field="dst"), validate=False)
    return ProjectedPosition._from_validated(ds) if validate else ProjectedPosition._from_unvalidated(ds)


def transform_crs(value: object, *, dst: str, validate: bool = True):
    """Transform a geo position object to an explicit destination CRS.

    Parameters
    ----------
    value : object
        ``GeodeticPosition``, ``ProjectedPosition``, or ECEF ``Position`` with
        canonical geo CRS metadata.
    dst : str
        WGS84-compatible geographic, geocentric/ECEF, or projected CRS string.
    validate : bool, optional
        Whether to validate the output before returning.

    Returns
    -------
    GeodeticPosition, tal.spatial.Position, or ProjectedPosition
        Output type selected by destination CRS class.

    Raises
    ------
    ImportError
        If ``pyproj`` from the optional ``tal[geo]`` dependency group is not
        installed.
    TypeError
        If ``value`` or ``dst`` has an unsupported type.
    ValueError
        If source CRS metadata is missing, malformed, non-WGS84, or if height
        semantics would require inventing or dropping a component.

    Notes
    -----
    Public CRS inputs are strings only. CRS metadata is independent from TAL
    frame metadata; this operation does not search ``FrameGraph``.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.geo import GeodeticPosition, transform_crs
    >>> ds = xr.Dataset(
    ...     {"position": (("sample", "lla"), [[34.0, -118.0, 20.0]])},
    ...     coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    ... )
    >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
    >>> ecef = transform_crs(GeodeticPosition.from_lla(ao), dst="EPSG:4978")
    >>> list(ecef.unsafe_data["axis"].values)
    ['x', 'y', 'z']
    """
    owner = "geo.transform_crs"
    dst_text = _require_dst_string(dst, owner=owner)
    src = _source(value, owner=owner)
    dst_crs = normalize_crs_with_class(dst_text, owner=owner, field="dst")
    if dst_crs.kind == "geographic":
        return _to_geodetic(src, dst=dst_crs.text, validate=validate, owner=owner)
    if dst_crs.kind == "geocentric":
        return _to_ecef(src, dst=dst_crs.text, validate=validate, owner=owner)
    return _to_projected(src, dst=dst_crs.text, validate=validate, owner=owner)


__all__ = [
    "transform_crs",
]
