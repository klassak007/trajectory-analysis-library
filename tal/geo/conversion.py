from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema import merge_schema
from tal.core.orchestration.context import DatasetContext, DatasetContextOptions, resolve_dataset_context
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from tal.spatial import Position
from tal.spatial.conversion.finalize import allocate_free_dim_name, dataset_dim_names
from tal.spatial.metadata import set_expressed_in, set_position_rep
from tal.utils.frame_schema import get_frames, set_frames

from .backends import transform_ecef_to_lla, transform_lla_to_ecef
from .metadata import (
    normalize_geodetic_metadata,
    options_from_ecef_provenance,
    read_geo_block,
    set_ecef_metadata,
)
from .options import GeodeticOptions, coerce_geodetic_options

if TYPE_CHECKING:
    from .geodetic import GeodeticPosition

_LLA_LABELS = ("lat", "lon", "alt")
_XYZ_LABELS = ("x", "y", "z")
_CTX_OPTIONS = DatasetContextOptions(
    require_roles=True,
    select_numeric_var=True,
    require_single_numeric_var=True,
    allowed_core_arity=(1,),
    require_semantic_dims_in_var=True,
)


def _context(value: AnalysisObject, *, owner: str) -> DatasetContext:
    return resolve_dataset_context(value, owner=owner, options=_CTX_OPTIONS)


def _component(data: xr.DataArray, *, dim: str, label: str) -> xr.DataArray:
    return data.sel({dim: label}, drop=True)


def _convert_lla_components(
    data: xr.DataArray,
    *,
    core_dim: str,
    opts: GeodeticOptions,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    lat = _component(data, dim=core_dim, label="lat")
    lon = _component(data, dim=core_dim, label="lon")
    alt = _component(data, dim=core_dim, label="alt")
    return xr.apply_ufunc(
        transform_lla_to_ecef,
        lat,
        lon,
        alt,
        input_core_dims=((), (), ()),
        output_core_dims=((), (), ()),
        output_dtypes=(np.float64, np.float64, np.float64),
        kwargs={"crs": opts.crs, "ecef_crs": opts.ecef_crs, "owner": owner},
        dask="parallelized",
    )


def _convert_ecef_components(
    data: xr.DataArray,
    *,
    core_dim: str,
    opts: GeodeticOptions,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    x = _component(data, dim=core_dim, label="x")
    y = _component(data, dim=core_dim, label="y")
    z = _component(data, dim=core_dim, label="z")
    return xr.apply_ufunc(
        transform_ecef_to_lla,
        x,
        y,
        z,
        input_core_dims=((), (), ()),
        output_core_dims=((), (), ()),
        output_dtypes=(np.float64, np.float64, np.float64),
        kwargs={"crs": opts.crs, "ecef_crs": opts.ecef_crs, "owner": owner},
        dask="parallelized",
    )


def _assemble(
    components: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    *,
    labels: tuple[str, str, str],
    core_dim: str,
    target_dims: tuple[str, ...],
    var_name: str,
    attrs_source: xr.Dataset,
) -> xr.Dataset:
    dim = xr.IndexVariable(core_dim, list(labels))
    arr = xr.concat(list(components), dim=dim).transpose(*target_dims)
    arr.name = var_name
    out = arr.to_dataset(name=var_name)
    out.attrs = dict(attrs_source.attrs)
    return out


def _target_dims(data: xr.DataArray, *, old_core_dim: str, new_core_dim: str) -> tuple[str, ...]:
    return tuple(new_core_dim if dim == old_core_dim else dim for dim in data.dims)


def _schema_spec(ctx: DatasetContext, *, core_dim: str) -> CoreSchemaFinalizeSpec:
    return CoreSchemaFinalizeSpec(
        sequence_dim=ctx.sequence_dim,
        batch_dims=ctx.batch_dims,
        core_dims=(core_dim,),
        param_name=ctx.param_coord,
        size_name=ctx.sequence_size_coord,
    )


def _finalize_core_schema(ctx: DatasetContext, ds: xr.Dataset, *, core_dim: str, validate: bool, owner: str) -> xr.Dataset:
    source = AnalysisObject._from_validated(ctx.ds)
    finalized = finalize_with_schema(
        source,
        ds,
        spec=_schema_spec(ctx, core_dim=core_dim),
        validate=validate,
        owner=owner,
        optional_sources=(ctx.data,) if ctx.data is not None else (),
    )
    return finalized.unsafe_data


def _resolved_frame_parent(ds: xr.Dataset, *, opts: GeodeticOptions, owner: str) -> str | None:
    parent, _ = get_frames(ds)
    if opts.ecef_frame is None:
        return parent
    if parent == opts.ecef_frame:
        return parent
    if opts.strict_frame:
        raise ValueError(f"{owner}: source frame parent {parent!r} is not compatible with {opts.ecef_frame!r}.")
    return opts.ecef_frame


def _apply_frame_policy(ds: xr.Dataset, *, source_ds: xr.Dataset, opts: GeodeticOptions, owner: str) -> xr.Dataset:
    _, child = get_frames(source_ds)
    next_parent = _resolved_frame_parent(source_ds, opts=opts, owner=owner)
    out = set_frames(ds, parent=next_parent, child=child, validate=False)
    if next_parent is not None:
        out = set_expressed_in(out, expressed_in=next_parent, validate=False, owner=owner)
    return out


def _ecef_core_dim(source_ds: xr.Dataset) -> str:
    return allocate_free_dim_name(
        existing_dims=dataset_dim_names(source_ds),
        candidates=("axis", "xyz"),
        base="axis",
        owner="geo.to_ecef",
        what="ECEF axis",
    )


def _lla_core_dim(source_ds: xr.Dataset) -> str:
    return allocate_free_dim_name(
        existing_dims=dataset_dim_names(source_ds),
        candidates=("lla", "geodetic"),
        base="lla",
        owner="geo.from_ecef",
        what="LLA axis",
    )


def from_lla(value: object, *, opts: GeodeticOptions | None = None, validate: bool = True) -> "GeodeticPosition":
    """Normalize an AO-like LLA payload into a GeodeticPosition."""
    from .geodetic import GeodeticPosition

    owner = "geo.GeodeticPosition.from_lla"
    source = coerce_analysis_object_input(value, owner=owner)
    ds = normalize_geodetic_metadata(
        source.unsafe_data,
        opts=opts,
        validate=False,
        validate_crs=opts is not None,
        owner=owner,
    )
    if validate:
        return GeodeticPosition._from_validated(ds)
    return GeodeticPosition._from_unvalidated(ds)


def to_ecef(position: "GeodeticPosition", *, opts: GeodeticOptions | None = None, validate: bool = True) -> Position:
    """Convert a GeodeticPosition to Cartesian ECEF Position."""
    owner = "geo.GeodeticPosition.to_ecef"
    normalized_opts = coerce_geodetic_options(opts, owner=owner, validate_crs=True)
    ctx = _context(position, owner=owner)
    assert ctx.data is not None and ctx.var_name is not None
    core_dim = ctx.core_dims[0]
    out_core_dim = _ecef_core_dim(ctx.ds)
    components = _convert_lla_components(ctx.data, core_dim=core_dim, opts=normalized_opts, owner=owner)
    ds = _assemble(
        components,
        labels=_XYZ_LABELS,
        core_dim=out_core_dim,
        target_dims=_target_dims(ctx.data, old_core_dim=core_dim, new_core_dim=out_core_dim),
        var_name=ctx.var_name,
        attrs_source=ctx.ds,
    )
    ds = _finalize_core_schema(ctx, ds, core_dim=out_core_dim, validate=validate, owner=owner)
    ds = set_position_rep(ds, rep="cart", validate=False, owner=owner)
    ds = set_ecef_metadata(ds, opts=normalized_opts, validate=False)
    ds = _apply_frame_policy(ds, source_ds=ctx.ds, opts=normalized_opts, owner=owner)
    if validate:
        return Position._from_validated(ds)
    return Position._from_unvalidated(ds)


def _options_for_ecef_source(position: Position, opts: GeodeticOptions | None, *, owner: str) -> GeodeticOptions:
    if opts is not None:
        return coerce_geodetic_options(opts, owner=owner, validate_crs=True)
    block = read_geo_block(position.unsafe_data, owner=owner)
    if block is None or block.get("kind") != "ecef_position":
        return coerce_geodetic_options(None, owner=owner, validate_crs=True)
    provenance = options_from_ecef_provenance(block, owner=owner)
    parent, _ = get_frames(position.unsafe_data)
    return coerce_geodetic_options(
        replace(provenance, ecef_frame=parent),
        owner=owner,
        validate_crs=True,
    )


def from_ecef(value: object, *, opts: GeodeticOptions | None = None, validate: bool = True) -> "GeodeticPosition":
    """Convert Cartesian ECEF position input to geodetic LLA coordinates.

    Parameters
    ----------
    value : object
        ``Position`` or AO-like value that can be coerced to cartesian
        ``tal.spatial.Position``.
    opts : GeodeticOptions | None, optional
        Geodetic conversion options controlling ``crs``, ``ecef_crs``,
        ``ecef_frame``, and ``strict_frame``. ``None`` uses ECEF provenance
        when present, otherwise default WGS84 options.
    validate : bool, optional
        Whether to validate the output before returning.

    Returns
    -------
    GeodeticPosition
        Converted geodetic position.

    Raises
    ------
    ImportError
        If ``pyproj`` from the optional ``tal[geo]`` dependency group is not
        installed.
    TypeError
        If ``value`` cannot be coerced to ``Position`` or ``opts`` is not
        ``GeodeticOptions`` or ``None``.
    ValueError
        If the ECEF payload has malformed TAL roles or core labels, option
        values or CRS identifiers are unsupported, or ``strict_frame=True``
        rejects incompatible frame metadata.

    Notes
    -----
    Conversion requires the optional ``tal[geo]`` dependency group. ECEF inputs
    use ``x``, ``y``, ``z`` core labels, while outputs use the public LLA label
    order ``lat``, ``lon``, ``alt``. This is an explicit CRS conversion, not a
    spatial frame-graph transform.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.geo import GeodeticOptions, from_ecef
    >>> from tal.spatial import Position
    >>> ds = xr.Dataset(
    ...     {"position": (("sample", "axis"), [[6378137.0, 0.0, 0.0]])},
    ...     coords={"sample": [0], "axis": ["x", "y", "z"]},
    ... )
    >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> opts = GeodeticOptions(longitude_wrap="[0, 360)")
    >>> lla = from_ecef(Position(ao), opts=opts)
    >>> list(lla.unsafe_data["lla"].values)
    ['lat', 'lon', 'alt']
    """
    from .geodetic import GeodeticPosition

    owner = "geo.from_ecef"
    position = value if isinstance(value, Position) else Position(value)
    normalized_opts = _options_for_ecef_source(position, opts, owner=owner)
    ctx = _context(position, owner=owner)
    assert ctx.data is not None and ctx.var_name is not None
    core_dim = ctx.core_dims[0]
    out_core_dim = _lla_core_dim(ctx.ds)
    components = _convert_ecef_components(ctx.data, core_dim=core_dim, opts=normalized_opts, owner=owner)
    ds = _assemble(
        components,
        labels=_LLA_LABELS,
        core_dim=out_core_dim,
        target_dims=_target_dims(ctx.data, old_core_dim=core_dim, new_core_dim=out_core_dim),
        var_name=ctx.var_name,
        attrs_source=ctx.ds,
    )
    ds = _finalize_core_schema(ctx, ds, core_dim=out_core_dim, validate=validate, owner=owner)
    ds = merge_schema(ds, {"ext": {"geo": None}}, validate=False)
    ds = normalize_geodetic_metadata(
        ds,
        opts=normalized_opts,
        validate=False,
        validate_crs=False,
        owner=owner,
    )
    ds = _apply_frame_policy(ds, source_ds=ctx.ds, opts=normalized_opts, owner=owner)
    if validate:
        return GeodeticPosition._from_validated(ds)
    return GeodeticPosition._from_unvalidated(ds)


__all__ = [
    "from_ecef",
    "from_lla",
    "to_ecef",
]
