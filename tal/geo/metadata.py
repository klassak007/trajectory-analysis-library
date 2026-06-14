from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import xarray as xr

from tal.core.schema import merge_schema

from .backends import base_geodetic_crs, normalize_crs_for_class
from .options import GeodeticOptions, LocalOrigin, coerce_geodetic_options, coerce_local_origin

_GEODETIC_KIND = "geodetic_position"
_CARTESIAN_KIND = "cartesian_geo_position"
_PROJECTED_KIND = "projected_position"
_ECEF_SYSTEM = "ecef"
_ENU_SYSTEM = "enu"

_GEODETIC_KEYS = {
    "angular_unit",
    "crs",
    "datum",
    "height_reference",
    "kind",
    "longitude_wrap",
}
_ECEF_KEYS = {
    "angular_unit",
    "cartesian_system",
    "crs",
    "datum",
    "geodetic_crs",
    "height_reference",
    "kind",
    "longitude_wrap",
}
_ENU_KEYS = {
    "angular_unit",
    "cartesian_system",
    "datum",
    "geodetic_crs",
    "height_reference",
    "kind",
    "longitude_wrap",
    "origin",
}
_PROJECTED_KEYS = {
    "crs",
    "datum",
    "geodetic_crs",
    "height_reference",
    "kind",
}
_INLINE_ORIGIN_KEYS = {"storage", "lat", "lon", "alt"}
_COORD_ORIGIN_KEYS = {"storage", "lat_coord", "lon_coord", "alt_coord"}
_ORIGIN_COORD_BASES = (
    "tal_geo_origin_lat",
    "tal_geo_origin_lon",
    "tal_geo_origin_alt",
)


def _require_dataset(ds: object, *, owner: str) -> xr.Dataset:
    if isinstance(ds, xr.Dataset):
        return ds
    raise TypeError(f"{owner}: expected xr.Dataset; got {type(ds).__name__}.")


def _read_ext(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    tal = ds.attrs.get("tal", {})
    if not isinstance(tal, Mapping):
        raise ValueError(f"{owner}: tal must be a mapping when present.")
    ext = tal.get("ext")
    if ext is None:
        return None
    if isinstance(ext, Mapping):
        return ext
    raise ValueError(f"{owner}: tal.ext must be a mapping when present.")


def _allowed_keys(block: Mapping[str, Any], *, owner: str) -> set[str]:
    kind = block.get("kind")
    if kind == _GEODETIC_KIND:
        return _GEODETIC_KEYS
    if kind == _PROJECTED_KIND:
        return _PROJECTED_KEYS
    if kind != _CARTESIAN_KIND:
        if kind == "ecef_position":
            raise ValueError(
                f"{owner}: tal.ext.geo.kind='ecef_position' was superseded by "
                "'cartesian_geo_position' in Geo G2."
            )
        raise ValueError(f"{owner}: unsupported tal.ext.geo.kind {kind!r}.")
    system = block.get("cartesian_system")
    if system == _ECEF_SYSTEM:
        return _ECEF_KEYS
    if system == _ENU_SYSTEM:
        return _ENU_KEYS
    raise ValueError(f"{owner}: unsupported tal.ext.geo.cartesian_system {system!r}.")


def _validate_keys(block: Mapping[Any, Any], *, owner: str) -> None:
    allowed = _allowed_keys(block, owner=owner)
    for key in sorted(block, key=lambda item: (type(item).__name__, repr(item))):
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"{owner}: unknown key {key!r} at tal.ext.geo; allowed keys are {sorted(allowed)!r}.")


def _validate_origin_keys(origin: Mapping[Any, Any], *, owner: str) -> None:
    storage = origin.get("storage")
    if storage == "inline":
        allowed = _INLINE_ORIGIN_KEYS
    elif storage == "coordinates":
        allowed = _COORD_ORIGIN_KEYS
    else:
        raise ValueError(f"{owner}: tal.ext.geo.origin.storage must be 'inline' or 'coordinates'.")
    for key in sorted(origin, key=lambda item: (type(item).__name__, repr(item))):
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(
                f"{owner}: unknown key {key!r} at tal.ext.geo.origin; allowed keys are {sorted(allowed)!r}."
            )


def read_geo_block(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    """Read and validate the TAL geo extension block."""
    ds = _require_dataset(ds, owner=owner)
    ext = _read_ext(ds, owner=owner)
    if ext is None or "geo" not in ext:
        return None
    block = ext["geo"]
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}: tal.ext.geo must be a mapping when present.")
    _validate_keys(block, owner=owner)
    if block.get("cartesian_system") == _ENU_SYSTEM:
        origin = block.get("origin")
        if not isinstance(origin, Mapping):
            raise ValueError(f"{owner}: ENU tal.ext.geo.origin must be a mapping.")
        _validate_origin_keys(origin, owner=owner)
    return block


def _string_field(block: Mapping[str, Any], name: str, default: str, *, owner: str) -> str:
    value = block.get(name, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{owner}: tal.ext.geo.{name} must be a non-empty string.")


def _finite_field(block: Mapping[str, Any], name: str, *, owner: str) -> float:
    value = block.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{owner}: tal.ext.geo.origin.{name} must be a finite numeric scalar.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{owner}: tal.ext.geo.origin.{name} must be finite.")
    return numeric


def _options_from_geodetic_block(
    block: Mapping[str, Any],
    *,
    owner: str,
    validate_crs: bool,
) -> GeodeticOptions:
    return coerce_geodetic_options(
        GeodeticOptions(
            datum=_string_field(block, "datum", "WGS84", owner=owner),  # type: ignore[arg-type]
            crs=_string_field(block, "crs", "EPSG:4979", owner=owner),
            angular_unit=_string_field(block, "angular_unit", "degree", owner=owner),  # type: ignore[arg-type]
            height_reference=_string_field(block, "height_reference", "ellipsoidal", owner=owner),  # type: ignore[arg-type]
            longitude_wrap=_string_field(block, "longitude_wrap", "[-180, 180)", owner=owner),  # type: ignore[arg-type]
        ),
        owner=owner,
        validate_crs=validate_crs,
    )


def options_from_ecef_provenance(block: Mapping[str, Any], *, owner: str) -> GeodeticOptions:
    """Build geodetic options from a canonical ECEF provenance block."""
    if block.get("kind") != _CARTESIAN_KIND or block.get("cartesian_system") != _ECEF_SYSTEM:
        raise ValueError(
            f"{owner}: expected tal.ext.geo.kind='cartesian_geo_position' "
            "with cartesian_system='ecef'."
        )
    crs = _string_field(block, "geodetic_crs", "EPSG:4979", owner=owner)
    ecef_crs = _string_field(block, "crs", "EPSG:4978", owner=owner)
    return coerce_geodetic_options(
        GeodeticOptions(
            datum=_string_field(block, "datum", "WGS84", owner=owner),  # type: ignore[arg-type]
            crs=crs,
            ecef_crs=ecef_crs,
            angular_unit=_string_field(block, "angular_unit", "degree", owner=owner),  # type: ignore[arg-type]
            height_reference=_string_field(block, "height_reference", "ellipsoidal", owner=owner),  # type: ignore[arg-type]
            longitude_wrap=_string_field(block, "longitude_wrap", "[-180, 180)", owner=owner),  # type: ignore[arg-type]
        ),
        owner=owner,
        validate_crs=crs != "EPSG:4979" or ecef_crs != "EPSG:4978",
    )


def options_from_enu_provenance(block: Mapping[str, Any], *, owner: str) -> GeodeticOptions:
    """Build geodetic options from canonical ENU provenance."""
    if block.get("kind") != _CARTESIAN_KIND or block.get("cartesian_system") != _ENU_SYSTEM:
        raise ValueError(
            f"{owner}: expected tal.ext.geo.kind='cartesian_geo_position' "
            "with cartesian_system='enu'."
        )
    crs = _string_field(block, "geodetic_crs", "EPSG:4979", owner=owner)
    return coerce_geodetic_options(
        GeodeticOptions(
            datum=_string_field(block, "datum", "WGS84", owner=owner),  # type: ignore[arg-type]
            crs=crs,
            angular_unit=_string_field(block, "angular_unit", "degree", owner=owner),  # type: ignore[arg-type]
            height_reference=_string_field(block, "height_reference", "ellipsoidal", owner=owner),  # type: ignore[arg-type]
            longitude_wrap=_string_field(block, "longitude_wrap", "[-180, 180)", owner=owner),  # type: ignore[arg-type]
        ),
        owner=owner,
        validate_crs=crs != "EPSG:4979",
    )


def options_from_geodetic_metadata(ds: xr.Dataset, *, owner: str) -> GeodeticOptions:
    """Build geodetic options from normalized geodetic metadata."""
    block = read_geo_block(ds, owner=owner)
    if block is None or block.get("kind") != _GEODETIC_KIND:
        raise ValueError(f"{owner}: expected tal.ext.geo.kind='geodetic_position'.")
    crs = _string_field(block, "crs", "EPSG:4979", owner=owner)
    return _options_from_geodetic_block(block, owner=owner, validate_crs=crs != "EPSG:4979")


def _geodetic_payload(opts: GeodeticOptions) -> dict[str, str]:
    return {
        "kind": _GEODETIC_KIND,
        "crs": opts.crs,
        "datum": opts.datum,
        "angular_unit": opts.angular_unit,
        "height_reference": opts.height_reference,
        "longitude_wrap": opts.longitude_wrap,
    }


def _ecef_payload(opts: GeodeticOptions) -> dict[str, str]:
    return {
        "kind": _CARTESIAN_KIND,
        "cartesian_system": _ECEF_SYSTEM,
        "crs": opts.ecef_crs,
        "geodetic_crs": opts.crs,
        "datum": opts.datum,
        "angular_unit": opts.angular_unit,
        "height_reference": opts.height_reference,
        "longitude_wrap": opts.longitude_wrap,
    }


def _enu_payload(opts: GeodeticOptions, *, origin: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": _CARTESIAN_KIND,
        "cartesian_system": _ENU_SYSTEM,
        "geodetic_crs": opts.crs,
        "datum": opts.datum,
        "angular_unit": opts.angular_unit,
        "height_reference": opts.height_reference,
        "longitude_wrap": opts.longitude_wrap,
        "origin": dict(origin),
    }


def _projected_payload(*, crs: str, geodetic_crs: str) -> dict[str, str]:
    return {
        "kind": _PROJECTED_KIND,
        "crs": crs,
        "geodetic_crs": geodetic_crs,
        "datum": "WGS84",
        "height_reference": "ellipsoidal",
    }


def normalize_geodetic_metadata(
    ds: xr.Dataset,
    *,
    opts: GeodeticOptions | None = None,
    validate: bool,
    validate_crs: bool,
    owner: str,
) -> xr.Dataset:
    """Normalize `tal.ext.geo` metadata for a GeodeticPosition."""
    block = read_geo_block(ds, owner=owner)
    if block is not None and block.get("kind") != _GEODETIC_KIND:
        raise ValueError(f"{owner}: GeodeticPosition requires tal.ext.geo.kind='geodetic_position'.")
    base = (
        _options_from_geodetic_block(block, owner=owner, validate_crs=validate_crs)
        if block is not None
        else GeodeticOptions()
    )
    effective = opts if opts is not None else base
    normalized = coerce_geodetic_options(effective, owner=owner, validate_crs=validate_crs)
    return merge_schema(ds, {"ext": {"geo": _geodetic_payload(normalized)}}, validate=validate)


def normalize_existing_projected_metadata(ds: xr.Dataset, *, validate: bool, owner: str) -> xr.Dataset:
    """Normalize an existing projected metadata block."""
    block = read_projected_geo_block(ds, owner=owner)
    return merge_schema(ds, {"ext": {"geo": dict(block)}}, validate=validate)


def normalize_projected_metadata(ds: xr.Dataset, *, crs: str, validate: bool, owner: str) -> xr.Dataset:
    """Stamp canonical projected CRS metadata."""
    normalized_crs = normalize_crs_for_class(crs, expected="projected", owner=owner, field="crs")
    geodetic_crs = base_geodetic_crs(normalized_crs, owner=owner, field="crs")
    payload = _projected_payload(crs=normalized_crs, geodetic_crs=geodetic_crs)
    return merge_schema(ds, {"ext": {"geo": payload}}, validate=validate)


def read_projected_geo_block(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any]:
    """Read a closed projected geo metadata block."""
    block = read_geo_block(ds, owner=owner)
    if block is None or block.get("kind") != _PROJECTED_KIND:
        raise ValueError(f"{owner}: expected tal.ext.geo.kind='projected_position'.")
    if _string_field(block, "datum", "", owner=owner) != "WGS84":
        raise ValueError(f"{owner}: tal.ext.geo.datum must be 'WGS84'.")
    if _string_field(block, "height_reference", "", owner=owner) != "ellipsoidal":
        raise ValueError(f"{owner}: tal.ext.geo.height_reference must be 'ellipsoidal'.")
    crs = _string_field(block, "crs", "", owner=owner)
    geodetic_crs = _string_field(block, "geodetic_crs", "", owner=owner)
    canonical_crs = normalize_crs_for_class(crs, expected="projected", owner=owner, field="crs")
    expected = base_geodetic_crs(canonical_crs, owner=owner, field="crs")
    actual = normalize_crs_for_class(geodetic_crs, expected="geographic", owner=owner, field="geodetic_crs")
    if actual != expected:
        raise ValueError(f"{owner}: tal.ext.geo.geodetic_crs conflicts with projected crs.")
    return _projected_payload(crs=canonical_crs, geodetic_crs=actual)


def read_cartesian_geo_block(ds: xr.Dataset, *, system: str, owner: str) -> Mapping[str, Any]:
    """Read canonical Cartesian geo metadata for the requested system."""
    block = read_cartesian_geo_block_if_present(ds, system=system, owner=owner)
    if block is None:
        raise ValueError(f"{owner}: tal.ext.geo metadata is required.")
    return block


def read_cartesian_geo_block_if_present(
    ds: xr.Dataset,
    *,
    system: str,
    owner: str,
) -> Mapping[str, Any] | None:
    """Read canonical Cartesian geo metadata, allowing a missing block."""
    block = read_geo_block(ds, owner=owner)
    if block is None:
        return None
    if block.get("kind") != _CARTESIAN_KIND or block.get("cartesian_system") != system:
        raise ValueError(
            f"{owner}: expected tal.ext.geo.kind='cartesian_geo_position' "
            f"with cartesian_system={system!r}."
        )
    return block


def is_cartesian_geo_system(ds: xr.Dataset, *, system: str, owner: str) -> bool:
    """Return whether geo metadata identifies a canonical Cartesian system."""
    block = read_geo_block(ds, owner=owner)
    return block is not None and block.get("kind") == _CARTESIAN_KIND and block.get("cartesian_system") == system


def inline_origin_metadata(origin: LocalOrigin, *, owner: str) -> dict[str, float | str]:
    """Serialize a scalar local origin for ENU metadata."""
    normalized = coerce_local_origin(origin, owner=owner)
    return {
        "storage": "inline",
        "lat": normalized.lat,
        "lon": normalized.lon,
        "alt": normalized.alt,
    }


def coordinate_origin_metadata(*, lat_coord: str, lon_coord: str, alt_coord: str) -> dict[str, str]:
    """Serialize coordinate-backed origin metadata."""
    return {
        "storage": "coordinates",
        "lat_coord": lat_coord,
        "lon_coord": lon_coord,
        "alt_coord": alt_coord,
    }


def allocate_origin_coordinate_names(ds: xr.Dataset) -> tuple[str, str, str]:
    """Allocate a collision-free origin-coordinate triplet."""
    occupied = set(ds.sizes) | set(ds.coords) | set(ds.data_vars)
    if not occupied.intersection(_ORIGIN_COORD_BASES):
        return _ORIGIN_COORD_BASES
    suffix = 2
    while True:
        names = tuple(f"{base}_{suffix}" for base in _ORIGIN_COORD_BASES)
        if not occupied.intersection(names):
            return names  # type: ignore[return-value]
        suffix += 1


def assign_origin_coordinates(
    ds: xr.Dataset,
    *,
    lat: xr.DataArray,
    lon: xr.DataArray,
    alt: xr.DataArray,
) -> tuple[xr.Dataset, dict[str, str]]:
    """Attach coordinate-backed origin arrays and return their provenance."""
    lat_name, lon_name, alt_name = allocate_origin_coordinate_names(ds)
    out = ds.assign_coords(
        {
            lat_name: lat.rename(lat_name),
            lon_name: lon.rename(lon_name),
            alt_name: alt.rename(alt_name),
        }
    )
    return out, coordinate_origin_metadata(lat_coord=lat_name, lon_coord=lon_name, alt_coord=alt_name)


def _validate_coordinate_name(
    ds: xr.Dataset,
    origin: Mapping[str, Any],
    key: str,
    *,
    core_dim: str,
    semantic_dims: tuple[str, ...],
    owner: str,
) -> xr.DataArray:
    name = origin.get(key)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{owner}: tal.ext.geo.origin.{key} must be a coordinate name string.")
    if name in ds.data_vars:
        raise ValueError(f"{owner}: tal.ext.geo.origin.{key} must reference a coordinate, not a data variable.")
    if name not in ds.coords:
        raise ValueError(f"{owner}: tal.ext.geo.origin.{key} references missing coordinate {name!r}.")
    coord = ds.coords[name]
    if not np.issubdtype(np.dtype(coord.dtype), np.number):
        raise TypeError(f"{owner}: origin coordinate {name!r} must be numeric; got {coord.dtype!r}.")
    if core_dim in coord.dims:
        raise ValueError(f"{owner}: origin coordinate {name!r} must not include output core dim {core_dim!r}.")
    illegal = tuple(dim for dim in coord.dims if dim not in semantic_dims)
    if illegal:
        raise ValueError(f"{owner}: origin coordinate {name!r} has non-semantic dims {illegal!r}.")
    return coord


def read_enu_origin_metadata(
    ds: xr.Dataset,
    *,
    core_dim: str,
    semantic_dims: tuple[str, ...],
    owner: str,
) -> Mapping[str, Any]:
    """Read and validate ENU origin provenance from a dataset."""
    block = read_cartesian_geo_block(ds, system=_ENU_SYSTEM, owner=owner)
    origin = block.get("origin")
    if not isinstance(origin, Mapping):
        raise ValueError(f"{owner}: tal.ext.geo.origin must be a mapping.")
    _validate_origin_keys(origin, owner=owner)
    if origin.get("storage") == "inline":
        _ = _finite_field(origin, "lat", owner=owner)
        _ = _finite_field(origin, "lon", owner=owner)
        _ = _finite_field(origin, "alt", owner=owner)
        return origin
    _ = _validate_coordinate_name(ds, origin, "lat_coord", core_dim=core_dim, semantic_dims=semantic_dims, owner=owner)
    _ = _validate_coordinate_name(ds, origin, "lon_coord", core_dim=core_dim, semantic_dims=semantic_dims, owner=owner)
    _ = _validate_coordinate_name(ds, origin, "alt_coord", core_dim=core_dim, semantic_dims=semantic_dims, owner=owner)
    return origin


def set_ecef_metadata(
    ds: xr.Dataset,
    *,
    opts: GeodeticOptions,
    validate: bool,
) -> xr.Dataset:
    """Stamp canonical ECEF geo provenance metadata."""
    cleared = merge_schema(ds, {"ext": {"geo": None}}, validate=False)
    return merge_schema(cleared, {"ext": {"geo": _ecef_payload(opts)}}, validate=validate)


def set_projected_metadata(
    ds: xr.Dataset,
    *,
    crs: str,
    geodetic_crs: str,
    validate: bool,
) -> xr.Dataset:
    """Stamp canonical projected geo metadata."""
    cleared = merge_schema(ds, {"ext": {"geo": None}}, validate=False)
    payload = _projected_payload(crs=crs, geodetic_crs=geodetic_crs)
    return merge_schema(cleared, {"ext": {"geo": payload}}, validate=validate)


def set_enu_metadata(
    ds: xr.Dataset,
    *,
    opts: GeodeticOptions,
    origin: Mapping[str, Any],
    validate: bool,
) -> xr.Dataset:
    """Stamp canonical ENU geo provenance metadata."""
    cleared = merge_schema(ds, {"ext": {"geo": None}}, validate=False)
    return merge_schema(cleared, {"ext": {"geo": _enu_payload(opts, origin=origin)}}, validate=validate)


__all__ = [
    "allocate_origin_coordinate_names",
    "assign_origin_coordinates",
    "coordinate_origin_metadata",
    "inline_origin_metadata",
    "is_cartesian_geo_system",
    "normalize_geodetic_metadata",
    "normalize_existing_projected_metadata",
    "normalize_projected_metadata",
    "options_from_ecef_provenance",
    "options_from_enu_provenance",
    "options_from_geodetic_metadata",
    "read_cartesian_geo_block",
    "read_cartesian_geo_block_if_present",
    "read_enu_origin_metadata",
    "read_geo_block",
    "read_projected_geo_block",
    "set_ecef_metadata",
    "set_enu_metadata",
    "set_projected_metadata",
]
