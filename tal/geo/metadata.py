from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from tal.core.schema import merge_schema

from .options import GeodeticOptions, coerce_geodetic_options

_GEO_KEYS = {
    "angular_unit",
    "crs",
    "datum",
    "geodetic_crs",
    "height_reference",
    "kind",
    "longitude_wrap",
}
_GEODETIC_KEYS = _GEO_KEYS - {"geodetic_crs"}
_ECEF_KEYS = _GEO_KEYS


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


def _validate_keys(block: Mapping[Any, Any], *, kind: str, owner: str) -> None:
    allowed = _GEODETIC_KEYS if kind == "geodetic_position" else _ECEF_KEYS
    for key in sorted(block, key=lambda item: (type(item).__name__, repr(item))):
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"{owner}: unknown key {key!r} at tal.ext.geo; allowed keys are {sorted(allowed)!r}.")


def read_geo_block(ds: xr.Dataset, *, owner: str) -> Mapping[str, Any] | None:
    """Read and validate the TAL geo extension block."""
    ds = _require_dataset(ds, owner=owner)
    ext = _read_ext(ds, owner=owner)
    if ext is None or "geo" not in ext:
        return None
    block = ext["geo"]
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}: tal.ext.geo must be a mapping when present.")
    kind = block.get("kind")
    if kind not in {"geodetic_position", "ecef_position"}:
        raise ValueError(f"{owner}: unsupported tal.ext.geo.kind {kind!r}.")
    _validate_keys(block, kind=str(kind), owner=owner)
    return block


def _string_field(block: Mapping[str, Any], name: str, default: str, *, owner: str) -> str:
    value = block.get(name, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{owner}: tal.ext.geo.{name} must be a non-empty string.")


def _options_from_geodetic_block(block: Mapping[str, Any], *, owner: str) -> GeodeticOptions:
    return coerce_geodetic_options(
        GeodeticOptions(
            datum=_string_field(block, "datum", "WGS84", owner=owner),  # type: ignore[arg-type]
            crs=_string_field(block, "crs", "EPSG:4979", owner=owner),
            angular_unit=_string_field(block, "angular_unit", "degree", owner=owner),  # type: ignore[arg-type]
            height_reference=_string_field(block, "height_reference", "ellipsoidal", owner=owner),  # type: ignore[arg-type]
            longitude_wrap=_string_field(block, "longitude_wrap", "[-180, 180)", owner=owner),  # type: ignore[arg-type]
        ),
        owner=owner,
        validate_crs=False,
    )


def options_from_ecef_provenance(block: Mapping[str, Any], *, owner: str) -> GeodeticOptions:
    """Build geodetic options from an ECEF provenance block."""
    if block.get("kind") != "ecef_position":
        raise ValueError(f"{owner}: expected tal.ext.geo.kind='ecef_position'.")
    return coerce_geodetic_options(
        GeodeticOptions(
            datum=_string_field(block, "datum", "WGS84", owner=owner),  # type: ignore[arg-type]
            crs=_string_field(block, "geodetic_crs", "EPSG:4979", owner=owner),
            ecef_crs=_string_field(block, "crs", "EPSG:4978", owner=owner),
            angular_unit=_string_field(block, "angular_unit", "degree", owner=owner),  # type: ignore[arg-type]
            height_reference=_string_field(block, "height_reference", "ellipsoidal", owner=owner),  # type: ignore[arg-type]
            longitude_wrap=_string_field(block, "longitude_wrap", "[-180, 180)", owner=owner),  # type: ignore[arg-type]
        ),
        owner=owner,
        validate_crs=False,
    )


def _geodetic_payload(opts: GeodeticOptions) -> dict[str, str]:
    return {
        "kind": "geodetic_position",
        "crs": opts.crs,
        "datum": opts.datum,
        "angular_unit": opts.angular_unit,
        "height_reference": opts.height_reference,
        "longitude_wrap": opts.longitude_wrap,
    }


def _ecef_payload(opts: GeodeticOptions) -> dict[str, str]:
    return {
        "kind": "ecef_position",
        "crs": opts.ecef_crs,
        "geodetic_crs": opts.crs,
        "datum": opts.datum,
        "angular_unit": opts.angular_unit,
        "height_reference": opts.height_reference,
        "longitude_wrap": opts.longitude_wrap,
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
    if block is not None and block.get("kind") != "geodetic_position":
        raise ValueError(f"{owner}: GeodeticPosition requires tal.ext.geo.kind='geodetic_position'.")
    base = _options_from_geodetic_block(block, owner=owner) if block is not None else GeodeticOptions()
    effective = opts if opts is not None else base
    normalized = coerce_geodetic_options(effective, owner=owner, validate_crs=validate_crs)
    return merge_schema(ds, {"ext": {"geo": _geodetic_payload(normalized)}}, validate=validate)


def set_ecef_metadata(
    ds: xr.Dataset,
    *,
    opts: GeodeticOptions,
    validate: bool,
) -> xr.Dataset:
    """Stamp ECEF geo provenance metadata."""
    return merge_schema(ds, {"ext": {"geo": _ecef_payload(opts)}}, validate=validate)


__all__ = [
    "normalize_geodetic_metadata",
    "options_from_ecef_provenance",
    "read_geo_block",
    "set_ecef_metadata",
]
