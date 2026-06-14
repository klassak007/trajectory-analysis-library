from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any, Literal


_PYPROJ_IMPORT_HINT = "Install geo support with 'tal[geo]' to use TAL geodetic conversions."
_WGS84_SEMI_MAJOR_M = 6378137.0
_WGS84_INVERSE_FLATTENING = 298.257223563
_SEMI_MAJOR_TOL = 1e-6
_INVERSE_FLATTENING_TOL = 1e-9

CRSClass = Literal["geographic", "geocentric", "projected"]


@dataclass(frozen=True)
class NormalizedCRS:
    """Canonical CRS text plus supported CRS class."""

    text: str
    kind: CRSClass


def _require_pyproj(owner: str) -> Any:
    try:
        return import_module("pyproj")
    except ImportError as exc:
        raise ImportError(f"{owner}: pyproj is required. {_PYPROJ_IMPORT_HINT}") from exc


def normalize_supported_crs(value: str, *, expected: str, owner: str) -> str:
    """Normalize one supported CRS through pyproj."""
    pyproj = _require_pyproj(owner)
    try:
        crs = pyproj.CRS.from_user_input(value)
        expected_crs = pyproj.CRS.from_user_input(expected)
    except Exception as exc:
        raise ValueError(f"{owner}: unsupported CRS {value!r}; expected {expected!r}.") from exc
    if crs != expected_crs:
        raise ValueError(f"{owner}: unsupported CRS {value!r}; expected {expected!r}.")
    return expected


def _require_crs_string(value: object, *, field: str, owner: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise TypeError(f"{owner}: {field} must be a non-empty CRS string.")


def _parse_crs(value: str, *, owner: str) -> Any:
    pyproj = _require_pyproj(owner)
    try:
        return pyproj.CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError(f"{owner}: unsupported CRS {value!r}.") from exc


def _canonical_crs(crs: Any) -> str:
    authority = crs.to_authority()
    if authority is not None:
        auth, code = authority
        return f"{auth}:{code}"
    return crs.to_wkt(version="WKT2_2019", pretty=False)


def _classify_crs(crs: Any, *, owner: str) -> CRSClass:
    if crs.is_geographic:
        return "geographic"
    if crs.is_geocentric:
        return "geocentric"
    if crs.is_projected:
        return "projected"
    raise ValueError(f"{owner}: unsupported CRS class for {crs.to_string()!r}.")


def _base_geodetic_crs(crs: Any) -> Any:
    return crs.geodetic_crs if crs.is_projected else crs


def _require_wgs84(base: Any, *, owner: str) -> None:
    names = " ".join(
        str(getattr(item, "name", ""))
        for item in (getattr(base, "datum", None), getattr(base, "datum_ensemble", None))
        if item is not None
    ).upper()
    if not (("WGS" in names and "84" in names) or "WORLD GEODETIC SYSTEM 1984" in names):
        raise ValueError(f"{owner}: CRS datum must identify WGS 84.")
    ellipsoid = base.ellipsoid
    semi_major = float(ellipsoid.semi_major_metre)
    inverse_flattening = float(ellipsoid.inverse_flattening)
    if abs(semi_major - _WGS84_SEMI_MAJOR_M) > _SEMI_MAJOR_TOL:
        raise ValueError(f"{owner}: CRS ellipsoid semi-major axis is not WGS84-compatible.")
    if abs(inverse_flattening - _WGS84_INVERSE_FLATTENING) > _INVERSE_FLATTENING_TOL:
        raise ValueError(f"{owner}: CRS ellipsoid inverse flattening is not WGS84-compatible.")


def normalize_crs_with_class(value: object, *, owner: str, field: str = "crs") -> NormalizedCRS:
    """Normalize a public CRS string and return its supported CRS class."""
    text = _require_crs_string(value, field=field, owner=owner)
    crs = _parse_crs(text, owner=owner)
    kind = _classify_crs(crs, owner=owner)
    _require_wgs84(_base_geodetic_crs(crs), owner=owner)
    return NormalizedCRS(text=_canonical_crs(crs), kind=kind)


def normalize_crs_for_class(value: object, *, expected: CRSClass, owner: str, field: str = "crs") -> str:
    """Normalize a CRS string and require a specific supported CRS class."""
    normalized = normalize_crs_with_class(value, owner=owner, field=field)
    if normalized.kind != expected:
        raise ValueError(f"{owner}: {field} must be a WGS84-compatible {expected} CRS; got {normalized.kind}.")
    return normalized.text


def base_geodetic_crs(value: object, *, owner: str, field: str = "crs") -> str:
    """Return canonical base/geodetic CRS text for a supported CRS string."""
    text = _require_crs_string(value, field=field, owner=owner)
    crs = _parse_crs(text, owner=owner)
    _ = _classify_crs(crs, owner=owner)
    base = _base_geodetic_crs(crs)
    _require_wgs84(base, owner=owner)
    return _canonical_crs(base)


def crs_has_height_axis(value: object, *, owner: str, field: str = "crs") -> bool:
    """Return whether the CRS exposes three or more axes."""
    text = _require_crs_string(value, field=field, owner=owner)
    return len(_parse_crs(text, owner=owner).axis_info) >= 3


@lru_cache(maxsize=4)
def _transformer(source_crs: str, target_crs: str, *, owner: str) -> Any:
    pyproj = _require_pyproj(owner)
    return pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)


@lru_cache(maxsize=4)
def _geod(crs: str, *, owner: str) -> Any:
    pyproj = _require_pyproj(owner)
    try:
        geod = pyproj.CRS.from_user_input(crs).get_geod()
    except Exception as exc:
        raise ValueError(f"{owner}: unsupported geodetic CRS {crs!r} for geodesic calculations.") from exc
    if geod is None:
        raise ValueError(f"{owner}: CRS {crs!r} does not provide a geodesic ellipsoid.")
    return geod


def transform_lla_to_ecef(
    lat: Any,
    lon: Any,
    alt: Any,
    *,
    crs: str,
    ecef_crs: str,
    owner: str,
) -> tuple[Any, Any, Any]:
    """Convert latitude, longitude, altitude arrays to ECEF arrays."""
    transformer = _transformer(crs, ecef_crs, owner=owner)
    x, y, z = transformer.transform(lon, lat, alt)
    return x, y, z


def transform_ecef_to_lla(
    x: Any,
    y: Any,
    z: Any,
    *,
    crs: str,
    ecef_crs: str,
    owner: str,
) -> tuple[Any, Any, Any]:
    """Convert ECEF arrays to latitude, longitude, altitude arrays."""
    transformer = _transformer(ecef_crs, crs, owner=owner)
    lon, lat, alt = transformer.transform(x, y, z)
    return lat, lon, alt


def transform_crs_xyz(
    x: Any,
    y: Any,
    z: Any,
    *,
    src_crs: str,
    dst_crs: str,
    owner: str,
) -> tuple[Any, Any, Any]:
    """Transform three CRS components with pyproj always-xy semantics."""
    return _transformer(src_crs, dst_crs, owner=owner).transform(x, y, z)


def transform_crs_xy(
    x: Any,
    y: Any,
    *,
    src_crs: str,
    dst_crs: str,
    owner: str,
) -> tuple[Any, Any]:
    """Transform two CRS components with pyproj always-xy semantics."""
    return _transformer(src_crs, dst_crs, owner=owner).transform(x, y)


def geod_inverse(
    lat1: Any,
    lon1: Any,
    lat2: Any,
    lon2: Any,
    *,
    crs: str,
    owner: str,
) -> tuple[Any, Any, Any]:
    """Return forward azimuth, back azimuth, and distance on the CRS ellipsoid."""
    az12, az21, distance = _geod(crs, owner=owner).inv(lon1, lat1, lon2, lat2)
    return az12, az21, distance


def geod_interpolate(
    lat1: Any,
    lon1: Any,
    lat2: Any,
    lon2: Any,
    alpha: Any,
    *,
    crs: str,
    owner: str,
) -> tuple[Any, Any]:
    """Return latitude and longitude at fractional distance between endpoints."""
    geod = _geod(crs, owner=owner)
    az12, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    lon, lat, _ = geod.fwd(lon1, lat1, az12, distance * alpha)
    return lat, lon


__all__ = [
    "CRSClass",
    "NormalizedCRS",
    "base_geodetic_crs",
    "crs_has_height_axis",
    "geod_interpolate",
    "geod_inverse",
    "normalize_crs_for_class",
    "normalize_crs_with_class",
    "normalize_supported_crs",
    "transform_crs_xy",
    "transform_crs_xyz",
    "transform_ecef_to_lla",
    "transform_lla_to_ecef",
]
