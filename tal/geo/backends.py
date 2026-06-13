from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any


_PYPROJ_IMPORT_HINT = "Install geo support with 'tal[geo]' to use TAL geodetic conversions."


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
    "geod_interpolate",
    "geod_inverse",
    "normalize_supported_crs",
    "transform_ecef_to_lla",
    "transform_lla_to_ecef",
]
