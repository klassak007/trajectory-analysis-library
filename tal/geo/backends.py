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


__all__ = [
    "normalize_supported_crs",
    "transform_ecef_to_lla",
    "transform_lla_to_ecef",
]
