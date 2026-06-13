from __future__ import annotations

import numpy as np


def enu_basis(lat_deg, lon_deg):
    """Return east, north, and up basis vectors for geodetic degrees."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    east = (-sin_lon, cos_lon, np.zeros_like(sin_lon))
    north = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    up = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
    return east, north, up


def ecef_delta_to_enu(dx, dy, dz, lat_deg, lon_deg):
    """Project ECEF deltas into an ENU basis."""
    east, north, up = enu_basis(lat_deg, lon_deg)
    e = east[0] * dx + east[1] * dy + east[2] * dz
    n = north[0] * dx + north[1] * dy + north[2] * dz
    u = up[0] * dx + up[1] * dy + up[2] * dz
    return e, n, u


def enu_delta_to_ecef(east_value, north_value, up_value, lat_deg, lon_deg):
    """Project ENU deltas into an ECEF basis."""
    east, north, up = enu_basis(lat_deg, lon_deg)
    dx = east[0] * east_value + north[0] * north_value + up[0] * up_value
    dy = east[1] * east_value + north[1] * north_value + up[1] * up_value
    dz = east[2] * east_value + north[2] * north_value + up[2] * up_value
    return dx, dy, dz


def _normalize_degrees(angle):
    return np.mod(np.mod(angle, 360.0) + 360.0, 360.0)


def geodesic_initial_bearing(azimuth, distance, lat1, lat2):
    """Normalize forward azimuth to an initial bearing."""
    pole = (np.isclose(np.abs(lat1), 90.0) | np.isclose(np.abs(lat2), 90.0)) & (distance != 0.0)
    if np.any(pole):
        raise ValueError("geo bearing: bearing is ambiguous at a pole.")
    out = _normalize_degrees(azimuth)
    return np.where(distance == 0.0, np.nan, out)


def geodesic_final_bearing(back_azimuth, distance, lat1, lat2):
    """Normalize back azimuth to a final bearing."""
    pole = (np.isclose(np.abs(lat1), 90.0) | np.isclose(np.abs(lat2), 90.0)) & (distance != 0.0)
    if np.any(pole):
        raise ValueError("geo bearing: bearing is ambiguous at a pole.")
    out = _normalize_degrees(back_azimuth + 180.0)
    return np.where(distance == 0.0, np.nan, out)


def require_unambiguous_geodesic_interpolation(lat1, lon1, lat2, lon2, alpha):
    """Return alpha after rejecting ambiguous pole interpolation."""
    pole = np.isclose(np.abs(lat1), 90.0) | np.isclose(np.abs(lat2), 90.0)
    interior = np.isfinite(alpha) & (alpha > 0.0) & (alpha < 1.0)
    lon_delta = ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    same_point = np.isclose(lat1, lat2) & np.isclose(lon_delta, 0.0)
    if np.any(pole & interior & ~same_point):
        raise ValueError("geo interpolation: geodesic interpolation is ambiguous at a pole.")
    return alpha


def enu_distance(east_value, north_value, up_value, include_altitude):
    """Return ENU horizontal or full endpoint distance."""
    horizontal_sq = east_value * east_value + north_value * north_value
    if include_altitude:
        horizontal_sq = horizontal_sq + up_value * up_value
    return np.sqrt(horizontal_sq)


def enu_bearing(east_value, north_value):
    """Return ENU heading in degrees clockwise from north."""
    distance = np.sqrt(east_value * east_value + north_value * north_value)
    angle = np.rad2deg(np.arctan2(east_value, north_value))
    return np.where(distance == 0.0, np.nan, _normalize_degrees(angle))


__all__ = [
    "ecef_delta_to_enu",
    "enu_bearing",
    "enu_distance",
    "enu_basis",
    "enu_delta_to_ecef",
    "geodesic_final_bearing",
    "geodesic_initial_bearing",
    "require_unambiguous_geodesic_interpolation",
]
