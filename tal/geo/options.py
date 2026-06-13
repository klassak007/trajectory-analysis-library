from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .backends import normalize_supported_crs

GeodeticDatum = Literal["WGS84"]

_ANGULAR_UNIT = "degree"
_HEIGHT_REFERENCE = "ellipsoidal"
_LONGITUDE_WRAPS = frozenset({"[-180, 180)", "[0, 360)"})
_GEODETIC_CRS = "EPSG:4979"
_ECEF_CRS = "EPSG:4978"


@dataclass(frozen=True)
class GeodeticOptions:
    """Options controlling G1 geodetic CRS metadata and conversion policy.

    Parameters
    ----------
    datum : {'WGS84'}, optional
        Supported geodetic datum for G1.
    crs : str, optional
        Geographic 3D CRS. G1 supports ``"EPSG:4979"``.
    ecef_crs : str, optional
        ECEF CRS. G1 supports ``"EPSG:4978"``.
    angular_unit : {'degree'}, optional
        Angular unit for latitude and longitude.
    height_reference : {'ellipsoidal'}, optional
        Altitude height reference.
    longitude_wrap : {'[-180, 180)', '[0, 360)'}, optional
        Longitude wrap convention recorded in metadata.
    ecef_frame : str | None, optional
        Frame id for the ECEF coordinate basis.
    strict_frame : bool, optional
        Whether missing or incompatible frame metadata fails closed.

    Notes
    -----
    CRS validation is performed by operation boundaries through pyproj-backed
    backend helpers; constructing this dataclass does not import pyproj.

    Examples
    --------
    >>> from tal.geo import GeodeticOptions
    >>> GeodeticOptions().datum
    'WGS84'
    """

    datum: GeodeticDatum = "WGS84"
    crs: str = _GEODETIC_CRS
    ecef_crs: str = _ECEF_CRS
    angular_unit: Literal["degree"] = "degree"
    height_reference: Literal["ellipsoidal"] = "ellipsoidal"
    longitude_wrap: Literal["[-180, 180)", "[0, 360)"] = "[-180, 180)"
    ecef_frame: str | None = "earth_ecef"
    strict_frame: bool = False


def _require_string(value: object, *, field: str, owner: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{owner}: {field} must be a non-empty string.")


def _validate_non_crs_fields(opts: GeodeticOptions, *, owner: str) -> None:
    if opts.datum != "WGS84":
        raise ValueError(f"{owner}: unsupported datum {opts.datum!r}; expected 'WGS84'.")
    if opts.angular_unit != _ANGULAR_UNIT:
        raise ValueError(f"{owner}: unsupported angular_unit {opts.angular_unit!r}.")
    if opts.height_reference != _HEIGHT_REFERENCE:
        raise ValueError(f"{owner}: unsupported height_reference {opts.height_reference!r}.")
    if opts.longitude_wrap not in _LONGITUDE_WRAPS:
        raise ValueError(f"{owner}: unsupported longitude_wrap {opts.longitude_wrap!r}.")
    if opts.ecef_frame is not None:
        _require_string(opts.ecef_frame, field="ecef_frame", owner=owner)
    if not isinstance(opts.strict_frame, bool):
        raise TypeError(f"{owner}: strict_frame must be bool.")
    if opts.strict_frame and opts.ecef_frame is None:
        raise ValueError(f"{owner}: strict_frame=True requires ecef_frame.")


def coerce_geodetic_options(
    opts: GeodeticOptions | None,
    *,
    owner: str,
    validate_crs: bool,
) -> GeodeticOptions:
    """Coerce and validate geodetic options at a public boundary."""
    if opts is None:
        opts = GeodeticOptions()
    if not isinstance(opts, GeodeticOptions):
        raise TypeError(f"{owner}: opts must be GeodeticOptions or None.")
    _validate_non_crs_fields(opts, owner=owner)
    crs = _require_string(opts.crs, field="crs", owner=owner)
    ecef_crs = _require_string(opts.ecef_crs, field="ecef_crs", owner=owner)
    if not validate_crs:
        if crs != _GEODETIC_CRS:
            raise ValueError(f"{owner}: unsupported CRS {crs!r}; expected {_GEODETIC_CRS!r}.")
        if ecef_crs != _ECEF_CRS:
            raise ValueError(f"{owner}: unsupported CRS {ecef_crs!r}; expected {_ECEF_CRS!r}.")
        return replace(opts, crs=crs, ecef_crs=ecef_crs)
    normalized_crs = normalize_supported_crs(crs, expected=_GEODETIC_CRS, owner=owner)
    normalized_ecef_crs = normalize_supported_crs(ecef_crs, expected=_ECEF_CRS, owner=owner)
    return replace(opts, crs=normalized_crs, ecef_crs=normalized_ecef_crs)


__all__ = [
    "GeodeticDatum",
    "GeodeticOptions",
    "coerce_geodetic_options",
]
