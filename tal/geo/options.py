from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Literal

from tal.core.param_ops.guards import validate_query_dim_name

from .backends import normalize_crs_for_class, normalize_supported_crs

if TYPE_CHECKING:
    from .geodetic import GeodeticPosition

GeodeticDatum = Literal["WGS84"]

_ANGULAR_UNIT = "degree"
_HEIGHT_REFERENCE = "ellipsoidal"
_LONGITUDE_WRAPS = frozenset({"[-180, 180)", "[0, 360)"})
_INTERPOLATION_LONGITUDE_WRAPS = frozenset({"shortest", "preserve", "[-180, 180)", "[0, 360)"})
_GEODETIC_CRS = "EPSG:4979"
_ECEF_CRS = "EPSG:4978"
_GEODESIC_METHODS = frozenset({"geodesic", "local_enu"})
_INTERPOLATION_METHODS = frozenset({"nearest", "geodesic_linear", "ecef_linear", "local_enu_linear"})
_DUPLICATE_POLICIES = frozenset({"invalid", "left", "right", "raise"})


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


@dataclass(frozen=True)
class LocalOrigin:
    """Scalar geodetic origin for local ENU conversion.

    Parameters
    ----------
    lat : float
        Origin latitude in degrees.
    lon : float
        Origin longitude in degrees.
    alt : float, optional
        Origin ellipsoidal altitude.
    opts : GeodeticOptions | None, optional
        Geodetic CRS options for interpreting the scalar origin, including
        ``crs``, ``ecef_crs``, and ``longitude_wrap``.

    Returns
    -------
    LocalOrigin
        Frozen scalar origin option object.

    Raises
    ------
    TypeError
        If public conversion boundaries receive an unsupported ``opts`` type.
    ValueError
        If public conversion boundaries receive unsupported geodetic option
        fields or non-finite scalar values.

    Notes
    -----
    ``LocalOrigin`` is a typed scalar option object. Raw latitude/longitude
    tuples are not accepted at public ENU conversion boundaries.

    Examples
    --------
    >>> from tal.geo import GeodeticOptions, LocalOrigin
    >>> opts = GeodeticOptions(longitude_wrap="[0, 360)")
    >>> origin = LocalOrigin(45.0, -75.0, 100.0, opts=opts)
    >>> origin.opts.longitude_wrap
    '[0, 360)'
    """

    lat: float
    lon: float
    alt: float = 0.0
    opts: GeodeticOptions | None = None


@dataclass(frozen=True)
class ENUOptions:
    """Options controlling ECEF/LLA to local ENU conversion.

    Parameters
    ----------
    origin : GeodeticPosition | LocalOrigin | None, optional
        Default local origin. A public ``origin=`` argument overrides this
        field.
    ecef_frame : str | None, optional
        Expected source ECEF frame id.
    output_frame : str | None, optional
        Optional ENU output frame id.
    strict_frame : bool, optional
        Whether missing or incompatible source ECEF frame metadata fails
        closed.

    Returns
    -------
    ENUOptions
        Frozen local ENU conversion option object.

    Raises
    ------
    TypeError
        If public conversion boundaries receive unsupported option types.
    ValueError
        If public conversion boundaries receive unsupported frame ids or an
        incompatible strict-frame policy.

    Notes
    -----
    ENU output uses Cartesian ``Position`` labels ``x``, ``y``, ``z`` where
    x=east, y=north, and z=up. Frame policy is metadata-only; no frame graph is
    searched.

    Examples
    --------
    >>> from tal.geo import ENUOptions, LocalOrigin
    >>> opts = ENUOptions(origin=LocalOrigin(45.0, -75.0), output_frame="site_enu")
    >>> opts.output_frame
    'site_enu'
    """

    origin: "GeodeticPosition | LocalOrigin | None" = None
    ecef_frame: str | None = "earth_ecef"
    output_frame: str | None = None
    strict_frame: bool = False


@dataclass(frozen=True)
class GeodesicOptions:
    """Options controlling geodetic distance and bearing methods.

    Parameters
    ----------
    method : {'geodesic', 'local_enu'}, optional
        Distance/bearing method family. ``'geodesic'`` uses WGS84 geodesic
        calculations; ``'local_enu'`` uses an explicit local tangent-plane
        approximation.
    include_altitude : bool, optional
        Whether distance includes endpoint altitude difference. Bearings are
        always horizontal.
    local_origin : GeodeticPosition | LocalOrigin | None, optional
        Required origin for ``method='local_enu'``.

    Returns
    -------
    GeodesicOptions
        Frozen distance/bearing option object.

    Raises
    ------
    TypeError
        If public operation boundaries receive unsupported option types.
    ValueError
        If public operation boundaries receive unsupported option values or
        missing origin semantics for local ENU methods.

    Notes
    -----
    ``include_altitude=True`` is an endpoint distance adjustment, not 3D path
    integration. Constructing this dataclass does not import ``pyproj``.

    Examples
    --------
    >>> from tal.geo import GeodesicOptions, LocalOrigin
    >>> opts = GeodesicOptions(method="local_enu", local_origin=LocalOrigin(45.0, -75.0))
    >>> opts.method
    'local_enu'
    """

    method: Literal["geodesic", "local_enu"] = "geodesic"
    include_altitude: bool = False
    local_origin: "GeodeticPosition | LocalOrigin | None" = None


@dataclass(frozen=True)
class GeodeticInterpolationOptions:
    """Options controlling geodetic parameter interpolation.

    Parameters
    ----------
    method : {'nearest', 'geodesic_linear', 'ecef_linear', 'local_enu_linear'}, optional
        Interpolation method for LLA payloads.
    local_origin : GeodeticPosition | LocalOrigin | None, optional
        Required origin for ``method='local_enu_linear'``.
    longitude_wrap : {'shortest', 'preserve', '[-180, 180)', '[0, 360)'}, optional
        Longitude wrapping policy applied to interpolated output.
    duplicate_policy : {'invalid', 'left', 'right', 'raise'}, optional
        Duplicate parameter policy for linear methods. ``nearest`` follows
        core nearest-selection behavior.
    query_dim : str, optional
        Query dimension name used by core param evaluation.

    Returns
    -------
    GeodeticInterpolationOptions
        Frozen geodetic interpolation option object.

    Raises
    ------
    TypeError
        If public operation boundaries receive unsupported option types.
    ValueError
        If public operation boundaries receive unsupported option values,
        query dimension names, or missing local origins.

    Notes
    -----
    Interpolation reuses TAL core param mapping. Query-map construction follows
    the existing core eager query boundary, while payload interpolation remains
    xarray/Dask-aware where the chosen method supports it.

    Examples
    --------
    >>> from tal.geo import GeodeticInterpolationOptions
    >>> opts = GeodeticInterpolationOptions(method="ecef_linear", longitude_wrap="[0, 360)")
    >>> opts.longitude_wrap
    '[0, 360)'
    """

    method: Literal["nearest", "geodesic_linear", "ecef_linear", "local_enu_linear"] = "geodesic_linear"
    local_origin: "GeodeticPosition | LocalOrigin | None" = None
    longitude_wrap: Literal["shortest", "preserve", "[-180, 180)", "[0, 360)"] = "shortest"
    duplicate_policy: Literal["invalid", "left", "right", "raise"] = "invalid"
    query_dim: str = "query"


def _require_finite_number(value: object, *, field: str, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{owner}: {field} must be a finite numeric scalar.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{owner}: {field} must be finite.")
    return numeric


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
    if crs == _GEODETIC_CRS:
        normalized_crs = normalize_supported_crs(crs, expected=_GEODETIC_CRS, owner=owner)
    else:
        normalized_crs = normalize_crs_for_class(crs, expected="geographic", owner=owner, field="crs")
    if ecef_crs == _ECEF_CRS:
        normalized_ecef_crs = normalize_supported_crs(ecef_crs, expected=_ECEF_CRS, owner=owner)
    else:
        normalized_ecef_crs = normalize_crs_for_class(
            ecef_crs,
            expected="geocentric",
            owner=owner,
            field="ecef_crs",
        )
    return replace(opts, crs=normalized_crs, ecef_crs=normalized_ecef_crs)


def coerce_local_origin(origin: LocalOrigin, *, owner: str) -> LocalOrigin:
    """Validate and normalize a scalar local origin."""
    if not isinstance(origin, LocalOrigin):
        raise TypeError(f"{owner}: origin must be LocalOrigin.")
    lat = _require_finite_number(origin.lat, field="origin.lat", owner=owner)
    lon = _require_finite_number(origin.lon, field="origin.lon", owner=owner)
    alt = _require_finite_number(origin.alt, field="origin.alt", owner=owner)
    opts = None
    if origin.opts is not None:
        opts = coerce_geodetic_options(origin.opts, owner=owner, validate_crs=True)
    return replace(origin, lat=lat, lon=lon, alt=alt, opts=opts)


def _coerce_optional_local_origin(origin: object, *, owner: str):
    if origin is None:
        return None
    if isinstance(origin, LocalOrigin):
        return coerce_local_origin(origin, owner=owner)
    from tal.core.orchestration.inputs import coerce_analysis_object_input

    from .geodetic import GeodeticPosition

    try:
        ao = coerce_analysis_object_input(origin, owner=owner)
    except TypeError as exc:
        raise TypeError(
            f"{owner}: local_origin must be LocalOrigin, GeodeticPosition, AO-like LLA payload, or None."
        ) from exc
    try:
        return GeodeticPosition.from_lla(ao, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: local_origin must be a valid geodetic LLA origin.") from exc


def coerce_enu_options(opts: ENUOptions | None, *, owner: str) -> ENUOptions:
    """Coerce and validate ENU options at a public boundary."""
    if opts is None:
        opts = ENUOptions()
    if not isinstance(opts, ENUOptions):
        raise TypeError(f"{owner}: opts must be ENUOptions or None.")
    ecef_frame = None if opts.ecef_frame is None else _require_string(opts.ecef_frame, field="ecef_frame", owner=owner)
    output_frame = None
    if opts.output_frame is not None:
        output_frame = _require_string(opts.output_frame, field="output_frame", owner=owner)
    if not isinstance(opts.strict_frame, bool):
        raise TypeError(f"{owner}: strict_frame must be bool.")
    if opts.strict_frame and ecef_frame is None:
        raise ValueError(f"{owner}: strict_frame=True requires ecef_frame.")
    origin = opts.origin
    if isinstance(origin, LocalOrigin):
        origin = coerce_local_origin(origin, owner=owner)
    return replace(opts, origin=origin, ecef_frame=ecef_frame, output_frame=output_frame)


def coerce_geodesic_options(opts: GeodesicOptions | None, *, owner: str) -> GeodesicOptions:
    """Coerce and validate geodesic distance/bearing options."""
    if opts is None:
        opts = GeodesicOptions()
    if not isinstance(opts, GeodesicOptions):
        raise TypeError(f"{owner}: opts must be GeodesicOptions or None.")
    if opts.method not in _GEODESIC_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_GEODESIC_METHODS)!r}.")
    if not isinstance(opts.include_altitude, bool):
        raise TypeError(f"{owner}: include_altitude must be bool.")
    origin = _coerce_optional_local_origin(opts.local_origin, owner=owner)
    if opts.method == "local_enu" and origin is None:
        raise ValueError(f"{owner}: opts.local_origin is required when method='local_enu'.")
    return replace(opts, local_origin=origin)


def coerce_geodetic_interpolation_options(
    opts: GeodeticInterpolationOptions | None,
    *,
    owner: str,
) -> GeodeticInterpolationOptions:
    """Coerce and validate geodetic interpolation options."""
    if opts is None:
        opts = GeodeticInterpolationOptions()
    if not isinstance(opts, GeodeticInterpolationOptions):
        raise TypeError(f"{owner}: opts must be GeodeticInterpolationOptions or None.")
    if opts.method not in _INTERPOLATION_METHODS:
        raise ValueError(f"{owner}: opts.method must be one of {sorted(_INTERPOLATION_METHODS)!r}.")
    if opts.longitude_wrap not in _INTERPOLATION_LONGITUDE_WRAPS:
        raise ValueError(
            f"{owner}: opts.longitude_wrap must be one of {sorted(_INTERPOLATION_LONGITUDE_WRAPS)!r}."
        )
    if opts.duplicate_policy not in _DUPLICATE_POLICIES:
        raise ValueError(f"{owner}: opts.duplicate_policy must be one of {sorted(_DUPLICATE_POLICIES)!r}.")
    validate_query_dim_name(opts.query_dim, owner=owner)
    origin = _coerce_optional_local_origin(opts.local_origin, owner=owner)
    if opts.method == "local_enu_linear" and origin is None:
        raise ValueError(f"{owner}: opts.local_origin is required when method='local_enu_linear'.")
    return replace(opts, local_origin=origin)


__all__ = [
    "ENUOptions",
    "GeodeticDatum",
    "GeodeticInterpolationOptions",
    "GeodeticOptions",
    "GeodesicOptions",
    "LocalOrigin",
    "coerce_geodetic_interpolation_options",
    "coerce_geodetic_options",
    "coerce_enu_options",
    "coerce_geodesic_options",
    "coerce_local_origin",
]
