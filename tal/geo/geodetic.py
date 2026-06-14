from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_single_core_dim_with_length,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.schema_read import validate_schema_if_needed
from tal.core.typed_lifecycle import TypedAnalysisObject, TypedLifecycleContext, TypedLifecycleSpec

from .metadata import normalize_geodetic_metadata, read_geo_block

if TYPE_CHECKING:
    from tal.linalg import Array
    from tal.spatial import Position

    from .options import ENUOptions, GeodesicOptions, GeodeticOptions
    from .projected import ProjectedPosition
    from .temporal import GeodeticParamAccessor

_LLA_LABELS: tuple[str, str, str] = ("lat", "lon", "alt")
_G1_GEODETIC_CRS = "EPSG:4979"


def _needs_crs_validation(ds: xr.Dataset, ctx: TypedLifecycleContext) -> bool:
    if ctx.phase == "init":
        return False
    block = read_geo_block(ds, owner=ctx.owner)
    if block is None or block.get("kind") != "geodetic_position":
        return False
    return block.get("crs", _G1_GEODETIC_CRS) != _G1_GEODETIC_CRS


def _normalize_metadata(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    return normalize_geodetic_metadata(
        ds,
        opts=None,
        validate=ctx.phase != "from_unvalidated",
        validate_crs=_needs_crs_validation(ds, ctx),
        owner=ctx.owner,
    )


def _enforce_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    candidate = validate_schema_if_needed(ds)
    var_name = select_single_numeric_var(candidate, owner=ctx.owner, what="GeodeticPosition")
    core_dim = require_single_core_dim_with_length(
        candidate,
        expected_length=3,
        owner=ctx.owner,
        what="GeodeticPosition",
    )
    require_var_contains_dims(
        candidate,
        var_name=var_name,
        required_dims=(core_dim,),
        owner=ctx.owner,
        what="GeodeticPosition",
    )
    labels = require_explicit_unique_dim_labels(candidate, dim=core_dim, owner=ctx.owner, what="GeodeticPosition")
    require_exact_labels(labels, expected=_LLA_LABELS, owner=ctx.owner, what="GeodeticPosition core")


class GeodeticPosition(TypedAnalysisObject):
    """Geodetic latitude, longitude, altitude position payload.

    Notes
    -----
    ``GeodeticPosition`` is separate from ``tal.spatial.Position`` so generic
    cartesian vector operations never treat LLA values as xyz vectors.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.geo import GeodeticPosition
    >>> ds = xr.Dataset(
    ...     {"lla": (("sample", "lla_axis"), [[45.0, -75.0, 100.0]])},
    ...     coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
    ... )
    >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True)
    >>> isinstance(GeodeticPosition(ao), GeodeticPosition)
    True
    """

    LLA_LABELS: tuple[str, str, str] = _LLA_LABELS
    LIFECYCLE = TypedLifecycleSpec(
        type_name="GeodeticPosition",
        owner_prefix="geo.GeodeticPosition",
        normalize=_normalize_metadata,
        enforce=_enforce_invariants,
    )

    @classmethod
    def from_lla(
        cls,
        value: object,
        *,
        opts: "GeodeticOptions | None" = None,
        validate: bool = True,
    ) -> "GeodeticPosition":
        """Normalize an LLA payload into a geodetic position.

        Parameters
        ----------
        value : object
            ``GeodeticPosition``, ``AnalysisObject``, ``xarray.Dataset``, or
            ``xarray.DataArray`` with declared TAL roles and an LLA core dim.
        opts : GeodeticOptions | None, optional
            Explicit geo metadata options controlling ``crs``, ``datum``,
            ``longitude_wrap``, and related geo metadata fields.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        GeodeticPosition
            Typed geodetic position with normalized geo metadata.

        Raises
        ------
        TypeError
            If ``value`` cannot be coerced to a TAL analysis object or
            ``opts`` is not ``GeodeticOptions`` or ``None``.
        ValueError
            If the payload has malformed TAL roles, does not have exactly one
            LLA core dim with ``lat``, ``lon``, ``alt`` labels, or contains
            unsupported geo metadata or option values.
        ImportError
            If explicit ``opts`` require CRS normalization and ``pyproj`` from
            the optional ``tal[geo]`` dependency group is not installed.

        Notes
        -----
        This method stamps metadata on an already-shaped LLA payload; it does
        not parse separate latitude, longitude, and altitude arrays. The public
        LLA component order is ``lat``, ``lon``, ``alt``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodeticOptions, GeodeticPosition
        >>> ds = xr.Dataset(
        ...     {"lla": (("sample", "lla_axis"), [[45.0, -75.0, 100.0]])},
        ...     coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True)
        >>> opts = GeodeticOptions(longitude_wrap="[0, 360)")
        >>> lla = GeodeticPosition.from_lla(ao, opts=opts)
        >>> lla.unsafe_data.attrs["tal"]["ext"]["geo"]["longitude_wrap"]
        '[0, 360)'
        """
        from .conversion import from_lla

        return from_lla(value, opts=opts, validate=validate)

    @classmethod
    def from_ecef(
        cls,
        value: object,
        *,
        opts: "GeodeticOptions | None" = None,
        validate: bool = True,
    ) -> "GeodeticPosition":
        """Convert a Cartesian ECEF position into geodetic LLA coordinates.

        Parameters
        ----------
        value : object
            ``Position`` or AO-like value that can be coerced to cartesian
            ``tal.spatial.Position``.
        opts : GeodeticOptions | None, optional
            Geodetic conversion options controlling ``crs``, ``ecef_crs``,
            ``ecef_frame``, and ``strict_frame``. ``None`` uses ECEF
            provenance when present, otherwise default WGS84 options.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        GeodeticPosition
            Converted geodetic position.

        Raises
        ------
        ImportError
            If ``pyproj`` from the optional ``tal[geo]`` dependency group is
            not installed.
        TypeError
            If ``value`` cannot be coerced to ``Position`` or ``opts`` is not
            ``GeodeticOptions`` or ``None``.
        ValueError
            If the ECEF payload has malformed TAL roles or core labels,
            present geo provenance is non-ECEF, option values or CRS
            identifiers are unsupported, or ``strict_frame=True`` rejects
            incompatible frame metadata.

        Notes
        -----
        ECEF inputs use ``x``, ``y``, ``z`` core labels. Outputs use the public
        LLA label order ``lat``, ``lon``, ``alt``. This is an explicit CRS
        conversion, not a spatial frame-graph transform.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodeticOptions, GeodeticPosition
        >>> from tal.spatial import Position
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "axis"), [[6378137.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "axis": ["x", "y", "z"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> opts = GeodeticOptions(longitude_wrap="[0, 360)")
        >>> lla = GeodeticPosition.from_ecef(Position(ao), opts=opts)
        >>> list(lla.unsafe_data["lla"].values)
        ['lat', 'lon', 'alt']
        """
        from .conversion import from_ecef

        return from_ecef(value, opts=opts, validate=validate)

    def to_ecef(
        self,
        *,
        opts: "GeodeticOptions | None" = None,
        validate: bool = True,
    ) -> "Position":
        """Convert geodetic LLA coordinates to Cartesian ECEF position.

        Parameters
        ----------
        opts : GeodeticOptions | None, optional
            Geodetic conversion options controlling ``crs``, ``ecef_crs``,
            ``ecef_frame``, and ``strict_frame``. ``None`` uses default WGS84
            options.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        tal.spatial.Position
            Cartesian ECEF position with xyz core labels.

        Raises
        ------
        ImportError
            If ``pyproj`` from the optional ``tal[geo]`` dependency group is
            not installed.
        TypeError
            If ``opts`` is not ``GeodeticOptions`` or ``None``.
        ValueError
            If the LLA payload has malformed TAL roles or core labels, option
            values or CRS identifiers are unsupported, or ``strict_frame=True``
            rejects incompatible frame metadata.

        Notes
        -----
        Input LLA values are selected by component labels in public
        ``lat``, ``lon``, ``alt`` order. The pyproj boundary uses longitude,
        latitude, altitude ordering internally. The returned cartesian position
        has ``x``, ``y``, ``z`` core labels and records ECEF provenance
        metadata under ``tal.ext.geo``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodeticOptions, GeodeticPosition
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "lla"), [[0.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
        >>> opts = GeodeticOptions(longitude_wrap="[0, 360)")
        >>> ecef = GeodeticPosition.from_lla(ao).to_ecef(opts=opts)
        >>> list(ecef.unsafe_data["axis"].values)
        ['x', 'y', 'z']
        """
        from .conversion import to_ecef

        return to_ecef(self, opts=opts, validate=validate)

    def to_enu(
        self,
        origin: object | None = None,
        *,
        opts: "ENUOptions | None" = None,
        validate: bool = True,
    ) -> "Position":
        """Convert geodetic LLA coordinates to local ENU position.

        Parameters
        ----------
        origin : object | None, optional
            ``LocalOrigin`` or AO-like value coercible to ``GeodeticPosition``.
            This argument overrides ``opts.origin``.
        opts : ENUOptions | None, optional
            ENU conversion options controlling ``origin``, ``ecef_frame``,
            ``output_frame``, and ``strict_frame``.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        tal.spatial.Position
            Cartesian ENU position with ``x``, ``y``, ``z`` labels where
            x=east, y=north, and z=up.

        Raises
        ------
        ImportError
            If ``pyproj`` from the optional ``tal[geo]`` dependency group is
            not installed.
        TypeError
            If ``opts`` is not ``ENUOptions`` or ``origin`` is not a supported
            local-origin object.
        ValueError
            If the origin is missing, source/origin topology is incompatible,
            CRS metadata is unsupported, the payload is malformed, or
            ``strict_frame=True`` rejects source frame metadata.

        Notes
        -----
        This is an explicit geodetic-to-ECEF-to-ENU conversion. It does not
        search a frame graph, and the returned ENU payload remains a Cartesian
        ``Position`` rather than a geodetic object.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import ENUOptions, GeodeticPosition, LocalOrigin
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "lla"), [[45.0, -75.0, 100.0]])},
        ...     coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
        >>> opts = ENUOptions(output_frame="site_enu")
        >>> enu = GeodeticPosition.from_lla(ao).to_enu(LocalOrigin(45.0, -75.0, 100.0), opts=opts)
        >>> list(enu.unsafe_data["axis"].values)
        ['x', 'y', 'z']
        """
        from .local import geodetic_to_enu

        return geodetic_to_enu(self, origin=origin, opts=opts, validate=validate)

    def to_crs(
        self,
        dst: str,
        *,
        validate: bool = True,
    ) -> "GeodeticPosition | Position | ProjectedPosition":
        """Transform geodetic coordinates to another CRS.

        Parameters
        ----------
        dst : str
            WGS84-compatible geographic, geocentric/ECEF, or projected CRS
            string.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        GeodeticPosition, tal.spatial.Position, or ProjectedPosition
            Output type selected by destination CRS class.

        Raises
        ------
        ImportError
            If ``pyproj`` from the optional ``tal[geo]`` dependency group is
            not installed.
        TypeError
            If ``dst`` is not a string.
        ValueError
            If source metadata or the destination CRS is malformed,
            non-WGS84, or unsupported.

        Notes
        -----
        CRS transforms are explicit pyproj-backed conversions. Projected
        destinations return ``ProjectedPosition``; ECEF destinations return
        cartesian ``Position``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodeticPosition
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "lla"), [[34.0, -118.0, 20.0]])},
        ...     coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
        >>> projected = GeodeticPosition.from_lla(ao).to_crs("EPSG:32611")
        >>> list(projected.unsafe_data["projected"].values)
        ['easting', 'northing', 'height']
        """
        from .crs_transform import transform_crs

        return transform_crs(self, dst=dst, validate=validate)

    def distance_to(
        self,
        other: object,
        *,
        opts: "GeodesicOptions | None" = None,
        validate: bool = True,
    ) -> "Array":
        """Compute geodetic distance to another LLA position.

        Parameters
        ----------
        other : object
            ``GeodeticPosition`` or AO-like LLA payload aligned with this
            object by TAL topology rules.
        opts : GeodesicOptions | None, optional
            Distance options including ``method``, ``include_altitude``, and
            ``local_origin``. ``None`` uses WGS84 geodesic surface distance.
        validate : bool, optional
            Whether to validate the scalar output before returning.

        Returns
        -------
        tal.linalg.Array
            Scalar-core distance payload in meters, with data variable
            ``distance_m``.

        Raises
        ------
        ImportError
            If ``pyproj`` from ``tal[geo]`` is required and unavailable.
        TypeError
            If ``other`` or ``opts`` has an unsupported type.
        ValueError
            If operands have malformed LLA metadata, incompatible CRS
            provenance, unsupported option values, or missing local ENU origin.

        Notes
        -----
        ``include_altitude=True`` applies endpoint altitude adjustment; it is
        not 3D path integration. Scalar results strip geo/spatial/frame
        extension metadata because they are not positions.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodesicOptions, GeodeticPosition
        >>> ds = xr.Dataset(
        ...     {"lla": (("sample", "lla_axis"), [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])},
        ...     coords={"sample": [0, 1], "lla_axis": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True)
        >>> opts = GeodesicOptions(include_altitude=True)
        >>> out = GeodeticPosition.from_lla(ao).distance_to(GeodeticPosition.from_lla(ao), opts=opts)
        >>> tuple(out.unsafe_data.attrs["tal"]["core"]["roles"]["core_dims"])
        ()
        """
        from .distance import distance_to

        return distance_to(self, other, opts=opts, validate=validate)

    def initial_bearing_to(
        self,
        other: object,
        *,
        opts: "GeodesicOptions | None" = None,
        validate: bool = True,
    ) -> "Array":
        """Compute initial bearing to another LLA position.

        Parameters
        ----------
        other : object
            ``GeodeticPosition`` or AO-like LLA payload.
        opts : GeodesicOptions | None, optional
            Bearing method options. ``method='local_enu'`` requires an
            explicit local origin. ``include_altitude`` does not affect
            bearings.
        validate : bool, optional
            Whether to validate the scalar output before returning.

        Returns
        -------
        tal.linalg.Array
            Scalar-core bearing payload in degrees clockwise from north, with
            data variable ``initial_bearing_deg``.

        Raises
        ------
        ImportError
            If ``pyproj`` from ``tal[geo]`` is required and unavailable.
        TypeError
            If ``other`` or ``opts`` has an unsupported type.
        ValueError
            If operands have malformed LLA metadata, incompatible CRS
            provenance, unsupported option values, missing local ENU origin, or
            ambiguous pole bearing geometry.

        Notes
        -----
        Coincident endpoints produce ``NaN``. Bearings are horizontal even when
        distance options include altitude.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodesicOptions, GeodeticPosition
        >>> ds = xr.Dataset(
        ...     {"lla": (("sample", "lla_axis"), [[0.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True)
        >>> opts = GeodesicOptions(include_altitude=True)
        >>> out = GeodeticPosition.from_lla(ao).initial_bearing_to(GeodeticPosition.from_lla(ao), opts=opts)
        >>> "initial_bearing_deg" in out.unsafe_data
        True
        """
        from .distance import initial_bearing_to

        return initial_bearing_to(self, other, opts=opts, validate=validate)

    def final_bearing_to(
        self,
        other: object,
        *,
        opts: "GeodesicOptions | None" = None,
        validate: bool = True,
    ) -> "Array":
        """Compute final bearing arriving at another LLA position.

        Parameters
        ----------
        other : object
            ``GeodeticPosition`` or AO-like LLA payload.
        opts : GeodesicOptions | None, optional
            Bearing options including ``method``, ``include_altitude``, and
            ``local_origin``. ``method='local_enu'`` uses the same local
            straight-line heading approximation as initial bearing.
        validate : bool, optional
            Whether to validate the scalar output before returning.

        Returns
        -------
        tal.linalg.Array
            Scalar-core bearing payload in degrees clockwise from north, with
            data variable ``final_bearing_deg``.

        Raises
        ------
        ImportError
            If ``pyproj`` from ``tal[geo]`` is required and unavailable.
        TypeError
            If ``other`` or ``opts`` has an unsupported type.
        ValueError
            If operands have malformed LLA metadata, incompatible CRS
            provenance, unsupported option values, missing local ENU origin, or
            ambiguous pole bearing geometry.

        Notes
        -----
        Final bearing is derived from the geodesic back azimuth for global
        geodesic mode. Local ENU mode is an explicit tangent-plane
        approximation.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodesicOptions, GeodeticPosition
        >>> ds = xr.Dataset(
        ...     {"lla": (("sample", "lla_axis"), [[0.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True)
        >>> opts = GeodesicOptions(include_altitude=True)
        >>> out = GeodeticPosition.from_lla(ao).final_bearing_to(GeodeticPosition.from_lla(ao), opts=opts)
        >>> "final_bearing_deg" in out.unsafe_data
        True
        """
        from .distance import final_bearing_to

        return final_bearing_to(self, other, opts=opts, validate=validate)

    @property
    def param(self) -> "GeodeticParamAccessor":
        """Return the parameter-domain accessor for geodetic interpolation.

        Returns
        -------
        GeodeticParamAccessor
            Typed accessor whose interpolation defaults are geodetic-safe.
        """
        from .temporal import GeodeticParamAccessor

        return GeodeticParamAccessor(self)


__all__ = [
    "GeodeticPosition",
]
