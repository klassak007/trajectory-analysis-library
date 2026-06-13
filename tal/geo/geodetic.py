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

from .metadata import normalize_geodetic_metadata

if TYPE_CHECKING:
    from tal.spatial import Position

    from .options import GeodeticOptions

_LLA_LABELS: tuple[str, str, str] = ("lat", "lon", "alt")


def _normalize_metadata(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    return normalize_geodetic_metadata(
        ds,
        opts=None,
        validate=ctx.phase != "from_unvalidated",
        validate_crs=False,
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
            If the ECEF payload has malformed TAL roles or core labels, option
            values or CRS identifiers are unsupported, or ``strict_frame=True``
            rejects incompatible frame metadata.

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


__all__ = [
    "GeodeticPosition",
]
