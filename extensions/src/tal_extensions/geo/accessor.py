from __future__ import annotations

from inspect import getattr_static
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tal.spatial import Position

    from .geodetic import GeodeticPosition
    from .options import ENUOptions, GeodeticOptions


class PositionGeoAccessor:
    """Explicit geo conversion accessor for Cartesian ``Position``.

    Notes
    -----
    The accessor is ergonomic sugar only. It does not change ``Position``
    identity and generic spatial operations still treat the payload as
    Cartesian.
    """

    def __init__(self, position: Position) -> None:
        self._position = position

    def to_lla(
        self,
        *,
        opts: GeodeticOptions | None = None,
        validate: bool = True,
    ) -> GeodeticPosition:
        """Convert canonical ECEF ``Position`` data to geodetic LLA.

        Parameters
        ----------
        opts : GeodeticOptions | None, optional
            Geodetic conversion options controlling ``crs``, ``ecef_crs``,
            ``ecef_frame``, and ``strict_frame``. ``None`` requires canonical
            ECEF geo provenance on the source.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        GeodeticPosition
            Converted geodetic position.

        Raises
        ------
        ImportError
            If ``pyproj`` from the optional ``tal-extensions[geo]`` dependency group is
            not installed.
        TypeError
            If ``opts`` is not ``GeodeticOptions`` or ``None``.
        ValueError
            If source geo provenance is missing when ``opts is None``, the
            source has non-ECEF Cartesian geo provenance, the payload is
            malformed, CRS values are unsupported, or strict frame policy
            rejects the input.

        Notes
        -----
        This method is a CRS conversion boundary, not a spatial frame transform.

        Examples
        --------
        >>> from tal_extensions.geo import register_position_accessor
        >>> register_position_accessor()
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal_extensions.geo import GeodeticOptions
        >>> from tal.spatial import Position
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "axis"), [[6378137.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "axis": ["x", "y", "z"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> opts = GeodeticOptions(ecef_frame=None)
        >>> lla = Position(ao).geo.to_lla(opts=opts)
        >>> lla.as_dataset()["lla"].values.tolist()
        ['lat', 'lon', 'alt']
        """
        from .local import position_to_lla

        return position_to_lla(self._position, opts=opts, validate=validate)

    def to_enu(
        self,
        origin: object | None = None,
        *,
        opts: ENUOptions | None = None,
        validate: bool = True,
    ) -> Position:
        """Convert canonical ECEF ``Position`` data to local ENU.

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
        Position
            Cartesian ENU position with labels ``x``, ``y``, ``z`` where
            x=east, y=north, and z=up.

        Raises
        ------
        ImportError
            If ``pyproj`` from the optional ``tal-extensions[geo]`` dependency group is
            not installed.
        TypeError
            If options or origin objects have unsupported types.
        ValueError
            If the origin is missing, source/origin topology is incompatible,
            source metadata identifies a non-ECEF Cartesian geo system, or
            strict frame policy rejects the input.

        Notes
        -----
        ENU conversion uses xarray alignment by named dimensions and coordinate
        labels. It does not positionally match origin samples.

        Examples
        --------
        >>> from tal_extensions.geo import register_position_accessor
        >>> register_position_accessor()
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal_extensions.geo import ENUOptions, LocalOrigin
        >>> from tal.spatial import Position
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "axis"), [[6378137.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "axis": ["x", "y", "z"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> opts = ENUOptions(origin=LocalOrigin(0.0, 0.0, 0.0), output_frame="site_enu")
        >>> enu = Position(ao).geo.to_enu(opts=opts)
        >>> enu.as_dataset()["axis"].values.tolist()
        ['x', 'y', 'z']
        """
        from .local import ecef_to_enu

        return ecef_to_enu(self._position, origin=origin, opts=opts, validate=validate)

    def to_ecef(
        self,
        origin: object | None = None,
        *,
        opts: ENUOptions | None = None,
        validate: bool = True,
    ) -> Position:
        """Convert local ENU ``Position`` data back to ECEF.

        Parameters
        ----------
        origin : object | None, optional
            Optional origin override. When omitted, valid ENU provenance on the
            source supplies the origin.
        opts : ENUOptions | None, optional
            ENU conversion options controlling ``origin``, ``ecef_frame``,
            ``output_frame``, and ``strict_frame``.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        Position
            Cartesian ECEF position with ``x``, ``y``, ``z`` labels.

        Raises
        ------
        ImportError
            If ``pyproj`` from the optional ``tal-extensions[geo]`` dependency group is
            not installed.
        TypeError
            If options or origin objects have unsupported types.
        ValueError
            If the source is not canonical ENU geo metadata, origin provenance
            is malformed or missing, or source/origin topology is incompatible.

        Notes
        -----
        This method converts ENU coordinates to ECEF. It rejects ECEF
        inputs rather than returning them unchanged.

        Examples
        --------
        >>> from tal_extensions.geo import register_position_accessor
        >>> register_position_accessor()
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal_extensions.geo import ENUOptions, GeodeticPosition, LocalOrigin
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "lla"), [[0.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
        >>> opts = ENUOptions(origin=LocalOrigin(0.0, 0.0, 0.0), ecef_frame="custom_ecef")
        >>> enu = GeodeticPosition.from_lla(ao).to_enu(opts=opts)
        >>> ecef = enu.geo.to_ecef(opts=opts)
        >>> ecef.as_dataset()["axis"].values.tolist()
        ['x', 'y', 'z']
        """
        from .local import enu_to_ecef

        return enu_to_ecef(self._position, origin=origin, opts=opts, validate=validate)


def _geo_accessor(position: Position) -> PositionGeoAccessor:
    return PositionGeoAccessor(position)


_GEO_PROPERTY = property(_geo_accessor)
_MISSING_ATTRIBUTE = object()


def register_position_accessor() -> None:
    """Explicitly install the geo accessor on TAL Position objects.

    Returns
    -------
    None
        Installs one extension-owned class property. Repeated calls are harmless.

    Raises
    ------
    RuntimeError
        If another attribute already occupies ``Position.geo``.

    Notes
    -----
    Importing the extension does not register the accessor or load pyproj.
    Registration changes the Position class for the current Python process.
    Typed constructors, conversions, and interpolation do not require it.

    Examples
    --------
    >>> from tal_extensions.geo import register_position_accessor
    >>> from tal.spatial import Position
    >>> register_position_accessor()
    >>> register_position_accessor()
    >>> assert isinstance(Position.geo, property)
    """
    from tal.spatial import Position

    existing = getattr_static(Position, "geo", _MISSING_ATTRIBUTE)
    if existing is _GEO_PROPERTY:
        return
    if existing is not _MISSING_ATTRIBUTE:
        raise RuntimeError(
            "geo.register_position_accessor: Position.geo is already occupied."
        )
    Position.geo = _GEO_PROPERTY


__all__ = [
    "PositionGeoAccessor",
    "register_position_accessor",
]
