from __future__ import annotations

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

    def __init__(self, position: "Position") -> None:
        self._position = position

    def to_lla(
        self,
        *,
        opts: "GeodeticOptions | None" = None,
        validate: bool = True,
    ) -> "GeodeticPosition":
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
            If ``pyproj`` from the optional ``tal[geo]`` dependency group is
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
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import GeodeticOptions
        >>> from tal.spatial import Position
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "axis"), [[6378137.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "axis": ["x", "y", "z"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> opts = GeodeticOptions(ecef_frame=None)
        >>> lla = Position(ao).geo.to_lla(opts=opts)
        >>> list(lla.unsafe_data["lla"].values)
        ['lat', 'lon', 'alt']
        """
        from .local import position_to_lla

        return position_to_lla(self._position, opts=opts, validate=validate)

    def to_enu(
        self,
        origin: object | None = None,
        *,
        opts: "ENUOptions | None" = None,
        validate: bool = True,
    ) -> "Position":
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
            If ``pyproj`` from the optional ``tal[geo]`` dependency group is
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
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import ENUOptions, LocalOrigin
        >>> from tal.spatial import Position
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "axis"), [[6378137.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "axis": ["x", "y", "z"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> opts = ENUOptions(origin=LocalOrigin(0.0, 0.0, 0.0), output_frame="site_enu")
        >>> enu = Position(ao).geo.to_enu(opts=opts)
        >>> list(enu.unsafe_data["axis"].values)
        ['x', 'y', 'z']
        """
        from .local import ecef_to_enu

        return ecef_to_enu(self._position, origin=origin, opts=opts, validate=validate)

    def to_ecef(
        self,
        origin: object | None = None,
        *,
        opts: "ENUOptions | None" = None,
        validate: bool = True,
    ) -> "Position":
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
            If ``pyproj`` from the optional ``tal[geo]`` dependency group is
            not installed.
        TypeError
            If options or origin objects have unsupported types.
        ValueError
            If the source is not canonical ENU geo metadata, origin provenance
            is malformed or missing, or source/origin topology is incompatible.

        Notes
        -----
        This method is the public ENU-to-ECEF inverse in G2. It rejects ECEF
        inputs rather than returning them unchanged.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import ENUOptions, GeodeticPosition, LocalOrigin
        >>> ds = xr.Dataset(
        ...     {"position": (("sample", "lla"), [[0.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True)
        >>> opts = ENUOptions(origin=LocalOrigin(0.0, 0.0, 0.0), ecef_frame="custom_ecef")
        >>> enu = GeodeticPosition.from_lla(ao).to_enu(opts=opts)
        >>> ecef = enu.geo.to_ecef(opts=opts)
        >>> list(ecef.unsafe_data["axis"].values)
        ['x', 'y', 'z']
        """
        from .local import enu_to_ecef

        return enu_to_ecef(self._position, origin=origin, opts=opts, validate=validate)


def install_position_geo_accessor() -> None:
    """Install ``Position.geo`` without importing ``tal.geo`` from spatial."""
    from tal.spatial import Position

    if isinstance(getattr(Position, "geo", None), property):
        return

    def geo(self: "Position") -> PositionGeoAccessor:
        from .accessor import PositionGeoAccessor

        return PositionGeoAccessor(self)

    setattr(Position, "geo", property(geo))


__all__ = [
    "PositionGeoAccessor",
    "install_position_geo_accessor",
]
