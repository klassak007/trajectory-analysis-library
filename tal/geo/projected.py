from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.core.typed_lifecycle import TypedAnalysisObject, TypedLifecycleContext, TypedLifecycleSpec

from .metadata import normalize_existing_projected_metadata, normalize_projected_metadata

if TYPE_CHECKING:
    from tal.spatial import Position

    from .geodetic import GeodeticPosition

_PROJECTED_2D_LABELS: tuple[str, str] = ("easting", "northing")
_PROJECTED_3D_LABELS: tuple[str, str, str] = ("easting", "northing", "height")


def _normalize_metadata(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    return normalize_existing_projected_metadata(
        ds,
        validate=ctx.phase != "from_unvalidated",
        owner=ctx.owner,
    )


def _projected_core_dim(ds: xr.Dataset, *, owner: str) -> str:
    declared, _, _, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: ProjectedPosition requires declared roles.")
    if len(core_dims) != 1:
        raise ValueError(f"{owner}: ProjectedPosition requires exactly one core dim; got {core_dims!r}.")
    core_dim = core_dims[0]
    length = int(ds.sizes.get(core_dim, -1))
    if length not in {2, 3}:
        raise ValueError(f"{owner}: ProjectedPosition core dim {core_dim!r} must have length 2 or 3.")
    return core_dim


def _enforce_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    candidate = validate_schema_if_needed(ds)
    var_name = select_single_numeric_var(candidate, owner=ctx.owner, what="ProjectedPosition")
    core_dim = _projected_core_dim(candidate, owner=ctx.owner)
    require_var_contains_dims(
        candidate,
        var_name=var_name,
        required_dims=(core_dim,),
        owner=ctx.owner,
        what="ProjectedPosition",
    )
    labels = require_explicit_unique_dim_labels(candidate, dim=core_dim, owner=ctx.owner, what="ProjectedPosition")
    expected = _PROJECTED_3D_LABELS if len(labels) == 3 else _PROJECTED_2D_LABELS
    require_exact_labels(labels, expected=expected, owner=ctx.owner, what="ProjectedPosition core")


class ProjectedPosition(TypedAnalysisObject):
    """Projected coordinate position payload.

    Parameters
    ----------
    data : object
        TAL analysis object, dataset, or data array with projected metadata
        and core labels ``easting``, ``northing`` and optional ``height``.

    Notes
    -----
    ``ProjectedPosition`` is separate from ``tal.spatial.Position``. It does
    not expose Cartesian vector semantics for projected coordinates.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.geo import ProjectedPosition
    >>> ds = xr.Dataset(
    ...     {"xy": (("sample", "projected"), [[500000.0, 4100000.0]])},
    ...     coords={"sample": [0], "projected": ["easting", "northing"]},
    ... )
    >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("projected",), validate=True)
    >>> projected = ProjectedPosition.from_projected(ao, crs="EPSG:32611")
    >>> list(projected.unsafe_data["projected"].values)
    ['easting', 'northing']
    """

    PROJECTED_2D_LABELS: tuple[str, str] = _PROJECTED_2D_LABELS
    PROJECTED_3D_LABELS: tuple[str, str, str] = _PROJECTED_3D_LABELS
    LIFECYCLE = TypedLifecycleSpec(
        type_name="ProjectedPosition",
        owner_prefix="geo.ProjectedPosition",
        normalize=_normalize_metadata,
        enforce=_enforce_invariants,
    )

    @classmethod
    def from_projected(cls, value: object, *, crs: str, validate: bool = True) -> "ProjectedPosition":
        """Normalize a projected-coordinate payload.

        Parameters
        ----------
        value : object
            AO-like projected payload with one projected core dimension.
        crs : str
            WGS84-compatible projected CRS string, for example
            ``"EPSG:32611"``.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        ProjectedPosition
            Typed projected coordinate payload.

        Raises
        ------
        ImportError
            If ``pyproj`` from ``tal[geo]`` is required and unavailable.
        TypeError
            If ``value`` cannot be coerced to a TAL analysis object or
            ``crs`` is not a string.
        ValueError
            If payload roles, labels, metadata, or CRS class are unsupported.

        Notes
        -----
        G4 accepts public CRS inputs as strings only. Projected payload labels
        are ``easting``, ``northing`` and optional ``height``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import ProjectedPosition
        >>> ds = xr.Dataset(
        ...     {"xy": (("sample", "projected"), [[500000.0, 4100000.0]])},
        ...     coords={"sample": [0], "projected": ["easting", "northing"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("projected",), validate=True)
        >>> projected = ProjectedPosition.from_projected(ao, crs="EPSG:32611")
        >>> projected.unsafe_data.attrs["tal"]["ext"]["geo"]["kind"]
        'projected_position'
        """

        owner = "geo.ProjectedPosition.from_projected"
        source = coerce_analysis_object_input(value, owner=owner)
        ds = normalize_projected_metadata(source.unsafe_data, crs=crs, validate=False, owner=owner)
        if validate:
            return cls._from_validated(ds)
        return cls._from_unvalidated(ds)

    def to_crs(self, dst: str, *, validate: bool = True) -> "GeodeticPosition | Position | ProjectedPosition":
        """Transform projected coordinates to another CRS.

        Parameters
        ----------
        dst : str
            WGS84-compatible geographic, geocentric/ECEF, or projected CRS.
        validate : bool, optional
            Whether to validate the output before returning.

        Returns
        -------
        GeodeticPosition, tal.spatial.Position, or ProjectedPosition
            Output type selected by destination CRS class.

        Raises
        ------
        ImportError
            If ``pyproj`` from ``tal[geo]`` is required and unavailable.
        TypeError
            If ``dst`` is not a string.
        ValueError
            If source metadata, destination CRS, or projected height semantics
            are unsupported.

        Notes
        -----
        A 2D projected source cannot be transformed to geographic or ECEF
        output because TAL does not invent height.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.geo import ProjectedPosition
        >>> ds = xr.Dataset(
        ...     {"xy": (("sample", "projected"), [[500000.0, 4100000.0, 20.0]])},
        ...     coords={"sample": [0], "projected": ["easting", "northing", "height"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("projected",), validate=True)
        >>> lla = ProjectedPosition.from_projected(ao, crs="EPSG:32611").to_crs("EPSG:4979")
        >>> list(lla.unsafe_data["lla"].values)
        ['lat', 'lon', 'alt']
        """

        from .crs_transform import transform_crs

        return transform_crs(self, dst=dst, validate=validate)


__all__ = [
    "ProjectedPosition",
]
