from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_single_core_dim_with_length,
    require_var_contains_dims,
)
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
    validate_schema_if_needed,
)
from tal.core.schema import merge_schema
from tal.core.typed_lifecycle import TypedAnalysisObject, TypedLifecycleContext, TypedLifecycleSpec

from .metadata import normalize_topocentric_metadata

if TYPE_CHECKING:
    from tal.linalg import Vector3

_DIRECTION_VAR = "direction"
_ALTITUDE_VAR = "altitude_deg"
_AZIMUTH_VAR = "azimuth_deg"
_ENU_LABELS: tuple[str, str, str] = ("east", "north", "up")
_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")
_DATA_VARS = frozenset({_DIRECTION_VAR, _ALTITUDE_VAR, _AZIMUTH_VAR})


def _coerce_direction_source(value: object, ctx: TypedLifecycleContext) -> AnalysisObject:
    if isinstance(value, xr.DataArray):
        named = value.rename(_DIRECTION_VAR)
        return AnalysisObject(named)
    return coerce_analysis_object_input(value, owner=ctx.owner)


def _semantic_dims(sequence_dim: str | None, batch_dims: tuple[str, ...]) -> tuple[str, ...]:
    if sequence_dim is None:
        return batch_dims
    return (sequence_dim, *batch_dims)


def _require_non_empty_string(value: object, *, field: str, owner: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{owner}: {field} must be a non-empty string.")


def _component(data: xr.DataArray, *, dim: str, label: str) -> xr.DataArray:
    return data.sel({dim: label}, drop=True)


def _alt_az_from_direction(data: xr.DataArray, *, core_dim: str) -> tuple[xr.DataArray, xr.DataArray]:
    east = _component(data, dim=core_dim, label="east")
    north = _component(data, dim=core_dim, label="north")
    up = _component(data, dim=core_dim, label="up")
    horizontal = np.sqrt(east * east + north * north)
    altitude = np.degrees(np.arctan2(up, horizontal)).rename(_ALTITUDE_VAR)
    azimuth = ((np.degrees(np.arctan2(east, north)) + 360.0) % 360.0).rename(_AZIMUTH_VAR)
    return altitude, azimuth


def _synthesize_payload_vars(ds: xr.Dataset) -> xr.Dataset:
    if _DIRECTION_VAR not in ds.data_vars:
        return ds
    declared, _, _, core_dims = read_roles(ds)
    if not declared or len(core_dims) != 1:
        return ds
    core_dim = core_dims[0]
    if core_dim not in ds[_DIRECTION_VAR].dims or int(ds.sizes.get(core_dim, -1)) != 3:
        return ds
    labels = tuple(ds.coords[core_dim].to_index().tolist()) if core_dim in ds.coords else ()
    if labels != _ENU_LABELS:
        return ds
    altitude, azimuth = _alt_az_from_direction(ds[_DIRECTION_VAR], core_dim=core_dim)
    updates: dict[str, xr.DataArray] = {}
    if _ALTITUDE_VAR not in ds.data_vars:
        updates[_ALTITUDE_VAR] = altitude
    if _AZIMUTH_VAR not in ds.data_vars:
        updates[_AZIMUTH_VAR] = azimuth
    if not updates:
        return ds
    return ds.assign(updates)


def _normalize_direction_metadata(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    with_payload = _synthesize_payload_vars(ds)
    return normalize_topocentric_metadata(
        with_payload,
        validate=ctx.phase != "from_unvalidated",
        owner=ctx.owner,
    )


def _require_known_data_vars(ds: xr.Dataset, *, owner: str) -> None:
    extra = sorted(str(name) for name in ds.data_vars if str(name) not in _DATA_VARS)
    if extra:
        raise ValueError(f"{owner}: TopocentricDirection data variables must be {sorted(_DATA_VARS)!r}; got extra {extra!r}.")


def _require_payload_var(ds: xr.Dataset, *, name: str, semantic_dims: tuple[str, ...], owner: str) -> None:
    if name not in ds.data_vars:
        raise ValueError(f"{owner}: TopocentricDirection requires data variable {name!r}.")
    arr = ds[name]
    if not np.issubdtype(np.dtype(arr.dtype), np.number):
        raise TypeError(f"{owner}: TopocentricDirection variable {name!r} must be numeric; got {arr.dtype!r}.")
    illegal = tuple(dim for dim in arr.dims if dim not in semantic_dims)
    if illegal:
        raise ValueError(f"{owner}: TopocentricDirection variable {name!r} has non-semantic dims {illegal!r}.")


def _enforce_direction_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    candidate = validate_schema_if_needed(ds)
    _require_known_data_vars(candidate, owner=ctx.owner)
    declared, sequence_dim, batch_dims, _ = read_roles(candidate)
    if not declared:
        raise ValueError(f"{ctx.owner}: TopocentricDirection requires declared roles.")
    core_dim = require_single_core_dim_with_length(
        candidate,
        expected_length=3,
        owner=ctx.owner,
        what="TopocentricDirection",
    )
    if _DIRECTION_VAR not in candidate.data_vars:
        raise ValueError(f"{ctx.owner}: TopocentricDirection requires data variable 'direction'.")
    if not np.issubdtype(np.dtype(candidate[_DIRECTION_VAR].dtype), np.number):
        raise TypeError(f"{ctx.owner}: TopocentricDirection direction variable must be numeric.")
    require_var_contains_dims(
        candidate,
        var_name=_DIRECTION_VAR,
        required_dims=(core_dim,),
        owner=ctx.owner,
        what="TopocentricDirection",
    )
    labels = require_explicit_unique_dim_labels(candidate, dim=core_dim, owner=ctx.owner, what="TopocentricDirection")
    require_exact_labels(labels, expected=_ENU_LABELS, owner=ctx.owner, what="TopocentricDirection core")
    semantic_dims = _semantic_dims(sequence_dim, batch_dims)
    _require_payload_var(candidate, name=_DIRECTION_VAR, semantic_dims=(*semantic_dims, core_dim), owner=ctx.owner)
    _require_payload_var(candidate, name=_ALTITUDE_VAR, semantic_dims=semantic_dims, owner=ctx.owner)
    _require_payload_var(candidate, name=_AZIMUTH_VAR, semantic_dims=semantic_dims, owner=ctx.owner)


def _vector3_dataset(
    ds: xr.Dataset,
    *,
    axis: str,
    output_var: str,
    owner: str,
) -> tuple[xr.Dataset, str | None, tuple[str, ...]]:
    candidate = validate_schema_if_needed(ds)
    declared, sequence_dim, batch_dims, core_dims = read_roles(candidate)
    if not declared or len(core_dims) != 1:
        raise ValueError(f"{owner}: TopocentricDirection requires declared roles and one ENU core dim.")
    core_dim = core_dims[0]
    semantic_dims = _semantic_dims(sequence_dim, batch_dims)
    if axis in semantic_dims and axis != core_dim:
        raise ValueError(f"{owner}: axis {axis!r} conflicts with semantic dims {semantic_dims!r}.")
    out = candidate[[_DIRECTION_VAR]]
    if core_dim != axis:
        out = out.rename({core_dim: axis})
    out = out.rename({_DIRECTION_VAR: output_var})
    out = out.assign_coords({axis: list(_XYZ_LABELS)})
    return merge_schema(out, {"ext": {"astro": None}}, validate=False), sequence_dim, batch_dims


class TopocentricDirection(TypedAnalysisObject):
    """Backend-neutral topocentric ENU direction payload.

    Parameters
    ----------
    data : object
        AO-like dataset or data array with declared TAL roles and one ENU core
        dimension labelled ``east``, ``north``, ``up``.

    Raises
    ------
    TypeError
        If payload variables are not numeric or the source is not AO-like.
    ValueError
        If roles, core labels, astro metadata, or payload variables are
        malformed.

    Notes
    -----
    Public construction preserves the supplied direction magnitude. It does not
    normalize vectors or validate unit norm so Dask-backed payloads remain lazy.
    Backend operations such as ``tal.astro.sun.direction_to_sun`` produce unit
    vectors and test that invariant at the operation boundary.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.astro import TopocentricDirection
    >>> ds = xr.Dataset(
    ...     {"direction": (("sample", "enu"), [[1.0, 0.0, 0.0]])},
    ...     coords={"sample": [0], "enu": ["east", "north", "up"]},
    ... )
    >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("enu",), validate=True)
    >>> direction = TopocentricDirection(ao)
    >>> sorted(direction.unsafe_data.data_vars)
    ['altitude_deg', 'azimuth_deg', 'direction']
    """

    ENU_LABELS: tuple[str, str, str] = _ENU_LABELS
    LIFECYCLE = TypedLifecycleSpec(
        type_name="TopocentricDirection",
        owner_prefix="astro.TopocentricDirection",
        coerce_source=_coerce_direction_source,
        normalize=_normalize_direction_metadata,
        enforce=_enforce_direction_invariants,
    )

    def to_vector3(
        self,
        *,
        axis: str = "axis",
        output_var: str = "direction",
        validate: bool = True,
    ) -> "Vector3":
        """Convert ENU direction labels to a ``Vector3`` xyz payload.

        Parameters
        ----------
        axis : str, optional
            Output core dimension name for ``x``, ``y``, ``z`` labels.
        output_var : str, optional
            Output vector variable name.
        validate : bool, optional
            Whether to validate the output core schema before returning.

        Returns
        -------
        tal.linalg.Vector3
            Vector3 with ``x=east``, ``y=north``, and ``z=up``.

        Raises
        ------
        ValueError
            If ``axis`` or ``output_var`` is empty, or if ``axis`` conflicts
            with an existing semantic dimension.

        Notes
        -----
        This is a label adapter for ENU sightline math. It preserves direction
        magnitude and metadata topology; it does not normalize the vector.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.astro import TopocentricDirection
        >>> from tal.core import AnalysisObject
        >>> ds = xr.Dataset(
        ...     {"direction": (("sample", "enu"), [[2.0, 3.0, 4.0]])},
        ...     coords={"sample": [0], "enu": ["east", "north", "up"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("enu",), validate=True)
        >>> vector = TopocentricDirection(ao).to_vector3()
        >>> tuple(vector.unsafe_data.coords["axis"].to_numpy().tolist())
        ('x', 'y', 'z')
        """
        owner = "astro.TopocentricDirection.to_vector3"
        resolved_axis = _require_non_empty_string(axis, field="axis", owner=owner)
        resolved_var = _require_non_empty_string(output_var, field="output_var", owner=owner)
        ds, sequence_dim, batch_dims = _vector3_dataset(
            self.unsafe_data,
            axis=resolved_axis,
            output_var=resolved_var,
            owner=owner,
        )
        ao = AnalysisObject.from_data(
            ds,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            core_dims=(resolved_axis,),
            param_coord=read_param_coord_name(self.unsafe_data),
            sequence_size_coord=read_sequence_size_coord_name(self.unsafe_data),
            validate=validate,
        )
        from tal.linalg import Vector3

        if validate:
            return Vector3._from_validated(ao.unsafe_data)
        return Vector3._from_unvalidated(ao.unsafe_data)


__all__ = ["TopocentricDirection"]
