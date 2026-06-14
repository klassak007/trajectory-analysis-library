from __future__ import annotations

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
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.core.typed_lifecycle import TypedAnalysisObject, TypedLifecycleContext, TypedLifecycleSpec

from .metadata import normalize_topocentric_metadata

_DIRECTION_VAR = "direction"
_ALTITUDE_VAR = "altitude_deg"
_AZIMUTH_VAR = "azimuth_deg"
_ENU_LABELS: tuple[str, str, str] = ("east", "north", "up")
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


class TopocentricDirection(TypedAnalysisObject):
    """Backend-neutral topocentric ENU unit direction payload.

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


__all__ = ["TopocentricDirection"]
