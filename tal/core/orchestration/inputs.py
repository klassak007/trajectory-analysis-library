from __future__ import annotations

"""Canonical orchestration input coercion boundaries for AO/xarray operands."""

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .alignment_intent import read_alignment_intent
from .broadcast_intent import read_broadcast_intent


def coerce_analysis_object_input(
    value: object,
    *,
    owner: str,
    index: int | None = None,
) -> "AnalysisObject":
    """Coerce one AO-like input for orchestration boundaries.

    Parameters
    ----------
    value : object
        Input value to normalize. Existing ``AnalysisObject`` instances are
        returned unchanged. ``xarray.Dataset`` and ``xarray.DataArray`` inputs
        are wrapped in ``AnalysisObject``.
    owner : str
        Public owner string used to build deterministic diagnostics.
    index : int | None, optional
        Operand index used in variadic boundary diagnostics.

    Returns
    -------
    AnalysisObject
        Coerced AO-like input.

    Raises
    ------
    TypeError
        If ``value`` is not an ``AnalysisObject``, ``xarray.Dataset``, or
        ``xarray.DataArray``.

    Notes
    -----
    Existing AO inputs are also checked for pending alignment or broadcast
    intents so invalid intent metadata fails at the operation boundary that
    consumes it.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.inputs import coerce_analysis_object_input
    >>> ds = xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]})
    >>> ao = coerce_analysis_object_input(ds, owner="thermal.Temperature.__init__")
    >>> isinstance(ao, AnalysisObject)
    True
    """
    from ..analysis_object import AnalysisObject

    if isinstance(value, AnalysisObject):
        label = "input" if index is None else f"operand {index}"
        _ = read_broadcast_intent(value, owner=owner, label=label)
        _ = read_alignment_intent(value, owner=owner, label=label)
        return value
    if isinstance(value, (xr.Dataset, xr.DataArray)):
        return AnalysisObject(value)
    expected = "AnalysisObject, xr.Dataset, or xr.DataArray"
    if index is None:
        raise TypeError(f"{owner}: expected {expected}; got {type(value).__name__}.")
    raise TypeError(
        f"{owner}: invalid input at index {index}; expected {expected}; got {type(value).__name__}."
    )


def normalize_analysis_object_inputs(
    values: Sequence[object],
    *,
    owner: str,
    require_nonempty: bool = True,
) -> list["AnalysisObject"]:
    """Normalize a sequence of AO-like values into AnalysisObject instances.

    Parameters
    ----------
    values : Sequence[object]
        Input values consumed by this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    require_nonempty : bool, optional
        Behavior flag/policy controlling boundary semantics.

    Returns
    -------
    list['AnalysisObject']
        Ordered collection produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if require_nonempty and not values:
        raise ValueError(f"{owner}: expected at least one input.")
    out: list[AnalysisObject] = []
    for idx, value in enumerate(values):
        out.append(coerce_analysis_object_input(value, owner=owner, index=idx))
    return out


def coerce_operand(
    value: object,
    *,
    owner: str,
    label: str | None = None,
    role: str = "operand",
    allow_scalar: bool = False,
    return_scalar_none: bool = False,
) -> object | None:
    """Coerce one user-facing operation operand.

    Parameters
    ----------
    value : object
        Operand supplied by user code.
    owner : str
        Public owner string used to build deterministic diagnostics.
    label : str | None, optional
        Operand label inserted before ``role`` in error messages.
    role : str, optional
        Diagnostic noun for this operand. Defaults to ``"operand"``.
    allow_scalar : bool, optional
        Whether NumPy scalar-like values are accepted.
    return_scalar_none : bool, optional
        When ``True`` and a scalar is accepted, return ``None`` instead of the
        scalar value. This is useful for scalar-or-AO operations that handle the
        scalar path separately.

    Returns
    -------
    object | None
        ``AnalysisObject`` for AO-like inputs, a scalar for accepted scalar
        inputs, or ``None`` when ``return_scalar_none`` is selected.

    Raises
    ------
    TypeError
        If the operand does not match the AO-like or scalar policy.

    Notes
    -----
    Use this helper at public operation boundaries so diagnostics include both
    the public owner and the operand label. Internal AO-only plumbing should use
    ``coerce_analysis_object_input(...)`` directly.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.inputs import coerce_operand
    >>> ds = xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]})
    >>> operand = coerce_operand(ds, owner="thermal.bias_temperature", label="temperature")
    >>> isinstance(operand, AnalysisObject)
    True
    >>> coerce_operand(1.5, owner="thermal.bias_temperature", label="bias", allow_scalar=True)
    1.5
    """
    if allow_scalar and np.isscalar(value):
        return None if return_scalar_none else value
    try:
        return coerce_analysis_object_input(value, owner=owner)
    except TypeError as exc:
        item = role if label is None else f"{label} {role}"
        expected = "AnalysisObject, xr.Dataset, or xr.DataArray"
        if allow_scalar:
            expected = f"{expected}, or scalar"
        raise TypeError(
            f"{owner}: invalid {item} ({type(value).__name__}); expected {expected}."
        ) from exc


def dataset_from_other_input(
    other: "AnalysisObject | xr.Dataset | xr.DataArray",
    *,
    owner: str,
) -> xr.Dataset:
    """Resolve `other` input into a Dataset for query-coordinate extraction.

    Parameters
    ----------
    other : AnalysisObject | xr.Dataset | xr.DataArray
        Secondary operand combined with the receiver/source operand.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    from ..analysis_object import AnalysisObject

    if isinstance(other, AnalysisObject):
        return other.unsafe_data
    if isinstance(other, xr.Dataset):
        return other
    if isinstance(other, xr.DataArray):
        return other.to_dataset(name=other.name or "__grid__")
    raise TypeError(
        f"{owner}: expected AnalysisObject, xarray.Dataset, or xarray.DataArray."
    )


def _coerce_query_numeric(query: xr.DataArray, *, owner: str) -> xr.DataArray:
    try:
        return query.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: query values must be numeric (coercible to float64).") from exc


def query_coord_from_other_input(
    other: "AnalysisObject | xr.Dataset | xr.DataArray",
    *,
    coord_name: str,
    owner: str,
) -> xr.DataArray:
    """Extract and numeric-normalize a query coordinate from `other`.

    Parameters
    ----------
    other : AnalysisObject | xr.Dataset | xr.DataArray
        Secondary operand combined with the receiver/source operand.
    coord_name : str, optional
        Coordinate name/value used by this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if isinstance(other, xr.DataArray):
        if other.name == coord_name:
            return _coerce_query_numeric(other, owner=owner)
        if coord_name in other.coords:
            return _coerce_query_numeric(other.coords[coord_name], owner=owner)
        raise ValueError(f"{owner}: could not find coord {coord_name!r} on DataArray input.")
    ds = dataset_from_other_input(other, owner=owner)
    if coord_name not in ds.coords:
        raise ValueError(f"{owner}: coord {coord_name!r} not found on other object.")
    return _coerce_query_numeric(ds.coords[coord_name], owner=owner)


__all__ = [
    "coerce_operand",
    "coerce_analysis_object_input",
    "dataset_from_other_input",
    "normalize_analysis_object_inputs",
    "query_coord_from_other_input",
]
