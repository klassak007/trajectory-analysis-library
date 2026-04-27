from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from ..orchestration.alignment import align_exact
from ..orchestration.lazy import require_unchunked_dataarray
from .types import ReducerOp, WeightInput, weighted_supported


_WEIGHTED_UNSUPPORTED = "weights are supported only for reducers {'mean','sum'}."


def _scalar_true(cond: xr.DataArray, *, owner: str, field: str) -> bool:
    require_unchunked_dataarray(
        cond,
        owner=owner,
        field=f"chunked weight validation for {field}",
        guidance="pass unchunked weights or pre-validate weight values explicitly.",
    )
    return bool(np.asarray(cond.data).item())


def _coerce_1d_weight(raw: xr.DataArray | np.ndarray, *, dim: str, var: xr.DataArray, owner: str) -> xr.DataArray:
    if isinstance(raw, xr.DataArray):
        da = raw
    else:
        arr = np.asarray(raw)
        if arr.ndim != 1:
            raise ValueError(f"{owner}: ndarray weights for dim {dim!r} must be 1-D.")
        if int(arr.shape[0]) != int(var.sizes[dim]):
            raise ValueError(
                f"{owner}: ndarray weights length for dim {dim!r} must equal dim size {int(var.sizes[dim])}."
            )
        coord = var.coords[dim] if dim in var.coords else xr.DataArray(np.arange(int(var.sizes[dim])), dims=(dim,))
        da = xr.DataArray(arr, dims=(dim,), coords={dim: coord})
    if tuple(da.dims) != (dim,):
        raise ValueError(f"{owner}: per-dim weights for {dim!r} must have dims ({dim!r},).")
    aligned, _ = align_exact(da, var, exclude=set(), owner=owner, what=f"weights[{dim}] alignment")
    return aligned


def _coerce_mapping_weights(
    mapping: Mapping[str, xr.DataArray | np.ndarray],
    *,
    reduce_dims: tuple[str, ...],
    var: xr.DataArray,
    owner: str,
) -> xr.DataArray:
    missing = tuple(dim for dim in reduce_dims if dim not in mapping)
    if missing:
        raise ValueError(f"{owner}: weights mapping must include every reduced dim; missing={missing!r}.")
    combined: xr.DataArray | None = None
    for dim in reduce_dims:
        part = _coerce_1d_weight(mapping[dim], dim=dim, var=var, owner=owner)
        combined = part if combined is None else (combined * part)
    assert combined is not None
    return combined


def coerce_aligned_weights(
    var: xr.DataArray,
    *,
    reduce_dims: tuple[str, ...],
    weights: WeightInput,
    op: ReducerOp,
    owner: str,
) -> xr.DataArray | None:
    if weights is None:
        return None
    if not weighted_supported(op):
        raise ValueError(f"{owner}: {_WEIGHTED_UNSUPPORTED}")
    if not reduce_dims:
        raise ValueError(f"{owner}: weights require at least one reduced dim.")
    if isinstance(weights, Mapping):
        return _coerce_mapping_weights(weights, reduce_dims=reduce_dims, var=var, owner=owner)
    if isinstance(weights, np.ndarray):
        if len(reduce_dims) != 1:
            raise ValueError(f"{owner}: ndarray weights are only valid for single-dim reduction.")
        return _coerce_1d_weight(weights, dim=reduce_dims[0], var=var, owner=owner)
    if isinstance(weights, xr.DataArray):
        extra = tuple(dim for dim in weights.dims if dim not in reduce_dims)
        if extra:
            raise ValueError(f"{owner}: weight dims must be a subset of reduced dims; extras={extra!r}.")
        aligned, _ = align_exact(weights, var, exclude=set(), owner=owner, what="weights alignment")
        return aligned
    raise TypeError(f"{owner}: weights must be xr.DataArray, ndarray, mapping, or None.")


def _valid_zone_for_weights(weight: xr.DataArray, *, mask: xr.DataArray | None) -> xr.DataArray:
    if mask is None:
        return xr.ones_like(weight, dtype=bool)
    projected = mask
    for dim in tuple(dim for dim in projected.dims if dim not in weight.dims):
        projected = projected.any(dim=dim)
    return projected.broadcast_like(weight)


def validate_weight_values(
    weight: xr.DataArray,
    *,
    mask: xr.DataArray | None,
    skipna: bool,
    owner: str,
) -> xr.DataArray:
    zone = _valid_zone_for_weights(weight, mask=mask)
    negative = (weight < 0) & zone
    if _scalar_true(negative.any(), owner=owner, field="negative weights"):
        raise ValueError(f"{owner}: negative weights are not allowed.")
    nonfinite = (~np.isfinite(weight)) & zone
    if skipna:
        return weight.where(~nonfinite)
    if _scalar_true(nonfinite.any(), owner=owner, field="non-finite weights"):
        raise ValueError(
            f"{owner}: skipna=False requires finite weights on the valid prefix of the reduced domain."
        )
    return weight


def require_no_unsupported_weights(*, weights: WeightInput, op: ReducerOp, owner: str) -> None:
    if weights is None or weighted_supported(op):
        return
    raise ValueError(f"{owner}: {_WEIGHTED_UNSUPPORTED}")


__all__ = [
    "coerce_aligned_weights",
    "require_no_unsupported_weights",
    "validate_weight_values",
]
