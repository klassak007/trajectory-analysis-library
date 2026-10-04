from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from ..orchestration.alignment import align_exact
from ..orchestration.indexing import isel_rows
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
        da = xr.DataArray(arr, dims=(dim,))
    if tuple(da.dims) != (dim,):
        raise ValueError(f"{owner}: per-dim weights for {dim!r} must have dims ({dim!r},).")
    aligned, _ = align_exact(da, var, exclude=set(), owner=owner, what=f"weights[{dim}] alignment")
    return aligned


def _coerce_mapping_weight_factors(
    mapping: Mapping[str, xr.DataArray | np.ndarray],
    *,
    reduce_dims: tuple[str, ...],
    var: xr.DataArray,
    owner: str,
) -> tuple[xr.DataArray, ...]:
    missing = tuple(dim for dim in reduce_dims if dim not in mapping)
    if missing:
        raise ValueError(f"{owner}: weights mapping must include every reduced dim; missing={missing!r}.")
    return tuple(
        _coerce_1d_weight(mapping[dim], dim=dim, var=var, owner=owner)
        for dim in reduce_dims
    )


def _aligned_weight_factors(
    var: xr.DataArray,
    *,
    reduce_dims: tuple[str, ...],
    weights: WeightInput,
    op: ReducerOp,
    owner: str,
) -> tuple[xr.DataArray, ...]:
    if weights is None:
        return ()
    if not weighted_supported(op):
        raise ValueError(f"{owner}: {_WEIGHTED_UNSUPPORTED}")
    if not reduce_dims:
        raise ValueError(f"{owner}: weights require at least one reduced dim.")
    if isinstance(weights, Mapping):
        return _coerce_mapping_weight_factors(
            weights,
            reduce_dims=reduce_dims,
            var=var,
            owner=owner,
        )
    if isinstance(weights, np.ndarray):
        if len(reduce_dims) != 1:
            raise ValueError(f"{owner}: ndarray weights are only valid for single-dim reduction.")
        return (_coerce_1d_weight(weights, dim=reduce_dims[0], var=var, owner=owner),)
    if isinstance(weights, xr.DataArray):
        extra = tuple(dim for dim in weights.dims if dim not in reduce_dims)
        if extra:
            raise ValueError(f"{owner}: weight dims must be a subset of reduced dims; extras={extra!r}.")
        aligned, _ = align_exact(
            weights,
            var,
            exclude=set(),
            owner=owner,
            what="weights alignment",
        )
        return (aligned,)
    raise TypeError(f"{owner}: weights must be xr.DataArray, ndarray, mapping, or None.")


def coerce_aligned_weights(
    var: xr.DataArray,
    *,
    reduce_dims: tuple[str, ...],
    weights: WeightInput,
    mask: xr.DataArray | None,
    op: ReducerOp,
    owner: str,
) -> xr.DataArray | None:
    factors = _aligned_weight_factors(
        var,
        reduce_dims=reduce_dims,
        weights=weights,
        op=op,
        owner=owner,
    )
    if not factors:
        return None
    # Empty payloads have no participating factor entries. Alignment above
    # still validates every consumed factor, but their value graphs must not
    # enter numerical topology that cannot consume them.
    if var.size == 0:
        return xr.DataArray(1.0)
    if isinstance(weights, Mapping):
        for factor in factors:
            zone = _valid_zone_for_weights(factor, mask=mask)
            _require_nonnegative_weights(factor, zone=zone, owner=owner)
    combined = factors[0]
    for factor in factors[1:]:
        combined = combined * factor
    return combined


def _active_reduce_dims(var: xr.DataArray, reduce_dims: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dim for dim in reduce_dims if dim in var.dims)


def _require_active_weight_dims(
    active_dims: tuple[str, ...],
    *,
    owner: str,
) -> None:
    if active_dims:
        return
    raise ValueError(
        f"{owner}: weights require at least one reduced payload dimension."
    )


def _preflight_1d_weight(
    raw: object,
    *,
    dim: str,
    size: int,
    owner: str,
) -> None:
    if isinstance(raw, xr.DataArray):
        if tuple(raw.dims) != (dim,):
            raise ValueError(f"{owner}: per-dim weights for {dim!r} must have dims ({dim!r},).")
        if int(raw.sizes[dim]) != size:
            raise ValueError(
                f"{owner}: DataArray weights length for dim {dim!r} must equal "
                f"dim size {size}."
            )
        return
    if not isinstance(raw, np.ndarray):
        raise TypeError(f"{owner}: mapping weights must contain xr.DataArray or ndarray values.")
    if raw.ndim != 1:
        raise ValueError(f"{owner}: ndarray weights for dim {dim!r} must be 1-D.")
    if int(raw.shape[0]) != size:
        raise ValueError(
            f"{owner}: ndarray weights length for dim {dim!r} must equal dim size {size}."
        )


def _preflight_mapping_weights(
    mapping: Mapping[object, object],
    *,
    ds: xr.Dataset,
    active_dims: tuple[str, ...],
    owner: str,
) -> None:
    _require_active_weight_dims(active_dims, owner=owner)
    missing = tuple(dim for dim in active_dims if dim not in mapping)
    if missing:
        raise ValueError(
            f"{owner}: weights mapping must include every reduced dim; missing={missing!r}."
        )
    for dim in active_dims:
        _preflight_1d_weight(
            mapping[dim],
            dim=dim,
            size=int(ds.sizes[dim]),
            owner=owner,
        )


def _preflight_ndarray_weights(
    weights: np.ndarray,
    *,
    ds: xr.Dataset,
    active_dims: tuple[str, ...],
    owner: str,
) -> None:
    _require_active_weight_dims(active_dims, owner=owner)
    if len(active_dims) != 1:
        raise ValueError(
            f"{owner}: ndarray weights require exactly one active reduced "
            f"payload dimension; got {active_dims!r}."
        )
    dim = active_dims[0]
    _preflight_1d_weight(
        weights,
        dim=dim,
        size=int(ds.sizes[dim]),
        owner=owner,
    )


def _preflight_dataarray_weight_for_var(
    weights: xr.DataArray,
    *,
    var: xr.DataArray,
    reduce_dims: tuple[str, ...],
    owner: str,
) -> None:
    active = _active_reduce_dims(var, reduce_dims)
    if not active:
        return
    extra = tuple(dim for dim in weights.dims if dim not in active)
    if extra:
        raise ValueError(f"{owner}: weight dims must be a subset of reduced dims; extras={extra!r}.")
    for dim in weights.dims:
        if int(weights.sizes[dim]) != int(var.sizes[dim]):
            raise ValueError(
                f"{owner}: DataArray weights length for dim {dim!r} must "
                f"equal dim size {int(var.sizes[dim])}."
            )


def _preflight_dataarray_weights(
    weights: xr.DataArray,
    *,
    ds: xr.Dataset,
    names: tuple[str, ...],
    reduce_dims: tuple[str, ...],
    active_dims: tuple[str, ...],
    owner: str,
) -> None:
    _require_active_weight_dims(active_dims, owner=owner)
    for name in names:
        _preflight_dataarray_weight_for_var(
            weights, var=ds[name], reduce_dims=reduce_dims, owner=f"{owner}.{name}"
        )


def _slice_dim_weight(
    weight: xr.DataArray | np.ndarray,
    *,
    dim: str,
    rows: np.ndarray,
) -> xr.DataArray | np.ndarray:
    if isinstance(weight, xr.DataArray):
        if dim not in weight.dims:
            return weight
        selected = isel_rows(weight, dim=dim, rows=rows)
        assert isinstance(selected, xr.DataArray)
        return selected
    return weight[rows]


def slice_weights_for_rows(
    weights: WeightInput,
    *,
    dim: str,
    rows: np.ndarray,
    ndarray_dims: tuple[str, ...],
) -> WeightInput:
    """Select one grouped row partition without realizing lazy weights."""
    if weights is None:
        return None
    if isinstance(weights, Mapping):
        return {
            name: _slice_dim_weight(value, dim=dim, rows=rows)
            if name == dim
            else value
            for name, value in weights.items()
        }
    if isinstance(weights, xr.DataArray):
        return _slice_dim_weight(weights, dim=dim, rows=rows)
    return {
        name: weights[rows] if name == dim else weights
        for name in ndarray_dims
    }


def require_weight_alignment(
    ds: xr.Dataset,
    *,
    names: tuple[str, ...],
    reduce_dims: tuple[str, ...],
    weights: WeightInput,
    op: ReducerOp,
    owner: str,
) -> None:
    """Validate full-domain weight indexes after deferred key realization."""
    if weights is None:
        return
    for name in names:
        active = _active_reduce_dims(ds[name], reduce_dims)
        if active:
            _aligned_weight_factors(
                ds[name],
                reduce_dims=active,
                weights=weights,
                op=op,
                owner=f"{owner}.{name}",
            )


def preflight_weight_structure(
    ds: xr.Dataset,
    *,
    names: tuple[str, ...],
    reduce_dims: tuple[str, ...],
    active_dims: tuple[str, ...],
    weights: WeightInput,
    owner: str,
) -> None:
    """Validate metadata-only weight structure without alignment or realization."""
    if weights is None:
        return
    if isinstance(weights, Mapping):
        _preflight_mapping_weights(
            weights,
            ds=ds,
            active_dims=active_dims,
            owner=owner,
        )
        return
    if isinstance(weights, np.ndarray):
        _preflight_ndarray_weights(
            weights,
            ds=ds,
            active_dims=active_dims,
            owner=owner,
        )
        return
    if isinstance(weights, xr.DataArray):
        _preflight_dataarray_weights(
            weights,
            ds=ds,
            names=names,
            reduce_dims=reduce_dims,
            active_dims=active_dims,
            owner=owner,
        )
        return
    raise TypeError(f"{owner}: weights must be xr.DataArray, ndarray, mapping, or None.")


def _valid_zone_for_weights(weight: xr.DataArray, *, mask: xr.DataArray | None) -> xr.DataArray:
    if mask is None:
        return xr.ones_like(weight, dtype=bool)
    projected = mask
    for dim in tuple(dim for dim in projected.dims if dim not in weight.dims):
        projected = projected.any(dim=dim)
    return projected


def _require_nonnegative_weights(
    weight: xr.DataArray,
    *,
    zone: xr.DataArray,
    owner: str,
) -> None:
    negative = (weight < 0) & zone
    if _scalar_true(negative.any(), owner=owner, field="negative weights"):
        raise ValueError(f"{owner}: negative weights are not allowed.")


def validate_weight_values(
    weight: xr.DataArray,
    *,
    mask: xr.DataArray | None,
    payload_has_entries: bool,
    skipna: bool,
    owner: str,
) -> xr.DataArray:
    if not payload_has_entries:
        return weight
    zone = _valid_zone_for_weights(weight, mask=mask)
    _require_nonnegative_weights(weight, zone=zone, owner=owner)
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
    "preflight_weight_structure",
    "require_no_unsupported_weights",
    "require_weight_alignment",
    "slice_weights_for_rows",
    "validate_weight_values",
]
