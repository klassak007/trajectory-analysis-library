from __future__ import annotations

import xarray as xr

from .types import ReducerOp, WeightInput
from .validity import apply_structural_mask, reduce_missing_on_valid_prefix
from .weights import coerce_aligned_weights, validate_weight_values


_SKIPNA_REDUCERS = frozenset({"mean", "sum", "std", "var", "median", "min", "max"})


def _reduce_unweighted_skipna(
    var: xr.DataArray,
    *,
    op: ReducerOp,
    reduce_dims: tuple[str, ...],
    skipna: bool,
    ddof: int,
    mask: xr.DataArray | None,
) -> xr.DataArray:
    data = apply_structural_mask(var, mask=mask)
    kwargs: dict[str, object] = {"dim": reduce_dims, "keep_attrs": True, "skipna": True}
    if op in {"std", "var"}:
        kwargs["ddof"] = int(ddof)
    result = getattr(data, op)(**kwargs)
    if skipna:
        return result
    poison = reduce_missing_on_valid_prefix(var, reduce_dims=reduce_dims, mask=mask)
    return result.where(~poison)


def _reduce_count(var: xr.DataArray, *, reduce_dims: tuple[str, ...], mask: xr.DataArray | None) -> xr.DataArray:
    data = apply_structural_mask(var, mask=mask)
    return data.count(dim=reduce_dims, keep_attrs=True)


def _reduce_any_all(
    var: xr.DataArray,
    *,
    op: ReducerOp,
    reduce_dims: tuple[str, ...],
    mask: xr.DataArray | None,
) -> xr.DataArray:
    if mask is None:
        return getattr(var, op)(dim=reduce_dims, keep_attrs=True)
    if op == "any":
        neutral = 0 if var.dtype.kind in {"i", "u", "f", "c"} else False
    else:
        neutral = 1 if var.dtype.kind in {"i", "u", "f", "c"} else True
    data = var.where(mask, other=neutral)
    return getattr(data, op)(dim=reduce_dims, keep_attrs=True)


def _reduce_weighted_mean_sum(
    var: xr.DataArray,
    *,
    op: ReducerOp,
    reduce_dims: tuple[str, ...],
    skipna: bool,
    weights: WeightInput,
    mask: xr.DataArray | None,
    owner: str,
) -> xr.DataArray:
    data = apply_structural_mask(var, mask=mask)
    weight = coerce_aligned_weights(var, reduce_dims=reduce_dims, weights=weights, op=op, owner=owner)
    assert weight is not None
    validated = validate_weight_values(weight, mask=mask, skipna=skipna, owner=owner)
    weight_clean = validated.fillna(0) if skipna else validated
    product = (data * weight_clean).where(data.notnull(), other=0).fillna(0)
    numerator = product.sum(dim=reduce_dims, keep_attrs=True, skipna=True)
    if op == "sum":
        if skipna:
            return numerator
        poison = reduce_missing_on_valid_prefix(var, reduce_dims=reduce_dims, mask=mask)
        return numerator.where(~poison)
    denom = weight_clean.where(data.notnull(), other=0).sum(dim=reduce_dims, keep_attrs=True, skipna=True)
    out = numerator / denom.where(denom != 0)
    if skipna:
        return out
    poison = reduce_missing_on_valid_prefix(var, reduce_dims=reduce_dims, mask=mask)
    return out.where(~poison)


def reduce_dataarray(
    var: xr.DataArray,
    *,
    op: ReducerOp,
    reduce_dims: tuple[str, ...],
    skipna: bool,
    ddof: int,
    weights: WeightInput,
    mask: xr.DataArray | None,
    owner: str,
) -> xr.DataArray:
    if not reduce_dims:
        return var
    if op in {"mean", "sum"} and weights is not None:
        return _reduce_weighted_mean_sum(
            var,
            op=op,
            reduce_dims=reduce_dims,
            skipna=skipna,
            weights=weights,
            mask=mask,
            owner=owner,
        )
    if op in _SKIPNA_REDUCERS:
        return _reduce_unweighted_skipna(
            var,
            op=op,
            reduce_dims=reduce_dims,
            skipna=skipna,
            ddof=ddof,
            mask=mask,
        )
    if op == "count":
        return _reduce_count(var, reduce_dims=reduce_dims, mask=mask)
    return _reduce_any_all(var, op=op, reduce_dims=reduce_dims, mask=mask)


__all__ = ["reduce_dataarray"]
