from __future__ import annotations

from functools import reduce

import pandas as pd
import xarray as xr

from .topology import ResolvedTopologyPlan, realize_operands_for_plan


def non_core_dims(da: xr.DataArray, *, core_dims: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dim for dim in da.dims if dim not in core_dims)


def align_exact(
    *arrays: xr.DataArray,
    exclude: set[str],
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    try:
        aligned = xr.align(*arrays, join="exact", copy=False, exclude=exclude)
    except ValueError as exc:
        raise ValueError(f"{owner}: {what} inputs are not label-aligned under strict exact policy: {exc}") from exc
    return tuple(aligned)


def _dim_index(
    array: xr.DataArray,
    *,
    dim: str,
) -> pd.Index:
    try:
        return array.get_index(dim)
    except (KeyError, TypeError, ValueError):
        return pd.RangeIndex(int(array.sizes[dim]), name=dim)


def _join_target_index(
    indices: tuple[pd.Index, ...],
    *,
    mode: str,
    owner: str,
    what: str,
    dim: str,
) -> pd.Index:
    if mode == "exact":
        base = indices[0]
        if all(base.equals(index) for index in indices[1:]):
            return base
        raise ValueError(f"{owner}: {what} requires exact {dim!r} labels under sequence/batch policy.")
    if mode == "inner":
        return reduce(lambda left, right: left.intersection(right), indices[1:], indices[0])
    if mode == "outer":
        return reduce(lambda left, right: left.union(right), indices[1:], indices[0])
    if mode == "left":
        return indices[0]
    if mode == "right":
        return indices[-1]
    raise ValueError(f"{owner}: {what} join mode {mode!r} is not supported.")


def _reconcile_dim(
    arrays: tuple[xr.DataArray, ...],
    *,
    dim: str,
    mode: str | None,
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    if mode is None:
        return arrays
    present = tuple(index for index, array in enumerate(arrays) if dim in array.dims)
    if len(present) <= 1:
        return arrays
    target = _join_target_index(
        tuple(_dim_index(arrays[index], dim=dim) for index in present),
        mode=mode,
        owner=owner,
        what=what,
        dim=dim,
    )
    out: list[xr.DataArray] = []
    for index, array in enumerate(arrays):
        if index not in present:
            out.append(array)
            continue
        if mode in {"outer", "left", "right"}:
            out.append(array.reindex({dim: target}))
            continue
        out.append(array.sel({dim: target}))
    return tuple(out)


def _param_index(
    array: xr.DataArray,
    *,
    plan: ResolvedTopologyPlan,
    owner: str,
    what: str,
) -> pd.Index:
    assert plan.sequence_dim is not None
    assert plan.param_coord is not None
    if plan.param_coord not in array.coords:
        raise ValueError(
            f"{owner}: {what} requires param coord {plan.param_coord!r} on every operand for param alignment."
        )
    coord = array.coords[plan.param_coord]
    if tuple(coord.dims) != (plan.sequence_dim,):
        raise ValueError(
            f"{owner}: {what} requires 1-D param coord {plan.param_coord!r} over {plan.sequence_dim!r}; "
            f"got dims={tuple(coord.dims)!r}."
        )
    index = coord.to_index()
    if index.is_unique:
        return index
    raise ValueError(
        f"{owner}: {what} requires unique param labels for param alignment on {plan.param_coord!r}."
    )


def _align_by_param_key(
    arrays: tuple[xr.DataArray, ...],
    *,
    plan: ResolvedTopologyPlan,
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    assert plan.sequence_dim is not None
    assert plan.param_coord is not None
    if plan.sequence_join not in {None, "exact"}:
        raise ValueError(
            f"{owner}: {what} sequence_join={plan.sequence_join!r} would re-key correspondence under on='param'."
        )
    param_indices = tuple(_param_index(array, plan=plan, owner=owner, what=what) for array in arrays)
    target_param = _join_target_index(
        param_indices,
        mode="exact",
        owner=owner,
        what=what,
        dim=plan.param_coord,
    )
    canonical_sequence = _dim_index(arrays[0], dim=plan.sequence_dim)
    swapped = tuple(
        array.swap_dims({plan.sequence_dim: plan.param_coord}).sel({plan.param_coord: target_param})
        for array in arrays
    )
    residual = swapped
    for batch_dim in plan.batch_dims:
        residual = _reconcile_dim(
            residual,
            dim=batch_dim,
            mode=plan.batch_join,
            owner=owner,
            what=what,
        )
    out: list[xr.DataArray] = []
    for array in residual:
        prepared = array.drop_vars(plan.sequence_dim, errors="ignore")
        restored = prepared.rename({plan.param_coord: plan.sequence_dim})
        restored = restored.assign_coords(
            {
                plan.sequence_dim: canonical_sequence,
                plan.param_coord: (plan.sequence_dim, target_param),
            }
        )
        out.append(restored)
    return tuple(out)


def _align_by_primary_key(
    arrays: tuple[xr.DataArray, ...],
    *,
    plan: ResolvedTopologyPlan,
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    if plan.primary_key == "param":
        return _align_by_param_key(arrays, plan=plan, owner=owner, what=what)
    out = arrays
    if plan.sequence_dim is not None:
        out = _reconcile_dim(
            out,
            dim=plan.sequence_dim,
            mode=plan.sequence_join,
            owner=owner,
            what=what,
        )
    for batch_dim in plan.batch_dims:
        out = _reconcile_dim(out, dim=batch_dim, mode=plan.batch_join, owner=owner, what=what)
    return out


def align_exact_for_plan(
    plan: ResolvedTopologyPlan,
    *,
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    arrays = realize_operands_for_plan(plan, owner=owner, what=what)
    arrays = _align_by_primary_key(arrays, plan=plan, owner=owner, what=what)
    return align_exact(
        *arrays,
        exclude=set(plan.align_exclude_dims),
        owner=owner,
        what=what,
    )


__all__ = [
    "align_exact_for_plan",
    "align_exact",
    "non_core_dims",
]
