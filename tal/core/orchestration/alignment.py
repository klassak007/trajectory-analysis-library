from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import pandas as pd
import xarray as xr

from .topology import ResolvedTopologyPlan, realize_operands_for_plan


@dataclass(frozen=True)
class _DimIndexState:
    labels: pd.Index
    has_coord: bool
    has_xindex: bool
    used_positional_fallback: bool


@dataclass(frozen=True)
class _ParamAlignmentState:
    keyed: tuple[xr.DataArray, ...]
    canonical: tuple[bool, ...]
    target_param: pd.Index
    canonical_sequence: pd.Index
    batch_targets: tuple[tuple[str, pd.Index | None], ...]


@dataclass(frozen=True)
class _ParamKeyTargets:
    indices: tuple[pd.Index, ...]
    param: pd.Index
    sequence: pd.Index


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


def _dim_index_state(
    array: xr.DataArray,
    *,
    dim: str,
) -> _DimIndexState:
    has_xindex = dim in array.xindexes
    try:
        labels = array.get_index(dim)
        used_fallback = not has_xindex
    except (KeyError, TypeError, ValueError):
        labels = pd.RangeIndex(int(array.sizes[dim]), name=dim)
        used_fallback = True
    return _DimIndexState(
        labels=labels,
        has_coord=dim in array.coords,
        has_xindex=has_xindex,
        used_positional_fallback=used_fallback,
    )


def _dim_index(
    array: xr.DataArray,
    *,
    dim: str,
) -> pd.Index:
    return _dim_index_state(array, dim=dim).labels


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


def _dim_target(
    arrays: tuple[xr.DataArray, ...],
    *,
    dim: str,
    mode: str | None,
    owner: str,
    what: str,
) -> pd.Index | None:
    if mode is None:
        return None
    indices = tuple(_dim_index(array, dim=dim) for array in arrays if dim in array.dims)
    if len(indices) <= 1:
        return None
    return _join_target_index(indices, mode=mode, owner=owner, what=what, dim=dim)


def _align_to_dim_target(
    array: xr.DataArray,
    *,
    dim: str,
    mode: str,
    target: pd.Index,
) -> xr.DataArray:
    if dim not in array.dims or _dim_index(array, dim=dim).equals(target):
        return array
    if mode in {"outer", "left", "right"}:
        return array.reindex({dim: target})
    return array.sel({dim: target})


def _apply_dim_target(
    arrays: tuple[xr.DataArray, ...],
    *,
    dim: str,
    mode: str,
    target: pd.Index,
) -> tuple[xr.DataArray, ...]:
    return tuple(
        _align_to_dim_target(array, dim=dim, mode=mode, target=target)
        for array in arrays
    )


def _reconcile_dim(
    arrays: tuple[xr.DataArray, ...],
    *,
    dim: str,
    mode: str | None,
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    target = _dim_target(arrays, dim=dim, mode=mode, owner=owner, what=what)
    if mode is None or target is None:
        return arrays
    return _apply_dim_target(arrays, dim=dim, mode=mode, target=target)


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


def _batch_targets(
    arrays: tuple[xr.DataArray, ...],
    *,
    plan: ResolvedTopologyPlan,
    owner: str,
    what: str,
) -> tuple[tuple[str, pd.Index | None], ...]:
    return tuple(
        (
            dim,
            _dim_target(arrays, dim=dim, mode=plan.batch_join, owner=owner, what=what),
        )
        for dim in plan.batch_dims
    )


def _batch_indexes_match(
    array: xr.DataArray,
    *,
    targets: tuple[tuple[str, pd.Index | None], ...],
) -> bool:
    for dim, target in targets:
        if target is None or dim not in array.dims:
            continue
        if not _dim_index(array, dim=dim).equals(target):
            return False
    return True


def _batch_index_state_is_stable(array: xr.DataArray, *, dim: str) -> bool:
    if dim not in array.dims:
        return True
    state = _dim_index_state(array, dim=dim)
    if not state.has_coord:
        return True
    return state.has_xindex and not state.used_positional_fallback


def _batch_index_states_are_stable(
    array: xr.DataArray,
    *,
    batch_dims: tuple[str, ...],
) -> bool:
    return all(_batch_index_state_is_stable(array, dim=dim) for dim in batch_dims)


def _param_coord_order_is_canonical(
    array: xr.DataArray,
    *,
    sequence_dim: str,
    param_coord: str,
) -> bool:
    key_names = {sequence_dim, param_coord}
    non_key = tuple(name for name in array.coords if name not in key_names)
    return tuple(array.coords) == (*non_key, sequence_dim, param_coord)


def _sequence_index_state_is_canonical(
    array: xr.DataArray,
    *,
    sequence_dim: str,
    target: pd.Index,
) -> bool:
    state = _dim_index_state(array, dim=sequence_dim)
    return (
        state.has_coord
        and state.has_xindex
        and not state.used_positional_fallback
        and state.labels.equals(target)
    )


def _param_operand_structure_is_canonical(
    array: xr.DataArray,
    *,
    param_index: pd.Index,
    target_param: pd.Index,
    canonical_sequence: pd.Index,
    plan: ResolvedTopologyPlan,
) -> bool:
    assert plan.sequence_dim is not None
    assert plan.param_coord is not None
    if not param_index.equals(target_param):
        return False
    if plan.param_coord in array.xindexes:
        return False
    if not _sequence_index_state_is_canonical(
        array,
        sequence_dim=plan.sequence_dim,
        target=canonical_sequence,
    ):
        return False
    if not _batch_index_states_are_stable(array, batch_dims=plan.batch_dims):
        return False
    return _param_coord_order_is_canonical(
        array,
        sequence_dim=plan.sequence_dim,
        param_coord=plan.param_coord,
    )


def _select_by_param(
    array: xr.DataArray,
    *,
    plan: ResolvedTopologyPlan,
    target_param: pd.Index,
) -> xr.DataArray:
    assert plan.sequence_dim is not None
    assert plan.param_coord is not None
    return array.swap_dims({plan.sequence_dim: plan.param_coord}).sel({plan.param_coord: target_param})


def _restore_sequence_dim(
    array: xr.DataArray,
    *,
    plan: ResolvedTopologyPlan,
    target_param: pd.Index,
    canonical_sequence: pd.Index,
) -> xr.DataArray:
    assert plan.sequence_dim is not None
    assert plan.param_coord is not None
    prepared = array.drop_vars(plan.sequence_dim, errors="ignore")
    restored = prepared.rename({plan.param_coord: plan.sequence_dim})
    return restored.assign_coords(
        {
            plan.sequence_dim: canonical_sequence,
            plan.param_coord: (plan.sequence_dim, target_param),
        }
    )


def _structurally_canonical_param_operands(
    arrays: tuple[xr.DataArray, ...],
    *,
    param_indices: tuple[pd.Index, ...],
    target_param: pd.Index,
    canonical_sequence: pd.Index,
    plan: ResolvedTopologyPlan,
) -> tuple[bool, ...]:
    return tuple(
        _param_operand_structure_is_canonical(
            array,
            param_index=param_index,
            target_param=target_param,
            canonical_sequence=canonical_sequence,
            plan=plan,
        )
        for array, param_index in zip(arrays, param_indices, strict=True)
    )


def _select_param_operands(
    arrays: tuple[xr.DataArray, ...],
    *,
    canonical: tuple[bool, ...],
    plan: ResolvedTopologyPlan,
    target_param: pd.Index,
) -> tuple[xr.DataArray, ...]:
    return tuple(
        array if is_canonical else _select_by_param(array, plan=plan, target_param=target_param)
        for array, is_canonical in zip(arrays, canonical, strict=True)
    )


def _batch_canonical_param_operands(
    arrays: tuple[xr.DataArray, ...],
    *,
    structurally_canonical: tuple[bool, ...],
    targets: tuple[tuple[str, pd.Index | None], ...],
) -> tuple[bool, ...]:
    return tuple(
        structural and _batch_indexes_match(array, targets=targets)
        for array, structural in zip(arrays, structurally_canonical, strict=True)
    )


def _select_new_batch_mismatches(
    source: tuple[xr.DataArray, ...],
    keyed: tuple[xr.DataArray, ...],
    *,
    structurally_canonical: tuple[bool, ...],
    canonical: tuple[bool, ...],
    plan: ResolvedTopologyPlan,
    target_param: pd.Index,
) -> tuple[xr.DataArray, ...]:
    out: list[xr.DataArray] = []
    states = zip(source, keyed, structurally_canonical, canonical, strict=True)
    for original, current, structural, final in states:
        if not structural or final:
            out.append(current)
        else:
            out.append(_select_by_param(original, plan=plan, target_param=target_param))
    return tuple(out)


def _apply_batch_targets(
    arrays: tuple[xr.DataArray, ...],
    *,
    mode: str,
    targets: tuple[tuple[str, pd.Index | None], ...],
) -> tuple[xr.DataArray, ...]:
    out = arrays
    for dim, target in targets:
        if target is not None:
            out = _apply_dim_target(out, dim=dim, mode=mode, target=target)
    return out


def _restore_param_operands(
    arrays: tuple[xr.DataArray, ...],
    *,
    canonical: tuple[bool, ...],
    plan: ResolvedTopologyPlan,
    target_param: pd.Index,
    canonical_sequence: pd.Index,
) -> tuple[xr.DataArray, ...]:
    return tuple(
        array
        if is_canonical
        else _restore_sequence_dim(
            array,
            plan=plan,
            target_param=target_param,
            canonical_sequence=canonical_sequence,
        )
        for array, is_canonical in zip(arrays, canonical, strict=True)
    )


def _param_key_targets(
    arrays: tuple[xr.DataArray, ...],
    *,
    plan: ResolvedTopologyPlan,
    owner: str,
    what: str,
) -> _ParamKeyTargets:
    assert plan.sequence_dim is not None
    assert plan.param_coord is not None
    indices = tuple(_param_index(array, plan=plan, owner=owner, what=what) for array in arrays)
    param = _join_target_index(
        indices,
        mode="exact",
        owner=owner,
        what=what,
        dim=plan.param_coord,
    )
    sequence = _dim_index(arrays[0], dim=plan.sequence_dim)
    return _ParamKeyTargets(indices=indices, param=param, sequence=sequence)


def _plan_param_alignment(
    arrays: tuple[xr.DataArray, ...],
    *,
    plan: ResolvedTopologyPlan,
    owner: str,
    what: str,
) -> _ParamAlignmentState:
    targets = _param_key_targets(arrays, plan=plan, owner=owner, what=what)
    structural = _structurally_canonical_param_operands(
        arrays,
        param_indices=targets.indices,
        target_param=targets.param,
        canonical_sequence=targets.sequence,
        plan=plan,
    )
    keyed = _select_param_operands(
        arrays,
        canonical=structural,
        plan=plan,
        target_param=targets.param,
    )
    batch_targets = _batch_targets(keyed, plan=plan, owner=owner, what=what)
    canonical = _batch_canonical_param_operands(
        keyed,
        structurally_canonical=structural,
        targets=batch_targets,
    )
    keyed = _select_new_batch_mismatches(
        arrays,
        keyed,
        structurally_canonical=structural,
        canonical=canonical,
        plan=plan,
        target_param=targets.param,
    )
    return _ParamAlignmentState(
        keyed=keyed,
        canonical=canonical,
        target_param=targets.param,
        canonical_sequence=targets.sequence,
        batch_targets=batch_targets,
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
    state = _plan_param_alignment(arrays, plan=plan, owner=owner, what=what)
    residual = _apply_batch_targets(
        state.keyed,
        mode=plan.batch_join,
        targets=state.batch_targets,
    )
    return _restore_param_operands(
        residual,
        canonical=state.canonical,
        plan=plan,
        target_param=state.target_param,
        canonical_sequence=state.canonical_sequence,
    )


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
    "align_exact",
    "align_exact_for_plan",
    "non_core_dims",
]
