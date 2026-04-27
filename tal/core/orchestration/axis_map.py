from __future__ import annotations

"""Semantic role-axis mapping helpers for cross-object orchestration paths."""

import xarray as xr

from ..param_ops.types import ParamRuntimeContext


def _lock_name_matched_batch_dims(
    src_batch: tuple[str, ...],
    *,
    candidates: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    locked: dict[str, str] = {}
    for src in src_batch:
        if src in candidates[src]:
            locked[src] = src
    return locked


def _select_unique_batch_mapping(
    src_batch: tuple[str, ...],
    *,
    candidates: dict[str, tuple[str, ...]],
    owner: str,
) -> dict[str, str]:
    ordered = sorted(src_batch, key=lambda dim: (len(candidates[dim]), dim))
    solutions: list[dict[str, str]] = []

    def _search(i: int, used: set[str], current: dict[str, str]) -> None:
        if len(solutions) > 1:
            return
        if i == len(ordered):
            solutions.append(dict(current))
            return
        src = ordered[i]
        for dst in candidates[src]:
            if dst in used:
                continue
            current[src] = dst
            used.add(dst)
            _search(i + 1, used, current)
            used.remove(dst)
            del current[src]

    _search(0, set(), {})
    if len(solutions) == 1:
        return solutions[0]
    if not solutions:
        raise ValueError(f"{owner}: AO operand batch topology is not representable under label-safe mapping.")
    raise ValueError(
        f"{owner}: AO operand batch mapping is ambiguous under labels; "
        "use unambiguous batch labels or matching batch-dim names."
    )


def _resolve_batch_dim_map(
    src_batch: tuple[str, ...],
    dst_batch: tuple[str, ...],
    *,
    source_clock: xr.DataArray,
    target_runtime: ParamRuntimeContext,
    owner: str,
) -> dict[str, str]:
    if not src_batch:
        return {}
    if src_batch == dst_batch:
        return {dim: dim for dim in src_batch}
    src_idx = {dim: source_clock.get_index(dim) for dim in src_batch}
    dst_idx = {dim: target_runtime.ds.get_index(dim) for dim in dst_batch}
    candidates = {
        src: tuple(dst for dst in dst_batch if src_idx[src].equals(dst_idx[dst]))
        for src in src_batch
    }
    missing = [src for src, matches in candidates.items() if not matches]
    if missing:
        raise ValueError(
            f"{owner}: AO operand batch labels are incompatible for dims {missing!r}; "
            "batch-dim indexes must be label-equivalent under a one-to-one mapping."
        )
    locked = _lock_name_matched_batch_dims(src_batch, candidates=candidates)
    unresolved = tuple(src for src in src_batch if src not in locked)
    if not unresolved:
        return locked
    used = set(locked.values())
    unresolved_candidates = {
        src: tuple(dst for dst in candidates[src] if dst not in used)
        for src in unresolved
    }
    blocked = [src for src, matches in unresolved_candidates.items() if not matches]
    if blocked:
        raise ValueError(f"{owner}: AO operand batch topology is not representable under label-safe mapping.")
    resolved = _select_unique_batch_mapping(unresolved, candidates=unresolved_candidates, owner=owner)
    return {**locked, **resolved}


def resolve_role_axis_map(
    source_runtime: ParamRuntimeContext,
    target_runtime: ParamRuntimeContext,
    *,
    source_clock: xr.DataArray,
    owner: str,
) -> dict[str, str]:
    src_batch = source_runtime.batch_dims
    dst_batch = target_runtime.batch_dims
    if len(src_batch) != len(dst_batch):
        raise ValueError(
            f"{owner}: AO operand batch topology is incompatible with context "
            f"({src_batch!r} vs {dst_batch!r})."
        )
    out = _resolve_batch_dim_map(
        src_batch,
        dst_batch,
        source_clock=source_clock,
        target_runtime=target_runtime,
        owner=owner,
    )
    out[source_runtime.sequence_dim] = target_runtime.sequence_dim
    if len(set(out.values())) != len(out):
        raise ValueError(f"{owner}: AO operand role dims are not injectively mappable to context.")
    return out


__all__ = ["resolve_role_axis_map"]

