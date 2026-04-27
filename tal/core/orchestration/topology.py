from __future__ import annotations

"""Shared batch/sequence topology orchestration wrappers.

This module is the orchestration-level owner for flatten/query/restore and
batch-index join wiring. It composes existing `param_ops` topology primitives
without duplicating their algorithmic behavior.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import xarray as xr

from .topology_batch import (
    align_combine_batch_axis,
    allocate_flat_batch_dim_name,
    batch_index_for_dataset,
    flatten_param_contexts,
    flatten_query_for_batch_plan,
    join_batch_indices,
    join_combine_batch_labels,
    restore_combine_batch_axis,
    restore_dataset_batch_topology,
    restore_dataset_multi_batch,
    stack_combine_batch_axis,
)

@dataclass(frozen=True)
class SemanticTopology:
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]

@dataclass(frozen=True)
class TopologyPolicy:
    mode: Literal["strict", "semantic_broadcast"] = "strict"
    core_dim_alignment: Literal["exact", "exclude_all_core"] = "exact"
    alignment_on: Literal["sequence", "param", "auto"] = "sequence"
    sequence_join: Literal["exact", "inner", "outer", "left", "right"] | None = "exact"
    batch_join: Literal["exact", "inner", "outer", "left", "right"] = "exact"
    core_policy: Literal["strict", "numpy_named"] = "strict"
    strict_core_match_required: bool = False

@dataclass(frozen=True)
class TopologyOperand:
    index: int
    data: xr.DataArray
    semantic: SemanticTopology
    param_coord: str | None = None

@dataclass(frozen=True)
class ResolvedTopologyPlan:
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    operands: tuple[TopologyOperand, ...]
    align_exclude_dims: frozenset[str]
    mode: Literal["strict", "semantic_broadcast"] = "strict"
    primary_key: Literal["sequence", "param"] = "sequence"
    sequence_join: Literal["exact", "inner", "outer", "left", "right"] | None = "exact"
    batch_join: Literal["exact", "inner", "outer", "left", "right"] = "exact"
    core_policy: Literal["strict", "numpy_named"] = "strict"
    param_coord: str | None = None
    output_core_dims: tuple[str, ...] = ()

STRICT_EXACT_POLICY = TopologyPolicy(mode="strict", core_dim_alignment="exact")
STRICT_NON_CORE_POLICY = TopologyPolicy(mode="strict", core_dim_alignment="exclude_all_core")
SEMANTIC_EXACT_POLICY = TopologyPolicy(mode="semantic_broadcast", core_dim_alignment="exact")
SEMANTIC_NON_CORE_POLICY = TopologyPolicy(
    mode="semantic_broadcast",
    core_dim_alignment="exclude_all_core",
)

_JOIN_POLICIES = frozenset({"exact", "inner", "outer", "left", "right"})
_ALIGNMENT_KEYS = frozenset({"sequence", "param", "auto"})
_CORE_POLICIES = frozenset({"strict", "numpy_named"})

def _normalize_policy(
    policy: TopologyPolicy | None,
    *,
    owner: str,
) -> TopologyPolicy:
    resolved = policy if policy is not None else STRICT_EXACT_POLICY
    if resolved.mode not in {"strict", "semantic_broadcast"}:
        raise ValueError(f"{owner}: topology mode {resolved.mode!r} is not supported.")
    if resolved.core_dim_alignment not in {"exact", "exclude_all_core"}:
        raise ValueError(
            f"{owner}: core_dim_alignment={resolved.core_dim_alignment!r} is not supported."
        )
    if resolved.alignment_on not in _ALIGNMENT_KEYS:
        raise ValueError(f"{owner}: alignment_on={resolved.alignment_on!r} is not supported.")
    if resolved.sequence_join is not None and resolved.sequence_join not in _JOIN_POLICIES:
        raise ValueError(f"{owner}: sequence_join={resolved.sequence_join!r} is not supported.")
    if resolved.batch_join not in _JOIN_POLICIES:
        raise ValueError(f"{owner}: batch_join={resolved.batch_join!r} is not supported.")
    if resolved.core_policy not in _CORE_POLICIES:
        raise ValueError(f"{owner}: core_policy={resolved.core_policy!r} is not supported.")
    if not isinstance(resolved.strict_core_match_required, bool):
        raise ValueError(
            f"{owner}: strict_core_match_required={resolved.strict_core_match_required!r} "
            "must be bool."
        )
    if resolved.alignment_on == "sequence" and resolved.sequence_join is None:
        raise ValueError(f"{owner}: sequence_join=None is not valid when alignment_on='sequence'.")
    return resolved

def _require_operands(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
) -> None:
    if operands:
        return
    raise ValueError(f"{owner}: {what} requires at least one operand.")

def _require_sequence_batch_match_strict(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
) -> tuple[str | None, tuple[str, ...]]:
    base = operands[0].semantic
    for operand in operands[1:]:
        if operand.semantic.sequence_dim != base.sequence_dim:
            raise ValueError(
                f"{owner}: {what} requires matching sequence_dim; "
                f"operand 0={base.sequence_dim!r}, operand {operand.index}={operand.semantic.sequence_dim!r}."
            )
        if operand.semantic.batch_dims != base.batch_dims:
            raise ValueError(
                f"{owner}: {what} requires matching batch_dims; "
                f"operand 0={base.batch_dims!r}, operand {operand.index}={operand.semantic.batch_dims!r}."
            )
    return base.sequence_dim, base.batch_dims

def _resolve_sequence_batch_match_semantic(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
) -> tuple[str | None, tuple[str, ...]]:
    sequence_dim: str | None = None
    for operand in operands:
        current = operand.semantic.sequence_dim
        if current is None:
            continue
        if sequence_dim is None:
            sequence_dim = current
            continue
        if current == sequence_dim:
            continue
        raise ValueError(
            f"{owner}: {what} requires matching sequence_dim; "
            f"got {sequence_dim!r} vs {current!r}."
        )
    batch_dims: tuple[str, ...] = ()
    for operand in operands:
        current = operand.semantic.batch_dims
        if not current:
            continue
        if not batch_dims:
            batch_dims = current
            continue
        if current == batch_dims:
            continue
        raise ValueError(
            f"{owner}: {what} requires matching batch_dims; "
            f"got {batch_dims!r} vs {current!r}."
        )
    return sequence_dim, batch_dims

def _resolve_sequence_batch_match(
    operands: tuple[TopologyOperand, ...],
    *,
    policy: TopologyPolicy,
    owner: str,
    what: str,
) -> tuple[str | None, tuple[str, ...]]:
    if policy.mode == "strict":
        return _require_sequence_batch_match_strict(operands, owner=owner, what=what)
    return _resolve_sequence_batch_match_semantic(operands, owner=owner, what=what)

def _non_core_dim_name_set(operand: TopologyOperand) -> set[str]:
    core = set(operand.semantic.core_dims)
    return {dim for dim in operand.data.dims if dim not in core}

def _require_non_core_dim_names_match(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
) -> None:
    base = operands[0]
    base_set = _non_core_dim_name_set(base)
    for operand in operands[1:]:
        current = _non_core_dim_name_set(operand)
        if current == base_set:
            continue
        left_only = tuple(sorted(base_set - current))
        right_only = tuple(sorted(current - base_set))
        raise ValueError(
            f"{owner}: {what} requires matching non-core dim names; "
            f"operand 0={tuple(sorted(base_set))!r}, operand {operand.index}={tuple(sorted(current))!r}, "
            f"left_only={left_only!r}, right_only={right_only!r}."
        )

def _semantic_dim_set(
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
) -> set[str]:
    names = set(batch_dims)
    if sequence_dim is not None:
        names.add(sequence_dim)
    return names

def _require_semantic_non_core_compatibility(
    operands: tuple[TopologyOperand, ...],
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    owner: str,
    what: str,
) -> None:
    base = operands[0]
    base_set = _non_core_dim_name_set(base)
    semantic_dims = _semantic_dim_set(sequence_dim, batch_dims)
    for operand in operands[1:]:
        current = _non_core_dim_name_set(operand)
        left_only = base_set - current
        right_only = current - base_set
        invalid_left = tuple(sorted(dim for dim in left_only if dim not in semantic_dims))
        invalid_right = tuple(sorted(dim for dim in right_only if dim not in semantic_dims))
        if not invalid_left and not invalid_right:
            continue
        raise ValueError(
            f"{owner}: {what} semantic broadcast only permits missing sequence/batch dims; "
            f"operand 0={tuple(sorted(base_set))!r}, operand {operand.index}={tuple(sorted(current))!r}, "
            f"invalid_left_only={invalid_left!r}, invalid_right_only={invalid_right!r}."
        )

def _require_core_dims_present(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
) -> None:
    for operand in operands:
        missing = tuple(dim for dim in operand.semantic.core_dims if dim not in operand.data.dims)
        if not missing:
            continue
        raise ValueError(
            f"{owner}: {what} requires core dims {missing!r} on operand {operand.index}; "
            f"present dims={tuple(operand.data.dims)!r}."
        )

def _shared_param_coord(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
) -> str | None:
    names = {operand.param_coord for operand in operands}
    if len(names) == 1:
        return next(iter(names))
    non_null = {name for name in names if name is not None}
    if len(non_null) <= 1:
        return None
    raise ValueError(
        f"{owner}: {what} requires matching param_coord names; "
        f"got {tuple(sorted(non_null))!r}."
    )

def _resolve_primary_key(
    *,
    policy: TopologyPolicy,
    sequence_dim: str | None,
    param_coord: str | None,
    owner: str,
    what: str,
) -> tuple[Literal["sequence", "param"], str | None]:
    if policy.alignment_on == "sequence":
        return "sequence", None
    if policy.alignment_on == "param":
        if sequence_dim is None:
            raise ValueError(f"{owner}: {what} cannot use alignment_on='param' without sequence_dim.")
        if param_coord is None:
            raise ValueError(f"{owner}: {what} requires shared param_coord for alignment_on='param'.")
        return "param", param_coord
    sequence_usable = sequence_dim is not None
    param_usable = sequence_dim is not None and param_coord is not None
    if sequence_usable and param_usable:
        raise ValueError(f"{owner}: {what} alignment_on='auto' is ambiguous between sequence and param keys.")
    if param_usable:
        return "param", param_coord
    if sequence_usable:
        return "sequence", None
    raise ValueError(f"{owner}: {what} alignment_on='auto' requires a usable sequence or param key.")

def _require_matching_core_dims(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
) -> tuple[str, ...]:
    base = operands[0].semantic.core_dims
    for operand in operands[1:]:
        if operand.semantic.core_dims == base:
            continue
        raise ValueError(
            f"{owner}: {what} requires matching core dims under core_policy='strict'; "
            f"operand 0={base!r}, operand {operand.index}={operand.semantic.core_dims!r}."
        )
    return base

def _resolve_numpy_named_core_dims(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
) -> tuple[str, ...]:
    non_scalar = [operand.semantic.core_dims for operand in operands if operand.semantic.core_dims]
    if not non_scalar:
        return ()
    expected = non_scalar[0]
    for operand in operands:
        core_dims = operand.semantic.core_dims
        if not core_dims:
            continue
        if core_dims == expected:
            continue
        raise ValueError(
            f"{owner}: {what} core_policy='numpy_named' only supports scalar cores and "
            "matching ordered core dim names; "
            f"expected {expected!r}, operand {operand.index}={core_dims!r}."
        )
    return expected

def _resolve_output_core_dims(
    operands: tuple[TopologyOperand, ...],
    *,
    policy: TopologyPolicy,
    owner: str,
    what: str,
) -> tuple[str, ...]:
    if policy.core_policy == "numpy_named":
        return _resolve_numpy_named_core_dims(operands, owner=owner, what=what)
    if policy.strict_core_match_required:
        return _require_matching_core_dims(operands, owner=owner, what=what)
    return operands[0].semantic.core_dims

def _resolve_align_exclude_dims(
    operands: tuple[TopologyOperand, ...],
    *,
    policy: TopologyPolicy,
) -> frozenset[str]:
    if policy.core_policy == "numpy_named":
        return frozenset(dim for operand in operands for dim in operand.semantic.core_dims)
    if policy.core_dim_alignment == "exact":
        return frozenset()
    return frozenset(dim for operand in operands for dim in operand.semantic.core_dims)

def _validate_non_core_dims_for_mode(
    operands: tuple[TopologyOperand, ...],
    *,
    policy: TopologyPolicy,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    owner: str,
    what: str,
) -> None:
    if policy.mode == "strict":
        _require_non_core_dim_names_match(operands, owner=owner, what=what)
        return
    _require_semantic_non_core_compatibility(
        operands,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        owner=owner,
        what=what,
    )

def _build_topology_plan(
    operands: tuple[TopologyOperand, ...],
    *,
    policy: TopologyPolicy,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    param_coord: str | None,
    owner: str,
    what: str,
) -> ResolvedTopologyPlan:
    primary_key, key_param_coord = _resolve_primary_key(
        policy=policy,
        sequence_dim=sequence_dim,
        param_coord=param_coord,
        owner=owner,
        what=what,
    )
    return ResolvedTopologyPlan(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        operands=operands,
        align_exclude_dims=_resolve_align_exclude_dims(operands, policy=policy),
        mode=policy.mode,
        primary_key=primary_key,
        sequence_join=policy.sequence_join,
        batch_join=policy.batch_join,
        core_policy=policy.core_policy,
        param_coord=key_param_coord,
        output_core_dims=_resolve_output_core_dims(operands, policy=policy, owner=owner, what=what),
    )

def _resolve_topology_plan(
    operands: tuple[TopologyOperand, ...],
    *,
    owner: str,
    what: str,
    policy: TopologyPolicy | None,
) -> ResolvedTopologyPlan:
    resolved_policy = _normalize_policy(policy, owner=owner)
    _require_operands(operands, owner=owner, what=what)
    sequence_dim, batch_dims = _resolve_sequence_batch_match(
        operands,
        policy=resolved_policy,
        owner=owner,
        what=what,
    )
    _require_core_dims_present(operands, owner=owner, what=what)
    _validate_non_core_dims_for_mode(
        operands,
        policy=resolved_policy,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        owner=owner,
        what=what,
    )
    return _build_topology_plan(
        operands,
        policy=resolved_policy,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        param_coord=_shared_param_coord(operands, owner=owner, what=what),
        owner=owner,
        what=what,
    )

def resolve_unary_topology(
    operand: TopologyOperand,
    *,
    owner: str,
    what: str,
    policy: TopologyPolicy | None = None,
) -> ResolvedTopologyPlan:
    return _resolve_topology_plan((operand,), owner=owner, what=what, policy=policy)

def resolve_binary_topology(
    left: TopologyOperand,
    right: TopologyOperand,
    *,
    owner: str,
    what: str,
    policy: TopologyPolicy | None = None,
) -> ResolvedTopologyPlan:
    return _resolve_topology_plan((left, right), owner=owner, what=what, policy=policy)

def resolve_nary_topology(
    operands: Sequence[TopologyOperand],
    *,
    owner: str,
    what: str,
    policy: TopologyPolicy | None = None,
) -> ResolvedTopologyPlan:
    return _resolve_topology_plan(tuple(operands), owner=owner, what=what, policy=policy)

def realize_operands_for_plan(
    plan: ResolvedTopologyPlan,
    *,
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    if not plan.operands:
        raise ValueError(f"{owner}: {what} resolved to an empty topology plan.")
    if len(plan.operands) == 1:
        # Unary operations must remain topology-preserving in semantic mode.
        return (plan.operands[0].data,)
    if plan.mode == "strict":
        return tuple(operand.data for operand in plan.operands)
    return _realize_semantic_broadcast_operands(plan, owner=owner, what=what)

def _reference_operand_for_dim(
    plan: ResolvedTopologyPlan,
    *,
    dim: str,
    owner: str,
    what: str,
) -> xr.DataArray:
    for operand in plan.operands:
        if dim in operand.data.dims:
            return operand.data
    raise ValueError(
        f"{owner}: {what} semantic broadcast requires at least one operand carrying dim {dim!r}."
    )

def _expand_spec_for_dim(
    reference: xr.DataArray,
    *,
    dim: str,
) -> xr.DataArray | int:
    if dim in reference.coords and tuple(reference.coords[dim].dims) == (dim,):
        return reference.coords[dim]
    return int(reference.sizes[dim])

def _expand_operand_semantic_dims(
    operand: TopologyOperand,
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    references: dict[str, xr.DataArray],
    owner: str,
    what: str,
) -> xr.DataArray:
    dims_to_add = [
        dim for dim in ((sequence_dim,) + batch_dims if sequence_dim is not None else batch_dims)
        if dim not in operand.data.dims
    ]
    if not dims_to_add:
        return operand.data
    expand_spec = {
        dim: _expand_spec_for_dim(references[dim], dim=dim)
        for dim in dims_to_add
    }
    try:
        return operand.data.expand_dims(expand_spec)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{owner}: {what} failed to realize semantic broadcast dims {tuple(dims_to_add)!r} "
            f"for operand {operand.index}."
        ) from exc

def _realize_semantic_broadcast_operands(
    plan: ResolvedTopologyPlan,
    *,
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    semantic_dims = tuple(
        dim for dim in ((plan.sequence_dim,) + plan.batch_dims if plan.sequence_dim is not None else plan.batch_dims)
    )
    references = {
        dim: _reference_operand_for_dim(plan, dim=dim, owner=owner, what=what)
        for dim in semantic_dims
    }
    return tuple(
        _expand_operand_semantic_dims(
            operand,
            sequence_dim=plan.sequence_dim,
            batch_dims=plan.batch_dims,
            references=references,
            owner=owner,
            what=what,
        )
        for operand in plan.operands
    )

__all__ = [
    "ResolvedTopologyPlan",
    "SEMANTIC_EXACT_POLICY",
    "SEMANTIC_NON_CORE_POLICY",
    "SemanticTopology",
    "STRICT_EXACT_POLICY",
    "STRICT_NON_CORE_POLICY",
    "TopologyOperand",
    "TopologyPolicy",
    "align_combine_batch_axis",
    "allocate_flat_batch_dim_name",
    "batch_index_for_dataset",
    "flatten_param_contexts",
    "flatten_query_for_batch_plan",
    "join_combine_batch_labels",
    "join_batch_indices",
    "restore_combine_batch_axis",
    "restore_dataset_batch_topology",
    "restore_dataset_multi_batch",
    "resolve_binary_topology",
    "resolve_nary_topology",
    "resolve_unary_topology",
    "realize_operands_for_plan",
    "stack_combine_batch_axis",
]
