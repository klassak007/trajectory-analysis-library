from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import xarray as xr

from ..core.analysis_object import AnalysisObject
from ..core.metadata_optional import shared_optional_name
from ..core.orchestration.alignment import align_exact_for_plan
from ..core.orchestration.alignment_intent import (
    select_topology_policy_with_intents,
)
from ..core.orchestration.context import (
    DatasetContextOptions,
    resolve_dataset_context,
    resolve_semantic_topology_from_dataset,
)
from ..core.orchestration.topology import (
    ResolvedTopologyPlan,
    SEMANTIC_EXACT_POLICY,
    STRICT_EXACT_POLICY,
    SemanticTopology,
    TopologyPolicy,
    TopologyOperand,
    resolve_binary_topology,
    resolve_nary_topology,
    resolve_unary_topology,
)
from ..utils.topology_operation_families import operation_intent_support_for_operation_family


@dataclass(frozen=True)
class ArrayOperandContext:
    ao: AnalysisObject
    var_name: str
    data: xr.DataArray
    roles_declared: bool
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    param_coord: str | None
    sequence_size_coord: str | None


@dataclass(frozen=True)
class ArrayPlan:
    operands: tuple[ArrayOperandContext, ...]
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    core_policy: str
    output_core_dims: tuple[str, ...]
    shared_param_coord: str | None
    shared_sequence_size_coord: str | None


def _align_contexts_for_topology(
    operands: tuple[ArrayOperandContext, ...],
    *,
    plan: ResolvedTopologyPlan,
    owner: str,
) -> tuple[ArrayOperandContext, ...]:
    aligned = align_exact_for_plan(
        plan,
        owner=owner,
        what="array plan",
    )
    return tuple(
        replace(operand, data=arr)
        for operand, arr in zip(operands, aligned, strict=True)
    )


def _resolve_topology_plan(
    contexts: tuple[ArrayOperandContext, ...],
    *,
    owner: str,
    require_declared_roles: bool,
    policy: TopologyPolicy,
) -> ResolvedTopologyPlan:
    operands = tuple(
        _topology_operand_for_context(
            context,
            owner=owner,
            index=index,
            require_declared_roles=require_declared_roles,
            policy=policy,
        )
        for index, context in enumerate(contexts)
    )
    if len(operands) == 1:
        return resolve_unary_topology(
            operands[0],
            owner=owner,
            what="array plan",
            policy=policy,
        )
    if len(operands) == 2:
        return resolve_binary_topology(
            operands[0],
            operands[1],
            owner=owner,
            what="array plan",
            policy=policy,
        )
    return resolve_nary_topology(
        operands,
        owner=owner,
        what="array plan",
        policy=policy,
    )


def _semantic_topology_for_context(
    context: ArrayOperandContext,
    *,
    owner: str,
    index: int,
    require_declared_roles: bool,
    policy: TopologyPolicy,
) -> SemanticTopology:
    if context.roles_declared:
        return resolve_semantic_topology_from_dataset(
            context.ao.unsafe_data,
            var_name=context.var_name,
            core_dims=context.core_dims,
            owner=owner,
            what=f"operand {index}",
            allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
            allow_missing_batch_dims=policy.mode == "semantic_broadcast",
        )
    if require_declared_roles:
        raise ValueError(f"{owner}: operand {index} requires declared roles.")
    return SemanticTopology(
        sequence_dim=None,
        batch_dims=(),
        core_dims=context.core_dims,
    )


def _topology_operand_for_context(
    context: ArrayOperandContext,
    *,
    owner: str,
    index: int,
    require_declared_roles: bool,
    policy: TopologyPolicy,
) -> TopologyOperand:
    return TopologyOperand(
        index=index,
        data=context.data,
        semantic=_semantic_topology_for_context(
            context,
            owner=owner,
            index=index,
            require_declared_roles=require_declared_roles,
            policy=policy,
        ),
        param_coord=context.param_coord,
    )


def _build_context_options(
    *,
    require_declared_roles: bool,
    require_single_numeric_var: bool,
    allowed_core_arity: tuple[int, ...],
    require_semantic_dims_in_var: bool,
) -> DatasetContextOptions:
    return DatasetContextOptions(
        require_roles=require_declared_roles,
        require_sequence_dim=False,
        select_numeric_var=True,
        require_single_numeric_var=require_single_numeric_var,
        allowed_core_arity=allowed_core_arity,
        require_semantic_dims_in_var=require_semantic_dims_in_var,
    )


def _resolve_operand_context(
    operand: AnalysisObject,
    *,
    owner: str,
    index: int,
    options: DatasetContextOptions,
) -> ArrayOperandContext:
    ctx = resolve_dataset_context(
        operand,
        owner=owner,
        index=index,
        options=options,
    )
    if ctx.data is None or ctx.var_name is None:
        raise ValueError(f"{owner}: operand {index} requires a selected numeric data variable.")
    return ArrayOperandContext(
        ao=ctx.ao,
        var_name=ctx.var_name,
        data=ctx.data,
        roles_declared=ctx.roles_declared,
        sequence_dim=ctx.sequence_dim,
        batch_dims=ctx.batch_dims,
        core_dims=ctx.core_dims,
        param_coord=ctx.param_coord,
        sequence_size_coord=ctx.sequence_size_coord,
    )


def _collect_operand_contexts(
    operands: Sequence[AnalysisObject],
    *,
    owner: str,
    options: DatasetContextOptions,
) -> tuple[ArrayOperandContext, ...]:
    context_items: list[ArrayOperandContext] = []
    for idx, operand in enumerate(operands):
        context_items.append(
            _resolve_operand_context(
                operand,
                owner=owner,
                index=idx,
                options=options,
            )
        )
    return tuple(context_items)


def _validate_align_mode(*, align_mode: str, owner: str) -> None:
    if align_mode == "exact":
        return
    raise ValueError(f"{owner}: align_mode={align_mode!r} is not supported; expected 'exact'.")


def _assemble_array_plan(
    contexts: tuple[ArrayOperandContext, ...],
    *,
    topology: ResolvedTopologyPlan,
) -> ArrayPlan:
    return ArrayPlan(
        operands=contexts,
        sequence_dim=topology.sequence_dim,
        batch_dims=topology.batch_dims,
        core_policy=topology.core_policy,
        output_core_dims=topology.output_core_dims,
        shared_param_coord=shared_optional_name(tuple(op.param_coord for op in contexts)),
        shared_sequence_size_coord=shared_optional_name(
            tuple(op.sequence_size_coord for op in contexts)
        ),
    )


def build_array_plan(
    operands: Sequence[AnalysisObject],
    *,
    owner: str,
    require_declared_roles: bool = True,
    allowed_core_arity: tuple[int, ...] = (1, 2),
    align_mode: str = "exact",
    require_single_numeric_var: bool = True,
) -> ArrayPlan:
    if not operands:
        raise ValueError(f"{owner}: expected at least one operand.")
    selection = select_topology_policy_with_intents(
        operands,
        owner=owner,
        operation_family="linalg.elementwise",
        support=operation_intent_support_for_operation_family(
            "linalg.elementwise",
            owner=owner,
        ),
        strict_policy=STRICT_EXACT_POLICY,
        semantic_policy=SEMANTIC_EXACT_POLICY,
    )
    policy = selection.policy
    resolved_allowed_core_arity = allowed_core_arity
    if policy.core_policy == "numpy_named" and 0 not in resolved_allowed_core_arity:
        resolved_allowed_core_arity = (0,) + resolved_allowed_core_arity
    options = _build_context_options(
        require_declared_roles=require_declared_roles,
        require_single_numeric_var=require_single_numeric_var,
        allowed_core_arity=resolved_allowed_core_arity,
        require_semantic_dims_in_var=policy.mode == "strict",
    )
    contexts = _collect_operand_contexts(
        operands,
        owner=owner,
        options=options,
    )
    _validate_align_mode(align_mode=align_mode, owner=owner)
    topology = _resolve_topology_plan(
        contexts,
        owner=owner,
        require_declared_roles=require_declared_roles,
        policy=policy,
    )
    contexts = _align_contexts_for_topology(contexts, plan=topology, owner=owner)
    return _assemble_array_plan(contexts, topology=topology)


def build_array_binary_plan(
    left: AnalysisObject,
    right: AnalysisObject,
    *,
    owner: str,
    require_declared_roles: bool = True,
    allowed_core_arity: tuple[int, ...] = (1, 2),
    align_mode: str = "exact",
    require_single_numeric_var: bool = True,
) -> ArrayPlan:
    return build_array_plan(
        [left, right],
        owner=owner,
        require_declared_roles=require_declared_roles,
        allowed_core_arity=allowed_core_arity,
        align_mode=align_mode,
        require_single_numeric_var=require_single_numeric_var,
    )


__all__ = [
    "ArrayOperandContext",
    "ArrayPlan",
    "build_array_binary_plan",
    "build_array_plan",
]
