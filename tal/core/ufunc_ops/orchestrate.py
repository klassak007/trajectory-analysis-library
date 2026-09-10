from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..analysis_object import AnalysisObject
from ..dataset_ownership import analysis_object_dataset
from ..event_ops.types import Condition, CoordOperand, VarOperand
from ..orchestration.alignment import align_exact_for_plan
from ..orchestration.alignment_intent import (
    select_topology_policy_with_intents,
)
from ..orchestration.context import (
    DatasetContext,
    DatasetContextOptions,
    resolve_dataset_context,
    resolve_semantic_topology_from_dataset,
)
from ..schema_read import read_roles
from ..orchestration.topology import (
    SemanticTopology,
    TopologyOperand,
    TopologyPolicy,
    resolve_binary_topology,
    resolve_unary_topology,
)
from ...utils.topology_operation_families import operation_intent_support_for_operation_family


@dataclass(frozen=True)
class UnaryAOContext:
    source: AnalysisObject | None
    operand: object


@dataclass(frozen=True)
class BinaryAOContext:
    source: AnalysisObject | None
    left: object
    right: object
    output_core_dims: tuple[str, ...] | None = None
    rewrap_context: object | None = None


@dataclass(frozen=True)
class _SemanticOperand:
    index: int
    source: AnalysisObject
    context: DatasetContext


@dataclass(frozen=True)
class _BinaryRuntimeOperands:
    source: AnalysisObject | None
    left_source: AnalysisObject | None
    right_source: AnalysisObject | None
    left_operand: object
    right_operand: object
    rewrap_context: object | None


_SEMANTIC_UFUNC_CONTEXT_OPTIONS = DatasetContextOptions(
    require_roles=False,
    require_sequence_dim=False,
    select_numeric_var=True,
    require_single_numeric_var=True,
    require_semantic_dims_in_var=False,
)

_UFUNC_STRICT_EXACT_POLICY = TopologyPolicy(
    mode="strict",
    core_dim_alignment="exact",
    strict_core_match_required=True,
)
_UFUNC_SEMANTIC_EXACT_POLICY = TopologyPolicy(
    mode="semantic_broadcast",
    core_dim_alignment="exact",
    strict_core_match_required=True,
)


def _semantic_policy(values: tuple[object, ...], *, owner: str) -> TopologyPolicy:
    return select_topology_policy_with_intents(
        values,
        owner=owner,
        operation_family="ufunc.arithmetic",
        support=operation_intent_support_for_operation_family(
            "ufunc.arithmetic",
            owner=owner,
        ),
        strict_policy=_UFUNC_STRICT_EXACT_POLICY,
        semantic_policy=_UFUNC_SEMANTIC_EXACT_POLICY,
    ).policy


def _semantic_operand(
    value: object,
    *,
    index: int,
    owner: str,
    what: str,
) -> _SemanticOperand | None:
    if not isinstance(value, AnalysisObject):
        return None
    context = resolve_dataset_context(
        value,
        owner=owner,
        options=_SEMANTIC_UFUNC_CONTEXT_OPTIONS,
        index=index,
    )
    if context.data is None or context.var_name is None:
        raise ValueError(f"{owner}: {what} requires a single numeric variable for semantic broadcast planning.")
    return _SemanticOperand(index=index, source=value, context=context)


def _topology_operand(spec: _SemanticOperand, *, owner: str, what: str, policy: TopologyPolicy) -> TopologyOperand:
    context = spec.context
    assert context.var_name is not None
    assert context.data is not None
    if context.roles_declared:
        semantic = resolve_semantic_topology_from_dataset(
            context.ds,
            var_name=context.var_name,
            core_dims=context.core_dims,
            owner=owner,
            what=what,
            allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
            allow_missing_batch_dims=policy.mode == "semantic_broadcast",
        )
    else:
        semantic = SemanticTopology(
            sequence_dim=None,
            batch_dims=(),
            core_dims=context.core_dims,
        )
    return TopologyOperand(
        index=spec.index,
        data=context.data,
        semantic=semantic,
        param_coord=context.param_coord,
    )


def _align_semantic_unary_operand(spec: _SemanticOperand, *, owner: str, what: str, policy: TopologyPolicy) -> xr.DataArray:
    plan = resolve_unary_topology(
        _topology_operand(spec, owner=owner, what=what, policy=policy),
        owner=owner,
        what=what,
        policy=policy,
    )
    (aligned,) = align_exact_for_plan(plan, owner=owner, what=what)
    return aligned


def _align_semantic_binary_operands(
    left: _SemanticOperand,
    right: _SemanticOperand,
    *,
    owner: str,
    what: str,
    policy: TopologyPolicy,
) -> tuple[xr.DataArray, xr.DataArray, tuple[str, ...]]:
    plan = resolve_binary_topology(
        _topology_operand(left, owner=owner, what=f"{what} left", policy=policy),
        _topology_operand(right, owner=owner, what=f"{what} right", policy=policy),
        owner=owner,
        what=what,
        policy=policy,
    )
    aligned_left, aligned_right = align_exact_for_plan(plan, owner=owner, what=what)
    return aligned_left, aligned_right, plan.output_core_dims


def prepare_unary_ao_context(value: object, *, owner: str) -> UnaryAOContext:
    if isinstance(value, AnalysisObject):
        policy = _semantic_policy((value,), owner=owner)
        if policy.mode == "semantic_broadcast":
            spec = _semantic_operand(value, index=0, owner=owner, what="unary ufunc operand")
            assert spec is not None
            aligned = _align_semantic_unary_operand(
                spec,
                owner=owner,
                what="unary ufunc semantic broadcast",
                policy=policy,
            )
            return UnaryAOContext(source=value, operand=aligned)
        return UnaryAOContext(source=value, operand=analysis_object_dataset(value))
    return UnaryAOContext(source=None, operand=value)


def _bootstrap_binary_runtime_operands(
    left: object,
    right: object,
    *,
    owner: str,
) -> _BinaryRuntimeOperands:
    left_source = left if isinstance(left, AnalysisObject) else None
    right_source = right if isinstance(right, AnalysisObject) else None
    source = left_source if left_source is not None else right_source
    rewrap_context = (
        None
        if source is None
        else source._prepare_result_rewrap_context((left, right), owner=owner)
    )
    left_operand = analysis_object_dataset(left) if left_source is not None else left
    right_operand = analysis_object_dataset(right) if right_source is not None else right
    return _BinaryRuntimeOperands(
        source=source,
        left_source=left_source,
        right_source=right_source,
        left_operand=left_operand,
        right_operand=right_operand,
        rewrap_context=rewrap_context,
    )


def _read_binary_output_core_dims(runtime: _BinaryRuntimeOperands) -> tuple[str, ...] | None:
    if runtime.left_source is not None:
        _, _, _, core_dims = read_roles(analysis_object_dataset(runtime.left_source))
        return core_dims
    if runtime.right_source is not None:
        _, _, _, core_dims = read_roles(analysis_object_dataset(runtime.right_source))
        return core_dims
    return None


def _prepare_strict_binary_ao_context(runtime: _BinaryRuntimeOperands) -> BinaryAOContext:
    return BinaryAOContext(
        source=runtime.source,
        left=runtime.left_operand,
        right=runtime.right_operand,
        output_core_dims=_read_binary_output_core_dims(runtime),
        rewrap_context=runtime.rewrap_context,
    )


def _prepare_semantic_binary_ao_context(
    runtime: _BinaryRuntimeOperands,
    left: object,
    right: object,
    *,
    owner: str,
    policy: TopologyPolicy,
) -> BinaryAOContext:
    left_spec = _semantic_operand(left, index=0, owner=owner, what="binary ufunc left operand")
    right_spec = _semantic_operand(right, index=1, owner=owner, what="binary ufunc right operand")
    if left_spec is not None:
        if right_spec is None:
            return _prepare_one_sided_semantic_context(
                runtime,
                left_spec,
                is_left=True,
                owner=owner,
                policy=policy,
            )
        left_operand, right_operand, output_core_dims = _align_semantic_binary_operands(
            left_spec,
            right_spec,
            owner=owner,
            what="binary ufunc semantic broadcast",
            policy=policy,
        )
    if right_spec is not None and left_spec is None:
        return _prepare_one_sided_semantic_context(
            runtime,
            right_spec,
            is_left=False,
            owner=owner,
            policy=policy,
        )
    if left_spec is None or right_spec is None:
        return _prepare_strict_binary_ao_context(runtime)
    return BinaryAOContext(
        source=runtime.source,
        left=left_operand,
        right=right_operand,
        output_core_dims=output_core_dims,
        rewrap_context=runtime.rewrap_context,
    )


def _prepare_one_sided_semantic_context(
    runtime: _BinaryRuntimeOperands,
    spec: _SemanticOperand,
    *,
    is_left: bool,
    owner: str,
    policy: TopologyPolicy,
) -> BinaryAOContext:
    side = "left" if is_left else "right"
    aligned = _align_semantic_unary_operand(
        spec,
        owner=owner,
        what=f"binary ufunc {side} semantic broadcast",
        policy=policy,
    )
    return BinaryAOContext(
        source=runtime.source,
        left=aligned if is_left else runtime.left_operand,
        right=runtime.right_operand if is_left else aligned,
        output_core_dims=spec.context.core_dims,
        rewrap_context=runtime.rewrap_context,
    )


def prepare_binary_ao_context(left: object, right: object, *, owner: str) -> BinaryAOContext:
    runtime = _bootstrap_binary_runtime_operands(left, right, owner=owner)
    policy = _semantic_policy((left, right), owner=owner)
    if policy.mode != "semantic_broadcast":
        return _prepare_strict_binary_ao_context(runtime)
    return _prepare_semantic_binary_ao_context(
        runtime,
        left,
        right,
        owner=owner,
        policy=policy,
    )


def _is_comparison_operand(value: object) -> bool:
    if isinstance(value, (AnalysisObject, xr.Dataset, xr.DataArray, VarOperand, CoordOperand)):
        return True
    return bool(np.isscalar(value))


def validate_condition_compare_operands(left: object, right: object, *, owner: str) -> None:
    if not _is_comparison_operand(left):
        raise TypeError(
            f"{owner}: unsupported left operand type {type(left).__name__} for comparison condition."
        )
    if not _is_comparison_operand(right):
        raise TypeError(
            f"{owner}: unsupported right operand type {type(right).__name__} for comparison condition."
        )


def require_condition(value: object, *, owner: str, side: str) -> Condition:
    if isinstance(value, Condition):
        return value
    raise TypeError(f"{owner}: {side} operand must be Condition.")


__all__ = [
    "BinaryAOContext",
    "UnaryAOContext",
    "prepare_binary_ao_context",
    "prepare_unary_ao_context",
    "require_condition",
    "validate_condition_compare_operands",
]
