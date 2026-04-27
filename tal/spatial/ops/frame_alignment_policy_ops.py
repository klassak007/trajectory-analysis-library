from __future__ import annotations

import xarray as xr

from tal.core.orchestration.alignment import align_exact_for_plan
from tal.core.orchestration.alignment_intent import select_topology_policy_with_intents
from tal.core.orchestration.context import resolve_semantic_topology_from_dataset
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.orchestration.topology import (
    SEMANTIC_NON_CORE_POLICY,
    STRICT_NON_CORE_POLICY,
    TopologyOperand,
    resolve_binary_topology,
)
from tal.core.schema_read import read_param_coord_name
from tal.utils.topology_operation_families import operation_intent_support_for_operation_family


def _pair_operand(
    value,
    data: xr.DataArray,
    *,
    core_dim: str,
    index: int,
    policy,
    owner: str,
    what: str,
) -> TopologyOperand:
    var_name = "__tal_alignment_operand__"
    ds = value.unsafe_data.copy()
    if ds.data_vars:
        ds = ds.drop_vars(tuple(ds.data_vars), errors="ignore")
    ds[var_name] = data
    semantic = resolve_semantic_topology_from_dataset(
        ds,
        var_name=var_name,
        core_dims=(core_dim,),
        owner=owner,
        what=what,
        allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
        allow_missing_batch_dims=policy.mode == "semantic_broadcast",
    )
    return TopologyOperand(
        index=index,
        data=ds[var_name],
        semantic=semantic,
        param_coord=read_param_coord_name(ds),
    )


def _preflight_param_primary(
    left_value,
    right_value,
    *,
    param_coord: str,
    owner: str,
    what: str,
) -> None:
    try:
        _ = resolve_param_runtime_context(left_value, on=param_coord)
        _ = resolve_param_runtime_context(right_value, on=param_coord)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{owner}: {what} requires usable numeric param domain for on='param'.") from exc


def align_frame_pair_by_policy(
    left_value,
    right_value,
    *,
    left_da: xr.DataArray,
    right_da: xr.DataArray,
    left_core_dim: str,
    right_core_dim: str,
    owner: str,
    what: str,
    operation_family: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    selection = select_topology_policy_with_intents(
        (left_value, right_value),
        owner=owner,
        operation_family=operation_family,
        support=operation_intent_support_for_operation_family(operation_family, owner=owner),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    left = _pair_operand(
        left_value,
        left_da,
        core_dim=left_core_dim,
        index=0,
        policy=selection.policy,
        owner=owner,
        what=what,
    )
    right = _pair_operand(
        right_value,
        right_da,
        core_dim=right_core_dim,
        index=1,
        policy=selection.policy,
        owner=owner,
        what=what,
    )
    plan = resolve_binary_topology(left, right, owner=owner, what=what, policy=selection.policy)
    if plan.primary_key == "param" and plan.param_coord is not None:
        _preflight_param_primary(
            left_value,
            right_value,
            param_coord=plan.param_coord,
            owner=owner,
            what=what,
        )
    return align_exact_for_plan(plan, owner=owner, what=what)


__all__ = ["align_frame_pair_by_policy"]
