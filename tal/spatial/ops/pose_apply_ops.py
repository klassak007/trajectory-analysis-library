from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.orchestration.alignment import align_exact_for_plan
from tal.core.orchestration.alignment_intent import select_topology_policy_with_intents
from tal.core.orchestration.context import resolve_semantic_topology_from_dataset
from tal.core.orchestration.topology import (
    SEMANTIC_NON_CORE_POLICY,
    STRICT_NON_CORE_POLICY,
    TopologyPolicy,
    TopologyOperand,
    resolve_nary_topology,
)
from tal.core.orchestration.runtime_checks import resolve_single_numeric_var_single_core_dim
from tal.core.schema_read import read_param_coord_name, validate_schema_if_needed
from tal.core.schema import copy_dataset_attrs
from tal.utils.frame_schema import set_frames
from tal.utils.topology_operation_families import operation_intent_support_for_operation_family

from ..acceleration import Acceleration, AngularAcceleration, LinearAcceleration
from ..policies.frame import resolve_apply_output_frames
from ..kernels.pose_apply_kernels import pose_apply_position_kernel
from ..position import Position
from .rotation_apply_ops import _rotation_apply_with_owner
from ..policies.wrap import wrap_like
from ..velocity import AngularVelocity, LinearVelocity, Velocity

if TYPE_CHECKING:
    from ..pose import Pose

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")

_VECTOR_TARGET_TYPES = (
    Position,
    LinearVelocity,
    AngularVelocity,
    LinearAcceleration,
    AngularAcceleration,
)

_SPATIAL_TARGET_TYPES = (
    Velocity,
    Acceleration,
)

_SUPPORTED_TARGET_TYPES = _VECTOR_TARGET_TYPES + _SPATIAL_TARGET_TYPES


@dataclass(frozen=True)
class PoseApplyOperandSpecs:
    target_ds: xr.Dataset
    target_var: str
    target_dim: str
    translation_ds: xr.Dataset
    translation_var: str
    translation_dim: str
    quat_ds: xr.Dataset
    quat_var: str
    quat_dim: str


@dataclass(frozen=True)
class PoseApplyResolvedInputs:
    specs: PoseApplyOperandSpecs
    target_da: xr.DataArray
    translation_da: xr.DataArray
    quat_da: xr.DataArray


def _topology_operand(
    ds: xr.Dataset,
    *,
    var_name: str,
    core_dim: str,
    index: int,
    owner: str,
    what: str,
    policy: TopologyPolicy,
) -> TopologyOperand:
    return TopologyOperand(
        index=index,
        data=ds[var_name],
        semantic=resolve_semantic_topology_from_dataset(
            ds,
            var_name=var_name,
            core_dims=(core_dim,),
            owner=owner,
            what=what,
            allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
            allow_missing_batch_dims=policy.mode == "semantic_broadcast",
        ),
        param_coord=read_param_coord_name(ds),
    )


def _wrap_pose_apply_position_kernel(
    position: np.ndarray,
    translation: np.ndarray,
    quat: np.ndarray,
    *,
    owner: str,
) -> np.ndarray:
    try:
        return pose_apply_position_kernel(position, translation, quat)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose apply position kernel failed.") from exc


def _resolve_pose_position_apply_inputs(
    pose: "Pose",
    target: Position,
    *,
    owner: str,
) -> PoseApplyResolvedInputs:
    specs = _resolve_pose_position_input_specs(pose, target, owner=owner)
    selection = select_topology_policy_with_intents(
        (pose, target),
        owner=owner,
        operation_family="spatial.pose.apply",
        support=operation_intent_support_for_operation_family(
            "spatial.pose.apply",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    policy = selection.policy
    target_da, translation_da, quat_da = _align_pose_position_operands(
        specs,
        owner=owner,
        policy=policy,
    )
    return PoseApplyResolvedInputs(
        specs=specs,
        target_da=target_da,
        translation_da=translation_da,
        quat_da=quat_da,
    )


def _resolve_pose_position_input_specs(
    pose: "Pose",
    target: Position,
    *,
    owner: str,
) -> PoseApplyOperandSpecs:
    translation, rotation = pose.decompose(validate=False)
    quat_rotation = rotation.as_quat(validate=False)
    target_ds = validate_schema_if_needed(target.unsafe_data)
    translation_ds = validate_schema_if_needed(translation.unsafe_data)
    quat_ds = validate_schema_if_needed(quat_rotation.unsafe_data)
    target_var, target_dim = resolve_single_numeric_var_single_core_dim(target_ds, owner=owner, what="Position target")
    translation_var, translation_dim = resolve_single_numeric_var_single_core_dim(
        translation_ds,
        owner=owner,
        what="Pose translation",
    )
    quat_var, quat_dim = resolve_single_numeric_var_single_core_dim(quat_ds, owner=owner, what="Pose rotation")
    return PoseApplyOperandSpecs(
        target_ds=target_ds,
        target_var=target_var,
        target_dim=target_dim,
        translation_ds=translation_ds,
        translation_var=translation_var,
        translation_dim=translation_dim,
        quat_ds=quat_ds,
        quat_var=quat_var,
        quat_dim=quat_dim,
    )


def _align_pose_position_operands(
    specs: PoseApplyOperandSpecs,
    *,
    owner: str,
    policy: TopologyPolicy,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    plan = _resolve_pose_position_plan(
        specs,
        owner=owner,
        policy=policy,
    )
    return align_exact_for_plan(
        plan,
        owner=owner,
        what="pose apply",
    )


def _resolve_pose_position_plan(
    specs: PoseApplyOperandSpecs,
    *,
    owner: str,
    policy: TopologyPolicy,
):
    operands = _pose_apply_topology_operands(
        specs,
        owner=owner,
        policy=policy,
    )
    plan = resolve_nary_topology(
        operands,
        owner=owner,
        what="pose apply",
        policy=policy,
    )
    return plan


def _pose_apply_topology_operands(
    specs: PoseApplyOperandSpecs,
    *,
    owner: str,
    policy: TopologyPolicy,
):
    return (
        _topology_operand(
            specs.target_ds,
            var_name=specs.target_var,
            core_dim=specs.target_dim,
            index=0,
            owner=owner,
            what="Position target",
            policy=policy,
        ),
        _topology_operand(
            specs.translation_ds,
            var_name=specs.translation_var,
            core_dim=specs.translation_dim,
            index=1,
            owner=owner,
            what="Pose translation",
            policy=policy,
        ),
        _topology_operand(
            specs.quat_ds,
            var_name=specs.quat_var,
            core_dim=specs.quat_dim,
            index=2,
            owner=owner,
            what="Pose rotation",
            policy=policy,
        ),
    )


def _apply_pose_position_kernel(
    target_da: xr.DataArray,
    translation_da: xr.DataArray,
    quat_da: xr.DataArray,
    *,
    target_dim: str,
    translation_dim: str,
    quat_dim: str,
    owner: str,
) -> xr.DataArray:
    kernel = _wrap_pose_apply_position_kernel
    try:
        output = xr.apply_ufunc(
            kernel,
            target_da,
            translation_da,
            quat_da,
            input_core_dims=[[target_dim], [translation_dim], [quat_dim]],
            output_core_dims=[[target_dim]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[np.float64],
            kwargs={"owner": owner},
            dask_gufunc_kwargs={"output_sizes": {target_dim: 3}},
        )
    except ValueError as exc:
        raise ValueError(f"{owner}: pose apply failed after alignment: {exc}") from exc
    return output.assign_coords({target_dim: list(_XYZ_LABELS)})


def _apply_pose_to_position(
    pose: "Pose",
    target: Position,
    *,
    parent: str | None,
    child: str | None,
    validate: bool,
    owner: str,
) -> Position:
    resolved = _resolve_pose_position_apply_inputs(
        pose,
        target,
        owner=owner,
    )
    output = _apply_pose_position_kernel(
        resolved.target_da,
        resolved.translation_da,
        resolved.quat_da,
        target_dim=resolved.specs.target_dim,
        translation_dim=resolved.specs.translation_dim,
        quat_dim=resolved.specs.quat_dim,
        owner=owner,
    )
    out_ds = output.to_dataset(name=resolved.specs.target_var)
    out_ds = copy_dataset_attrs(
        resolved.specs.target_ds,
        out_ds,
        validate=False,
    )
    out_ds = set_frames(out_ds, parent=parent, child=child, validate=False)
    return wrap_like(target, out_ds, validate=validate)


def _apply_pose_to_spatial_target(
    pose: "Pose",
    target: object,
    *,
    parent: str | None,
    child: str | None,
    validate: bool,
    owner: str,
) -> object:
    _, rotation = pose.decompose(validate=False)
    selection = select_topology_policy_with_intents(
        (pose, target),
        owner=owner,
        operation_family="spatial.pose.apply",
        support=operation_intent_support_for_operation_family(
            "spatial.pose.apply",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    rotation = _rotation_with_selection_intents(rotation, selection=selection)
    out = _rotation_apply_with_owner(rotation, target, validate=False, owner=owner)
    out_ds = set_frames(out.unsafe_data, parent=parent, child=child, validate=False)
    return wrap_like(target, out_ds, validate=validate)


def _rotation_with_selection_intents(rotation: "Rotation", *, selection):
    out = rotation
    alignment = selection.alignment
    if alignment is not None:
        out = out.a(
            on=alignment.on,
            sequence_join=alignment.sequence_join,
            batch_join=alignment.batch_join,
            core_policy=alignment.core_policy,
        )
    if selection.policy.mode == "semantic_broadcast":
        out = out.b()
    return out


def _pose_apply_with_owner(
    pose: "Pose",
    target: object,
    *,
    validate: bool,
    owner: str,
) -> object:
    if not isinstance(target, _SUPPORTED_TARGET_TYPES):
        raise TypeError(
            f"{owner}: target must be Position, LinearVelocity, AngularVelocity, "
            "LinearAcceleration, AngularAcceleration, Velocity, or Acceleration."
        )
    pose._enforce_invariants(owner=owner)
    target._enforce_invariants(owner=owner)
    parent, child = resolve_apply_output_frames(pose.unsafe_data, target.unsafe_data, owner=owner)
    if isinstance(target, Position):
        return _apply_pose_to_position(
            pose,
            target,
            parent=parent,
            child=child,
            validate=validate,
            owner=owner,
        )
    return _apply_pose_to_spatial_target(
        pose,
        target,
        parent=parent,
        child=child,
        validate=validate,
        owner=owner,
    )


def pose_apply(pose: "Pose", target: object, *, validate: bool) -> object:
    return _pose_apply_with_owner(
        pose,
        target,
        validate=validate,
        owner="spatial.pose.apply",
    )


__all__ = ["_pose_apply_with_owner", "pose_apply"]
