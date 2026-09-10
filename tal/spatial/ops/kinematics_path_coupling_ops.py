from __future__ import annotations

import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.runtime_checks import (
    resolve_single_numeric_var_single_core_dim,
    select_single_numeric_var,
)
from tal.frames import Frame

from ..metadata.roles import get_kinematics_kind
from ..policies.wrap import wrap_like
from .edge_resolver_ops import (
    PreparedEdgeResolver,
    _call_prepared_edge_callback,
    _trusted_edge_resolver_adapter,
    call_prepared_edge_resolver,
)
from .frame_alignment_policy_ops import align_frame_pair_by_policy
from .kinematics_path_support_ops import KinematicsPathSupportContext
from .path_configuration import SelectedPathConfiguration


def _edge_rotation_from_pose(prepared: PreparedEdgeResolver, *, owner: str):
    if prepared.resolver is None:
        return None
    from ..pose import Pose

    def _resolver(child, parent):
        payload = call_prepared_edge_resolver(prepared, child, parent, owner=owner)
        try:
            pose = payload if isinstance(payload, Pose) else Pose(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{owner}: edge_pose_fn must return Pose-coercible payload.") from exc
        return pose.decompose(validate=False)[1]

    return _trusted_edge_resolver_adapter(_resolver)


def _accumulator_like(source, *, owner: str) -> xr.DataArray:
    source_ds = analysis_object_dataset(source)
    var_name = select_single_numeric_var(source_ds, owner=owner, what="kinematic vector")
    return xr.zeros_like(source_ds[var_name])


def _coerce_velocity_payload(payload, *, owner: str):
    from ..velocity import Velocity

    if isinstance(payload, Velocity):
        return payload
    raise ValueError(f"{owner}: edge_velocity_fn must return Velocity payloads.")


def _coerce_acceleration_payload(payload, *, owner: str):
    from ..acceleration import Acceleration

    if isinstance(payload, Acceleration):
        return payload
    raise ValueError(f"{owner}: edge_acceleration_fn must return Acceleration payloads.")


def _add_by_policy(
    left_value,
    right_value,
    *,
    left: xr.DataArray,
    right: xr.DataArray,
    core_dim: str,
    owner: str,
    what: str,
) -> xr.DataArray:
    left_aligned, right_aligned = align_frame_pair_by_policy(
        left_value,
        right_value,
        left_da=left,
        right_da=right,
        left_core_dim=core_dim,
        right_core_dim=core_dim,
        owner=owner,
        what=what,
        operation_family="spatial.kinematics.path_coupling",
    )
    return left_aligned + right_aligned


def _motion_component(
    payload,
    *,
    operation: str,
    component: str,
    target_dim: str,
    src_parent: Frame,
    edge_rot_fn,
    configuration: SelectedPathConfiguration,
    owner: str,
) -> xr.DataArray:
    expressed = _express_motion_in_selected_basis(
        payload,
        dst=src_parent,
        edge_rotation_fn=edge_rot_fn,
        configuration=configuration,
        owner=owner,
    )
    target = expressed.linear(validate=False) if component == "linear" else expressed.angular(validate=False)
    target_ds = analysis_object_dataset(target)
    var_name, payload_dim = resolve_single_numeric_var_single_core_dim(
        target_ds,
        owner=owner,
        what=f"{operation} {component} payload",
    )
    out = target_ds[var_name]
    if payload_dim != target_dim:
        return out.rename({payload_dim: target_dim})
    return out


def _express_motion_in_selected_basis(
    payload,
    *,
    dst: Frame,
    edge_rotation_fn,
    configuration: SelectedPathConfiguration,
    owner: str,
):
    from ..acceleration import Acceleration
    from ..velocity import Velocity
    from .kinematics_family_frame_ops import (
        FamilyFrameRequest,
        acceleration_family_parts,
        run_family_pair_operation,
        velocity_family_parts,
    )
    from .kinematics_frame_ops import _run_vector_express_in

    is_velocity = isinstance(payload, Velocity)
    request = FamilyFrameRequest(
        payload,
        dst,
        edge_rotation_fn,
        configuration.options,
        False,
        owner,
        selected_configuration=configuration,
    )
    return run_family_pair_operation(
        request,
        parts_resolver=velocity_family_parts if is_velocity else acceleration_family_parts,
        member_runner=_run_vector_express_in,
        compose=Velocity.from_linear_angular if is_velocity else Acceleration.from_linear_angular,
        operation="express_in",
    )


def _edge_motion_payload(step, motion_class: str, *, context: KinematicsPathSupportContext, owner: str):
    callback = context.motion_callback
    if context.operation == "velocity":
        if motion_class not in {"galilean", "dynamic"}:
            return None
        if callback is None:
            raise RuntimeError("velocity support preflight must prepare edge_velocity_fn")
        payload = _call_prepared_edge_callback(callback, step.child, step.parent, owner=owner)
        return _coerce_velocity_payload(payload, owner=owner)
    if motion_class != "dynamic":
        return None
    if callback is None:
        raise RuntimeError("acceleration support preflight must prepare edge_acceleration_fn")
    payload = _call_prepared_edge_callback(callback, step.child, step.parent, owner=owner)
    return _coerce_acceleration_payload(payload, owner=owner)


def _accumulate_parent_motion(
    source_value,
    source,
    *,
    context: KinematicsPathSupportContext,
    component: str,
    edge_pose_resolver: PreparedEdgeResolver,
    configuration: SelectedPathConfiguration,
    owner: str,
) -> xr.DataArray:
    _, target_dim = resolve_single_numeric_var_single_core_dim(
        analysis_object_dataset(source),
        owner=owner,
        what="kinematic vector",
    )
    acc = _accumulator_like(source, owner=owner)
    edge_rot_fn = _edge_rotation_from_pose(edge_pose_resolver, owner=owner)
    for step, motion_class in zip(context.path.steps, context.edge_classes, strict=True):
        sign = -1.0 if step.invert else 1.0
        payload = _edge_motion_payload(step, motion_class, context=context, owner=owner)
        if payload is None:
            continue
        acc = _add_by_policy(
            source_value,
            payload,
            right=sign
            * _motion_component(
                payload,
                operation=context.operation,
                component=component,
                target_dim=target_dim,
                src_parent=context.src_parent,
                edge_rot_fn=edge_rot_fn,
                configuration=configuration,
                owner=owner,
            ),
            left=acc,
            core_dim=target_dim,
            owner=owner,
            what="kinematic path coupling accumulation",
        )
    return acc


def _target_component(source, *, owner: str) -> str:
    kind = get_kinematics_kind(analysis_object_dataset(source), owner=owner)
    if kind is None:
        raise ValueError(f"{owner}: source kinematics kind metadata is required.")
    if kind.startswith("linear_"):
        return "linear"
    if kind.startswith("angular_"):
        return "angular"
    raise ValueError(f"{owner}: unsupported source kind for vector-path coupling: {kind!r}.")


def apply_vector_path_coupling(
    source,
    prepared,
    *,
    context: KinematicsPathSupportContext,
    edge_pose_resolver: PreparedEdgeResolver,
    configuration: SelectedPathConfiguration,
    owner: str,
):
    if not context.path.steps:
        return prepared
    component = _target_component(source, owner=owner)
    add = _accumulate_parent_motion(
        source,
        prepared,
        context=context,
        component=component,
        edge_pose_resolver=edge_pose_resolver,
        configuration=configuration,
        owner=owner,
    )
    prepared_ds = analysis_object_dataset(prepared)
    var_name = select_single_numeric_var(prepared_ds, owner=owner, what="kinematic vector")
    ds = prepared_ds.copy()
    _, target_dim = resolve_single_numeric_var_single_core_dim(
        prepared_ds,
        owner=owner,
        what="kinematic vector",
    )
    ds[var_name] = _add_by_policy(
        source,
        source,
        left=ds[var_name],
        right=add,
        core_dim=target_dim,
        owner=owner,
        what="kinematic path coupling output",
    )
    return wrap_like(prepared, ds, validate=False)


__all__ = ["apply_vector_path_coupling"]
