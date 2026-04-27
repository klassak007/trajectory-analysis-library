from __future__ import annotations

import xarray as xr

from tal.core.orchestration.runtime_checks import select_single_numeric_var
from tal.core.orchestration.runtime_checks import resolve_single_numeric_var_single_core_dim

from ..metadata.roles import get_kinematics_kind
from ..path_solve import PathSolveOptions
from ..policies.wrap import wrap_like
from .frame_alignment_policy_ops import align_frame_pair_by_policy
from .kinematics_path_support_ops import KinematicsPathSupportContext


def _edge_rotation_from_pose(edge_pose_fn, *, owner: str):
    from ..pose import Pose

    def _resolver(child, parent):
        payload = edge_pose_fn(child, parent)
        try:
            pose = payload if isinstance(payload, Pose) else Pose(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{owner}: edge_pose_fn must return Pose-coercible payload.") from exc
        return pose.rotation(validate=False)

    return _resolver


def _accumulator_like(source, *, owner: str) -> xr.DataArray:
    var_name = select_single_numeric_var(source.unsafe_data, owner=owner, what="kinematic vector")
    return xr.zeros_like(source.unsafe_data[var_name])


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
    src_parent: str,
    edge_rot_fn,
    opts,
    owner: str,
) -> xr.DataArray:
    expressed = payload.express_in(src_parent, edge_rotation_fn=edge_rot_fn, opts=opts, validate=False)
    target = expressed.linear(validate=False) if component == "linear" else expressed.angular(validate=False)
    var_name, payload_dim = resolve_single_numeric_var_single_core_dim(
        target.unsafe_data,
        owner=owner,
        what=f"{operation} {component} payload",
    )
    out = target.unsafe_data[var_name]
    if payload_dim != target_dim:
        return out.rename({payload_dim: target_dim})
    return out


def _accumulate_parent_motion(
    source_value,
    source,
    *,
    context: KinematicsPathSupportContext,
    component: str,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    owner: str,
) -> xr.DataArray:
    _, target_dim = resolve_single_numeric_var_single_core_dim(
        source.unsafe_data,
        owner=owner,
        what="kinematic vector",
    )
    acc = _accumulator_like(source, owner=owner)
    edge_rot_fn = _edge_rotation_from_pose(edge_pose_fn, owner=owner)
    for step, motion_class in zip(context.path.steps, context.edge_classes, strict=True):
        sign = -1.0 if step.invert else 1.0
        if context.operation == "velocity":
            if motion_class not in {"galilean", "dynamic"}:
                continue
            payload = _coerce_velocity_payload(context.support.edge_velocity_fn(step.child, step.parent), owner=owner)
        else:
            if motion_class != "dynamic":
                continue
            payload = _coerce_acceleration_payload(
                context.support.edge_acceleration_fn(step.child, step.parent),
                owner=owner,
            )
        acc = _add_by_policy(
            source_value,
            payload,
            right=sign
            * _motion_component(
                payload,
                operation=context.operation,
                component=component,
                target_dim=target_dim,
                src_parent=context.src_parent.id,
                edge_rot_fn=edge_rot_fn,
                opts=opts,
                owner=owner,
            ),
            left=acc,
            core_dim=target_dim,
            owner=owner,
            what="kinematic path coupling accumulation",
        )
    return acc


def _target_component(source, *, owner: str) -> str:
    kind = get_kinematics_kind(source.unsafe_data, owner=owner)
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
    edge_pose_fn,
    opts: PathSolveOptions | None,
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
        edge_pose_fn=edge_pose_fn,
        opts=opts,
        owner=owner,
    )
    var_name = select_single_numeric_var(prepared.unsafe_data, owner=owner, what="kinematic vector")
    ds = prepared.unsafe_data.copy()
    _, target_dim = resolve_single_numeric_var_single_core_dim(
        prepared.unsafe_data,
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
