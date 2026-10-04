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
from .edge_resolver_ops import PreparedEdgeResolver
from .frame_alignment_policy_ops import align_frame_pair_by_policy
from .kinematics_path_support_ops import KinematicsPathSupportContext
from .path_configuration import SelectedPathConfiguration


def _accumulator_like(source, *, owner: str) -> xr.DataArray:
    source_ds = analysis_object_dataset(source)
    var_name = select_single_numeric_var(source_ds, owner=owner, what="kinematic vector")
    return xr.zeros_like(source_ds[var_name])


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
    owner: str,
) -> xr.DataArray:
    target = payload.linear(validate=False) if component == "linear" else payload.angular(validate=False)
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
    prepared_path,
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
        None,
        prepared_path.configuration.options,
        False,
        owner,
        selected_configuration=prepared_path.configuration,
    )
    return run_family_pair_operation(
        request,
        parts_resolver=velocity_family_parts if is_velocity else acceleration_family_parts,
        member_runner=_run_vector_express_in,
        compose=Velocity.from_linear_angular if is_velocity else Acceleration.from_linear_angular,
        operation="express_in",
        prepared_path=prepared_path,
    )


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
    _ = edge_pose_resolver, configuration
    occurrences = zip(context.path.steps, context.motion_payloads, strict=True)
    for step, payload in occurrences:
        sign = -1.0 if step.invert else 1.0
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
