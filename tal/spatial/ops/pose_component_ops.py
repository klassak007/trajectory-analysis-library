from __future__ import annotations

import xarray as xr

from tal.core.orchestration.topology import TopologyPolicy

from ..kinematics.paired_components import (
    PairAssemblyOptions,
    align_paired_component_payloads,
    build_paired_components_dataset,
    component_var_names,
    resolve_component_spec,
    resolve_pair_registry,
    resolve_paired_roles,
)

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")
_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")
_POSE_PAIR_OPTS = PairAssemblyOptions(
    left_what="Position",
    right_what="Rotation",
    pair_what="pose",
    left_component_name="position",
    right_component_name="rotation",
    left_expected_labels=_XYZ_LABELS,
    right_expected_labels=_QUAT_LABELS,
)


def resolve_pose_component_specs(
    ds: xr.Dataset,
    *,
    owner: str,
    core_dims: tuple[str, ...],
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Resolve registered Position and Rotation payloads for one Pose."""
    registry = resolve_pair_registry(
        ds,
        owner=owner,
        pair_what="Pose",
        left_component_name=_POSE_PAIR_OPTS.left_component_name,
        right_component_name=_POSE_PAIR_OPTS.right_component_name,
    )
    position = resolve_component_spec(
        registry[_POSE_PAIR_OPTS.left_component_name],
        component_name=_POSE_PAIR_OPTS.left_component_name,
        core_dims=core_dims,
        expected_labels=_POSE_PAIR_OPTS.left_expected_labels,
        owner=owner,
        pair_what="Pose",
    )
    rotation = resolve_component_spec(
        registry[_POSE_PAIR_OPTS.right_component_name],
        component_name=_POSE_PAIR_OPTS.right_component_name,
        core_dims=core_dims,
        expected_labels=_POSE_PAIR_OPTS.right_expected_labels,
        owner=owner,
        pair_what="Pose",
    )
    return position, rotation


def build_components_pose_dataset(
    rotation_ds: xr.Dataset,
    position_ds: xr.Dataset,
    *,
    owner: str,
    validate: bool,
    policy: TopologyPolicy,
) -> xr.Dataset:
    """Align and assemble one component-representation Pose Dataset."""
    pos_dim, rot_dim = resolve_paired_roles(
        position_ds,
        rotation_ds,
        owner=owner,
        left_what="Position",
        right_what="Rotation",
    )
    pos_var, rot_var = component_var_names(
        position_ds,
        rotation_ds,
        owner=owner,
        opts=_POSE_PAIR_OPTS,
    )
    pos_ds, rot_ds, sequence_dim, batch_dims = align_paired_component_payloads(
        position_ds,
        rotation_ds,
        left_var=pos_var,
        right_var=rot_var,
        left_dim=pos_dim,
        right_dim=rot_dim,
        owner=owner,
        opts=_POSE_PAIR_OPTS,
        policy=policy,
    )
    return build_paired_components_dataset(
        left_ds=pos_ds,
        right_ds=rot_ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        left_dim=pos_dim,
        right_dim=rot_dim,
        left_var=pos_var,
        right_var=rot_var,
        owner=owner,
        validate=validate,
        opts=_POSE_PAIR_OPTS,
        policy=policy,
    )


__all__ = ["build_components_pose_dataset", "resolve_pose_component_specs"]
