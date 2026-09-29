from __future__ import annotations

import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.alignment_intent import OperationTopologyIntent
from tal.core.orchestration.runtime_checks import (
    resolve_single_numeric_var_single_core_dim,
)

from ..position import Position
from ..rotation import Rotation
from .numerical_validity import numerical_valid_mask, safe_rotation_values


def rotation_with_selection_intents(rotation: Rotation, *, selection: OperationTopologyIntent) -> Rotation:
    """Carry a composite operation's resolved intent into its Rotation delegate."""
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


def compose_rotation(left: Rotation, right: Rotation, *, selection: OperationTopologyIntent, owner: str) -> Rotation:
    try:
        return rotation_with_selection_intents(left, selection=selection).compose(right, validate=False)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose rotation compose failed: {exc}") from exc


def inverse_rotation(rotation: Rotation, *, owner: str) -> Rotation:
    try:
        return rotation.inverse(validate=False)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose inverse rotation failed: {exc}") from exc


def safe_pose_operands(
    position_ds: xr.Dataset,
    rotation_ds: xr.Dataset,
    *,
    pos_var: str,
    quat_var: str,
    quat_dim: str,
) -> tuple[xr.DataArray, Rotation]:
    """Prepare finite placeholders for unreachable Pose numerical rows."""
    valid = numerical_valid_mask(rotation_ds)
    if valid is None:
        return position_ds[pos_var], Rotation._from_unvalidated(rotation_ds)
    translation = position_ds[pos_var].where(valid, 0.0)
    quaternion = safe_rotation_values(rotation_ds[quat_var], core_dims=(quat_dim,), valid=valid)
    safe_rotation = Rotation._from_unvalidated(rotation_ds.assign({quat_var: quaternion.variable}))
    return translation, safe_rotation


def safe_pose_components(
    position: Position,
    rotation: Rotation,
    *,
    owner: str,
) -> tuple[Position, Rotation]:
    """Replace only structurally unreachable component rows."""
    position_ds = analysis_object_dataset(position)
    rotation_ds = analysis_object_dataset(rotation)
    pos_var, _ = resolve_single_numeric_var_single_core_dim(
        position_ds, owner=owner, what="Pose translation"
    )
    quat_var, quat_dim = resolve_single_numeric_var_single_core_dim(
        rotation_ds, owner=owner, what="Pose rotation"
    )
    translation, safe_rotation = safe_pose_operands(
        position_ds,
        rotation_ds,
        pos_var=pos_var,
        quat_var=quat_var,
        quat_dim=quat_dim,
    )
    safe_position = Position._from_unvalidated(position_ds.assign({pos_var: translation.variable}))
    return safe_position, safe_rotation
