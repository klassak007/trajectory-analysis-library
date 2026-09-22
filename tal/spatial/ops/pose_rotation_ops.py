from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.runtime_checks import (
    resolve_single_numeric_var_single_core_dim,
)
from tal.core.param_ops.guards import reserved_coord_is_owned
from tal.core.schema_read import read_roles, read_sequence_size_coord_name
from tal.core.validity_mask import resolve_validated_structural_mask_base

from ..position import Position
from ..rotation import Rotation


def compose_rotation(left: Rotation, right: Rotation, *, owner: str) -> Rotation:
    try:
        return left.compose(right, validate=False)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose rotation compose failed: {exc}") from exc


def inverse_rotation(rotation: Rotation, *, owner: str) -> Rotation:
    try:
        return rotation.inverse(validate=False)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose inverse rotation failed: {exc}") from exc


def _runtime_valid_mask(ds: xr.Dataset) -> xr.DataArray | None:
    if "valid" in ds.coords and reserved_coord_is_owned(ds, name="valid"):
        return ds.coords["valid"].astype(bool)
    _, sequence_dim, _, _ = read_roles(ds)
    return resolve_validated_structural_mask_base(
        ds,
        sequence_dim=sequence_dim,
        sequence_size_coord=read_sequence_size_coord_name(ds),
    )


def safe_pose_operands(
    position_ds: xr.Dataset,
    rotation_ds: xr.Dataset,
    *,
    pos_var: str,
    quat_var: str,
    quat_dim: str,
) -> tuple[xr.DataArray, Rotation]:
    """Prepare finite placeholders for unreachable Pose numerical rows."""
    valid = _runtime_valid_mask(rotation_ds)
    if valid is None:
        return position_ds[pos_var], Rotation._from_unvalidated(rotation_ds)
    translation = position_ds[pos_var].where(valid.broadcast_like(position_ds[pos_var]), 0.0)
    identity = xr.DataArray(
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        dims=(quat_dim,),
        coords={quat_dim: rotation_ds.coords[quat_dim]},
    )
    quaternion = xr.where(valid.broadcast_like(rotation_ds[quat_var]), rotation_ds[quat_var], identity)
    safe_rotation = Rotation._from_unvalidated(rotation_ds.assign({quat_var: quaternion}))
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
    safe_position = Position._from_unvalidated(position_ds.assign({pos_var: translation}))
    return safe_position, safe_rotation
