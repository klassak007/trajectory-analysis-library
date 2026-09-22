"""Pose subtype layout enforcement."""

from __future__ import annotations

import xarray as xr

from tal.core.component_ops.runtime_checks import require_component_numeric_var
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.utils.frame_schema import get_frames

from ..metadata import get_pose_rep, validate_spatial_roles
from .pose_component_ops import resolve_pose_component_specs
from .pose_matrix_validation import select_pose_matrix_payload

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


def _enforce_components(ds: xr.Dataset, *, owner: str) -> None:
    declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: Pose requires declared roles.")
    if len(core_dims) != 2:
        raise ValueError(
            f"{owner}: Pose components layout requires exactly two core dims; "
            f"got {core_dims!r}."
        )
    (pos_dim, pos_var), (rot_dim, rot_var) = resolve_pose_component_specs(
        ds, owner=owner, core_dims=core_dims
    )
    if pos_dim == rot_dim:
        raise ValueError(
            f"{owner}: Pose position/rotation components must use distinct core dims."
        )
    required = ((sequence_dim,) if sequence_dim is not None else ()) + batch_dims
    require_component_numeric_var(
        ds, component_name="position", var_name=pos_var,
        required_dims=required + (pos_dim,), owner=owner, operand="Pose",
    )
    require_component_numeric_var(
        ds, component_name="rotation", var_name=rot_var,
        required_dims=required + (rot_dim,), owner=owner, operand="Pose",
    )
    pos_labels = require_explicit_unique_dim_labels(
        ds, dim=pos_dim, owner=owner, what="Pose position"
    )
    rot_labels = require_explicit_unique_dim_labels(
        ds, dim=rot_dim, owner=owner, what="Pose rotation"
    )
    require_exact_labels(
        pos_labels, expected=_XYZ, owner=owner, what="Pose position core"
    )
    require_exact_labels(
        rot_labels, expected=_QUAT, owner=owner, what="Pose rotation core"
    )


def enforce_matrix_layout(
    ds: xr.Dataset,
    *,
    owner: str,
    allow_auxiliary: bool = False,
) -> None:
    declared, _, _, core_dims = read_roles(ds)
    if not declared or len(core_dims) != 2:
        raise ValueError(
            f"{owner}: Pose matrix layout requires exactly two core dims; "
            f"got {core_dims!r}."
        )
    row_dim, col_dim = core_dims
    if row_dim == col_dim:
        raise ValueError(
            f"{owner}: Pose matrix core dims must be distinct; got {core_dims!r}."
        )
    var_name = select_pose_matrix_payload(
        ds, core_dims=(row_dim, col_dim), owner=owner,
    ) if allow_auxiliary else select_single_numeric_var(
        ds, owner=owner, what="Pose matrix layout",
    )
    require_var_contains_dims(
        ds, var_name=var_name, required_dims=core_dims,
        owner=owner, what="Pose matrix layout",
    )
    if int(ds.sizes.get(row_dim, -1)) != 4 or int(ds.sizes.get(col_dim, -1)) != 4:
        raise ValueError(f"{owner}: Pose matrix core dims must both have length 4.")
    for dim in core_dims:
        labels = require_explicit_unique_dim_labels(
            ds, dim=dim, owner=owner, what="Pose matrix"
        )
        require_exact_labels(
            labels, expected=_QUAT, owner=owner,
            what=f"Pose matrix dim {dim!r}",
        )


def enforce_pose_layout(
    ds: xr.Dataset,
    *,
    owner: str,
    schema_prepared: bool = False,
    allow_matrix_auxiliary: bool = False,
) -> None:
    candidate = ds if schema_prepared else validate_schema_if_needed(ds)
    validate_spatial_roles(candidate, owner=owner)
    _ = get_frames(candidate)
    representation = get_pose_rep(candidate, owner=owner)
    if representation == "components":
        _enforce_components(candidate, owner=owner)
        return
    if representation == "matrix":
        enforce_matrix_layout(
            candidate, owner=owner, allow_auxiliary=allow_matrix_auxiliary
        )
        return
    raise ValueError(f"{owner}: unsupported pose representation {representation!r}.")


__all__: list[str] = []
