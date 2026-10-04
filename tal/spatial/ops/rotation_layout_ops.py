from __future__ import annotations

import xarray as xr

from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.schema_read import validate_schema_if_needed
from tal.utils.frame_schema import get_frames

from ..metadata import get_rotation_rep, validate_spatial_roles
from .quat_role_dim_ops import (
    require_matrix_core_dims,
    resolve_quat_dim_with_role_fallback,
)

_QUAT_LABELS = ("x", "y", "z", "w")
_MATRIX_LABELS = ("x", "y", "z")


def require_quat_labels(ds: xr.Dataset, *, axis: str, owner: str) -> None:
    labels = require_explicit_unique_dim_labels(ds, dim=axis, owner=owner, what="Rotation")
    require_exact_labels(labels, expected=_QUAT_LABELS, owner=owner, what="Rotation core")


def require_quat_var_and_dim(ds: xr.Dataset, *, owner: str) -> tuple[str, str]:
    var_name = select_single_numeric_var(ds, owner=owner, what="Rotation")
    quat_dim = resolve_quat_dim_with_role_fallback(ds, var_name=var_name, owner=owner, what="Rotation")
    require_var_contains_dims(ds, var_name=var_name, required_dims=(quat_dim,), owner=owner, what="Rotation")
    require_quat_labels(ds, axis=quat_dim, owner=owner)
    return var_name, quat_dim


def require_matrix_labels(ds: xr.Dataset, *, row_dim: str, col_dim: str, owner: str) -> None:
    row_labels = require_explicit_unique_dim_labels(ds, dim=row_dim, owner=owner, what="Rotation matrix")
    col_labels = require_explicit_unique_dim_labels(ds, dim=col_dim, owner=owner, what="Rotation matrix")
    require_exact_labels(row_labels, expected=_MATRIX_LABELS, owner=owner, what=f"Rotation matrix row dim {row_dim!r}")
    require_exact_labels(col_labels, expected=_MATRIX_LABELS, owner=owner, what=f"Rotation matrix col dim {col_dim!r}")


def enforce_quat_layout_invariants(ds: xr.Dataset, *, owner: str) -> None:
    require_quat_var_and_dim(ds, owner=owner)


def enforce_matrix_layout_invariants(ds: xr.Dataset, *, owner: str) -> None:
    var_name = select_single_numeric_var(ds, owner=owner, what="Rotation matrix layout")
    row_dim, col_dim = require_matrix_core_dims(ds, owner=owner)
    require_var_contains_dims(
        ds,
        var_name=var_name,
        required_dims=(row_dim, col_dim),
        owner=owner,
        what="Rotation matrix layout",
    )
    require_matrix_labels(ds, row_dim=row_dim, col_dim=col_dim, owner=owner)


def enforce_rotation_dataset_invariants(ds: xr.Dataset, *, owner: str) -> None:
    candidate = validate_schema_if_needed(ds)
    validate_spatial_roles(candidate, owner=owner)
    _ = get_frames(candidate)
    rep = get_rotation_rep(candidate, owner=owner)
    if rep == "quat":
        enforce_quat_layout_invariants(candidate, owner=owner)
        return
    if rep == "matrix":
        enforce_matrix_layout_invariants(candidate, owner=owner)
        return
    raise ValueError(f"{owner}: unsupported rotation representation {rep!r}.")
