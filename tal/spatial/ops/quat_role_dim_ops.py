from __future__ import annotations

import xarray as xr

from tal.core.orchestration.runtime_checks import (
    require_single_core_dim_with_length,
    select_single_numeric_var,
)
from tal.core.schema_read import read_roles

from ..metadata import get_rotation_rep


def require_rotation_ingress_core_roles(ds: xr.Dataset, *, owner: str) -> None:
    """Keep public quaternion ingress strict; matrix invariants enforce two roles."""
    if get_rotation_rep(ds, owner=owner) == "quat":
        require_single_core_dim_with_length(ds, expected_length=4, owner=owner, what="Rotation")


def require_matrix_core_dims(ds: xr.Dataset, *, owner: str) -> tuple[str, str]:
    """Resolve and validate the declared matrix component dimensions."""
    declared, _, _, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: Rotation requires declared roles.")
    if len(core_dims) != 2:
        raise ValueError(
            f"{owner}: Rotation matrix layout requires exactly two core dims; got {core_dims!r}."
        )
    row_dim, col_dim = core_dims
    if row_dim == col_dim:
        raise ValueError(
            f"{owner}: Rotation matrix layout core dims must be distinct; got {core_dims!r}."
        )
    if int(ds.sizes.get(row_dim, -1)) != 3 or int(ds.sizes.get(col_dim, -1)) != 3:
        raise ValueError(f"{owner}: Rotation matrix core dims must both have length 3.")
    return row_dim, col_dim


def resolve_quat_dim_with_role_fallback(
    ds: xr.Dataset,
    *,
    var_name: str,
    owner: str,
    what: str,
) -> str:
    _, sequence_dim, batch_dims, core_dims = read_roles(ds)
    if sequence_dim is not None or batch_dims or core_dims:
        return require_single_core_dim_with_length(ds, expected_length=4, owner=owner, what=what)
    candidates = tuple(dim for dim in ds[var_name].dims if int(ds.sizes.get(dim, -1)) == 4)
    if len(candidates) != 1:
        raise ValueError(
            f"{owner}: {what} without a declared quaternion core role requires exactly one length-4 quaternion dim "
            f"on variable {var_name!r}; got candidates={candidates!r}."
        )
    return candidates[0]


def resolve_rotation_component_dims_for_reduce(
    ds: xr.Dataset,
    *,
    owner: str,
) -> tuple[str, ...]:
    rep = get_rotation_rep(ds, owner=owner)
    if rep == "quat":
        var_name = select_single_numeric_var(ds, owner=owner, what="Rotation.reduce component guard")
        quat_dim = resolve_quat_dim_with_role_fallback(ds, var_name=var_name, owner=owner, what="Rotation")
        return (quat_dim,)
    if rep == "matrix":
        return require_matrix_core_dims(ds, owner=owner)
    raise ValueError(f"{owner}: unsupported rotation representation {rep!r} for reduce component guard.")


__all__ = [
    "require_matrix_core_dims",
    "require_rotation_ingress_core_roles",
    "resolve_quat_dim_with_role_fallback",
    "resolve_rotation_component_dims_for_reduce",
]
