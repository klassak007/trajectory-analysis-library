from __future__ import annotations

import xarray as xr

from tal.core.orchestration.runtime_checks import (
    require_single_core_dim_with_length,
    select_single_numeric_var,
)
from tal.core.schema_read import read_roles

from ..metadata import get_rotation_rep


def resolve_quat_dim_with_role_fallback(
    ds: xr.Dataset,
    *,
    var_name: str,
    owner: str,
    what: str,
) -> str:
    declared, sequence_dim, _, _ = read_roles(ds)
    if declared and sequence_dim is not None:
        return require_single_core_dim_with_length(ds, expected_length=4, owner=owner, what=what)
    candidates = tuple(dim for dim in ds[var_name].dims if int(ds.sizes.get(dim, -1)) == 4)
    if len(candidates) != 1:
        raise ValueError(
            f"{owner}: {what} without declared sequence roles requires exactly one length-4 quaternion dim "
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
        declared, sequence_dim, _, core_dims = read_roles(ds)
        if not declared or sequence_dim is None:
            raise ValueError(f"{owner}: matrix Rotation.reduce component guard requires declared roles with sequence_dim.")
        if len(core_dims) != 2:
            raise ValueError(
                f"{owner}: matrix Rotation.reduce component guard requires exactly two core dims; got {core_dims!r}."
            )
        row_dim, col_dim = core_dims
        if row_dim == col_dim:
            raise ValueError(f"{owner}: matrix Rotation.reduce component guard requires distinct row/col core dims.")
        if row_dim not in ds.dims or col_dim not in ds.dims:
            raise ValueError(f"{owner}: matrix Rotation.reduce component guard core dims must exist on dataset.")
        if int(ds.sizes[row_dim]) != 3 or int(ds.sizes[col_dim]) != 3:
            raise ValueError(f"{owner}: matrix Rotation.reduce component guard requires 3x3 matrix core dims.")
        return (row_dim, col_dim)
    raise ValueError(f"{owner}: unsupported rotation representation {rep!r} for reduce component guard.")


__all__ = ["resolve_quat_dim_with_role_fallback", "resolve_rotation_component_dims_for_reduce"]
