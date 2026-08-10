"""Pose-matrix validation orchestration with structural-validity scoping."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core.orchestration.runtime_checks import require_var_contains_dims, select_single_numeric_var
from tal.core.schema_read import read_roles, read_sequence_size_coord_name
from tal.core.validity_mask import resolve_structural_valid_mask_base

from ..kernels.rigid_matrix_validation import validate_pose_matrix_block


def _matrix_spec(ds: xr.Dataset, *, owner: str) -> tuple[str, str, str, xr.DataArray]:
    declared, _, _, core_dims = read_roles(ds)
    if not declared or len(core_dims) != 2:
        raise ValueError(f"{owner}: Pose matrix layout requires exactly two declared core dims.")
    row_dim, col_dim = core_dims
    var_name = select_single_numeric_var(ds, owner=owner, what="Pose matrix layout")
    require_var_contains_dims(
        ds,
        var_name=var_name,
        required_dims=(row_dim, col_dim),
        owner=owner,
        what="Pose matrix layout",
    )
    return var_name, row_dim, col_dim, ds[var_name]


def _validation_scope(
    ds: xr.Dataset,
    matrix: xr.DataArray,
    *,
    owner: str,
) -> xr.DataArray:
    _, sequence_dim, _, _ = read_roles(ds)
    size_name = read_sequence_size_coord_name(ds)
    mask = resolve_structural_valid_mask_base(
        ds,
        sequence_dim=sequence_dim,
        sequence_size_coord=size_name,
        owner=owner,
    )
    if mask is None:
        return xr.DataArray(True)
    omitted = tuple(dim for dim in mask.dims if dim not in matrix.dims)
    return mask.any(dim=omitted) if omitted else mask


def _validate_matrix(
    ds: xr.Dataset,
    *,
    owner: str,
    sanitize_invalid: bool,
) -> tuple[str, xr.DataArray]:
    var_name, row_dim, col_dim, matrix = _matrix_spec(ds, owner=owner)
    scope = _validation_scope(ds, matrix, owner=owner)
    out = xr.apply_ufunc(
        validate_pose_matrix_block,
        matrix,
        scope,
        kwargs={"owner": owner, "sanitize_invalid": sanitize_invalid},
        input_core_dims=[[row_dim, col_dim], []],
        output_core_dims=[[row_dim, col_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.dtype(matrix.dtype)],
        keep_attrs=True,
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).transpose(*matrix.dims)
    coords = {
        name: coord
        for name, coord in matrix.coords.items()
        if set(coord.dims).issubset(out.dims)
    }
    out = out.assign_coords(coords)
    out.name = matrix.name
    return var_name, out


def validate_pose_matrix_dataset(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    """Attach eager or lazy pass-through validation to a matrix-layout dataset."""
    var_name, matrix = _validate_matrix(ds, owner=owner, sanitize_invalid=False)
    return ds.assign({var_name: matrix})


def prepare_pose_matrix_for_conversion(ds: xr.Dataset, *, owner: str) -> xr.DataArray:
    """Validate active matrices and sanitize structurally invalid rows for kernels."""
    _, matrix = _validate_matrix(ds, owner=owner, sanitize_invalid=True)
    return matrix


__all__ = ["prepare_pose_matrix_for_conversion", "validate_pose_matrix_dataset"]
