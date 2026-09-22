from __future__ import annotations

from functools import partial

import numpy as np
import xarray as xr

from ..kernels.fixed_size_backends import (
    pose_components_to_matrix_block_backend,
    pose_compose_translation_block_backend,
    pose_inverse_translation_block_backend,
)
from ..kernels.pose_kernels import _matrix_to_components_prevalidated_kernel
from .core_chunks import single_core_chunk

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")
_MATRIX_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")


def _wrap_components_to_matrix_kernel(
    translation: np.ndarray,
    quat: np.ndarray,
    *,
    owner: str,
) -> np.ndarray:
    try:
        return pose_components_to_matrix_block_backend(translation, quat)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose components->matrix conversion kernel failed.") from exc


def _wrap_compose_translation_kernel(
    left_t: np.ndarray,
    right_t: np.ndarray,
    right_quat: np.ndarray,
    *,
    owner: str,
) -> np.ndarray:
    try:
        return pose_compose_translation_block_backend(left_t, right_t, right_quat)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose compose translation kernel failed.") from exc


def _wrap_inverse_translation_kernel(
    translation: np.ndarray,
    quat: np.ndarray,
    *,
    owner: str,
) -> np.ndarray:
    try:
        return pose_inverse_translation_block_backend(translation, quat)
    except ValueError as exc:
        raise ValueError(f"{owner}: pose inverse translation failed.") from exc


def apply_components_to_matrix_kernel(
    pos_da: xr.DataArray,
    rot_da: xr.DataArray,
    *,
    pos_dim: str,
    quat_dim: str,
    row_dim: str,
    col_dim: str,
    owner: str,
) -> xr.DataArray:
    pos_da = single_core_chunk(pos_da, dim=pos_dim)
    rot_da = single_core_chunk(rot_da, dim=quat_dim)
    matrix = xr.apply_ufunc(
        partial(_wrap_components_to_matrix_kernel, owner=owner),
        pos_da,
        rot_da,
        input_core_dims=[[pos_dim], [quat_dim]],
        output_core_dims=[[row_dim, col_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {row_dim: 4, col_dim: 4}},
    )
    return matrix.assign_coords({row_dim: list(_MATRIX_LABELS), col_dim: list(_MATRIX_LABELS)})


def apply_matrix_to_components_kernel(
    matrix: xr.DataArray,
    *,
    row_dim: str,
    col_dim: str,
    pos_dim: str,
    quat_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Split validated homogeneous matrices without constructing typed wrappers."""
    matrix = single_core_chunk(single_core_chunk(matrix, dim=row_dim), dim=col_dim)
    position, quaternion = xr.apply_ufunc(
        _matrix_to_components_prevalidated_kernel,
        matrix,
        input_core_dims=[[row_dim, col_dim]],
        output_core_dims=[[pos_dim], [quat_dim]],
        exclude_dims={row_dim} if pos_dim == row_dim else set(),
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64],
        dask_gufunc_kwargs={"output_sizes": {pos_dim: 3, quat_dim: 4}},
    )
    return (
        position.assign_coords({pos_dim: list(_XYZ_LABELS)}),
        quaternion.assign_coords({quat_dim: ["x", "y", "z", "w"]}),
    )


def apply_pose_compose_translation_kernel(
    left_t: xr.DataArray,
    right_t: xr.DataArray,
    right_q: xr.DataArray,
    *,
    left_dim: str,
    right_dim: str,
    right_quat_dim: str,
    owner: str,
) -> xr.DataArray:
    left_t = single_core_chunk(left_t, dim=left_dim)
    right_t = single_core_chunk(right_t, dim=right_dim)
    right_q = single_core_chunk(right_q, dim=right_quat_dim)
    out_t = xr.apply_ufunc(
        partial(_wrap_compose_translation_kernel, owner=owner),
        left_t,
        right_t,
        right_q,
        input_core_dims=[[left_dim], [right_dim], [right_quat_dim]],
        output_core_dims=[[left_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {left_dim: 3},
        },
    )
    return out_t.assign_coords({left_dim: list(_XYZ_LABELS)})


def apply_pose_inverse_translation_kernel(
    translation: xr.DataArray,
    quat: xr.DataArray,
    *,
    pos_dim: str,
    quat_dim: str,
    owner: str,
) -> xr.DataArray:
    translation = single_core_chunk(translation, dim=pos_dim)
    quat = single_core_chunk(quat, dim=quat_dim)
    out_t = xr.apply_ufunc(
        partial(_wrap_inverse_translation_kernel, owner=owner),
        translation,
        quat,
        input_core_dims=[[pos_dim], [quat_dim]],
        output_core_dims=[[pos_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {pos_dim: 3}},
    )
    return out_t.assign_coords({pos_dim: list(_XYZ_LABELS)})
