from __future__ import annotations

import numpy as np

from ._fixed_size_common import (
    prepare_binary_quat_rows,
    prepare_pose_compose_rows,
    prepare_pose_inverse_rows,
    prepare_pose_matrix_rows,
    prepare_quat_to_matrix_rows,
    prepare_unary_quat_rows,
    prepare_vec_quat_rows,
    validate_quat_rows,
)
from .pose_kernels import components_to_matrix_kernel, compose_translation_kernel, inverse_translation_kernel
from .rotation_apply_kernels import rotate_vec3_kernel
from .rotation_compose_kernels import compose_quat_kernel, inverse_quat_kernel
from .rotation_kernels import _matrix_to_quat_prevalidated_kernel, quat_to_matrix_kernel
from .rigid_matrix_validation import require_real_matrix_dtype, validate_rotation_matrix_rows

SPATIAL_FIXED_BACKEND_NUMBA = "numba"
SPATIAL_FIXED_BACKEND_SCIPY = "scipy"
_OWNER = "spatial.fixed_size_backend"


def _run_scipy_kernel(func, *args: np.ndarray) -> np.ndarray:
    try:
        return func(*args)
    except ValueError as exc:
        raise ValueError(f"{_OWNER}: baseline fixed-size scipy kernel failed: {exc}") from exc


def quat_compose_block_backend(left: object, right: object, *, backend: str = SPATIAL_FIXED_BACKEND_SCIPY) -> np.ndarray:
    if backend == SPATIAL_FIXED_BACKEND_SCIPY:
        rows = prepare_binary_quat_rows(left, right, owner=_OWNER)
        validate_quat_rows(*rows.row_arrays, owner=_OWNER)
        return _run_scipy_kernel(
            compose_quat_kernel,
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
        )
    if backend == SPATIAL_FIXED_BACKEND_NUMBA:
        from .fixed_size_numba_backends import quat_compose_block_numba

        return quat_compose_block_numba(left, right, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported fixed-size backend {backend!r}.")


def quat_inverse_block_backend(values: object, *, backend: str = SPATIAL_FIXED_BACKEND_SCIPY) -> np.ndarray:
    if backend == SPATIAL_FIXED_BACKEND_SCIPY:
        rows = prepare_unary_quat_rows(values, owner=_OWNER)
        validate_quat_rows(rows.row_arrays[0], owner=_OWNER)
        return _run_scipy_kernel(inverse_quat_kernel, np.asarray(values, dtype=np.float64))
    if backend == SPATIAL_FIXED_BACKEND_NUMBA:
        from .fixed_size_numba_backends import quat_inverse_block_numba

        return quat_inverse_block_numba(values, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported fixed-size backend {backend!r}.")


def quat_to_matrix_block_backend(values: object, *, backend: str = SPATIAL_FIXED_BACKEND_SCIPY) -> np.ndarray:
    if backend == SPATIAL_FIXED_BACKEND_SCIPY:
        rows = prepare_quat_to_matrix_rows(values, owner=_OWNER)
        validate_quat_rows(rows.row_arrays[0], owner=_OWNER)
        return _run_scipy_kernel(quat_to_matrix_kernel, np.asarray(values, dtype=np.float64))
    if backend == SPATIAL_FIXED_BACKEND_NUMBA:
        from .fixed_size_numba_backends import quat_to_matrix_block_numba

        return quat_to_matrix_block_numba(values, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported fixed-size backend {backend!r}.")


def matrix_to_quat_block_backend(matrix: object, *, backend: str = SPATIAL_FIXED_BACKEND_SCIPY) -> np.ndarray:
    if backend not in (SPATIAL_FIXED_BACKEND_SCIPY, SPATIAL_FIXED_BACKEND_NUMBA):
        raise ValueError(f"{_OWNER}: unsupported fixed-size backend {backend!r}.")
    if backend == SPATIAL_FIXED_BACKEND_SCIPY:
        raw = np.asarray(matrix)
        require_real_matrix_dtype(raw, owner=_OWNER)
        validate_rotation_matrix_rows(raw, owner=_OWNER)
        values = raw.astype(np.float64, copy=False)
        return _run_scipy_kernel(_matrix_to_quat_prevalidated_kernel, values)
    from .fixed_size_numba_backends import matrix_to_quat_block_numba

    return matrix_to_quat_block_numba(matrix, owner=_OWNER)


def rotate_vec3_block_backend(values: object, quat: object, *, backend: str = SPATIAL_FIXED_BACKEND_SCIPY) -> np.ndarray:
    if backend == SPATIAL_FIXED_BACKEND_SCIPY:
        rows = prepare_vec_quat_rows(values, quat, owner=_OWNER)
        validate_quat_rows(rows.row_arrays[1], owner=_OWNER)
        return _run_scipy_kernel(
            rotate_vec3_kernel,
            np.asarray(values, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
        )
    if backend == SPATIAL_FIXED_BACKEND_NUMBA:
        from .fixed_size_numba_backends import rotate_vec3_block_numba

        return rotate_vec3_block_numba(values, quat, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported fixed-size backend {backend!r}.")


def pose_compose_translation_block_backend(
    left_t: object,
    right_t: object,
    right_q: object,
    *,
    backend: str = SPATIAL_FIXED_BACKEND_SCIPY,
) -> np.ndarray:
    if backend == SPATIAL_FIXED_BACKEND_SCIPY:
        rows = prepare_pose_compose_rows(left_t, right_t, right_q, owner=_OWNER)
        validate_quat_rows(rows.row_arrays[2], owner=_OWNER)
        return _run_scipy_kernel(
            compose_translation_kernel,
            np.asarray(left_t, dtype=np.float64),
            np.asarray(right_t, dtype=np.float64),
            np.asarray(right_q, dtype=np.float64),
        )
    if backend == SPATIAL_FIXED_BACKEND_NUMBA:
        from .fixed_size_numba_backends import pose_compose_translation_block_numba

        return pose_compose_translation_block_numba(left_t, right_t, right_q, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported fixed-size backend {backend!r}.")


def pose_inverse_translation_block_backend(
    translation: object,
    quat: object,
    *,
    backend: str = SPATIAL_FIXED_BACKEND_SCIPY,
) -> np.ndarray:
    if backend == SPATIAL_FIXED_BACKEND_SCIPY:
        rows = prepare_pose_inverse_rows(translation, quat, owner=_OWNER)
        validate_quat_rows(rows.row_arrays[1], owner=_OWNER)
        return _run_scipy_kernel(
            inverse_translation_kernel,
            np.asarray(translation, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
        )
    if backend == SPATIAL_FIXED_BACKEND_NUMBA:
        from .fixed_size_numba_backends import pose_inverse_translation_block_numba

        return pose_inverse_translation_block_numba(translation, quat, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported fixed-size backend {backend!r}.")


def pose_components_to_matrix_block_backend(
    translation: object,
    quat: object,
    *,
    backend: str = SPATIAL_FIXED_BACKEND_SCIPY,
) -> np.ndarray:
    if backend == SPATIAL_FIXED_BACKEND_SCIPY:
        rows = prepare_pose_matrix_rows(translation, quat, owner=_OWNER)
        validate_quat_rows(rows.row_arrays[1], owner=_OWNER)
        return _run_scipy_kernel(
            components_to_matrix_kernel,
            np.asarray(translation, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
        )
    if backend == SPATIAL_FIXED_BACKEND_NUMBA:
        from .fixed_size_numba_backends import pose_components_to_matrix_block_numba

        return pose_components_to_matrix_block_numba(translation, quat, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported fixed-size backend {backend!r}.")


__all__ = [
    "SPATIAL_FIXED_BACKEND_NUMBA",
    "SPATIAL_FIXED_BACKEND_SCIPY",
    "matrix_to_quat_block_backend",
    "pose_compose_translation_block_backend",
    "pose_components_to_matrix_block_backend",
    "pose_inverse_translation_block_backend",
    "quat_compose_block_backend",
    "quat_inverse_block_backend",
    "quat_to_matrix_block_backend",
    "rotate_vec3_block_backend",
]
