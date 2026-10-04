"""Shared validation kernels for rigid rotation and pose matrices."""

from __future__ import annotations

import numpy as np

from tal.core.ordered_dtypes import is_ordered_real_numeric_dtype

from ._fixed_size_constants import DET_ATOL, ORTHO_ATOL, POSE_MATRIX_SIZE, ROT_MATRIX_SIZE

_HOMOGENEOUS_ATOL = 1.0e-6


def require_real_matrix_dtype(values: object, *, owner: str) -> None:
    """Reject matrix payloads that are not real integer or floating-point."""
    values = np.asarray(values)
    if is_ordered_real_numeric_dtype(values.dtype):
        return
    raise ValueError(
        f"{owner}: matrix dtype must be real integer or floating-point, got {values.dtype!r}."
    )


def _require_trailing_matrix_shape(
    values: np.ndarray,
    *,
    size: int,
    owner: str,
) -> np.ndarray:
    if values.ndim < 2 or values.shape[-2:] != (size, size):
        raise ValueError(f"{owner}: matrix must have trailing shape ({size}, {size}).")
    return values.reshape((-1, size, size))


def validate_rotation_matrix_rows(values: object, *, owner: str) -> None:
    """Validate real finite proper-rotation matrices with fixed tolerances."""
    matrix = np.asarray(values)
    require_real_matrix_dtype(matrix, owner=owner)
    rows = _require_trailing_matrix_shape(matrix, size=ROT_MATRIX_SIZE, owner=owner)
    if not np.isfinite(rows).all():
        raise ValueError(f"{owner}: matrix values must be finite.")
    work = rows.astype(np.float64, copy=False)
    eye = np.eye(ROT_MATRIX_SIZE, dtype=np.float64)
    gram = np.matmul(np.swapaxes(work, -1, -2), work)
    error = np.max(np.abs(gram - eye), axis=(-2, -1))
    if np.any(error > ORTHO_ATOL):
        raise ValueError(f"{owner}: matrix is not orthonormal within tolerance.")
    determinant = np.linalg.det(work)
    if not np.isfinite(determinant).all():
        raise ValueError(f"{owner}: matrix determinants must be finite.")
    if np.any(np.abs(determinant - 1.0) > DET_ATOL):
        raise ValueError(f"{owner}: matrix determinant must be 1 within tolerance.")


def validate_pose_matrix_rows(values: object, *, owner: str) -> None:
    """Validate real finite homogeneous SE(3) matrices."""
    matrix = np.asarray(values)
    require_real_matrix_dtype(matrix, owner=owner)
    rows = _require_trailing_matrix_shape(matrix, size=POSE_MATRIX_SIZE, owner=owner)
    if not np.isfinite(rows).all():
        raise ValueError(f"{owner}: matrix values must be finite.")
    validate_rotation_matrix_rows(rows[:, :ROT_MATRIX_SIZE, :ROT_MATRIX_SIZE], owner=owner)
    expected = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    bottom = rows[:, ROT_MATRIX_SIZE, :].astype(np.float64, copy=False)
    if np.any(np.abs(bottom - expected) > _HOMOGENEOUS_ATOL):
        raise ValueError(
            f"{owner}: homogeneous bottom row must equal [0, 0, 0, 1] within tolerance."
        )


def validate_pose_matrix_block(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    owner: str,
    sanitize_invalid: bool,
) -> np.ndarray:
    """Validate active pose rows and optionally replace inactive rows by identity."""
    matrix = np.asarray(values)
    rows = _require_trailing_matrix_shape(matrix, size=POSE_MATRIX_SIZE, owner=owner)
    leading = matrix.shape[:-2]
    active = np.broadcast_to(np.asarray(valid, dtype=bool), leading).reshape(-1)
    validate_pose_matrix_rows(rows[active], owner=owner)
    if not sanitize_invalid or bool(active.all()):
        return matrix
    out = matrix.copy()
    out.reshape((-1, POSE_MATRIX_SIZE, POSE_MATRIX_SIZE))[~active] = np.eye(
        POSE_MATRIX_SIZE,
        dtype=matrix.dtype,
    )
    return out


__all__ = [
    "require_real_matrix_dtype",
    "validate_pose_matrix_block",
    "validate_pose_matrix_rows",
    "validate_rotation_matrix_rows",
]
