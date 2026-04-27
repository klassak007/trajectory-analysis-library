from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

_QUAT_SIZE = 4
_MATRIX_SIZE = 3
_ORTHO_ATOL = 1e-6
_DET_ATOL = 1e-6


def _reshape_quat_input(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.shape[-1] != _QUAT_SIZE:
        raise ValueError(
            "spatial.rotation.kernel.quat_to_matrix: expected last dim length 4 for quaternion input."
        )
    return values.reshape((-1, _QUAT_SIZE)), values.shape[:-1]


def _reshape_matrix_input(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.shape[-2:] != (_MATRIX_SIZE, _MATRIX_SIZE):
        raise ValueError(
            "spatial.rotation.kernel.matrix_to_quat: expected trailing matrix dims of shape (3, 3)."
        )
    return values.reshape((-1, _MATRIX_SIZE, _MATRIX_SIZE)), values.shape[:-2]


def _validate_rotation_matrices(values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError("spatial.rotation.kernel.matrix_to_quat: matrix values must be finite.")
    eye = np.eye(_MATRIX_SIZE, dtype=np.float64)
    gram = np.matmul(np.swapaxes(values, -1, -2), values)
    ortho_error = np.max(np.abs(gram - eye), axis=(1, 2))
    if np.any(ortho_error > _ORTHO_ATOL):
        raise ValueError(
            "spatial.rotation.kernel.matrix_to_quat: matrix is not orthonormal within tolerance."
        )
    det = np.linalg.det(values)
    if not np.isfinite(det).all():
        raise ValueError("spatial.rotation.kernel.matrix_to_quat: matrix determinants must be finite.")
    if np.any(np.abs(det - 1.0) > _DET_ATOL):
        raise ValueError(
            "spatial.rotation.kernel.matrix_to_quat: matrix determinant must be 1 within tolerance."
        )


def quat_to_matrix_kernel(values: np.ndarray) -> np.ndarray:
    flat, leading = _reshape_quat_input(values)
    try:
        out = SciRotation.from_quat(flat).as_matrix()
    except ValueError as exc:
        raise ValueError(f"spatial.rotation.kernel.quat_to_matrix: invalid quaternion input: {exc}") from exc
    return out.reshape(leading + (_MATRIX_SIZE, _MATRIX_SIZE))


def matrix_to_quat_kernel(values: np.ndarray) -> np.ndarray:
    flat, leading = _reshape_matrix_input(values)
    _validate_rotation_matrices(flat)
    try:
        out = SciRotation.from_matrix(flat).as_quat()
    except ValueError as exc:
        raise ValueError(f"spatial.rotation.kernel.matrix_to_quat: invalid rotation matrix input: {exc}") from exc
    return out.reshape(leading + (_QUAT_SIZE,))


__all__ = ["matrix_to_quat_kernel", "quat_to_matrix_kernel"]
