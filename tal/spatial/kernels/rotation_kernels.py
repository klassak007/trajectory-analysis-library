from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from .rigid_matrix_validation import validate_rotation_matrix_rows

_QUAT_SIZE = 4
_MATRIX_SIZE = 3


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


def quat_to_matrix_kernel(values: np.ndarray) -> np.ndarray:
    flat, leading = _reshape_quat_input(values)
    try:
        out = SciRotation.from_quat(flat).as_matrix()
    except ValueError as exc:
        raise ValueError(f"spatial.rotation.kernel.quat_to_matrix: invalid quaternion input: {exc}") from exc
    return out.reshape(leading + (_MATRIX_SIZE, _MATRIX_SIZE))


def matrix_to_quat_kernel(values: np.ndarray) -> np.ndarray:
    flat, leading = _reshape_matrix_input(values)
    validate_rotation_matrix_rows(flat, owner="spatial.rotation.kernel.matrix_to_quat")
    return _matrix_to_quat_prevalidated(flat, leading=leading)


def _matrix_to_quat_prevalidated(
    flat: np.ndarray,
    *,
    leading: tuple[int, ...],
) -> np.ndarray:
    try:
        out = SciRotation.from_matrix(flat).as_quat()
    except ValueError as exc:
        raise ValueError(f"spatial.rotation.kernel.matrix_to_quat: invalid rotation matrix input: {exc}") from exc
    return out.reshape(leading + (_QUAT_SIZE,))


def _matrix_to_quat_prevalidated_kernel(values: np.ndarray) -> np.ndarray:
    """Convert already validated rotation matrices to quaternion form."""
    flat, leading = _reshape_matrix_input(values)
    return _matrix_to_quat_prevalidated(flat, leading=leading)


__all__ = ["matrix_to_quat_kernel", "quat_to_matrix_kernel"]
