from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

_VEC3_SIZE = 3
_QUAT_SIZE = 4
_MATRIX_SIZE = 4
_ROT_MATRIX_SIZE = 3


def _reshape_vec3(values: np.ndarray, *, owner: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.shape[-1] != _VEC3_SIZE:
        raise ValueError(f"{owner}: expected trailing vector dim length 3.")
    return values.reshape((-1, _VEC3_SIZE)), values.shape[:-1]


def _reshape_quat(values: np.ndarray, *, owner: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.shape[-1] != _QUAT_SIZE:
        raise ValueError(f"{owner}: expected trailing quaternion dim length 4.")
    return values.reshape((-1, _QUAT_SIZE)), values.shape[:-1]


def _reshape_matrix(values: np.ndarray, *, owner: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.shape[-2:] != (_MATRIX_SIZE, _MATRIX_SIZE):
        raise ValueError(f"{owner}: expected trailing matrix dims of shape (4, 4).")
    return values.reshape((-1, _MATRIX_SIZE, _MATRIX_SIZE)), values.shape[:-2]


def compose_translation_kernel(left_t: np.ndarray, right_t: np.ndarray, right_quat: np.ndarray) -> np.ndarray:
    owner = "spatial.pose.kernel.compose_translation"
    left_flat, leading = _reshape_vec3(left_t, owner=owner)
    right_flat, _ = _reshape_vec3(right_t, owner=owner)
    quat_flat, _ = _reshape_quat(right_quat, owner=owner)
    try:
        rotated = SciRotation.from_quat(quat_flat).apply(left_flat)
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid quaternion input: {exc}") from exc
    return (rotated + right_flat).reshape(leading + (_VEC3_SIZE,))


def inverse_translation_kernel(translation: np.ndarray, quat: np.ndarray) -> np.ndarray:
    owner = "spatial.pose.kernel.inverse_translation"
    translation_flat, leading = _reshape_vec3(translation, owner=owner)
    quat_flat, _ = _reshape_quat(quat, owner=owner)
    try:
        rotated = SciRotation.from_quat(quat_flat).inv().apply(translation_flat)
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid quaternion input: {exc}") from exc
    return (-rotated).reshape(leading + (_VEC3_SIZE,))


def components_to_matrix_kernel(translation: np.ndarray, quat: np.ndarray) -> np.ndarray:
    owner = "spatial.pose.kernel.components_to_matrix"
    translation_flat, leading = _reshape_vec3(translation, owner=owner)
    quat_flat, _ = _reshape_quat(quat, owner=owner)
    try:
        rotm = SciRotation.from_quat(quat_flat).as_matrix()
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid quaternion input: {exc}") from exc
    out = np.zeros((rotm.shape[0], _MATRIX_SIZE, _MATRIX_SIZE), dtype=np.float64)
    out[:, :_ROT_MATRIX_SIZE, :_ROT_MATRIX_SIZE] = rotm
    out[:, :_ROT_MATRIX_SIZE, _ROT_MATRIX_SIZE] = translation_flat
    out[:, _ROT_MATRIX_SIZE, _ROT_MATRIX_SIZE] = 1.0
    return out.reshape(leading + (_MATRIX_SIZE, _MATRIX_SIZE))


def matrix3_to_quat_kernel(matrix3: np.ndarray) -> np.ndarray:
    owner = "spatial.pose.kernel.matrix3_to_quat"
    if matrix3.shape[-2:] != (_ROT_MATRIX_SIZE, _ROT_MATRIX_SIZE):
        raise ValueError(f"{owner}: expected trailing matrix dims of shape (3, 3).")
    flat = matrix3.reshape((-1, _ROT_MATRIX_SIZE, _ROT_MATRIX_SIZE))
    try:
        quat = SciRotation.from_matrix(flat).as_quat()
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid 3x3 rotation block: {exc}") from exc
    return quat.reshape(matrix3.shape[:-2] + (_QUAT_SIZE,))


def matrix_to_components_kernel(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    owner = "spatial.pose.kernel.matrix_to_components"
    matrix_flat, leading = _reshape_matrix(matrix, owner=owner)
    rotm = matrix_flat[:, :_ROT_MATRIX_SIZE, :_ROT_MATRIX_SIZE]
    translation = matrix_flat[:, :_ROT_MATRIX_SIZE, _ROT_MATRIX_SIZE]
    if not np.isfinite(matrix_flat).all():
        raise ValueError(f"{owner}: matrix values must be finite.")
    quat = matrix3_to_quat_kernel(rotm)
    return translation.reshape(leading + (_VEC3_SIZE,)), quat.reshape(leading + (_QUAT_SIZE,))


__all__ = [
    "compose_translation_kernel",
    "components_to_matrix_kernel",
    "inverse_translation_kernel",
    "matrix3_to_quat_kernel",
    "matrix_to_components_kernel",
]
