from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

_VEC3_SIZE = 3
_QUAT_SIZE = 4


def _reshape_vec3(values: np.ndarray, *, owner: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.shape[-1] != _VEC3_SIZE:
        raise ValueError(f"{owner}: expected trailing vector dim length 3.")
    return values.reshape((-1, _VEC3_SIZE)), values.shape[:-1]


def _reshape_quat(values: np.ndarray, *, owner: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.shape[-1] != _QUAT_SIZE:
        raise ValueError(f"{owner}: expected trailing quaternion dim length 4.")
    return values.reshape((-1, _QUAT_SIZE)), values.shape[:-1]


def rotate_vec3_kernel(values: np.ndarray, quat: np.ndarray) -> np.ndarray:
    owner = "spatial.rotation.kernel.apply_vec3"
    vec_flat, leading = _reshape_vec3(values, owner=owner)
    quat_flat, _ = _reshape_quat(quat, owner=owner)
    try:
        rotated = SciRotation.from_quat(quat_flat).apply(vec_flat)
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid quaternion input: {exc}") from exc
    return rotated.reshape(leading + (_VEC3_SIZE,))


__all__ = ["rotate_vec3_kernel"]
