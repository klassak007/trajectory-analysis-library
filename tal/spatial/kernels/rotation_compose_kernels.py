from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

_QUAT_SIZE = 4


def _reshape_quat(values: np.ndarray, *, owner: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if values.shape[-1] != _QUAT_SIZE:
        raise ValueError(f"{owner}: expected quaternion trailing dim length 4.")
    return values.reshape((-1, _QUAT_SIZE)), values.shape[:-1]


def compose_quat_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_flat, leading = _reshape_quat(left, owner="spatial.rotation.kernel.compose")
    right_flat, _ = _reshape_quat(right, owner="spatial.rotation.kernel.compose")
    try:
        left_rot = SciRotation.from_quat(left_flat)
        right_rot = SciRotation.from_quat(right_flat)
        out = (right_rot * left_rot).as_quat()
    except ValueError as exc:
        raise ValueError(f"spatial.rotation.kernel.compose: invalid quaternion input: {exc}") from exc
    return out.reshape(leading + (_QUAT_SIZE,))


def inverse_quat_kernel(values: np.ndarray) -> np.ndarray:
    flat, leading = _reshape_quat(values, owner="spatial.rotation.kernel.inverse")
    try:
        out = SciRotation.from_quat(flat).inv().as_quat()
    except ValueError as exc:
        raise ValueError(f"spatial.rotation.kernel.inverse: invalid quaternion input: {exc}") from exc
    return out.reshape(leading + (_QUAT_SIZE,))


__all__ = ["compose_quat_kernel", "inverse_quat_kernel"]
