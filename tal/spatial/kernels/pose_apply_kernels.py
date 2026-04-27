from __future__ import annotations

import numpy as np

from .pose_kernels import compose_translation_kernel


def pose_apply_position_kernel(position: np.ndarray, translation: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return compose_translation_kernel(position, translation, quat)


__all__ = ["pose_apply_position_kernel"]
