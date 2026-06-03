from __future__ import annotations

import numpy as np

from .kinematics_temporal_kernels import gaussian_partial_renorm_kernel, moving_average_partial_renorm_kernel

KINEMATICS_SMOOTHING_BACKEND_NUMPY = "numpy"
KINEMATICS_SMOOTHING_BACKEND_NUMBA = "numba"

_OWNER = "spatial.kinematics.temporal.smoothing_backend"


def moving_average_smoothing_block_backend(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    backend: str = KINEMATICS_SMOOTHING_BACKEND_NUMPY,
) -> np.ndarray:
    if backend == KINEMATICS_SMOOTHING_BACKEND_NUMPY:
        return moving_average_partial_renorm_kernel(values, param, valid, window=window)
    if backend == KINEMATICS_SMOOTHING_BACKEND_NUMBA:
        from .kinematics_smoothing_numba_backends import moving_average_smoothing_block_numba

        return moving_average_smoothing_block_numba(values, param, valid, window=window, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported smoothing backend {backend!r}.")


def gaussian_smoothing_block_backend(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    sigma: float,
    backend: str = KINEMATICS_SMOOTHING_BACKEND_NUMPY,
) -> np.ndarray:
    if backend == KINEMATICS_SMOOTHING_BACKEND_NUMPY:
        return gaussian_partial_renorm_kernel(values, param, valid, window=window, sigma=sigma)
    if backend == KINEMATICS_SMOOTHING_BACKEND_NUMBA:
        from .kinematics_smoothing_numba_backends import gaussian_smoothing_block_numba

        return gaussian_smoothing_block_numba(values, param, valid, window=window, sigma=sigma, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported smoothing backend {backend!r}.")


__all__ = [
    "KINEMATICS_SMOOTHING_BACKEND_NUMBA",
    "KINEMATICS_SMOOTHING_BACKEND_NUMPY",
    "gaussian_smoothing_block_backend",
    "moving_average_smoothing_block_backend",
]
