from __future__ import annotations

import numpy as np

from .kinematics_temporal_kernels import cumulative_trapezoid_kernel

KINEMATICS_TEMPORAL_BACKEND_NUMPY = "numpy"
KINEMATICS_TEMPORAL_BACKEND_NUMBA = "numba"


def cumulative_trapezoid_block_backend(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    initial_value: float,
    backend: str = KINEMATICS_TEMPORAL_BACKEND_NUMPY,
) -> np.ndarray:
    owner = "spatial.kinematics.temporal.integral_backend"
    if backend == KINEMATICS_TEMPORAL_BACKEND_NUMPY:
        return cumulative_trapezoid_kernel(values, param, valid, initial_value=initial_value)
    if backend == KINEMATICS_TEMPORAL_BACKEND_NUMBA:
        from .kinematics_temporal_numba_backends import cumulative_trapezoid_block_numba

        return cumulative_trapezoid_block_numba(values, param, valid, initial_value=initial_value, owner=owner)
    raise ValueError(f"{owner}: unsupported temporal backend {backend!r}.")


__all__ = [
    "KINEMATICS_TEMPORAL_BACKEND_NUMBA",
    "KINEMATICS_TEMPORAL_BACKEND_NUMPY",
    "cumulative_trapezoid_block_backend",
]
