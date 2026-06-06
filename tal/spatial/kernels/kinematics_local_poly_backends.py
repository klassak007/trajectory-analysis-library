from __future__ import annotations

import numpy as np

from .kinematics_local_poly_common import (
    LOCAL_POLY_BACKEND_OWNER,
    validate_local_poly_options,
    validate_local_poly_shapes,
)
from .kinematics_temporal_kernels import local_poly_first_derivative_kernel, local_poly_smooth_kernel

KINEMATICS_LOCAL_POLY_BACKEND_NUMPY = "numpy"
KINEMATICS_LOCAL_POLY_BACKEND_NUMBA = "numba"


def local_poly_smoothing_block_backend(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    poly_order: int,
    backend: str = KINEMATICS_LOCAL_POLY_BACKEND_NUMPY,
) -> np.ndarray:
    owner = LOCAL_POLY_BACKEND_OWNER
    validate_local_poly_shapes(values, param, valid, owner=owner)
    window_value, poly_value = validate_local_poly_options(window, poly_order, owner=owner)
    if backend == KINEMATICS_LOCAL_POLY_BACKEND_NUMPY:
        return local_poly_smooth_kernel(values, param, valid, window=window_value, poly_order=poly_value)
    if backend == KINEMATICS_LOCAL_POLY_BACKEND_NUMBA:
        raise ValueError(f"{owner}: numba local-poly backend is not retained; use backend='numpy'.")
    raise ValueError(f"{owner}: unsupported local-poly backend {backend!r}.")


def local_poly_derivative_block_backend(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    poly_order: int,
    backend: str = KINEMATICS_LOCAL_POLY_BACKEND_NUMPY,
) -> np.ndarray:
    owner = LOCAL_POLY_BACKEND_OWNER
    validate_local_poly_shapes(values, param, valid, owner=owner)
    window_value, poly_value = validate_local_poly_options(window, poly_order, owner=owner)
    if backend == KINEMATICS_LOCAL_POLY_BACKEND_NUMPY:
        return local_poly_first_derivative_kernel(values, param, valid, window=window_value, poly_order=poly_value)
    if backend == KINEMATICS_LOCAL_POLY_BACKEND_NUMBA:
        raise ValueError(f"{owner}: numba local-poly backend is not retained; use backend='numpy'.")
    raise ValueError(f"{owner}: unsupported local-poly backend {backend!r}.")


__all__ = [
    "KINEMATICS_LOCAL_POLY_BACKEND_NUMBA",
    "KINEMATICS_LOCAL_POLY_BACKEND_NUMPY",
    "local_poly_derivative_block_backend",
    "local_poly_smoothing_block_backend",
]
