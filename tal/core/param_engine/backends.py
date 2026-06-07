from __future__ import annotations

import numpy as np

PARAM_MAP_BACKEND_NUMPY_BLOCK = "numpy_block"
PARAM_BOUNDS_BACKEND_NUMPY_BLOCK = "numpy_block"
PARAM_MAP_BACKEND_NUMBA = "numba"
PARAM_BOUNDS_BACKEND_NUMBA = "numba"


def map_block_backend(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
    *,
    method: str,
    dup_code: int,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if backend == PARAM_MAP_BACKEND_NUMPY_BLOCK:
        from .numpy_backends import map_block_numpy

        return map_block_numpy(param_block, valid_block, query_block, method=method, dup_code=dup_code)
    if backend == PARAM_MAP_BACKEND_NUMBA:
        from .numba_backends import map_block_numba

        return map_block_numba(param_block, valid_block, query_block, method=method, dup_code=dup_code)
    raise ValueError(f"build_param_map: unsupported backend {backend!r}.")


def bounds_block_backend(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    start_block: np.ndarray,
    stop_block: np.ndarray,
    *,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    if backend == PARAM_BOUNDS_BACKEND_NUMPY_BLOCK:
        from .numpy_backends import bounds_block_numpy

        return bounds_block_numpy(param_block, valid_block, start_block, stop_block)
    if backend == PARAM_BOUNDS_BACKEND_NUMBA:
        from .numba_backends import bounds_block_numba

        return bounds_block_numba(param_block, valid_block, start_block, stop_block)
    raise ValueError(f"build_param_bounds_map: unsupported backend {backend!r}.")


__all__ = [
    "PARAM_BOUNDS_BACKEND_NUMBA",
    "PARAM_BOUNDS_BACKEND_NUMPY_BLOCK",
    "PARAM_MAP_BACKEND_NUMBA",
    "PARAM_MAP_BACKEND_NUMPY_BLOCK",
    "bounds_block_backend",
    "map_block_backend",
]
