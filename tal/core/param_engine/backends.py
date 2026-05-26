from __future__ import annotations

import numpy as np

PARAM_MAP_BACKEND_NUMPY_ROW = "numpy_row"
PARAM_BOUNDS_BACKEND_NUMPY_ROW = "numpy_row"
PARAM_MAP_BACKEND_NUMBA = "numba"
PARAM_BOUNDS_BACKEND_NUMBA = "numba"


def map_row_backend(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    query_row: np.ndarray,
    *,
    method: str,
    dup_code: int,
    backend: str = PARAM_MAP_BACKEND_NUMPY_ROW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if backend != PARAM_MAP_BACKEND_NUMPY_ROW:
        raise ValueError(f"build_param_map: unsupported backend {backend!r}.")
    from .map_build import _map_row

    return _map_row(param_row, valid_row, query_row, method=method, dup_code=dup_code)


def bounds_row_backend(
    param_row: np.ndarray,
    valid_row: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    *,
    backend: str = PARAM_BOUNDS_BACKEND_NUMPY_ROW,
) -> tuple[np.ndarray, np.ndarray]:
    if backend != PARAM_BOUNDS_BACKEND_NUMPY_ROW:
        raise ValueError(f"build_param_bounds_map: unsupported backend {backend!r}.")
    from .map_build import _bounds_row

    return _bounds_row(param_row, valid_row, start, stop)


def map_block_backend(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
    *,
    method: str,
    dup_code: int,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if backend != PARAM_MAP_BACKEND_NUMBA:
        raise ValueError(f"build_param_map: unsupported backend {backend!r}.")
    from .numba_backends import map_block_numba

    return map_block_numba(param_block, valid_block, query_block, method=method, dup_code=dup_code)


def bounds_block_backend(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    start_block: np.ndarray,
    stop_block: np.ndarray,
    *,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    if backend != PARAM_BOUNDS_BACKEND_NUMBA:
        raise ValueError(f"build_param_bounds_map: unsupported backend {backend!r}.")
    from .numba_backends import bounds_block_numba

    return bounds_block_numba(param_block, valid_block, start_block, stop_block)


__all__ = [
    "PARAM_BOUNDS_BACKEND_NUMBA",
    "PARAM_BOUNDS_BACKEND_NUMPY_ROW",
    "PARAM_MAP_BACKEND_NUMBA",
    "PARAM_MAP_BACKEND_NUMPY_ROW",
    "bounds_block_backend",
    "bounds_row_backend",
    "map_block_backend",
    "map_row_backend",
]
