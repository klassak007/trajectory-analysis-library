from __future__ import annotations

import numpy as np

LSTSQ_BACKEND_NUMPY_BLOCK = "numpy_block"
LSTSQ_BACKEND_NUMBA = "numba"
_NUMBA_MIN_ROWS = 512
_NUMBA_MAX_EQUATIONS = 16
_NUMBA_MAX_SOLUTIONS = 8
_NUMBA_MAX_RHS_COLS = 8


def _lstsq_numpy_row(a: np.ndarray, b: np.ndarray, *, rcond: float | None) -> np.ndarray:
    return np.linalg.lstsq(a, b, rcond=rcond)[0]


def _lstsq_work_dtype(a: np.ndarray, b: np.ndarray) -> np.dtype:
    a_dtype = np.asarray(a).dtype
    b_dtype = np.asarray(b).dtype
    if np.issubdtype(a_dtype, np.complexfloating) or np.issubdtype(b_dtype, np.complexfloating):
        return np.result_type(a_dtype, b_dtype, np.complex64)
    if a_dtype == np.dtype("float32") and b_dtype == np.dtype("float32"):
        return np.dtype("float32")
    return np.dtype("float64")


def _rcond_eps_dtype(dtype: np.dtype) -> np.dtype:
    if dtype == np.dtype("float32") or dtype == np.dtype("complex64"):
        return np.dtype("float32")
    return np.dtype("float64")


def _effective_lstsq_rcond(
    rcond: float | None,
    *,
    rows: int,
    cols: int,
    dtype: np.dtype,
) -> float:
    if rcond is not None:
        return float(rcond)
    return float(max(rows, cols) * np.finfo(_rcond_eps_dtype(dtype)).eps)


def _lstsq_outer_rows_from_shapes(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    *,
    rhs_is_vector: bool,
) -> int:
    if rhs_is_vector:
        outer = np.broadcast_shapes(a_shape[:-2], b_shape[:-1])
    else:
        outer = np.broadcast_shapes(a_shape[:-2], b_shape[:-2])
    return int(np.prod(outer, dtype=np.int64)) if outer else 1


def _lstsq_rhs_cols_from_shape(b_shape: tuple[int, ...], *, rhs_is_vector: bool) -> int:
    return 1 if rhs_is_vector else int(b_shape[-1])


def _is_lstsq_supported_dtype(dtype: np.dtype) -> bool:
    return (
        np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
        or np.issubdtype(dtype, np.complexfloating)
    )


def _select_lstsq_backend(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rhs_is_vector: bool,
    numba_available: bool = True,
) -> str:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    return _select_lstsq_backend_from_metadata(
        tuple(int(size) for size in a_arr.shape),
        tuple(int(size) for size in b_arr.shape),
        a_arr.dtype,
        b_arr.dtype,
        rhs_is_vector=rhs_is_vector,
        numba_available=numba_available,
    )


def _select_lstsq_backend_from_metadata(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    a_dtype: object,
    b_dtype: object,
    *,
    rhs_is_vector: bool,
    numba_available: bool,
) -> str:
    if not numba_available:
        return LSTSQ_BACKEND_NUMPY_BLOCK
    if not _is_lstsq_supported_dtype(np.dtype(a_dtype)) or not _is_lstsq_supported_dtype(np.dtype(b_dtype)):
        return LSTSQ_BACKEND_NUMPY_BLOCK
    rows = _lstsq_outer_rows_from_shapes(a_shape, b_shape, rhs_is_vector=rhs_is_vector)
    equations = int(a_shape[-2])
    solutions = int(a_shape[-1])
    rhs_cols = _lstsq_rhs_cols_from_shape(b_shape, rhs_is_vector=rhs_is_vector)
    if rows < _NUMBA_MIN_ROWS:
        return LSTSQ_BACKEND_NUMPY_BLOCK
    if equations > _NUMBA_MAX_EQUATIONS or solutions > _NUMBA_MAX_SOLUTIONS:
        return LSTSQ_BACKEND_NUMPY_BLOCK
    if rhs_cols > _NUMBA_MAX_RHS_COLS:
        return LSTSQ_BACKEND_NUMPY_BLOCK
    return LSTSQ_BACKEND_NUMBA


def lstsq_block_backend(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rcond: float | None,
    rhs_is_vector: bool,
    backend: str,
    owner: str = "linalg.solve",
) -> np.ndarray:
    if backend not in (LSTSQ_BACKEND_NUMPY_BLOCK, LSTSQ_BACKEND_NUMBA):
        raise ValueError(f"{owner}: unsupported lstsq backend {backend!r}.")
    work_dtype = _lstsq_work_dtype(a, b)
    a_arr = np.asarray(a, dtype=work_dtype)
    b_arr = np.asarray(b, dtype=work_dtype)
    if backend == LSTSQ_BACKEND_NUMPY_BLOCK:
        from .numpy_backends import lstsq_block_numpy

        return lstsq_block_numpy(
            a_arr,
            b_arr,
            rcond=rcond,
            rhs_is_vector=rhs_is_vector,
            owner=owner,
        )
    if backend == LSTSQ_BACKEND_NUMBA:
        from tal.utils.numba_support import require_numba

        require_numba(owner)
        from .numba_backends import lstsq_block_numba

        return lstsq_block_numba(
            a_arr,
            b_arr,
            rcond=rcond,
            rhs_is_vector=rhs_is_vector,
            owner=owner,
        )
    raise AssertionError("unreachable lstsq backend dispatch")


__all__ = [
    "LSTSQ_BACKEND_NUMBA",
    "LSTSQ_BACKEND_NUMPY_BLOCK",
    "_select_lstsq_backend",
    "_select_lstsq_backend_from_metadata",
    "lstsq_block_backend",
]
