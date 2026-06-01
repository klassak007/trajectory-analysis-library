from __future__ import annotations

import numpy as np

LSTSQ_BACKEND_NUMPY_ROW = "numpy_row"
LSTSQ_BACKEND_NUMBA = "numba"
_NUMBA_MIN_ROWS = 512
_NUMBA_MAX_EQUATIONS = 16
_NUMBA_MAX_SOLUTIONS = 8
_NUMBA_MAX_RHS_COLS = 8


def _lstsq_numpy_row(a: np.ndarray, b: np.ndarray, *, rcond: float | None) -> np.ndarray:
    return np.linalg.lstsq(a, b, rcond=rcond)[0]


def lstsq_solution_backend(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rcond: float | None,
    backend: str = LSTSQ_BACKEND_NUMPY_ROW,
) -> np.ndarray:
    if backend != LSTSQ_BACKEND_NUMPY_ROW:
        raise ValueError(f"linalg.solve: unsupported lstsq backend {backend!r}.")
    return _lstsq_numpy_row(a, b, rcond=rcond)


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


def _lstsq_outer_rows(a: np.ndarray, b: np.ndarray, *, rhs_is_vector: bool) -> int:
    if rhs_is_vector:
        outer = np.broadcast_shapes(a.shape[:-2], b.shape[:-1])
    else:
        outer = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return int(np.prod(outer, dtype=np.int64)) if outer else 1


def _lstsq_rhs_cols(b: np.ndarray, *, rhs_is_vector: bool) -> int:
    return 1 if rhs_is_vector else int(b.shape[-1])


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
) -> str:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if not _is_lstsq_supported_dtype(a_arr.dtype) or not _is_lstsq_supported_dtype(b_arr.dtype):
        return LSTSQ_BACKEND_NUMPY_ROW
    rows = _lstsq_outer_rows(a_arr, b_arr, rhs_is_vector=rhs_is_vector)
    equations = int(a_arr.shape[-2])
    solutions = int(a_arr.shape[-1])
    rhs_cols = _lstsq_rhs_cols(b_arr, rhs_is_vector=rhs_is_vector)
    if rows < _NUMBA_MIN_ROWS:
        return LSTSQ_BACKEND_NUMPY_ROW
    if equations > _NUMBA_MAX_EQUATIONS or solutions > _NUMBA_MAX_SOLUTIONS:
        return LSTSQ_BACKEND_NUMPY_ROW
    if rhs_cols > _NUMBA_MAX_RHS_COLS:
        return LSTSQ_BACKEND_NUMPY_ROW
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
    if backend != LSTSQ_BACKEND_NUMBA:
        raise ValueError(f"{owner}: unsupported lstsq backend {backend!r}.")
    from tal.utils.numba_support import require_numba

    require_numba(owner)
    work_dtype = _lstsq_work_dtype(a, b)
    a_arr = np.asarray(a, dtype=work_dtype)
    b_arr = np.asarray(b, dtype=work_dtype)
    effective_rcond = _effective_lstsq_rcond(
        rcond,
        rows=int(a_arr.shape[-2]),
        cols=int(a_arr.shape[-1]),
        dtype=work_dtype,
    )
    from .numba_backends import lstsq_block_numba

    return lstsq_block_numba(
        a_arr,
        b_arr,
        rcond=effective_rcond,
        rhs_is_vector=rhs_is_vector,
        owner=owner,
    )


__all__ = [
    "LSTSQ_BACKEND_NUMBA",
    "LSTSQ_BACKEND_NUMPY_ROW",
    "lstsq_block_backend",
    "lstsq_solution_backend",
]
