from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_support import require_numba


def _row_count(shape: tuple[int, ...]) -> int:
    rows = 1
    for size in shape:
        rows *= int(size)
    return rows


@lru_cache(maxsize=1)
def _compiled_lstsq_vector_block():
    numba = require_numba("linalg.solve")
    return numba.njit(cache=True, fastmath=False)(_lstsq_vector_block_impl)


@lru_cache(maxsize=1)
def _compiled_lstsq_matrix_block():
    numba = require_numba("linalg.solve")
    return numba.njit(cache=True, fastmath=False)(_lstsq_matrix_block_impl)


def _broadcast_vector_blocks(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if a_block.ndim < 2 or b_block.ndim < 1:
        raise ValueError(f"{owner}: lstsq blocks must include trailing core dimensions.")
    equations = int(a_block.shape[-2])
    solutions = int(a_block.shape[-1])
    if int(b_block.shape[-1]) != equations:
        raise ValueError(f"{owner}: vector rhs trailing dimension must match equation dimension.")
    try:
        outer = np.broadcast_shapes(a_block.shape[:-2], b_block.shape[:-1])
    except ValueError as exc:
        raise ValueError(f"{owner}: lstsq vector rhs blocks are not broadcast-compatible.") from exc
    rows = _row_count(outer)
    a_rows = np.broadcast_to(a_block, outer + (equations, solutions)).reshape(rows, equations, solutions)
    b_rows = np.broadcast_to(b_block, outer + (equations,)).reshape(rows, equations)
    return np.ascontiguousarray(a_rows), np.ascontiguousarray(b_rows), outer + (solutions,)


def _broadcast_matrix_blocks(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if a_block.ndim < 2 or b_block.ndim < 2:
        raise ValueError(f"{owner}: matrix rhs lstsq blocks must include trailing core dimensions.")
    equations = int(a_block.shape[-2])
    solutions = int(a_block.shape[-1])
    rhs_cols = int(b_block.shape[-1])
    if int(b_block.shape[-2]) != equations:
        raise ValueError(f"{owner}: matrix rhs leading core dimension must match equation dimension.")
    try:
        outer = np.broadcast_shapes(a_block.shape[:-2], b_block.shape[:-2])
    except ValueError as exc:
        raise ValueError(f"{owner}: lstsq matrix rhs blocks are not broadcast-compatible.") from exc
    rows = _row_count(outer)
    a_rows = np.broadcast_to(a_block, outer + (equations, solutions)).reshape(rows, equations, solutions)
    b_rows = np.broadcast_to(b_block, outer + (equations, rhs_cols)).reshape(rows, equations, rhs_cols)
    return np.ascontiguousarray(a_rows), np.ascontiguousarray(b_rows), outer + (solutions, rhs_cols)


def lstsq_block_numba(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    rcond: float,
    rhs_is_vector: bool,
    owner: str,
) -> np.ndarray:
    require_numba(owner)
    if rhs_is_vector:
        a_rows, b_rows, output_shape = _broadcast_vector_blocks(a_block, b_block, owner=owner)
        out = _compiled_lstsq_vector_block()(a_rows, b_rows, float(rcond))
        return out.reshape(output_shape)
    a_rows, b_rows, output_shape = _broadcast_matrix_blocks(a_block, b_block, owner=owner)
    out = _compiled_lstsq_matrix_block()(a_rows, b_rows, float(rcond))
    return out.reshape(output_shape)


def _lstsq_vector_block_impl(a_rows, b_rows, rcond):
    out = np.empty((a_rows.shape[0], a_rows.shape[2]), dtype=a_rows.dtype)
    for row in range(a_rows.shape[0]):
        out[row] = np.linalg.lstsq(a_rows[row], b_rows[row], rcond=rcond)[0]
    return out


def _lstsq_matrix_block_impl(a_rows, b_rows, rcond):
    out = np.empty((a_rows.shape[0], a_rows.shape[2], b_rows.shape[2]), dtype=a_rows.dtype)
    for row in range(a_rows.shape[0]):
        out[row] = np.linalg.lstsq(a_rows[row], b_rows[row], rcond=rcond)[0]
    return out


__all__ = ["lstsq_block_numba"]
