from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows
from tal.utils.numba_support import njit_kernel, require_numba


@lru_cache(maxsize=1)
def _compiled_lstsq_vector_block():
    numba = require_numba("linalg.solve")
    return njit_kernel(numba, _lstsq_vector_block_impl)


@lru_cache(maxsize=1)
def _compiled_lstsq_matrix_block():
    numba = require_numba("linalg.solve")
    return njit_kernel(numba, _lstsq_matrix_block_impl)


def _broadcast_vector_blocks(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (a_block, b_block),
        (BlockInputSpec("a", 2, None), BlockInputSpec("b", 1, None)),
        output_core_shape=(),
        owner=owner,
    )
    a_rows, b_rows = prepared.row_arrays
    equations = int(a_rows.shape[-2])
    solutions = int(a_rows.shape[-1])
    if int(b_rows.shape[-1]) != equations:
        raise ValueError(f"{owner}: vector rhs trailing dimension must match equation dimension.")
    return a_rows, b_rows, prepared.outer_shape + (solutions,)


def _broadcast_matrix_blocks(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (a_block, b_block),
        (BlockInputSpec("a", 2, None), BlockInputSpec("b", 2, None)),
        output_core_shape=(),
        owner=owner,
    )
    a_rows, b_rows = prepared.row_arrays
    equations = int(a_rows.shape[-2])
    solutions = int(a_rows.shape[-1])
    rhs_cols = int(b_rows.shape[-1])
    if int(b_rows.shape[-2]) != equations:
        raise ValueError(f"{owner}: matrix rhs leading core dimension must match equation dimension.")
    return a_rows, b_rows, prepared.outer_shape + (solutions, rhs_cols)


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
