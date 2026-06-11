from __future__ import annotations

from functools import lru_cache

import numpy as np

from .block_prep import prepare_lstsq_matrix_block_rows, prepare_lstsq_vector_block_rows
from tal.utils.numba_support import njit_kernel, require_numba


@lru_cache(maxsize=1)
def _compiled_lstsq_vector_block():
    numba = require_numba("linalg.solve")
    return njit_kernel(numba, _lstsq_vector_block_impl)


@lru_cache(maxsize=1)
def _compiled_lstsq_matrix_block():
    numba = require_numba("linalg.solve")
    return njit_kernel(numba, _lstsq_matrix_block_impl)


def lstsq_block_numba(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    rcond: float | None,
    rhs_is_vector: bool,
    owner: str,
) -> np.ndarray:
    require_numba(owner)
    if rhs_is_vector:
        a_rows, b_rows, output_shape = prepare_lstsq_vector_block_rows(a_block, b_block, owner=owner)
        out = _compiled_lstsq_vector_block()(a_rows, b_rows, _effective_rcond_for_rows(a_rows, rcond))
        return out.reshape(output_shape)
    a_rows, b_rows, output_shape = prepare_lstsq_matrix_block_rows(a_block, b_block, owner=owner)
    out = _compiled_lstsq_matrix_block()(a_rows, b_rows, _effective_rcond_for_rows(a_rows, rcond))
    return out.reshape(output_shape)


def _effective_rcond_for_rows(a_rows: np.ndarray, rcond: float | None) -> float:
    from .solve_backends import _effective_lstsq_rcond

    return _effective_lstsq_rcond(
        rcond,
        rows=int(a_rows.shape[-2]),
        cols=int(a_rows.shape[-1]),
        dtype=a_rows.dtype,
    )


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
