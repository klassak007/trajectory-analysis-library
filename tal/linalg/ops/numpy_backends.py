from __future__ import annotations

import numpy as np

from .block_prep import prepare_lstsq_matrix_block_rows, prepare_lstsq_vector_block_rows


def _effective_rcond_for_rows(a_rows: np.ndarray, rcond: float | None) -> float:
    from .solve_backends import _effective_lstsq_rcond

    return _effective_lstsq_rcond(
        rcond,
        rows=int(a_rows.shape[-2]),
        cols=int(a_rows.shape[-1]),
        dtype=a_rows.dtype,
    )


def lstsq_block_numpy(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    rcond: float | None,
    rhs_is_vector: bool,
    owner: str,
) -> np.ndarray:
    if rhs_is_vector:
        return _lstsq_vector_block_numpy(a_block, b_block, rcond=rcond, owner=owner)
    return _lstsq_matrix_block_numpy(a_block, b_block, rcond=rcond, owner=owner)


def _lstsq_vector_block_numpy(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    rcond: float | None,
    owner: str,
) -> np.ndarray:
    from .solve_backends import _lstsq_numpy_row

    a_rows, b_rows, output_shape = prepare_lstsq_vector_block_rows(a_block, b_block, owner=owner)
    effective_rcond = _effective_rcond_for_rows(a_rows, rcond)
    out = np.empty((int(a_rows.shape[0]), int(a_rows.shape[2])), dtype=a_rows.dtype)
    for row in range(int(a_rows.shape[0])):
        out[row] = _lstsq_numpy_row(a_rows[row], b_rows[row], rcond=effective_rcond)
    return out.reshape(output_shape)


def _lstsq_matrix_block_numpy(
    a_block: np.ndarray,
    b_block: np.ndarray,
    *,
    rcond: float | None,
    owner: str,
) -> np.ndarray:
    from .solve_backends import _lstsq_numpy_row

    a_rows, b_rows, output_shape = prepare_lstsq_matrix_block_rows(a_block, b_block, owner=owner)
    effective_rcond = _effective_rcond_for_rows(a_rows, rcond)
    out = np.empty(
        (int(a_rows.shape[0]), int(a_rows.shape[2]), int(b_rows.shape[2])),
        dtype=a_rows.dtype,
    )
    for row in range(int(a_rows.shape[0])):
        out[row] = _lstsq_numpy_row(a_rows[row], b_rows[row], rcond=effective_rcond)
    return out.reshape(output_shape)


__all__ = ["lstsq_block_numpy"]
