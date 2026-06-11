from __future__ import annotations

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows


def prepare_lstsq_vector_block_rows(
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


def prepare_lstsq_matrix_block_rows(
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


__all__ = ["prepare_lstsq_matrix_block_rows", "prepare_lstsq_vector_block_rows"]
