from __future__ import annotations

import numpy as np

from .block_prep import prepare_bounds_block_rows, prepare_map_block_rows


def map_block_numpy(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .map_build import _map_row

    param_rows, valid_rows, query_rows, output_shape = prepare_map_block_rows(param_block, valid_block, query_block)
    rows = int(param_rows.shape[0])
    query_size = int(query_rows.shape[1])
    i0 = np.zeros((rows, query_size), dtype=np.int64)
    i1 = np.zeros((rows, query_size), dtype=np.int64)
    alpha = np.zeros((rows, query_size), dtype=np.float64)
    valid = np.zeros((rows, query_size), dtype=bool)
    for row in range(rows):
        row_i0, row_i1, row_alpha, row_valid = _map_row(
            param_rows[row],
            valid_rows[row],
            query_rows[row],
            method=method,
            dup_code=dup_code,
        )
        i0[row] = row_i0
        i1[row] = row_i1
        alpha[row] = row_alpha
        valid[row] = row_valid
    return i0.reshape(output_shape), i1.reshape(output_shape), alpha.reshape(output_shape), valid.reshape(output_shape)


def bounds_block_numpy(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    start_block: np.ndarray,
    stop_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from .map_build import _bounds_row

    param_rows, valid_rows, start_rows, stop_rows, output_shape = prepare_bounds_block_rows(
        param_block,
        valid_block,
        start_block,
        stop_block,
    )
    rows = int(param_rows.shape[0])
    i0 = np.zeros(rows, dtype=np.int64)
    i1 = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        row_i0, row_i1 = _bounds_row(param_rows[row], valid_rows[row], start_rows[row], stop_rows[row])
        i0[row] = row_i0
        i1[row] = row_i1
    return i0.reshape(output_shape), i1.reshape(output_shape)


__all__ = ["bounds_block_numpy", "map_block_numpy"]
