from __future__ import annotations

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows


def prepare_boundary_block_rows(
    mask_block: np.ndarray,
    valid_block: np.ndarray,
    clock_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (mask_block, valid_block, clock_block),
        (
            BlockInputSpec("mask", 1, bool),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("clock", 1, np.float64),
        ),
        output_core_shape=(),
        owner=owner,
    )
    mask_rows, valid_rows, clock_rows = prepared.row_arrays
    seq_size = int(mask_rows.shape[-1])
    if int(valid_rows.shape[-1]) != seq_size or int(clock_rows.shape[-1]) != seq_size:
        raise ValueError(f"{owner}: event boundary block trailing dimensions must match.")
    return mask_rows, valid_rows, clock_rows, prepared.outer_shape


def prepare_intervals_block_rows(
    time_block: np.ndarray,
    edge_block: np.ndarray,
    before_block: np.ndarray,
    after_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (time_block, edge_block, before_block, after_block),
        (
            BlockInputSpec("time", 1, np.float64),
            BlockInputSpec("edge", 1, np.int8),
            BlockInputSpec("before", 1, np.int64),
            BlockInputSpec("after", 1, np.int64),
        ),
        output_core_shape=(),
        owner=owner,
    )
    time_rows, edge_rows, before_rows, after_rows = prepared.row_arrays
    event_size = int(time_rows.shape[-1])
    mismatched = (
        int(edge_rows.shape[-1]) != event_size
        or int(before_rows.shape[-1]) != event_size
        or int(after_rows.shape[-1]) != event_size
    )
    if mismatched:
        raise ValueError(f"{owner}: interval block trailing dimensions must match.")
    return time_rows, edge_rows, before_rows, after_rows, prepared.outer_shape


__all__ = ["prepare_boundary_block_rows", "prepare_intervals_block_rows"]
