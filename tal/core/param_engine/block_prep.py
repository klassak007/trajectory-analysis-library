from __future__ import annotations

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows


def prepare_map_block_rows(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (param_block, valid_block, query_block),
        (
            BlockInputSpec("param", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("query", 1, np.float64),
        ),
        output_core_shape=(),
        owner="build_param_map",
    )
    param_rows, valid_rows, query_rows = prepared.row_arrays
    seq_size = int(param_rows.shape[-1])
    query_size = int(query_rows.shape[-1])
    if int(valid_rows.shape[-1]) != seq_size:
        raise ValueError("build_param_map: valid block trailing dimension must match param block.")
    return param_rows, valid_rows, query_rows, prepared.outer_shape + (query_size,)


def prepare_datetime_map_block_rows(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (param_block, valid_block, query_block),
        (
            BlockInputSpec("param", 1, np.dtype("datetime64[ns]")),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("query", 1, np.dtype("datetime64[ns]")),
        ),
        output_core_shape=(),
        owner="build_param_map",
    )
    param_rows, valid_rows, query_rows = prepared.row_arrays
    seq_size = int(param_rows.shape[-1])
    query_size = int(query_rows.shape[-1])
    if int(valid_rows.shape[-1]) != seq_size:
        raise ValueError("build_param_map: valid block trailing dimension must match param block.")
    return param_rows, valid_rows, query_rows, prepared.outer_shape + (query_size,)


def prepare_bounds_block_rows(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    start_block: np.ndarray,
    stop_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (param_block, valid_block, start_block, stop_block),
        (
            BlockInputSpec("param", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("start", 0, np.float64),
            BlockInputSpec("stop", 0, np.float64),
        ),
        output_core_shape=(),
        owner="build_param_bounds_map",
    )
    param_rows, valid_rows, start_rows, stop_rows = prepared.row_arrays
    seq_size = int(param_rows.shape[-1])
    if int(valid_rows.shape[-1]) != seq_size:
        raise ValueError("build_param_bounds_map: valid block trailing dimension must match param block.")
    return param_rows, valid_rows, start_rows, stop_rows, prepared.outer_shape


def prepare_datetime_bounds_block_rows(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    start_block: np.ndarray,
    stop_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    prepared = prepare_block_rows(
        (param_block, valid_block, start_block, stop_block),
        (
            BlockInputSpec("param", 1, np.dtype("datetime64[ns]")),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("start", 0, np.dtype("datetime64[ns]")),
            BlockInputSpec("stop", 0, np.dtype("datetime64[ns]")),
        ),
        output_core_shape=(),
        owner="build_param_bounds_map",
    )
    param_rows, valid_rows, start_rows, stop_rows = prepared.row_arrays
    seq_size = int(param_rows.shape[-1])
    if int(valid_rows.shape[-1]) != seq_size:
        raise ValueError("build_param_bounds_map: valid block trailing dimension must match param block.")
    return param_rows, valid_rows, start_rows, stop_rows, prepared.outer_shape


__all__ = [
    "prepare_bounds_block_rows",
    "prepare_datetime_bounds_block_rows",
    "prepare_datetime_map_block_rows",
    "prepare_map_block_rows",
]
