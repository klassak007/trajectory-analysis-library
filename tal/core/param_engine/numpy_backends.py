from __future__ import annotations

import numpy as np

from .block_prep import (
    prepare_bounds_block_rows,
    prepare_datetime_bounds_block_rows,
    prepare_datetime_map_block_rows,
    prepare_map_block_rows,
)
from .map_failures import ParamMapFailure


def _map_rows_with_status(
    mapper,
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]],
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, ...]:
    param_rows, valid_rows, query_rows, output_shape = prepared
    buffers = (
        np.zeros(output_shape, dtype=np.int64),
        np.zeros(output_shape, dtype=np.int64),
        np.zeros(output_shape, dtype=np.float64),
        np.zeros(output_shape, dtype=bool),
    )
    outer_shape = output_shape[:-1]
    status = np.zeros(outer_shape, dtype=np.int8).reshape(-1)
    position = np.full(outer_shape, -1, dtype=np.int64).reshape(-1)
    detail = np.zeros(outer_shape, dtype=object).reshape(-1)
    outputs = tuple(
        value.reshape(param_rows.shape[0], output_shape[-1]) for value in buffers
    )
    kwargs = {"method": method, "dup_code": dup_code}
    for row in range(param_rows.shape[0]):
        args = (param_rows[row], valid_rows[row])
        try:
            result = mapper(*args, query_rows[row], **kwargs)
        except ParamMapFailure as exc:
            status[row] = exc.status
            position[row] = exc.position
            detail[row] = exc.detail
            continue
        for target, values in zip(outputs, result, strict=True):
            target[row] = values
    return (
        *buffers,
        status.reshape(outer_shape),
        position.reshape(outer_shape),
        detail.reshape(outer_shape),
    )


def map_block_numpy_status(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, ...]:
    from .numeric_rows import numeric_map_row

    prepared = prepare_map_block_rows(param_block, valid_block, query_block)
    return _map_rows_with_status(numeric_map_row, prepared, method=method, dup_code=dup_code)


def map_block_numpy(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .numeric_rows import numeric_map_row

    param_rows, valid_rows, query_rows, output_shape = prepare_map_block_rows(param_block, valid_block, query_block)
    rows = int(param_rows.shape[0])
    query_size = int(query_rows.shape[1])
    i0 = np.zeros((rows, query_size), dtype=np.int64)
    i1 = np.zeros((rows, query_size), dtype=np.int64)
    alpha = np.zeros((rows, query_size), dtype=np.float64)
    valid = np.zeros((rows, query_size), dtype=bool)
    for row in range(rows):
        row_i0, row_i1, row_alpha, row_valid = numeric_map_row(
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
    from .numeric_rows import numeric_bounds_row

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
        row_i0, row_i1 = numeric_bounds_row(param_rows[row], valid_rows[row], start_rows[row], stop_rows[row])
        i0[row] = row_i0
        i1[row] = row_i1
    return i0.reshape(output_shape), i1.reshape(output_shape)


def datetime_map_block_numpy(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .datetime_rows import datetime_map_row

    param_rows, valid_rows, query_rows, output_shape = prepare_datetime_map_block_rows(
        param_block,
        valid_block,
        query_block,
    )
    rows = int(param_rows.shape[0])
    query_size = int(query_rows.shape[1])
    i0 = np.zeros((rows, query_size), dtype=np.int64)
    i1 = np.zeros((rows, query_size), dtype=np.int64)
    alpha = np.zeros((rows, query_size), dtype=np.float64)
    valid = np.zeros((rows, query_size), dtype=bool)
    for row in range(rows):
        row_i0, row_i1, row_alpha, row_valid = datetime_map_row(
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


def datetime_map_block_numpy_status(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    query_block: np.ndarray,
    *,
    method: str,
    dup_code: int,
) -> tuple[np.ndarray, ...]:
    from .datetime_rows import datetime_map_row

    prepared = prepare_datetime_map_block_rows(param_block, valid_block, query_block)
    return _map_rows_with_status(datetime_map_row, prepared, method=method, dup_code=dup_code)


def validate_map_domain_block_numpy_status(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    *,
    param_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compact source-domain validation state for each public row."""
    shape = np.shape(param_block)
    if param_kind == "datetime64":
        from .datetime_rows import datetime_map_row

        query = np.empty(shape[:-1] + (0,), dtype="datetime64[ns]")
        prepared = prepare_datetime_map_block_rows(param_block, valid_block, query)
        result = _map_rows_with_status(
            datetime_map_row, prepared, method="nearest", dup_code=0
        )
    else:
        from .numeric_rows import numeric_map_row

        query = np.empty(shape[:-1] + (0,), dtype=np.asarray(param_block).dtype)
        prepared = prepare_map_block_rows(param_block, valid_block, query)
        result = _map_rows_with_status(
            numeric_map_row, prepared, method="nearest", dup_code=0
        )
    return result[4], result[5], result[6]


def datetime_bounds_block_numpy(
    param_block: np.ndarray,
    valid_block: np.ndarray,
    start_block: np.ndarray,
    stop_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from .datetime_rows import datetime_bounds_row

    param_rows, valid_rows, start_rows, stop_rows, output_shape = prepare_datetime_bounds_block_rows(
        param_block,
        valid_block,
        start_block,
        stop_block,
    )
    rows = int(param_rows.shape[0])
    i0 = np.zeros(rows, dtype=np.int64)
    i1 = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        row_i0, row_i1 = datetime_bounds_row(param_rows[row], valid_rows[row], start_rows[row], stop_rows[row])
        i0[row] = row_i0
        i1[row] = row_i1
    return i0.reshape(output_shape), i1.reshape(output_shape)


__all__ = [
    "bounds_block_numpy",
    "datetime_bounds_block_numpy",
    "datetime_map_block_numpy",
    "datetime_map_block_numpy_status",
    "map_block_numpy",
    "map_block_numpy_status",
    "validate_map_domain_block_numpy_status",
]
