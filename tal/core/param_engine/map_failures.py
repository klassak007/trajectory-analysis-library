from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from tal.utils.xarray_namespace import unique_temp_dim

from .blocking import PARAM_LOGICAL_ROW_LIMIT

MAP_STATUS_OK = 0
MAP_STATUS_MONOTONIC = 1
MAP_STATUS_DUPLICATE = 2
MAP_STATUS_UNSAFE_SOURCE = 3
MAP_STATUS_UNSAFE_QUERY = 4
MAP_STATUS_DATETIME_SPAN = 5

_MONOTONIC_ERROR = "build_param_map: parameter coordinate must be monotonic non-decreasing on valid domain."
_DUPLICATE_ERROR = "build_param_map: duplicate parameter bracket encountered for linear interpolation."
_UNSAFE_MIXED_ERROR = (
    "cannot be represented exactly as float64; use matching integer parameter/query "
    "dtypes or rescale the parameter domain"
)
_DATETIME_SPAN_ERROR = "build_param_map: datetime64 values span more than int64 nanoseconds from row anchor."


class ParamMapFailure(ValueError):
    """Internal typed failure carrying stable public ordering state."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        position: int = -1,
        detail: int = 0,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.position = position
        self.detail = detail


def monotonic_map_failure(*, owner: str) -> ParamMapFailure:
    return ParamMapFailure(
        MAP_STATUS_MONOTONIC,
        f"{owner}: parameter coordinate must be monotonic non-decreasing on valid domain.",
    )


def duplicate_map_failure(*, position: int) -> ParamMapFailure:
    return ParamMapFailure(
        MAP_STATUS_DUPLICATE,
        _DUPLICATE_ERROR,
        position=position,
    )


def unsafe_source_map_failure(value: int, *, owner: str) -> ParamMapFailure:
    return ParamMapFailure(
        MAP_STATUS_UNSAFE_SOURCE,
        f"{owner}: integer parameter value {value!r} {_UNSAFE_MIXED_ERROR}.",
        detail=value,
    )


def unsafe_query_map_failure(value: int, *, owner: str, position: int) -> ParamMapFailure:
    return ParamMapFailure(
        MAP_STATUS_UNSAFE_QUERY,
        f"{owner}: integer operand {value!r} {_UNSAFE_MIXED_ERROR}.",
        position=position,
        detail=value,
    )


def datetime_span_map_failure(*, owner: str, position: int) -> ParamMapFailure:
    return ParamMapFailure(
        MAP_STATUS_DATETIME_SPAN,
        f"{owner}: datetime64 values span more than int64 nanoseconds from row anchor.",
        position=position,
    )


def raise_map_status(status: int, detail: int = 0) -> None:
    if status == MAP_STATUS_MONOTONIC:
        raise ValueError(_MONOTONIC_ERROR)
    if status == MAP_STATUS_DUPLICATE:
        raise ValueError(_DUPLICATE_ERROR)
    if status == MAP_STATUS_UNSAFE_SOURCE:
        raise ValueError(f"build_param_map: integer parameter value {detail!r} {_UNSAFE_MIXED_ERROR}.")
    if status == MAP_STATUS_UNSAFE_QUERY:
        raise ValueError(f"build_param_map: integer operand {detail!r} {_UNSAFE_MIXED_ERROR}.")
    if status == MAP_STATUS_DATETIME_SPAN:
        raise ValueError(_DATETIME_SPAN_ERROR)


def _failure_index(statuses: np.ndarray, positions: np.ndarray, row: int) -> int | None:
    failed = np.flatnonzero(statuses[row] != MAP_STATUS_OK)
    if failed.size == 0:
        return None
    return int(failed[int(np.argmin(positions[row, failed]))])


def _ordered_failure(
    status: np.ndarray,
    position: np.ndarray,
    detail: np.ndarray,
) -> np.int8:
    statuses = np.asarray(status).reshape(-1, status.shape[-1])
    positions = np.asarray(position).reshape(statuses.shape)
    details = np.asarray(detail).reshape(statuses.shape)
    for row in range(statuses.shape[0]):
        selected = _failure_index(statuses, positions, row)
        if selected is not None:
            raise_map_status(int(statuses[row, selected]), int(details[row, selected]))
    return np.int8(0)


def raise_ordered_map_failures(
    status: np.ndarray,
    position: np.ndarray,
    detail: np.ndarray,
) -> None:
    """Raise the first failure in public outer-row/query order."""
    _ordered_failure(status[..., None], position[..., None], detail[..., None])


def _first_failure(
    status: np.ndarray,
    position: np.ndarray,
    detail: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    statuses = np.asarray(status).reshape(-1, status.shape[-1])
    positions = np.asarray(position).reshape(statuses.shape)
    details = np.asarray(detail).reshape(statuses.shape)
    selected = np.zeros(statuses.shape[0], dtype=np.int8)
    selected_position = np.full(statuses.shape[0], -1, dtype=np.int64)
    selected_detail = np.zeros(statuses.shape[0], dtype=object)
    for row in range(statuses.shape[0]):
        index = _failure_index(statuses, positions, row)
        if index is None:
            continue
        selected[row] = statuses[row, index]
        selected_position[row] = positions[row, index]
        selected_detail[row] = details[row, index]
    outer_shape = status.shape[:-1]
    return (
        selected.reshape(outer_shape),
        selected_position.reshape(outer_shape),
        selected_detail.reshape(outer_shape),
    )


def _absolute_positions(
    positions: Sequence[xr.DataArray],
    starts: Sequence[int],
) -> tuple[xr.DataArray, ...]:
    return tuple(
        xr.where(value >= 0, value + start, value)
        for value, start in zip(positions, starts, strict=True)
    )


def first_map_failure(
    statuses: Sequence[xr.DataArray],
    positions: Sequence[xr.DataArray],
    details: Sequence[xr.DataArray],
    *,
    starts: Sequence[int],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Select the earliest query failure for each retained outer row."""
    taken = tuple(str(dim) for value in statuses for dim in value.dims)
    block_dim = unique_temp_dim("__tal_map_block__", taken_dims=taken)
    inputs = (
        xr.concat(tuple(statuses), dim=block_dim),
        xr.concat(_absolute_positions(positions, starts), dim=block_dim),
        xr.concat(tuple(details), dim=block_dim),
    )
    return xr.apply_ufunc(
        _first_failure,
        *inputs,
        input_core_dims=[[block_dim], [block_dim], [block_dim]],
        output_core_dims=[[], [], []],
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int8, np.int64, object],
    )


def _first_ordered_failure(
    status: np.ndarray,
    position: np.ndarray,
    detail: np.ndarray,
) -> tuple[np.int8, np.int64, object]:
    statuses = np.asarray(status).reshape(-1)
    failed = np.flatnonzero(statuses != MAP_STATUS_OK)
    if failed.size == 0:
        return np.int8(0), np.int64(-1), 0
    selected = int(failed[0])
    positions = np.asarray(position).reshape(-1)
    details = np.asarray(detail).reshape(-1)
    return np.int8(statuses[selected]), np.int64(positions[selected]), details[selected]


def summarize_map_failure(
    status: xr.DataArray,
    position: xr.DataArray,
    detail: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Reduce one bounded public-row block to its earliest failure."""
    dims = tuple(status.dims)
    return xr.apply_ufunc(
        _first_ordered_failure,
        status.transpose(*dims),
        position.transpose(*dims),
        detail.transpose(*dims),
        input_core_dims=[list(dims), list(dims), list(dims)],
        output_core_dims=[[], [], []],
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int8, np.int64, object],
    )


def _summarize_failure_group(
    summaries: Sequence[tuple[xr.DataArray, xr.DataArray, xr.DataArray]],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    taken = tuple(str(dim) for summary in summaries for value in summary for dim in value.dims)
    block_dim = unique_temp_dim("__tal_map_block__", taken_dims=taken)
    columns = tuple(
        xr.concat(tuple(summary[index] for summary in summaries), dim=block_dim)
        for index in range(3)
    )
    return summarize_map_failure(*columns)


def _bounded_failure_summary(
    summaries: Sequence[tuple[xr.DataArray, xr.DataArray, xr.DataArray]],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    current = tuple(summaries)
    while len(current) > 1:
        groups = (
            current[start : start + PARAM_LOGICAL_ROW_LIMIT]
            for start in range(0, len(current), PARAM_LOGICAL_ROW_LIMIT)
        )
        current = tuple(_summarize_failure_group(group) for group in groups)
    return current[0]


def _raise_failure_summary(status: np.ndarray, position: np.ndarray, detail: np.ndarray) -> np.int8:
    raise_ordered_map_failures(np.asarray(status), np.asarray(position), np.asarray(detail))
    return np.int8(0)


def ordered_map_failure_dependency(
    summaries: Sequence[tuple[xr.DataArray, xr.DataArray, xr.DataArray]],
) -> xr.DataArray:
    """Raise from bounded block summaries in public block order."""
    status, position, detail = _bounded_failure_summary(summaries)
    return xr.apply_ufunc(
        _raise_failure_summary,
        status,
        position,
        detail,
        input_core_dims=[[], [], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[np.int8],
    )


def attach_map_failure_dependency(
    columns: tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray],
    dependency: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    if dependency.chunks is None:
        return columns
    i0, i1, alpha, valid = columns
    return (
        i0 + dependency.astype(i0.dtype),
        i1 + dependency.astype(i1.dtype),
        alpha + dependency.astype(alpha.dtype),
        valid | dependency.astype(bool),
    )


__all__ = [
    "MAP_STATUS_DATETIME_SPAN",
    "MAP_STATUS_DUPLICATE",
    "MAP_STATUS_MONOTONIC",
    "MAP_STATUS_OK",
    "MAP_STATUS_UNSAFE_QUERY",
    "MAP_STATUS_UNSAFE_SOURCE",
    "ParamMapFailure",
    "attach_map_failure_dependency",
    "datetime_span_map_failure",
    "duplicate_map_failure",
    "first_map_failure",
    "monotonic_map_failure",
    "ordered_map_failure_dependency",
    "raise_map_status",
    "raise_ordered_map_failures",
    "summarize_map_failure",
    "unsafe_query_map_failure",
    "unsafe_source_map_failure",
]
