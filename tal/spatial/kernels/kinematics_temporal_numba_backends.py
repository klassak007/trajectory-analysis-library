from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_scan import ScanAxisSpec, ScanInputSpec, prepare_scan_rows
from tal.utils.numba_support import njit_kernel, require_numba

_STATUS_OK = 0
_STATUS_LEFT_PACKED = 1
_STATUS_EMPTY = 2
_STATUS_MONOTONIC = 3


@lru_cache(maxsize=1)
def _compiled_trapezoid_block():
    numba = require_numba("spatial.kinematics.temporal.integral_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _trapezoid_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _trapezoid_count_valid, _trapezoid_row_impl
    _trapezoid_count_valid = njit_kernel(numba, _trapezoid_count_valid)
    _trapezoid_row_impl = njit_kernel(numba, _trapezoid_row_impl)


def _validate_shapes(values: object, param: object, valid: object, *, owner: str) -> None:
    value_shape = np.shape(values)
    param_shape = np.shape(param)
    if len(value_shape) < 2:
        raise ValueError(f"{owner}: values must include sequence and core dimensions.")
    if param_shape != np.shape(valid):
        raise ValueError(f"{owner}: param/valid shapes must match; got {param_shape!r} vs {np.shape(valid)!r}.")
    if value_shape[:-1] != param_shape:
        raise ValueError(
            f"{owner}: values non-core shape must match param shape; got {value_shape[:-1]!r} vs {param_shape!r}."
        )


def _prepare_trapezoid_blocks(
    values: object,
    param: object,
    valid: object,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    _validate_shapes(values, param, valid, owner=owner)
    output_core_shape = (int(np.shape(values)[-1]),)
    prepared = prepare_scan_rows(
        (values, param, valid),
        (
            ScanInputSpec("values", 1, 1, np.float64),
            ScanInputSpec("param", 1, 0, np.float64),
            ScanInputSpec("valid", 1, 0, bool),
        ),
        ordered_axes=(ScanAxisSpec("sequence", "scan"),),
        output_core_shapes=(output_core_shape,),
        owner=owner,
    )
    value_rows, param_rows, valid_rows = prepared.row_arrays
    return value_rows, param_rows, valid_rows, prepared.output_shapes[0]


def _raise_trapezoid_status(status: int, *, owner: str) -> None:
    if status == _STATUS_LEFT_PACKED:
        raise ValueError(f"{owner}: valid mask must be left-packed on the temporal sequence axis.")
    if status == _STATUS_EMPTY:
        raise ValueError(f"{owner}: temporal operation requires at least 1 valid samples per batch row.")
    if status == _STATUS_MONOTONIC:
        raise ValueError(f"{owner}: param domain must be finite and strictly increasing on the valid temporal domain.")


def cumulative_trapezoid_block_numba(
    values: object,
    param: object,
    valid: object,
    *,
    initial_value: float,
    owner: str,
) -> np.ndarray:
    require_numba(owner)
    value_rows, param_rows, valid_rows, output_shape = _prepare_trapezoid_blocks(
        values,
        param,
        valid,
        owner=owner,
    )
    out, status = _compiled_trapezoid_block()(value_rows, param_rows, valid_rows, float(initial_value))
    _raise_trapezoid_status(int(status), owner=owner)
    return out.reshape(output_shape)


def _trapezoid_count_valid(valid_row):
    count = 0
    seen_false = False
    for idx in range(valid_row.shape[0]):
        if not valid_row[idx]:
            seen_false = True
            continue
        if seen_false:
            return -1
        count += 1
    return count


def _trapezoid_row_impl(values, param, valid, row, initial_value, out):
    count = _trapezoid_count_valid(valid[row])
    if count < 0:
        return _STATUS_LEFT_PACKED
    if count == 0:
        return _STATUS_EMPTY
    total = initial_value
    for comp in range(values.shape[2]):
        out[row, 0, comp] = total
    for idx in range(1, count):
        dt = param[row, idx] - param[row, idx - 1]
        if not np.isfinite(dt) or dt <= 0.0:
            return _STATUS_MONOTONIC
        for comp in range(values.shape[2]):
            total = out[row, idx - 1, comp] + 0.5 * (values[row, idx, comp] + values[row, idx - 1, comp]) * dt
            out[row, idx, comp] = total
    return _STATUS_OK


def _trapezoid_block_impl(values, param, valid, initial_value):
    out = np.empty_like(values)
    out.fill(np.nan)
    for row in range(values.shape[0]):
        status = _trapezoid_row_impl(values, param, valid, row, initial_value, out)
        if status != _STATUS_OK:
            return out, status
    return out, _STATUS_OK


__all__ = ["cumulative_trapezoid_block_numba"]
