from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_scan import ScanAxisSpec, ScanInputSpec, prepare_scan_rows
from tal.utils.numba_support import njit_kernel, require_numba

_STATUS_OK = 0
_STATUS_LEFT_PACKED = 1
_STATUS_EMPTY = 2
_STATUS_MONOTONIC = 3

_HELPERS_JITTED = False


@lru_cache(maxsize=1)
def _compiled_trapezoid_block():
    numba = require_numba("spatial.kinematics.temporal.integral_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _trapezoid_block_impl)


@lru_cache(maxsize=1)
def _compiled_simpson_block():
    numba = require_numba("spatial.kinematics.temporal.integral_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _simpson_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _HELPERS_JITTED
    global _simpson_forward_unequal_interval, _simpson_interval_integral
    global _simpson_param_at, _simpson_reverse_unequal_interval, _simpson_row_impl
    global _simpson_subinterval_integral, _simpson_value_at
    global _trapezoid_count_valid, _trapezoid_row_impl, _validate_temporal_param_prefix
    if _HELPERS_JITTED:
        return
    _trapezoid_count_valid = njit_kernel(numba, _trapezoid_count_valid)
    _validate_temporal_param_prefix = njit_kernel(numba, _validate_temporal_param_prefix)
    _trapezoid_row_impl = njit_kernel(numba, _trapezoid_row_impl)
    _simpson_param_at = njit_kernel(numba, _simpson_param_at)
    _simpson_value_at = njit_kernel(numba, _simpson_value_at)
    _simpson_subinterval_integral = njit_kernel(numba, _simpson_subinterval_integral)
    _simpson_forward_unequal_interval = njit_kernel(numba, _simpson_forward_unequal_interval)
    _simpson_reverse_unequal_interval = njit_kernel(numba, _simpson_reverse_unequal_interval)
    _simpson_interval_integral = njit_kernel(numba, _simpson_interval_integral)
    _simpson_row_impl = njit_kernel(numba, _simpson_row_impl)
    _HELPERS_JITTED = True


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


def _prepare_integral_blocks(
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


def _raise_integral_status(status: int, *, owner: str, minimum: int) -> None:
    if status == _STATUS_LEFT_PACKED:
        raise ValueError(f"{owner}: valid mask must be left-packed on the temporal sequence axis.")
    if status == _STATUS_EMPTY:
        raise ValueError(f"{owner}: temporal operation requires at least {minimum} valid samples per batch row.")
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
    value_rows, param_rows, valid_rows, output_shape = _prepare_integral_blocks(
        values,
        param,
        valid,
        owner=owner,
    )
    out, status = _compiled_trapezoid_block()(value_rows, param_rows, valid_rows, float(initial_value))
    _raise_integral_status(int(status), owner=owner, minimum=1)
    return out.reshape(output_shape)


def cumulative_simpson_block_numba(
    values: object,
    param: object,
    valid: object,
    *,
    initial_value: float,
    owner: str,
) -> np.ndarray:
    require_numba(owner)
    value_rows, param_rows, valid_rows, output_shape = _prepare_integral_blocks(
        values,
        param,
        valid,
        owner=owner,
    )
    out, status = _compiled_simpson_block()(value_rows, param_rows, valid_rows, float(initial_value))
    _raise_integral_status(int(status), owner=owner, minimum=3)
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


def _validate_temporal_param_prefix(param_row, count):
    for idx in range(1, count):
        dt = param_row[idx] - param_row[idx - 1]
        if not np.isfinite(dt) or dt <= 0.0:
            return _STATUS_MONOTONIC
    return _STATUS_OK


def _trapezoid_row_impl(values, param, valid, row, initial_value, out):
    count = _trapezoid_count_valid(valid[row])
    if count < 0:
        return _STATUS_LEFT_PACKED
    if count == 0:
        return _STATUS_EMPTY
    status = _validate_temporal_param_prefix(param[row], count)
    if status != _STATUS_OK:
        return status
    total = initial_value
    for comp in range(values.shape[2]):
        out[row, 0, comp] = total
    for idx in range(1, count):
        dt = param[row, idx] - param[row, idx - 1]
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


def _simpson_subinterval_integral(f0, f1, f2, h0, h1):
    full = h0 + h1
    h0_full = h0 / full
    h0_h1 = h0 / h1
    h0h0_full_h1 = h0_full * h0_h1
    coeff0 = 3.0 - h0_full
    coeff1 = 3.0 + h0h0_full_h1 + h0_full
    coeff2 = -h0h0_full_h1
    return h0 / 6.0 * (coeff0 * f0 + coeff1 * f1 + coeff2 * f2)


def _simpson_param_at(param, row, idx, count):
    if idx < count:
        return param[row, idx]
    last = count - 1
    last_dt = param[row, last] - param[row, last - 1]
    return param[row, last] + last_dt * (idx - last)


def _simpson_value_at(values, row, idx, count, comp):
    if idx < count:
        return values[row, idx, comp]
    return values[row, count - 1, comp]


def _simpson_forward_unequal_interval(values, param, row, start, count, comp):
    x0 = _simpson_param_at(param, row, start, count)
    x1 = _simpson_param_at(param, row, start + 1, count)
    x2 = _simpson_param_at(param, row, start + 2, count)
    h0 = x1 - x0
    h1 = x2 - x1
    f0 = _simpson_value_at(values, row, start, count, comp)
    f1 = _simpson_value_at(values, row, start + 1, count, comp)
    f2 = _simpson_value_at(values, row, start + 2, count, comp)
    return _simpson_subinterval_integral(f0, f1, f2, h0, h1)


def _simpson_reverse_unequal_interval(values, param, row, start, count, comp):
    x0 = _simpson_param_at(param, row, start, count)
    x1 = _simpson_param_at(param, row, start + 1, count)
    x2 = _simpson_param_at(param, row, start + 2, count)
    h0 = x1 - x0
    h1 = x2 - x1
    f0 = _simpson_value_at(values, row, start, count, comp)
    f1 = _simpson_value_at(values, row, start + 1, count, comp)
    f2 = _simpson_value_at(values, row, start + 2, count, comp)
    return _simpson_subinterval_integral(f2, f1, f0, h1, h0)


def _simpson_interval_integral(values, param, row, interval, count, seq_size, comp):
    if interval == seq_size - 2:
        return _simpson_reverse_unequal_interval(values, param, row, seq_size - 3, count, comp)
    if interval % 2 == 0:
        return _simpson_forward_unequal_interval(values, param, row, interval, count, comp)
    return _simpson_reverse_unequal_interval(values, param, row, interval - 1, count, comp)


def _simpson_row_impl(values, param, valid, row, initial_value, out):
    count = _trapezoid_count_valid(valid[row])
    if count < 0:
        return _STATUS_LEFT_PACKED
    if count < 3:
        return _STATUS_EMPTY
    status = _validate_temporal_param_prefix(param[row], count)
    if status != _STATUS_OK:
        return status
    for comp in range(values.shape[2]):
        total = initial_value
        out[row, 0, comp] = total
        for interval in range(count - 1):
            total += _simpson_interval_integral(values, param, row, interval, count, values.shape[1], comp)
            out[row, interval + 1, comp] = total
    return _STATUS_OK


def _simpson_block_impl(values, param, valid, initial_value):
    out = np.empty_like(values)
    out.fill(np.nan)
    for row in range(values.shape[0]):
        status = _simpson_row_impl(values, param, valid, row, initial_value, out)
        if status != _STATUS_OK:
            return out, status
    return out, _STATUS_OK


__all__ = ["cumulative_simpson_block_numba", "cumulative_trapezoid_block_numba"]
