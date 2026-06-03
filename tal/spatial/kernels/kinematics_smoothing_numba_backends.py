from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows
from tal.utils.numba_support import njit_kernel, require_numba

_STATUS_OK = 0
_STATUS_LEFT_PACKED = 1
_STATUS_EMPTY = 2
_STATUS_MONOTONIC = 3


@lru_cache(maxsize=1)
def _compiled_smoothing_blocks():
    numba = require_numba("spatial.kinematics.temporal.smoothing_backend")
    _jit_kernel_helpers(numba)
    return (
        njit_kernel(numba, _moving_average_block_impl),
        njit_kernel(numba, _gaussian_block_impl),
    )


def _jit_kernel_helpers(numba) -> None:
    global _validated_count, _moving_average_sample_components, _gaussian_sample_components
    global _moving_average_row_impl, _gaussian_row_impl
    _validated_count = njit_kernel(numba, _validated_count)
    _moving_average_sample_components = njit_kernel(numba, _moving_average_sample_components)
    _gaussian_sample_components = njit_kernel(numba, _gaussian_sample_components)
    _moving_average_row_impl = njit_kernel(numba, _moving_average_row_impl)
    _gaussian_row_impl = njit_kernel(numba, _gaussian_row_impl)


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


def _validate_window(window: int, *, owner: str) -> int:
    if not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError(f"{owner}: window must be a positive odd integer.")
    return int(window)


def _validate_sigma(sigma: float, *, owner: str) -> float:
    sigma_value = float(sigma)
    if not (sigma_value > 0.0 and sigma_value == sigma_value and abs(sigma_value) != float("inf")):
        raise ValueError(f"{owner}: sigma must be positive and finite.")
    return sigma_value


def _prepare_smoothing_blocks(
    values: object,
    param: object,
    valid: object,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    _validate_shapes(values, param, valid, owner=owner)
    output_core_shape = tuple(int(size) for size in np.shape(values)[-2:])
    prepared = prepare_block_rows(
        (values, param, valid),
        (
            BlockInputSpec("values", 2, np.float64),
            BlockInputSpec("param", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
        ),
        output_core_shape=output_core_shape,
        owner=owner,
    )
    value_rows, param_rows, valid_rows = prepared.row_arrays
    return value_rows, param_rows, valid_rows, prepared.output_shape


def _raise_smoothing_status(status: int, *, owner: str) -> None:
    if status == _STATUS_LEFT_PACKED:
        raise ValueError(f"{owner}: valid mask must be left-packed on the temporal sequence axis.")
    if status == _STATUS_EMPTY:
        raise ValueError(f"{owner}: temporal operation requires at least 1 valid samples per batch row.")
    if status == _STATUS_MONOTONIC:
        raise ValueError(f"{owner}: param domain must be finite and strictly increasing on the valid temporal domain.")


def moving_average_smoothing_block_numba(
    values: object,
    param: object,
    valid: object,
    *,
    window: int,
    owner: str,
) -> np.ndarray:
    require_numba(owner)
    window_value = _validate_window(window, owner=owner)
    prepared = _prepare_smoothing_blocks(values, param, valid, owner=owner)
    out, status = _compiled_smoothing_blocks()[0](*prepared[:3], window_value)
    _raise_smoothing_status(int(status), owner=owner)
    return out.reshape(prepared[3])


def gaussian_smoothing_block_numba(
    values: object,
    param: object,
    valid: object,
    *,
    window: int,
    sigma: float,
    owner: str,
) -> np.ndarray:
    require_numba(owner)
    window_value = _validate_window(window, owner=owner)
    sigma_value = _validate_sigma(sigma, owner=owner)
    prepared = _prepare_smoothing_blocks(values, param, valid, owner=owner)
    out, status = _compiled_smoothing_blocks()[1](*prepared[:3], window_value, sigma_value)
    _raise_smoothing_status(int(status), owner=owner)
    return out.reshape(prepared[3])


def _validated_count(param_row, valid_row):
    count = 0
    seen_false = False
    for idx in range(valid_row.shape[0]):
        if not valid_row[idx]:
            seen_false = True
            continue
        if seen_false:
            return _STATUS_LEFT_PACKED, 0
        count += 1
    if count == 0:
        return _STATUS_EMPTY, 0
    for idx in range(1, count):
        dt = param_row[idx] - param_row[idx - 1]
        if not np.isfinite(dt) or dt <= 0.0:
            return _STATUS_MONOTONIC, 0
    return _STATUS_OK, count


def _moving_average_sample_components(values, valid, row, idx, radius, out):
    denom = 0.0
    for comp in range(values.shape[2]):
        out[row, idx, comp] = 0.0
    for offset in range(-radius, radius + 1):
        window_idx = idx + offset
        if window_idx < 0 or window_idx >= values.shape[1]:
            continue
        if not valid[row, window_idx]:
            continue
        denom += 1.0
        for comp in range(values.shape[2]):
            out[row, idx, comp] += values[row, window_idx, comp]
    if denom > 0.0:
        for comp in range(values.shape[2]):
            out[row, idx, comp] /= denom
        return
    for comp in range(values.shape[2]):
        out[row, idx, comp] = np.nan


def _gaussian_sample_components(values, param, valid, row, idx, radius, sigma, out):
    denom = 0.0
    center = param[row, idx]
    for comp in range(values.shape[2]):
        out[row, idx, comp] = 0.0
    for offset in range(-radius, radius + 1):
        window_idx = idx + offset
        if window_idx < 0 or window_idx >= values.shape[1]:
            continue
        if not valid[row, window_idx]:
            continue
        scaled = (param[row, window_idx] - center) / sigma
        weight = np.exp(-0.5 * scaled * scaled)
        denom += weight
        for comp in range(values.shape[2]):
            out[row, idx, comp] += values[row, window_idx, comp] * weight
    if denom > 0.0:
        for comp in range(values.shape[2]):
            out[row, idx, comp] /= denom
        return
    for comp in range(values.shape[2]):
        out[row, idx, comp] = np.nan


def _moving_average_row_impl(values, param, valid, row, window, out):
    status, count = _validated_count(param[row], valid[row])
    if status != _STATUS_OK:
        return status
    radius = window // 2
    for idx in range(count):
        _moving_average_sample_components(values, valid, row, idx, radius, out)
    return _STATUS_OK


def _gaussian_row_impl(values, param, valid, row, window, sigma, out):
    status, count = _validated_count(param[row], valid[row])
    if status != _STATUS_OK:
        return status
    radius = window // 2
    for idx in range(count):
        _gaussian_sample_components(values, param, valid, row, idx, radius, sigma, out)
    return _STATUS_OK


def _moving_average_block_impl(values, param, valid, window):
    out = np.empty_like(values)
    out.fill(np.nan)
    for row in range(values.shape[0]):
        status = _moving_average_row_impl(values, param, valid, row, window, out)
        if status != _STATUS_OK:
            return out, status
    return out, _STATUS_OK


def _gaussian_block_impl(values, param, valid, window, sigma):
    out = np.empty_like(values)
    out.fill(np.nan)
    for row in range(values.shape[0]):
        status = _gaussian_row_impl(values, param, valid, row, window, sigma, out)
        if status != _STATUS_OK:
            return out, status
    return out, _STATUS_OK


__all__ = ["gaussian_smoothing_block_numba", "moving_average_smoothing_block_numba"]
