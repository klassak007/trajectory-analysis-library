from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.integrate import cumulative_simpson as scipy_cumulative_simpson

from .local_poly_weights import build_local_poly_weight_stack


def _require_shapes(values: np.ndarray, param: np.ndarray, valid: np.ndarray, *, owner: str) -> None:
    if values.ndim < 2:
        raise ValueError(f"{owner}: values must include sequence and core dimensions.")
    if param.shape != valid.shape:
        raise ValueError(f"{owner}: param/valid shapes must match; got {param.shape!r} vs {valid.shape!r}.")
    if values.shape[:-1] != param.shape:
        raise ValueError(
            f"{owner}: values non-core shape must match param shape; "
            f"got {values.shape[:-1]!r} vs {param.shape!r}."
        )


def _left_packed_counts(valid: np.ndarray, *, owner: str) -> np.ndarray:
    seq_size = int(valid.shape[-1])
    counts = valid.sum(axis=-1, dtype="int64")
    expected = np.arange(seq_size, dtype="int64") < counts[..., None]
    if np.any(valid != expected):
        raise ValueError(f"{owner}: valid mask must be left-packed on the temporal sequence axis.")
    return counts


def _require_min_count(counts: np.ndarray, *, minimum: int, owner: str) -> None:
    if np.any(counts < minimum):
        raise ValueError(
            f"{owner}: temporal operation requires at least {minimum} valid samples per batch row."
        )


def _require_positive_odd_window(window: int, *, owner: str) -> None:
    if not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError(f"{owner}: window must be a positive odd integer.")


def _centered_value_windows(values: np.ndarray, *, window: int) -> np.ndarray:
    radius = window // 2
    pad = [(0, 0)] * values.ndim
    pad[-2] = (radius, radius)
    padded = np.pad(values, pad, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, window_shape=window, axis=-2)
    return np.moveaxis(windows, -1, -2)


def _centered_sequence_windows(sequence: np.ndarray, *, window: int, fill_value: float | bool) -> np.ndarray:
    radius = window // 2
    pad = [(0, 0)] * sequence.ndim
    pad[-1] = (radius, radius)
    padded = np.pad(sequence, pad, mode="constant", constant_values=fill_value)
    return sliding_window_view(padded, window_shape=window, axis=-1)


def _simpson_safe_tail_param(
    param: np.ndarray,
    counts: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seq_size = int(param.shape[-1])
    sample_index = np.arange(seq_size, dtype="int64")
    sample_active = sample_index < counts[..., None]
    last_index = counts - 1
    last_param = np.take_along_axis(param, last_index[..., None], axis=-1)[..., 0]
    last_dt = np.take_along_axis(dt, (counts - 2)[..., None], axis=-1)[..., 0]
    offset = np.clip(sample_index - last_index[..., None], 0, None)
    synthetic_tail = last_param[..., None] + last_dt[..., None] * offset
    safe_param = np.where(sample_active, param, synthetic_tail)
    return safe_param, sample_active


def _simpson_safe_tail_values(
    values: np.ndarray,
    counts: np.ndarray,
    sample_active: np.ndarray,
) -> np.ndarray:
    last_index = counts - 1
    last_value = np.take_along_axis(values, last_index[..., None, None], axis=-2)[..., 0, :]
    return np.where(sample_active[..., None], values, last_value[..., None, :])


def _require_monotonic_param(param: np.ndarray, counts: np.ndarray, *, owner: str) -> np.ndarray:
    seq_size = int(param.shape[-1])
    if seq_size <= 1:
        return np.empty(param.shape[:-1] + (0,), dtype="float64")
    dt = param[..., 1:] - param[..., :-1]
    active_pairs = np.arange(seq_size - 1, dtype="int64") < (counts - 1)[..., None]
    bad = active_pairs & (~np.isfinite(dt) | (dt <= 0.0))
    if np.any(bad):
        raise ValueError(
            f"{owner}: param domain must be finite and strictly increasing on the valid temporal domain."
        )
    return dt


def finite_difference_one_sided_kernel(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    owner = "spatial.kinematics.temporal.derivative.kernel"
    vals = np.asarray(values, dtype="float64")
    prm = np.asarray(param, dtype="float64")
    mask = np.asarray(valid, dtype=bool)
    _require_shapes(vals, prm, mask, owner=owner)
    counts = _left_packed_counts(mask, owner=owner)
    _require_min_count(counts, minimum=2, owner=owner)
    dt = _require_monotonic_param(prm, counts, owner=owner)
    seq_size = int(vals.shape[-2])
    active_pairs = np.arange(seq_size - 1, dtype="int64") < (counts - 1)[..., None]
    dv = vals[..., 1:, :] - vals[..., :-1, :]
    safe_dt = np.where(active_pairs, dt, 1.0)
    slope = np.divide(
        dv,
        safe_dt[..., None],
        out=np.full_like(dv, np.nan, dtype="float64"),
        where=active_pairs[..., None],
    )
    out = np.full_like(vals, np.nan, dtype="float64")
    out[..., 0, :] = slope[..., 0, :]
    out[..., 1:, :] = slope
    sample_active = np.arange(seq_size, dtype="int64") < counts[..., None]
    return np.where(sample_active[..., None], out, np.nan)


def moving_average_partial_renorm_kernel(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    owner = "spatial.kinematics.temporal.smoothing.kernel"
    vals = np.asarray(values, dtype="float64")
    prm = np.asarray(param, dtype="float64")
    mask = np.asarray(valid, dtype=bool)
    _require_shapes(vals, prm, mask, owner=owner)
    _require_positive_odd_window(window, owner=owner)
    counts = _left_packed_counts(mask, owner=owner)
    _require_min_count(counts, minimum=1, owner=owner)
    _ = _require_monotonic_param(prm, counts, owner=owner)
    value_windows = _centered_value_windows(vals, window=window)
    radius = window // 2
    pad = [(0, 0)] * mask.ndim
    pad[-1] = (radius, radius)
    mask_windows = sliding_window_view(np.pad(mask, pad, mode="constant", constant_values=False), window, axis=-1)
    active_vals = np.where(mask_windows[..., None], value_windows, 0.0)
    numer = active_vals.sum(axis=-2, dtype="float64")
    denom = mask_windows.sum(axis=-1, dtype="float64")
    averaged = np.divide(numer, denom[..., None], out=np.full_like(numer, np.nan), where=denom[..., None] > 0.0)
    return np.where(mask[..., None], averaged, np.nan)


def gaussian_partial_renorm_kernel(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    sigma: float,
) -> np.ndarray:
    owner = "spatial.kinematics.temporal.smoothing.kernel"
    vals = np.asarray(values, dtype="float64")
    prm = np.asarray(param, dtype="float64")
    mask = np.asarray(valid, dtype=bool)
    _require_shapes(vals, prm, mask, owner=owner)
    _require_positive_odd_window(window, owner=owner)
    sigma_value = float(sigma)
    if not (sigma_value > 0.0 and sigma_value == sigma_value and abs(sigma_value) != float("inf")):
        raise ValueError(f"{owner}: sigma must be positive and finite.")
    counts = _left_packed_counts(mask, owner=owner)
    _require_min_count(counts, minimum=1, owner=owner)
    _ = _require_monotonic_param(prm, counts, owner=owner)
    value_windows = _centered_value_windows(vals, window=window)
    param_windows = _centered_sequence_windows(prm, window=window, fill_value=np.nan)
    mask_windows = _centered_sequence_windows(mask, window=window, fill_value=False)
    center = prm[..., :, None]
    scaled = (param_windows - center) / sigma_value
    raw_weights = np.exp(-0.5 * scaled * scaled)
    weights = np.where(mask_windows, raw_weights, 0.0)
    numer = np.where(mask_windows[..., None], value_windows, 0.0) * weights[..., None]
    denom = weights.sum(axis=-1, dtype="float64")
    out = np.divide(
        numer.sum(axis=-2, dtype="float64"),
        denom[..., None],
        out=np.full_like(vals, np.nan, dtype="float64"),
        where=denom[..., None] > 0.0,
    )
    return np.where(mask[..., None], out, np.nan)


def local_poly_smooth_kernel(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    poly_order: int,
) -> np.ndarray:
    owner = "spatial.kinematics.temporal.smoothing.kernel"
    vals = np.asarray(values, dtype="float64")
    prm = np.asarray(param, dtype="float64")
    mask = np.asarray(valid, dtype=bool)
    _require_shapes(vals, prm, mask, owner=owner)
    _require_positive_odd_window(window, owner=owner)
    counts = _left_packed_counts(mask, owner=owner)
    _require_min_count(counts, minimum=poly_order + 1, owner=owner)
    _ = _require_monotonic_param(prm, counts, owner=owner)
    weights, _, window_valid, sample_valid = build_local_poly_weight_stack(
        prm, mask, window=window, poly_order=poly_order
    )
    windows = _centered_value_windows(vals, window=window)
    weighted = np.where(window_valid[..., None], windows, 0.0) * weights[..., None]
    smoothed = weighted.sum(axis=-2, dtype="float64")
    return np.where(sample_valid[..., None], smoothed, np.nan)


def local_poly_first_derivative_kernel(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    poly_order: int,
) -> np.ndarray:
    owner = "spatial.kinematics.temporal.derivative.kernel"
    vals = np.asarray(values, dtype="float64")
    prm = np.asarray(param, dtype="float64")
    mask = np.asarray(valid, dtype=bool)
    _require_shapes(vals, prm, mask, owner=owner)
    _require_positive_odd_window(window, owner=owner)
    counts = _left_packed_counts(mask, owner=owner)
    _require_min_count(counts, minimum=poly_order + 1, owner=owner)
    _ = _require_monotonic_param(prm, counts, owner=owner)
    _, weights, window_valid, sample_valid = build_local_poly_weight_stack(
        prm, mask, window=window, poly_order=poly_order
    )
    windows = _centered_value_windows(vals, window=window)
    weighted = np.where(window_valid[..., None], windows, 0.0) * weights[..., None]
    derived = weighted.sum(axis=-2, dtype="float64")
    return np.where(sample_valid[..., None], derived, np.nan)


def cumulative_trapezoid_kernel(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    initial_value: float,
) -> np.ndarray:
    owner = "spatial.kinematics.temporal.integral.kernel"
    vals = np.asarray(values, dtype="float64")
    prm = np.asarray(param, dtype="float64")
    mask = np.asarray(valid, dtype=bool)
    _require_shapes(vals, prm, mask, owner=owner)
    counts = _left_packed_counts(mask, owner=owner)
    _require_min_count(counts, minimum=1, owner=owner)
    dt = _require_monotonic_param(prm, counts, owner=owner)
    seq_size = int(vals.shape[-2])
    active_pairs = np.arange(seq_size - 1, dtype="int64") < (counts - 1)[..., None]
    increments = 0.5 * (vals[..., 1:, :] + vals[..., :-1, :]) * dt[..., None]
    increments = np.where(active_pairs[..., None], increments, 0.0)
    cumulative = np.cumsum(increments, axis=-2, dtype="float64")
    base = float(initial_value)
    out = np.full_like(vals, np.nan, dtype="float64")
    out[..., 0, :] = base
    out[..., 1:, :] = base + cumulative
    sample_active = np.arange(seq_size, dtype="int64") < counts[..., None]
    return np.where(sample_active[..., None], out, np.nan)


def cumulative_simpson_kernel(
    values: np.ndarray,
    param: np.ndarray,
    valid: np.ndarray,
    *,
    initial_value: float,
) -> np.ndarray:
    owner = "spatial.kinematics.temporal.integral.kernel"
    vals = np.asarray(values, dtype="float64")
    prm = np.asarray(param, dtype="float64")
    mask = np.asarray(valid, dtype=bool)
    _require_shapes(vals, prm, mask, owner=owner)
    counts = _left_packed_counts(mask, owner=owner)
    _require_min_count(counts, minimum=3, owner=owner)
    dt = _require_monotonic_param(prm, counts, owner=owner)
    safe_param, sample_active = _simpson_safe_tail_param(prm, counts, dt)
    safe_values = _simpson_safe_tail_values(vals, counts, sample_active)
    seq_size = int(vals.shape[-2])
    rows = int(np.prod(vals.shape[:-2], dtype="int64")) if vals.ndim > 2 else 1
    core_size = int(vals.shape[-1])
    vals_rows = safe_values.reshape(rows, seq_size, core_size)
    param_rows = safe_param.reshape(rows, seq_size)
    flat_vals = np.transpose(vals_rows, (0, 2, 1)).reshape(rows * core_size, seq_size)
    flat_param = np.repeat(param_rows, core_size, axis=0)
    integrated = scipy_cumulative_simpson(flat_vals, x=flat_param, axis=-1, initial=float(initial_value))
    rows_out = integrated.reshape(rows, core_size, seq_size).transpose(0, 2, 1)
    out = rows_out.reshape(vals.shape)
    return np.where(sample_active[..., None], out, np.nan)


__all__ = [
    "cumulative_simpson_kernel",
    "cumulative_trapezoid_kernel",
    "finite_difference_one_sided_kernel",
    "gaussian_partial_renorm_kernel",
    "local_poly_first_derivative_kernel",
    "local_poly_smooth_kernel",
    "moving_average_partial_renorm_kernel",
]
