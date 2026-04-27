from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _centered_windows(values: np.ndarray, *, window: int, fill_value: float) -> np.ndarray:
    radius = window // 2
    pad = [(0, 0)] * values.ndim
    pad[-1] = (radius, radius)
    padded = np.pad(values, pad, mode="constant", constant_values=fill_value)
    return sliding_window_view(padded, window_shape=window, axis=-1)


def _centered_boolean_windows(values: np.ndarray, *, window: int) -> np.ndarray:
    radius = window // 2
    pad = [(0, 0)] * values.ndim
    pad[-1] = (radius, radius)
    padded = np.pad(values, pad, mode="constant", constant_values=False)
    return sliding_window_view(padded, window_shape=window, axis=-1)


def build_local_poly_weight_stack(
    param: np.ndarray,
    valid: np.ndarray,
    *,
    window: int,
    poly_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    param_windows = _centered_windows(param, window=window, fill_value=np.nan)
    valid_windows = _centered_boolean_windows(valid, window=window)
    dx = param_windows - param[..., :, None]
    window_valid = valid_windows & np.isfinite(dx)
    exponents = np.arange(poly_order + 1, dtype="float64")
    design = np.power(dx[..., None], exponents)
    design = np.where(window_valid[..., None], design, 0.0)
    transposed = np.swapaxes(design, -2, -1)
    gram = transposed @ design
    gram_pinv = np.linalg.pinv(gram, rcond=1e-12)
    weight_stack = gram_pinv @ transposed
    smooth_weights = weight_stack[..., 0, :]
    deriv_weights = weight_stack[..., 1, :]
    sufficient = window_valid.sum(axis=-1, dtype="int64") >= (poly_order + 1)
    sample_valid = valid & sufficient
    return smooth_weights, deriv_weights, window_valid, sample_valid


__all__ = ["build_local_poly_weight_stack"]
