from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows
from tal.utils.numba_support import njit_kernel, require_numba

from .fixed_size_primitives import normalize_quat_row as _normalized_quat
from .fixed_size_primitives import normalize_quat_tuple as _normalize_quat_tuple

_QUAT_SIZE = 4
_LERP_DOT_THRESHOLD = 0.9995
_STATUS_OK = 0
_STATUS_INVALID_QUAT = 1
_STATUS_ALPHA_RANGE = 2


@lru_cache(maxsize=1)
def _compiled_slerp_block():
    numba = require_numba("spatial.rotation.interp_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _slerp_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _fill_nan, _normalize_quat_tuple, _normalized_quat, _slerp_sample, _write_lerp, _write_normalized_tuple, _write_slerp
    _fill_nan = njit_kernel(numba, _fill_nan)
    _normalized_quat = njit_kernel(numba, _normalized_quat)
    _normalize_quat_tuple = njit_kernel(numba, _normalize_quat_tuple)
    _write_normalized_tuple = njit_kernel(numba, _write_normalized_tuple)
    _write_lerp = njit_kernel(numba, _write_lerp)
    _write_slerp = njit_kernel(numba, _write_slerp)
    _slerp_sample = njit_kernel(numba, _slerp_sample)


def _prepare_slerp_blocks(
    q0_block: np.ndarray,
    q1_block: np.ndarray,
    alpha_block: np.ndarray,
    valid_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    _validate_slerp_shapes(q0_block, q1_block, alpha_block, valid_block, owner=owner)
    prepared = prepare_block_rows(
        (q0_block, q1_block, alpha_block, valid_block),
        (
            BlockInputSpec("q0", 2, np.float64),
            BlockInputSpec("q1", 2, np.float64),
            BlockInputSpec("alpha", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
        ),
        output_core_shape=(),
        owner=owner,
    )
    q0_rows, q1_rows, alpha_rows, valid_rows = prepared.row_arrays
    if int(q0_rows.shape[-1]) != _QUAT_SIZE or int(q1_rows.shape[-1]) != _QUAT_SIZE:
        raise ValueError(f"{owner}: expected trailing quaternion dim length 4.")
    query_size = int(q0_rows.shape[-2])
    if (
        int(q1_rows.shape[-2]) != query_size
        or int(alpha_rows.shape[-1]) != query_size
        or int(valid_rows.shape[-1]) != query_size
    ):
        raise ValueError(f"{owner}: q0, q1, alpha, and valid query dimensions must match.")
    return q0_rows, q1_rows, alpha_rows, valid_rows, prepared.outer_shape + (query_size, _QUAT_SIZE)


def _validate_slerp_shapes(
    q0_block: np.ndarray,
    q1_block: np.ndarray,
    alpha_block: np.ndarray,
    valid_block: np.ndarray,
    *,
    owner: str,
) -> None:
    if np.shape(q0_block) != np.shape(q1_block):
        raise ValueError(f"{owner}: q0 and q1 must share shape.")
    if np.shape(alpha_block) != np.shape(valid_block):
        raise ValueError(f"{owner}: alpha and valid must share shape.")
    q0_shape = np.shape(q0_block)
    if q0_shape[:-1] != np.shape(alpha_block):
        raise ValueError(f"{owner}: alpha/valid must match q0/q1 non-core shape.")


def _raise_slerp_status(status: int, *, owner: str) -> None:
    if status == _STATUS_INVALID_QUAT:
        raise ValueError(f"{owner}: quaternion norm must be finite and > 0.")
    if status == _STATUS_ALPHA_RANGE:
        raise ValueError(f"{owner}: finite alpha values must be within [0, 1].")


def slerp_quat_numba(
    q0_block: np.ndarray,
    q1_block: np.ndarray,
    alpha_block: np.ndarray,
    valid_block: np.ndarray,
    *,
    owner: str,
) -> np.ndarray:
    require_numba(owner)
    q0_rows, q1_rows, alpha_rows, valid_rows, output_shape = _prepare_slerp_blocks(
        q0_block,
        q1_block,
        alpha_block,
        valid_block,
        owner=owner,
    )
    out, status = _compiled_slerp_block()(q0_rows, q1_rows, alpha_rows, valid_rows)
    _raise_slerp_status(int(status), owner=owner)
    return out.reshape(output_shape)


def _fill_nan(out):
    out.fill(np.nan)


def _write_normalized_tuple(out, row, idx, quat):
    status, x, y, z, w = _normalize_quat_tuple(quat)
    if status != _STATUS_OK:
        return _STATUS_INVALID_QUAT
    out[row, idx, 0] = x
    out[row, idx, 1] = y
    out[row, idx, 2] = z
    out[row, idx, 3] = w
    return status


def _write_lerp(out, row, idx, q0, q1, t):
    return _write_normalized_tuple(
        out,
        row,
        idx,
        (
            q0[0] + t * (q1[0] - q0[0]),
            q0[1] + t * (q1[1] - q0[1]),
            q0[2] + t * (q1[2] - q0[2]),
            q0[3] + t * (q1[3] - q0[3]),
        ),
    )


def _write_slerp(out, row, idx, q0, q1, dot, t):
    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = t * theta0
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return _write_normalized_tuple(
        out,
        row,
        idx,
        (
            s0 * q0[0] + s1 * q1[0],
            s0 * q0[1] + s1 * q1[1],
            s0 * q0[2] + s1 * q1[2],
            s0 * q0[3] + s1 * q1[3],
        ),
    )


def _slerp_sample(q0_rows, q1_rows, alpha_rows, row, idx, out):
    t = alpha_rows[row, idx]
    if not np.isfinite(t):
        return _STATUS_OK
    if t < 0.0 or t > 1.0:
        return _STATUS_ALPHA_RANGE
    status0, q00, q01, q02, q03 = _normalized_quat(q0_rows, row, idx)
    status1, q10, q11, q12, q13 = _normalized_quat(q1_rows, row, idx)
    if status0 != _STATUS_OK or status1 != _STATUS_OK:
        return _STATUS_INVALID_QUAT
    q0 = (q00, q01, q02, q03)
    q1 = (q10, q11, q12, q13)
    dot = q00 * q10 + q01 * q11 + q02 * q12 + q03 * q13
    if dot < 0.0:
        dot = -dot
        q1 = (-q10, -q11, -q12, -q13)
    if dot > 1.0:
        dot = 1.0
    if dot > _LERP_DOT_THRESHOLD:
        return _write_lerp(out, row, idx, q0, q1, t)
    return _write_slerp(out, row, idx, q0, q1, dot, t)


def _slerp_block_impl(q0_rows, q1_rows, alpha_rows, valid_rows):
    out = np.empty_like(q0_rows)
    _fill_nan(out)
    query_size = q0_rows.shape[1]
    for flat_idx in range(q0_rows.shape[0] * query_size):
        row = flat_idx // query_size
        idx = flat_idx - row * query_size
        if not valid_rows[row, idx]:
            continue
        status = _slerp_sample(q0_rows, q1_rows, alpha_rows, row, idx, out)
        if status != _STATUS_OK:
            return out, status
    return out, _STATUS_OK


__all__ = ["slerp_quat_numba"]
