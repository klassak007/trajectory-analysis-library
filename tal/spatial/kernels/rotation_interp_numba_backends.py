from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_support import njit_kernel, require_numba

from . import quaternion_interp_primitives as _quat_interp_primitives
from .fixed_size_primitives import normalize_quat_tuple as _normalize_quat_tuple
from .fixed_size_primitives import quat_multiply as _quat_multiply
from .quaternion_interp_primitives import slerp_quat as _slerp_quat
from .rotation_interp_blocks import SlerpBlock, iter_slerp_blocks, validate_slerp_shapes
from .rotation_interp_reference import (
    half_turn_tolerance,
    prepare_numba_slerp_endpoints,
)

_QUAT_SIZE = 4
_STATUS_OK = 0
_STATUS_INVALID_QUAT = 1
_STATUS_ALPHA_RANGE = 2
_STATUS_AMBIGUOUS_HALF_TURN = 3


@lru_cache(maxsize=1)
def _compiled_slerp_block():
    numba = require_numba("spatial.rotation.interp_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _slerp_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _fill_nan, _shared_slerp_quat, _slerp_sample
    _, _, _shared_slerp_quat = _compile_slerp_dependencies(numba)
    _fill_nan = njit_kernel(numba, _fill_nan)
    _slerp_sample = njit_kernel(numba, _slerp_sample)


def _compile_slerp_dependencies(numba):
    normalize = getattr(_normalize_quat_tuple, "py_func", _normalize_quat_tuple)
    multiply = getattr(_quat_multiply, "py_func", _quat_multiply)
    normalize_compiled = njit_kernel(numba, normalize)
    multiply_compiled = njit_kernel(numba, multiply)
    _quat_interp_primitives.normalize_quat_tuple = normalize_compiled
    slerp = getattr(_slerp_quat, "py_func", _slerp_quat)
    return normalize_compiled, multiply_compiled, njit_kernel(numba, slerp)


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
    validate_slerp_shapes(q0_block, q1_block, alpha_block, valid_block, owner=owner)
    tolerance = half_turn_tolerance((q0_block.dtype, q1_block.dtype))
    out = np.empty(q0_block.shape, dtype=np.float64)
    output_rows = out.reshape(-1, _QUAT_SIZE)
    for block in iter_slerp_blocks(q0_block, q1_block, alpha_block, valid_block):
        output_rows[block.start : block.start + block.alpha.size] = _evaluate_slerp_block(
            block, tolerance=tolerance, owner=owner,
        )
        del block
    return out


def _evaluate_slerp_block(block: SlerpBlock, *, tolerance: float, owner: str) -> np.ndarray:
    left = block.left.astype(np.float64, copy=False)
    right = block.right.astype(np.float64, copy=False)
    alpha = block.alpha.astype(np.float64, copy=False)
    valid = block.valid.astype(bool, copy=False)
    out, status, ambiguous = _compiled_slerp_block()(left, right, alpha, valid, tolerance, False)
    _raise_slerp_status(int(status), owner=owner)
    if np.any(ambiguous):
        out[ambiguous] = _evaluate_principal_arcs(left[ambiguous], right[ambiguous], alpha[ambiguous], owner=owner)
    return out


def _evaluate_principal_arcs(left: np.ndarray, right: np.ndarray, alpha: np.ndarray, *, owner: str) -> np.ndarray:
    prepared = prepare_numba_slerp_endpoints(left, right, owner=owner)
    valid = np.ones(alpha.size, dtype=bool)
    out, status, _ = _compiled_slerp_block()(left, prepared, alpha, valid, -1.0, True)
    _raise_slerp_status(int(status), owner=owner)
    return out


def _fill_nan(out):
    out.fill(np.nan)


def _slerp_sample(q0_rows, q1_rows, alpha_rows, idx, out, half_turn_tolerance, principal_arc):
    t = alpha_rows[idx]
    if not np.isfinite(t):
        return _STATUS_OK
    if t < 0.0 or t > 1.0:
        return _STATUS_ALPHA_RANGE
    status, result = _shared_slerp_quat(q0_rows[idx], q1_rows[idx], t, half_turn_tolerance, principal_arc)
    if status != _STATUS_OK:
        return status
    for component in range(_QUAT_SIZE):
        out[idx, component] = result[component]
    return status


def _slerp_block_impl(q0_rows, q1_rows, alpha_rows, valid_rows, half_turn_tolerance, principal_arc):
    out = np.empty_like(q0_rows)
    _fill_nan(out)
    ambiguous = np.zeros(alpha_rows.size, dtype=np.bool_)
    for idx in range(alpha_rows.size):
        if not valid_rows[idx]:
            continue
        status = _slerp_sample(q0_rows, q1_rows, alpha_rows, idx, out, half_turn_tolerance, principal_arc)
        if status == _STATUS_AMBIGUOUS_HALF_TURN:
            ambiguous[idx] = True
        elif status != _STATUS_OK:
            return out, status, ambiguous
    return out, _STATUS_OK, ambiguous


__all__ = ["slerp_quat_numba"]
