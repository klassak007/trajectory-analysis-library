from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_support import njit_kernel, require_numba

from . import fixed_size_primitives as _fixed_primitives
from . import higher_order_interp_primitives as _interp_primitives
from ._fixed_size_constants import STATUS_INVALID_QUAT, STATUS_OK
from .higher_order_interp_backends import (
    PoseInterpWindow,
    QuatInterpWindow,
    prepare_pose_cubic_rows,
    prepare_squad_rows,
)

_STATUS_ALPHA_RANGE = 2
_HELPERS_JITTED = False


@lru_cache(maxsize=1)
def _compiled_squad_block():
    numba = require_numba("spatial.rotation.higher_order_interp_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _squad_block_impl)


@lru_cache(maxsize=1)
def _compiled_pose_block():
    numba = require_numba("spatial.pose.higher_order_interp_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _pose_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _HELPERS_JITTED
    global _squad_sample, _pose_sample
    if _HELPERS_JITTED:
        return
    _jit_primitive_helpers(numba)
    _squad_sample = njit_kernel(numba, _squad_sample)
    _pose_sample = njit_kernel(numba, _pose_sample)
    _HELPERS_JITTED = True


def _jit_primitive_helpers(numba) -> None:
    _interp_primitives.normalize_quat_tuple = njit_kernel(numba, _fixed_primitives.normalize_quat_tuple)
    _interp_primitives.quat_multiply = njit_kernel(numba, _fixed_primitives.quat_multiply)
    for name in (
        "_dot_quat",
        "_flip_quat",
        "_inverse_unit_quat",
        "_normalize_tuple_or_status",
        "_canonical_next",
        "canonical_quat_window",
        "quat_log_unit",
        "quat_exp_vector",
        "_quat_tangent",
        "_normalize_quat_result",
        "slerp_unit",
        "squad_quat",
        "catmull_rom_vec3",
    ):
        setattr(_interp_primitives, name, njit_kernel(numba, getattr(_interp_primitives, name)))


def _raise_status(status: int, *, owner: str) -> None:
    if status == STATUS_INVALID_QUAT:
        raise ValueError(f"{owner}: quaternion norm must be finite and > 0.")
    if status == _STATUS_ALPHA_RANGE:
        raise ValueError(f"{owner}: finite alpha values must be within [0, 1].")


def squad_quat_block_numba(window: QuatInterpWindow, alpha: object, valid: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    rows = prepare_squad_rows(window, alpha, valid, owner=owner)
    out, status = _compiled_squad_block()(*rows.row_arrays)
    _raise_status(int(status), owner=owner)
    return out.reshape(rows.output_shape)


def pose_cubic_squad_block_numba(
    window: PoseInterpWindow,
    alpha: object,
    valid: object,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray]:
    require_numba(owner)
    rows = prepare_pose_cubic_rows(window, alpha, valid, owner=owner)
    out_t, out_q, status = _compiled_pose_block()(*rows.row_arrays)
    _raise_status(int(status), owner=owner)
    return out_t.reshape(rows.output_shape + (3,)), out_q.reshape(rows.output_shape + (4,))


def _squad_sample(q_rows, alpha_rows, row, idx, out):
    t = alpha_rows[row, idx]
    if not np.isfinite(t) or t < 0.0 or t > 1.0:
        return _STATUS_ALPHA_RANGE
    status, quat = _interp_primitives.squad_quat(
        q_rows[0][row, idx],
        q_rows[1][row, idx],
        q_rows[2][row, idx],
        q_rows[3][row, idx],
        t,
    )
    if status != STATUS_OK:
        return status
    for comp in range(4):
        out[row, idx, comp] = quat[comp]
    return STATUS_OK


def _pose_sample(t_rows, q_rows, alpha_rows, row, idx, out_t, out_q):
    status = _squad_sample(q_rows, alpha_rows, row, idx, out_q)
    if status != STATUS_OK:
        return status
    interp = _interp_primitives.catmull_rom_vec3(
        t_rows[0][row, idx],
        t_rows[1][row, idx],
        t_rows[2][row, idx],
        t_rows[3][row, idx],
        alpha_rows[row, idx],
    )
    for comp in range(3):
        out_t[row, idx, comp] = interp[comp]
    return STATUS_OK


def _squad_block_impl(q_prev, q0, q1, q_next, alpha, valid):
    out = np.empty_like(q_prev)
    out.fill(np.nan)
    q_rows = (q_prev, q0, q1, q_next)
    query = q_prev.shape[1]
    for flat_idx in range(q_prev.shape[0] * query):
        row = flat_idx // query
        idx = flat_idx - row * query
        if not valid[row, idx]:
            continue
        status = _squad_sample(q_rows, alpha, row, idx, out)
        if status != STATUS_OK:
            return out, status
    return out, STATUS_OK


def _pose_block_impl(t_prev, t0, t1, t_next, q_prev, q0, q1, q_next, alpha, valid):
    out_t = np.empty_like(t_prev)
    out_q = np.empty_like(q_prev)
    out_t.fill(np.nan)
    out_q.fill(np.nan)
    t_rows = (t_prev, t0, t1, t_next)
    q_rows = (q_prev, q0, q1, q_next)
    query = t_prev.shape[1]
    for flat_idx in range(t_prev.shape[0] * query):
        row = flat_idx // query
        idx = flat_idx - row * query
        if not valid[row, idx]:
            continue
        status = _pose_sample(t_rows, q_rows, alpha, row, idx, out_t, out_q)
        if status != STATUS_OK:
            return out_t, out_q, status
    return out_t, out_q, STATUS_OK


__all__ = ["pose_cubic_squad_block_numba", "squad_quat_block_numba"]
