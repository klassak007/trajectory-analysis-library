from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows

from ._fixed_size_constants import STATUS_INVALID_QUAT, STATUS_OK
from .higher_order_interp_primitives import catmull_rom_vec3, squad_quat

ROTATION_HIGHER_ORDER_BACKEND_NUMPY = "numpy"
ROTATION_HIGHER_ORDER_BACKEND_NUMBA = "numba"
POSE_HIGHER_ORDER_BACKEND_NUMPY = "numpy"
POSE_HIGHER_ORDER_BACKEND_NUMBA = "numba"

_ROTATION_OWNER = "spatial.rotation.higher_order_interp_backend"
_POSE_OWNER = "spatial.pose.higher_order_interp_backend"
_STATUS_ALPHA_RANGE = 2
_QUAT_SIZE = 4
_VEC3_SIZE = 3


@dataclass(frozen=True)
class QuatInterpWindow:
    q_prev: object
    q0: object
    q1: object
    q_next: object


@dataclass(frozen=True)
class PoseInterpWindow:
    t_prev: object
    t0: object
    t1: object
    t_next: object
    q_prev: object
    q0: object
    q1: object
    q_next: object


@dataclass(frozen=True)
class _PreparedRows:
    row_arrays: tuple[np.ndarray, ...]
    output_shape: tuple[int, ...]


def _shape(value: object) -> tuple[int, ...]:
    return tuple(int(size) for size in np.shape(value))


def _require_tail(shape: tuple[int, ...], tail: tuple[int, ...], *, name: str, owner: str) -> None:
    if len(shape) < len(tail) or shape[-len(tail) :] != tail:
        raise ValueError(f"{owner}: {name} must have trailing shape {tail}.")


def _validate_alpha_valid(shape: tuple[int, ...], alpha: object, valid: object, *, owner: str) -> None:
    if np.shape(alpha) != np.shape(valid):
        raise ValueError(f"{owner}: alpha and valid must share shape.")
    if shape[:-1] != np.shape(alpha):
        raise ValueError(f"{owner}: alpha/valid must match window non-core shape.")


def _validate_quat_window_shapes(window: QuatInterpWindow, alpha: object, valid: object, *, owner: str) -> None:
    shapes = tuple(_shape(value) for value in (window.q_prev, window.q0, window.q1, window.q_next))
    for name, shape in zip(("q_prev", "q0", "q1", "q_next"), shapes, strict=True):
        _require_tail(shape, (_QUAT_SIZE,), name=name, owner=owner)
    if len(set(shapes)) != 1:
        raise ValueError(f"{owner}: quaternion window shapes must match exactly.")
    _validate_alpha_valid(shapes[0], alpha, valid, owner=owner)


def _validate_pose_window_shapes(window: PoseInterpWindow, alpha: object, valid: object, *, owner: str) -> None:
    t_shapes = tuple(_shape(value) for value in (window.t_prev, window.t0, window.t1, window.t_next))
    q_shapes = tuple(_shape(value) for value in (window.q_prev, window.q0, window.q1, window.q_next))
    for name, shape in zip(("t_prev", "t0", "t1", "t_next"), t_shapes, strict=True):
        _require_tail(shape, (_VEC3_SIZE,), name=name, owner=owner)
    for name, shape in zip(("q_prev", "q0", "q1", "q_next"), q_shapes, strict=True):
        _require_tail(shape, (_QUAT_SIZE,), name=name, owner=owner)
    if len(set(t_shapes)) != 1:
        raise ValueError(f"{owner}: translation window shapes must match exactly.")
    if len(set(q_shapes)) != 1:
        raise ValueError(f"{owner}: quaternion window shapes must match exactly.")
    if t_shapes[0][:-1] != q_shapes[0][:-1]:
        raise ValueError(f"{owner}: translation and quaternion non-core shapes must match exactly.")
    _validate_alpha_valid(q_shapes[0], alpha, valid, owner=owner)


def prepare_squad_rows(window: QuatInterpWindow, alpha: object, valid: object, *, owner: str) -> _PreparedRows:
    _validate_quat_window_shapes(window, alpha, valid, owner=owner)
    prepared = prepare_block_rows(
        (window.q_prev, window.q0, window.q1, window.q_next, alpha, valid),
        (
            BlockInputSpec("q_prev", 2, np.float64),
            BlockInputSpec("q0", 2, np.float64),
            BlockInputSpec("q1", 2, np.float64),
            BlockInputSpec("q_next", 2, np.float64),
            BlockInputSpec("alpha", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
        ),
        output_core_shape=(),
        owner=owner,
    )
    query = int(prepared.row_arrays[0].shape[-2])
    return _PreparedRows(prepared.row_arrays, prepared.outer_shape + (query, _QUAT_SIZE))


def prepare_pose_cubic_rows(window: PoseInterpWindow, alpha: object, valid: object, *, owner: str) -> _PreparedRows:
    _validate_pose_window_shapes(window, alpha, valid, owner=owner)
    blocks = (
        window.t_prev,
        window.t0,
        window.t1,
        window.t_next,
        window.q_prev,
        window.q0,
        window.q1,
        window.q_next,
        alpha,
        valid,
    )
    specs = (
        BlockInputSpec("t_prev", 2, np.float64),
        BlockInputSpec("t0", 2, np.float64),
        BlockInputSpec("t1", 2, np.float64),
        BlockInputSpec("t_next", 2, np.float64),
        BlockInputSpec("q_prev", 2, np.float64),
        BlockInputSpec("q0", 2, np.float64),
        BlockInputSpec("q1", 2, np.float64),
        BlockInputSpec("q_next", 2, np.float64),
        BlockInputSpec("alpha", 1, np.float64),
        BlockInputSpec("valid", 1, bool),
    )
    prepared = prepare_block_rows(blocks, specs, output_core_shape=(), owner=owner)
    query = int(prepared.row_arrays[0].shape[-2])
    return _PreparedRows(prepared.row_arrays, prepared.outer_shape + (query,))


def _raise_status(status: int, *, owner: str) -> None:
    if status == STATUS_INVALID_QUAT:
        raise ValueError(f"{owner}: quaternion norm must be finite and > 0.")
    if status == _STATUS_ALPHA_RANGE:
        raise ValueError(f"{owner}: finite alpha values must be within [0, 1].")


def _squad_numpy(rows: _PreparedRows, *, owner: str) -> np.ndarray:
    q_prev, q0, q1, q_next, alpha, valid = rows.row_arrays
    out = np.full(q_prev.shape, np.nan, dtype=np.float64)
    for row in range(q_prev.shape[0]):
        for idx in range(q_prev.shape[1]):
            if not valid[row, idx]:
                continue
            t = float(alpha[row, idx])
            if not np.isfinite(t) or t < 0.0 or t > 1.0:
                _raise_status(_STATUS_ALPHA_RANGE, owner=owner)
            status, quat = squad_quat(q_prev[row, idx], q0[row, idx], q1[row, idx], q_next[row, idx], t)
            _raise_status(status, owner=owner)
            out[row, idx] = quat
    return out.reshape(rows.output_shape)


def _pose_numpy(rows: _PreparedRows, *, owner: str) -> tuple[np.ndarray, np.ndarray]:
    t_prev, t0, t1, t_next, q_prev, q0, q1, q_next, alpha, valid = rows.row_arrays
    out_t = np.full(t_prev.shape, np.nan, dtype=np.float64)
    out_q = np.full(q_prev.shape, np.nan, dtype=np.float64)
    for row in range(t_prev.shape[0]):
        for idx in range(t_prev.shape[1]):
            if not valid[row, idx]:
                continue
            t = float(alpha[row, idx])
            if not np.isfinite(t) or t < 0.0 or t > 1.0:
                _raise_status(_STATUS_ALPHA_RANGE, owner=owner)
            status, quat = squad_quat(q_prev[row, idx], q0[row, idx], q1[row, idx], q_next[row, idx], t)
            _raise_status(status, owner=owner)
            out_t[row, idx] = catmull_rom_vec3(t_prev[row, idx], t0[row, idx], t1[row, idx], t_next[row, idx], t)
            out_q[row, idx] = quat
    return out_t.reshape(rows.output_shape + (_VEC3_SIZE,)), out_q.reshape(rows.output_shape + (_QUAT_SIZE,))


def squad_quat_block_backend(
    window: QuatInterpWindow,
    alpha: object,
    valid: object,
    *,
    backend: str = ROTATION_HIGHER_ORDER_BACKEND_NUMPY,
) -> np.ndarray:
    if backend == ROTATION_HIGHER_ORDER_BACKEND_NUMPY:
        return _squad_numpy(prepare_squad_rows(window, alpha, valid, owner=_ROTATION_OWNER), owner=_ROTATION_OWNER)
    if backend == ROTATION_HIGHER_ORDER_BACKEND_NUMBA:
        from .higher_order_interp_numba_backends import squad_quat_block_numba

        return squad_quat_block_numba(window, alpha, valid, owner=_ROTATION_OWNER)
    raise ValueError(f"{_ROTATION_OWNER}: unsupported higher-order rotation backend {backend!r}.")


def pose_cubic_squad_block_backend(
    window: PoseInterpWindow,
    alpha: object,
    valid: object,
    *,
    backend: str = POSE_HIGHER_ORDER_BACKEND_NUMPY,
) -> tuple[np.ndarray, np.ndarray]:
    if backend == POSE_HIGHER_ORDER_BACKEND_NUMPY:
        return _pose_numpy(prepare_pose_cubic_rows(window, alpha, valid, owner=_POSE_OWNER), owner=_POSE_OWNER)
    if backend == POSE_HIGHER_ORDER_BACKEND_NUMBA:
        from .higher_order_interp_numba_backends import pose_cubic_squad_block_numba

        return pose_cubic_squad_block_numba(window, alpha, valid, owner=_POSE_OWNER)
    raise ValueError(f"{_POSE_OWNER}: unsupported higher-order pose backend {backend!r}.")


__all__ = [
    "POSE_HIGHER_ORDER_BACKEND_NUMBA",
    "POSE_HIGHER_ORDER_BACKEND_NUMPY",
    "ROTATION_HIGHER_ORDER_BACKEND_NUMBA",
    "ROTATION_HIGHER_ORDER_BACKEND_NUMPY",
    "PoseInterpWindow",
    "QuatInterpWindow",
    "pose_cubic_squad_block_backend",
    "prepare_pose_cubic_rows",
    "prepare_squad_rows",
    "squad_quat_block_backend",
]
