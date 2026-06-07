from __future__ import annotations

import numpy as np

from ._fixed_size_constants import STATUS_INVALID_QUAT, STATUS_OK
from .fixed_size_primitives import normalize_quat_tuple, quat_multiply

_LERP_DOT_THRESHOLD = 0.9995
_SMALL_VECTOR_NORM = 1.0e-12


def _dot_quat(left, right) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2] + left[3] * right[3]


def _flip_quat(quat):
    return (-quat[0], -quat[1], -quat[2], -quat[3])


def _inverse_unit_quat(quat):
    return (-quat[0], -quat[1], -quat[2], quat[3])


def _normalize_tuple_or_status(quat):
    status, x, y, z, w = normalize_quat_tuple(quat)
    return status, (x, y, z, w)


def _canonical_next(previous, current):
    if _dot_quat(previous, current) < 0.0:
        return _flip_quat(current)
    return current


def canonical_quat_window(q_prev, q0, q1, q_next):
    status, previous = _normalize_tuple_or_status(q_prev)
    if status != STATUS_OK:
        return status, previous, previous, previous, previous
    status, center0 = _normalize_tuple_or_status(q0)
    if status != STATUS_OK:
        return status, previous, previous, previous, previous
    center0 = _canonical_next(previous, center0)
    status, center1 = _normalize_tuple_or_status(q1)
    if status != STATUS_OK:
        return status, previous, previous, previous, previous
    center1 = _canonical_next(center0, center1)
    status, next_quat = _normalize_tuple_or_status(q_next)
    if status != STATUS_OK:
        return status, previous, previous, previous, previous
    return STATUS_OK, previous, center0, center1, _canonical_next(center1, next_quat)


def quat_log_unit(quat):
    vx, vy, vz, w = quat
    vector_norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    if vector_norm <= _SMALL_VECTOR_NORM:
        return (0.0, 0.0, 0.0)
    scale = np.arctan2(vector_norm, w) / vector_norm
    return (vx * scale, vy * scale, vz * scale)


def quat_exp_vector(vector):
    x, y, z = vector
    norm = np.sqrt(x * x + y * y + z * z)
    if norm <= _SMALL_VECTOR_NORM:
        return (0.0, 0.0, 0.0, 1.0)
    scale = np.sin(norm) / norm
    return (x * scale, y * scale, z * scale, np.cos(norm))


def _quat_tangent(center, previous, next_quat):
    inv_center = _inverse_unit_quat(center)
    prev_delta = quat_log_unit(quat_multiply(inv_center, previous))
    next_delta = quat_log_unit(quat_multiply(inv_center, next_quat))
    tangent = (
        -0.25 * (prev_delta[0] + next_delta[0]),
        -0.25 * (prev_delta[1] + next_delta[1]),
        -0.25 * (prev_delta[2] + next_delta[2]),
    )
    return quat_multiply(center, quat_exp_vector(tangent))


def _normalize_quat_result(quat):
    status, normalized = _normalize_tuple_or_status(quat)
    if status != STATUS_OK:
        return (np.nan, np.nan, np.nan, np.nan)
    return normalized


def slerp_unit(q0, q1, alpha: float):
    right = q1
    dot = _dot_quat(q0, right)
    if dot < 0.0:
        dot = -dot
        right = _flip_quat(right)
    if dot > 1.0:
        dot = 1.0
    if dot > _LERP_DOT_THRESHOLD:
        return _normalize_quat_result(
            (
                q0[0] + alpha * (right[0] - q0[0]),
                q0[1] + alpha * (right[1] - q0[1]),
                q0[2] + alpha * (right[2] - q0[2]),
                q0[3] + alpha * (right[3] - q0[3]),
            )
        )
    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = alpha * theta0
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return _normalize_quat_result(
        (
            s0 * q0[0] + s1 * right[0],
            s0 * q0[1] + s1 * right[1],
            s0 * q0[2] + s1 * right[2],
            s0 * q0[3] + s1 * right[3],
        )
    )


def squad_quat(q_prev, q0, q1, q_next, alpha: float):
    status, previous, center0, center1, next_quat = canonical_quat_window(q_prev, q0, q1, q_next)
    if status != STATUS_OK:
        return status, (np.nan, np.nan, np.nan, np.nan)
    tangent0 = _quat_tangent(center0, previous, center1)
    tangent1 = _quat_tangent(center1, center0, next_quat)
    path = slerp_unit(center0, center1, alpha)
    control = slerp_unit(tangent0, tangent1, alpha)
    return STATUS_OK, slerp_unit(path, control, 2.0 * alpha * (1.0 - alpha))


def catmull_rom_vec3(p0, p1, p2, p3, alpha: float):
    a2 = alpha * alpha
    a3 = a2 * alpha
    return (
        0.5
        * (
            2.0 * p1[0]
            + (-p0[0] + p2[0]) * alpha
            + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * a2
            + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * a3
        ),
        0.5
        * (
            2.0 * p1[1]
            + (-p0[1] + p2[1]) * alpha
            + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * a2
            + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * a3
        ),
        0.5
        * (
            2.0 * p1[2]
            + (-p0[2] + p2[2]) * alpha
            + (2.0 * p0[2] - 5.0 * p1[2] + 4.0 * p2[2] - p3[2]) * a2
            + (-p0[2] + 3.0 * p1[2] - 3.0 * p2[2] + p3[2]) * a3
        ),
    )


__all__ = [
    "canonical_quat_window",
    "catmull_rom_vec3",
    "quat_exp_vector",
    "quat_log_unit",
    "slerp_unit",
    "squad_quat",
]
