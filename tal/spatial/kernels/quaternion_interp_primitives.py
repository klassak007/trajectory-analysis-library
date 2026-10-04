from __future__ import annotations

import numpy as np

from ._fixed_size_constants import STATUS_OK
from .fixed_size_primitives import normalize_quat_tuple

_NLERP_DOT_THRESHOLD = 1.0 - 1.0e-12
_STATUS_AMBIGUOUS_HALF_TURN = 3


def slerp_quat(q0, q1, alpha: float, half_turn_tolerance: float = -1.0, principal_arc: bool = False):
    """Return status and exact shortest-arc SLERP for two quaternions."""
    status, x0, y0, z0, w0 = normalize_quat_tuple(q0)
    if status != STATUS_OK:
        return status, (np.nan, np.nan, np.nan, np.nan)
    status, x1, y1, z1, w1 = normalize_quat_tuple(q1)
    if status != STATUS_OK:
        return status, (np.nan, np.nan, np.nan, np.nan)
    left = (x0, y0, z0, w0)
    right = (x1, y1, z1, w1)
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if not principal_arc and half_turn_tolerance >= 0.0 and abs(dot) <= half_turn_tolerance:
        return _STATUS_AMBIGUOUS_HALF_TURN, right
    if not principal_arc and dot < 0.0:
        dot = -dot
        right = (-x1, -y1, -z1, -w1)
    if alpha == 0.0:
        return STATUS_OK, left
    if alpha == 1.0:
        return STATUS_OK, right
    dot = min(dot, 1.0)
    if dot > _NLERP_DOT_THRESHOLD:
        candidate = (
            left[0] + alpha * (right[0] - left[0]),
            left[1] + alpha * (right[1] - left[1]),
            left[2] + alpha * (right[2] - left[2]),
            left[3] + alpha * (right[3] - left[3]),
        )
    else:
        angle = np.arccos(dot)
        denominator = np.sin(angle)
        left_scale = np.sin((1.0 - alpha) * angle) / denominator
        right_scale = np.sin(alpha * angle) / denominator
        candidate = (
            left_scale * left[0] + right_scale * right[0],
            left_scale * left[1] + right_scale * right[1],
            left_scale * left[2] + right_scale * right[2],
            left_scale * left[3] + right_scale * right[3],
        )
    status, x, y, z, w = normalize_quat_tuple(candidate)
    return status, (x, y, z, w)


__all__ = ["slerp_quat"]
