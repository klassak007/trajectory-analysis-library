from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

_HALF_TURN_EPSILON_FACTOR = 32.0


def _quaternion_norms(values: np.ndarray, *, owner: str) -> np.ndarray:
    work = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        norms = np.linalg.norm(work, axis=1)
    if np.any(~np.isfinite(norms) | (norms <= 0.0)):
        raise ValueError(f"{owner}: quaternion norm must be finite and > 0.")
    return norms


def invalid_quaternion_rows(values: np.ndarray) -> np.ndarray:
    """Return rows whose quaternion norm is non-finite or zero."""
    work = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        norms = np.linalg.norm(work, axis=1)
    return ~np.isfinite(norms) | (norms <= 0.0)


def scipy_slerp_rows(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: np.ndarray,
    *,
    owner: str,
) -> np.ndarray:
    left = np.asarray(q0, dtype=np.float64)
    right = np.asarray(q1, dtype=np.float64)
    _quaternion_norms(left, owner=owner)
    _quaternion_norms(right, owner=owner)
    interleaved = np.empty((2 * q0.shape[0], 4), dtype=np.float64)
    interleaved[0::2] = left
    interleaved[1::2] = right
    rotations = SciRotation.from_quat(interleaved)
    left_rotation = rotations[0::2]
    relative = left_rotation.inv() * rotations[1::2]
    scaled = SciRotation.from_rotvec(relative.as_rotvec() * alpha[:, None])
    return (left_rotation * scaled).as_quat()


def half_turn_tolerance(dtypes: tuple[np.dtype, ...]) -> float:
    floating = tuple(dtype for dtype in dtypes if dtype.kind == "f")
    if not floating:
        return _HALF_TURN_EPSILON_FACTOR * np.finfo(np.float64).eps
    return _HALF_TURN_EPSILON_FACTOR * max(np.finfo(dtype).eps for dtype in floating)


def prepare_numba_slerp_endpoints(
    left: np.ndarray,
    right: np.ndarray,
    *,
    owner: str,
) -> np.ndarray:
    """Orient only rows classified as ambiguous by the numerical kernel."""
    principal = scipy_slerp_rows(left, right, np.ones(left.shape[0]), owner=owner)
    flip = np.sum(principal * right, axis=1) < 0.0
    return np.where(flip[:, None], -right, right)


__all__ = [
    "half_turn_tolerance",
    "invalid_quaternion_rows",
    "prepare_numba_slerp_endpoints",
    "scipy_slerp_rows",
]
