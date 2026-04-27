from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

ROTATION_INTERP_BACKEND_SCIPY = "scipy"


def _normalize_quaternion(values: np.ndarray, *, owner: str) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{owner}: quaternion norm must be finite and > 0.")
    return np.asarray(values, dtype=np.float64) / norm


def _slerp_pair_scipy(q0: np.ndarray, q1: np.ndarray, t: float, *, owner: str) -> np.ndarray:
    qa = _normalize_quaternion(q0, owner=owner)
    qb = _normalize_quaternion(q1, owner=owner)
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    rots = SciRotation.from_quat(np.stack([qa, qb], axis=0))
    interp = Slerp(np.array([0.0, 1.0], dtype=np.float64), rots)
    return interp(np.array([t], dtype=np.float64)).as_quat()[0]


def _slerp_quat_scipy_stopgap(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """SciPy SLERP stopgap: row-local loops are isolated to backend owner only."""
    owner = "spatial.rotation.interp_backend"
    if q0.shape != q1.shape:
        raise ValueError(f"{owner}: q0 and q1 must share shape.")
    if q0.shape[-1] != 4:
        raise ValueError(f"{owner}: expected trailing quaternion dim length 4.")
    if alpha.shape != valid.shape:
        raise ValueError(f"{owner}: alpha and valid must share shape.")
    if q0.shape[:-1] != alpha.shape:
        raise ValueError(f"{owner}: alpha/valid must match q0/q1 non-core shape.")
    rows = int(np.prod(alpha.shape[:-1])) if alpha.ndim > 1 else 1
    qsize = int(alpha.shape[-1])
    flat_q0 = q0.reshape(rows, qsize, 4)
    flat_q1 = q1.reshape(rows, qsize, 4)
    flat_alpha = alpha.reshape(rows, qsize)
    flat_valid = valid.reshape(rows, qsize)
    out = np.full_like(flat_q0, np.nan, dtype=np.float64)
    for row in range(rows):
        for idx in range(qsize):
            if not bool(flat_valid[row, idx]):
                continue
            t = float(flat_alpha[row, idx])
            if not np.isfinite(t):
                continue
            out[row, idx] = _slerp_pair_scipy(flat_q0[row, idx], flat_q1[row, idx], t, owner=owner)
    return out.reshape(q0.shape)


def slerp_quat_backend(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
    *,
    backend: str = ROTATION_INTERP_BACKEND_SCIPY,
) -> np.ndarray:
    if backend != ROTATION_INTERP_BACKEND_SCIPY:
        raise ValueError(f"spatial.rotation.interp_backend: unsupported backend {backend!r}.")
    return _slerp_quat_scipy_stopgap(q0, q1, alpha, valid)


__all__ = ["ROTATION_INTERP_BACKEND_SCIPY", "slerp_quat_backend"]
