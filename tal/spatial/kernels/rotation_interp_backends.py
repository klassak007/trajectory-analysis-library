from __future__ import annotations

import numpy as np

from tal.utils.numba_support import _numba_available

from .rotation_interp_blocks import SlerpBlock, iter_slerp_blocks, validate_slerp_shapes
from .rotation_interp_reference import invalid_quaternion_rows, scipy_slerp_rows

ROTATION_INTERP_BACKEND_AUTO = "auto"
ROTATION_INTERP_BACKEND_SCIPY = "scipy"
ROTATION_INTERP_BACKEND_NUMBA = "numba"


def _slerp_valid_rows(block: SlerpBlock, *, owner: str) -> np.ndarray:
    alpha = block.alpha.astype(np.float64, copy=False)
    active = block.valid.astype(bool, copy=False) & np.isfinite(alpha)
    alpha_error = active & ((alpha < 0.0) | (alpha > 1.0))
    quat_error = active & (invalid_quaternion_rows(block.left) | invalid_quaternion_rows(block.right))
    failures = np.flatnonzero(alpha_error | quat_error)
    if failures.size and alpha_error[failures[0]]:
        raise ValueError(f"{owner}: finite alpha values must be within [0, 1].")
    if failures.size:
        raise ValueError(f"{owner}: quaternion norm must be finite and > 0.")
    out = np.full(block.left.shape, np.nan, dtype=np.float64)
    if np.any(active):
        out[active] = scipy_slerp_rows(block.left[active], block.right[active], alpha[active], owner=owner)
    return out


def _slerp_quat_scipy_block(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray, valid: np.ndarray) -> np.ndarray:
    owner = "spatial.rotation.interp_backend"
    validate_slerp_shapes(q0, q1, alpha, valid, owner=owner)
    out = np.empty(q0.shape, dtype=np.float64)
    output_rows = out.reshape(-1, 4)
    for block in iter_slerp_blocks(q0, q1, alpha, valid):
        output_rows[block.start : block.start + block.alpha.size] = _slerp_valid_rows(block, owner=owner)
        del block
    return out


def _select_rotation_interp_backend(q0: np.ndarray, q1: np.ndarray) -> str:
    dtypes = (np.asarray(q0).dtype, np.asarray(q1).dtype)
    eligible = all(dtype.kind == "f" and dtype.itemsize in {4, 8} for dtype in dtypes)
    return ROTATION_INTERP_BACKEND_NUMBA if eligible and _numba_available() else ROTATION_INTERP_BACKEND_SCIPY


def slerp_quat_backend(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
    *,
    backend: str = ROTATION_INTERP_BACKEND_AUTO,
) -> np.ndarray:
    owner = "spatial.rotation.interp_backend"
    if backend == ROTATION_INTERP_BACKEND_AUTO:
        backend = _select_rotation_interp_backend(q0, q1)
    if backend == ROTATION_INTERP_BACKEND_SCIPY:
        return _slerp_quat_scipy_block(q0, q1, alpha, valid)
    if backend == ROTATION_INTERP_BACKEND_NUMBA:
        from .rotation_interp_numba_backends import slerp_quat_numba

        return slerp_quat_numba(q0, q1, alpha, valid, owner=owner)
    raise ValueError(f"{owner}: unsupported backend {backend!r}.")


__all__ = [
    "ROTATION_INTERP_BACKEND_AUTO",
    "ROTATION_INTERP_BACKEND_NUMBA",
    "ROTATION_INTERP_BACKEND_SCIPY",
    "slerp_quat_backend",
]
