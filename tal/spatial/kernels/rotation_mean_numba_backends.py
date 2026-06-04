from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_support import njit_kernel, require_numba

from .rotation_mean_backends import prepare_quat_mean_rows

_QUAT_SIZE = 4
_HELPERS_JITTED = False


@lru_cache(maxsize=1)
def _compiled_quat_mean_block():
    numba = require_numba("spatial.rotation.mean_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _quat_mean_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _HELPERS_JITTED
    global _add_weighted_outer, _largest_eigenvector, _valid_sample, _write_row_mean
    if _HELPERS_JITTED:
        return
    _valid_sample = njit_kernel(numba, _valid_sample)
    _add_weighted_outer = njit_kernel(numba, _add_weighted_outer)
    _largest_eigenvector = njit_kernel(numba, _largest_eigenvector)
    _write_row_mean = njit_kernel(numba, _write_row_mean)
    _HELPERS_JITTED = True


def quat_mean_block_numba(values: object, weights: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    rows = prepare_quat_mean_rows(values, weights, owner=owner)
    out = _compiled_quat_mean_block()(*rows.row_arrays)
    return out.reshape(rows.output_shape)


def _valid_sample(values, weights, row, idx):
    weight = weights[row, idx]
    if not np.isfinite(weight):
        return False
    for comp in range(_QUAT_SIZE):
        if not np.isfinite(values[row, idx, comp]):
            return False
    return True


def _add_weighted_outer(gram, quat, weight):
    for left in range(_QUAT_SIZE):
        for right in range(_QUAT_SIZE):
            gram[left, right] += quat[left] * quat[right] * weight


def _largest_eigenvector(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    best = 0
    for idx in range(1, _QUAT_SIZE):
        if eigvals[idx] > eigvals[best]:
            best = idx
    return eigvecs[:, best]


def _write_row_mean(values, weights, row, out):
    gram = np.zeros((_QUAT_SIZE, _QUAT_SIZE), dtype=np.float64)
    total = 0.0
    for idx in range(values.shape[1]):
        if not _valid_sample(values, weights, row, idx):
            continue
        weight = weights[row, idx]
        quat = values[row, idx]
        total += weight
        _add_weighted_outer(gram, quat, weight)
    if not np.isfinite(total) or total <= 0.0:
        return
    quat = _largest_eigenvector(gram)
    norm = 0.0
    for comp in range(_QUAT_SIZE):
        norm += quat[comp] * quat[comp]
    if not np.isfinite(norm) or norm <= 0.0:
        return
    scale = 1.0 / np.sqrt(norm)
    if quat[3] < 0.0:
        scale = -scale
    for comp in range(_QUAT_SIZE):
        out[row, comp] = quat[comp] * scale


def _quat_mean_block_impl(values, weights):
    out = np.empty((values.shape[0], _QUAT_SIZE), dtype=np.float64)
    out.fill(np.nan)
    for row in range(values.shape[0]):
        _write_row_mean(values, weights, row, out)
    return out


__all__ = ["quat_mean_block_numba"]
