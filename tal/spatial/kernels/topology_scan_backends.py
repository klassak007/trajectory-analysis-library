from __future__ import annotations

import numpy as np

from ._topology_scan_common import prepare_chain_pose_rows, raise_chain_pose_status
from ._topology_scan_constants import (
    QUAT_SIZE,
    STATUS_EMPTY,
    STATUS_INVALID_DIRECTION,
    STATUS_INVALID_QUAT,
    STATUS_LEFT_PACKED,
    STATUS_OK,
)

SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY = "numpy"
SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA = "numba"


def _count_valid(valid_row: np.ndarray) -> int:
    count = 0
    seen_false = False
    for idx in range(valid_row.shape[0]):
        if not valid_row[idx]:
            seen_false = True
            continue
        if seen_false:
            return -1
        count += 1
    return count


def _normalize_quat(quat: np.ndarray) -> tuple[int, np.ndarray]:
    norm = float(np.sqrt(np.sum(quat * quat)))
    if not np.isfinite(norm) or norm <= 0.0:
        return STATUS_INVALID_QUAT, np.zeros((QUAT_SIZE,), dtype=np.float64)
    return STATUS_OK, quat / norm


def _quat_multiply(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    rx, ry, rz, rw = right
    lx, ly, lz, lw = left
    return np.asarray(
        [
            rw * lx + rx * lw + ry * lz - rz * ly,
            rw * ly - rx * lz + ry * lw + rz * lx,
            rw * lz + rx * ly - ry * lx + rz * lw,
            rw * lw - rx * lx - ry * ly - rz * lz,
        ],
        dtype=np.float64,
    )


def _rotate_vec3(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    qvec = quat[:3]
    uv = np.cross(qvec, vector)
    uuv = np.cross(qvec, uv)
    return vector + 2.0 * (quat[3] * uv + uuv)


def _inverse_pose(translation: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse_quat = np.asarray([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)
    return -_rotate_vec3(inverse_quat, translation), inverse_quat


def _compose_pose(left_t: np.ndarray, left_q: np.ndarray, right_t: np.ndarray, right_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_t = _rotate_vec3(right_q, left_t) + right_t
    status, out_q = _normalize_quat(_quat_multiply(right_q, left_q))
    if status != STATUS_OK:
        return out_t, out_q
    return out_t, out_q


def _load_local(
    translation: np.ndarray,
    quat: np.ndarray,
    direction: np.ndarray,
    row: int,
    idx: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    sign = int(direction[row, idx])
    if sign != 1 and sign != -1:
        return STATUS_INVALID_DIRECTION, np.zeros((3,), dtype=np.float64), np.zeros((4,), dtype=np.float64)
    status, local_q = _normalize_quat(quat[row, idx])
    if status != STATUS_OK:
        return status, np.zeros((3,), dtype=np.float64), local_q
    local_t = translation[row, idx].astype(np.float64, copy=True)
    if sign == -1:
        local_t, local_q = _inverse_pose(local_t, local_q)
    return STATUS_OK, local_t, local_q


def _chain_pose_numpy(rows) -> tuple[np.ndarray, np.ndarray, int]:
    out_t = np.full_like(rows.translation, np.nan, dtype=np.float64)
    out_q = np.full_like(rows.quat, np.nan, dtype=np.float64)
    for row in range(rows.translation.shape[0]):
        count = _count_valid(rows.valid[row])
        if count < 0:
            return out_t, out_q, STATUS_LEFT_PACKED
        if count == 0:
            return out_t, out_q, STATUS_EMPTY
        status, acc_t, acc_q = _load_local(rows.translation, rows.quat, rows.direction, row, 0)
        if status != STATUS_OK:
            return out_t, out_q, status
        out_t[row, 0] = acc_t
        out_q[row, 0] = acc_q
        for idx in range(1, count):
            status, local_t, local_q = _load_local(rows.translation, rows.quat, rows.direction, row, idx)
            if status != STATUS_OK:
                return out_t, out_q, status
            acc_t, acc_q = _compose_pose(acc_t, acc_q, local_t, local_q)
            out_t[row, idx] = acc_t
            out_q[row, idx] = acc_q
    return out_t, out_q, STATUS_OK


def chain_pose_compose_block_backend(
    translation: np.ndarray,
    quat: np.ndarray,
    valid: np.ndarray,
    direction: np.ndarray,
    *,
    backend: str = SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY,
) -> tuple[np.ndarray, np.ndarray]:
    owner = "spatial.topology_scan.pose_backend"
    if backend == SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY:
        rows = prepare_chain_pose_rows(translation, quat, valid, direction, owner=owner)
        out_t, out_q, status = _chain_pose_numpy(rows)
        raise_chain_pose_status(int(status), owner=owner)
        return out_t.reshape(rows.translation_output_shape), out_q.reshape(rows.quat_output_shape)
    if backend == SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA:
        from .topology_scan_numba_backends import chain_pose_compose_block_numba

        return chain_pose_compose_block_numba(translation, quat, valid, direction, owner=owner)
    raise ValueError(f"{owner}: unsupported topology scan backend {backend!r}.")


__all__ = [
    "SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA",
    "SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY",
    "chain_pose_compose_block_backend",
]
