from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_support import njit_kernel, require_numba

from ._topology_scan_common import prepare_chain_pose_rows, raise_chain_pose_status
from ._topology_scan_constants import (
    STATUS_EMPTY,
    STATUS_INVALID_DIRECTION,
    STATUS_INVALID_QUAT,
    STATUS_LEFT_PACKED,
    STATUS_OK,
)
from .fixed_size_primitives import compose_pose as _compose_pose
from .fixed_size_primitives import inverse_pose as _inverse_pose
from .fixed_size_primitives import normalize_quat_row as _normalize_quat


@lru_cache(maxsize=1)
def _compiled_chain_pose_block():
    numba = require_numba("spatial.topology_scan.pose_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _chain_pose_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _chain_pose_row_impl, _compose_pose, _count_valid, _inverse_pose, _load_local, _normalize_quat
    _count_valid = njit_kernel(numba, _count_valid)
    _normalize_quat = njit_kernel(numba, _normalize_quat)
    _inverse_pose = njit_kernel(numba, _inverse_pose)
    _compose_pose = njit_kernel(numba, _compose_pose)
    _load_local = njit_kernel(numba, _load_local)
    _chain_pose_row_impl = njit_kernel(numba, _chain_pose_row_impl)


def chain_pose_compose_block_numba(
    translation: object,
    quat: object,
    valid: object,
    direction: object,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray]:
    require_numba(owner)
    rows = prepare_chain_pose_rows(translation, quat, valid, direction, owner=owner)
    out_t, out_q, status = _compiled_chain_pose_block()(
        rows.translation,
        rows.quat,
        rows.valid,
        rows.direction,
    )
    raise_chain_pose_status(int(status), owner=owner)
    return out_t.reshape(rows.translation_output_shape), out_q.reshape(rows.quat_output_shape)


def _count_valid(valid_row):
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


def _load_local(translation, quat, direction, row, idx):
    sign = direction[row, idx]
    if sign != 1 and sign != -1:
        return STATUS_INVALID_DIRECTION, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
    status, qx, qy, qz, qw = _normalize_quat(quat, row, idx)
    if status != STATUS_OK:
        return STATUS_INVALID_QUAT, (0.0, 0.0, 0.0), (qx, qy, qz, qw)
    local_t = (translation[row, idx, 0], translation[row, idx, 1], translation[row, idx, 2])
    local_q = (qx, qy, qz, qw)
    if sign == -1:
        local_t, local_q = _inverse_pose(local_t, local_q)
    return STATUS_OK, local_t, local_q


def _chain_pose_row_impl(translation, quat, valid, direction, row, out_t, out_q):
    count = _count_valid(valid[row])
    if count < 0:
        return STATUS_LEFT_PACKED
    if count == 0:
        return STATUS_EMPTY
    status, acc_t, acc_q = _load_local(translation, quat, direction, row, 0)
    if status != STATUS_OK:
        return status
    for idx in range(count):
        if idx > 0:
            status, local_t, local_q = _load_local(translation, quat, direction, row, idx)
            if status != STATUS_OK:
                return status
            acc_t, acc_q = _compose_pose(acc_t, acc_q, local_t, local_q)
        for comp in range(3):
            out_t[row, idx, comp] = acc_t[comp]
        for comp in range(4):
            out_q[row, idx, comp] = acc_q[comp]
    return STATUS_OK


def _chain_pose_block_impl(translation, quat, valid, direction):
    out_t = np.empty_like(translation)
    out_q = np.empty_like(quat)
    out_t.fill(np.nan)
    out_q.fill(np.nan)
    for row in range(translation.shape[0]):
        status = _chain_pose_row_impl(translation, quat, valid, direction, row, out_t, out_q)
        if status != STATUS_OK:
            return out_t, out_q, status
    return out_t, out_q, STATUS_OK


__all__ = ["chain_pose_compose_block_numba"]
