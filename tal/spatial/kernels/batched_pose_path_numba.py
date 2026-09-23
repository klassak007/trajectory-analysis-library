from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_support import njit_kernel, require_numba

from . import fixed_size_primitives as _fixed
from .path_kernel_status import (
    PATH_STATUS_INVALID_ALPHA,
    PATH_STATUS_INVALID_BRACKET,
    PATH_STATUS_INVALID_DIRECTION,
    PATH_STATUS_OK,
)
from .rotation_interp_numba_backends import _compile_slerp_dependencies

_AMBIGUOUS_ARC = 3
_PARALLEL_MIN_ROWS = 32_768


def _sample_edge(translation, quaternion, source, row, pos_map, rot_map, tolerance):
    pos_i0, pos_i1, pos_alpha = pos_map
    rot_i0, rot_i1, rot_alpha = rot_map
    pt = pos_alpha[row]
    rt = rot_alpha[row]
    if not np.isfinite(pt) or pt < 0.0 or pt > 1.0:
        return PATH_STATUS_INVALID_ALPHA, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), False
    if not np.isfinite(rt) or rt < 0.0 or rt > 1.0:
        return PATH_STATUS_INVALID_ALPHA, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), False
    pi0, pi1 = pos_i0[row], pos_i1[row]
    ri0, ri1 = rot_i0[row], rot_i1[row]
    if pi0 < 0 or pi1 < 0 or pi0 >= translation.shape[1] or pi1 >= translation.shape[1]:
        return PATH_STATUS_INVALID_BRACKET, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), False
    if ri0 < 0 or ri1 < 0 or ri0 >= quaternion.shape[1] or ri1 >= quaternion.shape[1]:
        return PATH_STATUS_INVALID_BRACKET, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), False
    sampled_t = (
        (1.0 - pt) * translation[source, pi0, 0] + pt * translation[source, pi1, 0],
        (1.0 - pt) * translation[source, pi0, 1] + pt * translation[source, pi1, 1],
        (1.0 - pt) * translation[source, pi0, 2] + pt * translation[source, pi1, 2],
    )
    left_q = (
        quaternion[source, ri0, 0], quaternion[source, ri0, 1],
        quaternion[source, ri0, 2], quaternion[source, ri0, 3],
    )
    right_q = (
        quaternion[source, ri1, 0], quaternion[source, ri1, 1],
        quaternion[source, ri1, 2], quaternion[source, ri1, 3],
    )
    if left_q == right_q:
        normalized = _shared_normalize(left_q)
        status = normalized[0]
        sampled_q = normalized[1:]
    else:
        status, sampled_q = _shared_slerp(left_q, right_q, rt, tolerance, False)
    ambiguous = status == _AMBIGUOUS_ARC
    if ambiguous:
        status, sampled_q = _shared_slerp(left_q, right_q, rt, -1.0, False)
    return status, sampled_t, sampled_q, ambiguous


def _edge_row(
    translation,
    quaternion,
    batch_index,
    pos_mapping,
    rot_mapping,
    direction,
    tolerance,
    output,
    first_edge,
    row,
):
    pos_i0, pos_i1, pos_alpha, pos_valid = pos_mapping
    rot_i0, rot_i1, rot_alpha, rot_valid = rot_mapping
    accumulated_t, accumulated_q, accumulated_valid = output
    query_count = pos_alpha.size // batch_index.size
    if not pos_valid[row] or not rot_valid[row] or (not first_edge and not accumulated_valid[row]):
        accumulated_valid[row] = False
        return PATH_STATUS_OK, False
    source = batch_index[row // query_count]
    pos_map = (pos_i0, pos_i1, pos_alpha)
    rot_map = (rot_i0, rot_i1, rot_alpha)
    status, edge_t, edge_q, ambiguous = _compiled_sample_edge(
        translation, quaternion, source, row, pos_map, rot_map, tolerance,
    )
    if status != PATH_STATUS_OK:
        return status, ambiguous
    if direction == -1:
        edge_t, edge_q = _compiled_inverse(edge_t, edge_q)
    elif direction != 1:
        return PATH_STATUS_INVALID_DIRECTION, ambiguous
    if first_edge:
        accumulated_t[row], accumulated_q[row] = edge_t, edge_q
    else:
        accumulated_t[row], accumulated_q[row] = _compiled_compose(
            accumulated_t[row], accumulated_q[row], edge_t, edge_q,
        )
    accumulated_valid[row] = True
    return PATH_STATUS_OK, ambiguous


def _edge_block_serial_impl(
    translation, quaternion, batch_index, pos_mapping, rot_mapping,
    direction, tolerance, output, first_edge,
):
    rows = pos_mapping[2].size
    any_ambiguous = False
    for row in range(rows):
        status, ambiguous = _compiled_edge_row(
            translation, quaternion, batch_index, pos_mapping, rot_mapping,
            direction, tolerance, output, first_edge, row,
        )
        if status != PATH_STATUS_OK:
            return status, row, ambiguous
        any_ambiguous = any_ambiguous or ambiguous
    return PATH_STATUS_OK, -1, any_ambiguous


def _edge_block_parallel_impl(
    translation, quaternion, batch_index, pos_mapping, rot_mapping,
    direction, tolerance, output, first_edge,
):
    rows = pos_mapping[2].size
    status = np.zeros(rows, dtype=np.int8)
    ambiguous = np.zeros(rows, dtype=np.bool_)
    for row in _parallel_range(rows):
        status[row], ambiguous[row] = _compiled_edge_row(
            translation, quaternion, batch_index, pos_mapping, rot_mapping,
            direction, tolerance, output, first_edge, row,
        )
    return status, ambiguous


def _apply_position_impl(values, pose_t, pose_q, pose_valid, caller_valid, output):
    for row in range(pose_valid.size):
        if not pose_valid[row] or not caller_valid[row]:
            continue
        vector = (values[row, 0], values[row, 1], values[row, 2])
        quaternion = pose_q[row]
        if (
            quaternion[0] == 0.0
            and quaternion[1] == 0.0
            and quaternion[2] == 0.0
            and quaternion[3] == 1.0
        ):
            rotated = vector
        else:
            rotated = _compiled_rotate(quaternion, vector)
        for axis in range(3):
            output[row, axis] = rotated[axis] + pose_t[row, axis]


@lru_cache(maxsize=1)
def _compiled_kernels():
    numba = require_numba("spatial.path_solve.pose")
    global _shared_normalize, _shared_slerp, _compiled_sample_edge, _compiled_edge_row
    global _compiled_inverse, _compiled_compose, _compiled_rotate, _parallel_range
    _shared_normalize, _, _shared_slerp = _compile_slerp_dependencies(numba)
    _compiled_inverse = njit_kernel(numba, _fixed.inverse_pose)
    _compiled_compose = njit_kernel(numba, _fixed.compose_pose)
    _compiled_rotate = njit_kernel(numba, _fixed.rotate_vec3)
    _compiled_sample_edge = njit_kernel(numba, _sample_edge)
    _compiled_edge_row = njit_kernel(numba, _edge_row)
    _parallel_range = numba.prange
    serial = njit_kernel(numba, _edge_block_serial_impl)
    parallel = numba.njit(cache=True, fastmath=False, parallel=True)(_edge_block_parallel_impl)
    return serial, parallel, njit_kernel(numba, _apply_position_impl)


def batched_pose_edge_numba(*args):
    """Update one bounded Pose block and return its compact status."""
    serial, parallel, _ = _compiled_kernels()
    rows = args[3][2].size
    if rows < _PARALLEL_MIN_ROWS:
        return serial(*args)
    status, ambiguous = parallel(*args)
    failures = np.flatnonzero(status)
    row = int(failures[0]) if failures.size else -1
    code = int(status[row]) if row >= 0 else PATH_STATUS_OK
    return code, row, bool(np.any(ambiguous))


def batched_position_apply_numba(*args) -> None:
    """Apply one bounded completed Pose block directly to Position rows."""
    _, _, apply_kernel = _compiled_kernels()
    apply_kernel(*args)


__all__ = ["batched_pose_edge_numba", "batched_position_apply_numba"]
