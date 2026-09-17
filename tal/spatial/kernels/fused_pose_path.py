from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_support import njit_kernel, require_numba

from . import fixed_size_primitives as _fixed
from .rotation_interp_numba_backends import _compile_slerp_dependencies
from .rotation_interp_reference import half_turn_tolerance
from .streaming_pose_path import stream_pose_path_blocks

_OK = 0
_INVALID_QUATERNION = 1
_INVALID_ALPHA = 2
_INVALID_BRACKET = 3
_INVALID_DIRECTION = 4
_AMBIGUOUS_ARC = 3
_REPAIR_BLOCK_SIZE = 65_536
_PARALLEL_MIN_ROWS = 100_000


def _edge_value(translation, quaternion, edge, left, right, fraction, tolerance):
    sampled_t = (
        (1.0 - fraction) * translation[edge, left, 0] + fraction * translation[edge, right, 0],
        (1.0 - fraction) * translation[edge, left, 1] + fraction * translation[edge, right, 1],
        (1.0 - fraction) * translation[edge, left, 2] + fraction * translation[edge, right, 2],
    )
    left_q = (
        quaternion[edge, left, 0], quaternion[edge, left, 1],
        quaternion[edge, left, 2], quaternion[edge, left, 3],
    )
    right_q = (
        quaternion[edge, right, 0], quaternion[edge, right, 1],
        quaternion[edge, right, 2], quaternion[edge, right, 3],
    )
    status, sampled_q = _shared_slerp(left_q, right_q, fraction, tolerance, False)
    ambiguous = status == _AMBIGUOUS_ARC
    if ambiguous:
        status, sampled_q = _shared_slerp(left_q, right_q, fraction, -1.0, False)
    return status, sampled_t, sampled_q, ambiguous


def _query_value(translation, quaternion, directions, left, right, fraction, tolerance):
    result_t = (0.0, 0.0, 0.0)
    result_q = (0.0, 0.0, 0.0, 1.0)
    ambiguous = False
    for edge in range(translation.shape[0]):
        direction = directions[edge]
        if direction != 1 and direction != -1:
            return result_t, result_q, _INVALID_DIRECTION, edge, ambiguous
        status, edge_t, edge_q, edge_ambiguous = _compiled_edge_value(
            translation, quaternion, edge, left, right, fraction, tolerance,
        )
        ambiguous = ambiguous or edge_ambiguous
        if status != _OK:
            return result_t, result_q, status, edge, ambiguous
        if direction == -1:
            edge_t, edge_q = _compiled_inverse(edge_t, edge_q)
        result_t, result_q = _compiled_compose(result_t, result_q, edge_t, edge_q)
    return result_t, result_q, _OK, -1, ambiguous


def _empty_result(count):
    return (
        np.full((count, 3), np.nan),
        np.full((count, 4), np.nan),
        np.zeros(count, dtype=np.int8),
        np.full(count, -1, dtype=np.int32),
        np.zeros(count, dtype=np.bool_),
    )


def _fused_row(query, native, mapping, output, tolerance):
    translation, quaternion, directions = native
    i0, i1, alpha, valid = mapping
    out_t, out_q, status, failure_edge, ambiguous = output
    if not valid[query]:
        return
    fraction = alpha[query]
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        status[query], failure_edge[query] = _INVALID_ALPHA, 0
        return
    left, right = i0[query], i1[query]
    samples = translation.shape[1]
    if left < 0 or right < 0 or left >= samples or right >= samples:
        status[query], failure_edge[query] = _INVALID_BRACKET, 0
        return
    result_t, result_q, code, edge, needs_repair = _compiled_query_value(
        translation, quaternion, directions, left, right, fraction, tolerance,
    )
    status[query], failure_edge[query], ambiguous[query] = code, edge, needs_repair
    if code == _OK:
        out_t[query], out_q[query] = result_t, result_q


def _fused_serial_impl(translation, quaternion, directions, i0, i1, alpha, valid, tolerance):
    output = _compiled_empty_result(alpha.size)
    native = (translation, quaternion, directions)
    mapping = (i0, i1, alpha, valid)
    for query in range(alpha.size):
        _compiled_fused_row(query, native, mapping, output, tolerance)
    return output


def _fused_parallel_impl(translation, quaternion, directions, i0, i1, alpha, valid, tolerance):
    output = _compiled_empty_result(alpha.size)
    native = (translation, quaternion, directions)
    mapping = (i0, i1, alpha, valid)
    for query in _parallel_range(alpha.size):
        _compiled_fused_row(query, native, mapping, output, tolerance)
    return output


@lru_cache(maxsize=1)
def _compiled_fused():
    numba = require_numba("spatial.path_solve.pose")
    global _shared_slerp, _compiled_edge_value, _compiled_query_value
    global _compiled_inverse, _compiled_compose, _parallel_range
    global _compiled_empty_result, _compiled_fused_row
    _, _, _shared_slerp = _compile_slerp_dependencies(numba)
    _compiled_inverse = njit_kernel(numba, _fixed.inverse_pose)
    _compiled_compose = njit_kernel(numba, _fixed.compose_pose)
    _compiled_edge_value = njit_kernel(numba, _edge_value)
    _compiled_query_value = njit_kernel(numba, _query_value)
    _compiled_empty_result = njit_kernel(numba, _empty_result)
    _compiled_fused_row = njit_kernel(numba, _fused_row)
    _parallel_range = numba.prange
    serial = njit_kernel(numba, _fused_serial_impl)
    parallel = numba.njit(cache=True, fastmath=False, parallel=True)(_fused_parallel_impl)
    return serial, parallel


def _raise_first_failure(status: np.ndarray, edges: np.ndarray) -> None:
    first: tuple[int, int] | None = None
    for query in np.flatnonzero(status):
        candidate = (int(edges[query]), int(query))
        if first is None or candidate < first:
            first = candidate
    if first is None:
        return
    edge, query = first
    descriptions = {
        _INVALID_QUATERNION: "quaternion norm must be finite and > 0",
        _INVALID_ALPHA: "finite alpha values must be within [0, 1]",
        _INVALID_BRACKET: "parameter bracket index is out of range",
        _INVALID_DIRECTION: "path direction must be +1 or -1",
    }
    raise ValueError(f"edge {edge}, query {query}: {descriptions[int(status[query])]}")


def _repair_principal_arcs(
    result: tuple[np.ndarray, np.ndarray],
    ambiguous: np.ndarray,
    status: np.ndarray,
    translation: np.ndarray,
    quaternion: np.ndarray,
    directions: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
) -> None:
    rows = np.flatnonzero(ambiguous & (status == _OK))
    for start in range(0, rows.size, _REPAIR_BLOCK_SIZE):
        selected = rows[start : start + _REPAIR_BLOCK_SIZE]
        repaired = stream_pose_path_blocks(
            tuple(translation), tuple(quaternion), directions,
            i0=i0[selected], i1=i1[selected], alpha=alpha[selected], valid=valid[selected],
        )
        result[0][selected], result[1][selected] = repaired


def fuse_pose_path(
    translation: np.ndarray,
    quaternion: np.ndarray,
    directions: np.ndarray,
    *,
    i0: np.ndarray,
    i1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
    quaternion_dtypes: tuple[np.dtype, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one eager dynamic path using core-owned bracket decisions."""
    if translation.ndim != 3 or translation.shape[-1] != 3:
        raise ValueError("fused translation must have shape (edge, sample, 3).")
    if quaternion.shape != (*translation.shape[:2], 4):
        raise ValueError("fused quaternion must have shape (edge, sample, 4).")
    if directions.shape != (translation.shape[0],):
        raise ValueError("fused directions must have one entry per edge.")
    if any(value.shape != alpha.shape for value in (i0, i1, valid)):
        raise ValueError("fused parameter-map columns must have matching shapes.")
    serial, parallel = _compiled_fused()
    selected = parallel if alpha.size >= _PARALLEL_MIN_ROWS else serial
    result_t, result_q, status, edges, ambiguous = selected(
        np.ascontiguousarray(translation, dtype=np.float64),
        np.ascontiguousarray(quaternion, dtype=np.float64),
        np.ascontiguousarray(directions, dtype=np.int8),
        np.ascontiguousarray(i0, dtype=np.int64),
        np.ascontiguousarray(i1, dtype=np.int64),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.ascontiguousarray(valid, dtype=np.bool_),
        half_turn_tolerance(quaternion_dtypes),
    )
    result = (result_t, result_q)
    _repair_principal_arcs(result, ambiguous, status, translation, quaternion, directions, i0, i1, alpha, valid)
    _raise_first_failure(status, edges)
    return result


__all__ = ["fuse_pose_path"]
