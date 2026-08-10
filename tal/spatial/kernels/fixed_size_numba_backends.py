from __future__ import annotations

from functools import lru_cache

import numpy as np

from tal.utils.numba_support import njit_kernel, require_numba

from ._fixed_size_common import (
    DET_ATOL,
    FixedRows,
    prepare_binary_quat_rows,
    prepare_matrix_to_quat_rows,
    prepare_pose_compose_rows,
    prepare_pose_inverse_rows,
    prepare_pose_matrix_rows,
    prepare_quat_to_matrix_rows,
    prepare_unary_quat_rows,
    prepare_vec_quat_rows,
    raise_fixed_status,
)
from ._fixed_size_constants import (
    ORTHO_ATOL,
    POSE_MATRIX_SIZE,
    STATUS_INVALID_DET,
    STATUS_INVALID_QUAT,
    STATUS_NONFINITE_DET,
    STATUS_NONFINITE_MATRIX,
    STATUS_NONORTHONORMAL_MATRIX,
    STATUS_OK,
)
from .fixed_size_primitives import (
    inverse_pose as _inverse_pose,
    normalize_quat_array as _normalize_quat_array,
    normalize_quat_components as _normalize_quat_components,
    quat_multiply as _quat_multiply,
    rotate_vec3 as _rotate_vec3,
)
from .rigid_matrix_validation import require_real_matrix_dtype

_HELPERS_JITTED = False


def _compile(func):
    numba = require_numba("spatial.fixed_size_backend")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, func)


def _jit_kernel_helpers(numba) -> None:
    global _HELPERS_JITTED
    global _column_dot, _inverse_pose, _matrix_det, _matrix_validation_status, _normalize_quat_array
    global _normalize_quat_components, _orthonormal_status, _quat_from_matrix, _quat_multiply, _rotate_vec3
    global _write_matrix_from_quat
    if _HELPERS_JITTED:
        return
    _normalize_quat_components = njit_kernel(numba, _normalize_quat_components)
    _normalize_quat_array = njit_kernel(numba, _normalize_quat_array)
    _quat_multiply = njit_kernel(numba, _quat_multiply)
    _rotate_vec3 = njit_kernel(numba, _rotate_vec3)
    _inverse_pose = njit_kernel(numba, _inverse_pose)
    _write_matrix_from_quat = njit_kernel(numba, _write_matrix_from_quat)
    _matrix_det = njit_kernel(numba, _matrix_det)
    _column_dot = njit_kernel(numba, _column_dot)
    _orthonormal_status = njit_kernel(numba, _orthonormal_status)
    _matrix_validation_status = njit_kernel(numba, _matrix_validation_status)
    _quat_from_matrix = njit_kernel(numba, _quat_from_matrix)
    _HELPERS_JITTED = True


@lru_cache(maxsize=1)
def _compiled_quat_compose():
    return _compile(_quat_compose_impl)


@lru_cache(maxsize=1)
def _compiled_quat_inverse():
    return _compile(_quat_inverse_impl)


@lru_cache(maxsize=1)
def _compiled_quat_to_matrix():
    return _compile(_quat_to_matrix_impl)


@lru_cache(maxsize=1)
def _compiled_matrix_to_quat():
    return _compile(_matrix_to_quat_impl)


@lru_cache(maxsize=1)
def _compiled_rotate_vec3():
    return _compile(_rotate_vec3_impl)


@lru_cache(maxsize=1)
def _compiled_pose_compose_translation():
    return _compile(_pose_compose_translation_impl)


@lru_cache(maxsize=1)
def _compiled_pose_inverse_translation():
    return _compile(_pose_inverse_translation_impl)


@lru_cache(maxsize=1)
def _compiled_pose_components_to_matrix():
    return _compile(_pose_components_to_matrix_impl)


def _run_unary(rows: FixedRows, compiled, *, owner: str) -> np.ndarray:
    out, status = compiled(rows.row_arrays[0])
    raise_fixed_status(int(status), owner=owner)
    return out.reshape(rows.output_shape)


def quat_compose_block_numba(left: object, right: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    rows = prepare_binary_quat_rows(left, right, owner=owner)
    out, status = _compiled_quat_compose()(*rows.row_arrays)
    raise_fixed_status(int(status), owner=owner)
    return out.reshape(rows.output_shape)


def quat_inverse_block_numba(values: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    return _run_unary(prepare_unary_quat_rows(values, owner=owner), _compiled_quat_inverse(), owner=owner)


def quat_to_matrix_block_numba(values: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    return _run_unary(prepare_quat_to_matrix_rows(values, owner=owner), _compiled_quat_to_matrix(), owner=owner)


def matrix_to_quat_block_numba(matrix: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    require_real_matrix_dtype(matrix, owner=owner)
    return _run_unary(prepare_matrix_to_quat_rows(matrix, owner=owner), _compiled_matrix_to_quat(), owner=owner)


def rotate_vec3_block_numba(values: object, quat: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    rows = prepare_vec_quat_rows(values, quat, owner=owner)
    out, status = _compiled_rotate_vec3()(*rows.row_arrays)
    raise_fixed_status(int(status), owner=owner)
    return out.reshape(rows.output_shape)


def pose_compose_translation_block_numba(left_t: object, right_t: object, right_q: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    rows = prepare_pose_compose_rows(left_t, right_t, right_q, owner=owner)
    out, status = _compiled_pose_compose_translation()(*rows.row_arrays)
    raise_fixed_status(int(status), owner=owner)
    return out.reshape(rows.output_shape)


def pose_inverse_translation_block_numba(translation: object, quat: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    rows = prepare_pose_inverse_rows(translation, quat, owner=owner)
    out, status = _compiled_pose_inverse_translation()(*rows.row_arrays)
    raise_fixed_status(int(status), owner=owner)
    return out.reshape(rows.output_shape)


def pose_components_to_matrix_block_numba(translation: object, quat: object, *, owner: str) -> np.ndarray:
    require_numba(owner)
    rows = prepare_pose_matrix_rows(translation, quat, owner=owner)
    out, status = _compiled_pose_components_to_matrix()(*rows.row_arrays)
    raise_fixed_status(int(status), owner=owner)
    return out.reshape(rows.output_shape)


def _quat_compose_impl(left, right):
    out = np.empty_like(left)
    for row in range(left.shape[0]):
        status_l, lx, ly, lz, lw = _normalize_quat_array(left, row)
        status_r, rx, ry, rz, rw = _normalize_quat_array(right, row)
        if status_l != STATUS_OK or status_r != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        status, qx, qy, qz, qw = _normalize_quat_components(*_quat_multiply((rx, ry, rz, rw), (lx, ly, lz, lw)))
        if status != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        out[row, 0] = qx
        out[row, 1] = qy
        out[row, 2] = qz
        out[row, 3] = qw
    return out, STATUS_OK


def _quat_inverse_impl(values):
    out = np.empty_like(values)
    for row in range(values.shape[0]):
        status, qx, qy, qz, qw = _normalize_quat_array(values, row)
        if status != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        out[row, 0] = -qx
        out[row, 1] = -qy
        out[row, 2] = -qz
        out[row, 3] = qw
    return out, STATUS_OK


def _write_matrix_from_quat(out, row, qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    out[row, 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[row, 0, 1] = 2.0 * (xy - wz)
    out[row, 0, 2] = 2.0 * (xz + wy)
    out[row, 1, 0] = 2.0 * (xy + wz)
    out[row, 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[row, 1, 2] = 2.0 * (yz - wx)
    out[row, 2, 0] = 2.0 * (xz - wy)
    out[row, 2, 1] = 2.0 * (yz + wx)
    out[row, 2, 2] = 1.0 - 2.0 * (xx + yy)


def _quat_to_matrix_impl(values):
    out = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    for row in range(values.shape[0]):
        status, qx, qy, qz, qw = _normalize_quat_array(values, row)
        if status != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        _write_matrix_from_quat(out, row, qx, qy, qz, qw)
    return out, STATUS_OK


def _matrix_det(matrix, row):
    m = matrix[row]
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


def _column_dot(matrix, row, col_a, col_b):
    dot = 0.0
    for idx in range(3):
        value = matrix[row, idx, col_a]
        other = matrix[row, idx, col_b]
        if not np.isfinite(value) or not np.isfinite(other):
            return STATUS_NONFINITE_MATRIX, 0.0
        dot += value * other
    return STATUS_OK, dot


def _orthonormal_status(matrix, row):
    max_error = 0.0
    for pair in range(9):
        col_a = pair // 3
        col_b = pair - col_a * 3
        status, dot = _column_dot(matrix, row, col_a, col_b)
        if status != STATUS_OK:
            return status
        expected = 1.0 if col_a == col_b else 0.0
        max_error = max(max_error, abs(dot - expected))
    if max_error <= ORTHO_ATOL:
        return STATUS_OK
    return STATUS_NONORTHONORMAL_MATRIX


def _matrix_validation_status(matrix, row):
    status = _orthonormal_status(matrix, row)
    if status != STATUS_OK:
        return status
    det = _matrix_det(matrix, row)
    if not np.isfinite(det):
        return STATUS_NONFINITE_DET
    if abs(det - 1.0) > DET_ATOL:
        return STATUS_INVALID_DET
    return STATUS_OK


def _quat_from_matrix(matrix, row):
    m = matrix[row]
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return (m[2, 1] - m[1, 2]) / scale, (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale, 0.25 * scale
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return 0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[2, 1] - m[1, 2]) / scale
    if m[1, 1] > m[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return (m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale, (m[0, 2] - m[2, 0]) / scale
    scale = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return (m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale, (m[1, 0] - m[0, 1]) / scale


def _matrix_to_quat_impl(matrix):
    out = np.empty((matrix.shape[0], 4), dtype=np.float64)
    for row in range(matrix.shape[0]):
        status = _matrix_validation_status(matrix, row)
        if status != STATUS_OK:
            return out, status
        status, qx, qy, qz, qw = _normalize_quat_components(*_quat_from_matrix(matrix, row))
        if status != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        out[row, 0] = qx
        out[row, 1] = qy
        out[row, 2] = qz
        out[row, 3] = qw
    return out, STATUS_OK


def _rotate_vec3_impl(values, quat):
    out = np.empty_like(values)
    for row in range(values.shape[0]):
        status, qx, qy, qz, qw = _normalize_quat_array(quat, row)
        if status != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        rotated = _rotate_vec3((qx, qy, qz, qw), (values[row, 0], values[row, 1], values[row, 2]))
        out[row, 0] = rotated[0]
        out[row, 1] = rotated[1]
        out[row, 2] = rotated[2]
    return out, STATUS_OK


def _pose_compose_translation_impl(left_t, right_t, right_q):
    out = np.empty_like(left_t)
    for row in range(left_t.shape[0]):
        status, qx, qy, qz, qw = _normalize_quat_array(right_q, row)
        if status != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        rotated = _rotate_vec3((qx, qy, qz, qw), (left_t[row, 0], left_t[row, 1], left_t[row, 2]))
        out[row, 0] = rotated[0] + right_t[row, 0]
        out[row, 1] = rotated[1] + right_t[row, 1]
        out[row, 2] = rotated[2] + right_t[row, 2]
    return out, STATUS_OK


def _pose_inverse_translation_impl(translation, quat):
    out = np.empty_like(translation)
    for row in range(translation.shape[0]):
        status, qx, qy, qz, qw = _normalize_quat_array(quat, row)
        if status != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        inverted, _ = _inverse_pose((translation[row, 0], translation[row, 1], translation[row, 2]), (qx, qy, qz, qw))
        out[row, 0] = inverted[0]
        out[row, 1] = inverted[1]
        out[row, 2] = inverted[2]
    return out, STATUS_OK


def _pose_components_to_matrix_impl(translation, quat):
    out = np.zeros((translation.shape[0], POSE_MATRIX_SIZE, POSE_MATRIX_SIZE), dtype=np.float64)
    for row in range(translation.shape[0]):
        status, qx, qy, qz, qw = _normalize_quat_array(quat, row)
        if status != STATUS_OK:
            return out, STATUS_INVALID_QUAT
        _write_matrix_from_quat(out, row, qx, qy, qz, qw)
        out[row, 0, 3] = translation[row, 0]
        out[row, 1, 3] = translation[row, 1]
        out[row, 2, 3] = translation[row, 2]
        out[row, 3, 3] = 1.0
    return out, STATUS_OK


__all__ = [
    "matrix_to_quat_block_numba",
    "pose_compose_translation_block_numba",
    "pose_components_to_matrix_block_numba",
    "pose_inverse_translation_block_numba",
    "quat_compose_block_numba",
    "quat_inverse_block_numba",
    "quat_to_matrix_block_numba",
    "rotate_vec3_block_numba",
]
