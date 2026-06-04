from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tal.utils.block_rows import BlockInputSpec, prepare_block_rows

from ._fixed_size_constants import (
    DET_ATOL,
    ORTHO_ATOL,
    POSE_MATRIX_SIZE,
    QUAT_SIZE,
    ROT_MATRIX_SIZE,
    STATUS_INVALID_DET,
    STATUS_INVALID_QUAT,
    STATUS_NONFINITE_DET,
    STATUS_NONFINITE_MATRIX,
    STATUS_NONORTHONORMAL_MATRIX,
    STATUS_OK,
    VEC3_SIZE,
)


@dataclass(frozen=True)
class FixedRows:
    row_arrays: tuple[np.ndarray, ...]
    output_shape: tuple[int, ...]


def _shape(value: object) -> tuple[int, ...]:
    return tuple(int(size) for size in np.shape(value))


def _require_trailing(shape: tuple[int, ...], tail: tuple[int, ...], *, name: str, owner: str) -> None:
    if len(shape) < len(tail) or shape[-len(tail) :] != tail:
        raise ValueError(f"{owner}: {name} must have trailing shape {tail}.")


def _prepare_rows(blocks: tuple[object, ...], specs: tuple[BlockInputSpec, ...], *, output: tuple[int, ...], owner: str) -> FixedRows:
    prepared = prepare_block_rows(blocks, specs, output_core_shape=output, owner=owner)
    return FixedRows(prepared.row_arrays, prepared.output_shape)


def prepare_binary_quat_rows(left: object, right: object, *, owner: str) -> FixedRows:
    left_shape = _shape(left)
    right_shape = _shape(right)
    _require_trailing(left_shape, (QUAT_SIZE,), name="left", owner=owner)
    _require_trailing(right_shape, (QUAT_SIZE,), name="right", owner=owner)
    if left_shape != right_shape:
        raise ValueError(f"{owner}: left and right quaternion shapes must match exactly.")
    specs = (BlockInputSpec("left", 1, np.float64), BlockInputSpec("right", 1, np.float64))
    return _prepare_rows((left, right), specs, output=(QUAT_SIZE,), owner=owner)


def prepare_unary_quat_rows(values: object, *, owner: str, name: str = "quat") -> FixedRows:
    _require_trailing(_shape(values), (QUAT_SIZE,), name=name, owner=owner)
    return _prepare_rows((values,), (BlockInputSpec(name, 1, np.float64),), output=(QUAT_SIZE,), owner=owner)


def prepare_quat_to_matrix_rows(values: object, *, owner: str) -> FixedRows:
    _require_trailing(_shape(values), (QUAT_SIZE,), name="quat", owner=owner)
    return _prepare_rows((values,), (BlockInputSpec("quat", 1, np.float64),), output=(ROT_MATRIX_SIZE, ROT_MATRIX_SIZE), owner=owner)


def prepare_matrix_to_quat_rows(matrix: object, *, owner: str) -> FixedRows:
    _require_trailing(_shape(matrix), (ROT_MATRIX_SIZE, ROT_MATRIX_SIZE), name="matrix", owner=owner)
    return _prepare_rows((matrix,), (BlockInputSpec("matrix", 2, np.float64),), output=(QUAT_SIZE,), owner=owner)


def prepare_vec_quat_rows(values: object, quat: object, *, owner: str) -> FixedRows:
    values_shape = _shape(values)
    quat_shape = _shape(quat)
    _require_trailing(values_shape, (VEC3_SIZE,), name="values", owner=owner)
    _require_trailing(quat_shape, (QUAT_SIZE,), name="quat", owner=owner)
    if values_shape[:-1] != quat_shape[:-1]:
        raise ValueError(f"{owner}: values and quat non-core shapes must match exactly.")
    specs = (BlockInputSpec("values", 1, np.float64), BlockInputSpec("quat", 1, np.float64))
    return _prepare_rows((values, quat), specs, output=(VEC3_SIZE,), owner=owner)


def prepare_pose_compose_rows(left_t: object, right_t: object, right_q: object, *, owner: str) -> FixedRows:
    left_shape = _shape(left_t)
    right_shape = _shape(right_t)
    quat_shape = _shape(right_q)
    _require_trailing(left_shape, (VEC3_SIZE,), name="left_t", owner=owner)
    _require_trailing(right_shape, (VEC3_SIZE,), name="right_t", owner=owner)
    _require_trailing(quat_shape, (QUAT_SIZE,), name="right_q", owner=owner)
    if left_shape[:-1] != right_shape[:-1] or left_shape[:-1] != quat_shape[:-1]:
        raise ValueError(f"{owner}: pose compose input non-core shapes must match exactly.")
    specs = (
        BlockInputSpec("left_t", 1, np.float64),
        BlockInputSpec("right_t", 1, np.float64),
        BlockInputSpec("right_q", 1, np.float64),
    )
    return _prepare_rows((left_t, right_t, right_q), specs, output=(VEC3_SIZE,), owner=owner)


def prepare_pose_inverse_rows(translation: object, quat: object, *, owner: str) -> FixedRows:
    rows = prepare_vec_quat_rows(translation, quat, owner=owner)
    return FixedRows(rows.row_arrays, rows.output_shape)


def prepare_pose_matrix_rows(translation: object, quat: object, *, owner: str) -> FixedRows:
    translation_shape = _shape(translation)
    quat_shape = _shape(quat)
    _require_trailing(translation_shape, (VEC3_SIZE,), name="translation", owner=owner)
    _require_trailing(quat_shape, (QUAT_SIZE,), name="quat", owner=owner)
    if translation_shape[:-1] != quat_shape[:-1]:
        raise ValueError(f"{owner}: translation and quat non-core shapes must match exactly.")
    specs = (BlockInputSpec("translation", 1, np.float64), BlockInputSpec("quat", 1, np.float64))
    return _prepare_rows((translation, quat), specs, output=(POSE_MATRIX_SIZE, POSE_MATRIX_SIZE), owner=owner)


def _quat_status(values: np.ndarray) -> int:
    totals = np.sum(values * values, axis=1)
    if not np.isfinite(values).all() or not np.isfinite(totals).all() or np.any(totals <= 0.0):
        return STATUS_INVALID_QUAT
    return STATUS_OK


def validate_quat_rows(*row_arrays: np.ndarray, owner: str) -> None:
    for values in row_arrays:
        raise_fixed_status(_quat_status(values), owner=owner)


def _matrix_det(matrix: np.ndarray, row: int) -> float:
    m = matrix[row]
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


def _column_dot(matrix: np.ndarray, row: int, col_a: int, col_b: int) -> tuple[int, float]:
    dot = 0.0
    for idx in range(ROT_MATRIX_SIZE):
        value = matrix[row, idx, col_a]
        other = matrix[row, idx, col_b]
        if not np.isfinite(value) or not np.isfinite(other):
            return STATUS_NONFINITE_MATRIX, 0.0
        dot += value * other
    return STATUS_OK, dot


def _orthonormal_status(matrix: np.ndarray, row: int) -> int:
    max_error = 0.0
    for pair in range(ROT_MATRIX_SIZE * ROT_MATRIX_SIZE):
        col_a = pair // ROT_MATRIX_SIZE
        col_b = pair - col_a * ROT_MATRIX_SIZE
        status, dot = _column_dot(matrix, row, col_a, col_b)
        if status != STATUS_OK:
            return status
        expected = 1.0 if col_a == col_b else 0.0
        max_error = max(max_error, abs(dot - expected))
    if max_error <= ORTHO_ATOL:
        return STATUS_OK
    return STATUS_NONORTHONORMAL_MATRIX


def _matrix_status(matrix: np.ndarray) -> int:
    for row in range(matrix.shape[0]):
        status = _orthonormal_status(matrix, row)
        if status != STATUS_OK:
            return status
        det = _matrix_det(matrix, row)
        if not np.isfinite(det):
            return STATUS_NONFINITE_DET
        if abs(det - 1.0) > DET_ATOL:
            return STATUS_INVALID_DET
    return STATUS_OK


def validate_matrix_rows(matrix: np.ndarray, *, owner: str) -> None:
    raise_fixed_status(_matrix_status(matrix), owner=owner)


def raise_fixed_status(status: int, *, owner: str) -> None:
    if status == STATUS_INVALID_QUAT:
        raise ValueError(f"{owner}: quaternion norm must be finite and > 0.")
    if status == STATUS_NONFINITE_MATRIX:
        raise ValueError(f"{owner}: matrix values must be finite.")
    if status == STATUS_NONORTHONORMAL_MATRIX:
        raise ValueError(f"{owner}: matrix is not orthonormal within tolerance.")
    if status == STATUS_NONFINITE_DET:
        raise ValueError(f"{owner}: matrix determinants must be finite.")
    if status == STATUS_INVALID_DET:
        raise ValueError(f"{owner}: matrix determinant must be 1 within tolerance.")


__all__ = [
    "DET_ATOL",
    "FixedRows",
    "prepare_binary_quat_rows",
    "prepare_matrix_to_quat_rows",
    "prepare_pose_compose_rows",
    "prepare_pose_inverse_rows",
    "prepare_pose_matrix_rows",
    "prepare_quat_to_matrix_rows",
    "prepare_unary_quat_rows",
    "prepare_vec_quat_rows",
    "raise_fixed_status",
    "validate_matrix_rows",
    "validate_quat_rows",
]
