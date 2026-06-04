from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tal.utils.numba_scan import ScanAxisSpec, ScanInputSpec, prepare_scan_rows

from ._topology_scan_constants import (
    QUAT_SIZE,
    STATUS_EMPTY,
    STATUS_INVALID_DIRECTION,
    STATUS_INVALID_QUAT,
    STATUS_LEFT_PACKED,
    VEC3_SIZE,
)


@dataclass(frozen=True)
class ChainPoseRows:
    translation: np.ndarray
    quat: np.ndarray
    valid: np.ndarray
    direction: np.ndarray
    translation_output_shape: tuple[int, ...]
    quat_output_shape: tuple[int, ...]


def _shape(value: object) -> tuple[int, ...]:
    return tuple(int(size) for size in np.shape(value))


def validate_chain_pose_shapes(translation: object, quat: object, valid: object, direction: object, *, owner: str) -> None:
    t_shape = _shape(translation)
    q_shape = _shape(quat)
    valid_shape = _shape(valid)
    direction_shape = _shape(direction)
    if len(t_shape) < 1 or t_shape[-1] != VEC3_SIZE:
        raise ValueError(f"{owner}: translation must have trailing vector dim length 3.")
    if len(q_shape) < 1 or q_shape[-1] != QUAT_SIZE:
        raise ValueError(f"{owner}: quat must have trailing quaternion dim length 4.")
    if not (t_shape[:-1] == q_shape[:-1] == valid_shape == direction_shape):
        raise ValueError(f"{owner}: translation, quat, valid, and direction shapes must match exactly.")
    direction_array = np.asarray(direction)
    if direction_array.dtype == np.dtype(bool) or not np.issubdtype(direction_array.dtype, np.integer):
        raise ValueError(f"{owner}: direction must use an integer dtype.")


def prepare_chain_pose_rows(translation: object, quat: object, valid: object, direction: object, *, owner: str) -> ChainPoseRows:
    validate_chain_pose_shapes(translation, quat, valid, direction, owner=owner)
    prepared = prepare_scan_rows(
        (translation, quat, valid, direction),
        (
            ScanInputSpec("translation", 1, 1, np.float64),
            ScanInputSpec("quat", 1, 1, np.float64),
            ScanInputSpec("valid", 1, 0, bool),
            ScanInputSpec("direction", 1, 0, np.int64),
        ),
        ordered_axes=(ScanAxisSpec("chain", "topology"),),
        output_core_shapes=((VEC3_SIZE,), (QUAT_SIZE,)),
        owner=owner,
    )
    translation_rows, quat_rows, valid_rows, direction_rows = prepared.row_arrays
    return ChainPoseRows(
        translation_rows,
        quat_rows,
        valid_rows,
        direction_rows,
        prepared.output_shapes[0],
        prepared.output_shapes[1],
    )


def raise_chain_pose_status(status: int, *, owner: str) -> None:
    if status == STATUS_LEFT_PACKED:
        raise ValueError(f"{owner}: valid mask must be left-packed on the topology chain axis.")
    if status == STATUS_EMPTY:
        raise ValueError(f"{owner}: topology scan requires at least 1 valid edge per batch row.")
    if status == STATUS_INVALID_QUAT:
        raise ValueError(f"{owner}: quaternion norm must be finite and > 0 on the valid topology prefix.")
    if status == STATUS_INVALID_DIRECTION:
        raise ValueError(f"{owner}: direction values must be 1 or -1 on the valid topology prefix.")


__all__ = ["ChainPoseRows", "prepare_chain_pose_rows", "raise_chain_pose_status", "validate_chain_pose_shapes"]
