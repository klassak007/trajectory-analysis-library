from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..kernels.batched_pose_path_numba import (
    batched_pose_edge_numba,
    batched_position_apply_numba,
)
from ..kernels.fixed_size_backends import (
    SPATIAL_FIXED_BACKEND_NUMBA,
    SPATIAL_FIXED_BACKEND_SCIPY,
    pose_compose_translation_block_backend,
    pose_inverse_translation_block_backend,
    quat_compose_block_backend,
    quat_inverse_block_backend,
)
from ..kernels.path_kernel_status import (
    PATH_STATUS_INVALID_ALPHA,
    PATH_STATUS_INVALID_BRACKET,
    PATH_STATUS_INVALID_QUATERNION,
    PathKernelFailure,
    raise_path_kernel_failure,
)
from ..kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_NUMBA,
    ROTATION_INTERP_BACKEND_SCIPY,
    slerp_quat_backend,
)
from ..kernels.rotation_interp_reference import (
    half_turn_tolerance,
    invalid_quaternion_rows,
)
from .batched_path_finalize import commit_batched_pose, commit_batched_position
from .batched_path_inputs import (
    PackedBatchedPathInputs,
    PackedPathMap,
    pack_batched_path_inputs,
)
from .batched_path_plan import PreparedBatchedPathExecution
from .path_query_ops import _raise_path_query_execution_error

_Backend = Literal["numba", "scipy"]


@dataclass(frozen=True)
class _PoseBlock:
    translation: np.ndarray
    quaternion: np.ndarray
    valid: np.ndarray


class _BatchedKernelFailure(Exception):
    def __init__(self, failure: PathKernelFailure) -> None:
        self.failure = failure
        super().__init__(failure)


def _partition_slices(
    plan: PreparedBatchedPathExecution,
    index: int,
) -> dict[str, slice]:
    block = plan.physical_rows.partitions[index].block
    return dict(block.selections)


def _map_block(
    mapping: PackedPathMap,
    plan: PreparedBatchedPathExecution,
    index: int,
) -> PackedPathMap:
    selected = _partition_slices(plan, index)
    slices = tuple(selected[dim] for dim in plan.logical_rows.dims)
    return PackedPathMap(
        mapping.i0[slices],
        mapping.i1[slices],
        mapping.alpha[slices],
        mapping.valid[slices],
    )


def _batch_block(
    values: np.ndarray,
    value_dims: tuple[str, ...],
    value_sizes: tuple[int, ...],
    plan: PreparedBatchedPathExecution,
    index: int,
) -> np.ndarray:
    return values[_batch_indices(value_dims, value_sizes, plan, index)]


def _batch_indices(
    value_dims: tuple[str, ...],
    value_sizes: tuple[int, ...],
    plan: PreparedBatchedPathExecution,
    index: int,
) -> np.ndarray:
    target_dims = plan.finalization.batch_dims
    partition = plan.physical_rows.partitions[index]
    target = tuple(partition.shape[:-1])
    origins = partition.origin[:-1]
    batch_index = np.zeros(target, dtype=np.int64)
    for dim, size in zip(value_dims, value_sizes, strict=True):
        axis = target_dims.index(dim)
        stride = int(np.prod(value_sizes[value_dims.index(dim) + 1 :], dtype=np.int64))
        coordinate = np.arange(origins[axis], origins[axis] + target[axis], dtype=np.int64)
        shape = [1] * len(target)
        shape[axis] = target[axis]
        batch_index += coordinate.reshape(shape) * stride
    return batch_index


def _gather_active(
    values: np.ndarray,
    batch_index: np.ndarray,
    mapping: PackedPathMap,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outer = int(np.prod(mapping.alpha.shape[:-1], dtype=np.int64))
    query = int(mapping.alpha.shape[-1])
    width = int(values.shape[-1])
    active_rows = active.reshape(-1)
    shape = (outer, query, width)
    if np.all(active_rows):
        sources = np.broadcast_to(batch_index[..., None], mapping.alpha.shape).reshape(-1)
        left = mapping.i0.reshape(-1)
        right = mapping.i1.reshape(-1)
        return values[sources, left].reshape(shape), values[sources, right].reshape(shape)
    rows = np.flatnonzero(active_rows)
    left_values = np.zeros((outer * query, width), dtype=values.dtype)
    right_values = np.zeros((outer * query, width), dtype=values.dtype)
    if not rows.size:
        return left_values.reshape(shape), right_values.reshape(shape)
    sources = np.broadcast_to(batch_index[..., None], mapping.alpha.shape).reshape(-1)
    left = mapping.i0.reshape(-1)
    right = mapping.i1.reshape(-1)
    left_values[rows] = values[sources[rows], left[rows]]
    right_values[rows] = values[sources[rows], right[rows]]
    return left_values.reshape(shape), right_values.reshape(shape)


def _backend_names(backend: _Backend) -> tuple[str, str]:
    if backend == "numba":
        return ROTATION_INTERP_BACKEND_NUMBA, SPATIAL_FIXED_BACKEND_NUMBA
    return ROTATION_INTERP_BACKEND_SCIPY, SPATIAL_FIXED_BACKEND_SCIPY


def _map_failure_status(
    position_map: PackedPathMap,
    rotation_map: PackedPathMap,
    sample_counts: tuple[int, int],
) -> np.ndarray:
    active = (position_map.valid & rotation_map.valid).reshape(-1)
    alphas = (position_map.alpha.reshape(-1), rotation_map.alpha.reshape(-1))
    alpha_error = active & (~np.isfinite(alphas[0]) | (alphas[0] < 0.0) | (alphas[0] > 1.0))
    alpha_error |= active & (~np.isfinite(alphas[1]) | (alphas[1] < 0.0) | (alphas[1] > 1.0))
    brackets = (
        position_map.i0.reshape(-1), position_map.i1.reshape(-1),
        rotation_map.i0.reshape(-1), rotation_map.i1.reshape(-1),
    )
    bracket_error = active & ((brackets[0] < 0) | (brackets[1] < 0))
    bracket_error |= active & ((brackets[0] >= sample_counts[0]) | (brackets[1] >= sample_counts[0]))
    bracket_error |= active & ((brackets[2] < 0) | (brackets[3] < 0))
    bracket_error |= active & ((brackets[2] >= sample_counts[1]) | (brackets[3] >= sample_counts[1]))
    status = np.zeros(active.shape, dtype=np.int8)
    status[bracket_error] = PATH_STATUS_INVALID_BRACKET
    status[alpha_error] = PATH_STATUS_INVALID_ALPHA
    return status


def _first_status_failure(status: np.ndarray) -> tuple[int, int] | None:
    rows = np.flatnonzero(status)
    if not rows.size:
        return None
    row = int(rows[0])
    return int(status[row]), row


def _first_quaternion_failure(
    left: np.ndarray,
    right: np.ndarray,
    active: np.ndarray,
) -> tuple[int, int] | None:
    invalid = active.reshape(-1) & (
        invalid_quaternion_rows(left.reshape(-1, 4))
        | invalid_quaternion_rows(right.reshape(-1, 4))
    )
    rows = np.flatnonzero(invalid)
    if not rows.size:
        return None
    return PATH_STATUS_INVALID_QUATERNION, int(rows[0])


def _raise_partition_failure(
    failure: tuple[int, int],
    plan: PreparedBatchedPathExecution,
    *,
    edge: int,
    index: int,
) -> None:
    status, row = failure
    partition = plan.physical_rows.partitions[index]
    raise _BatchedKernelFailure(
        PathKernelFailure(edge, partition.global_row_position(row), status)
    )


def _earliest_failure(
    *failures: tuple[int, int] | None,
) -> tuple[int, int] | None:
    present = tuple(failure for failure in failures if failure is not None)
    return min(present, key=lambda failure: failure[1]) if present else None


def _empty_edge_block(mapping: PackedPathMap) -> _PoseBlock:
    shape = mapping.valid.shape
    translation = np.zeros((*shape, 3), dtype=np.float64)
    quaternion = np.zeros((*shape, 4), dtype=np.float64)
    quaternion[..., 3] = 1.0
    return _PoseBlock(translation, quaternion, np.zeros(shape, dtype=np.bool_))


def _interpolate_edge(
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    edge: int,
    index: int,
    *,
    backend: _Backend,
) -> _PoseBlock:
    provider = packed.providers[edge]
    batch_index = _batch_indices(provider.batch_dims, provider.batch_sizes, plan, index)
    position_map = _map_block(packed.maps[provider.position_map], plan, index)
    rotation_map = _map_block(packed.maps[provider.rotation_map], plan, index)
    valid = position_map.valid & rotation_map.valid
    if not np.any(valid):
        return _empty_edge_block(position_map)
    map_status = _map_failure_status(
        position_map,
        rotation_map,
        (provider.translation.shape[1], provider.quaternion.shape[1]),
    )
    gather_valid = valid & ~map_status.reshape(valid.shape).astype(bool)
    left_t, right_t = _gather_active(
        provider.translation, batch_index, position_map, gather_valid,
    )
    left_q, right_q = _gather_active(
        provider.quaternion, batch_index, rotation_map, gather_valid,
    )
    if backend == "scipy":
        failure = _earliest_failure(
            _first_status_failure(map_status),
            _first_quaternion_failure(left_q, right_q, gather_valid),
        )
        if failure is not None:
            _raise_partition_failure(failure, plan, edge=edge, index=index)
    outer, query = left_t.shape[:2]
    position_alpha = position_map.alpha.reshape(outer, query)
    rotation_alpha = rotation_map.alpha.reshape(outer, query)
    active = valid.reshape(outer, query)
    translation = (1.0 - position_alpha[..., None]) * left_t + position_alpha[..., None] * right_t
    rotation_backend, _ = _backend_names(backend)
    quaternion = slerp_quat_backend(
        left_q,
        right_q,
        rotation_alpha,
        active,
        backend=rotation_backend,
    )
    shape = (*position_map.alpha.shape, 3)
    translation = np.where(valid[..., None], translation.reshape(shape), 0.0)
    quaternion = quaternion.reshape((*rotation_map.alpha.shape, 4))
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))
    quaternion = np.where(valid[..., None], quaternion, identity)
    return _PoseBlock(translation, quaternion, valid)


def _orient(block: _PoseBlock, *, invert: bool, backend: _Backend) -> _PoseBlock:
    if not invert:
        return block
    _, fixed_backend = _backend_names(backend)
    translation = pose_inverse_translation_block_backend(
        block.translation,
        block.quaternion,
        backend=fixed_backend,
    )
    quaternion = quat_inverse_block_backend(block.quaternion, backend=fixed_backend)
    return _PoseBlock(translation, quaternion, block.valid)


def _compose(left: _PoseBlock, right: _PoseBlock, *, backend: _Backend) -> _PoseBlock:
    _, fixed_backend = _backend_names(backend)
    valid = left.valid & right.valid
    translation = pose_compose_translation_block_backend(
        left.translation,
        right.translation,
        right.quaternion,
        backend=fixed_backend,
    )
    quaternion = quat_compose_block_backend(
        left.quaternion,
        right.quaternion,
        backend=fixed_backend,
    )
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))
    return _PoseBlock(
        np.where(valid[..., None], translation, 0.0),
        np.where(valid[..., None], quaternion, identity),
        valid,
    )


def _partition_pose_scipy(
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    index: int,
    *,
    backend: _Backend,
) -> _PoseBlock:
    accumulated: _PoseBlock | None = None
    for edge in range(len(packed.providers)):
        current = _interpolate_edge(packed, plan, edge, index, backend=backend)
        current = _orient(current, invert=plan.path.steps[edge].invert, backend=backend)
        accumulated = current if accumulated is None else _compose(accumulated, current, backend=backend)
    if accumulated is None:
        raise ValueError("batched path execution requires at least one edge.")
    return accumulated


def _flat_map(mapping: PackedPathMap) -> tuple[np.ndarray, ...]:
    return (
        np.ascontiguousarray(mapping.i0.reshape(-1), dtype=np.int64),
        np.ascontiguousarray(mapping.i1.reshape(-1), dtype=np.int64),
        np.ascontiguousarray(mapping.alpha.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(mapping.valid.reshape(-1), dtype=np.bool_),
    )


def _empty_pose_block(plan: PreparedBatchedPathExecution, index: int) -> _PoseBlock:
    shape = plan.physical_rows.partitions[index].shape
    rows = int(np.prod(shape, dtype=np.int64))
    translation = np.zeros((rows, 3), dtype=np.float64)
    quaternion = np.zeros((rows, 4), dtype=np.float64)
    quaternion[:, 3] = 1.0
    return _PoseBlock(translation, quaternion, np.zeros(rows, dtype=np.bool_))


def _run_numba_edge(
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    block: _PoseBlock,
    edge: int,
    index: int,
) -> bool:
    provider = packed.providers[edge]
    batch_index = _batch_indices(provider.batch_dims, provider.batch_sizes, plan, index)
    position_map = _flat_map(_map_block(packed.maps[provider.position_map], plan, index))
    rotation_map = _flat_map(_map_block(packed.maps[provider.rotation_map], plan, index))
    status, row, ambiguous = batched_pose_edge_numba(
        provider.translation,
        provider.quaternion,
        np.ascontiguousarray(batch_index.reshape(-1)),
        position_map,
        rotation_map,
        -1 if plan.path.steps[edge].invert else 1,
        half_turn_tolerance((provider.quaternion.dtype,)),
        (block.translation, block.quaternion, block.valid),
        edge == 0,
    )
    if status:
        partition = plan.physical_rows.partitions[index]
        raise _BatchedKernelFailure(
            PathKernelFailure(edge, partition.global_row_position(row), status)
        )
    return bool(ambiguous)


def _partition_pose_numba(
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    index: int,
) -> _PoseBlock:
    block = _empty_pose_block(plan, index)
    ambiguous = False
    for edge in range(len(packed.providers)):
        ambiguous = _run_numba_edge(packed, plan, block, edge, index) or ambiguous
    if ambiguous:
        return _partition_pose_scipy(packed, plan, index, backend="scipy")
    shape = plan.physical_rows.partitions[index].shape
    return _PoseBlock(
        block.translation.reshape(*shape, 3),
        block.quaternion.reshape(*shape, 4),
        block.valid.reshape(shape),
    )


def _target_slices(plan: PreparedBatchedPathExecution, index: int) -> tuple[object, ...]:
    selected = dict(plan.physical_rows.partitions[index].block.selections)
    return tuple(selected[dim] for dim in plan.logical_rows.dims)


def _caller_block(
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if packed.caller is None or packed.caller_valid is None:
        raise ValueError("batched Position execution requires a caller.")
    query_slice = plan.physical_rows.partitions[index].block.selection_for_dim(
        plan.finalization.query_dim
    )
    values = _batch_block(
        packed.caller[..., query_slice, :],
        packed.caller_batch_dims,
        packed.caller_batch_sizes,
        plan,
        index,
    )
    valid = _batch_block(
        packed.caller_valid[..., query_slice],
        packed.caller_batch_dims,
        packed.caller_batch_sizes,
        plan,
        index,
    )
    return values, valid


def _execute_partition(
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    index: int,
    backend: _Backend,
) -> _PoseBlock:
    if backend == "numba":
        return _partition_pose_numba(packed, plan, index)
    return _partition_pose_scipy(packed, plan, index, backend=backend)


def _apply_position_block(
    pose: _PoseBlock,
    caller: np.ndarray,
    caller_valid: np.ndarray,
    backend: _Backend,
) -> np.ndarray:
    active = pose.valid & caller_valid
    applied = np.full(caller.shape, np.nan, dtype=np.float64)
    if backend == "numba":
        batched_position_apply_numba(
            np.ascontiguousarray(caller.reshape(-1, 3)),
            pose.translation.reshape(-1, 3),
            pose.quaternion.reshape(-1, 4),
            pose.valid.reshape(-1),
            caller_valid.reshape(-1),
            applied.reshape(-1, 3),
        )
        return applied
    transformed = pose_compose_translation_block_backend(
        caller,
        pose.translation,
        pose.quaternion,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    return np.where(active[..., None], transformed, np.nan)


def _write_partition(
    outputs: tuple[np.ndarray, np.ndarray | None],
    pose: _PoseBlock,
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    index: int,
    backend: _Backend,
) -> None:
    first, second = outputs
    target = _target_slices(plan, index)
    if second is not None:
        first[target] = np.where(pose.valid[..., None], pose.translation, np.nan)
        second[target] = np.where(pose.valid[..., None], pose.quaternion, np.nan)
        return
    caller, caller_valid = _caller_block(packed, plan, index)
    applied = _apply_position_block(pose, caller, caller_valid, backend)
    first[target] = applied


def _execute_arrays(
    plan: PreparedBatchedPathExecution,
    packed: PackedBatchedPathInputs,
    *,
    backend: _Backend,
) -> tuple[np.ndarray, np.ndarray | None]:
    shape = plan.logical_rows.sizes
    pose_output = plan.finalization.output == "pose"
    first = np.full((*shape, 3), np.nan, dtype=np.float64)
    second = np.full((*shape, 4), np.nan, dtype=np.float64) if pose_output else None
    failures: list[PathKernelFailure] = []
    for index in range(len(plan.physical_rows.partitions)):
        try:
            pose = _execute_partition(packed, plan, index, backend)
        except _BatchedKernelFailure as exc:
            failures.append(exc.failure)
            continue
        _write_partition((first, second), pose, packed, plan, index, backend)
    if failures:
        raise_path_kernel_failure(min(failures, key=lambda failure: (failure.edge, failure.row)))
    return first, second


def execute_eager_batched_path(
    plan: PreparedBatchedPathExecution,
    *,
    backend: _Backend,
    owner: str,
):
    """Execute and commit one eligible eager batched path."""
    if not plan.eligible or plan.storage != "eager":
        raise ValueError("batched eager path executor requires an eligible eager plan.")
    try:
        packed = pack_batched_path_inputs(plan)
        first, second = _execute_arrays(plan, packed, backend=backend)
    except (TypeError, ValueError) as exc:
        _raise_path_query_execution_error(exc, owner=owner)
    if plan.finalization.output == "position":
        return commit_batched_position(plan, first)
    if second is None:
        raise ValueError("batched Pose execution did not produce quaternion output.")
    return commit_batched_pose(plan, first, second)


__all__ = ["execute_eager_batched_path"]
