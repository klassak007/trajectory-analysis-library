"""Benchmark-local packed comparator for eager batched path execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tal.spatial.kernels.batched_pose_path_numba import (
    batched_pose_edge_numba,
    batched_position_apply_numba,
)
from tal.spatial.kernels.fixed_size_backends import (
    SPATIAL_FIXED_BACKEND_NUMBA,
    SPATIAL_FIXED_BACKEND_SCIPY,
    pose_compose_translation_block_backend,
    pose_inverse_translation_block_backend,
    quat_compose_block_backend,
    quat_inverse_block_backend,
)
from tal.spatial.kernels.path_kernel_status import (
    PathKernelFailure,
    raise_path_kernel_failure,
)
from tal.spatial.kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_NUMBA,
    ROTATION_INTERP_BACKEND_SCIPY,
    slerp_quat_backend,
)
from tal.spatial.kernels.rotation_interp_reference import half_turn_tolerance
from tal.spatial.ops.batched_path_inputs import (
    PackedBatchedPathInputs,
    PackedPathMap,
    PackedPathProvider,
)
from tal.spatial.ops.batched_path_plan import PreparedBatchedPathExecution

_Backend = Literal["numba", "scipy"]


@dataclass(frozen=True)
class _DirectPoseBlock:
    translation: np.ndarray
    quaternion: np.ndarray
    valid: np.ndarray


def _partition_slices(
    plan: PreparedBatchedPathExecution,
    index: int,
) -> tuple[slice, ...]:
    selections = dict(plan.physical_rows.partitions[index].block.selections)
    return tuple(selections[dim] for dim in plan.logical_rows.dims)


def _map_block(
    mapping: PackedPathMap,
    plan: PreparedBatchedPathExecution,
    index: int,
) -> PackedPathMap:
    selected = _partition_slices(plan, index)
    return PackedPathMap(
        mapping.i0[selected],
        mapping.i1[selected],
        mapping.alpha[selected],
        mapping.valid[selected],
    )


def _batch_index(
    dims: tuple[str, ...],
    sizes: tuple[int, ...],
    plan: PreparedBatchedPathExecution,
    index: int,
) -> np.ndarray:
    partition = plan.physical_rows.partitions[index]
    target_shape = tuple(partition.shape[:-1])
    origins = partition.origin[:-1]
    result = np.zeros(target_shape, dtype=np.int64)
    for source_axis, (dim, size) in enumerate(zip(dims, sizes, strict=True)):
        target_axis = plan.finalization.batch_dims.index(dim)
        stride = int(np.prod(sizes[source_axis + 1 :], dtype=np.int64))
        rows = np.arange(
            origins[target_axis], origins[target_axis] + target_shape[target_axis],
            dtype=np.int64,
        )
        shape = [1] * len(target_shape)
        shape[target_axis] = target_shape[target_axis]
        result += rows.reshape(shape) * stride
    return result


def _gather(
    values: np.ndarray,
    batch_index: np.ndarray,
    mapping: PackedPathMap,
) -> tuple[np.ndarray, np.ndarray]:
    outer = int(np.prod(mapping.alpha.shape[:-1], dtype=np.int64))
    query = int(mapping.alpha.shape[-1])
    width = int(values.shape[-1])
    source = np.broadcast_to(batch_index[..., None], mapping.alpha.shape).reshape(-1)
    left = mapping.i0.reshape(-1)
    right = mapping.i1.reshape(-1)
    return (
        values[source, left].reshape(outer, query, width),
        values[source, right].reshape(outer, query, width),
    )


def _backend_names(backend: _Backend) -> tuple[str, str]:
    if backend == "numba":
        return ROTATION_INTERP_BACKEND_NUMBA, SPATIAL_FIXED_BACKEND_NUMBA
    return ROTATION_INTERP_BACKEND_SCIPY, SPATIAL_FIXED_BACKEND_SCIPY


def _sample_edge(
    provider: PackedPathProvider,
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    index: int,
    backend: _Backend,
) -> _DirectPoseBlock:
    batch = _batch_index(provider.batch_dims, provider.batch_sizes, plan, index)
    position = _map_block(packed.maps[provider.position_map], plan, index)
    rotation = _map_block(packed.maps[provider.rotation_map], plan, index)
    left_t, right_t = _gather(provider.translation, batch, position)
    left_q, right_q = _gather(provider.quaternion, batch, rotation)
    outer, query = left_t.shape[:2]
    position_alpha = position.alpha.reshape(outer, query)
    rotation_alpha = rotation.alpha.reshape(outer, query)
    valid = position.valid & rotation.valid
    translation = (1.0 - position_alpha[..., None]) * left_t + position_alpha[..., None] * right_t
    rotation_backend, _ = _backend_names(backend)
    quaternion = slerp_quat_backend(
        left_q, right_q, rotation_alpha, valid.reshape(outer, query),
        backend=rotation_backend,
    )
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))
    return _DirectPoseBlock(
        np.where(valid[..., None], translation.reshape((*valid.shape, 3)), 0.0),
        np.where(valid[..., None], quaternion.reshape((*valid.shape, 4)), identity),
        valid,
    )


def _orient(
    block: _DirectPoseBlock,
    *,
    invert: bool,
    backend: _Backend,
) -> _DirectPoseBlock:
    if not invert:
        return block
    _, fixed_backend = _backend_names(backend)
    return _DirectPoseBlock(
        pose_inverse_translation_block_backend(
            block.translation, block.quaternion, backend=fixed_backend,
        ),
        quat_inverse_block_backend(block.quaternion, backend=fixed_backend),
        block.valid,
    )


def _compose(
    left: _DirectPoseBlock,
    right: _DirectPoseBlock,
    *,
    backend: _Backend,
) -> _DirectPoseBlock:
    _, fixed_backend = _backend_names(backend)
    valid = left.valid & right.valid
    translation = pose_compose_translation_block_backend(
        left.translation, right.translation, right.quaternion,
        backend=fixed_backend,
    )
    quaternion = quat_compose_block_backend(
        left.quaternion, right.quaternion, backend=fixed_backend,
    )
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))
    return _DirectPoseBlock(
        np.where(valid[..., None], translation, 0.0),
        np.where(valid[..., None], quaternion, identity),
        valid,
    )


def _scipy_partition(
    plan: PreparedBatchedPathExecution,
    packed: PackedBatchedPathInputs,
    index: int,
) -> _DirectPoseBlock:
    result: _DirectPoseBlock | None = None
    for edge, provider in enumerate(packed.providers):
        current = _sample_edge(provider, packed, plan, index, "scipy")
        current = _orient(current, invert=plan.path.steps[edge].invert, backend="scipy")
        result = current if result is None else _compose(result, current, backend="scipy")
    if result is None:
        raise ValueError("packed direct comparator requires at least one path edge.")
    return result


def _flat_map(mapping: PackedPathMap) -> tuple[np.ndarray, ...]:
    return (
        np.ascontiguousarray(mapping.i0.reshape(-1), dtype=np.int64),
        np.ascontiguousarray(mapping.i1.reshape(-1), dtype=np.int64),
        np.ascontiguousarray(mapping.alpha.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(mapping.valid.reshape(-1), dtype=np.bool_),
    )


def _numba_edge(
    block: _DirectPoseBlock,
    provider: PackedPathProvider,
    packed: PackedBatchedPathInputs,
    plan: PreparedBatchedPathExecution,
    edge: int,
    index: int,
) -> bool:
    batch = _batch_index(provider.batch_dims, provider.batch_sizes, plan, index)
    position = _flat_map(_map_block(packed.maps[provider.position_map], plan, index))
    rotation = _flat_map(_map_block(packed.maps[provider.rotation_map], plan, index))
    status, row, ambiguous = batched_pose_edge_numba(
        provider.translation,
        provider.quaternion,
        np.ascontiguousarray(batch.reshape(-1)),
        position,
        rotation,
        -1 if plan.path.steps[edge].invert else 1,
        half_turn_tolerance((provider.quaternion.dtype,)),
        (block.translation, block.quaternion, block.valid),
        edge == 0,
    )
    if status:
        partition = plan.physical_rows.partitions[index]
        raise_path_kernel_failure(
            PathKernelFailure(edge, partition.global_row_position(row), status)
        )
    return bool(ambiguous)


def _numba_partition(
    plan: PreparedBatchedPathExecution,
    packed: PackedBatchedPathInputs,
    index: int,
) -> _DirectPoseBlock:
    shape = plan.physical_rows.partitions[index].shape
    rows = int(np.prod(shape, dtype=np.int64))
    translation = np.zeros((rows, 3), dtype=np.float64)
    quaternion = np.zeros((rows, 4), dtype=np.float64)
    quaternion[:, 3] = 1.0
    block = _DirectPoseBlock(translation, quaternion, np.zeros(rows, dtype=np.bool_))
    ambiguous = False
    for edge, provider in enumerate(packed.providers):
        ambiguous = _numba_edge(block, provider, packed, plan, edge, index) or ambiguous
    if ambiguous:
        return _scipy_partition(plan, packed, index)
    return _DirectPoseBlock(
        translation.reshape(*shape, 3),
        quaternion.reshape(*shape, 4),
        block.valid.reshape(shape),
    )


def _caller_block(
    plan: PreparedBatchedPathExecution,
    packed: PackedBatchedPathInputs,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if packed.caller is None or packed.caller_valid is None:
        raise ValueError("packed direct Position comparator requires a caller.")
    partition = plan.physical_rows.partitions[index]
    query_slice = partition.block.selection_for_dim(plan.finalization.query_dim)
    batch = _batch_index(
        packed.caller_batch_dims, packed.caller_batch_sizes, plan, index,
    )
    return packed.caller[..., query_slice, :][batch], packed.caller_valid[..., query_slice][batch]


def _apply_position(
    pose: _DirectPoseBlock,
    values: np.ndarray,
    valid: np.ndarray,
    backend: _Backend,
) -> np.ndarray:
    active = pose.valid & valid
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if backend == "numba":
        batched_position_apply_numba(
            np.ascontiguousarray(values.reshape(-1, 3)),
            pose.translation.reshape(-1, 3),
            pose.quaternion.reshape(-1, 4),
            pose.valid.reshape(-1),
            valid.reshape(-1),
            output.reshape(-1, 3),
        )
        return output
    transformed = pose_compose_translation_block_backend(
        values, pose.translation, pose.quaternion,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    return np.where(active[..., None], transformed, np.nan)


def execute_direct_packed(
    plan: PreparedBatchedPathExecution,
    packed: PackedBatchedPathInputs,
    *,
    backend: _Backend,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Execute a comparator from production-equivalent packed inputs."""
    shape = plan.logical_rows.sizes
    position_output = plan.finalization.output == "position"
    first = np.full((*shape, 3), np.nan, dtype=np.float64)
    second = None if position_output else np.full((*shape, 4), np.nan, dtype=np.float64)
    for index in range(len(plan.physical_rows.partitions)):
        pose = (
            _numba_partition(plan, packed, index)
            if backend == "numba"
            else _scipy_partition(plan, packed, index)
        )
        target = _partition_slices(plan, index)
        if second is not None:
            first[target] = np.where(pose.valid[..., None], pose.translation, np.nan)
            second[target] = np.where(pose.valid[..., None], pose.quaternion, np.nan)
            continue
        values, valid = _caller_block(plan, packed, index)
        first[target] = _apply_position(pose, values, valid, backend)
    return first, second


__all__ = ["execute_direct_packed"]
