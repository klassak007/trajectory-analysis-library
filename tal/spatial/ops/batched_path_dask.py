"""Lazy, partition-local numerical execution for eligible spatial paths."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, pairwise
from uuid import uuid4

import dask.array as da
import numpy as np
import xarray as xr
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.runtime_checks import select_single_numeric_var
from tal.core.schema_read import read_roles
from tal.utils.numba_support import _numba_available

from ..kernels.path_kernel_status import PathKernelFailure, raise_path_kernel_failure
from ..kernels.rotation_interp_backends import slerp_quat_backend
from .batched_path_execution import (
    _apply_position_block,
    _compose,
    _earliest_failure,
    _empty_edge_block,
    _first_quaternion_failure,
    _first_status_failure,
    _map_block,
    _map_failure_status,
    _orient,
    _PoseBlock,
)
from .batched_path_finalize import commit_batched_pose, commit_batched_position
from .batched_path_inputs import PackedPathMap, pack_batched_path_maps
from .batched_path_plan import PreparedBatchedPathExecution
from .path_query_ops import _raise_path_query_execution_error
from .pose_component_ops import resolve_pose_component_specs


@dataclass(frozen=True)
class _PartitionResult:
    first: np.ndarray | None
    second: np.ndarray | None
    failure: PathKernelFailure | None


@dataclass(frozen=True)
class _Source:
    value: xr.DataArray
    blocks: object | None
    limits: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _Provider:
    translation: _Source
    quaternion: _Source
    batch_dims: tuple[str, ...]
    counts: tuple[int, int]
    invert: bool


def _local_batch_index(
    dims: tuple[str, ...],
    partition,
    target_dims: tuple[str, ...],
) -> np.ndarray:
    shape = partition.shape[:-1]
    index = np.zeros(shape, dtype=np.int64)
    sizes = tuple(shape[target_dims.index(dim)] for dim in dims)
    for offset, dim in enumerate(dims):
        axis = target_dims.index(dim)
        stride = int(np.prod(sizes[offset + 1 :], dtype=np.int64))
        values = np.arange(shape[axis], dtype=np.int64) * stride
        view = [1] * len(shape)
        view[axis] = shape[axis]
        index += values.reshape(view)
    return index


def _one_block(value: xr.DataArray):
    data = value.data
    if isinstance(data, np.ndarray):
        return data
    # Each selected piece is at most one physical row partition or native
    # provider sequence chunk. Only its small component axis may need joining.
    # Preserve shared upstream source keys across position and rotation pieces.
    # Per-piece optimization can fuse their common source independently.
    if all(blocks == 1 for blocks in data.numblocks):
        return data.to_delayed(optimize_graph=False).ravel()[0]
    return data.rechunk(data.shape).to_delayed(optimize_graph=False).ravel()[0]


def _source(value: xr.DataArray, *, core_axis: bool = True) -> _Source:
    data = value.data
    if not isinstance(data, da.Array):
        return _Source(value, None, ())
    if core_axis and data.numblocks[-1] != 1:
        data = data.rechunk({data.ndim - 1: data.shape[-1]})
    limits = tuple(tuple(accumulate((0, *chunks))) for chunks in data.chunks)
    return _Source(value, data.to_delayed(optimize_graph=False), limits)


def _eager_slice(value: xr.DataArray, selected: dict[str, slice]):
    """Slice already planned eager storage without rebuilding labeled wrappers."""
    return value.data[tuple(selected.get(dim, slice(None)) for dim in value.dims)]


def _source_chunks(
    source: _Source,
    mapping: PackedPathMap,
    active: np.ndarray,
    partition,
) -> tuple[tuple[int, object, tuple[slice, ...]], ...]:
    value = source.value
    batch_dims = value.dims[:-2]
    sequence_dim = value.dims[-2]
    selected = dict(partition.block.selections)
    batch = {dim: selected[dim] for dim in batch_dims}
    size = value.sizes[sequence_dim]
    if source.blocks is None:
        return ((0, _eager_slice(value, batch), ()),)
    batch_indices = []
    batch_slices = []
    for dim, limits in zip(batch_dims, source.limits[:-2], strict=True):
        selection = batch[dim]
        index = int(np.searchsorted(limits, selection.start, side="right") - 1)
        if selection.stop > limits[index + 1]:
            return _sliced_source_chunks(value, mapping, active, partition)
        batch_indices.append(index)
        batch_slices.append(slice(selection.start - limits[index], selection.stop - limits[index]))
    limits = source.limits[-2]
    indexes = np.concatenate((mapping.i0[active], mapping.i1[active]))
    indexes = indexes[(indexes >= 0) & (indexes < size)]
    pieces = []
    for sequence_index, (start, end) in enumerate(pairwise(limits)):
        if not np.any((indexes >= start) & (indexes < end)):
            continue
        key = (*batch_indices, sequence_index, 0)
        pieces.append((start, source.blocks[key], tuple(batch_slices)))
    return tuple(pieces)


def _sliced_source_chunks(value, mapping, active, partition):
    selected = dict(partition.block.selections)
    batch = {dim: selected[dim] for dim in value.dims[:-2]}
    sequence_dim = value.dims[-2]
    limits = (0, *accumulate(value.data.chunks[-2]))
    indexes = np.concatenate((mapping.i0[active], mapping.i1[active]))
    pieces = []
    for start, end in pairwise(limits):
        if not np.any((indexes >= start) & (indexes < end)):
            continue
        selection = {**batch, sequence_dim: slice(start, end)}
        pieces.append((start, _one_block(value.isel(selection)), ()))
    return tuple(pieces)


def _gather(
    pieces: tuple[tuple[int, np.ndarray, tuple[slice, ...]], ...],
    batch_index: np.ndarray,
    mapping: PackedPathMap,
    active: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    shape = mapping.valid.shape
    nrows = int(np.prod(shape, dtype=np.int64))
    dtype = np.asarray(pieces[0][1]).dtype if pieces else np.float64
    left = np.zeros((nrows, width), dtype=dtype)
    right = np.zeros((nrows, width), dtype=dtype)
    batches = np.broadcast_to(batch_index[..., None], shape).reshape(-1)
    enabled = active.reshape(-1)
    for start, payload, batch_slices in pieces:
        values = np.asarray(payload)
        if batch_slices:
            values = values[(*batch_slices, slice(None), slice(None))]
        values = values.reshape(-1, values.shape[-2], width)
        end = start + values.shape[1]
        for brackets, output in ((mapping.i0, left), (mapping.i1, right)):
            indexes = brackets.reshape(-1)
            rows = np.flatnonzero(enabled & (indexes >= start) & (indexes < end))
            output[rows] = values[batches[rows], indexes[rows] - start]
    return left.reshape(*shape, width), right.reshape(*shape, width)


def _interpolate(
    edge,
    backend: str,
) -> tuple[_PoseBlock | None, tuple[int, int] | None]:
    translation, quaternion, batch_index, position_map, rotation_map, counts, _ = edge
    valid = position_map.valid & rotation_map.valid
    if not np.any(valid):
        return _empty_edge_block(position_map), None
    status = _map_failure_status(position_map, rotation_map, counts)
    gather_valid = valid & ~status.reshape(valid.shape).astype(bool)
    left_t, right_t = _gather(translation, batch_index, position_map, gather_valid, 3)
    left_q, right_q = _gather(quaternion, batch_index, rotation_map, gather_valid, 4)
    failure = _earliest_failure(
        _first_status_failure(status),
        _first_quaternion_failure(left_q, right_q, gather_valid),
    )
    if failure is not None:
        return None, failure
    alpha_t = position_map.alpha[..., None]
    translated = (1.0 - alpha_t) * left_t + alpha_t * right_t
    quat = slerp_quat_backend(left_q, right_q, rotation_map.alpha, valid, backend=backend)
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))
    block = _PoseBlock(
        np.where(valid[..., None], translated, 0.0),
        np.where(valid[..., None], quat, identity),
        valid,
    )
    return block, None


def _numerical_partition(partition, edges, output, caller, caller_valid) -> _PartitionResult:
    backend = "numba" if _numba_available() else "scipy"
    accumulated = None
    for edge_index, edge in enumerate(edges):
        current, failure = _interpolate(edge, backend)
        if failure is not None:
            status, row = failure
            return _PartitionResult(None, None, PathKernelFailure(
                edge_index, partition.global_row_position(row), status,
            ))
        if current is None:
            raise RuntimeError("batched path worker lost an edge result")
        current = _orient(current, invert=edge[-1], backend=backend)
        accumulated = current if accumulated is None else _compose(accumulated, current, backend=backend)
    if accumulated is None:
        raise RuntimeError("batched path worker received no edges")
    if output == "position":
        applied = _apply_position_block(
            accumulated, _unpack_piece(caller), _unpack_piece(caller_valid), backend,
        )
        return _PartitionResult(applied, None, None)
    first = np.where(accumulated.valid[..., None], accumulated.translation, np.nan)
    second = np.where(accumulated.valid[..., None], accumulated.quaternion, np.nan)
    return _PartitionResult(first, second, None)


def _failure_gate(results: tuple[_PartitionResult, ...], owner: str) -> None:
    present = tuple(result.failure for result in results if result.failure is not None)
    if not present:
        return
    try:
        raise_path_kernel_failure(min(present, key=lambda failure: (failure.edge, failure.row)))
    except ValueError as exc:
        _raise_path_query_execution_error(exc, owner=owner)


def _output_block(result: _PartitionResult, gate: None, component: int) -> np.ndarray:
    _ = gate
    output = result.first if component == 0 else result.second
    if output is None:
        raise RuntimeError("batched path worker did not return the requested component")
    return output


def _caller_sources(plan: PreparedBatchedPathExecution):
    topology = plan.query.topology
    if topology is None or topology.caller is None:
        return None, None
    ds = topology.caller.ds
    name = select_single_numeric_var(ds, owner="spatial.path_solve.pose", what="Position caller")
    _, sequence, batch_dims, core_dims = read_roles(ds)
    values = ds[name].transpose(*batch_dims, sequence, core_dims[0])
    valid = topology.caller.valid_mask.transpose(*batch_dims, sequence)
    return _source(values), _source(valid, core_axis=False)


def _single_chunk(source: _Source, selected: dict[str, slice]):
    value = source.value
    if source.blocks is None:
        return _eager_slice(value, selected), ()
    indices = []
    slices = []
    for dim, limits in zip(value.dims, source.limits, strict=True):
        selection = selected.get(dim, slice(0, value.sizes[dim]))
        index = int(np.searchsorted(limits, selection.start, side="right") - 1)
        if selection.stop > limits[index + 1]:
            return _one_block(value.isel(selected)), ()
        indices.append(index)
        slices.append(slice(selection.start - limits[index], selection.stop - limits[index]))
    return source.blocks[tuple(indices)], tuple(slices)


def _unpack_piece(piece) -> np.ndarray:
    payload, slices = piece
    values = np.asarray(payload)
    return values[slices] if slices else values


def _caller_piece(sources, partition, query_dim):
    values, valid = sources
    if values is None or valid is None:
        return None, None
    selected = dict(partition.block.selections)
    sequence = values.value.dims[-2]
    selected[sequence] = selected.pop(query_dim)
    return _single_chunk(values, selected), _single_chunk(valid, selected)


def _provider(plan, index) -> _Provider:
    item = plan.query.items[index]
    projection = item.native_projection or item.projection
    if projection is None:
        raise ValueError("batched path provider projection is missing")
    ds = analysis_object_dataset(projection.value)
    _, sequence, batch_dims, core_dims = read_roles(ds)
    (position_dim, position_var), (rotation_dim, rotation_var) = resolve_pose_component_specs(
        ds, owner="spatial.path_solve.pose", core_dims=core_dims,
    )
    translation = ds[position_var].transpose(*batch_dims, sequence, position_dim)
    quaternion = ds[rotation_var].transpose(*batch_dims, sequence, rotation_dim)
    counts = (int(translation.sizes[sequence]), int(quaternion.sizes[sequence]))
    return _Provider(
        _source(translation), _source(quaternion), batch_dims,
        counts, plan.path.steps[index].invert,
    )


def _provider_edge(plan, provider, physical_index, physical, maps, pair):
    position_map = _map_block(maps[pair[0]], plan, index=physical_index)
    rotation_map = _map_block(maps[pair[1]], plan, index=physical_index)
    active = position_map.valid & rotation_map.valid
    batches = _local_batch_index(provider.batch_dims, physical, plan.finalization.batch_dims)
    sources = (
        _source_chunks(provider.translation, position_map, active, physical),
        _source_chunks(provider.quaternion, rotation_map, active, physical),
    )
    return (*sources, batches, position_map, rotation_map, provider.counts, provider.invert)


def _piece_dependency(piece, dependencies):
    """Expose a known chunk reference without traversing numerical metadata."""
    if piece is None:
        return None
    payload, slices = piece
    if isinstance(payload, Delayed):
        dependencies.setdefault((id(payload.dask), payload.__dask_layers__()), payload)
        payload = payload.key
    return [payload, slices]


def _edge_dependencies(edge, dependencies):
    translation, quaternion, *metadata = edge
    pieces = [
        [[start, *_piece_dependency((payload, slices), dependencies)]
         for start, payload, slices in component]
        for component in (translation, quaternion)
    ]
    return [*pieces, *metadata]


def _partition_graph(plan, *, owner):
    maps, pairs = pack_batched_path_maps(plan)
    providers = tuple(_provider(plan, edge) for edge in range(len(pairs)))
    output = plan.finalization.output
    caller_sources = _caller_sources(plan) if output == "position" else (None, None)
    name = f"tal-path-{uuid4().hex}"
    tasks, dependencies, results = {}, {}, []
    for index, partition in enumerate(plan.physical_rows.partitions):
        edges = [
            _edge_dependencies(_provider_edge(plan, provider, index, partition, maps, pair), dependencies)
            for provider, pair in zip(providers, pairs, strict=True)
        ]
        caller, valid = _caller_piece(caller_sources, partition, plan.finalization.query_dim)
        key = (name, index)
        tasks[key] = (
            _numerical_partition, partition, edges, output,
            _piece_dependency(caller, dependencies), _piece_dependency(valid, dependencies),
        )
        results.append(key)
    gate = (name, "failure")
    tasks[gate] = (_failure_gate, results, owner)
    graph = HighLevelGraph.from_collections(name, tasks, dependencies=tuple(dependencies.values()))
    return name, graph, results, gate


def _partition_arrays(plan, *, owner):
    name, graph, results, gate = _partition_graph(plan, owner=owner)
    widths = (3, 4) if plan.finalization.output == "pose" else (3,)
    chunks = tuple(
        tuple(part.stop - part.start for part in lane)
        for lane in plan.physical_rows.execution.chunks
    )
    arrays = []
    for component, width in enumerate(widths):
        output_name = f"{name}-{component}"
        tasks = {
            (output_name, *partition.block.ordinal, 0): (_output_block, result, gate, component)
            for partition, result in zip(plan.physical_rows.partitions, results, strict=True)
        }
        output_graph = HighLevelGraph(
            {**graph.layers, output_name: tasks},
            {**graph.dependencies, output_name: {name}},
        )
        arrays.append(da.Array(output_graph, output_name, (*chunks, (width,)), dtype=np.float64))
    return arrays[0], arrays[1] if len(arrays) == 2 else None


def _coalesce_caller_batch_chunks(plan, values: da.Array) -> da.Array:
    topology = plan.query.topology
    if topology is None or topology.caller is None:
        return values
    ds = topology.caller.ds
    name = select_single_numeric_var(ds, owner="spatial.path_solve.pose", what="Position caller")
    source = ds[name]
    if source.chunks is None:
        return values
    targets = {
        axis: source.chunks[source.get_axis_num(dim)]
        for axis, dim in enumerate(plan.finalization.batch_dims)
        if dim in source.dims and values.chunks[axis] != source.chunks[source.get_axis_num(dim)]
    }
    return values.rechunk(targets) if targets else values


def execute_dask_batched_path(plan: PreparedBatchedPathExecution, *, owner: str):
    """Construct one shared worker task per physical partition and commit once."""
    if not plan.eligible or plan.storage != "dask":
        raise ValueError("batched Dask executor requires an eligible Dask plan")
    output = plan.finalization.output
    shape = plan.logical_rows.sizes
    if plan.physical_rows.has_no_rows:
        chunks = tuple(max(1, size) for size in shape)
        first = da.full((*shape, 3), np.nan, chunks=(*chunks, 3))
        second = da.full((*shape, 4), np.nan, chunks=(*chunks, 4)) if output == "pose" else None
    else:
        first, second = _partition_arrays(plan, owner=owner)
    if output == "position":
        first = _coalesce_caller_batch_chunks(plan, first)
        return commit_batched_position(plan, first)
    if second is None:
        raise ValueError("batched Pose execution did not produce quaternion output")
    return commit_batched_pose(plan, first, second)


__all__ = ["execute_dask_batched_path"]
