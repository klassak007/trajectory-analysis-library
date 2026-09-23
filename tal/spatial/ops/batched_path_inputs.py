from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.runtime_checks import select_single_numeric_var
from tal.core.schema_read import read_roles

from .batched_path_plan import PreparedBatchedPathExecution
from .pose_component_ops import resolve_pose_component_specs


@dataclass(frozen=True)
class PackedPathMap:
    i0: np.ndarray
    i1: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class PackedPathProvider:
    translation: np.ndarray
    quaternion: np.ndarray
    batch_dims: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    position_map: int
    rotation_map: int


@dataclass(frozen=True)
class PackedBatchedPathInputs:
    providers: tuple[PackedPathProvider, ...]
    maps: tuple[PackedPathMap, ...]
    caller: np.ndarray | None
    caller_valid: np.ndarray | None
    caller_batch_dims: tuple[str, ...]
    caller_batch_sizes: tuple[int, ...]


def _row_view(
    value: xr.DataArray,
    dims: tuple[str, ...],
    sizes: tuple[int, ...],
) -> xr.DataArray:
    expanded = value
    for dim, size in zip(dims, sizes, strict=True):
        if dim not in expanded.dims:
            expanded = expanded.expand_dims({dim: size})
    return expanded.transpose(*dims)


def _materialize_map_columns(
    mapping,
    dims: tuple[str, ...],
    sizes: tuple[int, ...],
) -> tuple[xr.DataArray, ...]:
    names = ("i0", "i1", "alpha", "valid")
    columns = tuple(
        _row_view(value, dims, sizes)
        for value in (mapping.i0, mapping.i1, mapping.alpha, mapping.valid)
    )
    if not any(column.chunks is not None for column in columns):
        return columns
    computed = xr.Dataset(dict(zip(names, columns, strict=True))).compute()
    return tuple(computed[name] for name in names)


def _pack_map(evaluation, plan: PreparedBatchedPathExecution) -> PackedPathMap:
    mapping = evaluation.param_map
    topology = plan.query.topology
    if topology is None:
        raise ValueError("batched path packing requires query topology.")
    dims = plan.logical_rows.dims
    i0, i1, alpha, valid = _materialize_map_columns(
        mapping, dims, plan.logical_rows.sizes,
    )
    return PackedPathMap(
        np.asarray(i0.data).astype(np.int64, copy=False),
        np.asarray(i1.data).astype(np.int64, copy=False),
        np.asarray(alpha.data).astype(np.float64, copy=False),
        np.asarray(valid.data).astype(bool, copy=False),
    )


def _provider_arrays(
    item,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[int, ...]]:
    projection = item.native_projection or item.projection
    if projection is None:
        raise ValueError("batched path provider projection is missing.")
    ds = analysis_object_dataset(projection.value)
    _, sequence_dim, batch_dims, core_dims = read_roles(ds)
    (position_dim, position_var), (rotation_dim, rotation_var) = resolve_pose_component_specs(
        ds,
        owner="spatial.path_solve.pose",
        core_dims=core_dims,
    )
    translation = np.asarray(ds[position_var].transpose(*batch_dims, sequence_dim, position_dim).data)
    quaternion = np.asarray(ds[rotation_var].transpose(*batch_dims, sequence_dim, rotation_dim).data)
    batch_sizes = tuple(int(ds.sizes[dim]) for dim in batch_dims)
    batch_count = int(np.prod(batch_sizes, dtype=np.int64)) if batch_dims else 1
    translation = np.ascontiguousarray(
        translation.reshape(batch_count, translation.shape[-2], translation.shape[-1])
    )
    quaternion = np.ascontiguousarray(
        quaternion.reshape(batch_count, quaternion.shape[-2], quaternion.shape[-1])
    )
    return translation, quaternion, batch_dims, batch_sizes


def _map_index(
    evaluation,
    plan: PreparedBatchedPathExecution,
    indexes: dict[int, int],
    maps: list[PackedPathMap],
) -> int:
    key = id(evaluation)
    if key not in indexes:
        indexes[key] = len(maps)
        maps.append(_pack_map(evaluation, plan))
    return indexes[key]


def _pack_providers(
    plan: PreparedBatchedPathExecution,
) -> tuple[tuple[PackedPathProvider, ...], tuple[PackedPathMap, ...]]:
    maps: list[PackedPathMap] = []
    indexes: dict[int, int] = {}
    providers: list[PackedPathProvider] = []
    for item in plan.query.items:
        translation, quaternion, batch_dims, batch_sizes = _provider_arrays(item)
        position_map = _map_index(item.evaluations[0], plan, indexes, maps)
        rotation_map = _map_index(item.evaluations[-1], plan, indexes, maps)
        providers.append(
            PackedPathProvider(
                translation,
                quaternion,
                batch_dims,
                batch_sizes,
                position_map,
                rotation_map,
            )
        )
    return tuple(providers), tuple(maps)


def _pack_caller(
    plan: PreparedBatchedPathExecution,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[str, ...], tuple[int, ...]]:
    topology = plan.query.topology
    if plan.finalization.output != "position" or topology is None or topology.caller is None:
        return None, None, (), ()
    ds = topology.caller.ds
    name = select_single_numeric_var(ds, owner="spatial.path_solve.pose", what="Position caller")
    _, sequence_dim, batch_dims, core_dims = read_roles(ds)
    caller = np.asarray(ds[name].transpose(*batch_dims, sequence_dim, core_dims[0]).data)
    valid = np.asarray(topology.caller.valid_mask.transpose(*batch_dims, sequence_dim).data, dtype=bool)
    batch_count = int(np.prod(tuple(int(ds.sizes[dim]) for dim in batch_dims), dtype=np.int64))
    batch_count = batch_count if batch_dims else 1
    return (
        np.ascontiguousarray(caller.reshape(batch_count, caller.shape[-2], caller.shape[-1])),
        np.ascontiguousarray(valid.reshape(batch_count, valid.shape[-1])),
        batch_dims,
        tuple(int(ds.sizes[dim]) for dim in batch_dims),
    )


def pack_batched_path_inputs(plan: PreparedBatchedPathExecution) -> PackedBatchedPathInputs:
    """Expose eager provider/map buffers once without expanding query rows."""
    providers, maps = _pack_providers(plan)
    caller, caller_valid, caller_batch_dims, caller_batch_sizes = _pack_caller(plan)
    return PackedBatchedPathInputs(
        providers,
        maps,
        caller,
        caller_valid,
        caller_batch_dims,
        caller_batch_sizes,
    )


__all__ = [
    "PackedBatchedPathInputs",
    "PackedPathMap",
    "PackedPathProvider",
    "pack_batched_path_inputs",
]
