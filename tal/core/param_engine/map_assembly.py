from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
import xarray as xr

from ..orchestration.indexing import (
    ResultCoordinateSnapshot,
    capture_result_coordinates,
    restore_result_coordinates,
    without_index_topology,
)
from .blocking import (
    LogicalRowBlock,
    LogicalRowBlockPlan,
    assemble_logical_blocks,
    project_logical_row_plan,
    without_logical_scalar_collisions,
)
from .map_failures import (
    attach_map_failure_dependency,
    first_map_failure,
    map_failure_dependency,
    raise_ordered_map_failures,
)

_COLUMN_DTYPES = (np.int64, np.int64, np.float64, bool)


def _map_outer_source(value: xr.DataArray, *, sequence_dim: str) -> xr.DataArray:
    dependent = tuple(name for name, coord in value.coords.items() if sequence_dim in coord.dims)
    value = value.drop_vars(dependent)
    if int(value.sizes[sequence_dim]):
        return value.isel({sequence_dim: 0}, drop=True)
    dims = tuple(dim for dim in value.dims if dim != sequence_dim)
    shape = tuple(int(value.sizes[dim]) for dim in dims)
    return xr.DataArray(np.broadcast_to(np.empty((), dtype=value.dtype), shape), dims=dims, coords=dict(value.coords))


def _map_output_template(
    param: xr.DataArray,
    mask: xr.DataArray,
    query: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
    logical_dims: tuple[str, ...],
) -> xr.DataArray:
    inputs = tuple(
        without_logical_scalar_collisions(without_index_topology(value, dims=logical_dims), logical_dims)
        for value in (param, mask, query)
    )
    sources = (_map_outer_source(inputs[0], sequence_dim=sequence_dim), _map_outer_source(inputs[1], sequence_dim=sequence_dim))
    template = xr.broadcast(*sources, inputs[2])[0]
    dims = [dim for dim in param.dims if dim != sequence_dim]
    dims.extend(dim for dim in mask.dims if dim != sequence_dim and dim not in dims)
    dims.extend(dim for dim in query.dims if dim != query_dim and dim not in dims)
    dims.append(query_dim)
    ordered = template.transpose(*dims)
    data = np.broadcast_to(np.empty((), dtype=np.int64), ordered.shape)
    return ordered.copy(data=data)


def _assemble_map_columns(
    blocks: tuple[tuple[xr.DataArray, ...], ...],
    *,
    plan: LogicalRowBlockPlan,
    template: xr.DataArray,
    sources: tuple[xr.DataArray, ...],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    return tuple(
        assemble_logical_blocks(
            (block[index] for block in blocks),
            plan=plan,
            template=template.astype(blocks[0][index].dtype),
            index_sources=sources,
            owner="build_param_map",
        )
        for index in range(4)
    )  # type: ignore[return-value]


def _outer_failure_blocks(
    blocks: tuple[tuple[xr.DataArray, ...], ...],
    *,
    plan: LogicalRowBlockPlan,
    query_dim: str,
) -> tuple[tuple[xr.DataArray, ...], ...]:
    query_axis = plan.dims.index(query_dim)
    grouped: dict[tuple[int, ...], list[tuple[int, tuple[xr.DataArray, ...]]]] = {}
    for spec, result in zip(plan.blocks, blocks, strict=True):
        key = tuple(value for index, value in enumerate(spec.ordinal) if index != query_axis)
        grouped.setdefault(key, []).append((spec.selection_for_dim(query_dim).start or 0, result))
    selected = tuple(
        first_map_failure(
            tuple(result[4] for _, result in grouped[key]),
            tuple(result[5] for _, result in grouped[key]),
            tuple(result[6] for _, result in grouped[key]),
            starts=tuple(start for start, _ in grouped[key]),
        )
        for key in sorted(grouped)
    )
    return tuple(tuple(item[index] for item in selected) for index in range(3))


def _map_failure_for_blocks(
    blocks: tuple[tuple[xr.DataArray, ...], ...],
    *,
    plan: LogicalRowBlockPlan,
    template: xr.DataArray,
    query_dim: str,
    sources: tuple[xr.DataArray, ...],
) -> xr.DataArray:
    outer_plan = project_logical_row_plan(plan, drop_dims=frozenset({query_dim}))
    status_blocks, position_blocks, detail_blocks = _outer_failure_blocks(blocks, plan=plan, query_dim=query_dim)
    outer_template = template.isel({query_dim: 0}, drop=True)
    assembled = tuple(
        assemble_logical_blocks(
            items,
            plan=outer_plan,
            template=outer_template.astype(dtype),
            index_sources=sources,
            owner="build_param_map",
        )
        for items, dtype in zip(
            (status_blocks, position_blocks, detail_blocks),
            (np.int8, np.int64, object),
            strict=True,
        )
    )
    return map_failure_dependency((assembled[0],), (assembled[1],), (assembled[2],), starts=(0,))


def _target_slices(template: xr.DataArray, block: LogicalRowBlock) -> tuple[slice, ...]:
    selected = dict(block.selections)
    return tuple(selected.get(dim, slice(None)) for dim in template.dims)


def _outer_slices(
    template: xr.DataArray,
    block: LogicalRowBlock,
    *,
    query_dim: str,
) -> tuple[slice, ...]:
    selected = dict(block.selections)
    return tuple(selected.get(dim, slice(None)) for dim in template.dims if dim != query_dim)


def _update_eager_failure(
    state: tuple[np.ndarray, np.ndarray, np.ndarray],
    result: tuple[xr.DataArray, ...],
    target: tuple[slice, ...],
    *,
    query_start: int,
) -> None:
    status, position, detail = (np.asarray(result[index].data) for index in range(4, 7))
    position = np.where(position >= 0, position + query_start, position)
    current_status, current_position, current_detail = (value[target] for value in state)
    replace = (status != 0) & ((current_status == 0) | (position < current_position))
    state[0][target] = np.where(replace, status, current_status)
    state[1][target] = np.where(replace, position, current_position)
    state[2][target] = np.where(replace, detail, current_detail)


def _restore_eager_columns(
    arrays: tuple[np.ndarray, ...],
    template: xr.DataArray,
    snapshot: ResultCoordinateSnapshot,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    return tuple(
        restore_result_coordinates(template.astype(dtype).copy(data=array), snapshot).rename(None)
        for array, dtype in zip(arrays, _COLUMN_DTYPES, strict=True)
    )  # type: ignore[return-value]


def _assemble_eager_map(
    first: tuple[xr.DataArray, ...],
    remaining: Iterator[tuple[xr.DataArray, ...]],
    *,
    plan: LogicalRowBlockPlan,
    template: xr.DataArray,
    query_dim: str,
    snapshot: ResultCoordinateSnapshot,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    arrays = tuple(np.empty(template.shape, dtype=dtype) for dtype in _COLUMN_DTYPES)
    outer_shape = tuple(template.sizes[dim] for dim in template.dims if dim != query_dim)
    failure = (np.zeros(outer_shape, np.int8), np.full(outer_shape, -1, np.int64), np.zeros(outer_shape, object))
    for spec, result in zip(plan.blocks, chain((first,), remaining), strict=True):
        target = _target_slices(template, spec)
        for index in range(4):
            arrays[index][target] = result[index].data
        _update_eager_failure(
            failure,
            result,
            _outer_slices(template, spec, query_dim=query_dim),
            query_start=spec.selection_for_dim(query_dim).start or 0,
        )
    raise_ordered_map_failures(*failure)
    return _restore_eager_columns(arrays, template, snapshot)


def assemble_param_map_blocks(
    blocks: Iterable[tuple[xr.DataArray, ...]],
    *,
    plan: LogicalRowBlockPlan,
    param: xr.DataArray,
    mask: xr.DataArray,
    query: xr.DataArray,
    sequence_dim: str,
    query_dim: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    template = _map_output_template(
        param,
        mask,
        query,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        logical_dims=plan.dims,
    )
    sources = (param, mask, query)
    snapshot = capture_result_coordinates(
        *sources,
        output_dims=tuple(template.dims),
        owner="build_param_map",
    )
    iterator = iter(blocks)
    first = next(iterator)
    if all(value.chunks is None for value in first):
        return _assemble_eager_map(
            first,
            iterator,
            plan=plan,
            template=template,
            query_dim=query_dim,
            snapshot=snapshot,
        )
    lazy_blocks = tuple(chain((first,), iterator))
    columns = _assemble_map_columns(lazy_blocks, plan=plan, template=template, sources=sources)
    dependency = _map_failure_for_blocks(
        lazy_blocks,
        plan=plan,
        template=template,
        query_dim=query_dim,
        sources=sources,
    )
    return attach_map_failure_dependency(columns, dependency)


__all__ = ["assemble_param_map_blocks"]
