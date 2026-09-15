from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain, product

import numpy as np
import xarray as xr

from ..orchestration.indexing import (
    capture_result_coordinates,
    restore_result_coordinates,
    without_index_topology,
)

PARAM_LOGICAL_ROW_LIMIT = 65_536


@dataclass(frozen=True)
class LogicalRowBlock:
    """One rectangular selection in public logical-row order."""

    ordinal: tuple[int, ...]
    selections: tuple[tuple[str, slice], ...]

    def for_value(self, value: xr.DataArray) -> dict[str, slice]:
        return {dim: selection for dim, selection in self.selections if dim in value.dims}

    def selection_for_dim(self, dim: str) -> slice:
        return dict(self.selections)[dim]


@dataclass(frozen=True)
class LogicalRowBlockPlan:
    """Immutable bounded partition of named logical-row dimensions."""

    dims: tuple[str, ...]
    sizes: tuple[int, ...]
    chunks: tuple[tuple[slice, ...], ...]
    blocks: tuple[LogicalRowBlock, ...]

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(len(parts) for parts in self.chunks)


def _ordered_logical_dims(
    values: tuple[xr.DataArray, ...],
    *,
    excluded_dims: frozenset[str],
    included_dims: tuple[str, ...] | None,
    fastest_dim: str,
) -> tuple[str, ...]:
    allowed = None if included_dims is None else set(included_dims)
    dims: list[str] = []
    for value in values:
        candidates = _eligible_logical_dims(
            value,
            excluded_dims=excluded_dims,
            allowed=allowed,
        )
        _append_unique_dims(dims, candidates)
    if fastest_dim in dims:
        dims.remove(fastest_dim)
        dims.append(fastest_dim)
    return tuple(dims)


def _append_unique_dims(target: list[str], candidates: tuple[str, ...]) -> None:
    for dim in candidates:
        if dim not in target:
            target.append(dim)


def _eligible_logical_dims(
    value: xr.DataArray,
    *,
    excluded_dims: frozenset[str],
    allowed: set[str] | None,
) -> tuple[str, ...]:
    return tuple(
        dim
        for dim in value.dims
        if dim not in excluded_dims and (allowed is None or dim in allowed)
    )


def _logical_sizes(values: tuple[xr.DataArray, ...], dims: tuple[str, ...]) -> tuple[int, ...]:
    sizes: list[int] = []
    for dim in dims:
        candidates = {int(value.sizes[dim]) for value in values if dim in value.dims}
        if len(candidates) != 1:
            raise ValueError(f"parameter blocking: dimension {dim!r} has incompatible sizes.")
        sizes.append(candidates.pop())
    return tuple(sizes)


def _chunk_lengths(sizes: tuple[int, ...], *, max_rows: int) -> tuple[int, ...]:
    remaining = max_rows
    reversed_lengths: list[int] = []
    for size in reversed(sizes):
        length = min(max(size, 1), max(remaining, 1))
        reversed_lengths.append(length)
        remaining = max(1, remaining // length)
    return tuple(reversed(reversed_lengths))


def _slices(size: int, length: int) -> tuple[slice, ...]:
    if size == 0:
        return (slice(0, 0),)
    return tuple(slice(start, min(start + length, size)) for start in range(0, size, length))


def prepare_logical_row_blocks(
    *values: xr.DataArray,
    excluded_dims: frozenset[str],
    fastest_dim: str,
    included_dims: tuple[str, ...] | None = None,
    max_rows: int = PARAM_LOGICAL_ROW_LIMIT,
) -> LogicalRowBlockPlan:
    """Partition combined named rows without inspecting array payloads."""
    if max_rows <= 0:
        raise ValueError("parameter blocking: max_rows must be positive.")
    items = tuple(values)
    dims = _ordered_logical_dims(
        items,
        excluded_dims=excluded_dims,
        included_dims=included_dims,
        fastest_dim=fastest_dim,
    )
    sizes = _logical_sizes(items, dims)
    lengths = _chunk_lengths(sizes, max_rows=max_rows)
    chunks = tuple(_slices(size, length) for size, length in zip(sizes, lengths, strict=True))
    blocks = tuple(
        LogicalRowBlock(ordinal, tuple(zip(dims, selections, strict=True)))
        for ordinal, selections in zip(
            product(*(range(len(parts)) for parts in chunks)),
            product(*chunks),
            strict=True,
        )
    )
    return LogicalRowBlockPlan(dims, sizes, chunks, blocks)


def project_logical_row_plan(
    plan: LogicalRowBlockPlan,
    *,
    drop_dims: frozenset[str],
) -> LogicalRowBlockPlan:
    """Project an existing block grid without changing retained partitions."""
    retained = tuple(index for index, dim in enumerate(plan.dims) if dim not in drop_dims)
    dims = tuple(plan.dims[index] for index in retained)
    sizes = tuple(plan.sizes[index] for index in retained)
    chunks = tuple(plan.chunks[index] for index in retained)
    blocks = tuple(
        LogicalRowBlock(ordinal, tuple(zip(dims, selections, strict=True)))
        for ordinal, selections in zip(
            product(*(range(len(parts)) for parts in chunks)),
            product(*chunks),
            strict=True,
        )
    )
    return LogicalRowBlockPlan(dims, sizes, chunks, blocks)


def without_logical_scalar_collisions(
    value: xr.DataArray,
    dims: tuple[str, ...],
) -> xr.DataArray:
    names = tuple(
        dim
        for dim in dims
        if dim not in value.dims and dim in value.coords and not value.coords[dim].dims
    )
    return value.drop_vars(names) if names else value


def select_logical_block(value: xr.DataArray, block: LogicalRowBlock) -> xr.DataArray:
    """Select one block only along dimensions present on ``value``."""
    value = without_logical_scalar_collisions(
        value,
        tuple(dim for dim, _ in block.selections),
    )
    selection = block.for_value(value)
    if not selection:
        return value
    unindexed = without_index_topology(value, dims=tuple(selection))
    return unindexed.isel(selection)  # type: ignore[return-value]


def _combine_block_data(
    arrays: tuple[object, ...],
    *,
    plan: LogicalRowBlockPlan,
    output_dims: tuple[str, ...],
) -> object:
    iterator = iter(arrays)

    def combine(level: int) -> object:
        if level == len(plan.dims):
            return next(iterator)
        children = tuple(combine(level + 1) for _ in plan.chunks[level])
        if len(children) == 1:
            return children[0]
        return np.concatenate(children, axis=output_dims.index(plan.dims[level]))

    combined = combine(0)
    try:
        next(iterator)
    except StopIteration:
        return combined
    raise ValueError("parameter blocking: block result count does not match the plan.")


def _assemble_eager_block_data(
    first: xr.DataArray,
    remaining: Iterable[xr.DataArray],
    *,
    plan: LogicalRowBlockPlan,
    template: xr.DataArray,
) -> np.ndarray:
    output = np.empty(template.shape, dtype=template.dtype)
    blocks = chain((first,), remaining)
    for spec, block in zip(plan.blocks, blocks, strict=True):
        selected = dict(spec.selections)
        target = tuple(selected.get(dim, slice(None)) for dim in template.dims)
        output[target] = block.data
    return output


def assemble_logical_blocks(
    blocks: Iterable[xr.DataArray],
    *,
    plan: LogicalRowBlockPlan,
    template: xr.DataArray,
    index_sources: tuple[xr.DataArray, ...] = (),
    owner: str = "parameter blocking",
) -> xr.DataArray:
    """Assemble raw block arrays under one full-topology xarray template."""
    sources = index_sources or (template,)
    snapshot = capture_result_coordinates(
        *sources,
        output_dims=tuple(template.dims),
        owner=owner,
    )
    iterator = iter(blocks)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("parameter blocking: block result count does not match the plan.")
    if first.chunks is None:
        data = _assemble_eager_block_data(first, iterator, plan=plan, template=template)
    else:
        lazy_blocks = tuple(chain((first,), iterator))
        if len(lazy_blocks) != len(plan.blocks):
            raise ValueError("parameter blocking: block result count does not match the plan.")
        data = _combine_block_data(
            tuple(block.data for block in lazy_blocks),
            plan=plan,
            output_dims=tuple(template.dims),
        )
    assembled = template.copy(data=data)
    return restore_result_coordinates(assembled, snapshot)  # type: ignore[return-value]


__all__ = [
    "PARAM_LOGICAL_ROW_LIMIT",
    "LogicalRowBlock",
    "LogicalRowBlockPlan",
    "assemble_logical_blocks",
    "prepare_logical_row_blocks",
    "project_logical_row_plan",
    "select_logical_block",
    "without_logical_scalar_collisions",
]
