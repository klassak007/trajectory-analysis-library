from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, pairwise, product
from math import prod

import xarray as xr

from ..orchestration.lazy import payload_chunks_for_dim
from .blocking import LogicalRowBlock, LogicalRowBlockPlan


@dataclass(frozen=True)
class PhysicalRowPartition:
    """One native-chunk-aware subdivision of a logical row block."""

    block: LogicalRowBlock
    logical_ordinal: tuple[int, ...]
    origin: tuple[int, ...]
    shape: tuple[int, ...]
    row_strides: tuple[int, ...]

    @property
    def row_count(self) -> int:
        return int(prod(self.shape))

    def global_row_position(self, local_position: int) -> int:
        """Map a local row-major position to the public logical row position."""
        if local_position < 0 or local_position >= self.row_count:
            raise IndexError("physical row partition: local position is out of bounds.")
        local = _unravel_position(local_position, self.shape)
        return sum(
            (start + offset) * stride
            for start, offset, stride in zip(self.origin, local, self.row_strides, strict=True)
        )


@dataclass(frozen=True)
class PhysicalRowPartitionPlan:
    """Immutable physical subdivision retaining the public logical plan."""

    logical: LogicalRowBlockPlan
    execution: LogicalRowBlockPlan
    partitions: tuple[PhysicalRowPartition, ...]

    @property
    def has_no_rows(self) -> bool:
        return self.logical.has_no_rows


def _unravel_position(position: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0] * len(shape)
    remaining = position
    for index in range(len(shape) - 1, -1, -1):
        offsets[index] = remaining % shape[index]
        remaining //= shape[index]
    return tuple(offsets)


def _row_strides(sizes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(prod(sizes[index + 1 :]) for index in range(len(sizes)))


def _chunk_boundaries(chunks: tuple[int, ...], *, size: int, dim: str) -> tuple[int, ...]:
    if sum(chunks) != size or any(chunk <= 0 for chunk in chunks):
        raise ValueError(f"physical row partition: invalid chunks for dimension {dim!r}.")
    return tuple(accumulate(chunks[:-1]))


def _logical_boundaries(plan: LogicalRowBlockPlan, index: int) -> set[int]:
    boundaries = {0, plan.sizes[index]}
    for selection in plan.chunks[index]:
        boundaries.update((int(selection.start or 0), int(selection.stop or 0)))
    return boundaries


def _physical_chunks(
    plan: LogicalRowBlockPlan,
    values: tuple[xr.DataArray, ...],
) -> tuple[tuple[slice, ...], ...]:
    return tuple(
        _physical_chunks_for_dim(plan, index, dim, size, values)
        for index, (dim, size) in enumerate(zip(plan.dims, plan.sizes, strict=True))
    )


def _physical_chunks_for_dim(
    plan: LogicalRowBlockPlan,
    index: int,
    dim: str,
    size: int,
    values: tuple[xr.DataArray, ...],
) -> tuple[slice, ...]:
    boundaries = _logical_boundaries(plan, index)
    for value in values:
        chunks = payload_chunks_for_dim(value, dim=dim)
        if chunks is not None:
            boundaries.update(_chunk_boundaries(chunks, size=size, dim=dim))
    ordered = sorted(boundaries)
    return tuple(slice(left, right) for left, right in pairwise(ordered))


def _logical_ordinal(plan: LogicalRowBlockPlan, origin: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        next(
            index
            for index, selection in enumerate(parts)
            if int(selection.start or 0) <= start < int(selection.stop or 0)
        )
        for start, parts in zip(origin, plan.chunks, strict=True)
    )


def _partition(
    plan: LogicalRowBlockPlan,
    ordinal: tuple[int, ...],
    selections: tuple[slice, ...],
) -> PhysicalRowPartition:
    origin = tuple(int(selection.start or 0) for selection in selections)
    shape = tuple(int(selection.stop or 0) - start for selection, start in zip(selections, origin, strict=True))
    block = LogicalRowBlock(ordinal, tuple(zip(plan.dims, selections, strict=True)))
    return PhysicalRowPartition(
        block,
        _logical_ordinal(plan, origin),
        origin,
        shape,
        _row_strides(plan.sizes),
    )


def prepare_physical_row_partitions(
    plan: LogicalRowBlockPlan,
    *values: xr.DataArray,
) -> PhysicalRowPartitionPlan:
    """Subdivide logical rows at native lazy chunk boundaries without computation."""
    if plan.has_no_rows:
        execution = LogicalRowBlockPlan(plan.dims, plan.sizes, plan.chunks, ())
        return PhysicalRowPartitionPlan(plan, execution, ())
    chunks = _physical_chunks(plan, tuple(values))
    selections = tuple(product(*chunks))
    partitions = tuple(
        _partition(plan, ordinal, selected)
        for ordinal, selected in zip(
            product(*(range(len(parts)) for parts in chunks)),
            selections,
            strict=True,
        )
    )
    blocks = tuple(partition.block for partition in partitions)
    execution = LogicalRowBlockPlan(plan.dims, plan.sizes, chunks, blocks)
    return PhysicalRowPartitionPlan(plan, execution, partitions)


__all__ = [
    "PhysicalRowPartition",
    "PhysicalRowPartitionPlan",
    "prepare_physical_row_partitions",
]
