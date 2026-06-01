from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockInputSpec:
    name: str
    core_ndim: int
    dtype: object | None = None


@dataclass(frozen=True)
class BlockRows:
    row_arrays: tuple[np.ndarray, ...]
    outer_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def row_count(shape: tuple[int, ...]) -> int:
    rows = 1
    for size in shape:
        rows *= int(size)
    return rows


def _coerce_block(block: object, spec: BlockInputSpec, *, owner: str) -> np.ndarray:
    try:
        if spec.dtype is None:
            return np.asarray(block)
        return np.asarray(block, dtype=spec.dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{owner}: block {spec.name!r} could not be coerced to dtype {spec.dtype!r}.") from exc


def _outer_shape(array: np.ndarray, core_ndim: int) -> tuple[int, ...]:
    if core_ndim == 0:
        return tuple(int(size) for size in array.shape)
    return tuple(int(size) for size in array.shape[:-core_ndim])


def _core_shape(array: np.ndarray, core_ndim: int) -> tuple[int, ...]:
    if core_ndim == 0:
        return ()
    return tuple(int(size) for size in array.shape[-core_ndim:])


def _validate_specs(blocks: tuple[object, ...], specs: tuple[BlockInputSpec, ...], *, owner: str) -> None:
    if len(blocks) != len(specs):
        raise ValueError(f"{owner}: block/spec count mismatch ({len(blocks)} blocks, {len(specs)} specs).")
    for spec in specs:
        if spec.core_ndim < 0:
            raise ValueError(f"{owner}: block {spec.name!r} core_ndim must be >= 0.")


def prepare_block_rows(
    blocks: tuple[object, ...],
    specs: tuple[BlockInputSpec, ...],
    *,
    output_core_shape: tuple[int, ...],
    owner: str,
    broadcast_shapes: tuple[tuple[int, ...], ...] | None = None,
) -> BlockRows:
    _validate_specs(blocks, specs, owner=owner)
    arrays = tuple(_coerce_block(block, spec, owner=owner) for block, spec in zip(blocks, specs, strict=True))
    for array, spec in zip(arrays, specs, strict=True):
        if array.ndim < spec.core_ndim:
            raise ValueError(f"{owner}: block {spec.name!r} must include {spec.core_ndim} trailing core dimensions.")
    outer_inputs = tuple(_outer_shape(array, spec.core_ndim) for array, spec in zip(arrays, specs, strict=True))
    if broadcast_shapes is not None:
        outer_inputs = outer_inputs + tuple(tuple(int(size) for size in shape) for shape in broadcast_shapes)
    try:
        outer = np.broadcast_shapes(*outer_inputs) if outer_inputs else ()
    except ValueError as exc:
        raise ValueError(f"{owner}: blocks are not broadcast-compatible.") from exc
    rows = row_count(outer)
    row_arrays = []
    for array, spec in zip(arrays, specs, strict=True):
        core = _core_shape(array, spec.core_ndim)
        target = outer + core
        try:
            broadcast = np.broadcast_to(array, target)
        except ValueError as exc:
            raise ValueError(f"{owner}: block {spec.name!r} is not broadcast-compatible.") from exc
        row_arrays.append(np.ascontiguousarray(broadcast.reshape((rows,) + core)))
    output_core = tuple(int(size) for size in output_core_shape)
    return BlockRows(tuple(row_arrays), outer, outer + output_core)


__all__ = ["BlockInputSpec", "BlockRows", "prepare_block_rows", "row_count"]
