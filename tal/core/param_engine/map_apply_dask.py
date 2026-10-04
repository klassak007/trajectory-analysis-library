"""Construct one lazy array application from already aligned parameter maps."""

from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr

from ..orchestration.indexing import (
    capture_result_coordinates,
    restore_result_coordinates,
)
from .blocking import LogicalRowBlockPlan
from .types import ParamMap


def _array_operand(
    value: xr.DataArray,
    dims: tuple[str, ...],
    chunks: dict[str, tuple[int, ...]],
) -> da.Array:
    present = tuple(dim for dim in dims if dim in value.dims)
    data = value.transpose(*present).data
    target = tuple(chunks.get(dim, (value.sizes[dim],)) for dim in present)
    if isinstance(data, da.Array):
        updates = {axis: chunks[dim] for axis, dim in enumerate(present) if dim in chunks}
        data = data.rechunk(updates)
    else:
        data = da.from_array(data, chunks=target, name=False)
    # Insert absent broadcast axes without expanding any source buffer.
    selection = tuple(slice(None) if dim in present else None for dim in dims)
    return data[selection] if len(present) != len(dims) else data


def apply_lazy_param_map(
    values: xr.DataArray,
    *,
    param_map: ParamMap,
    sequence_dim: str,
    plan: LogicalRowBlockPlan,
    template: xr.DataArray,
    kernel,
) -> xr.DataArray:
    """Emit blockwise interpolation and restore public coordinates once."""
    dims = tuple(template.dims)
    outer = dims[:-1]
    chunks = {
        dim: tuple(part.stop - part.start for part in lane)
        for dim, lane in zip(plan.dims, plan.chunks, strict=True)
    }
    source = _array_operand(values, (*outer, sequence_dim), chunks)
    columns = tuple(
        _array_operand(column, dims, chunks)
        for column in (param_map.i0, param_map.i1, param_map.alpha, param_map.valid)
    )
    output_axes = tuple(range(len(dims)))
    source_axes = (*output_axes[:-1], len(dims))
    operands = tuple(item for column in columns for item in (column, output_axes))
    data = da.blockwise(
        kernel, output_axes, source, source_axes, *operands,
        dtype=template.dtype, concatenate=True, align_arrays=False,
        meta=np.empty((0,) * len(dims), dtype=template.dtype),
    )
    snapshot = capture_result_coordinates(
        values, param_map.i0, output_dims=dims, owner="apply_param_map",
    )
    return restore_result_coordinates(template.copy(data=data), snapshot)


__all__: list[str] = []
