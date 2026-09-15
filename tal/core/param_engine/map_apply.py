from __future__ import annotations

import numpy as np
import xarray as xr

from ..orchestration.indexing import (
    capture_result_coordinates,
    restore_result_coordinates,
    without_index_topology,
)
from ..schema_validate.finalize import transfer_dataarray_metadata
from .blocking import (
    LogicalRowBlock,
    assemble_logical_blocks,
    prepare_logical_row_blocks,
    select_logical_block,
)
from .types import ParamMap


def _align_application_arrays(
    *values: xr.DataArray,
    sequence_dim: str,
    query_dim: str,
    owner: str,
    what: str,
) -> tuple[xr.DataArray, ...]:
    from ..orchestration.alignment import align_exact

    return align_exact(
        *values,
        exclude={sequence_dim, query_dim},
        owner=owner,
        what=what,
    )


def align_param_map_application(
    values: xr.DataArray,
    param_map: ParamMap,
    *,
    sequence_dim: str,
    owner: str,
) -> tuple[xr.DataArray, ParamMap]:
    query_dim = param_map.query_dim
    aligned = _align_application_arrays(
        values,
        param_map.i0,
        param_map.i1,
        param_map.alpha,
        param_map.valid,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        owner=owner,
        what="parameter-map application",
    )
    return aligned[0], ParamMap(*aligned[1:], query_dim=query_dim)


def _gather_block(values_block: np.ndarray, index_block: np.ndarray) -> np.ndarray:
    query_size = int(index_block.shape[-1])
    seq_size = int(values_block.shape[-1])
    target_outer = np.broadcast_shapes(values_block.shape[:-1], index_block.shape[:-1])
    values = np.broadcast_to(values_block, target_outer + (seq_size,))
    index = np.broadcast_to(index_block.astype("int64", copy=False), target_outer + (query_size,))
    return np.take_along_axis(values, index, axis=-1)


def _gather_single_block(
    values: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
) -> xr.DataArray:
    out = xr.apply_ufunc(
        _gather_block,
        values,
        indexer.astype("int64"),
        input_core_dims=[[sequence_dim], [query_dim]],
        output_core_dims=[[query_dim]],
        vectorize=False,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[values.dtype],
    )
    broadcast_dims: list[str] = []
    for dim in values.dims:
        if dim != sequence_dim and dim not in broadcast_dims:
            broadcast_dims.append(dim)
    for dim in indexer.dims:
        if dim != query_dim and dim not in broadcast_dims:
            broadcast_dims.append(dim)
    return out.transpose(*(broadcast_dims + [query_dim]))


def gather_sequence_block(
    values: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
) -> xr.DataArray:
    """Gather one block whose logical-row bound is already established."""
    return _gather_single_block(
        values,
        indexer,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
    )


def _gather_output_template(
    values: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
    logical_dims: tuple[str, ...],
    dtype: np.dtype,
) -> xr.DataArray:
    unindexed_values = without_index_topology(values, dims=logical_dims)
    unindexed_indexer = without_index_topology(indexer, dims=logical_dims)
    source = unindexed_values.isel({sequence_dim: 0}, drop=True)
    template = xr.broadcast(source, unindexed_indexer)[0]
    outer = [dim for dim in values.dims if dim != sequence_dim]
    outer.extend(dim for dim in indexer.dims if dim != query_dim and dim not in outer)
    ordered = template.transpose(*(outer + [query_dim]))
    prototype = np.broadcast_to(np.empty((), dtype=dtype), ordered.shape)
    return ordered.copy(data=prototype)


def _empty_gather(
    values: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
) -> xr.DataArray:
    logical_dims = tuple(dim for dim in values.dims if dim != sequence_dim)
    logical_dims += tuple(dim for dim in indexer.dims if dim not in logical_dims)
    snapshot = capture_result_coordinates(
        values,
        indexer,
        output_dims=logical_dims,
        owner="gather_along_sequence",
    )
    values = without_index_topology(values, dims=logical_dims)
    indexer = without_index_topology(indexer, dims=logical_dims)
    dependent = [name for name, coord in values.coords.items() if sequence_dim in coord.dims]
    empty = values.drop_vars(dependent).isel({sequence_dim: slice(0, 0)})
    empty = empty.rename({sequence_dim: query_dim})
    template = xr.broadcast(empty, indexer)[0].assign_coords(indexer.coords)
    dims = [dim for dim in values.dims if dim != sequence_dim]
    dims.extend(dim for dim in indexer.dims if dim != query_dim and dim not in dims)
    dims.append(query_dim)
    result = template.transpose(*dims)
    return restore_result_coordinates(result, snapshot)  # type: ignore[return-value]


def _prepare_gather_inputs(
    values: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    if sequence_dim not in values.dims:
        raise ValueError(f"{owner}: values missing sequence_dim {sequence_dim!r}.")
    if query_dim not in indexer.dims:
        raise ValueError(f"{owner}: indexer missing query_dim {query_dim!r}.")
    return _align_application_arrays(
        values,
        indexer,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        owner=owner,
        what="gather",
    )


def _gather_nonempty(
    values: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
    owner: str,
) -> xr.DataArray:
    plan = prepare_logical_row_blocks(
        values,
        indexer,
        excluded_dims=frozenset({sequence_dim}),
        fastest_dim=query_dim,
    )
    blocks = (
        gather_sequence_block(
            select_logical_block(values, block),
            select_logical_block(indexer, block),
            sequence_dim=sequence_dim,
            query_dim=query_dim,
        )
        for block in plan.blocks
    )
    template = _gather_output_template(
        values,
        indexer,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        logical_dims=plan.dims,
        dtype=np.dtype(values.dtype),
    )
    return assemble_logical_blocks(
        blocks,
        plan=plan,
        template=template,
        index_sources=(values, indexer),
        owner=owner,
    )


def gather_along_sequence(
    values: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
    owner: str,
) -> xr.DataArray:
    values, indexer = _prepare_gather_inputs(
        values,
        indexer,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        owner=owner,
    )
    query_size = int(indexer.sizes[query_dim])
    if query_size == 0:
        return _empty_gather(
            values,
            indexer,
            sequence_dim=sequence_dim,
            query_dim=query_dim,
        )
    return _gather_nonempty(
        values,
        indexer,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        owner=owner,
    )


def gather_dataset_along_sequence(
    ds: xr.Dataset,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
    owner: str,
) -> xr.Dataset:
    out = ds.copy(deep=False)
    var_updates: dict[str, xr.DataArray] = {}
    for name, var in ds.data_vars.items():
        if sequence_dim in var.dims:
            var_updates[str(name)] = gather_along_sequence(
                var,
                indexer,
                sequence_dim=sequence_dim,
                query_dim=query_dim,
                owner=owner,
            )
    if var_updates:
        out = out.assign(var_updates)
    coord_updates: dict[str, xr.DataArray] = {}
    for name, coord in ds.coords.items():
        if sequence_dim in coord.dims:
            coord_updates[str(name)] = gather_along_sequence(
                coord,
                indexer,
                sequence_dim=sequence_dim,
                query_dim=query_dim,
                owner=owner,
            )
    return out.assign_coords(coord_updates) if coord_updates else out


def _slice_param_map(param_map: ParamMap, block: LogicalRowBlock) -> ParamMap:
    return ParamMap(
        i0=select_logical_block(param_map.i0, block),
        i1=select_logical_block(param_map.i1, block),
        alpha=select_logical_block(param_map.alpha, block),
        valid=select_logical_block(param_map.valid, block),
        query_dim=param_map.query_dim,
    )


def _interpolate_param_block(
    values: xr.DataArray,
    *,
    param_map: ParamMap,
    sequence_dim: str,
) -> xr.DataArray:
    max_index = int(values.sizes[sequence_dim] - 1)
    param_map = ParamMap(
        i0=param_map.i0.clip(min=0, max=max_index),
        i1=param_map.i1.clip(min=0, max=max_index),
        alpha=param_map.alpha,
        valid=param_map.valid,
        query_dim=param_map.query_dim,
    )
    query_dim = param_map.query_dim
    left = gather_sequence_block(
        values,
        param_map.i0,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
    )
    right = gather_sequence_block(
        values,
        param_map.i1,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
    )
    blended = ((1.0 - param_map.alpha) * left + param_map.alpha * right).transpose(*left.dims)
    return blended.where(param_map.valid, np.nan)


def _apply_param_map_blocks(
    values: xr.DataArray,
    *,
    param_map: ParamMap,
    sequence_dim: str,
    logical_dims: tuple[str, ...] | None,
) -> xr.DataArray:
    query_dim = param_map.query_dim
    plan = prepare_logical_row_blocks(
        values,
        param_map.i0,
        excluded_dims=frozenset({sequence_dim}),
        fastest_dim=query_dim,
        included_dims=logical_dims,
    )
    blocks = (
        _interpolate_param_block(
            select_logical_block(values, block),
            param_map=_slice_param_map(param_map, block),
            sequence_dim=sequence_dim,
        )
        for block in plan.blocks
    )
    template = _gather_output_template(
        values,
        param_map.i0,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
        logical_dims=plan.dims,
        dtype=np.dtype(np.result_type(values.dtype, np.float64)),
    )
    return assemble_logical_blocks(
        blocks,
        plan=plan,
        template=template,
        index_sources=(values, param_map.i0),
        owner="apply_param_map",
    )


def empty_mapped_value(
    values: xr.DataArray,
    *,
    param_map: ParamMap,
    sequence_dim: str,
) -> xr.DataArray:
    order = [dim for dim in values.dims if dim != sequence_dim]
    order.extend(dim for dim in param_map.valid.dims if dim not in order)
    sizes = {
        dim: int(values.sizes[dim]) if dim in values.dims else int(param_map.valid.sizes[dim])
        for dim in order
    }
    snapshot = capture_result_coordinates(
        values,
        param_map.valid,
        output_dims=tuple(order),
        owner="apply_param_map",
    )
    shape = tuple(sizes[dim] for dim in order)
    data: object = np.full(shape, np.nan, dtype=np.float64)
    if values.chunks is not None or param_map.valid.chunks is not None:
        import dask.array as da

        sources = (values, param_map.valid)
        chunks = tuple(
            next(
                (source.chunksizes[dim] for source in sources if source.chunks is not None and dim in source.dims),
                (sizes[dim],),
            )
            for dim in order
        )
        data = da.full(shape, np.nan, chunks=chunks, dtype=np.float64)
    out = xr.DataArray(
        data,
        dims=order,
        name=values.name,
    )
    out = transfer_dataarray_metadata(values, out)
    restored = restore_result_coordinates(out, snapshot)
    return restored.where(param_map.valid, np.nan)  # type: ignore[return-value]


def _apply_param_map_request(
    values: xr.DataArray,
    *,
    param_map: ParamMap,
    sequence_dim: str,
    logical_dims: tuple[str, ...] | None,
) -> xr.DataArray:
    if sequence_dim not in values.dims:
        raise ValueError(f"apply_param_map: values missing sequence_dim {sequence_dim!r}.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("apply_param_map: values dtype must be numeric.")
    values, param_map = align_param_map_application(
        values,
        param_map,
        sequence_dim=sequence_dim,
        owner="apply_param_map",
    )
    if int(values.sizes.get(sequence_dim, 0)) == 0:
        return empty_mapped_value(values, param_map=param_map, sequence_dim=sequence_dim)
    if int(param_map.i0.sizes[param_map.query_dim]) == 0:
        return empty_mapped_value(values, param_map=param_map, sequence_dim=sequence_dim)
    return _apply_param_map_blocks(
        values,
        param_map=param_map,
        sequence_dim=sequence_dim,
        logical_dims=logical_dims,
    )


def apply_param_map_with_batch_dims(
    values: xr.DataArray,
    *,
    param_map: ParamMap,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
) -> xr.DataArray:
    """Apply a map using already-resolved semantic batch dimensions."""
    logical_dims = (*batch_dims, param_map.query_dim)
    return _apply_param_map_request(
        values,
        param_map=param_map,
        sequence_dim=sequence_dim,
        logical_dims=logical_dims,
    )


def apply_param_map(
    values: xr.DataArray,
    *,
    param_map: ParamMap,
    sequence_dim: str,
) -> xr.DataArray:
    """Apply a ``ParamMap`` to interpolate values along ``sequence_dim``.

    Parameters
    ----------
    values : xr.DataArray
        Input values consumed by this operation.
    param_map : ParamMap, optional
        Parameter-domain input used for temporal evaluation/alignment.
    sequence_dim : str, optional
        Optional override for the sequence dimension used by temporal semantics.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return _apply_param_map_request(
        values,
        param_map=param_map,
        sequence_dim=sequence_dim,
        logical_dims=None,
    )


__all__ = [
    "align_param_map_application",
    "apply_param_map",
    "empty_mapped_value",
    "gather_along_sequence",
    "gather_dataset_along_sequence",
]
