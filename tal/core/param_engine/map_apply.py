from __future__ import annotations

import numpy as np
import xarray as xr

from .types import ParamMap


def _gather_block(values_block: np.ndarray, index_block: np.ndarray) -> np.ndarray:
    query_size = int(index_block.shape[-1])
    seq_size = int(values_block.shape[-1])
    target_outer = np.broadcast_shapes(values_block.shape[:-1], index_block.shape[:-1])
    values = np.broadcast_to(values_block, target_outer + (seq_size,))
    index = np.broadcast_to(index_block.astype("int64", copy=False), target_outer + (query_size,))
    return np.take_along_axis(values, index, axis=-1)


def gather_along_sequence(
    values: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
    owner: str,
) -> xr.DataArray:
    if sequence_dim not in values.dims:
        raise ValueError(f"{owner}: values missing sequence_dim {sequence_dim!r}.")
    if query_dim not in indexer.dims:
        raise ValueError(f"{owner}: indexer missing query_dim {query_dim!r}.")
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


def _empty_sequence_output(
    values: xr.DataArray,
    *,
    param_map: ParamMap,
    sequence_dim: str,
) -> xr.DataArray:
    out = xr.DataArray(np.nan)
    for dim in values.dims:
        if dim == sequence_dim:
            continue
        if dim in values.coords and values.coords[dim].dims == (dim,):
            coord = values.coords[dim]
        else:
            coord = np.arange(values.sizes[dim], dtype="int64")
        out = out.expand_dims({dim: coord})
    out, _ = xr.broadcast(out, param_map.valid)
    order = [dim for dim in values.dims if dim != sequence_dim]
    order.extend(dim for dim in param_map.valid.dims if dim not in order)
    return out.transpose(*order).where(param_map.valid, np.nan).astype("float64")


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
    if sequence_dim not in values.dims:
        raise ValueError(f"apply_param_map: values missing sequence_dim {sequence_dim!r}.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("apply_param_map: values dtype must be numeric.")
    if int(values.sizes.get(sequence_dim, 0)) == 0:
        return _empty_sequence_output(values, param_map=param_map, sequence_dim=sequence_dim)
    max_index = int(values.sizes[sequence_dim] - 1)
    i0 = param_map.i0.clip(min=0, max=max_index)
    i1 = param_map.i1.clip(min=0, max=max_index)
    left = gather_along_sequence(
        values,
        i0,
        sequence_dim=sequence_dim,
        query_dim=param_map.query_dim,
        owner="apply_param_map",
    )
    right = gather_along_sequence(
        values,
        i1,
        sequence_dim=sequence_dim,
        query_dim=param_map.query_dim,
        owner="apply_param_map",
    )
    blended = (1.0 - param_map.alpha) * left + param_map.alpha * right
    blended = blended.transpose(*left.dims)
    return blended.where(param_map.valid, np.nan)


__all__ = [
    "apply_param_map",
    "gather_along_sequence",
    "gather_dataset_along_sequence",
]
