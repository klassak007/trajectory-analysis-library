from __future__ import annotations

"""Shared owner for stacking event-major window outputs onto one sequence axis."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import xarray as xr

from tal.utils.xarray_namespace import dataset_namespace_names, unique_temp_dim

from ..orchestration.indexing import (
    capture_index_topology,
    dimension_coordinates,
    without_index_topology,
)
from ..orchestration.lazy import is_chunked_dataarray
from ..param_engine.map_apply import gather_dataset_along_sequence
from ..param_ops.guards import reserved_coord_is_owned
from ..schema_read import read_param_coord_name
from .event_primitives import EDGE_INVALID, SAMPLE_SENTINEL

_STACK_META = ("window_tau", "window_event_index")
_STREAM_META = ("stream_segment_index",)
XrObj = TypeVar("XrObj", xr.DataArray, xr.Dataset)


@dataclass(frozen=True)
class StackedWindowResult:
    """Stacked window payload before AO finalization.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    ds: xr.Dataset
    stack_dim: str
    size_name: str
    valid_sample: xr.DataArray


@dataclass(frozen=True)
class StackedStreamResult:
    """Stacked stream payload before AO finalization.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    ds: xr.Dataset
    stream_dim: str
    size_name: str
    valid_sample: xr.DataArray



def _bounded_stream_indexer_block(valid_block: np.ndarray, *, out_len: int) -> np.ndarray:
    outer_shape = valid_block.shape[:-1]
    row_count = 1 if not outer_shape else int(np.prod(outer_shape, dtype=np.int64))
    stream_size = int(valid_block.shape[-1])
    valid_rows = valid_block.reshape(row_count, stream_size)
    out_rows = np.full((row_count, out_len), SAMPLE_SENTINEL, dtype="int64")
    if out_len == 0 or stream_size == 0:
        return out_rows.reshape(outer_shape + (out_len,))
    rank = np.cumsum(valid_rows, axis=1) - 1
    source_idx = np.broadcast_to(np.arange(stream_size, dtype="int64"), valid_rows.shape)
    keep = valid_rows & (rank < out_len)
    row_idx, col_idx = np.nonzero(keep)
    out_rows[row_idx, rank[row_idx, col_idx]] = source_idx[row_idx, col_idx]
    return out_rows.reshape(outer_shape + (out_len,))



def _bounded_stream_indexer(
    valid_sample: xr.DataArray,
    *,
    stream_dim: str,
    out_len: int,
    out_dim: str,
) -> xr.DataArray:
    if out_len == 0:
        return valid_sample.isel({stream_dim: slice(0, 0)}).astype("int64").rename("stream_indexer")
    chunked = is_chunked_dataarray(valid_sample)
    ufunc_kwargs: dict[str, object] = {}
    if chunked:
        ufunc_kwargs["dask_gufunc_kwargs"] = {"output_sizes": {out_dim: out_len}, "allow_rechunk": True}
    out = xr.apply_ufunc(
        _bounded_stream_indexer_block,
        valid_sample.astype(bool),
        input_core_dims=[[stream_dim]],
        output_core_dims=[[out_dim]],
        kwargs={"out_len": out_len},
        vectorize=False,
        dask="parallelized" if chunked else "allowed",
        output_dtypes=[np.int64],
        **ufunc_kwargs,
    )
    coord = np.arange(out_len, dtype="int64")
    return out.assign_coords({out_dim: coord}).rename({out_dim: stream_dim}).rename("stream_indexer")



def gather_packed_samples(
    ds: xr.Dataset, indexer: xr.DataArray, *, sequence_dim: str, owner: str,
) -> tuple[xr.Dataset, xr.DataArray]:
    """Gather a reviewed positional packing map without sampled carriers."""
    names = dataset_namespace_names(ds) + tuple(str(name) for name in indexer.coords)
    query_dim = unique_temp_dim("packed_query", taken_dims=names)
    batch_dims = tuple(dim for dim in indexer.dims if dim != sequence_dim)
    coordinates = dimension_coordinates(ds, dims=batch_dims)
    unindexed = tuple(dim for dim in batch_dims if dim in ds.coords and dim not in ds.xindexes and ds[dim].dims == (dim,))
    source = ds.drop_vars(unindexed)
    numerical = xr.DataArray(indexer.variable, coords=capture_index_topology(ds, dims=batch_dims).coordinates)
    numerical = numerical.assign_coords(dimension_coordinates(indexer, dims=(sequence_dim,)))
    query = numerical.rename({sequence_dim: query_dim})
    safe = query.where(query >= 0, 0).clip(min=0, max=max(ds.sizes[sequence_dim] - 1, 0)).astype("int64")
    gathered = gather_dataset_along_sequence(source, safe, sequence_dim=sequence_dim, query_dim=query_dim, owner=owner)
    booleans = {
        name: gathered[name].fillna(False).astype(bool).variable
        for name, coord in ds.coords.items()
        if sequence_dim in coord.dims and coord.dtype == bool and reserved_coord_is_owned(ds, name=name)
    }
    gathered = gathered.assign_coords(xr.Coordinates(booleans, indexes={}))
    valid = xr.DataArray(query.data >= 0, dims=query.dims, coords=dimension_coordinates(query, dims=tuple(query.dims)))
    return (gathered.rename({query_dim: sequence_dim}).assign_coords(coordinates),
            valid.rename({query_dim: sequence_dim}).assign_coords(coordinates))


def mask_aligned_samples(ds: xr.Dataset, valid: xr.DataArray, *, sequence_dim: str) -> xr.Dataset:
    """Mask aligned numerical variables, retaining reviewed topology separately."""
    condition = xr.DataArray(valid.variable)
    variables = {
        name: xr.DataArray(var.variable).where(condition).variable
        for name, var in ds.data_vars.items() if sequence_dim in var.dims
    }
    coordinates = {
        name: _masked_coordinate(coord, condition, generated=reserved_coord_is_owned(ds, name=name))
        for name, coord in ds.coords.items() if name != sequence_dim and sequence_dim in coord.dims
    }
    return ds.assign(variables).assign_coords(xr.Coordinates(coordinates, indexes={}))


def _masked_coordinate(coord: xr.DataArray, condition: xr.DataArray, *, generated: bool) -> xr.Variable:
    target = xr.DataArray(coord.variable)
    if generated and coord.dtype == bool:
        return target.where(condition, False).variable
    return target.where(condition).variable


def _pack_window_samples(ds: xr.Dataset, valid: xr.DataArray, *, stack_dim: str, owner: str):
    out_dim = unique_temp_dim("__tal_window_out__", taken_dims=dataset_namespace_names(ds))
    indexer = _bounded_stream_indexer(valid, stream_dim=stack_dim, out_len=ds.sizes[stack_dim], out_dim=out_dim)
    indexer = without_index_topology(indexer, dims=(stack_dim,))
    gathered, packed_valid = gather_packed_samples(ds, indexer, sequence_dim=stack_dim, owner=owner)
    out = mask_aligned_samples(gathered, packed_valid, sequence_dim=stack_dim)
    condition = xr.DataArray(packed_valid.variable)
    updates = {
        name: xr.DataArray(gathered[name].variable).where(condition, fill).astype(dtype).variable
        for name, fill, dtype in (("window_event_index", -1, "int64"), ("event_sample_index_before", -1, "int64"),
                                 ("event_sample_index_after", -1, "int64"), ("event_edge_code", 0, "int8"))
    }
    return out.assign_coords(xr.Coordinates(updates, indexes={})), packed_valid


def _stack_dim_name(ds: xr.Dataset, *, owner: str) -> str:
    _ = owner
    names = set(dataset_namespace_names(ds))
    names.update(_STACK_META)
    return unique_temp_dim("window_sample", taken_dims=tuple(sorted(names)))


def _size_coord_name(ds: xr.Dataset, *, owner: str) -> str:
    _ = owner
    names = set(dataset_namespace_names(ds))
    names.update(_STACK_META)
    return unique_temp_dim("window_size", taken_dims=tuple(sorted(names)))


def _stream_dim_name(ds: xr.Dataset, *, owner: str) -> str:
    _ = owner
    names = set(dataset_namespace_names(ds))
    names.update(_STREAM_META)
    return unique_temp_dim("stream_sample", taken_dims=tuple(sorted(names)))


def _stream_size_coord_name(ds: xr.Dataset, *, owner: str) -> str:
    _ = owner
    names = set(dataset_namespace_names(ds))
    names.update(_STREAM_META)
    return unique_temp_dim("stream_size", taken_dims=tuple(sorted(names)))


def _zero_batch_dims(*, sizes: Mapping[str, int], batch_dims: Sequence[str]) -> tuple[str, ...]:
    return tuple(dim for dim in batch_dims if int(sizes.get(dim, 0)) == 0)


def _captured_zero_dim_coords(obj: XrObj, *, zero_dims: Sequence[str]) -> xr.Coordinates:
    return dimension_coordinates(obj, dims=tuple(zero_dims))


def _pad_zero_batch_dims(obj: XrObj, *, zero_dims: Sequence[str]) -> XrObj:
    out = without_index_topology(obj, dims=tuple(zero_dims))
    fill = {name: False for name, coord in obj.coords.items()
            if coord.dtype == bool and reserved_coord_is_owned(obj, name=name)}
    for dim in zero_dims:
        out = out.assign_coords({dim: np.arange(0, dtype="int64")}).reindex({dim: [0]}, fill_value=fill)
    return out


def _restore_zero_batch_dims(
    obj: XrObj,
    *,
    zero_dims: Sequence[str],
    captured_coords: xr.Coordinates,
) -> XrObj:
    if not zero_dims:
        return obj
    indexers = {dim: slice(0, 0) for dim in zero_dims}
    out = without_index_topology(obj.isel(indexers), dims=tuple(zero_dims))
    return out.assign_coords(captured_coords)


def _stack_dataset(
    ds: xr.Dataset,
    *,
    batch_dims: Sequence[str],
    event_dim: str,
    tau_dim: str,
    stack_dim: str,
    owner: str,
) -> xr.Dataset:
    if event_dim not in ds.dims or tau_dim not in ds.dims:
        raise ValueError(f"{owner}: expected dims {event_dim!r} and {tau_dim!r} on window dataset.")
    zero_dims = _zero_batch_dims(sizes=ds.sizes, batch_dims=batch_dims)
    captured = _captured_zero_dim_coords(ds, zero_dims=zero_dims)
    padded = _pad_zero_batch_dims(ds, zero_dims=zero_dims)
    padded = _rechunk_empty_variables(padded)
    stacked = padded.stack({stack_dim: (event_dim, tau_dim)}, create_index=False)
    return _restore_zero_batch_dims(stacked, zero_dims=zero_dims, captured_coords=captured)


def _rechunk_empty_variables(ds: xr.Dataset) -> xr.Dataset:
    """Consolidate empty lazy products so Dask can stack their dimensions."""
    updates = {
        name: var.chunk({dim: -1 for dim in var.dims}).variable
        for name, var in ds.items() if var.size == 0 and var.chunks is not None
    }
    coords = {
        name: coord.chunk({dim: -1 for dim in coord.dims}).variable
        for name, coord in ds.coords.items() if coord.size == 0 and coord.chunks is not None
    }
    return ds.assign(updates).assign_coords(xr.Coordinates(coords, indexes={}))


def _valid_sample(
    ds: xr.Dataset,
    *,
    batch_dims: Sequence[str],
    event_dim: str,
    tau_dim: str,
    stack_dim: str,
) -> xr.DataArray:
    valid_event = (ds["event_edge_code"] != int(EDGE_INVALID)).astype(bool)
    tau_valid = xr.DataArray(np.ones(ds.sizes[tau_dim], dtype=bool), dims=(tau_dim,), coords={tau_dim: ds[tau_dim]})
    dims = tuple(batch_dims) + (event_dim, tau_dim)
    valid = (valid_event & tau_valid).transpose(*dims)
    zero_dims = _zero_batch_dims(sizes=ds.sizes, batch_dims=batch_dims)
    captured = _captured_zero_dim_coords(valid, zero_dims=zero_dims)
    padded = _pad_zero_batch_dims(valid, zero_dims=zero_dims)
    if padded.size == 0 and padded.chunks is not None:
        padded = padded.chunk({dim: -1 for dim in padded.dims})
    stacked = padded.stack({stack_dim: (event_dim, tau_dim)}, create_index=False)
    return _restore_zero_batch_dims(stacked, zero_dims=zero_dims, captured_coords=captured).rename("valid_sample")


def _broadcast_metadata(base: xr.DataArray, valid_sample: xr.DataArray) -> xr.DataArray:
    expanded = xr.DataArray(base.variable)
    for dim in valid_sample.dims:
        if dim not in expanded.dims:
            expanded = expanded.expand_dims({dim: valid_sample.sizes[dim]})
    expanded = expanded.transpose(*valid_sample.dims)
    return expanded


def _window_tau_coord(stacked: xr.Dataset, *, valid_sample: xr.DataArray, tau_dim: str) -> xr.DataArray:
    base = stacked.coords[tau_dim].astype("float64")
    expanded = _broadcast_metadata(base, valid_sample)
    return expanded.where(xr.DataArray(valid_sample.variable), np.nan).astype("float64").rename("window_tau")


def _window_event_index_coord(
    ds: xr.Dataset,
    *,
    valid_sample: xr.DataArray,
    event_dim: str,
    tau_dim: str,
) -> xr.DataArray:
    stack_dim = valid_sample.dims[-1]
    positions = xr.Variable((event_dim,), np.arange(ds.sizes[event_dim], dtype="int64"))
    positions = positions.set_dims({event_dim: ds.sizes[event_dim], tau_dim: ds.sizes[tau_dim]})
    base = xr.DataArray(positions).stack({stack_dim: (event_dim, tau_dim)}, create_index=False)
    expanded = _broadcast_metadata(base, valid_sample)
    return xr.where(xr.DataArray(valid_sample.variable), expanded, -1).astype("int64").rename("window_event_index")


def _valid_size(valid_sample: xr.DataArray, *, stack_dim: str) -> xr.DataArray:
    return valid_sample.astype("int64").sum(dim=stack_dim).astype("int64")


def _drop_level_coords(stacked: xr.Dataset, *, event_dim: str, tau_dim: str) -> xr.Dataset:
    names = [name for name in (event_dim, tau_dim) if name in stacked.coords]
    return stacked.drop_vars(names) if names else stacked


def _drop_level_coords_pair(stacked: xr.Dataset, *, first_dim: str, second_dim: str) -> xr.Dataset:
    param_name = read_param_coord_name(stacked)
    names = [name for name in (first_dim, second_dim) if name in stacked.coords and name != param_name]
    return stacked.drop_vars(names) if names else stacked


def _assert_metadata_namespace(
    ds: xr.Dataset,
    *,
    owner: str,
) -> None:
    names = set(dataset_namespace_names(ds))
    conflicts = sorted(name for name in _STACK_META if name in names)
    if conflicts:
        raise ValueError(f"{owner}: stacked metadata names conflict with dataset namespace: {conflicts!r}.")


def _assert_stream_metadata_namespace(
    ds: xr.Dataset,
    *,
    owner: str,
) -> None:
    names = set(dataset_namespace_names(ds))
    conflicts = sorted(name for name in _STREAM_META if name in names)
    if conflicts:
        raise ValueError(f"{owner}: stream metadata names conflict with dataset namespace: {conflicts!r}.")


def _stack_stream_dataset(
    ds: xr.Dataset,
    *,
    batch_dims: Sequence[str],
    segment_dim: str,
    sequence_dim: str,
    stream_dim: str,
    owner: str,
) -> xr.Dataset:
    if segment_dim not in ds.dims or sequence_dim not in ds.dims:
        raise ValueError(f"{owner}: expected dims {segment_dim!r} and {sequence_dim!r} on stream dataset.")
    zero_dims = _zero_batch_dims(sizes=ds.sizes, batch_dims=batch_dims)
    captured = _captured_zero_dim_coords(ds, zero_dims=zero_dims)
    padded = _pad_zero_batch_dims(ds, zero_dims=zero_dims)
    padded = _rechunk_empty_variables(padded)
    stacked = padded.stack({stream_dim: (segment_dim, sequence_dim)}, create_index=False)
    return _restore_zero_batch_dims(stacked, zero_dims=zero_dims, captured_coords=captured)


def _stream_valid_sample(
    ds: xr.Dataset,
    *,
    stream_dim: str,
    orig_index_name: str,
) -> xr.DataArray:
    if orig_index_name not in ds.coords:
        raise ValueError(f"events.when: expected coordinate {orig_index_name!r} on stacked stream dataset.")
    valid = (ds.coords[orig_index_name] >= 0).astype(bool)
    return valid.rename("valid_sample")


def _stream_segment_index_coord(
    stacked: xr.Dataset,
    *,
    valid_sample: xr.DataArray,
    segment_dim: str,
) -> xr.DataArray:
    base = stacked.coords[segment_dim].astype("int64")
    expanded = _broadcast_metadata(base, valid_sample)
    return xr.where(valid_sample, expanded, -1).astype("int64").rename("stream_segment_index")


def _stream_valid_size(valid_sample: xr.DataArray, *, stream_dim: str) -> xr.DataArray:
    return valid_sample.astype("int64").sum(dim=stream_dim).astype("int64")


def stack_segment_stream(
    ds: xr.Dataset,
    *,
    batch_dims: Sequence[str],
    segment_dim: str,
    sequence_dim: str,
    owner: str,
    orig_index_name: str = "orig_index",
) -> StackedStreamResult:
    """Flatten ``(segment_dim, sequence_dim)`` into one stream sequence axis.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    batch_dims : Sequence[str], optional
        Optional override for batch dimensions used by temporal semantics.
    segment_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    sequence_dim : str, optional
        Optional override for the sequence dimension used by temporal semantics.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    orig_index_name : str, optional
        Index selector/configuration applied to the source data.

    Returns
    -------
    StackedStreamResult
        String result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    stream_dim = _stream_dim_name(ds, owner=owner)
    stacked = _stack_stream_dataset(
        ds,
        batch_dims=batch_dims,
        segment_dim=segment_dim,
        sequence_dim=sequence_dim,
        stream_dim=stream_dim,
        owner=owner,
    )
    valid_sample = _stream_valid_sample(stacked, stream_dim=stream_dim, orig_index_name=orig_index_name)
    stream_segment_index = _stream_segment_index_coord(stacked, valid_sample=valid_sample, segment_dim=segment_dim)
    out = _drop_level_coords_pair(stacked, first_dim=segment_dim, second_dim=sequence_dim)
    _assert_stream_metadata_namespace(out, owner=owner)
    size_name = _stream_size_coord_name(out, owner=owner)
    out = out.assign_coords(
        {
            "stream_segment_index": stream_segment_index,
            size_name: _stream_valid_size(valid_sample, stream_dim=stream_dim),
        }
    )
    return StackedStreamResult(ds=out, stream_dim=stream_dim, size_name=size_name, valid_sample=valid_sample)


def stack_event_windows(
    ds: xr.Dataset,
    *,
    batch_dims: Sequence[str],
    event_dim: str,
    tau_dim: str,
    owner: str,
) -> StackedWindowResult:
    """Flatten ``(event_dim, tau_dim)`` windows into one stack sequence axis.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    batch_dims : Sequence[str], optional
        Optional override for batch dimensions used by temporal semantics.
    event_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    tau_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    StackedWindowResult
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    stack_dim = _stack_dim_name(ds, owner=owner)
    stacked = _stack_dataset(
        ds,
        batch_dims=batch_dims,
        event_dim=event_dim,
        tau_dim=tau_dim,
        stack_dim=stack_dim,
        owner=owner,
    )
    valid_sample = _valid_sample(
        ds,
        batch_dims=batch_dims,
        event_dim=event_dim,
        tau_dim=tau_dim,
        stack_dim=stack_dim,
    )
    window_tau = _window_tau_coord(stacked, valid_sample=valid_sample, tau_dim=tau_dim)
    window_event_index = _window_event_index_coord(ds, valid_sample=valid_sample, event_dim=event_dim, tau_dim=tau_dim)
    out = _drop_level_coords(stacked, event_dim=event_dim, tau_dim=tau_dim)
    _assert_metadata_namespace(out, owner=owner)
    size_name = _size_coord_name(out, owner=owner)
    out = out.assign_coords(
        {
            "window_tau": window_tau,
            "window_event_index": window_event_index,
            size_name: _valid_size(valid_sample, stack_dim=stack_dim),
        }
    )
    out, valid_sample = _pack_window_samples(out, valid_sample, stack_dim=stack_dim, owner=owner)
    out = out.assign_coords({size_name: _valid_size(valid_sample, stack_dim=stack_dim).variable})
    return StackedWindowResult(ds=out, stack_dim=stack_dim, size_name=size_name, valid_sample=valid_sample)


__all__ = ["StackedStreamResult", "StackedWindowResult", "stack_event_windows", "stack_segment_stream"]
