from __future__ import annotations

"""Shared owner for stacking event-major window outputs onto one sequence axis."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np
import xarray as xr

from tal.utils.xarray_namespace import dataset_namespace_names, unique_temp_dim

from .event_primitives import EDGE_INVALID

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


def _captured_zero_dim_coords(obj: XrObj, *, zero_dims: Sequence[str]) -> dict[str, xr.DataArray]:
    captured: dict[str, xr.DataArray] = {}
    for dim in zero_dims:
        if dim in obj.coords:
            captured[dim] = obj.coords[dim]
    return captured


def _pad_zero_batch_dims(obj: XrObj, *, zero_dims: Sequence[str]) -> XrObj:
    out = obj
    for dim in zero_dims:
        out = out.reindex({dim: np.asarray([0], dtype="int64")})
    return out


def _restore_zero_batch_dims(
    obj: XrObj,
    *,
    zero_dims: Sequence[str],
    captured_coords: Mapping[str, xr.DataArray] | None = None,
) -> XrObj:
    if not zero_dims:
        return obj
    indexers = {dim: slice(0, 0) for dim in zero_dims}
    out = obj.isel(indexers)
    if not captured_coords:
        return out
    restore = {name: coord for name, coord in captured_coords.items() if name in out.dims}
    return out.assign_coords(restore) if restore else out


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
    stacked = padded.stack({stack_dim: (event_dim, tau_dim)}).reset_index(stack_dim)
    return _restore_zero_batch_dims(stacked, zero_dims=zero_dims, captured_coords=captured)


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
    stacked = padded.stack({stack_dim: (event_dim, tau_dim)}).reset_index(stack_dim, drop=True)
    return _restore_zero_batch_dims(stacked, zero_dims=zero_dims, captured_coords=captured).rename("valid_sample")


def _window_tau_coord(stacked: xr.Dataset, *, valid_sample: xr.DataArray, tau_dim: str) -> xr.DataArray:
    base = stacked.coords[tau_dim].astype("float64")
    expanded = xr.broadcast(base, valid_sample)[0].transpose(*valid_sample.dims)
    return expanded.where(valid_sample, np.nan).astype("float64").rename("window_tau")


def _window_event_index_coord(
    stacked: xr.Dataset,
    *,
    valid_sample: xr.DataArray,
    event_dim: str,
) -> xr.DataArray:
    base = stacked.coords[event_dim].astype("int64")
    expanded = xr.broadcast(base, valid_sample)[0].transpose(*valid_sample.dims)
    return xr.where(valid_sample, expanded, -1).astype("int64").rename("window_event_index")


def _valid_size(valid_sample: xr.DataArray, *, stack_dim: str) -> xr.DataArray:
    return valid_sample.astype("int64").sum(dim=stack_dim).astype("int64")


def _drop_level_coords(stacked: xr.Dataset, *, event_dim: str, tau_dim: str) -> xr.Dataset:
    names = [name for name in (event_dim, tau_dim) if name in stacked.coords]
    return stacked.drop_vars(names) if names else stacked


def _drop_level_coords_pair(stacked: xr.Dataset, *, first_dim: str, second_dim: str) -> xr.Dataset:
    names = [name for name in (first_dim, second_dim) if name in stacked.coords]
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
    stacked = padded.stack({stream_dim: (segment_dim, sequence_dim)}).reset_index(stream_dim)
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
    expanded = xr.broadcast(base, valid_sample)[0].transpose(*valid_sample.dims)
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
    window_event_index = _window_event_index_coord(stacked, valid_sample=valid_sample, event_dim=event_dim)
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
    return StackedWindowResult(ds=out, stack_dim=stack_dim, size_name=size_name, valid_sample=valid_sample)


__all__ = ["StackedStreamResult", "StackedWindowResult", "stack_event_windows", "stack_segment_stream"]
