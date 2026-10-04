from __future__ import annotations

import numpy as np
import xarray as xr

from tal.utils.xarray_namespace import (
    dataarray_namespace_names,
    dataset_namespace_names,
    unique_temp_dim,
)

from ..orchestration.indexing import (
    capture_result_coordinates,
    restore_result_coordinates,
)
from .boundary import EventBoundaryPayload
from .intervals import IntervalPayload
from .resolve import EventEvalContext

_REQUIRED_EVENT_VARS = ("time", "edge_code", "sample_index_before", "sample_index_after")
_REQUIRED_INTERVAL_VARS = ("time", "sample_index", "is_trigger", "valid_segment")


def _assert_table_var_namespace_safe(
    context: EventEvalContext,
    *,
    required: tuple[str, ...],
    owner: str,
    incoming: xr.DataArray | None = None,
) -> None:
    names = set(context.runtime.batch_dims)
    source = capture_result_coordinates(context.runtime.ds, output_dims=context.runtime.batch_dims, owner=owner)
    names.update(source.coordinates)
    names.update(name for group in source.indexes for name in group.coordinates)
    if incoming is not None:
        consumed_axes = set(incoming.dims) - set(context.runtime.batch_dims)
        names.update(name for name in incoming.coords if name not in consumed_axes)
    conflicts = sorted(set(required) & names)
    if conflicts:
        raise ValueError(
            f"{owner}: output variable names {conflicts!r} conflict with surviving coordinates/indexes; "
            "rename the conflicting dimensions or coordinates before extraction."
        )


def _assert_event_var_namespace_safe(
    context: EventEvalContext,
    *,
    owner: str,
    incoming: xr.DataArray | None = None,
) -> None:
    _assert_table_var_namespace_safe(context, required=_REQUIRED_EVENT_VARS, owner=owner, incoming=incoming)


def _assert_interval_var_namespace_safe(
    context: EventEvalContext,
    *,
    owner: str,
) -> None:
    _assert_table_var_namespace_safe(context, required=_REQUIRED_INTERVAL_VARS, owner=owner)


def _event_dim_name(
    payload: EventBoundaryPayload,
    *,
    context: EventEvalContext,
) -> str:
    taken = set(dataset_namespace_names(context.runtime.ds))
    taken.update(dataarray_namespace_names(payload.time))
    taken.update(_REQUIRED_EVENT_VARS)
    taken.update(context.runtime.batch_dims)
    taken.add(payload.event_dim)
    return unique_temp_dim("event", taken_dims=tuple(sorted(taken)))


def _rename_payload_dim(payload: EventBoundaryPayload, *, event_dim: str) -> EventBoundaryPayload:
    if event_dim == payload.event_dim:
        return payload
    mapping = {payload.event_dim: event_dim}
    return EventBoundaryPayload(
        time=payload.time.rename(mapping),
        edge_code=payload.edge_code.rename(mapping),
        sample_index_before=payload.sample_index_before.rename(mapping),
        sample_index_after=payload.sample_index_after.rename(mapping),
        event_dim=event_dim,
    )


def _interval_dims(
    payload: IntervalPayload,
    *,
    context: EventEvalContext,
) -> tuple[str, str]:
    taken = set(dataset_namespace_names(context.runtime.ds))
    taken.update(dataarray_namespace_names(payload.time))
    taken.update(_REQUIRED_INTERVAL_VARS)
    taken.update(context.runtime.batch_dims)
    taken.add(payload.segment_dim)
    taken.add(payload.edge_dim)
    segment_dim = unique_temp_dim("segment", taken_dims=tuple(sorted(taken)))
    taken.add(segment_dim)
    edge_dim = unique_temp_dim("edge", taken_dims=tuple(sorted(taken)))
    return segment_dim, edge_dim


def _rename_interval_payload(
    payload: IntervalPayload,
    *,
    segment_dim: str,
    edge_dim: str,
) -> IntervalPayload:
    if segment_dim == payload.segment_dim and edge_dim == payload.edge_dim:
        return payload
    mapping = {payload.segment_dim: segment_dim, payload.edge_dim: edge_dim}
    return IntervalPayload(
        time=payload.time.rename(mapping),
        sample_index=payload.sample_index.rename(mapping),
        is_trigger=payload.is_trigger.rename({payload.segment_dim: segment_dim}),
        valid_segment=payload.valid_segment.rename({payload.segment_dim: segment_dim}),
        segment_dim=segment_dim,
        edge_dim=edge_dim,
    )


def pack_event_table(
    payload: EventBoundaryPayload,
    *,
    context: EventEvalContext,
    owner: str,
) -> xr.Dataset:
    """Pack extracted boundaries into a deterministic event table dataset.

    Parameters
    ----------
    payload : EventBoundaryPayload
        Resolved runtime context/payload used by this orchestration boundary.
    context : EventEvalContext, optional
        Resolved runtime context/payload used by this orchestration boundary.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    _assert_event_var_namespace_safe(context, owner=owner, incoming=payload.time)
    event_dim = _event_dim_name(payload, context=context)
    packed = _rename_payload_dim(payload, event_dim=event_dim)
    ds = xr.Dataset(
        data_vars={
            "time": packed.time.astype("float64").variable,
            "edge_code": packed.edge_code.astype("int8").variable,
            "sample_index_before": packed.sample_index_before.astype("int64").variable,
            "sample_index_after": packed.sample_index_after.astype("int64").variable,
        },
    )
    coordinates = capture_result_coordinates(context.runtime.ds, packed.time, output_dims=tuple(ds.dims), owner=owner)
    ds = restore_result_coordinates(ds, coordinates)
    if event_dim in ds.coords:
        return ds
    return ds.assign_coords({event_dim: np.arange(ds.sizes[event_dim], dtype="int64")})


def pack_interval_table(
    payload: IntervalPayload,
    *,
    context: EventEvalContext,
    owner: str,
) -> xr.Dataset:
    """Pack extracted intervals into a deterministic segment table dataset.

    Parameters
    ----------
    payload : IntervalPayload
        Resolved runtime context/payload used by this orchestration boundary.
    context : EventEvalContext, optional
        Resolved runtime context/payload used by this orchestration boundary.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    xr.Dataset
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    _assert_interval_var_namespace_safe(context, owner=owner)
    segment_dim, edge_dim = _interval_dims(payload, context=context)
    packed = _rename_interval_payload(payload, segment_dim=segment_dim, edge_dim=edge_dim)
    ds = xr.Dataset(
        data_vars={
            "time": packed.time.astype("float64").variable,
            "sample_index": packed.sample_index.astype("int64").variable,
            "is_trigger": packed.is_trigger.astype(bool).variable,
            "valid_segment": packed.valid_segment.astype(bool).variable,
        },
    )
    coordinates = capture_result_coordinates(context.runtime.ds, packed.time, output_dims=tuple(ds.dims), owner=owner)
    ds = restore_result_coordinates(ds, coordinates)
    if segment_dim not in ds.coords:
        ds = ds.assign_coords({segment_dim: np.arange(ds.sizes[segment_dim], dtype="int64")})
    if edge_dim not in ds.coords:
        ds = ds.assign_coords({edge_dim: np.asarray(["start", "end"], dtype=object)})
    return ds


__all__ = ["pack_event_table", "pack_interval_table"]
