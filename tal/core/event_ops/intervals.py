from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .backends import EVENT_INTERVALS_BACKEND_NUMPY_ROW, intervals_bounded_row_backend
from ..orchestration.lazy import fail_if_chunked_boundary, is_chunked_dataarray
from .boundary import (
    EventBoundaryPayload,
    extract_event_boundaries,
)
from .event_primitives import (
    EDGE_ENTER,
    EDGE_EXIT,
    EDGE_INVALID,
    EDGE_TRIGGER,
    SAMPLE_SENTINEL,
    batch_dims,
    lane_data,
)
from .resolve import EventEvalContext
from .types import Condition, EventExtractOptions, IntervalExtractOptions

_INTERNAL_SEGMENT_DIM = "__tal_segment__"
_INTERNAL_EDGE_DIM = "__tal_edge__"
_EDGE_COORDS = np.asarray(["start", "end"], dtype=object)


@dataclass(frozen=True)
class IntervalPayload:
    """Packed interval arrays before interval table assembly.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    time: xr.DataArray
    sample_index: xr.DataArray
    is_trigger: xr.DataArray
    valid_segment: xr.DataArray
    segment_dim: str = _INTERNAL_SEGMENT_DIM
    edge_dim: str = _INTERNAL_EDGE_DIM


def _empty_segment_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), dtype="float64"),
        np.empty((0, 2), dtype="int64"),
        np.empty((0,), dtype=bool),
        np.empty((0,), dtype=bool),
    )


def _rows_to_segment_arrays(
    time_rows: list[tuple[float, float]],
    sample_rows: list[tuple[int, int]],
    trigger_rows: list[bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not time_rows:
        return _empty_segment_arrays()
    time = np.asarray(time_rows, dtype="float64")
    sample = np.asarray(sample_rows, dtype="int64")
    is_trigger = np.asarray(trigger_rows, dtype=bool)
    valid = np.ones(len(time_rows), dtype=bool)
    return time, sample, is_trigger, valid


def _dedupe_degenerate_trigger_collisions(
    time_rows: list[tuple[float, float]],
    sample_rows: list[tuple[int, int]],
    trigger_rows: list[bool],
) -> tuple[list[tuple[float, float]], list[tuple[int, int]], list[bool]]:
    trigger_keys = {
        (time_pair[0], time_pair[1], sample_pair[0], sample_pair[1])
        for time_pair, sample_pair, is_trigger in zip(time_rows, sample_rows, trigger_rows)
        if is_trigger and time_pair[0] == time_pair[1] and sample_pair[0] == sample_pair[1]
    }
    out_time: list[tuple[float, float]] = []
    out_sample: list[tuple[int, int]] = []
    out_trigger: list[bool] = []
    for time_pair, sample_pair, is_trigger in zip(time_rows, sample_rows, trigger_rows):
        key = (time_pair[0], time_pair[1], sample_pair[0], sample_pair[1])
        is_degenerate = time_pair[0] == time_pair[1] and sample_pair[0] == sample_pair[1]
        if (not is_trigger) and is_degenerate and key in trigger_keys:
            continue
        out_time.append(time_pair)
        out_sample.append(sample_pair)
        out_trigger.append(is_trigger)
    return out_time, out_sample, out_trigger


def _row_segments(
    time_row: np.ndarray,
    edge_row: np.ndarray,
    before_row: np.ndarray,
    after_row: np.ndarray,
    *,
    bounded: bool,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(edge_row, dtype="int8") != int(EDGE_INVALID)
    if not bool(valid.any()):
        return _empty_segment_arrays()
    time = np.asarray(time_row[valid], dtype="float64")
    edge = np.asarray(edge_row[valid], dtype="int8")
    before = np.asarray(before_row[valid], dtype="int64")
    after = np.asarray(after_row[valid], dtype="int64")
    time_rows: list[tuple[float, float]] = []
    sample_rows: list[tuple[int, int]] = []
    trigger_rows: list[bool] = []
    open_start: tuple[float, int] | None = None
    for idx in range(int(edge.size)):
        code = int(edge[idx])
        if code == int(EDGE_ENTER):
            open_start = (float(time[idx]), int(after[idx]))
            continue
        if code == int(EDGE_TRIGGER):
            sample = int(after[idx])
            time_rows.append((float(time[idx]), float(time[idx])))
            sample_rows.append((sample, sample))
            trigger_rows.append(True)
            continue
        if code == int(EDGE_EXIT) and open_start is not None:
            time_rows.append((open_start[0], float(time[idx])))
            sample_rows.append((open_start[1], int(before[idx])))
            trigger_rows.append(False)
            open_start = None
    if open_start is not None and not bounded:
        raise ValueError(f"{owner}: interval pairing failed due to unmatched trailing enter boundary.")
    time_rows, sample_rows, trigger_rows = _dedupe_degenerate_trigger_collisions(
        time_rows,
        sample_rows,
        trigger_rows,
    )
    return _rows_to_segment_arrays(time_rows, sample_rows, trigger_rows)


def _pad_segments(
    time: np.ndarray,
    sample: np.ndarray,
    is_trigger: np.ndarray,
    valid: np.ndarray,
    *,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_time = np.full((max_segments, 2), np.nan, dtype="float64")
    out_sample = np.full((max_segments, 2), SAMPLE_SENTINEL, dtype="int64")
    out_trigger = np.zeros(max_segments, dtype=bool)
    out_valid = np.zeros(max_segments, dtype=bool)
    n = min(int(time.shape[0]), max_segments)
    if n == 0:
        return out_time, out_sample, out_trigger, out_valid
    out_time[:n, :] = time[:n, :]
    out_sample[:n, :] = sample[:n, :]
    out_trigger[:n] = is_trigger[:n]
    out_valid[:n] = valid[:n]
    return out_time, out_sample, out_trigger, out_valid


def _bounded_row_kernel(
    time_row: np.ndarray,
    edge_row: np.ndarray,
    before_row: np.ndarray,
    after_row: np.ndarray,
    *,
    max_segments: int,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time, sample, is_trigger, valid = _row_segments(
        time_row,
        edge_row,
        before_row,
        after_row,
        bounded=True,
        owner=owner,
    )
    return _pad_segments(
        time,
        sample,
        is_trigger,
        valid,
        max_segments=max_segments,
    )


def _build_edge_array(
    values: np.ndarray,
    *,
    context: EventEvalContext,
    segment_dim: str,
    edge_dim: str,
    name: str,
) -> xr.DataArray:
    context_batch_dims = batch_dims(context)
    dims = context_batch_dims + (segment_dim, edge_dim)
    coords: dict[str, object] = {
        segment_dim: np.arange(values.shape[-2], dtype="int64"),
        edge_dim: _EDGE_COORDS,
    }
    for dim in context_batch_dims:
        coords[dim] = context.clock.coords[dim]
    return xr.DataArray(values, dims=dims, coords=coords, name=name)


def _build_segment_array(
    values: np.ndarray,
    *,
    context: EventEvalContext,
    segment_dim: str,
    name: str,
) -> xr.DataArray:
    context_batch_dims = batch_dims(context)
    dims = context_batch_dims + (segment_dim,)
    coords: dict[str, object] = {segment_dim: np.arange(values.shape[-1], dtype="int64")}
    for dim in context_batch_dims:
        coords[dim] = context.clock.coords[dim]
    return xr.DataArray(values, dims=dims, coords=coords, name=name)


def _dynamic_rows(
    payload: EventBoundaryPayload,
    *,
    context: EventEvalContext,
    owner: str,
) -> tuple[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    time_lanes = lane_data(
        payload.time.astype("float64"),
        context=context,
        core_dim=payload.event_dim,
    )
    edge_lanes = lane_data(
        payload.edge_code.astype("int8"),
        context=context,
        core_dim=payload.event_dim,
    )
    before_lanes = lane_data(
        payload.sample_index_before.astype("int64"),
        context=context,
        core_dim=payload.event_dim,
    )
    after_lanes = lane_data(
        payload.sample_index_after.astype("int64"),
        context=context,
        core_dim=payload.event_dim,
    )
    rows = [
        _row_segments(time_lanes[i], edge_lanes[i], before_lanes[i], after_lanes[i], bounded=False, owner=owner)
        for i in range(time_lanes.shape[0])
    ]
    return time_lanes.shape[0], rows


def _pack_dynamic_rows(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    lane_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    max_segments = max((time.shape[0] for time, _, _, _ in rows), default=0)
    time_out = np.full((lane_count, max_segments, 2), np.nan, dtype="float64")
    sample_out = np.full((lane_count, max_segments, 2), SAMPLE_SENTINEL, dtype="int64")
    trigger_out = np.zeros((lane_count, max_segments), dtype=bool)
    valid_out = np.zeros((lane_count, max_segments), dtype=bool)
    for lane, (time, sample, trig, valid) in enumerate(rows):
        n = int(time.shape[0])
        if n == 0:
            continue
        time_out[lane, :n, :] = time
        sample_out[lane, :n, :] = sample
        trigger_out[lane, :n] = trig
        valid_out[lane, :n] = valid
    return time_out, sample_out, trigger_out, valid_out, max_segments


def _reshape_dynamic(
    values: np.ndarray,
    *,
    context: EventEvalContext,
    max_segments: int,
    with_edge: bool,
) -> np.ndarray:
    batch_shape = tuple(context.clock.sizes[dim] for dim in batch_dims(context))
    if not batch_shape:
        return values.reshape((max_segments, 2) if with_edge else (max_segments,))
    target = batch_shape + ((max_segments, 2) if with_edge else (max_segments,))
    return values.reshape(target)


def _extract_dynamic(
    payload: EventBoundaryPayload,
    *,
    context: EventEvalContext,
    owner: str,
) -> IntervalPayload:
    lane_count, rows = _dynamic_rows(payload, context=context, owner=owner)
    time_out, sample_out, trigger_out, valid_out, max_segments = _pack_dynamic_rows(rows, lane_count=lane_count)
    return IntervalPayload(
        time=_build_edge_array(
            _reshape_dynamic(time_out, context=context, max_segments=max_segments, with_edge=True),
            context=context,
            segment_dim=_INTERNAL_SEGMENT_DIM,
            edge_dim=_INTERNAL_EDGE_DIM,
            name="time",
        ),
        sample_index=_build_edge_array(
            _reshape_dynamic(sample_out, context=context, max_segments=max_segments, with_edge=True),
            context=context,
            segment_dim=_INTERNAL_SEGMENT_DIM,
            edge_dim=_INTERNAL_EDGE_DIM,
            name="sample_index",
        ),
        is_trigger=_build_segment_array(
            _reshape_dynamic(trigger_out, context=context, max_segments=max_segments, with_edge=False),
            context=context,
            segment_dim=_INTERNAL_SEGMENT_DIM,
            name="is_trigger",
        ),
        valid_segment=_build_segment_array(
            _reshape_dynamic(valid_out, context=context, max_segments=max_segments, with_edge=False),
            context=context,
            segment_dim=_INTERNAL_SEGMENT_DIM,
            name="valid_segment",
        ),
    )


def _extract_bounded(
    payload: EventBoundaryPayload,
    *,
    max_segments: int,
    owner: str,
) -> IntervalPayload:
    chunked = any(
        is_chunked_dataarray(da)
        for da in (payload.time, payload.edge_code, payload.sample_index_before, payload.sample_index_after)
    )
    dask_mode = "parallelized" if chunked else "allowed"
    kwargs = {
        "max_segments": max_segments,
        "owner": owner,
        "backend": EVENT_INTERVALS_BACKEND_NUMPY_ROW,
    }
    ufunc_kwargs: dict[str, object] = {}
    if chunked:
        ufunc_kwargs["dask_gufunc_kwargs"] = {
            "output_sizes": {_INTERNAL_SEGMENT_DIM: max_segments, _INTERNAL_EDGE_DIM: 2},
            "allow_rechunk": True,
        }
    time, sample, is_trigger, valid = xr.apply_ufunc(
        intervals_bounded_row_backend,
        payload.time.astype("float64"),
        payload.edge_code.astype("int8"),
        payload.sample_index_before.astype("int64"),
        payload.sample_index_after.astype("int64"),
        input_core_dims=[[payload.event_dim]] * 4,
        output_core_dims=[
            [_INTERNAL_SEGMENT_DIM, _INTERNAL_EDGE_DIM],
            [_INTERNAL_SEGMENT_DIM, _INTERNAL_EDGE_DIM],
            [_INTERNAL_SEGMENT_DIM],
            [_INTERNAL_SEGMENT_DIM],
        ],
        kwargs=kwargs,
        vectorize=True,
        dask=dask_mode,
        output_dtypes=[np.float64, np.int64, np.bool_, np.bool_],
        **ufunc_kwargs,
    )
    segment_coord = np.arange(max_segments, dtype="int64")
    edge_coord = _EDGE_COORDS
    return IntervalPayload(
        time=time.assign_coords({_INTERNAL_SEGMENT_DIM: segment_coord, _INTERNAL_EDGE_DIM: edge_coord}).rename("time"),
        sample_index=sample.assign_coords(
            {_INTERNAL_SEGMENT_DIM: segment_coord, _INTERNAL_EDGE_DIM: edge_coord}
        ).rename("sample_index"),
        is_trigger=is_trigger.assign_coords({_INTERNAL_SEGMENT_DIM: segment_coord}).rename("is_trigger"),
        valid_segment=valid.assign_coords({_INTERNAL_SEGMENT_DIM: segment_coord}).rename("valid_segment"),
    )


def _max_events(max_segments: int | None) -> int | None:
    if max_segments is None:
        return None
    return int(max_segments) * 3


def _boundary_options(opts: IntervalExtractOptions) -> EventExtractOptions:
    return EventExtractOptions(
        eval=opts.eval,
        include_initial=opts.include_initial,
        truth_eval=opts.truth_eval,
        max_events=_max_events(opts.max_segments),
    )


def _preflight_chunked(
    *,
    effective_mask: xr.DataArray,
    context: EventEvalContext,
    opts: IntervalExtractOptions,
    owner: str,
) -> None:
    chunked = any(
        is_chunked_dataarray(da)
        for da in (effective_mask, context.valid_mask, context.clock)
    )
    fail_if_chunked_boundary(
        (opts.max_segments is None) and chunked,
        owner=owner,
        message="chunked interval extraction requires opts.max_segments for bounded output sizing.",
    )


def extract_intervals(
    condition: Condition,
    *,
    effective_mask: xr.DataArray,
    context: EventEvalContext,
    opts: IntervalExtractOptions,
    owner: str,
    emit_triggers: bool | None = None,
) -> IntervalPayload:
    """Extract intervals from condition boundaries on the evaluation clock.

    Parameters
    ----------
    condition : Condition
        Condition/expression used for event or mask evaluation.
    effective_mask : xr.DataArray, optional
        Validity/mask payload used by this operation.
    context : EventEvalContext, optional
        Resolved runtime context/payload used by this orchestration boundary.
    opts : IntervalExtractOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    emit_triggers : bool | None, optional
        Behavior flag/policy controlling boundary semantics.

    Returns
    -------
    IntervalPayload
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    _preflight_chunked(effective_mask=effective_mask, context=context, opts=opts, owner=owner)
    boundaries = extract_event_boundaries(
        condition,
        effective_mask=effective_mask,
        context=context,
        opts=_boundary_options(opts),
        owner=owner,
        emit_triggers=emit_triggers,
    )
    if opts.max_segments is None:
        return _extract_dynamic(boundaries, context=context, owner=owner)
    return _extract_bounded(
        boundaries,
        max_segments=int(opts.max_segments),
        owner=owner,
    )


__all__ = ["IntervalPayload", "extract_intervals"]
