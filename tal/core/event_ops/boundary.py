from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .backends import (
    EVENT_BOUNDARY_BACKEND_NUMBA,
    EVENT_BOUNDARY_BACKEND_NUMPY_BLOCK,
    boundary_bounded_block_backend,
)
from ..orchestration.lazy import fail_if_chunked_boundary, is_chunked_dataarray
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
from .types import CompareNode, Condition, EventExtractOptions
from tal.utils.numba_support import _numba_available

_INTERNAL_EVENT_DIM = "__tal_event__"


@dataclass(frozen=True)
class EventBoundaryPayload:
    """Packed boundary arrays before event table assembly.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    time: xr.DataArray
    edge_code: xr.DataArray
    sample_index_before: xr.DataArray
    sample_index_after: xr.DataArray
    event_dim: str = _INTERNAL_EVENT_DIM


def _empty_candidate_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([], dtype="float64"),
        np.asarray([], dtype="int8"),
        np.asarray([], dtype="int64"),
        np.asarray([], dtype="int64"),
    )


def _transition_rows(
    *,
    valid_idx: np.ndarray,
    mask_vals: np.ndarray,
    clock_vals: np.ndarray,
    include_initial: bool,
) -> list[tuple[float, int, int, int]]:
    rows: list[tuple[float, int, int, int]] = []
    if include_initial and bool(mask_vals[0]):
        rows.append((float(clock_vals[0]), int(EDGE_ENTER), -1, int(valid_idx[0])))
    for pos in range(1, int(valid_idx.size)):
        prev_true = bool(mask_vals[pos - 1])
        cur_true = bool(mask_vals[pos])
        if not prev_true and cur_true:
            rows.append((float(clock_vals[pos]), int(EDGE_ENTER), int(valid_idx[pos - 1]), int(valid_idx[pos])))
        if prev_true and not cur_true:
            rows.append((float(clock_vals[pos - 1]), int(EDGE_EXIT), int(valid_idx[pos - 1]), int(valid_idx[pos])))
    if bool(mask_vals[-1]):
        rows.append((float(clock_vals[-1]), int(EDGE_EXIT), int(valid_idx[-1]), -1))
    return rows


def _trigger_rows(
    *,
    valid_idx: np.ndarray,
    mask_vals: np.ndarray,
    clock_vals: np.ndarray,
) -> list[tuple[float, int, int, int]]:
    rows: list[tuple[float, int, int, int]] = []
    trigger_idx = np.flatnonzero(mask_vals)
    for pos in trigger_idx:
        sample = int(valid_idx[int(pos)])
        rows.append((float(clock_vals[int(pos)]), int(EDGE_TRIGGER), sample, sample))
    return rows


def _rows_to_arrays(
    rows: list[tuple[float, int, int, int]],
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        return _empty_candidate_arrays()
    times = np.asarray([row[0] for row in rows], dtype="float64")
    if not np.isfinite(times).all():
        raise ValueError(f"{owner}: extracted event boundaries include non-finite clock values.")
    edge = np.asarray([row[1] for row in rows], dtype="int8")
    before = np.asarray([row[2] for row in rows], dtype="int64")
    after = np.asarray([row[3] for row in rows], dtype="int64")
    return times, edge, before, after


def _edge_sort_key(edge: np.ndarray) -> np.ndarray:
    lookup = np.asarray([3, 0, 2, 1], dtype="int8")
    return lookup[np.asarray(edge, dtype="int8")]


def _sort_dedupe(
    times: np.ndarray,
    edge: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    dedupe_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if times.size == 0:
        return times, edge, before, after
    order = np.lexsort((_edge_sort_key(edge), times))
    times = times[order]
    edge = edge[order]
    before = before[order]
    after = after[order]
    keep = np.ones(times.size, dtype=bool)
    seen: dict[int, float] = {}
    for idx in range(int(times.size)):
        code = int(edge[idx])
        prev = seen.get(code)
        now = float(times[idx])
        if prev is not None and abs(now - prev) <= dedupe_atol:
            keep[idx] = False
            continue
        seen[code] = now
    return times[keep], edge[keep], before[keep], after[keep]


def _row_candidates(
    mask_row: np.ndarray,
    valid_row: np.ndarray,
    clock_row: np.ndarray,
    *,
    include_initial: bool,
    emit_triggers: bool,
    dedupe_atol: float,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_idx = np.flatnonzero(valid_row)
    if valid_idx.size == 0:
        return _empty_candidate_arrays()
    mask_vals = np.asarray(mask_row[valid_idx], dtype=bool)
    clock_vals = np.asarray(clock_row[valid_idx], dtype="float64")
    rows = _transition_rows(
        valid_idx=valid_idx,
        mask_vals=mask_vals,
        clock_vals=clock_vals,
        include_initial=include_initial,
    )
    if emit_triggers:
        rows.extend(_trigger_rows(valid_idx=valid_idx, mask_vals=mask_vals, clock_vals=clock_vals))
    times, edge, before, after = _rows_to_arrays(rows, owner=owner)
    return _sort_dedupe(times, edge, before, after, dedupe_atol=dedupe_atol)


def _pad_to_max_events(
    times: np.ndarray,
    edge: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_time = np.full(max_events, np.nan, dtype="float64")
    out_edge = np.zeros(max_events, dtype="int8")
    out_before = np.full(max_events, SAMPLE_SENTINEL, dtype="int64")
    out_after = np.full(max_events, SAMPLE_SENTINEL, dtype="int64")
    if max_events == 0:
        return out_time, out_edge, out_before, out_after
    n = min(int(times.size), max_events)
    out_time[:n] = times[:n]
    out_edge[:n] = edge[:n]
    out_before[:n] = before[:n]
    out_after[:n] = after[:n]
    return out_time, out_edge, out_before, out_after


def _bounded_row_kernel(
    mask_row: np.ndarray,
    valid_row: np.ndarray,
    clock_row: np.ndarray,
    *,
    include_initial: bool,
    emit_triggers: bool,
    dedupe_atol: float,
    max_events: int,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times, edge, before, after = _row_candidates(
        mask_row,
        valid_row,
        clock_row,
        include_initial=include_initial,
        emit_triggers=emit_triggers,
        dedupe_atol=dedupe_atol,
        owner=owner,
    )
    return _pad_to_max_events(
        times,
        edge,
        before,
        after,
        max_events=max_events,
    )


def _build_dataarray(
    values: np.ndarray,
    *,
    context: EventEvalContext,
    name: str,
) -> xr.DataArray:
    context_batch_dims = batch_dims(context)
    dims = context_batch_dims + (_INTERNAL_EVENT_DIM,)
    coords: dict[str, object] = {_INTERNAL_EVENT_DIM: np.arange(values.shape[-1], dtype="int64")}
    for dim in context_batch_dims:
        coords[dim] = context.clock.coords[dim]
    return xr.DataArray(values, dims=dims, coords=coords, name=name)


def _extract_dynamic(
    *,
    effective_mask: xr.DataArray,
    context: EventEvalContext,
    opts: EventExtractOptions,
    emit_triggers: bool,
    owner: str,
) -> EventBoundaryPayload:
    sequence_dim = context.runtime.sequence_dim
    mask_lanes = lane_data(effective_mask.astype(bool), context=context, core_dim=sequence_dim)
    valid_lanes = lane_data(context.valid_mask.astype(bool), context=context, core_dim=sequence_dim)
    clock_lanes = lane_data(context.clock.astype("float64"), context=context, core_dim=sequence_dim)
    rows = [
        _row_candidates(
            mask_lanes[i],
            valid_lanes[i],
            clock_lanes[i],
            include_initial=opts.include_initial,
            emit_triggers=emit_triggers,
            dedupe_atol=float(opts.dedupe_atol),
            owner=owner,
        )
        for i in range(mask_lanes.shape[0])
    ]
    max_events = max((times.size for times, _, _, _ in rows), default=0)
    times = np.full((mask_lanes.shape[0], max_events), np.nan, dtype="float64")
    edge = np.zeros((mask_lanes.shape[0], max_events), dtype="int8")
    before = np.full((mask_lanes.shape[0], max_events), SAMPLE_SENTINEL, dtype="int64")
    after = np.full((mask_lanes.shape[0], max_events), SAMPLE_SENTINEL, dtype="int64")
    for lane, (t, e, b, a) in enumerate(rows):
        n = int(t.size)
        if n == 0:
            continue
        times[lane, :n] = t
        edge[lane, :n] = e
        before[lane, :n] = b
        after[lane, :n] = a
    batch_shape = tuple(context.clock.sizes[dim] for dim in batch_dims(context))
    target_shape = (max_events,) if not batch_shape else batch_shape + (max_events,)
    return EventBoundaryPayload(
        time=_build_dataarray(times.reshape(target_shape), context=context, name="time"),
        edge_code=_build_dataarray(edge.reshape(target_shape), context=context, name="edge_code"),
        sample_index_before=_build_dataarray(before.reshape(target_shape), context=context, name="sample_index_before"),
        sample_index_after=_build_dataarray(after.reshape(target_shape), context=context, name="sample_index_after"),
    )


def _select_boundary_normal_backend() -> str:
    if _numba_available():
        return EVENT_BOUNDARY_BACKEND_NUMBA
    return EVENT_BOUNDARY_BACKEND_NUMPY_BLOCK


def _extract_bounded(
    *,
    effective_mask: xr.DataArray,
    context: EventEvalContext,
    opts: EventExtractOptions,
    emit_triggers: bool,
    owner: str,
) -> EventBoundaryPayload:
    max_events = int(opts.max_events or 0)
    chunked = any(
        is_chunked_dataarray(da) for da in (effective_mask, context.valid_mask, context.clock)
    )
    dask_mode = "parallelized" if chunked else "allowed"
    kwargs = {
        "include_initial": bool(opts.include_initial),
        "emit_triggers": bool(emit_triggers),
        "dedupe_atol": float(opts.dedupe_atol),
        "max_events": max_events,
        "owner": owner,
        "backend": _select_boundary_normal_backend(),
    }
    ufunc_kwargs: dict[str, object] = {}
    if chunked:
        ufunc_kwargs["dask_gufunc_kwargs"] = {
            "output_sizes": {_INTERNAL_EVENT_DIM: max_events},
            "allow_rechunk": True,
        }
    time, edge, before, after = xr.apply_ufunc(
        boundary_bounded_block_backend,
        effective_mask.astype(bool),
        context.valid_mask.astype(bool),
        context.clock.astype("float64"),
        input_core_dims=[[context.runtime.sequence_dim]] * 3,
        output_core_dims=[[_INTERNAL_EVENT_DIM]] * 4,
        kwargs=kwargs,
        vectorize=False,
        dask=dask_mode,
        output_dtypes=[np.float64, np.int8, np.int64, np.int64],
        **ufunc_kwargs,
    )
    coord = np.arange(max_events, dtype="int64")
    return EventBoundaryPayload(
        time=time.assign_coords({_INTERNAL_EVENT_DIM: coord}).rename("time"),
        edge_code=edge.assign_coords({_INTERNAL_EVENT_DIM: coord}).rename("edge_code"),
        sample_index_before=before.assign_coords({_INTERNAL_EVENT_DIM: coord}).rename("sample_index_before"),
        sample_index_after=after.assign_coords({_INTERNAL_EVENT_DIM: coord}).rename("sample_index_after"),
    )


def _emit_triggers(condition: Condition) -> bool:
    return isinstance(condition.node, CompareNode) and condition.node.op == "eq"


def _supports_dynamic_extraction(
    *,
    effective_mask: xr.DataArray,
    context: EventEvalContext,
) -> bool:
    return not any(
        is_chunked_dataarray(da)
        for da in (effective_mask, context.valid_mask, context.clock)
    )


def extract_event_boundaries(
    condition: Condition,
    *,
    effective_mask: xr.DataArray,
    context: EventEvalContext,
    opts: EventExtractOptions,
    owner: str,
    emit_triggers: bool | None = None,
) -> EventBoundaryPayload:
    """Extract enter/exit/trigger boundaries from an effective condition mask.

    Parameters
    ----------
    condition : Condition
        Condition/expression used for event or mask evaluation.
    effective_mask : xr.DataArray, optional
        Validity/mask payload used by this operation.
    context : EventEvalContext, optional
        Resolved runtime context/payload used by this orchestration boundary.
    opts : EventExtractOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    emit_triggers : bool | None, optional
        Behavior flag/policy controlling boundary semantics.

    Returns
    -------
    EventBoundaryPayload
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    emit = _emit_triggers(condition) if emit_triggers is None else bool(emit_triggers)
    can_dynamic = _supports_dynamic_extraction(effective_mask=effective_mask, context=context)
    fail_if_chunked_boundary(
        (opts.max_events is None) and (not can_dynamic),
        owner=owner,
        message="chunked event extraction requires opts.max_events for bounded output sizing.",
    )
    if opts.max_events is None:
        return _extract_dynamic(
            effective_mask=effective_mask,
            context=context,
            opts=opts,
            emit_triggers=emit,
            owner=owner,
        )
    return _extract_bounded(
        effective_mask=effective_mask,
        context=context,
        opts=opts,
        emit_triggers=emit,
        owner=owner,
    )


__all__ = ["EventBoundaryPayload", "extract_event_boundaries"]
