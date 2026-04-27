from __future__ import annotations

"""Shared owner for event-boundary edge/mode row selection."""

from typing import Literal

import numpy as np
import xarray as xr

from ..orchestration.lazy import fail_if_chunked_boundary, is_chunked_dataarray
from .boundary import EventBoundaryPayload
from .event_primitives import EDGE_ENTER, EDGE_EXIT, EDGE_INVALID, SAMPLE_SENTINEL, batch_dims, lane_data
from .resolve import EventEvalContext


def _selected_edge_mask(edge_row: np.ndarray, *, edges: str) -> np.ndarray:
    if edges == "all":
        return (edge_row == int(EDGE_ENTER)) | (edge_row == int(EDGE_EXIT))
    if edges == "enter":
        return edge_row == int(EDGE_ENTER)
    return edge_row == int(EDGE_EXIT)


def _selection_output_len(*, mode: str, max_events: int | None, max_selected: int) -> int:
    if mode in ("first", "last"):
        return 1
    if mode == "first_n":
        return int(max_events or 0)
    if max_events is not None:
        return int(max_events)
    return int(max_selected)


def _select_row_indices(
    edge_row: np.ndarray,
    *,
    edges: str,
    mode: str,
    max_events: int | None,
) -> np.ndarray:
    edge = np.asarray(edge_row, dtype="int8")
    keep = (edge != int(EDGE_INVALID)) & _selected_edge_mask(edge, edges=edges)
    idx = np.flatnonzero(keep)
    if mode == "all":
        return idx if max_events is None else idx[: int(max_events)]
    if mode == "first":
        return idx[:1]
    if mode == "last":
        return idx[-1:] if idx.size else idx
    return idx[: int(max_events or 0)]


def _row_selected_arrays(
    time_row: np.ndarray,
    edge_row: np.ndarray,
    before_row: np.ndarray,
    after_row: np.ndarray,
    *,
    edges: str,
    mode: str,
    max_events: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = _select_row_indices(edge_row, edges=edges, mode=mode, max_events=max_events)
    if idx.size == 0:
        empty_f = np.asarray([], dtype="float64")
        empty_i8 = np.asarray([], dtype="int8")
        empty_i64 = np.asarray([], dtype="int64")
        return empty_f, empty_i8, empty_i64, empty_i64
    return (
        np.asarray(time_row[idx], dtype="float64"),
        np.asarray(edge_row[idx], dtype="int8"),
        np.asarray(before_row[idx], dtype="int64"),
        np.asarray(after_row[idx], dtype="int64"),
    )


def _pad_selected(
    time: np.ndarray,
    edge: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    out_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_time = np.full(out_len, np.nan, dtype="float64")
    out_edge = np.full(out_len, int(EDGE_INVALID), dtype="int8")
    out_before = np.full(out_len, SAMPLE_SENTINEL, dtype="int64")
    out_after = np.full(out_len, SAMPLE_SENTINEL, dtype="int64")
    n = min(int(time.size), int(out_len))
    if n:
        out_time[:n] = time[:n]
        out_edge[:n] = edge[:n]
        out_before[:n] = before[:n]
        out_after[:n] = after[:n]
    return out_time, out_edge, out_before, out_after


def _flat_row_count(shape: tuple[int, ...]) -> int:
    return 1 if not shape else int(np.prod(shape, dtype=np.int64))


def _selected_positions(
    keep_rows: np.ndarray,
    *,
    mode: str,
    out_len: int,
) -> np.ndarray:
    rows, event_count = keep_rows.shape
    selected = np.full((rows, out_len), -1, dtype="int64")
    if out_len == 0 or event_count == 0:
        return selected
    if mode == "last":
        keep_rev = keep_rows[:, ::-1]
        rank_rev = np.cumsum(keep_rev, axis=1) - 1
        take_rev = keep_rev & (rank_rev < out_len)
        row_idx, col_idx = np.nonzero(take_rev)
        selected[row_idx, rank_rev[row_idx, col_idx]] = event_count - 1 - col_idx
        return selected
    rank = np.cumsum(keep_rows, axis=1) - 1
    take = keep_rows & (rank < out_len)
    row_idx, col_idx = np.nonzero(take)
    selected[row_idx, rank[row_idx, col_idx]] = col_idx
    return selected


def _bounded_select_block(
    time_block: np.ndarray,
    edge_block: np.ndarray,
    before_block: np.ndarray,
    after_block: np.ndarray,
    *,
    edges: str,
    mode: str,
    out_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outer_shape = time_block.shape[:-1]
    row_count = _flat_row_count(outer_shape)
    event_count = int(time_block.shape[-1])
    if event_count == 0:
        target = outer_shape + (out_len,)
        return (
            np.full(target, np.nan, dtype="float64"),
            np.full(target, int(EDGE_INVALID), dtype="int8"),
            np.full(target, SAMPLE_SENTINEL, dtype="int64"),
            np.full(target, SAMPLE_SENTINEL, dtype="int64"),
        )
    time_rows = time_block.reshape(row_count, event_count)
    edge_rows = edge_block.reshape(row_count, event_count)
    before_rows = before_block.reshape(row_count, event_count)
    after_rows = after_block.reshape(row_count, event_count)
    keep_rows = (edge_rows != int(EDGE_INVALID)) & _selected_edge_mask(edge_rows, edges=edges)
    selected = _selected_positions(keep_rows, mode=mode, out_len=out_len)
    safe_selected = np.where(selected >= 0, selected, 0)
    keep_out = selected >= 0
    time = np.where(keep_out, np.take_along_axis(time_rows, safe_selected, axis=1), np.nan)
    edge = np.where(keep_out, np.take_along_axis(edge_rows, safe_selected, axis=1), int(EDGE_INVALID))
    before = np.where(
        keep_out,
        np.take_along_axis(before_rows, safe_selected, axis=1),
        SAMPLE_SENTINEL,
    )
    after = np.where(
        keep_out,
        np.take_along_axis(after_rows, safe_selected, axis=1),
        SAMPLE_SENTINEL,
    )
    target = outer_shape + (out_len,)
    return (
        time.reshape(target).astype("float64"),
        edge.reshape(target).astype("int8"),
        before.reshape(target).astype("int64"),
        after.reshape(target).astype("int64"),
    )


def _build_selected_array(
    values: np.ndarray,
    *,
    context: EventEvalContext,
    event_dim: str,
    name: str,
) -> xr.DataArray:
    context_batch_dims = batch_dims(context)
    dims = context_batch_dims + (event_dim,)
    coords: dict[str, object] = {event_dim: np.arange(values.shape[-1], dtype="int64")}
    for dim in context_batch_dims:
        coords[dim] = context.clock.coords[dim]
    return xr.DataArray(values, dims=dims, coords=coords, name=name)


def _dynamic_shape(context: EventEvalContext, *, out_len: int) -> tuple[int, ...]:
    shape = tuple(context.clock.sizes[dim] for dim in batch_dims(context))
    return (out_len,) if not shape else shape + (out_len,)


def _selected_rows(
    payload: EventBoundaryPayload,
    *,
    context: EventEvalContext,
    edges: str,
    mode: str,
    max_events: int | None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    time_lanes = lane_data(payload.time.astype("float64"), context=context, core_dim=payload.event_dim)
    edge_lanes = lane_data(payload.edge_code.astype("int8"), context=context, core_dim=payload.event_dim)
    before_lanes = lane_data(payload.sample_index_before.astype("int64"), context=context, core_dim=payload.event_dim)
    after_lanes = lane_data(payload.sample_index_after.astype("int64"), context=context, core_dim=payload.event_dim)
    rows = [
        _row_selected_arrays(
            time_lanes[idx],
            edge_lanes[idx],
            before_lanes[idx],
            after_lanes[idx],
            edges=edges,
            mode=mode,
            max_events=max_events,
        )
        for idx in range(time_lanes.shape[0])
    ]
    return time_lanes, rows


def _pack_dynamic_rows(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    lane_count: int,
    out_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.full((lane_count, out_len), np.nan, dtype="float64")
    edge = np.full((lane_count, out_len), int(EDGE_INVALID), dtype="int8")
    before = np.full((lane_count, out_len), SAMPLE_SENTINEL, dtype="int64")
    after = np.full((lane_count, out_len), SAMPLE_SENTINEL, dtype="int64")
    for lane, (t, e, b, a) in enumerate(rows):
        n = min(int(t.size), out_len)
        if n:
            time[lane, :n], edge[lane, :n] = t[:n], e[:n]
            before[lane, :n], after[lane, :n] = b[:n], a[:n]
    return time, edge, before, after


def _payload_from_arrays(
    *,
    time: np.ndarray,
    edge: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    context: EventEvalContext,
    event_dim: str,
    out_len: int,
) -> EventBoundaryPayload:
    shape = _dynamic_shape(context, out_len=out_len)
    return EventBoundaryPayload(
        time=_build_selected_array(time.reshape(shape), context=context, event_dim=event_dim, name="time"),
        edge_code=_build_selected_array(edge.reshape(shape), context=context, event_dim=event_dim, name="edge_code"),
        sample_index_before=_build_selected_array(
            before.reshape(shape),
            context=context,
            event_dim=event_dim,
            name="sample_index_before",
        ),
        sample_index_after=_build_selected_array(
            after.reshape(shape),
            context=context,
            event_dim=event_dim,
            name="sample_index_after",
        ),
        event_dim=event_dim,
    )


def _select_boundaries_dynamic(
    payload: EventBoundaryPayload,
    *,
    context: EventEvalContext,
    edges: str,
    mode: str,
    max_events: int | None,
) -> EventBoundaryPayload:
    time_lanes, rows = _selected_rows(
        payload,
        context=context,
        edges=edges,
        mode=mode,
        max_events=max_events,
    )
    out_len = _selection_output_len(
        mode=mode,
        max_events=max_events,
        max_selected=max((row[0].size for row in rows), default=0),
    )
    time, edge, before, after = _pack_dynamic_rows(
        rows,
        lane_count=time_lanes.shape[0],
        out_len=out_len,
    )
    return _payload_from_arrays(
        time=time,
        edge=edge,
        before=before,
        after=after,
        context=context,
        event_dim=payload.event_dim,
        out_len=out_len,
    )


def _select_boundaries_bounded(
    payload: EventBoundaryPayload,
    *,
    edges: str,
    mode: str,
    max_events: int,
) -> EventBoundaryPayload:
    out_dim = "__tal_selected_event__"
    out_len = _selection_output_len(mode=mode, max_events=max_events, max_selected=0)
    chunked = any(
        is_chunked_dataarray(da)
        for da in (payload.time, payload.edge_code, payload.sample_index_before, payload.sample_index_after)
    )
    kwargs = {"edges": edges, "mode": mode, "out_len": out_len}
    ufunc_kwargs: dict[str, object] = {}
    if chunked:
        ufunc_kwargs["dask_gufunc_kwargs"] = {"output_sizes": {out_dim: out_len}, "allow_rechunk": True}
    time, edge, before, after = xr.apply_ufunc(
        _bounded_select_block,
        payload.time.astype("float64"),
        payload.edge_code.astype("int8"),
        payload.sample_index_before.astype("int64"),
        payload.sample_index_after.astype("int64"),
        input_core_dims=[[payload.event_dim]] * 4,
        output_core_dims=[[out_dim]] * 4,
        kwargs=kwargs,
        vectorize=False,
        dask="parallelized" if chunked else "allowed",
        output_dtypes=[np.float64, np.int8, np.int64, np.int64],
        **ufunc_kwargs,
    )
    coord = np.arange(out_len, dtype="int64")
    rename = {out_dim: payload.event_dim}
    return EventBoundaryPayload(
        time=time.assign_coords({out_dim: coord}).rename(rename).rename("time"),
        edge_code=edge.assign_coords({out_dim: coord}).rename(rename).rename("edge_code"),
        sample_index_before=before.assign_coords({out_dim: coord}).rename(rename).rename("sample_index_before"),
        sample_index_after=after.assign_coords({out_dim: coord}).rename(rename).rename("sample_index_after"),
        event_dim=payload.event_dim,
    )


def _preflight_selection(
    payload: EventBoundaryPayload,
    *,
    mode: str,
    max_events: int | None,
    on_empty: Literal["empty", "error"],
    owner: str,
) -> None:
    chunked = any(
        is_chunked_dataarray(da)
        for da in (payload.time, payload.edge_code, payload.sample_index_before, payload.sample_index_after)
    )
    fail_if_chunked_boundary(
        chunked and max_events is None,
        owner=owner,
        message="chunked boundary selection requires opts.max_events for bounded output sizing.",
    )
    fail_if_chunked_boundary(
        chunked and mode == "last",
        owner=owner,
        message="opts.mode='last' requires unchunked boundary selection for exact semantics.",
    )
    fail_if_chunked_boundary(
        chunked and on_empty == "error",
        owner=owner,
        message="opts.on_empty='error' requires unchunked boundary selection.",
    )


def _selected_any(payload: EventBoundaryPayload) -> bool:
    return bool(np.any(np.asarray(payload.edge_code.data) != int(EDGE_INVALID)))


def _is_chunked_payload(payload: EventBoundaryPayload) -> bool:
    return any(
        is_chunked_dataarray(da)
        for da in (payload.time, payload.edge_code, payload.sample_index_before, payload.sample_index_after)
    )


def _collapse_empty_unchunked(
    payload: EventBoundaryPayload,
    *,
    on_empty: Literal["empty", "error"],
) -> EventBoundaryPayload:
    if on_empty != "empty" or _is_chunked_payload(payload) or _selected_any(payload):
        return payload
    indexer = {payload.event_dim: slice(0, 0)}
    return EventBoundaryPayload(
        time=payload.time.isel(indexer),
        edge_code=payload.edge_code.isel(indexer),
        sample_index_before=payload.sample_index_before.isel(indexer),
        sample_index_after=payload.sample_index_after.isel(indexer),
        event_dim=payload.event_dim,
    )


def _enforce_on_empty(
    payload: EventBoundaryPayload,
    *,
    on_empty: Literal["empty", "error"],
    edges: str,
    mode: str,
    owner: str,
) -> None:
    if on_empty == "error" and not _selected_any(payload):
        raise ValueError(f"{owner}: no selected boundaries for opts.edges={edges!r} and opts.mode={mode!r}.")


def select_event_boundaries(
    payload: EventBoundaryPayload,
    *,
    context: EventEvalContext,
    edges: Literal["all", "enter", "exit"],
    mode: Literal["all", "first", "first_n", "last"],
    max_events: int | None,
    on_empty: Literal["empty", "error"],
    owner: str,
) -> EventBoundaryPayload:
    """Select boundary rows by edge and mode with deterministic packing.

    Parameters
    ----------
    payload : EventBoundaryPayload
        Resolved runtime context/payload used by this orchestration boundary.
    context : EventEvalContext, optional
        Resolved runtime context/payload used by this orchestration boundary.
    edges : Literal['all', 'enter', 'exit'], optional
        Edge/event payload consumed by this operation.
    mode : Literal['all', 'first', 'first_n', 'last'], optional
        Policy selector controlling alignment/join behavior.
    max_events : int | None, optional
        Bound/limit value controlling operation behavior.
    on_empty : Literal['empty', 'error'], optional
        Policy controlling behavior when no matching rows/events are found.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    EventBoundaryPayload
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    _preflight_selection(
        payload,
        mode=mode,
        max_events=max_events,
        on_empty=on_empty,
        owner=owner,
    )
    selected = (
        _select_boundaries_dynamic(
            payload,
            context=context,
            edges=edges,
            mode=mode,
            max_events=max_events,
        )
        if max_events is None
        else _select_boundaries_bounded(payload, edges=edges, mode=mode, max_events=max_events)
    )
    selected = _collapse_empty_unchunked(selected, on_empty=on_empty)
    _enforce_on_empty(selected, on_empty=on_empty, edges=edges, mode=mode, owner=owner)
    return selected


__all__ = ["select_event_boundaries"]
