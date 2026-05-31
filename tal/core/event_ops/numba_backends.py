from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._event_constants import EDGE_ENTER, EDGE_EXIT, EDGE_INVALID, EDGE_TRIGGER, SAMPLE_SENTINEL
from tal.utils.numba_support import require_numba

_STATUS_OK = 0
_STATUS_NONFINITE_TIME = 1
_HELPERS_JITTED = False


def _row_count(shape: tuple[int, ...]) -> int:
    rows = 1
    for size in shape:
        rows *= int(size)
    return rows


@lru_cache(maxsize=1)
def _compiled_boundary_block():
    numba = require_numba("events.boundaries")
    _jit_kernel_helpers(numba)
    return numba.njit(cache=True, fastmath=False)(_boundary_block_impl)


@lru_cache(maxsize=1)
def _compiled_intervals_block():
    numba = require_numba("events.intervals")
    _jit_kernel_helpers(numba)
    return numba.njit(cache=True, fastmath=False)(_intervals_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _HELPERS_JITTED
    global _append_boundary, _boundary_row_impl, _collect_interval_segments, _dedupe_boundaries
    global _edge_rank, _interval_has_trigger_collision
    global _intervals_row_impl, _sort_boundaries, _transition_candidates
    global _trigger_candidates, _write_interval_segments
    if _HELPERS_JITTED:
        return
    _edge_rank = numba.njit(cache=True, fastmath=False)(_edge_rank)
    _append_boundary = numba.njit(cache=True, fastmath=False)(_append_boundary)
    _sort_boundaries = numba.njit(cache=True, fastmath=False)(_sort_boundaries)
    _dedupe_boundaries = numba.njit(cache=True, fastmath=False)(_dedupe_boundaries)
    _transition_candidates = numba.njit(cache=True, fastmath=False)(_transition_candidates)
    _trigger_candidates = numba.njit(cache=True, fastmath=False)(_trigger_candidates)
    _boundary_row_impl = numba.njit(cache=True, fastmath=False)(_boundary_row_impl)
    _interval_has_trigger_collision = numba.njit(cache=True, fastmath=False)(_interval_has_trigger_collision)
    _collect_interval_segments = numba.njit(cache=True, fastmath=False)(_collect_interval_segments)
    _write_interval_segments = numba.njit(cache=True, fastmath=False)(_write_interval_segments)
    _intervals_row_impl = numba.njit(cache=True, fastmath=False)(_intervals_row_impl)
    _HELPERS_JITTED = True


def _broadcast_boundary_blocks(
    mask_block: np.ndarray,
    valid_block: np.ndarray,
    clock_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    mask = np.asarray(mask_block, dtype=bool)
    valid = np.asarray(valid_block, dtype=bool)
    clock = np.asarray(clock_block, dtype=np.float64)
    if mask.ndim < 1 or valid.ndim < 1 or clock.ndim < 1:
        raise ValueError(f"{owner}: mask, valid, and clock blocks must include trailing sequence dimensions.")
    seq_size = int(mask.shape[-1])
    if int(valid.shape[-1]) != seq_size or int(clock.shape[-1]) != seq_size:
        raise ValueError(f"{owner}: event boundary block trailing dimensions must match.")
    try:
        outer = np.broadcast_shapes(mask.shape[:-1], valid.shape[:-1], clock.shape[:-1])
    except ValueError as exc:
        raise ValueError(f"{owner}: event boundary blocks are not broadcast-compatible.") from exc
    rows = _row_count(outer)
    mask_rows = np.broadcast_to(mask, outer + (seq_size,)).reshape(rows, seq_size)
    valid_rows = np.broadcast_to(valid, outer + (seq_size,)).reshape(rows, seq_size)
    clock_rows = np.broadcast_to(clock, outer + (seq_size,)).reshape(rows, seq_size)
    return (
        np.ascontiguousarray(mask_rows),
        np.ascontiguousarray(valid_rows),
        np.ascontiguousarray(clock_rows),
        outer,
    )


def _broadcast_interval_blocks(
    time_block: np.ndarray,
    edge_block: np.ndarray,
    before_block: np.ndarray,
    after_block: np.ndarray,
    *,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    time = np.asarray(time_block, dtype=np.float64)
    edge = np.asarray(edge_block, dtype=np.int8)
    before = np.asarray(before_block, dtype=np.int64)
    after = np.asarray(after_block, dtype=np.int64)
    if time.ndim < 1 or edge.ndim < 1 or before.ndim < 1 or after.ndim < 1:
        raise ValueError(f"{owner}: interval blocks must include trailing event dimensions.")
    event_size = int(time.shape[-1])
    if int(edge.shape[-1]) != event_size or int(before.shape[-1]) != event_size or int(after.shape[-1]) != event_size:
        raise ValueError(f"{owner}: interval block trailing dimensions must match.")
    try:
        outer = np.broadcast_shapes(time.shape[:-1], edge.shape[:-1], before.shape[:-1], after.shape[:-1])
    except ValueError as exc:
        raise ValueError(f"{owner}: interval blocks are not broadcast-compatible.") from exc
    rows = _row_count(outer)
    time_rows = np.broadcast_to(time, outer + (event_size,)).reshape(rows, event_size)
    edge_rows = np.broadcast_to(edge, outer + (event_size,)).reshape(rows, event_size)
    before_rows = np.broadcast_to(before, outer + (event_size,)).reshape(rows, event_size)
    after_rows = np.broadcast_to(after, outer + (event_size,)).reshape(rows, event_size)
    return (
        np.ascontiguousarray(time_rows),
        np.ascontiguousarray(edge_rows),
        np.ascontiguousarray(before_rows),
        np.ascontiguousarray(after_rows),
        outer,
    )


def boundary_bounded_block_numba(
    mask_block: np.ndarray,
    valid_block: np.ndarray,
    clock_block: np.ndarray,
    *,
    include_initial: bool,
    emit_triggers: bool,
    dedupe_atol: float,
    max_events: int,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require_numba(owner)
    mask_rows, valid_rows, clock_rows, outer = _broadcast_boundary_blocks(
        mask_block,
        valid_block,
        clock_block,
        owner=owner,
    )
    time, edge, before, after, status = _compiled_boundary_block()(
        mask_rows,
        valid_rows,
        clock_rows,
        bool(include_initial),
        bool(emit_triggers),
        float(dedupe_atol),
        int(max_events),
    )
    if int(status) == _STATUS_NONFINITE_TIME:
        raise ValueError(f"{owner}: extracted event boundaries include non-finite clock values.")
    output_shape = outer + (int(max_events),)
    return time.reshape(output_shape), edge.reshape(output_shape), before.reshape(output_shape), after.reshape(output_shape)


def intervals_bounded_block_numba(
    time_block: np.ndarray,
    edge_block: np.ndarray,
    before_block: np.ndarray,
    after_block: np.ndarray,
    *,
    max_segments: int,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require_numba(owner)
    time_rows, edge_rows, before_rows, after_rows, outer = _broadcast_interval_blocks(
        time_block,
        edge_block,
        before_block,
        after_block,
        owner=owner,
    )
    time, sample, is_trigger, valid = _compiled_intervals_block()(
        time_rows,
        edge_rows,
        before_rows,
        after_rows,
        int(max_segments),
    )
    edge_shape = outer + (int(max_segments), 2)
    segment_shape = outer + (int(max_segments),)
    return (
        time.reshape(edge_shape),
        sample.reshape(edge_shape),
        is_trigger.reshape(segment_shape),
        valid.reshape(segment_shape),
    )


def _edge_rank(edge):
    if edge == EDGE_ENTER:
        return 0
    if edge == EDGE_TRIGGER:
        return 1
    if edge == EDGE_EXIT:
        return 2
    return 3


def _append_boundary(count, time, edge, before, after, next_time, next_edge, next_before, next_after):
    if not np.isfinite(next_time):
        return count, _STATUS_NONFINITE_TIME
    time[count] = next_time
    edge[count] = next_edge
    before[count] = next_before
    after[count] = next_after
    return count + 1, _STATUS_OK


def _sort_boundaries(count, time, edge, before, after):
    for idx in range(1, count):
        t = time[idx]
        e = edge[idx]
        b = before[idx]
        a = after[idx]
        rank = _edge_rank(e)
        pos = idx - 1
        while pos >= 0 and (time[pos] > t or (time[pos] == t and _edge_rank(edge[pos]) > rank)):
            time[pos + 1] = time[pos]
            edge[pos + 1] = edge[pos]
            before[pos + 1] = before[pos]
            after[pos + 1] = after[pos]
            pos -= 1
        time[pos + 1] = t
        edge[pos + 1] = e
        before[pos + 1] = b
        after[pos + 1] = a


def _dedupe_boundaries(row, count, dedupe_atol, candidates, outputs):
    time, edge, before, after = candidates
    out_time, out_edge, out_before, out_after = outputs
    seen = np.zeros(4, dtype=np.bool_)
    seen_time = np.zeros(4, dtype=np.float64)
    written = 0
    max_events = out_time.shape[1]
    for idx in range(count):
        code = int(edge[idx])
        now = time[idx]
        if seen[code] and abs(now - seen_time[code]) <= dedupe_atol:
            continue
        seen[code] = True
        seen_time[code] = now
        if written < max_events:
            out_time[row, written] = now
            out_edge[row, written] = edge[idx]
            out_before[row, written] = before[idx]
            out_after[row, written] = after[idx]
            written += 1


def _transition_candidates(mask, valid, clock, include_initial, candidates):
    time, edge, before, after = candidates
    count = 0
    has_previous = False
    prev_idx = 0
    prev_mask = False
    prev_clock = 0.0
    for idx in range(mask.shape[0]):
        if not valid[idx]:
            continue
        current_mask = bool(mask[idx])
        current_clock = clock[idx]
        if not has_previous:
            if include_initial and current_mask:
                count, status = _append_boundary(
                    count, time, edge, before, after, current_clock, EDGE_ENTER, SAMPLE_SENTINEL, idx
                )
                if status != _STATUS_OK:
                    return count, status
        elif (not prev_mask) and current_mask:
            count, status = _append_boundary(count, time, edge, before, after, current_clock, EDGE_ENTER, prev_idx, idx)
            if status != _STATUS_OK:
                return count, status
        elif prev_mask and (not current_mask):
            count, status = _append_boundary(count, time, edge, before, after, prev_clock, EDGE_EXIT, prev_idx, idx)
            if status != _STATUS_OK:
                return count, status
        has_previous = True
        prev_idx = idx
        prev_mask = current_mask
        prev_clock = current_clock
    if has_previous and prev_mask:
        count, status = _append_boundary(count, time, edge, before, after, prev_clock, EDGE_EXIT, prev_idx, SAMPLE_SENTINEL)
        if status != _STATUS_OK:
            return count, status
    return count, _STATUS_OK


def _trigger_candidates(mask, valid, clock, count, candidates):
    time, edge, before, after = candidates
    for idx in range(mask.shape[0]):
        if valid[idx] and bool(mask[idx]):
            count, status = _append_boundary(count, time, edge, before, after, clock[idx], EDGE_TRIGGER, idx, idx)
            if status != _STATUS_OK:
                return count, status
    return count, _STATUS_OK


def _boundary_row_impl(row, mask, valid, clock, include_initial, emit_triggers, dedupe_atol, outputs):
    max_candidates = mask.shape[0] * 2 + 2
    time = np.empty(max_candidates, dtype=np.float64)
    edge = np.empty(max_candidates, dtype=np.int8)
    before = np.empty(max_candidates, dtype=np.int64)
    after = np.empty(max_candidates, dtype=np.int64)
    candidates = (time, edge, before, after)
    count, status = _transition_candidates(mask, valid, clock, include_initial, candidates)
    if status != _STATUS_OK:
        return status
    if emit_triggers:
        count, status = _trigger_candidates(mask, valid, clock, count, candidates)
        if status != _STATUS_OK:
            return status
    _sort_boundaries(count, time, edge, before, after)
    _dedupe_boundaries(row, count, dedupe_atol, candidates, outputs)
    return _STATUS_OK


def _boundary_block_impl(mask, valid, clock, include_initial, emit_triggers, dedupe_atol, max_events):
    rows = mask.shape[0]
    out_time = np.full((rows, max_events), np.nan, dtype=np.float64)
    out_edge = np.zeros((rows, max_events), dtype=np.int8)
    out_before = np.full((rows, max_events), SAMPLE_SENTINEL, dtype=np.int64)
    out_after = np.full((rows, max_events), SAMPLE_SENTINEL, dtype=np.int64)
    outputs = (out_time, out_edge, out_before, out_after)
    for row in range(rows):
        status = _boundary_row_impl(
            row,
            mask[row],
            valid[row],
            clock[row],
            include_initial,
            emit_triggers,
            dedupe_atol,
            outputs,
        )
        if status != _STATUS_OK:
            return out_time, out_edge, out_before, out_after, status
    return out_time, out_edge, out_before, out_after, _STATUS_OK


def _interval_has_trigger_collision(index, count, segments):
    time0, time1, sample0, sample1, trigger = segments
    for other in range(count):
        if not trigger[other]:
            continue
        if time0[index] == time0[other] and time1[index] == time1[other]:
            if sample0[index] == sample0[other] and sample1[index] == sample1[other]:
                return True
    return False


def _collect_interval_segments(time, edge, before, after, segments):
    seg_t0, seg_t1, seg_s0, seg_s1, seg_trigger = segments
    count = 0
    has_open = False
    open_time = 0.0
    open_sample = SAMPLE_SENTINEL
    for idx in range(time.shape[0]):
        code = edge[idx]
        if code == EDGE_INVALID:
            continue
        if code == EDGE_ENTER:
            has_open = True
            open_time = time[idx]
            open_sample = after[idx]
            continue
        if code == EDGE_TRIGGER:
            sample = after[idx]
            seg_t0[count] = time[idx]
            seg_t1[count] = time[idx]
            seg_s0[count] = sample
            seg_s1[count] = sample
            seg_trigger[count] = True
            count += 1
            continue
        if code == EDGE_EXIT and has_open:
            seg_t0[count] = open_time
            seg_t1[count] = time[idx]
            seg_s0[count] = open_sample
            seg_s1[count] = before[idx]
            seg_trigger[count] = False
            count += 1
            has_open = False
    return count


def _write_interval_segments(row, count, segments, outputs):
    seg_t0, seg_t1, seg_s0, seg_s1, seg_trigger = segments
    out_time, out_sample, out_trigger, out_valid = outputs
    written = 0
    max_segments = out_valid.shape[1]
    for idx in range(count):
        degenerate = seg_t0[idx] == seg_t1[idx] and seg_s0[idx] == seg_s1[idx]
        if (not seg_trigger[idx]) and degenerate:
            if _interval_has_trigger_collision(idx, count, segments):
                continue
        if written < max_segments:
            out_time[row, written, 0] = seg_t0[idx]
            out_time[row, written, 1] = seg_t1[idx]
            out_sample[row, written, 0] = seg_s0[idx]
            out_sample[row, written, 1] = seg_s1[idx]
            out_trigger[row, written] = seg_trigger[idx]
            out_valid[row, written] = True
            written += 1


def _intervals_row_impl(row, time, edge, before, after, outputs):
    event_size = time.shape[0]
    seg_t0 = np.empty(event_size, dtype=np.float64)
    seg_t1 = np.empty(event_size, dtype=np.float64)
    seg_s0 = np.empty(event_size, dtype=np.int64)
    seg_s1 = np.empty(event_size, dtype=np.int64)
    seg_trigger = np.zeros(event_size, dtype=np.bool_)
    segments = (seg_t0, seg_t1, seg_s0, seg_s1, seg_trigger)
    count = _collect_interval_segments(time, edge, before, after, segments)
    _write_interval_segments(row, count, segments, outputs)


def _intervals_block_impl(time, edge, before, after, max_segments):
    rows = time.shape[0]
    out_time = np.full((rows, max_segments, 2), np.nan, dtype=np.float64)
    out_sample = np.full((rows, max_segments, 2), SAMPLE_SENTINEL, dtype=np.int64)
    out_trigger = np.zeros((rows, max_segments), dtype=np.bool_)
    out_valid = np.zeros((rows, max_segments), dtype=np.bool_)
    outputs = (out_time, out_sample, out_trigger, out_valid)
    for row in range(rows):
        _intervals_row_impl(row, time[row], edge[row], before[row], after[row], outputs)
    return out_time, out_sample, out_trigger, out_valid


__all__ = ["boundary_bounded_block_numba", "intervals_bounded_block_numba"]
