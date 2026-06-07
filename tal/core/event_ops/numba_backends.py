from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._event_constants import EDGE_ENTER, EDGE_EXIT, EDGE_INVALID, EDGE_TRIGGER, SAMPLE_SENTINEL
from .block_prep import prepare_boundary_block_rows, prepare_intervals_block_rows
from tal.utils.numba_support import njit_kernel, require_numba

_STATUS_OK = 0
_STATUS_NONFINITE_TIME = 1
_HELPERS_JITTED = False


@lru_cache(maxsize=1)
def _compiled_boundary_block():
    numba = require_numba("events.boundaries")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _boundary_block_impl)


@lru_cache(maxsize=1)
def _compiled_intervals_block():
    numba = require_numba("events.intervals")
    _jit_kernel_helpers(numba)
    return njit_kernel(numba, _intervals_block_impl)


def _jit_kernel_helpers(numba) -> None:
    global _HELPERS_JITTED
    global _append_boundary, _append_transition, _boundary_row_impl, _collect_interval_segments, _dedupe_boundaries
    global _edge_rank, _interval_has_trigger_collision
    global _intervals_row_impl, _skip_interval_segment, _sort_boundaries, _transition_candidates
    global _trigger_candidates, _write_interval_segments
    if _HELPERS_JITTED:
        return
    _edge_rank = njit_kernel(numba, _edge_rank)
    _append_boundary = njit_kernel(numba, _append_boundary)
    _append_transition = njit_kernel(numba, _append_transition)
    _sort_boundaries = njit_kernel(numba, _sort_boundaries)
    _dedupe_boundaries = njit_kernel(numba, _dedupe_boundaries)
    _transition_candidates = njit_kernel(numba, _transition_candidates)
    _trigger_candidates = njit_kernel(numba, _trigger_candidates)
    _boundary_row_impl = njit_kernel(numba, _boundary_row_impl)
    _interval_has_trigger_collision = njit_kernel(numba, _interval_has_trigger_collision)
    _collect_interval_segments = njit_kernel(numba, _collect_interval_segments)
    _skip_interval_segment = njit_kernel(numba, _skip_interval_segment)
    _write_interval_segments = njit_kernel(numba, _write_interval_segments)
    _intervals_row_impl = njit_kernel(numba, _intervals_row_impl)
    _HELPERS_JITTED = True


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
    mask_rows, valid_rows, clock_rows, outer = prepare_boundary_block_rows(
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
    time_rows, edge_rows, before_rows, after_rows, outer = prepare_intervals_block_rows(
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


def _append_transition(count, current_clock, current_mask, previous, idx, include_initial, candidates):
    has_previous, prev_mask, prev_idx, prev_clock = previous
    time, edge, before, after = candidates
    if not has_previous:
        if not include_initial or not current_mask:
            return count, _STATUS_OK
        return _append_boundary(count, time, edge, before, after, current_clock, EDGE_ENTER, SAMPLE_SENTINEL, idx)
    if (not prev_mask) and current_mask:
        return _append_boundary(count, time, edge, before, after, current_clock, EDGE_ENTER, prev_idx, idx)
    if prev_mask and (not current_mask):
        return _append_boundary(count, time, edge, before, after, prev_clock, EDGE_EXIT, prev_idx, idx)
    return count, _STATUS_OK


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
        previous = (has_previous, prev_mask, prev_idx, prev_clock)
        count, status = _append_transition(count, current_clock, current_mask, previous, idx, include_initial, candidates)
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
        if not valid[idx] or not bool(mask[idx]):
            continue
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
        if time0[index] != time0[other] or time1[index] != time1[other]:
            continue
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


def _skip_interval_segment(index, count, segments):
    seg_t0, seg_t1, seg_s0, seg_s1, seg_trigger = segments
    if seg_trigger[index]:
        return False
    if seg_t0[index] != seg_t1[index] or seg_s0[index] != seg_s1[index]:
        return False
    return _interval_has_trigger_collision(index, count, segments)


def _write_interval_segments(row, count, segments, outputs):
    seg_t0, seg_t1, seg_s0, seg_s1, seg_trigger = segments
    out_time, out_sample, out_trigger, out_valid = outputs
    written = 0
    max_segments = out_valid.shape[1]
    for idx in range(count):
        if _skip_interval_segment(idx, count, segments):
            continue
        if written >= max_segments:
            continue
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
