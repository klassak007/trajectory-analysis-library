from __future__ import annotations

import numpy as np

from ._event_constants import EDGE_INVALID, SAMPLE_SENTINEL
from .block_prep import prepare_boundary_block_rows, prepare_intervals_block_rows


def boundary_bounded_block_numpy(
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
    from .boundary import _bounded_row_kernel

    mask_rows, valid_rows, clock_rows, outer = prepare_boundary_block_rows(
        mask_block,
        valid_block,
        clock_block,
        owner=owner,
    )
    rows = int(mask_rows.shape[0])
    time = np.full((rows, int(max_events)), np.nan, dtype=np.float64)
    edge = np.full((rows, int(max_events)), EDGE_INVALID, dtype=np.int8)
    before = np.full((rows, int(max_events)), SAMPLE_SENTINEL, dtype=np.int64)
    after = np.full((rows, int(max_events)), SAMPLE_SENTINEL, dtype=np.int64)
    for row in range(rows):
        row_time, row_edge, row_before, row_after = _bounded_row_kernel(
            mask_rows[row],
            valid_rows[row],
            clock_rows[row],
            include_initial=include_initial,
            emit_triggers=emit_triggers,
            dedupe_atol=dedupe_atol,
            max_events=int(max_events),
            owner=owner,
        )
        time[row] = row_time
        edge[row] = row_edge
        before[row] = row_before
        after[row] = row_after
    output_shape = outer + (int(max_events),)
    return time.reshape(output_shape), edge.reshape(output_shape), before.reshape(output_shape), after.reshape(output_shape)


def intervals_bounded_block_numpy(
    time_block: np.ndarray,
    edge_block: np.ndarray,
    before_block: np.ndarray,
    after_block: np.ndarray,
    *,
    max_segments: int,
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .intervals import _bounded_row_kernel

    time_rows, edge_rows, before_rows, after_rows, outer = prepare_intervals_block_rows(
        time_block,
        edge_block,
        before_block,
        after_block,
        owner=owner,
    )
    rows = int(time_rows.shape[0])
    edge_shape = (rows, int(max_segments), 2)
    segment_shape = (rows, int(max_segments))
    time = np.full(edge_shape, np.nan, dtype=np.float64)
    sample = np.full(edge_shape, SAMPLE_SENTINEL, dtype=np.int64)
    is_trigger = np.zeros(segment_shape, dtype=bool)
    valid = np.zeros(segment_shape, dtype=bool)
    for row in range(rows):
        row_time, row_sample, row_trigger, row_valid = _bounded_row_kernel(
            time_rows[row],
            edge_rows[row],
            before_rows[row],
            after_rows[row],
            max_segments=int(max_segments),
            owner=owner,
        )
        time[row] = row_time
        sample[row] = row_sample
        is_trigger[row] = row_trigger
        valid[row] = row_valid
    out_edge_shape = outer + (int(max_segments), 2)
    out_segment_shape = outer + (int(max_segments),)
    return (
        time.reshape(out_edge_shape),
        sample.reshape(out_edge_shape),
        is_trigger.reshape(out_segment_shape),
        valid.reshape(out_segment_shape),
    )


__all__ = ["boundary_bounded_block_numpy", "intervals_bounded_block_numpy"]
