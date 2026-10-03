from __future__ import annotations

import argparse

import numpy as np

from _numba_bench import cold_subprocess, time_once, warm_median
from tal.core.event_ops.backends import (
    EVENT_BOUNDARY_BACKEND_NUMBA,
    EVENT_BOUNDARY_BACKEND_NUMPY_BLOCK,
    EVENT_INTERVALS_BACKEND_NUMBA,
    EVENT_INTERVALS_BACKEND_NUMPY_BLOCK,
    boundary_bounded_block_backend,
    intervals_bounded_block_backend,
)
from tal.core.event_ops.boundary import _bounded_row_kernel as _boundary_row
from tal.core.event_ops.event_primitives import EDGE_ENTER, EDGE_EXIT, EDGE_INVALID, EDGE_TRIGGER, SAMPLE_SENTINEL
from tal.core.event_ops.intervals import _bounded_row_kernel as _intervals_row


_PROMOTION_SPEEDUP = 0.8
_PROMOTION_SLOWDOWN = 1.1


def _dense_boundary_many_short() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows = 4096
    seq = 32
    base = np.arange(seq)
    mask = np.broadcast_to((base % 3) != 0, (rows, seq)).copy()
    valid = np.ones_like(mask, dtype=bool)
    clock = np.broadcast_to(base.astype("float64"), (rows, seq)).copy()
    return mask, valid, clock, 32


def _sparse_boundary_many_short() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows = 4096
    seq = 64
    mask = np.zeros((rows, seq), dtype=bool)
    mask[:, 16:20] = True
    valid = np.ones_like(mask, dtype=bool)
    clock = np.broadcast_to(np.arange(seq, dtype="float64"), (rows, seq)).copy()
    return mask, valid, clock, 8


def _dense_boundary_fewer_long() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows = 128
    seq = 1024
    base = np.arange(seq)
    mask = np.broadcast_to((base % 4) < 2, (rows, seq)).copy()
    valid = np.ones_like(mask, dtype=bool)
    clock = np.broadcast_to(base.astype("float64"), (rows, seq)).copy()
    return mask, valid, clock, 1024


def _interval_rows_many_short() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    rows = 4096
    event_count = 12
    time = np.full((rows, event_count), np.nan, dtype="float64")
    edge = np.full((rows, event_count), EDGE_INVALID, dtype="int8")
    before = np.full((rows, event_count), SAMPLE_SENTINEL, dtype="int64")
    after = np.full((rows, event_count), SAMPLE_SENTINEL, dtype="int64")
    pattern_time = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 8.0], dtype="float64")
    pattern_edge = np.asarray(
        [EDGE_ENTER, EDGE_EXIT, EDGE_ENTER, EDGE_EXIT, EDGE_ENTER, EDGE_TRIGGER, EDGE_EXIT, EDGE_INVALID],
        dtype="int8",
    )
    pattern_before = np.asarray([0, 1, 2, 3, 4, 5, 5, SAMPLE_SENTINEL], dtype="int64")
    pattern_after = np.asarray([1, 2, 3, 4, 5, 5, 6, SAMPLE_SENTINEL], dtype="int64")
    time[:, : pattern_time.size] = pattern_time
    edge[:, : pattern_edge.size] = pattern_edge
    before[:, : pattern_before.size] = pattern_before
    after[:, : pattern_after.size] = pattern_after
    return time, edge, before, after, 8


def _interval_rows_fewer_long() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    rows = 128
    event_count = 512
    base = np.arange(event_count, dtype="int64")
    edge_row = np.full(event_count, EDGE_INVALID, dtype="int8")
    edge_row[base % 4 == 0] = EDGE_ENTER
    edge_row[base % 4 == 1] = EDGE_EXIT
    edge_row[base % 4 == 2] = EDGE_TRIGGER
    sample = base.copy()
    time = np.broadcast_to(base.astype("float64"), (rows, event_count)).copy()
    edge = np.broadcast_to(edge_row, (rows, event_count)).copy()
    before = np.full((rows, event_count), SAMPLE_SENTINEL, dtype="int64")
    after = np.full((rows, event_count), SAMPLE_SENTINEL, dtype="int64")
    before[:, edge_row == EDGE_EXIT] = sample[edge_row == EDGE_EXIT] - 1
    before[:, edge_row == EDGE_TRIGGER] = sample[edge_row == EDGE_TRIGGER]
    after[:, edge_row == EDGE_ENTER] = sample[edge_row == EDGE_ENTER]
    after[:, edge_row == EDGE_EXIT] = sample[edge_row == EDGE_EXIT]
    after[:, edge_row == EDGE_TRIGGER] = sample[edge_row == EDGE_TRIGGER]
    return time, edge, before, after, 256


def _baseline_boundary(mask: np.ndarray, valid: np.ndarray, clock: np.ndarray, max_events: int) -> None:
    for row in range(mask.shape[0]):
        _boundary_row(
            mask[row],
            valid[row],
            clock[row],
            include_initial=True,
            emit_triggers=True,
            dedupe_atol=0.0,
            max_events=max_events,
            owner="events.boundaries",
        )


def _numba_boundary(mask: np.ndarray, valid: np.ndarray, clock: np.ndarray, max_events: int) -> None:
    boundary_bounded_block_backend(
        mask,
        valid,
        clock,
        include_initial=True,
        emit_triggers=True,
        dedupe_atol=0.0,
        max_events=max_events,
        owner="events.boundaries",
        backend=EVENT_BOUNDARY_BACKEND_NUMBA,
    )


def _numpy_block_boundary(mask: np.ndarray, valid: np.ndarray, clock: np.ndarray, max_events: int) -> None:
    boundary_bounded_block_backend(
        mask,
        valid,
        clock,
        include_initial=True,
        emit_triggers=True,
        dedupe_atol=0.0,
        max_events=max_events,
        owner="events.boundaries",
        backend=EVENT_BOUNDARY_BACKEND_NUMPY_BLOCK,
    )


def _baseline_intervals(time: np.ndarray, edge: np.ndarray, before: np.ndarray, after: np.ndarray, max_segments: int) -> None:
    for row in range(time.shape[0]):
        _intervals_row(
            time[row],
            edge[row],
            before[row],
            after[row],
            max_segments=max_segments,
            owner="events.intervals",
        )


def _numba_intervals(time: np.ndarray, edge: np.ndarray, before: np.ndarray, after: np.ndarray, max_segments: int) -> None:
    intervals_bounded_block_backend(
        time,
        edge,
        before,
        after,
        max_segments=max_segments,
        owner="events.intervals",
        backend=EVENT_INTERVALS_BACKEND_NUMBA,
    )


def _numpy_block_intervals(
    time: np.ndarray,
    edge: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    max_segments: int,
) -> None:
    intervals_bounded_block_backend(
        time,
        edge,
        before,
        after,
        max_segments=max_segments,
        owner="events.intervals",
        backend=EVENT_INTERVALS_BACKEND_NUMPY_BLOCK,
    )


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-event-numba-cache-")


def _run_cold(case_name: str) -> None:
    if case_name == "boundary-dense-many-short":
        print(f"{time_once(_numba_boundary, *_dense_boundary_many_short()):.9f}")
        return
    if case_name == "boundary-sparse-many-short":
        print(f"{time_once(_numba_boundary, *_sparse_boundary_many_short()):.9f}")
        return
    if case_name == "boundary-dense-fewer-long":
        print(f"{time_once(_numba_boundary, *_dense_boundary_fewer_long()):.9f}")
        return
    if case_name == "intervals-many-short":
        print(f"{time_once(_numba_intervals, *_interval_rows_many_short()):.9f}")
        return
    if case_name == "intervals-fewer-long":
        print(f"{time_once(_numba_intervals, *_interval_rows_fewer_long()):.9f}")
        return
    raise ValueError(f"unknown benchmark case {case_name!r}")


def _report_boundary(
    case_name: str,
    title: str,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, int],
) -> tuple[bool, bool]:
    mask, valid, clock, max_events = data
    _numba_boundary(mask, valid, clock, max_events)
    baseline = warm_median(_baseline_boundary, mask, valid, clock, max_events)
    numpy_block = warm_median(_numpy_block_boundary, mask, valid, clock, max_events)
    warm = warm_median(_numba_boundary, mask, valid, clock, max_events)
    eligible = "many-short" in case_name
    print(f"\n{title}")
    print(f"  shape: rows={mask.shape[0]} seq={mask.shape[1]} max_events={max_events}")
    print(f"  boundary private row baseline warm:      {baseline:.6f}s")
    print(f"  boundary numpy_block fallback warm:      {numpy_block:.6f}s")
    print(f"  boundary numba first-call fresh process: {_cold_subprocess(case_name):.6f}s")
    print(f"  boundary numba warm median:              {warm:.6f}s")
    return (
        eligible and warm <= baseline * _PROMOTION_SPEEDUP,
        (not eligible) or warm <= baseline * _PROMOTION_SLOWDOWN,
    )


def _report_intervals(
    case_name: str,
    title: str,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
) -> tuple[bool, bool]:
    time, edge, before, after, max_segments = data
    _numba_intervals(time, edge, before, after, max_segments)
    baseline = warm_median(_baseline_intervals, time, edge, before, after, max_segments)
    numpy_block = warm_median(_numpy_block_intervals, time, edge, before, after, max_segments)
    warm = warm_median(_numba_intervals, time, edge, before, after, max_segments)
    eligible = "many-short" in case_name
    print(f"\n{title}")
    print(f"  shape: rows={time.shape[0]} events={time.shape[1]} max_segments={max_segments}")
    print(f"  intervals private row baseline warm:     {baseline:.6f}s")
    print(f"  intervals numpy_block fallback warm:     {numpy_block:.6f}s")
    print(f"  intervals numba first-call fresh process: {_cold_subprocess(case_name):.6f}s")
    print(f"  intervals numba warm median:             {warm:.6f}s")
    return (
        eligible and warm <= baseline * _PROMOTION_SPEEDUP,
        (not eligible) or warm <= baseline * _PROMOTION_SLOWDOWN,
    )


def _print_gate(boundary_results: list[tuple[bool, bool]], interval_results: list[tuple[bool, bool]]) -> None:
    boundary_win = any(result[0] for result in boundary_results)
    boundary_slowdown_ok = all(result[1] for result in boundary_results)
    intervals_win = any(result[0] for result in interval_results)
    intervals_slowdown_ok = all(result[1] for result in interval_results)
    print("\nevent F2C decision input")
    print(f"  boundary eligible high-row >=20% warm win:     {boundary_win}")
    print(f"  boundary eligible high-row <=10% slowdowns:    {boundary_slowdown_ok}")
    print(f"  boundary promotion gate:                       {'PASS' if boundary_win and boundary_slowdown_ok else 'FAIL'}")
    print(f"  intervals eligible high-row >=20% warm win:    {intervals_win}")
    print(f"  intervals eligible high-row <=10% slowdowns:   {intervals_slowdown_ok}")
    print(f"  intervals promotion gate:                      {'PASS' if intervals_win and intervals_slowdown_ok else 'FAIL'}")
    print("\nevent F2C-B2 normal-path migration")
    print("  boundary selected normal backend:              numba")
    print("  boundary no-numba fallback backend:            numpy_block")
    print("  intervals selected normal backend:             numba")
    print("  intervals no-numba fallback backend:           numpy_block")
    print("  public normal path:                            blockwise vectorize=False")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    boundary_results = [
        _report_boundary("boundary-dense-many-short", "many-short dense bounded boundary rows", _dense_boundary_many_short()),
        _report_boundary("boundary-sparse-many-short", "many-short sparse bounded boundary rows", _sparse_boundary_many_short()),
        _report_boundary("boundary-dense-fewer-long", "fewer-long dense bounded boundary rows", _dense_boundary_fewer_long()),
    ]
    interval_results = [
        _report_intervals("intervals-many-short", "many-short bounded interval rows", _interval_rows_many_short()),
        _report_intervals("intervals-fewer-long", "fewer-long bounded interval rows", _interval_rows_fewer_long()),
    ]
    _print_gate(boundary_results, interval_results)


if __name__ == "__main__":
    main()
