from pathlib import Path

import numpy as np
import pytest

from tal.core.event_ops.backends import (
    EVENT_BOUNDARY_BACKEND_NUMBA,
    EVENT_INTERVALS_BACKEND_NUMBA,
    boundary_bounded_block_backend,
    intervals_bounded_block_backend,
)
from tal.core.event_ops.boundary import _bounded_row_kernel as _boundary_row
from tal.core.event_ops.event_primitives import EDGE_ENTER, EDGE_EXIT, EDGE_INVALID, EDGE_TRIGGER, SAMPLE_SENTINEL
from tal.core.event_ops.intervals import _bounded_row_kernel as _intervals_row


def _require_numba() -> None:
    pytest.importorskip("numba")


def _baseline_boundary_block(
    mask: np.ndarray,
    valid: np.ndarray,
    clock: np.ndarray,
    *,
    include_initial: bool,
    emit_triggers: bool,
    dedupe_atol: float,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outer = np.broadcast_shapes(mask.shape[:-1], valid.shape[:-1], clock.shape[:-1])
    rows = int(np.prod(outer, dtype=np.int64)) if outer else 1
    mask_rows = np.broadcast_to(mask, outer + (mask.shape[-1],)).reshape(rows, mask.shape[-1])
    valid_rows = np.broadcast_to(valid, outer + (valid.shape[-1],)).reshape(rows, valid.shape[-1])
    clock_rows = np.broadcast_to(clock, outer + (clock.shape[-1],)).reshape(rows, clock.shape[-1])
    outputs = [
        _boundary_row(
            mask_rows[row],
            valid_rows[row],
            clock_rows[row],
            include_initial=include_initial,
            emit_triggers=emit_triggers,
            dedupe_atol=dedupe_atol,
            max_events=max_events,
            owner="events.boundaries",
        )
        for row in range(rows)
    ]
    return tuple(np.stack([out[idx] for out in outputs]).reshape(outer + (max_events,)) for idx in range(4))


def _baseline_intervals_block(
    time: np.ndarray,
    edge: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outer = np.broadcast_shapes(time.shape[:-1], edge.shape[:-1], before.shape[:-1], after.shape[:-1])
    rows = int(np.prod(outer, dtype=np.int64)) if outer else 1
    time_rows = np.broadcast_to(time, outer + (time.shape[-1],)).reshape(rows, time.shape[-1])
    edge_rows = np.broadcast_to(edge, outer + (edge.shape[-1],)).reshape(rows, edge.shape[-1])
    before_rows = np.broadcast_to(before, outer + (before.shape[-1],)).reshape(rows, before.shape[-1])
    after_rows = np.broadcast_to(after, outer + (after.shape[-1],)).reshape(rows, after.shape[-1])
    outputs = [
        _intervals_row(
            time_rows[row],
            edge_rows[row],
            before_rows[row],
            after_rows[row],
            max_segments=max_segments,
            owner="events.intervals",
        )
        for row in range(rows)
    ]
    return tuple(
        np.stack([out[idx] for out in outputs]).reshape(outer + outputs[0][idx].shape)
        for idx in range(4)
    )


def _assert_boundary_equal(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    assert actual[0].dtype == np.dtype("float64")
    assert actual[1].dtype == np.dtype("int8")
    assert actual[2].dtype == np.dtype("int64")
    assert actual[3].dtype == np.dtype("int64")
    np.testing.assert_allclose(actual[0], expected[0], equal_nan=True)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[3], expected[3])


def _assert_intervals_equal(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    assert actual[0].dtype == np.dtype("float64")
    assert actual[1].dtype == np.dtype("int64")
    assert actual[2].dtype == np.dtype("bool")
    assert actual[3].dtype == np.dtype("bool")
    np.testing.assert_allclose(actual[0], expected[0], equal_nan=True)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[3], expected[3])


def _decision_section(target: str) -> str:
    text = Path("contracts/114-numba-default-baseline-migration-slice-f2c.md").read_text(encoding="utf-8")
    section = text.split(f"### {target}", 1)[1]
    return section.split("\n### ", 1)[0]


def _assert_decision_record(target: str, benchmark: str, cases: tuple[str, ...]) -> None:
    section = _decision_section(target)
    for required in (
        "Decision: promote",
        "Gate result: PASS",
        "Benchmark evidence:",
        "Reason:",
        "Public routing status:",
        "No-Numba behavior:",
        "Next F2C-B action:",
    ):
        assert required in section
    assert benchmark in section
    for case in cases:
        assert case in section
    contract_083 = Path("contracts/083-compiled-kernel-backend-followon-phase-f2.md").read_text(encoding="utf-8")
    assert "Status: Draft" in contract_083
    assert "event/linalg closeout open" in contract_083


def test_event_f2c_001_boundary_default_or_baseline_migration_decision() -> None:
    """ID: EVENT_F2C_001_boundary_default_or_baseline_migration_decision."""
    _assert_decision_record(
        "event_boundary",
        "benchmarks/bench_event_numba_backends.py",
        ("boundary-dense-many-short", "boundary-sparse-many-short", "boundary-dense-fewer-long"),
    )
    text = Path("tal/core/event_ops/boundary.py").read_text(encoding="utf-8")
    assert '"backend": EVENT_BOUNDARY_BACKEND_NUMPY_ROW' in text
    assert "EVENT_BOUNDARY_BACKEND_NUMBA" not in text
    assert "vectorize=True" in text.split("def _extract_bounded(", 1)[1]


def test_event_f2c_002_intervals_default_or_baseline_migration_decision() -> None:
    """ID: EVENT_F2C_002_intervals_default_or_baseline_migration_decision."""
    _assert_decision_record(
        "event_intervals",
        "benchmarks/bench_event_numba_backends.py",
        ("intervals-many-short", "intervals-fewer-long"),
    )
    text = Path("tal/core/event_ops/intervals.py").read_text(encoding="utf-8")
    assert '"backend": EVENT_INTERVALS_BACKEND_NUMPY_ROW' in text
    assert "EVENT_INTERVALS_BACKEND_NUMBA" not in text
    assert "vectorize=True" in text.split("def _extract_bounded(", 1)[1]


def test_event_numba_001_boundary_bounded_transition_parity() -> None:
    """ID: EVENT_NUMBA_001_boundary_bounded_transition_parity."""
    _require_numba()
    mask = np.asarray([[False, True, True, False, True], [True, True, False, False, True]])
    valid = np.ones_like(mask, dtype=bool)
    clock = np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0, 14.0]])
    actual = boundary_bounded_block_backend(
        mask,
        valid,
        clock,
        include_initial=True,
        emit_triggers=False,
        dedupe_atol=0.0,
        max_events=5,
        owner="events.boundaries",
        backend=EVENT_BOUNDARY_BACKEND_NUMBA,
    )
    expected = _baseline_boundary_block(
        mask,
        valid,
        clock,
        include_initial=True,
        emit_triggers=False,
        dedupe_atol=0.0,
        max_events=5,
    )
    _assert_boundary_equal(actual, expected)
    with pytest.raises(ValueError, match="events.boundaries: extracted event boundaries include non-finite clock values"):
        boundary_bounded_block_backend(
            np.asarray([[True]]),
            np.asarray([[True]]),
            np.asarray([[np.nan]]),
            include_initial=True,
            emit_triggers=False,
            dedupe_atol=0.0,
            max_events=1,
            owner="events.boundaries",
            backend=EVENT_BOUNDARY_BACKEND_NUMBA,
        )


def test_event_numba_002_boundary_bounded_trigger_parity() -> None:
    """ID: EVENT_NUMBA_002_boundary_bounded_trigger_parity."""
    _require_numba()
    mask = np.asarray([[True, False, True, False], [False, True, True, False]])
    valid = np.asarray([[True, True, True, True], [True, True, False, True]])
    clock = np.asarray([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])
    actual = boundary_bounded_block_backend(
        mask,
        valid,
        clock,
        include_initial=False,
        emit_triggers=True,
        dedupe_atol=0.0,
        max_events=8,
        owner="events.boundaries",
        backend=EVENT_BOUNDARY_BACKEND_NUMBA,
    )
    expected = _baseline_boundary_block(
        mask,
        valid,
        clock,
        include_initial=False,
        emit_triggers=True,
        dedupe_atol=0.0,
        max_events=8,
    )
    _assert_boundary_equal(actual, expected)


def test_event_numba_003_boundary_bounded_dedupe_parity() -> None:
    """ID: EVENT_NUMBA_003_boundary_bounded_dedupe_parity."""
    _require_numba()
    mask = np.asarray([[True, True, False, True]])
    valid = np.asarray([[True, True, True, True]])
    clock = np.asarray([[0.0, 1e-13, 1.0, 1.0 + 5e-13]])
    actual = boundary_bounded_block_backend(
        mask,
        valid,
        clock,
        include_initial=True,
        emit_triggers=True,
        dedupe_atol=1e-12,
        max_events=8,
        owner="events.boundaries",
        backend=EVENT_BOUNDARY_BACKEND_NUMBA,
    )
    expected = _baseline_boundary_block(
        mask,
        valid,
        clock,
        include_initial=True,
        emit_triggers=True,
        dedupe_atol=1e-12,
        max_events=8,
    )
    _assert_boundary_equal(actual, expected)


def test_event_numba_004_boundary_bounded_truncation_padding_parity() -> None:
    """ID: EVENT_NUMBA_004_boundary_bounded_truncation_padding_parity."""
    _require_numba()
    mask = np.asarray([[False, True, False, True, False]])
    valid = np.ones_like(mask, dtype=bool)
    clock = np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]])
    for max_events in (2, 6):
        actual = boundary_bounded_block_backend(
            mask,
            valid,
            clock,
            include_initial=False,
            emit_triggers=True,
            dedupe_atol=0.0,
            max_events=max_events,
            owner="events.boundaries",
            backend=EVENT_BOUNDARY_BACKEND_NUMBA,
        )
        expected = _baseline_boundary_block(
            mask,
            valid,
            clock,
            include_initial=False,
            emit_triggers=True,
            dedupe_atol=0.0,
            max_events=max_events,
        )
        _assert_boundary_equal(actual, expected)


def test_event_numba_005_intervals_bounded_pairing_parity() -> None:
    """ID: EVENT_NUMBA_005_intervals_bounded_pairing_parity."""
    _require_numba()
    time = np.asarray([[1.0, 2.0, 4.0, np.nan]])
    edge = np.asarray([[EDGE_ENTER, EDGE_EXIT, EDGE_ENTER, EDGE_INVALID]], dtype="int8")
    before = np.asarray([[0, 1, 3, SAMPLE_SENTINEL]], dtype="int64")
    after = np.asarray([[1, 2, 4, SAMPLE_SENTINEL]], dtype="int64")
    actual = intervals_bounded_block_backend(
        time,
        edge,
        before,
        after,
        max_segments=3,
        owner="events.intervals",
        backend=EVENT_INTERVALS_BACKEND_NUMBA,
    )
    expected = _baseline_intervals_block(time, edge, before, after, max_segments=3)
    _assert_intervals_equal(actual, expected)


def test_event_numba_006_intervals_bounded_trigger_collision_parity() -> None:
    """ID: EVENT_NUMBA_006_intervals_bounded_trigger_collision_parity."""
    _require_numba()
    time = np.asarray([[1.0, 1.0, 1.0, 2.0]])
    edge = np.asarray([[EDGE_ENTER, EDGE_TRIGGER, EDGE_EXIT, EDGE_INVALID]], dtype="int8")
    before = np.asarray([[0, 1, 1, SAMPLE_SENTINEL]], dtype="int64")
    after = np.asarray([[1, 1, 2, SAMPLE_SENTINEL]], dtype="int64")
    actual = intervals_bounded_block_backend(
        time,
        edge,
        before,
        after,
        max_segments=3,
        owner="events.intervals",
        backend=EVENT_INTERVALS_BACKEND_NUMBA,
    )
    expected = _baseline_intervals_block(time, edge, before, after, max_segments=3)
    _assert_intervals_equal(actual, expected)


def test_event_numba_007_intervals_bounded_truncation_padding_parity() -> None:
    """ID: EVENT_NUMBA_007_intervals_bounded_truncation_padding_parity."""
    _require_numba()
    time = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    edge = np.asarray([[EDGE_ENTER, EDGE_EXIT, EDGE_TRIGGER, EDGE_ENTER, EDGE_EXIT, EDGE_INVALID]], dtype="int8")
    before = np.asarray([[0, 1, 3, 3, 4, SAMPLE_SENTINEL]], dtype="int64")
    after = np.asarray([[1, 2, 3, 4, 5, SAMPLE_SENTINEL]], dtype="int64")
    for max_segments in (1, 5):
        actual = intervals_bounded_block_backend(
            time,
            edge,
            before,
            after,
            max_segments=max_segments,
            owner="events.intervals",
            backend=EVENT_INTERVALS_BACKEND_NUMBA,
        )
        expected = _baseline_intervals_block(time, edge, before, after, max_segments=max_segments)
        _assert_intervals_equal(actual, expected)
