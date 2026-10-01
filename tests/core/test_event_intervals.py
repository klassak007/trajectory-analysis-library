from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

import tal.core.event_ops.intervals as intervals_mod
from tal.core import AnalysisObject
from tal.core.event_ops import (
    Condition,
    ConditionEvalOptions,
    EventExtractOptions,
    IntervalExtractOptions,
)


def _ao_series(*, values: list[float], time: list[float]) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray(values, dtype="float64"))},
        coords={
            "sample": np.arange(len(values), dtype="int64"),
            "time": ("sample", np.asarray(time, dtype="float64")),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )


def _segment_and_edge_dims(ds: xr.Dataset, *, batch_dims: tuple[str, ...] = ()) -> tuple[str, str]:
    dims = [dim for dim in ds["time"].dims if dim not in batch_dims]
    assert len(dims) == 2
    edge = [
        dim
        for dim in dims
        if ds.sizes[dim] == 2 and set(map(str, ds.coords[dim].values.tolist())) == {"start", "end"}
    ]
    assert len(edge) == 1
    segment = next(dim for dim in dims if dim != edge[0])
    return segment, edge[0]


def test_event_interval_001_composite_boundary_refine_and_or_semantics() -> None:
    """ID: EVENT_INTERVAL_001_composite_boundary_refine_and_or_semantics."""
    ao = _ao_series(values=[0.0, 1.0, 2.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = (
        Condition.compare(Condition.var("value"), "gt", 0.5)
        & Condition.compare(Condition.var("value"), "lt", 1.5)
    ) | Condition.compare(Condition.var("value"), "eq", 2.0)
    out = ao.events.intervals(cond)
    segment_dim, _ = _segment_and_edge_dims(out)
    np.testing.assert_array_equal(out["valid_segment"].values, np.asarray([True], dtype=bool))
    np.testing.assert_allclose(out["time"].isel({segment_dim: 0}).values, np.asarray([1.0, 3.0], dtype="float64"))
    np.testing.assert_array_equal(
        out["sample_index"].isel({segment_dim: 0}).values,
        np.asarray([1, 3], dtype="int64"),
    )
    assert bool(out["is_trigger"].isel({segment_dim: 0}).item()) is False


def test_event_interval_002_equality_trigger_degenerate_segment() -> None:
    """ID: EVENT_INTERVAL_002_equality_trigger_degenerate_segment."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    out = ao.events.intervals(Condition.compare(Condition.var("value"), "eq", 1.0))
    segment_dim, _ = _segment_and_edge_dims(out)
    valid = np.asarray(out["valid_segment"].values, dtype=bool)
    is_trigger = np.asarray(out["is_trigger"].values, dtype=bool) & valid
    trigger_idx = np.flatnonzero(is_trigger)
    assert trigger_idx.size >= 1
    for idx in trigger_idx:
        time_row = out["time"].isel({segment_dim: int(idx)}).values
        sample_row = out["sample_index"].isel({segment_dim: int(idx)}).values
        assert float(time_row[0]) == float(time_row[1])
        assert int(sample_row[0]) == int(sample_row[1])


def test_event_interval_003_empty_result_shape_is_deterministic() -> None:
    """ID: EVENT_INTERVAL_003_empty_result_shape_is_deterministic."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    out = ao.events.intervals(Condition.compare(Condition.var("value"), "gt", 1.0))
    segment_dim, edge_dim = _segment_and_edge_dims(out)
    assert out.sizes[segment_dim] == 0
    assert out.sizes[edge_dim] == 2
    assert out["valid_segment"].size == 0


def test_event_interval_004_eq_isolated_hit_no_duplicate_degenerate_rows() -> None:
    """ID: EVENT_INTERVAL_004_eq_isolated_hit_no_duplicate_degenerate_rows."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    out = ao.events.intervals(Condition.compare(Condition.var("value"), "eq", 1.0))
    segment_dim, _ = _segment_and_edge_dims(out)
    valid = np.asarray(out["valid_segment"].values, dtype=bool)
    valid_idx = np.flatnonzero(valid)
    assert valid_idx.size == 1
    idx = int(valid_idx[0])
    assert bool(out["is_trigger"].isel({segment_dim: idx}).item()) is True
    np.testing.assert_allclose(out["time"].isel({segment_dim: idx}).values, np.asarray([1.0, 1.0], dtype="float64"))
    np.testing.assert_array_equal(
        out["sample_index"].isel({segment_dim: idx}).values,
        np.asarray([1, 1], dtype="int64"),
    )


def test_event_interval_005_eq_contiguous_run_preserves_nontrigger_span_and_trigger_rows() -> None:
    """ID: EVENT_INTERVAL_005_eq_contiguous_run_preserves_nontrigger_span_and_trigger_rows."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    out = ao.events.intervals(Condition.compare(Condition.var("value"), "eq", 1.0))
    segment_dim, _ = _segment_and_edge_dims(out)
    valid = np.asarray(out["valid_segment"].values, dtype=bool)
    trigger = np.asarray(out["is_trigger"].values, dtype=bool)
    valid_idx = np.flatnonzero(valid)
    trigger_idx = np.flatnonzero(valid & trigger)
    span_idx = np.flatnonzero(valid & (~trigger))
    assert valid_idx.size == 3
    assert trigger_idx.size == 2
    assert span_idx.size == 1
    span = int(span_idx[0])
    np.testing.assert_allclose(out["time"].isel({segment_dim: span}).values, np.asarray([1.0, 2.0], dtype="float64"))
    np.testing.assert_array_equal(
        out["sample_index"].isel({segment_dim: span}).values,
        np.asarray([1, 2], dtype="int64"),
    )
    for idx in trigger_idx:
        row = int(idx)
        time_row = out["time"].isel({segment_dim: row}).values
        sample_row = out["sample_index"].isel({segment_dim: row}).values
        assert float(time_row[0]) == float(time_row[1])
        assert int(sample_row[0]) == int(sample_row[1])


def test_event_interval_hard_001_chunked_without_max_segments_fails_fast() -> None:
    """ID: EVENT_INTERVAL_HARD_001_chunked_without_max_segments_fails_fast."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 1.0, 0.0]), chunks=2))},
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="time")
    with pytest.raises(ValueError) as err:
        ao.events.intervals(Condition.compare(Condition.var("value"), "gt", 0.5))
    assert "chunked interval extraction requires opts.max_segments" in str(err.value)


def test_event_interval_hard_002_chunked_with_explicit_max_segments_allowed() -> None:
    """ID: EVENT_INTERVAL_HARD_002_chunked_with_explicit_max_segments_allowed."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 1.0, 0.0]), chunks=2))},
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="time")
    out = ao.events.intervals(
        Condition.compare(Condition.var("value"), "gt", 0.5),
        opts=IntervalExtractOptions(max_segments=3),
    )
    segment_dim, edge_dim = _segment_and_edge_dims(out)
    assert out.sizes[segment_dim] == 3
    assert out.sizes[edge_dim] == 2
    assert hasattr(out["time"].data, "chunks")


def test_event_interval_hard_003_grouped_empty_batch_intervals_returns_empty_table() -> None:
    """ID: EVENT_INTERVAL_HARD_003_grouped_empty_batch_intervals_returns_empty_table."""
    values = np.empty((0, 3), dtype="float64")
    time = np.empty((0, 3), dtype="float64")
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), values)},
        coords={
            "trial": np.asarray([], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sample"), time),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    out = ao.events.intervals(Condition.compare(Condition.var("value"), "gt", 0.5))
    segment_dim, edge_dim = _segment_and_edge_dims(out, batch_dims=("trial",))
    assert out.sizes["trial"] == 0
    assert out.sizes[segment_dim] == 0
    assert out.sizes[edge_dim] == 2
    np.testing.assert_array_equal(out.coords["trial"].values, np.asarray([], dtype="int64"))


def test_event_hard_023_intervals_bounded_stopgap_backend_interface_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: EVENT_HARD_023_intervals_bounded_stopgap_backend_interface_lock."""
    calls = {"n": 0}
    original = intervals_mod.intervals_bounded_block_backend

    def _count(*args: object, **kwargs: object):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(intervals_mod, "_numba_available", lambda: False)
    monkeypatch.setattr(intervals_mod, "intervals_bounded_block_backend", _count)
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    _ = ao.events.intervals(
        Condition.compare(Condition.var("value"), "gt", 0.5),
        opts=IntervalExtractOptions(max_segments=3),
    )
    assert calls["n"] >= 1


@pytest.mark.parametrize("batch", ["__tal_segment__", "__tal_edge__", "__tal_event__"])
@pytest.mark.parametrize("lazy,limit,empty", [(False, None, False), (True, 2, False), (False, None, True), (True, 2, True)])
def test_tut_009_interval_dimensions_allocated_before_payload(batch, lazy, limit, empty):
    """TUT-009: source names and occupied suffixes stay available to users."""
    from dask.callbacks import Callback

    n = 0 if empty else 4
    ds = xr.Dataset({"value": ((batch, "sample"), np.tile([0., 3., 4., 0.][:n], (2, 1)))},
                    coords={batch: ["a", "b"], "sample": np.arange(n), "time": ("sample", np.array([0., .2, .7, 1.5])[:n]),
                            "__tal_segment___": "occupied", "__tal_edge___": "occupied", "__tal_event___": "occupied"})
    source = AnalysisObject.from_data(ds.chunk({"sample": 2}) if lazy else ds, sequence_dim="sample", batch_dims=(batch,), param_coord="time")
    snapshot = source.as_dataset(copy="deep")
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.events.intervals(Condition.compare(Condition.var("value"), "gt", 2.),
                                        opts=IntervalExtractOptions(max_segments=limit))
    assert tasks == []
    actual = result.compute(scheduler="synchronous")
    segment, edge = _segment_and_edge_dims(actual, batch_dims=(batch,))
    assert len({batch, segment, edge}) == 3
    if not empty:
        np.testing.assert_allclose(actual.time.isel({segment: 0}), [[.2, .7], [.2, .7]])
        np.testing.assert_array_equal(actual.sample_index.isel({segment: 0}), [[1, 2], [1, 2]])
    assert actual.xindexes[batch].equals(snapshot.xindexes[batch])
    xr.testing.assert_identical(source.as_dataset(), snapshot)


COLUMNS = {
    "events": ["time", "edge_code", "sample_index_before", "sample_index_after"],
    "intervals": ["time", "sample_index", "is_trigger", "valid_segment"],
}
TABLE_CASES = [
    (op, name, form, lazy)
    for op, names in COLUMNS.items()
    for name in names
    for form, lazy in [("batch", False), ("batch", True), ("scalar", False)]
]


@pytest.mark.parametrize("operation,name,form,lazy", TABLE_CASES)
def test_tut_014_source_carrier_column_conflicts(
    tutorial_audit_source, operation, name, form, lazy
):
    """TUT-014: public ownership regression and controls."""
    a = tutorial_audit_source(
        clock_values=[0.0, 1.0, 3.0, 5.0, 8.0, 10.0],
        lazy=lazy,
        batch=(2,),
        aux=(name, form),
    )
    before = a.as_dataset(copy="deep")
    c = Condition.compare(Condition.var("value"), "gt", 0.5)
    ev = ConditionEvalOptions(coord_name="clock")
    tasks = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(ValueError, match="^events\\..*: .*conflict.*rename"),
    ):
        (
            a.events.events(c, opts=EventExtractOptions(eval=ev, max_events=4))
            if operation == "events"
            else a.events.intervals(
                c, opts=IntervalExtractOptions(eval=ev, max_segments=3)
            )
        )
    assert not tasks
    xr.testing.assert_identical(a.as_dataset(), before)


@pytest.mark.parametrize(
    "name", ["time", "sample_index", "is_trigger", "valid_segment", "tag"]
)
def test_tut_014_dynamic_carriers(tutorial_audit_source, name):
    """TUT-014: public ownership regression and controls."""
    a = tutorial_audit_source(
        clock_values=[0.0, 1.0, 3.0, 5.0, 8.0, 10.0], batch=(2,), aux=(name, "batch")
    )
    c = Condition.compare(Condition.var("value"), "gt", 0.5)
    if name != "tag":
        with pytest.raises(ValueError, match=r"^events.intervals: .*conflict.*rename"):
            a.events.intervals(
                c,
                opts=IntervalExtractOptions(
                    eval=ConditionEvalOptions(coord_name="clock")
                ),
            )
        return
    out = a.events.intervals(
        c, opts=IntervalExtractOptions(eval=ConditionEvalOptions(coord_name="clock"))
    )
    assert set(COLUMNS["intervals"]) <= set(out.data_vars)
    np.testing.assert_allclose(out.time[0], [[1.0, 3.0], [8.0, 8.0]])
    np.testing.assert_array_equal(out.tag, [20, 21])


@pytest.mark.parametrize("operation", ["events", "intervals"])
@pytest.mark.parametrize(
    "case,lazy",
    [
        ("unbounded", False),
        ("bounded", False),
        ("bounded", True),
        ("empty", True),
        ("zero", True),
        ("no_episodes", False),
    ],
)
def test_tut_014_reviewed_nonconflicting_carriers(
    tutorial_audit_source, operation, case, lazy
):
    """TUT-014: ordinary carriers survive all table layouts without reserving variables."""
    source = tutorial_audit_source(
        batch=(0,) if case == "zero" else (2,),
        n=0 if case == "empty" else 6,
        lazy=lazy,
        aux=("tag", "batch"),
    )
    ds = source.as_dataset().assign_coords(station="lab")
    ds["time"] = xr.Variable(("b0",), np.zeros(ds.sizes["b0"]))
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("b0",), param_coord="clock"
    )
    before = source.as_dataset(copy="deep")
    condition = Condition.compare(
        Condition.var("value"), "gt", 10.0 if case == "no_episodes" else 0.5
    )
    limit = None if case in ("unbounded", "no_episodes") else 3
    evaluation = ConditionEvalOptions(coord_name="clock")
    opts = (
        EventExtractOptions(eval=evaluation, max_events=limit)
        if operation == "events"
        else IntervalExtractOptions(eval=evaluation, max_segments=limit)
    )
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = getattr(source.events, operation)(condition, opts=opts)
    assert not tasks
    assert set(result.data_vars) == set(COLUMNS[operation])
    assert result.tag.dims == ("b0",) and result.station.dims == ()
    np.testing.assert_array_equal(result.tag, before.tag)
    assert result.station.data == "lab"
    assert type(result.xindexes["b0"]) is type(before.xindexes["b0"])
    assert result.xindexes["b0"].equals(before.xindexes["b0"])
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize(
    "operation", ["events", "intervals", "boundaries", "segments", "stream", "windows"]
)
@pytest.mark.parametrize("case", ["unbounded", "empty", "zero", "no_episodes"])
def test_tut_014_ordinary_conflicts_before_empty_or_dependent_work(
    tutorial_audit_source, operation, case
):
    """TUT-014: ordinary claims receive the same early error as native groups."""
    from tal.core.event_ops import AroundOptions, AtBoundariesOptions, WhenOptions

    source = tutorial_audit_source(
        lazy=case != "unbounded",
        batch=(0,) if case == "zero" else (2,),
        n=0 if case == "empty" else 6,
        aux=("time", "batch"),
    )
    before = source.as_dataset(copy="deep")
    condition = Condition.compare(
        Condition.var("value"), "gt", 10.0 if case == "no_episodes" else 0.5
    )
    evaluation = ConditionEvalOptions(coord_name="clock")
    limit = None if case == "unbounded" else 3
    tasks = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(ValueError, match=r"^events\..*: .*conflict.*rename"),
    ):
        if operation == "events":
            source.events.events(
                condition, opts=EventExtractOptions(eval=evaluation, max_events=limit)
            )
        elif operation == "intervals":
            source.events.intervals(
                condition,
                opts=IntervalExtractOptions(eval=evaluation, max_segments=limit),
            )
        elif operation == "boundaries":
            source.events.at_boundaries(
                condition, opts=AtBoundariesOptions(eval=evaluation, max_events=limit)
            )
        elif operation in ("segments", "stream"):
            source.events.when(
                condition,
                opts=WhenOptions(eval=evaluation, layout=operation, max_segments=limit),
            )
        else:
            source.events.around(
                condition, opts=AroundOptions(eval=evaluation, grid=np.array([0.0]))
            )
    assert not tasks
    xr.testing.assert_identical(source.as_dataset(), before)
