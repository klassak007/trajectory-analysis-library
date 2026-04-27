from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.event_ops import Condition, IntervalExtractOptions
import tal.core.event_ops.intervals as intervals_mod


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
    original = intervals_mod.intervals_bounded_row_backend

    def _count(*args: object, **kwargs: object):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(intervals_mod, "intervals_bounded_row_backend", _count)
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    _ = ao.events.intervals(
        Condition.compare(Condition.var("value"), "gt", 0.5),
        opts=IntervalExtractOptions(max_segments=3),
    )
    assert calls["n"] >= 1
