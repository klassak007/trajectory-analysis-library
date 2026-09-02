from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.event_ops import Condition, AtBoundariesOptions


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


def _chunked_ao_series(*, values: list[float], time: list[float], chunks: int = 2) -> AnalysisObject:
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray(values, dtype="float64"), chunks=chunks))},
        coords={
            "sample": np.arange(len(values), dtype="int64"),
            "time": ("sample", da.from_array(np.asarray(time, dtype="float64"), chunks=chunks)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )


def _sequence_dim(ao: AnalysisObject) -> str:
    roles = ao.as_dataset(copy="none").attrs["tal"]["core"]["roles"]
    return str(roles["sequence_dim"])


def test_event_when_001_edges_enter_exit_all_selection_semantics() -> None:
    """ID: EVENT_WHEN_001_edges_enter_exit_all_selection_semantics."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 1.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out_all = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="all", mode="all"))
    out_enter = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="enter", mode="all"))
    out_exit = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="exit", mode="all"))
    seq_all = _sequence_dim(out_all)
    seq_enter = _sequence_dim(out_enter)
    seq_exit = _sequence_dim(out_exit)
    np.testing.assert_allclose(out_all.as_dataset(copy="none")["time"].values, np.asarray([1.0, 2.0, 4.0, 4.0], dtype="float64"))
    np.testing.assert_array_equal(out_all.as_dataset(copy="none")["event_edge_code"].values, np.asarray([1, 2, 1, 2], dtype="int8"))
    np.testing.assert_allclose(out_enter.as_dataset(copy="none")["time"].values, np.asarray([1.0, 4.0], dtype="float64"))
    np.testing.assert_array_equal(out_enter.as_dataset(copy="none")["event_edge_code"].values, np.asarray([1, 1], dtype="int8"))
    np.testing.assert_allclose(out_exit.as_dataset(copy="none")["time"].values, np.asarray([2.0, 4.0], dtype="float64"))
    np.testing.assert_array_equal(out_exit.as_dataset(copy="none")["event_edge_code"].values, np.asarray([2, 2], dtype="int8"))
    assert out_all.as_dataset(copy="none").sizes[seq_all] == 4
    assert out_enter.as_dataset(copy="none").sizes[seq_enter] == 2
    assert out_exit.as_dataset(copy="none").sizes[seq_exit] == 2


def test_event_when_002_mode_all_first_firstn_last_semantics() -> None:
    """ID: EVENT_WHEN_002_mode_all_first_firstn_last_semantics."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 1.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out_all = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(mode="all"))
    out_first = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(mode="first"))
    out_last = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(mode="last"))
    out_first_n = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(mode="first_n", max_events=2))
    np.testing.assert_allclose(out_all.as_dataset(copy="none")["time"].values, np.asarray([1.0, 2.0, 4.0, 4.0], dtype="float64"))
    np.testing.assert_allclose(out_first.as_dataset(copy="none")["time"].values, np.asarray([1.0], dtype="float64"))
    np.testing.assert_allclose(out_last.as_dataset(copy="none")["time"].values, np.asarray([4.0], dtype="float64"))
    np.testing.assert_array_equal(out_last.as_dataset(copy="none")["event_edge_code"].values, np.asarray([2], dtype="int8"))
    np.testing.assert_allclose(out_first_n.as_dataset(copy="none")["time"].values, np.asarray([1.0, 2.0], dtype="float64"))
    with pytest.raises(ValueError) as err:
        ao.events.at_boundaries(cond, opts=AtBoundariesOptions(mode="first_n"))
    assert "opts.mode='first_n' requires opts.max_events" in str(err.value)


def test_event_when_003_on_empty_empty_and_error_contract() -> None:
    """ID: EVENT_WHEN_003_on_empty_empty_and_error_contract."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    out = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(on_empty="empty"))
    seq = _sequence_dim(out)
    assert out.as_dataset(copy="none").sizes[seq] == 0
    with pytest.raises(ValueError) as err:
        ao.events.at_boundaries(cond, opts=AtBoundariesOptions(on_empty="error"))
    assert "no selected boundaries" in str(err.value)


def test_event_when_004_when_attaches_interval_boundary_metadata_deterministically() -> None:
    """ID: EVENT_WHEN_004_when_attaches_interval_boundary_metadata_deterministically."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 1.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="all", mode="all"))
    seq = _sequence_dim(out)
    assert "event_edge_code" in out.as_dataset(copy="none").coords
    assert "event_sample_index_before" in out.as_dataset(copy="none").coords
    assert "event_sample_index_after" in out.as_dataset(copy="none").coords
    for name in ("event_edge_code", "event_sample_index_before", "event_sample_index_after"):
        assert out.as_dataset(copy="none").coords[name].dims == (seq,)
    np.testing.assert_array_equal(out.as_dataset(copy="none")["event_edge_code"].values, np.asarray([1, 2, 1, 2], dtype="int8"))
    np.testing.assert_array_equal(
        out.as_dataset(copy="none")["event_sample_index_before"].values,
        np.asarray([0, 2, 3, 4], dtype="int64"),
    )
    np.testing.assert_array_equal(
        out.as_dataset(copy="none")["event_sample_index_after"].values,
        np.asarray([1, 3, 4, -1], dtype="int64"),
    )


def test_event_when_hard_001_chunked_without_max_events_fails_fast() -> None:
    """ID: EVENT_WHEN_HARD_001_chunked_without_max_events_fails_fast."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError) as err:
        ao.events.at_boundaries(Condition.compare(Condition.var("value"), "gt", 0.5))
    assert "events.at_boundaries: chunked event extraction requires opts.max_events" in str(err.value)


def test_event_when_hard_002_chunked_with_explicit_max_events_allowed() -> None:
    """ID: EVENT_WHEN_HARD_002_chunked_with_explicit_max_events_allowed."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    out = ao.events.at_boundaries(
        Condition.compare(Condition.var("value"), "gt", 0.5),
        opts=AtBoundariesOptions(mode="first_n", max_events=3),
    )
    seq = _sequence_dim(out)
    assert out.as_dataset(copy="none").sizes[seq] == 3
    assert hasattr(out.as_dataset(copy="none")["value"].data, "chunks")


def test_event_when_hard_003_grouped_empty_batch_when_returns_empty_with_on_empty_empty() -> None:
    """ID: EVENT_WHEN_HARD_003_grouped_empty_batch_when_returns_empty_with_on_empty_empty."""
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
    out = ao.events.at_boundaries(Condition.compare(Condition.var("value"), "gt", 0.5), opts=AtBoundariesOptions(on_empty="empty"))
    seq = _sequence_dim(out)
    assert out.as_dataset(copy="none").sizes["trial"] == 0
    assert out.as_dataset(copy="none").sizes[seq] == 0
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["trial"].values, np.asarray([], dtype="int64"))


def test_event_when_hard_004_exit_first_n_not_truncated_before_filtering() -> None:
    """ID: EVENT_WHEN_HARD_004_exit_first_n_not_truncated_before_filtering."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="exit", mode="first_n", max_events=1))
    np.testing.assert_allclose(out.as_dataset(copy="none")["time"].values, np.asarray([2.0], dtype="float64"))
    np.testing.assert_array_equal(out.as_dataset(copy="none")["event_edge_code"].values, np.asarray([2], dtype="int8"))


def test_event_when_hard_005_mode_first_with_max_events_gt1_no_gufunc_dim_crash() -> None:
    """ID: EVENT_WHEN_HARD_005_mode_first_with_max_events_gt1_no_gufunc_dim_crash."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 1.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(mode="first", max_events=2))
    np.testing.assert_allclose(out.as_dataset(copy="none")["time"].values, np.asarray([1.0], dtype="float64"))
    np.testing.assert_array_equal(out.as_dataset(copy="none")["event_edge_code"].values, np.asarray([1], dtype="int8"))


def test_event_when_hard_006_mode_last_with_max_events_gt1_no_gufunc_dim_crash_unchunked() -> None:
    """ID: EVENT_WHEN_HARD_006_mode_last_with_max_events_gt1_no_gufunc_dim_crash_unchunked."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 1.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(mode="last", max_events=2))
    np.testing.assert_allclose(out.as_dataset(copy="none")["time"].values, np.asarray([4.0], dtype="float64"))
    np.testing.assert_array_equal(out.as_dataset(copy="none")["event_edge_code"].values, np.asarray([2], dtype="int8"))


def test_event_when_hard_007_chunked_last_mode_failfast_if_exact_last_not_supported() -> None:
    """ID: EVENT_WHEN_HARD_007_chunked_last_mode_failfast_if_exact_last_not_supported."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.at_boundaries(cond, opts=AtBoundariesOptions(mode="last", max_events=2))
    assert "opts.mode='last' requires unchunked boundary selection for exact semantics" in str(err.value)


def test_event_when_hard_008_chunked_edge_filtered_first_n_selection_stable() -> None:
    """ID: EVENT_WHEN_HARD_008_chunked_edge_filtered_first_n_selection_stable."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="exit", mode="first_n", max_events=1))
    np.testing.assert_allclose(out.as_dataset(copy="none")["time"].values, np.asarray([2.0], dtype="float64"))
    np.testing.assert_array_equal(out.as_dataset(copy="none")["event_edge_code"].values, np.asarray([2], dtype="int8"))


def test_event_when_hard_009_on_empty_empty_fixed_modes_return_zero_length() -> None:
    """ID: EVENT_WHEN_HARD_009_on_empty_empty_fixed_modes_return_zero_length."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    cases = [
        AtBoundariesOptions(mode="first", on_empty="empty"),
        AtBoundariesOptions(mode="last", on_empty="empty"),
        AtBoundariesOptions(mode="first_n", max_events=2, on_empty="empty"),
    ]
    for options in cases:
        out = ao.events.at_boundaries(cond, opts=options)
        seq = _sequence_dim(out)
        assert out.as_dataset(copy="none").sizes[seq] == 0


def test_event_hard_019_boundary_select_bounded_blockwise_selection_parity() -> None:
    """ID: EVENT_HARD_019_boundary_select_bounded_blockwise_selection_parity."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 1.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out_all = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="all", mode="all", max_events=3))
    out_first = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="all", mode="first", max_events=3))
    out_last = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="all", mode="last", max_events=3))
    out_first_n = ao.events.at_boundaries(cond, opts=AtBoundariesOptions(edges="all", mode="first_n", max_events=2))
    np.testing.assert_allclose(out_all.as_dataset(copy="none")["time"].values, np.asarray([1.0, 2.0, 4.0], dtype="float64"))
    np.testing.assert_allclose(out_first.as_dataset(copy="none")["time"].values, np.asarray([1.0], dtype="float64"))
    np.testing.assert_allclose(out_last.as_dataset(copy="none")["time"].values, np.asarray([4.0], dtype="float64"))
    np.testing.assert_allclose(out_first_n.as_dataset(copy="none")["time"].values, np.asarray([1.0, 2.0], dtype="float64"))
