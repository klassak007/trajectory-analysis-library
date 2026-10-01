from __future__ import annotations

import itertools

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisObject
from tal.core.event_ops import AroundOptions, Condition, ConditionEvalOptions


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


def _event_dim(ao: AnalysisObject) -> str:
    roles = ao.as_dataset(copy="none").attrs["tal"]["core"]["roles"]
    batch_dims = tuple(roles["batch_dims"])
    assert len(batch_dims) >= 1
    return str(batch_dims[-1])


def _validity_coord_name(ao: AnalysisObject) -> str:
    validity = ao.as_dataset(copy="none").attrs["tal"]["core"]["validity"]
    return str(validity["sequence_size_coord"])


def test_event_around_001_condition_enter_windows() -> None:
    """ID: EVENT_AROUND_001_condition_enter_windows."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 2.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(cond, opts=AroundOptions(edge="enter", dt=1.0, pre=1.0, post=1.0))
    seq = _sequence_dim(out)
    event_dim = _event_dim(out)
    assert out.as_dataset(copy="none").sizes[event_dim] == 2
    np.testing.assert_allclose(out.as_dataset(copy="none").coords[seq].values, np.asarray([-1.0, 0.0, 1.0], dtype="float64"))
    np.testing.assert_allclose(out.as_dataset(copy="none").coords["event_time"].values, np.asarray([1.0, 4.0], dtype="float64"))
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["event_edge_code"].values, np.asarray([1, 1], dtype="int8"))
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel({event_dim: 0}).values,
        np.asarray([0.0, 1.0, 1.0], dtype="float64"),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel({event_dim: 1}).values,
        np.asarray([0.0, 2.0, np.nan], dtype="float64"),
        equal_nan=True,
    )


def test_event_around_002_explicit_event_times_windows() -> None:
    """ID: EVENT_AROUND_002_explicit_event_times_windows."""
    ao = _ao_series(values=[0.0, 1.0, 2.0, 3.0, 4.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    events = xr.DataArray(np.asarray([1.5, 3.0], dtype="float64"), dims=("anchor",), coords={"anchor": [10, 20]})
    out = ao.events.around(events, opts=AroundOptions(dt=0.5, pre=0.5, post=0.5))
    seq = _sequence_dim(out)
    event_dim = _event_dim(out)
    assert out.as_dataset(copy="none").sizes[event_dim] == 2
    np.testing.assert_allclose(out.as_dataset(copy="none").coords[seq].values, np.asarray([-0.5, 0.0, 0.5], dtype="float64"))
    np.testing.assert_allclose(out.as_dataset(copy="none").coords["event_time"].values, np.asarray([1.5, 3.0], dtype="float64"))
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["event_edge_code"].values, np.asarray([3, 3], dtype="int8"))
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["event_sample_index_before"].values,
        np.asarray([-1, -1], dtype="int64"),
    )
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["event_sample_index_after"].values,
        np.asarray([-1, -1], dtype="int64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel({event_dim: 0}).values,
        np.asarray([1.0, 1.5, 2.0], dtype="float64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel({event_dim: 1}).values,
        np.asarray([2.5, 3.0, 3.5], dtype="float64"),
    )


def test_event_around_003_grouped_ragged_windows_truthful_validity() -> None:
    """ID: EVENT_AROUND_003_grouped_ragged_windows_truthful_validity."""
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                np.asarray(
                    [
                        [0.0, 1.0, 0.0, 0.0, 2.0],
                        [0.0, 1.0, 0.0, 99.0, 99.0],
                    ],
                    dtype="float64",
                ),
            )
        },
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(5, dtype="int64"),
            "time": (
                ("trial", "sample"),
                np.asarray(
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                    ],
                    dtype="float64",
                ),
            ),
            "group_size": ("trial", np.asarray([5, 3], dtype="int64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="group_size",
    )
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(
        cond,
        opts=AroundOptions(
            edge="enter",
            dt=1.0,
            pre=0.0,
            post=0.0,
            eval=ConditionEvalOptions(validity_mode="auto"),
        ),
    )
    event_dim = _event_dim(out)
    size_name = _validity_coord_name(out)
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords[size_name].values,
        np.asarray([[1, 1], [1, 0]], dtype="int64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel(trial=1, **{event_dim: 1}).values,
        np.asarray([np.nan], dtype="float64"),
        equal_nan=True,
    )


def test_event_around_003a_grouped_sequence_only_clock_broadcasts_across_batch() -> None:
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                np.asarray(
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype="float64",
                ),
            ),
        },
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", np.asarray([0.0, 1.0, 2.0], dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(cond, opts=AroundOptions(edge="enter", dt=1.0, pre=0.0, post=0.0))
    event_dim = _event_dim(out)
    np.testing.assert_allclose(
        out.as_dataset(copy="none").coords["event_time"].isel({event_dim: 0}).values,
        np.asarray([1.0, 2.0], dtype="float64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel({event_dim: 0}).values,
        np.asarray([[1.0], [1.0]], dtype="float64"),
    )


def test_event_around_004_stacked_layout_flattens_event_tau_in_packed_order() -> None:
    """ID: EVENT_AROUND_004_stacked_layout_flattens_event_tau_in_packed_order."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 2.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(cond, opts=AroundOptions(layout="stacked", edge="enter", dt=1.0, pre=1.0, post=1.0))
    seq = _sequence_dim(out)
    size_name = _validity_coord_name(out)
    assert out.as_dataset(copy="none").sizes[seq] == 6
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["window_event_index"].values,
        np.asarray([0, 0, 0, 1, 1, 1], dtype="int64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none").coords["window_tau"].values,
        np.asarray([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], dtype="float64"),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none").coords["event_time"].values,
        np.asarray([1.0, 1.0, 1.0, 4.0, 4.0, 4.0], dtype="float64"),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].values,
        np.asarray([0.0, 1.0, 1.0, 0.0, 2.0, np.nan], dtype="float64"),
        equal_nan=True,
    )
    assert int(out.as_dataset(copy="none").coords[size_name].values) == 6


def test_event_around_005_stacked_layout_parity_with_segments_values() -> None:
    """ID: EVENT_AROUND_005_stacked_layout_parity_with_segments_values."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 2.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    segments = ao.events.around(cond, opts=AroundOptions(layout="segments", edge="enter", dt=1.0, pre=1.0, post=1.0))
    stacked = ao.events.around(cond, opts=AroundOptions(layout="stacked", edge="enter", dt=1.0, pre=1.0, post=1.0))
    event_dim = _event_dim(segments)
    tau_dim = _sequence_dim(segments)
    stack_dim = _sequence_dim(stacked)
    expected = segments.as_dataset(copy="none")["value"].stack({"window": (event_dim, tau_dim)}).reset_index("window", drop=True)
    got = stacked.as_dataset(copy="none")["value"].rename({stack_dim: "window"})
    np.testing.assert_allclose(got.values, expected.values, equal_nan=True)


def test_event_around_006_stacked_grouped_ragged_validity_truthful() -> None:
    """ID: EVENT_AROUND_006_stacked_grouped_ragged_validity_truthful."""
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                np.asarray(
                    [
                        [0.0, 1.0, 0.0, 0.0, 2.0],
                        [0.0, 1.0, 0.0, 99.0, 99.0],
                    ],
                    dtype="float64",
                ),
            )
        },
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(5, dtype="int64"),
            "time": (
                ("trial", "sample"),
                np.asarray(
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                    ],
                    dtype="float64",
                ),
            ),
            "group_size": ("trial", np.asarray([5, 3], dtype="int64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="group_size",
    )
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(
        cond,
        opts=AroundOptions(
            layout="stacked",
            edge="enter",
            dt=1.0,
            pre=0.0,
            post=0.0,
            eval=ConditionEvalOptions(validity_mode="auto"),
        ),
    )
    seq = _sequence_dim(out)
    size_name = _validity_coord_name(out)
    assert out.as_dataset(copy="none").sizes[seq] == 2
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords[size_name].values, np.asarray([2, 1], dtype="int64"))
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["window_event_index"].isel(trial=1).values,
        np.asarray([0, -1], dtype="int64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none").coords["window_tau"].isel(trial=1).values,
        np.asarray([0.0, np.nan], dtype="float64"),
        equal_nan=True,
    )


def test_event_around_hard_002_grid_and_dt_mutually_exclusive() -> None:
    """ID: EVENT_AROUND_HARD_002_grid_and_dt_mutually_exclusive."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.around(cond, opts=AroundOptions(dt=1.0, grid=np.asarray([0.0], dtype="float64")))
    assert "events.around: opts.dt and opts.grid are mutually exclusive." in str(err.value)


def test_event_around_hard_003_chunked_condition_source_failfast_unbounded() -> None:
    """ID: EVENT_AROUND_HARD_003_chunked_condition_source_failfast_unbounded."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.around(cond, opts=AroundOptions(dt=1.0, pre=0.0, post=0.0))
    assert "events.around: chunked condition-source around extraction requires explicit event anchors." in str(err.value)


def test_event_around_hard_004_metadata_namespace_collision_failfast() -> None:
    """ID: EVENT_AROUND_HARD_004_metadata_namespace_collision_failfast."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray([0.0, 1.0, 0.0], dtype="float64"))},
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", np.asarray([0.0, 1.0, 2.0], dtype="float64")),
            "event_time": np.asarray(123.0, dtype="float64"),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.around(cond, opts=AroundOptions(dt=1.0, pre=0.0, post=0.0))
    assert "events.around: around metadata names conflict with dataset namespace" in str(err.value)


def test_event_around_004a_compatible_unowned_valid_coord_does_not_trigger_reserved_collision() -> None:
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                np.asarray(
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype="float64",
                ),
            )
        },
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (
                ("trial", "sample"),
                np.asarray(
                    [
                        [0.0, 1.0, 2.0],
                        [0.0, 1.0, 2.0],
                    ],
                    dtype="float64",
                ),
            ),
            "valid": (
                ("trial", "sample"),
                np.asarray(
                    [
                        [True, True, True],
                        [True, True, True],
                    ],
                    dtype=bool,
                ),
            ),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(cond, opts=AroundOptions(edge="enter", dt=1.0, pre=0.0, post=0.0))
    assert "value" in out.as_dataset(copy="none").data_vars


def test_event_around_hard_005_explicit_source_invalid_shape_rejected() -> None:
    """ID: EVENT_AROUND_HARD_005_explicit_source_invalid_shape_rejected."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    bad = xr.DataArray(np.zeros((2, 2), dtype="float64"), dims=("a", "b"))
    with pytest.raises(ValueError) as err:
        ao.events.around(bad, opts=AroundOptions(dt=1.0, pre=0.0, post=0.0))
    assert "events.around: explicit event time source must be 0-D or 1-D." in str(err.value)


def test_event_around_hard_006_condition_empty_anchor_set_returns_empty_output() -> None:
    """ID: EVENT_AROUND_HARD_006_condition_empty_anchor_set_returns_empty_output."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(cond, opts=AroundOptions(dt=1.0, pre=1.0, post=1.0))
    seq = _sequence_dim(out)
    event_dim = _event_dim(out)
    assert out.as_dataset(copy="none").sizes[event_dim] == 0
    assert out.as_dataset(copy="none").sizes[seq] == 3
    for name in ("event_time", "event_edge_code", "event_sample_index_before", "event_sample_index_after"):
        assert out.as_dataset(copy="none").coords[name].sizes[event_dim] == 0


def test_event_around_hard_007_explicit_empty_anchor_set_returns_empty_output() -> None:
    """ID: EVENT_AROUND_HARD_007_explicit_empty_anchor_set_returns_empty_output."""
    ao = _ao_series(values=[0.0, 1.0, 2.0], time=[0.0, 1.0, 2.0])
    anchors = xr.DataArray(np.asarray([], dtype="float64"), dims=("anchor",))
    out = ao.events.around(anchors, opts=AroundOptions(dt=1.0, pre=1.0, post=1.0))
    seq = _sequence_dim(out)
    event_dim = _event_dim(out)
    assert out.as_dataset(copy="none").sizes[event_dim] == 0
    assert out.as_dataset(copy="none").sizes[seq] == 3
    assert out.as_dataset(copy="none").coords["event_time"].sizes[event_dim] == 0


def test_event_around_hard_008_grouped_explicit_anchor_preserves_batch_topology() -> None:
    """ID: EVENT_AROUND_HARD_008_grouped_explicit_anchor_preserves_batch_topology."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype="float64"))},
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    anchors = xr.DataArray(
        np.asarray([0.5, 1.5], dtype="float64"),
        dims=("trial",),
        coords={"trial": np.asarray([0, 1], dtype="int64")},
    )
    out = ao.events.around(anchors, opts=AroundOptions(dt=0.5, pre=0.5, post=0.5))
    event_dim = _event_dim(out)
    assert out.as_dataset(copy="none").sizes["trial"] == 2
    assert out.as_dataset(copy="none").sizes[event_dim] == 1
    np.testing.assert_allclose(
        out.as_dataset(copy="none").coords["event_time"].isel({event_dim: 0}).values,
        np.asarray([0.5, 1.5], dtype="float64"),
    )


def test_event_around_hard_009_chunked_grid_rejected_failfast() -> None:
    """ID: EVENT_AROUND_HARD_009_chunked_grid_rejected_failfast."""
    da = pytest.importorskip("dask.array")
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    grid = xr.DataArray(da.from_array(np.asarray([-1.0, 0.0, 1.0], dtype="float64"), chunks=2), dims=("tau",))
    with pytest.raises(ValueError) as err:
        ao.events.around(cond, opts=AroundOptions(grid=grid, pre=0.0, post=0.0))
    assert "events.around: opts.grid must be unchunked" in str(err.value)


def test_event_around_hard_010_nonfinite_grid_rejected_deterministically() -> None:
    """ID: EVENT_AROUND_HARD_010_nonfinite_grid_rejected_deterministically."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.around(
            cond,
            opts=AroundOptions(grid=np.asarray([-1.0, np.nan, 1.0], dtype="float64"), pre=0.0, post=0.0),
        )
    assert "events.around: opts.grid must contain only finite values." in str(err.value)


def test_event_around_hard_011_grouped_explicit_batch_event_layout_preserved() -> None:
    """ID: EVENT_AROUND_HARD_011_grouped_explicit_batch_event_layout_preserved."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype="float64"))},
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    anchors = xr.DataArray(
        np.asarray([[0.5, 1.5], [1.0, 2.0]], dtype="float64"),
        dims=("trial", "anchor"),
        coords={"trial": np.asarray([0, 1], dtype="int64"), "anchor": np.asarray([10, 20], dtype="int64")},
    )
    out = ao.events.around(anchors, opts=AroundOptions(dt=0.5, pre=0.0, post=0.0))
    event_dim = _event_dim(out)
    assert out.as_dataset(copy="none").sizes[event_dim] == 2
    np.testing.assert_allclose(out.as_dataset(copy="none").coords["event_time"].values, anchors.values)


def test_event_around_hard_012_stacked_empty_anchor_returns_empty_output() -> None:
    """ID: EVENT_AROUND_HARD_012_stacked_empty_anchor_returns_empty_output."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(cond, opts=AroundOptions(layout="stacked", dt=1.0, pre=1.0, post=1.0))
    seq = _sequence_dim(out)
    size_name = _validity_coord_name(out)
    assert out.as_dataset(copy="none").sizes[seq] == 0
    assert int(out.as_dataset(copy="none").coords[size_name].values) == 0


def test_event_around_hard_013_stacked_metadata_namespace_collision_failfast() -> None:
    """ID: EVENT_AROUND_HARD_013_stacked_metadata_namespace_collision_failfast."""
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), np.asarray([0.0, 1.0, 0.0], dtype="float64")),
            "window_tau": (("sample",), np.asarray([0.0, 0.0, 0.0], dtype="float64")),
        },
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", np.asarray([0.0, 1.0, 2.0], dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="time")
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.around(cond, opts=AroundOptions(layout="stacked", dt=1.0, pre=0.0, post=0.0))
    assert "events.around: stacked metadata names conflict with dataset namespace" in str(err.value)


def test_event_around_hard_014_stacked_condition_chunked_source_failfast() -> None:
    """ID: EVENT_AROUND_HARD_014_stacked_condition_chunked_source_failfast."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.around(cond, opts=AroundOptions(layout="stacked", dt=1.0, pre=0.0, post=0.0))
    assert "events.around: chunked condition-source around extraction requires explicit event anchors." in str(err.value)


def test_event_around_hard_015_stacked_explicit_grouped_zero_lane_deterministic() -> None:
    """ID: EVENT_AROUND_HARD_015_stacked_explicit_grouped_zero_lane_deterministic."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.empty((0, 3), dtype="float64"))},
        coords={
            "trial": np.asarray([], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sample"), np.empty((0, 3), dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    anchors = xr.DataArray(
        np.empty((0, 2), dtype="float64"),
        dims=("trial", "anchor"),
        coords={"trial": np.asarray([], dtype="int64"), "anchor": np.asarray([0, 1], dtype="int64")},
    )
    out = ao.events.around(anchors, opts=AroundOptions(layout="stacked", dt=1.0, pre=0.0, post=0.0))
    seq = _sequence_dim(out)
    assert out.as_dataset(copy="none").sizes["trial"] == 0
    assert out.as_dataset(copy="none").sizes[seq] == 2


def test_event_around_hard_016_stacked_no_hidden_eager_count_discovery() -> None:
    """ID: EVENT_AROUND_HARD_016_stacked_no_hidden_eager_count_discovery."""
    da = pytest.importorskip("dask.array")
    ao = _chunked_ao_series(values=[0.0, 1.0, 2.0], time=[0.0, 1.0, 2.0])
    anchors = xr.DataArray(da.from_array(np.asarray([0.5, 1.5], dtype="float64"), chunks=2), dims=("anchor",))
    out = ao.events.around(anchors, opts=AroundOptions(layout="stacked", dt=1.0, pre=0.0, post=0.0))
    assert out.as_dataset(copy="none")["value"].chunks is not None


def test_event_around_hard_017_stacked_zero_lane_batch_coord_dtype_preserved() -> None:
    """ID: EVENT_AROUND_HARD_017_stacked_zero_lane_batch_coord_dtype_preserved."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.empty((0, 3), dtype="float64"))},
        coords={
            "trial": np.asarray([], dtype=object),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sample"), np.empty((0, 3), dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    anchors = xr.DataArray(
        np.empty((0, 2), dtype="float64"),
        dims=("trial", "anchor"),
        coords={"trial": np.asarray([], dtype=object), "anchor": np.asarray([0, 1], dtype="int64")},
    )
    segments = ao.events.around(anchors, opts=AroundOptions(layout="segments", dt=1.0, pre=0.0, post=0.0))
    stacked = ao.events.around(anchors, opts=AroundOptions(layout="stacked", dt=1.0, pre=0.0, post=0.0))
    assert segments.as_dataset(copy="none").coords["trial"].dtype == ds.coords["trial"].dtype
    assert stacked.as_dataset(copy="none").coords["trial"].dtype == ds.coords["trial"].dtype
    assert stacked.as_dataset(copy="none").coords["trial"].dtype == segments.as_dataset(copy="none").coords["trial"].dtype


def test_event_around_hard_018_stacked_zero_lane_batch_coord_values_parity_with_source() -> None:
    """ID: EVENT_AROUND_HARD_018_stacked_zero_lane_batch_coord_values_parity_with_source."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.empty((0, 3), dtype="float64"))},
        coords={
            "trial": np.asarray([], dtype=object),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sample"), np.empty((0, 3), dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    anchors = xr.DataArray(
        np.empty((0, 2), dtype="float64"),
        dims=("trial", "anchor"),
        coords={"trial": np.asarray([], dtype=object), "anchor": np.asarray([0, 1], dtype="int64")},
    )
    stacked = ao.events.around(anchors, opts=AroundOptions(layout="stacked", dt=1.0, pre=0.0, post=0.0))
    np.testing.assert_array_equal(stacked.as_dataset(copy="none").coords["trial"].values, ds.coords["trial"].values)


def test_event_hard_018_around_finalize_owner_preserves_param_and_validity() -> None:
    """ID: EVENT_HARD_018_around_finalize_owner_preserves_param_and_validity."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.around(cond, opts=AroundOptions(layout="segments", edge="enter", dt=1.0, pre=0.0, post=0.0))
    seq = _sequence_dim(out)
    core = out.as_dataset(copy="none").attrs["tal"]["core"]
    assert core["param_coord"]["name"] == seq
    validity = core["validity"]
    assert isinstance(validity, dict)
    assert str(validity["sequence_size_coord"]) in out.as_dataset(copy="none").coords


ANCHOR_CASES = list(
    itertools.product(
        [False, True],
        ["segments", "stacked"],
        ["unindexed", "indexed", "axis_indexed"],
        [0, 2],
    )
)


@pytest.mark.parametrize("lazy,layout,index_kind,n", ANCHOR_CASES)
def test_tut_016_explicit_anchor_namespace(
    tutorial_audit_source, lazy, layout, index_kind, n
):
    """TUT-016: public ownership regression and controls."""
    a = tutorial_audit_source(
        clock_values=[0.0, 1.0, 3.0, 5.0, 8.0, 10.0], lazy=lazy, batch=()
    )
    coordinates = {
        "event": ("anchor", np.arange(n) + 20),
        "event_1": ("anchor", np.arange(n) + 30),
    }
    if index_kind == "axis_indexed":
        coordinates["anchor"] = np.arange(n) + 40
    q = xr.DataArray(np.array([1.0, 8.0])[:n], dims="anchor", coords=coordinates)
    if index_kind == "indexed":
        q = q.set_xindex("event")
    if lazy:
        q = q.chunk(anchor=1)
    before = a.as_dataset(copy="deep")
    original = q.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        out = a.events.around(
            q,
            opts=AroundOptions(
                eval=ConditionEvalOptions(coord_name="clock"),
                grid=np.array([0.0]),
                layout=layout,
            ),
        ).as_dataset(copy="none")
    assert not tasks
    computed = out.compute(scheduler="synchronous")
    if layout == "segments":
        event_dim = out.attrs["tal"]["core"]["roles"]["batch_dims"][-1]
        assert event_dim not in {"event", "event_1"}
        np.testing.assert_array_equal(computed.event_time, [1.0, 8.0][:n])
        np.testing.assert_array_equal(computed.value, np.ones((n, 1)))
        assert computed.event.dims == (event_dim,)
        assert ("event" in out.xindexes) == (index_kind == "indexed")
    else:
        np.testing.assert_array_equal(computed.value, np.ones(n))
        assert "event" in out.coords
    xr.testing.assert_identical(a.as_dataset(), before)
    xr.testing.assert_identical(q, original)


@pytest.mark.parametrize(
    "lazy,labels",
    list(itertools.product([False, True], [[0, 1], [10, 20], ["alpha", "beta"]])),
)
def test_tut_019_stacked_window_row_positions(tutorial_audit_source, lazy, labels):
    """TUT-019: public ownership regression and controls."""
    a = tutorial_audit_source(
        clock_values=[0.0, 1.0, 3.0, 5.0, 8.0, 10.0], lazy=lazy, batch=()
    )
    q = xr.DataArray([1.0, 8.0], dims="anchor", coords={"anchor": labels})
    if lazy:
        q = q.chunk(anchor=1)
    before = q.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        out = a.events.around(
            q,
            opts=AroundOptions(
                eval=ConditionEvalOptions(coord_name="clock"),
                grid=np.array([0.0]),
                layout="stacked",
            ),
        ).as_dataset()
    assert not tasks
    np.testing.assert_array_equal(
        out.window_event_index.compute(scheduler="synchronous"), [0, 1]
    )
    np.testing.assert_array_equal(
        out.value.compute(scheduler="synchronous"), [1.0, 1.0]
    )
    xr.testing.assert_identical(q, before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("layout", ["segments", "stacked"])
@pytest.mark.parametrize(
    "case", ["labels", "unindexed", "invalid", "empty", "zero_batch"]
)
def test_tut_018_019_window_topology_and_multi_tau_provenance(
    tutorial_audit_source, lazy, layout, case
):
    """TUT-018/019: labels survive as metadata while provenance enumerates source rows."""
    source = tutorial_audit_source(
        lazy=lazy,
        batch=(0,) if case == "zero_batch" else (),
        clock_values=[0.0, 1.0, 3.0, 5.0, 8.0, 10.0],
    )
    n = 0 if case == "empty" else 2
    variables = {
        "anchor": xr.Variable("anchor", np.array(["alpha", "beta"])[:n]),
        "anchor_note": xr.Variable("anchor", np.arange(n) + 30),
    }
    anchors = xr.DataArray(
        np.array([np.nan if case == "invalid" else 1.0, 8.0])[:n],
        dims="anchor",
        coords=xr.Coordinates(variables, indexes={}),
    )
    if case == "labels":
        anchors = anchors.set_xindex("anchor").set_xindex("anchor_note")
    if lazy:
        anchors = anchors.chunk(anchor=1)
    before, original = source.as_dataset(copy="deep"), anchors.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.events.around(
            anchors,
            opts=AroundOptions(
                eval=ConditionEvalOptions(coord_name="clock"),
                grid=np.array([0.0, 0.5]),
                layout=layout,
            ),
        ).as_dataset()
    assert not tasks
    computed = result.compute(scheduler="synchronous")
    assert set(result.data_vars) == {"value"}
    if layout == "segments":
        dim = result.attrs["tal"]["core"]["roles"]["batch_dims"][-1]
        for name, index in anchors.xindexes.items():
            target = dim if name == "anchor" else name
            expected = index.rename({"anchor": dim}, {"anchor": dim})
            assert type(result.xindexes[target]) is type(expected)
            assert result.xindexes[target].equals(expected)
        if case != "labels":
            assert dim not in result.xindexes
        assert result.anchor_note.dims == (dim,)
    else:
        expected = np.repeat(np.arange(n), 2)
        if case == "invalid":
            expected = np.array([1, 1, -1, -1])
        if case == "zero_batch":
            expected = np.empty((0, n * 2), dtype="int64")
        np.testing.assert_array_equal(computed.window_event_index, expected)
        assert computed.window_event_index.dtype == np.dtype("int64")
        assert set(result.xindexes) == ({"b0"} if case == "zero_batch" else set())
    if case not in ("empty", "zero_batch"):
        expected_clock = (
            np.array([[np.nan, np.nan], [8.0, 8.5]])
            if case == "invalid"
            else np.array([[1.0, 1.5], [8.0, 8.5]])
        )
        np.testing.assert_allclose(
            computed.clock,
            expected_clock if layout == "segments" else (np.r_[expected_clock[1], [np.nan, np.nan]] if case == "invalid" else expected_clock.ravel()),
            equal_nan=True,
        )
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(anchors, original)

@pytest.mark.parametrize("case", ["empty", "zero_batch", "nonempty"])
@pytest.mark.parametrize("name", ["value", "valid", "clock", "tag"])
@pytest.mark.parametrize("layout", ["segments", "stacked"])
@pytest.mark.parametrize("lazy", [False, True])
def test_tut_020_window_output_ownership(
    window_review_source, case, name, layout, lazy
):
    ao = window_review_source(lazy, 0 if case == "zero_batch" else None).rename(
        {"time": "clock"}
    )
    count = 0 if case == "empty" else 2
    anchor_values = np.array([1.0, 2.0])[:count]
    incoming = xr.DataArray(
        anchor_values, dims="anchor", coords={name: ("anchor", np.arange(count) + 100)}
    )
    if lazy:
        incoming = xr.DataArray(
            incoming.variable.chunk({"anchor": max(count, 1)}), coords=incoming.coords
        )
    before = ao.as_dataset()
    query_before = incoming.copy(deep=True)
    tasks = []
    if name == "value":
        with (
            Callback(pretask=lambda *args: tasks.append(args[0])),
            pytest.raises(ValueError, match="collid|conflict"),
        ):
            ao.events.around(
                incoming,
                opts=AroundOptions(
                    eval=ConditionEvalOptions(coord_name="clock"),
                    grid=np.array([-0.5, 0.0, 0.5]),
                    layout=layout,
                ),
            )
        assert not tasks
        xr.testing.assert_identical(ao.as_dataset(), before)
        xr.testing.assert_identical(incoming, query_before)
        return
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = ao.events.around(
            incoming,
            opts=AroundOptions(
                eval=ConditionEvalOptions(coord_name="clock"),
                grid=np.array([-0.5, 0.0, 0.5]),
                layout=layout,
            ),
        ).as_dataset()
    assert not tasks
    assert set(result.data_vars) == {"value"}
    assert result.clock.dims == result.value.dims
    assert result.valid.dims == result.value.dims
    if name == "valid":
        assert result.valid.dtype == np.dtype(bool)
    if name == "tag":
        assert "tag" in result.coords
    xr.testing.assert_identical(ao.as_dataset(), before)
    xr.testing.assert_identical(incoming, query_before)


@pytest.mark.parametrize("bad_row", [0, 1, 2])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("lazy", [False, True])
def test_tut_023_stacked_invalid_anchor_rows_are_left_packed(
    window_review_source, bad_row, width, lazy
):
    ao = window_review_source(lazy)
    anchors = np.array([0.5, 1.5, 2.5])
    anchors[bad_row] = np.nan
    incoming = xr.DataArray(
        da.from_array(anchors, chunks=1) if lazy else anchors,
        dims="anchor",
        coords={"anchor": ["a", "b", "c"]},
    )
    tau = np.array([0.0]) if width == 1 else np.array([-0.25, 0.0, 0.25])
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = ao.events.around(
            incoming, opts=AroundOptions(grid=tau, layout="stacked")
        ).as_dataset()
    assert not tasks
    computed = result.compute(scheduler="synchronous")
    size = 2 * width
    expected_rows = np.repeat(np.flatnonzero(np.isfinite(anchors)), width)
    expected_clock = (anchors[np.isfinite(anchors), None] + tau).ravel()
    np.testing.assert_array_equal(
        computed.window_event_index, np.r_[expected_rows, np.full(width, -1)]
    )
    np.testing.assert_allclose(
        computed.value, np.r_[expected_clock, np.full(width, np.nan)], equal_nan=True
    )
    np.testing.assert_allclose(
        computed.time, np.r_[expected_clock, np.full(width, np.nan)], equal_nan=True
    )
    assert computed.window_size.item() == size
    if not lazy:
        assert (
            computed.attrs["tal"]["core"]["validity"]["sequence_size_coord"]
            == "window_size"
        )
    else:
        assert "validity" not in computed.attrs["tal"]["core"]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_tut_020_empty_anchor_complete_batch_claims_are_checked(
    window_review_source, lazy, empty
):
    ao = window_review_source(lazy, 2, extra_index=True)
    incoming = xr.DataArray(
        np.empty((2, 0)) if empty else np.tile([1.0, 2.0], (2, 1)),
        dims=("trial", "anchor"),
        coords={"trial": [10, 11], "alias": ("trial", [101, 100])},
    ).set_xindex("alias")
    tasks = []
    with (
        Callback(pretask=lambda *args: tasks.append(args[0])),
        pytest.raises(ValueError, match="conflict|label|index|topology"),
    ):
        ao.events.around(incoming, opts=AroundOptions(grid=np.array([0.0])))
    assert not tasks


@pytest.mark.parametrize("name", ["valid", "clock", "tag"])
@pytest.mark.parametrize("layout", ["segments", "stacked"])
@pytest.mark.parametrize(
    "lazy_anchor,lazy_source", itertools.product([False, True], repeat=2)
)
def test_tut_024_lazy_anchor_generated_coordinate_precedence(
    window_review_source, name, layout, lazy_anchor, lazy_source
):
    ao = window_review_source(lazy_source).rename({"time": "clock"})
    anchors = np.array([1.0, 2.0])
    incoming = xr.DataArray(
        da.from_array(anchors, chunks=1) if lazy_anchor else anchors,
        dims="anchor",
        coords={name: ("anchor", [100, 200])},
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = ao.events.around(
            incoming,
            opts=AroundOptions(
                eval=ConditionEvalOptions(coord_name="clock"),
                grid=np.array([0.0]),
                layout=layout,
            ),
        ).as_dataset()
    assert not tasks
    expected = anchors[:, None] if layout == "segments" else anchors
    np.testing.assert_allclose(result.clock.compute(), expected)
    np.testing.assert_allclose(result.value.compute(), expected)
    assert result.valid.dtype == np.dtype(bool)
    if name == "tag":
        np.testing.assert_array_equal(result.tag.compute(), [100, 200])


@pytest.mark.parametrize("layout", ["segments", "stacked"])
@pytest.mark.parametrize("lazy_anchor", [False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_tut_024_window_valid_coordinate_stays_boolean(
    window_review_source, layout, lazy_anchor, width
):
    ao = window_review_source(False)
    anchors = np.array([1.0, np.nan])
    incoming = xr.DataArray(
        da.from_array(anchors, chunks=1) if lazy_anchor else anchors, dims="anchor"
    )
    tau = np.array([0.0]) if width == 1 else np.array([-0.25, 0.0, 0.25])
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = ao.events.around(
            incoming, opts=AroundOptions(grid=tau, layout=layout)
        ).as_dataset()
    assert not tasks
    assert result.valid.dtype == np.dtype(bool)
    expected = np.broadcast_to(np.array([True, False])[:, None], (2, width))
    if layout == "stacked":
        expected = expected.ravel()
    np.testing.assert_array_equal(result.valid.compute(), expected)


@pytest.mark.parametrize("layout", ["segments", "stacked"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("n", [0, 2])
def test_tut_020_empty_windows_protect_core_names(window_review_source, layout, lazy, n):
    """Protected core names reject before any sampling, including empty anchors."""
    base = window_review_source(lazy).as_dataset()
    base["value"] = base.value.expand_dims({"axis": ["x", "y"]}).transpose("sample", "axis")
    source = AnalysisObject.from_data(base.drop_attrs(), sequence_dim="sample", core_dims=("axis",), param_coord="time")
    anchors = xr.DataArray(np.array([1., 2.])[:n], dims="anchor", coords={"axis": ("anchor", np.arange(n))})
    before, original = source.as_dataset(), anchors.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])), pytest.raises(ValueError, match="events.around.*core"):
        source.events.around(anchors, opts=AroundOptions(grid=np.array([0.]), layout=layout))
    assert not tasks
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(anchors, original)


@pytest.mark.parametrize("layout", ["segments", "stacked"])
@pytest.mark.parametrize("lazy_part", ["anchor", "carrier", "both"])
@pytest.mark.parametrize("name", ["clock", "valid", "tag"])
def test_tut_024_independently_lazy_anchor_carriers(window_review_source, layout, lazy_part, name):
    """Generated precedence is established before numerical masking, without equality computation."""
    source = window_review_source(True).rename({"time": "clock"})
    anchors_data = np.array([1., np.nan, 2.])
    carrier_data = np.array([10., 20., 30.])
    anchors = xr.DataArray(
        da.from_array(anchors_data, chunks=1) if lazy_part != "carrier" else anchors_data,
        dims="anchor", coords={name: ("anchor", da.from_array(carrier_data, chunks=1) if lazy_part != "anchor" else carrier_data)},
    )
    before, original = source.as_dataset(), anchors.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = source.events.around(anchors, opts=AroundOptions(eval=ConditionEvalOptions(coord_name="clock"), grid=np.array([0.]), layout=layout)).as_dataset()
    assert not tasks
    computed = result.compute(scheduler="synchronous")
    expected = np.array([[1.], [np.nan], [2.]]) if layout == "segments" else [1., 2., np.nan]
    np.testing.assert_allclose(computed.clock, expected, equal_nan=True)
    assert computed.valid.dtype == np.dtype(bool)
    if name == "tag":
        np.testing.assert_allclose(computed.tag, carrier_data if layout == "segments" else [10., 30., np.nan], equal_nan=True)
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(anchors, original)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("case", ["mixed", "all_invalid", "empty", "zero_batch"])
def test_tut_023_per_batch_windows_keep_packed_prefix(window_review_source, lazy, case):
    """Independent row enumeration and a public mask consumer verify prefix truthfulness."""
    source = window_review_source(lazy, 0 if case == "zero_batch" else 2, extra_index=True)
    n = 0 if case == "empty" else 3
    values = np.array([[np.nan, .5, 2.5], [1.5, np.nan, 2.5]])[:, :n]
    if case == "all_invalid":
        values[:] = np.nan
    if case == "zero_batch":
        values = values[:0]
    source_ds = source.as_dataset()
    coordinates = xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(n, dim="anchor"))
    coordinates = coordinates.assign({"trial": source_ds.trial, "alias": source_ds.alias})
    anchors = xr.DataArray(da.from_array(values, chunks=1) if lazy else values, dims=("trial", "anchor"), coords=coordinates).set_xindex("alias")
    before, original = source.as_dataset(), anchors.copy(deep=True)
    tau = np.array([-.25, .25])
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        output = source.events.around(anchors, opts=AroundOptions(grid=tau, layout="stacked"))
        result = output.as_dataset()
        mask = output.events.mask(Condition.compare(Condition.var("value"), "ge", 0), opts=ConditionEvalOptions(coord_name="window_tau"))
    assert not tasks
    computed = result.compute(scheduler="synchronous")
    expected = np.full((values.shape[0], n * 2), np.nan)
    expected_rows = np.full((values.shape[0], n * 2), -1, dtype="int64")
    lengths = np.zeros(values.shape[0], dtype="int64")
    for lane, row in enumerate(values):
        rows = np.flatnonzero(np.isfinite(row))
        clocks = (row[rows, None] + tau).ravel()
        lengths[lane] = len(clocks)
        expected[lane, :len(clocks)] = clocks
        expected_rows[lane, :len(clocks)] = np.repeat(rows, 2)
    np.testing.assert_allclose(computed.value, expected, equal_nan=True)
    np.testing.assert_allclose(computed.time, expected, equal_nan=True)
    np.testing.assert_array_equal(computed.window_event_index, expected_rows)
    np.testing.assert_array_equal(computed.window_size, lengths)
    np.testing.assert_array_equal(mask.compute(), np.arange(n * 2)[None, :] < lengths[:, None])
    assert computed.valid.dtype == np.dtype(bool)
    assert computed.window_event_index.dtype == np.dtype("int64")
    assert computed.event_edge_code.dtype == np.dtype("int8")
    for name, index in source_ds.xindexes.items():
        if name != "sample":
            assert type(result.xindexes[name]) is type(index)
            assert result.xindexes[name].equals(index)
    assert set(result.data_vars) == {"value"}
    assert result.sizes[next(d for d in result.value.dims if d != "trial")] == n * 2
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(anchors, original)
