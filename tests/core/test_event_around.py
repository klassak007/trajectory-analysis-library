from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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
