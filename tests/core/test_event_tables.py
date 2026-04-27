from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.event_ops import Condition, EventExtractOptions
import tal.core.event_ops.boundary as boundary_mod


def _ao_series(*, values: list[float], time: list[float], event_coord: bool = False) -> AnalysisObject:
    coords: dict[str, object] = {
        "sample": np.arange(len(values), dtype="int64"),
        "time": ("sample", np.asarray(time, dtype="float64")),
    }
    if event_coord:
        coords["event"] = ("sample", np.arange(len(values), dtype="int64"))
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray(values, dtype="float64"))},
        coords=coords,
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )


def _event_dim(ds: xr.Dataset, *, batch_dims: tuple[str, ...] = ()) -> str:
    dims = [dim for dim in ds.dims if dim not in batch_dims]
    assert len(dims) == 1
    return dims[0]


def test_event_table_001_boundary_driven_enter_exit_basic() -> None:
    """ID: EVENT_TABLE_001_boundary_driven_enter_exit_basic."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 1.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    out = ao.events.events(Condition.compare(Condition.var("value"), "gt", 0.5))
    event_dim = _event_dim(out)
    np.testing.assert_array_equal(out["edge_code"].values, np.asarray([1, 2, 1, 2], dtype="int8"))
    np.testing.assert_allclose(out["time"].values, np.asarray([1.0, 2.0, 4.0, 4.0], dtype="float64"))
    np.testing.assert_array_equal(out["sample_index_before"].values, np.asarray([0, 2, 3, 4], dtype="int64"))
    np.testing.assert_array_equal(out["sample_index_after"].values, np.asarray([1, 3, 4, -1], dtype="int64"))
    assert out.sizes[event_dim] == 4


def test_event_table_002_include_initial_emits_initial_enter() -> None:
    """ID: EVENT_TABLE_002_include_initial_emits_initial_enter."""
    ao = _ao_series(values=[1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    out = ao.events.events(
        Condition.compare(Condition.var("value"), "gt", 0.5),
        opts=EventExtractOptions(include_initial=True),
    )
    np.testing.assert_array_equal(out["edge_code"].values, np.asarray([1, 2], dtype="int8"))
    np.testing.assert_allclose(out["time"].values, np.asarray([0.0, 1.0], dtype="float64"))
    np.testing.assert_array_equal(out["sample_index_before"].values, np.asarray([-1, 1], dtype="int64"))
    np.testing.assert_array_equal(out["sample_index_after"].values, np.asarray([0, 2], dtype="int64"))


def test_event_table_003_trigger_path_dedup_sorted_left_packed() -> None:
    """ID: EVENT_TABLE_003_trigger_path_dedup_sorted_left_packed."""
    ao = _ao_series(values=[1.0, 1.0, 0.0], time=[0.0, 1e-13, 1.0])
    out = ao.events.events(
        Condition.compare(Condition.var("value"), "eq", 1.0),
        opts=EventExtractOptions(dedupe_atol=1e-12),
    )
    np.testing.assert_array_equal(out["edge_code"].values, np.asarray([3, 2], dtype="int8"))
    np.testing.assert_allclose(out["time"].values, np.asarray([0.0, 1e-13], dtype="float64"))
    np.testing.assert_array_equal(out["sample_index_before"].values, np.asarray([0, 1], dtype="int64"))
    np.testing.assert_array_equal(out["sample_index_after"].values, np.asarray([0, 2], dtype="int64"))


def test_event_hard_001_dim_name_collisions_allocated_safely() -> None:
    """ID: EVENT_HARD_001_dim_name_collisions_allocated_safely."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0], event_coord=True)
    out = ao.events.events(Condition.compare(Condition.var("value"), "gt", 0.5))
    event_dim = _event_dim(out)
    assert event_dim != "event"
    assert event_dim.startswith("event")


def test_event_hard_002_chunked_without_max_events_fails_fast() -> None:
    """ID: EVENT_HARD_002_chunked_without_max_events_fails_fast."""
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
        ao.events.events(Condition.compare(Condition.var("value"), "gt", 0.5))
    assert "chunked event extraction requires opts.max_events" in str(err.value)


def test_event_hard_003_chunked_with_explicit_max_events_allowed() -> None:
    """ID: EVENT_HARD_003_chunked_with_explicit_max_events_allowed."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 1.0, 0.0]), chunks=2))},
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="time")
    out = ao.events.events(
        Condition.compare(Condition.var("value"), "gt", 0.5),
        opts=EventExtractOptions(max_events=3),
    )
    event_dim = _event_dim(out)
    assert out.sizes[event_dim] == 3
    assert hasattr(out["time"].data, "chunks")


def test_event_hard_004_no_hidden_eager_count_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: EVENT_HARD_004_no_hidden_eager_count_discovery."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 1.0, 0.0]), chunks=2))},
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="time")

    def _fail_compute(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected eager compute")

    monkeypatch.setattr(da.Array, "compute", _fail_compute, raising=True)
    with pytest.raises(ValueError) as err:
        ao.events.events(Condition.compare(Condition.var("value"), "gt", 0.5))
    assert "chunked event extraction requires opts.max_events" in str(err.value)


def test_event_hard_005_grouped_label_alignment_not_positional() -> None:
    """ID: EVENT_HARD_005_grouped_label_alignment_not_positional."""
    values = np.asarray([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype="float64")
    time = np.broadcast_to(np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64"), values.shape)
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), values)},
        coords={
            "trial": np.asarray(["b", "a"], dtype=object),
            "sample": np.arange(4, dtype="int64"),
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
    out = ao.events.events(Condition.compare(Condition.var("value"), "gt", 0.5))
    event_dim = _event_dim(out, batch_dims=("trial",))
    assert list(out.coords["trial"].values) == ["b", "a"]
    np.testing.assert_array_equal(out["edge_code"].sel(trial="b").values, np.asarray([1, 2], dtype="int8"))
    np.testing.assert_array_equal(out["sample_index_before"].sel(trial="a").values, np.asarray([1, -1], dtype="int64"))
    np.testing.assert_array_equal(out["sample_index_after"].sel(trial="a").values, np.asarray([2, -1], dtype="int64"))
    assert out.sizes[event_dim] == 2


def test_event_hard_012_grouped_empty_batch_dynamic_returns_empty_table() -> None:
    """ID: EVENT_HARD_012_grouped_empty_batch_dynamic_returns_empty_table."""
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
    out = ao.events.events(Condition.compare(Condition.var("value"), "gt", 0.5))
    event_dim = _event_dim(out, batch_dims=("trial",))
    assert out.sizes["trial"] == 0
    assert out.sizes[event_dim] == 0
    np.testing.assert_array_equal(out.coords["trial"].values, np.asarray([], dtype="int64"))


def test_event_hard_013_event_extract_eval_type_validation_tal_owned() -> None:
    """ID: EVENT_HARD_013_event_extract_eval_type_validation_tal_owned."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    with pytest.raises(TypeError) as err:
        ao.events.events(
            Condition.compare(Condition.var("value"), "gt", 0.5),
            opts=EventExtractOptions(eval="bad"),  # type: ignore[arg-type]
        )
    assert "events.events: opts.eval must be ConditionEvalOptions" in str(err.value)


@pytest.mark.parametrize("invalid", ["x", None])
def test_event_hard_015_event_dedupe_tolerance_invalid_raises_tal_valueerror(invalid: object) -> None:
    """ID: EVENT_HARD_015_event_dedupe_tolerance_invalid_raises_tal_valueerror."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError) as err:
        ao.events.events(
            Condition.compare(Condition.var("value"), "gt", 0.5),
            opts=EventExtractOptions(dedupe_atol=invalid),  # type: ignore[arg-type]
        )
    assert "events.events: opts.dedupe_atol must be a numeric scalar" in str(err.value)


def test_event_hard_022_boundary_bounded_stopgap_backend_interface_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: EVENT_HARD_022_boundary_bounded_stopgap_backend_interface_lock."""
    calls = {"n": 0}
    original = boundary_mod.boundary_bounded_row_backend

    def _count(*args: object, **kwargs: object):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(boundary_mod, "boundary_bounded_row_backend", _count)
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    _ = ao.events.events(
        Condition.compare(Condition.var("value"), "gt", 0.5),
        opts=EventExtractOptions(max_events=3),
    )
    assert calls["n"] >= 1
