from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.event_ops import Condition, WhenOptions


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


def _valid_size_coord(ao: AnalysisObject) -> str:
    validity = ao.as_dataset(copy="none").attrs["tal"]["core"]["validity"]
    return str(validity["sequence_size_coord"])


def test_event_during_009_stream_layout_flattens_segment_sequence_packed_order() -> None:
    """ID: EVENT_DURING_009_stream_layout_flattens_segment_sequence_packed_order."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 2.0, 0.0], time=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream"))
    seq = _sequence_dim(out)
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["orig_index"].values,
        np.asarray([1, 2, 4], dtype="int64"),
    )
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["stream_segment_index"].values,
        np.asarray([0, 0, 1], dtype="int64"),
    )
    assert out.as_dataset(copy="none").sizes[seq] == 3


def test_event_during_010_stream_layout_parity_with_segments_values() -> None:
    """ID: EVENT_DURING_010_stream_layout_parity_with_segments_values."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 2.0, 0.0], time=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    segments = ao.events.when(cond, opts=WhenOptions(layout="segments"))
    stream = ao.events.when(cond, opts=WhenOptions(layout="stream"))
    seg_dim = tuple(segments.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["batch_dims"])[-1]
    flat = (
        segments.as_dataset(copy="none")["value"]
        .stack({"stream_tmp": (str(seg_dim), "sample")})
        .reset_index("stream_tmp", drop=True)
        .where(
            segments.as_dataset(copy="none")["orig_index"]
            .stack({"stream_tmp": (str(seg_dim), "sample")})
            .reset_index("stream_tmp", drop=True)
            >= 0
        )
        .dropna("stream_tmp")
    )
    np.testing.assert_allclose(
        stream.as_dataset(copy="none")["value"].values,
        np.asarray(flat.values, dtype="float64"),
        equal_nan=True,
    )


def test_event_during_011_stream_layout_inside_false_valid_complement_parity() -> None:
    """ID: EVENT_DURING_011_stream_layout_inside_false_valid_complement_parity."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[0.0, 2.0, 0.0, 3.0, 0.0, 9.0, 9.0]], dtype="float64"))},
        coords={
            "trial": np.asarray([0], dtype="int64"),
            "sample": np.arange(7, dtype="int64"),
            "time": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype="float64")),
            "group_size": ("trial", np.asarray([5], dtype="int64")),
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
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream", inside=False))
    seq = _sequence_dim(out)
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["orig_index"].isel(trial=0).values,
        np.asarray([0, 2, 4], dtype="int64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel(trial=0).values,
        np.asarray([0.0, 0.0, 0.0], dtype="float64"),
        equal_nan=True,
    )
    assert out.as_dataset(copy="none").sizes[seq] == 3


def test_event_during_012_stream_layout_attaches_deterministic_metadata_coords() -> None:
    """ID: EVENT_DURING_012_stream_layout_attaches_deterministic_metadata_coords."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream", max_segments=2))
    seq = _sequence_dim(out)
    size_name = _valid_size_coord(out)
    for name in (
        "orig_index",
        "stream_segment_index",
        "segment_start_time",
        "segment_end_time",
        "segment_start_index",
        "segment_end_index",
    ):
        assert name in out.as_dataset(copy="none").coords
        assert out.as_dataset(copy="none").coords[name].dims == (seq,)
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["orig_index"].values,
        np.asarray([1, -1, -1, -1, -1, -1], dtype="int64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none").coords["segment_start_time"].values,
        np.asarray([1.0, np.nan, np.nan, np.nan, np.nan, np.nan], dtype="float64"),
        equal_nan=True,
    )
    assert int(out.as_dataset(copy="none").coords[size_name].values) == 1


def test_event_during_hard_009_chunked_stream_without_max_segments_fails_fast() -> None:
    """ID: EVENT_DURING_HARD_009_chunked_stream_without_max_segments_fails_fast."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.when(cond, opts=WhenOptions(layout="stream"))
    assert "events.when: chunked stream extraction requires opts.max_segments" in str(err.value)


def test_event_during_hard_010_chunked_stream_with_explicit_max_segments_allowed() -> None:
    """ID: EVENT_DURING_HARD_010_chunked_stream_with_explicit_max_segments_allowed."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream", max_segments=2))
    assert out.as_dataset(copy="none")["value"].chunks is not None


def test_event_during_hard_011_grouped_empty_batch_stream_deterministic() -> None:
    """ID: EVENT_DURING_HARD_011_grouped_empty_batch_stream_deterministic."""
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
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream", on_empty="empty"))
    seq = _sequence_dim(out)
    assert out.as_dataset(copy="none").sizes["trial"] == 0
    assert out.as_dataset(copy="none").sizes[seq] == 0
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["trial"].values, np.asarray([], dtype="int64"))


def test_event_during_hard_012_stream_metadata_namespace_collision_failfast() -> None:
    """ID: EVENT_DURING_HARD_012_stream_metadata_namespace_collision_failfast."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray([0.0, 1.0, 0.0], dtype="float64"))},
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", np.asarray([0.0, 1.0, 2.0], dtype="float64")),
            "stream_segment_index": ("sample", np.asarray([0, 0, 0], dtype="int64")),
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
        ao.events.when(cond, opts=WhenOptions(layout="stream"))
    assert "events.when: stream metadata names conflict with dataset namespace" in str(err.value)


def test_event_during_hard_013_stream_no_hidden_eager_count_discovery() -> None:
    """ID: EVENT_DURING_HARD_013_stream_no_hidden_eager_count_discovery."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 1.0, 0.0], dtype="float64"), chunks=2))},
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0], dtype="float64"), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="time")
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream", max_segments=2))
    assert out.as_dataset(copy="none")["value"].chunks is not None


def test_event_during_hard_014_stream_on_empty_error_message_layout_specific() -> None:
    """ID: EVENT_DURING_HARD_014_stream_on_empty_error_message_layout_specific."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    with pytest.raises(ValueError) as err:
        ao.events.when(cond, opts=WhenOptions(layout="stream", on_empty="error"))
    text = str(err.value)
    assert "events.when: no selected stream samples for opts.layout='stream'" in text
    assert "opts.layout='segments'" not in text


def test_event_during_hard_015_stream_on_empty_error_chunked_layout_specific_failfast() -> None:
    """ID: EVENT_DURING_HARD_015_stream_on_empty_error_chunked_layout_specific_failfast."""
    ao = _chunked_ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    with pytest.raises(ValueError) as err:
        ao.events.when(
            cond,
            opts=WhenOptions(layout="stream", max_segments=2, on_empty="error"),
        )
    assert "events.when: opts.on_empty='error' requires unchunked when stream selection." in str(err.value)


def test_event_hard_017_during_stream_finalize_owner_preserves_param_and_validity() -> None:
    """ID: EVENT_HARD_017_during_stream_finalize_owner_preserves_param_and_validity."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream", max_segments=2))
    core = out.as_dataset(copy="none").attrs["tal"]["core"]
    assert core["param_coord"]["name"] == "time"
    validity = core["validity"]
    assert isinstance(validity, dict)
    assert str(validity["sequence_size_coord"]) in out.as_dataset(copy="none").coords


def test_event_hard_020_during_stream_bounded_indexer_blockwise_frontpack_parity() -> None:
    """ID: EVENT_HARD_020_during_stream_bounded_indexer_blockwise_frontpack_parity."""
    ao = _ao_series(values=[0.0, 1.0, 0.0, 2.0, 0.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream", max_segments=2))
    np.testing.assert_array_equal(
        out.as_dataset(copy="none").coords["orig_index"].values,
        np.asarray([1, 3, -1, -1, -1, -1, -1, -1, -1, -1], dtype="int64"),
    )
