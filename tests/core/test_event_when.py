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


def _segment_dim(ao: AnalysisObject) -> str:
    roles = ao.unsafe_data.attrs["tal"]["core"]["roles"]
    batch_dims = tuple(roles["batch_dims"])
    assert len(batch_dims) >= 1
    return str(batch_dims[-1])


def test_event_during_001_mask_layout_preserves_shape_and_roles() -> None:
    """ID: EVENT_DURING_001_mask_layout_preserves_shape_and_roles."""
    ao = _ao_series(values=[0.0, 1.0, 2.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="mask"))
    assert dict(out.unsafe_data.sizes) == dict(ao.unsafe_data.sizes)
    assert out.unsafe_data.attrs["tal"]["core"]["roles"] == ao.unsafe_data.attrs["tal"]["core"]["roles"]
    np.testing.assert_allclose(out.unsafe_data["time"].values, ao.unsafe_data["time"].values)
    np.testing.assert_allclose(
        out.unsafe_data["value"].values,
        np.asarray([np.nan, 1.0, 2.0, np.nan], dtype="float64"),
        equal_nan=True,
    )


def test_event_during_002_inside_false_selects_valid_complement_only() -> None:
    """ID: EVENT_DURING_002_inside_false_selects_valid_complement_only."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0, 100.0]], dtype="float64"))},
        coords={
            "trial": np.asarray([0], dtype="int64"),
            "sample": np.arange(4, dtype="int64"),
            "time": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype="float64")),
            "group_size": ("trial", np.asarray([3], dtype="int64")),
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
    out = ao.events.when(cond, opts=WhenOptions(inside=False))
    np.testing.assert_allclose(
        out.unsafe_data["value"].values,
        np.asarray([[0.0, np.nan, np.nan, np.nan]], dtype="float64"),
        equal_nan=True,
    )
    assert np.isnan(float(out.unsafe_data["value"].values[0, 3]))


def test_event_during_003_on_empty_empty_and_error_contract() -> None:
    """ID: EVENT_DURING_003_on_empty_empty_and_error_contract."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    out = ao.events.when(cond, opts=WhenOptions(on_empty="empty"))
    np.testing.assert_allclose(
        out.unsafe_data["value"].values,
        np.asarray([np.nan, np.nan, np.nan], dtype="float64"),
        equal_nan=True,
    )
    with pytest.raises(ValueError) as err:
        ao.events.when(cond, opts=WhenOptions(on_empty="error"))
    assert "events.when: no selected samples" in str(err.value)


def test_event_during_hard_001_layout_stream_runtime_enabled() -> None:
    """ID: EVENT_DURING_HARD_001_layout_stream_runtime_enabled."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="stream"))
    roles = out.unsafe_data.attrs["tal"]["core"]["roles"]
    seq_dim = str(roles["sequence_dim"])
    assert seq_dim in out.unsafe_data.dims
    np.testing.assert_allclose(
        out.unsafe_data["value"].values,
        np.asarray([1.0], dtype="float64"),
        equal_nan=True,
    )


def test_event_during_hard_002_chunked_on_empty_error_fails_fast() -> None:
    """ID: EVENT_DURING_HARD_002_chunked_on_empty_error_fails_fast."""
    ao = _chunked_ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    with pytest.raises(ValueError) as err:
        ao.events.when(cond, opts=WhenOptions(on_empty="error"))
    assert "events.when: opts.on_empty='error' requires unchunked when selection." in str(err.value)


def test_event_during_hard_003_grouped_empty_batch_during_mask_is_deterministic() -> None:
    """ID: EVENT_DURING_HARD_003_grouped_empty_batch_during_mask_is_deterministic."""
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
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(on_empty="empty"))
    assert out.unsafe_data.sizes["trial"] == 0
    assert out.unsafe_data.sizes["sample"] == 3
    np.testing.assert_array_equal(out.unsafe_data.coords["trial"].values, np.asarray([], dtype="int64"))


def test_event_during_004_segments_layout_extracts_contiguous_runs() -> None:
    """ID: EVENT_DURING_004_segments_layout_extracts_contiguous_runs."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 2.0, 0.0], time=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments"))
    segment_dim = _segment_dim(out)
    assert out.unsafe_data.sizes[segment_dim] == 2
    np.testing.assert_array_equal(
        out.unsafe_data["segment_start_index"].values,
        np.asarray([1, 4], dtype="int64"),
    )
    np.testing.assert_array_equal(
        out.unsafe_data["segment_end_index"].values,
        np.asarray([2, 4], dtype="int64"),
    )
    np.testing.assert_allclose(
        out.unsafe_data["segment_start_time"].values,
        np.asarray([1.0, 4.0], dtype="float64"),
    )
    np.testing.assert_allclose(
        out.unsafe_data["segment_end_time"].values,
        np.asarray([2.0, 4.0], dtype="float64"),
    )
    np.testing.assert_allclose(
        out.unsafe_data["value"].isel({segment_dim: 0}).values,
        np.asarray([1.0, 1.0, np.nan, np.nan, np.nan, np.nan], dtype="float64"),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out.unsafe_data["value"].isel({segment_dim: 1}).values,
        np.asarray([2.0, np.nan, np.nan, np.nan, np.nan, np.nan], dtype="float64"),
        equal_nan=True,
    )


def test_event_during_005_segments_layout_inside_false_valid_complement_runs() -> None:
    """ID: EVENT_DURING_005_segments_layout_inside_false_valid_complement_runs."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0, 100.0]], dtype="float64"))},
        coords={
            "trial": np.asarray([0], dtype="int64"),
            "sample": np.arange(4, dtype="int64"),
            "time": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype="float64")),
            "group_size": ("trial", np.asarray([3], dtype="int64")),
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
    out = ao.events.when(cond, opts=WhenOptions(layout="segments", inside=False))
    segment_dim = _segment_dim(out)
    assert out.unsafe_data.sizes["trial"] == 1
    assert out.unsafe_data.sizes[segment_dim] == 1
    np.testing.assert_allclose(
        out.unsafe_data["value"].isel(trial=0, **{segment_dim: 0}).values,
        np.asarray([0.0, np.nan, np.nan, np.nan], dtype="float64"),
        equal_nan=True,
    )
    assert np.isnan(float(out.unsafe_data["value"].isel(trial=0, **{segment_dim: 0}).values[3]))


def test_event_during_006_segments_layout_attaches_deterministic_metadata_coords() -> None:
    """ID: EVENT_DURING_006_segments_layout_attaches_deterministic_metadata_coords."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments"))
    segment_dim = _segment_dim(out)
    for name in (
        "segment_start_time",
        "segment_end_time",
        "segment_start_index",
        "segment_end_index",
    ):
        assert name in out.unsafe_data.coords
        assert out.unsafe_data.coords[name].dims == (segment_dim,)
    assert "orig_index" in out.unsafe_data.coords
    assert out.unsafe_data.coords["orig_index"].dims == (segment_dim, "sample")
    np.testing.assert_array_equal(
        out.unsafe_data["orig_index"].isel({segment_dim: 0}).values,
        np.asarray([1, 2, -1, -1], dtype="int64"),
    )


def test_event_during_007_segments_layout_on_empty_contract() -> None:
    """ID: EVENT_DURING_007_segments_layout_on_empty_contract."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments", on_empty="empty"))
    segment_dim = _segment_dim(out)
    assert out.unsafe_data.sizes[segment_dim] == 0
    with pytest.raises(ValueError) as err:
        ao.events.when(cond, opts=WhenOptions(layout="segments", on_empty="error"))
    assert "events.when: no selected segments" in str(err.value)


def test_event_during_hard_004_segments_layout_stable_with_stream_enabled() -> None:
    """ID: EVENT_DURING_HARD_004_segments_layout_stable_with_stream_enabled."""
    ao = _ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    seg = ao.events.when(cond, opts=WhenOptions(layout="segments"))
    stream = ao.events.when(cond, opts=WhenOptions(layout="stream"))
    assert _segment_dim(seg)
    roles = stream.unsafe_data.attrs["tal"]["core"]["roles"]
    assert str(roles["sequence_dim"]) in stream.unsafe_data.dims


def test_event_during_hard_005_chunked_segments_without_max_segments_fails_fast() -> None:
    """ID: EVENT_DURING_HARD_005_chunked_segments_without_max_segments_fails_fast."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    with pytest.raises(ValueError) as err:
        ao.events.when(cond, opts=WhenOptions(layout="segments"))
    assert "chunked interval extraction requires opts.max_segments" in str(err.value)


def test_event_during_hard_006_chunked_segments_with_explicit_max_segments_allowed() -> None:
    """ID: EVENT_DURING_HARD_006_chunked_segments_with_explicit_max_segments_allowed."""
    ao = _chunked_ao_series(values=[0.0, 1.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments", max_segments=2))
    segment_dim = _segment_dim(out)
    assert out.unsafe_data.sizes[segment_dim] == 2
    assert hasattr(out.unsafe_data["value"].data, "chunks")


def test_event_during_hard_007_grouped_empty_batch_segments_deterministic() -> None:
    """ID: EVENT_DURING_HARD_007_grouped_empty_batch_segments_deterministic."""
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
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments", on_empty="empty"))
    segment_dim = _segment_dim(out)
    assert out.unsafe_data.sizes["trial"] == 0
    assert out.unsafe_data.sizes[segment_dim] == 0
    np.testing.assert_array_equal(out.unsafe_data.coords["trial"].values, np.asarray([], dtype="int64"))


def test_event_during_hard_008_segment_metadata_namespace_collision_failfast() -> None:
    """ID: EVENT_DURING_HARD_008_segment_metadata_namespace_collision_failfast."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray([0.0, 1.0, 0.0], dtype="float64"))},
        coords={
            "sample": np.arange(3, dtype="int64"),
            "time": ("sample", np.asarray([0.0, 1.0, 2.0], dtype="float64")),
            "segment_start_time": ("sample", np.asarray([0.0, 0.0, 0.0], dtype="float64")),
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
        ao.events.when(cond, opts=WhenOptions(layout="segments"))
    assert "events.when: segment metadata names conflict with dataset namespace" in str(err.value)


def test_event_during_008_inside_complement_semantics_parity_mask_vs_segments() -> None:
    """ID: EVENT_DURING_008_inside_complement_semantics_parity_mask_vs_segments."""
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
    mask_out = ao.events.when(cond, opts=WhenOptions(layout="mask", inside=False))
    seg_out = ao.events.when(cond, opts=WhenOptions(layout="segments", inside=False))
    segment_dim = _segment_dim(seg_out)
    np.testing.assert_array_equal(
        seg_out.unsafe_data["segment_start_index"].isel(trial=0).values,
        np.asarray([0, 2, 4], dtype="int64"),
    )
    np.testing.assert_array_equal(
        seg_out.unsafe_data["segment_end_index"].isel(trial=0).values,
        np.asarray([0, 2, 4], dtype="int64"),
    )
    expected_mask = np.asarray([[0.0, np.nan, 0.0, np.nan, 0.0, np.nan, np.nan]], dtype="float64")
    np.testing.assert_allclose(mask_out.unsafe_data["value"].values, expected_mask, equal_nan=True)


def test_event_hard_016_during_segments_finalize_owner_preserves_param_and_validity() -> None:
    """ID: EVENT_HARD_016_during_segments_finalize_owner_preserves_param_and_validity."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments"))
    core = out.unsafe_data.attrs["tal"]["core"]
    assert core["param_coord"]["name"] == "time"
    validity = core["validity"]
    assert isinstance(validity, dict)
    assert str(validity["sequence_size_coord"]) in out.unsafe_data.coords


def test_event_hard_021_during_segments_bounded_orig_index_blockwise_parity() -> None:
    """ID: EVENT_HARD_021_during_segments_bounded_orig_index_blockwise_parity."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 0.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments", max_segments=3))
    segment_dim = _segment_dim(out)
    np.testing.assert_array_equal(
        out.unsafe_data["orig_index"].isel({segment_dim: 0}).values,
        np.asarray([1, 2, -1, -1, -1], dtype="int64"),
    )
    np.testing.assert_array_equal(
        out.unsafe_data["orig_index"].isel({segment_dim: 1}).values,
        np.asarray([-1, -1, -1, -1, -1], dtype="int64"),
    )
