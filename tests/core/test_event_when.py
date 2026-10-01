from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisObject
from tal.core.event_ops import Condition, ConditionEvalOptions, WhenOptions


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
    roles = ao.as_dataset(copy="none").attrs["tal"]["core"]["roles"]
    batch_dims = tuple(roles["batch_dims"])
    assert len(batch_dims) >= 1
    return str(batch_dims[-1])


def test_event_during_001_mask_layout_preserves_shape_and_roles() -> None:
    """ID: EVENT_DURING_001_mask_layout_preserves_shape_and_roles."""
    ao = _ao_series(values=[0.0, 1.0, 2.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="mask"))
    assert dict(out.as_dataset(copy="none").sizes) == dict(ao.as_dataset(copy="none").sizes)
    assert out.as_dataset(copy="none").attrs["tal"]["core"]["roles"] == ao.as_dataset(copy="none").attrs["tal"]["core"]["roles"]
    np.testing.assert_allclose(out.as_dataset(copy="none")["time"].values, ao.as_dataset(copy="none")["time"].values)
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].values,
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
        out.as_dataset(copy="none")["value"].values,
        np.asarray([[0.0, np.nan, np.nan, np.nan]], dtype="float64"),
        equal_nan=True,
    )
    assert np.isnan(float(out.as_dataset(copy="none")["value"].values[0, 3]))


def test_event_during_003_on_empty_empty_and_error_contract() -> None:
    """ID: EVENT_DURING_003_on_empty_empty_and_error_contract."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    out = ao.events.when(cond, opts=WhenOptions(on_empty="empty"))
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].values,
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
    roles = out.as_dataset(copy="none").attrs["tal"]["core"]["roles"]
    seq_dim = str(roles["sequence_dim"])
    assert seq_dim in out.as_dataset(copy="none").dims
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].values,
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
    assert out.as_dataset(copy="none").sizes["trial"] == 0
    assert out.as_dataset(copy="none").sizes["sample"] == 3
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["trial"].values, np.asarray([], dtype="int64"))


def test_event_during_004_segments_layout_extracts_contiguous_runs() -> None:
    """ID: EVENT_DURING_004_segments_layout_extracts_contiguous_runs."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 2.0, 0.0], time=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments"))
    segment_dim = _segment_dim(out)
    assert out.as_dataset(copy="none").sizes[segment_dim] == 2
    np.testing.assert_array_equal(
        out.as_dataset(copy="none")["segment_start_index"].values,
        np.asarray([1, 4], dtype="int64"),
    )
    np.testing.assert_array_equal(
        out.as_dataset(copy="none")["segment_end_index"].values,
        np.asarray([2, 4], dtype="int64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["segment_start_time"].values,
        np.asarray([1.0, 4.0], dtype="float64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["segment_end_time"].values,
        np.asarray([2.0, 4.0], dtype="float64"),
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel({segment_dim: 0}).values,
        np.asarray([1.0, 1.0, np.nan, np.nan, np.nan, np.nan], dtype="float64"),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel({segment_dim: 1}).values,
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
    assert out.as_dataset(copy="none").sizes["trial"] == 1
    assert out.as_dataset(copy="none").sizes[segment_dim] == 1
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["value"].isel(trial=0, **{segment_dim: 0}).values,
        np.asarray([0.0, np.nan, np.nan, np.nan], dtype="float64"),
        equal_nan=True,
    )
    assert np.isnan(float(out.as_dataset(copy="none")["value"].isel(trial=0, **{segment_dim: 0}).values[3]))


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
        assert name in out.as_dataset(copy="none").coords
        assert out.as_dataset(copy="none").coords[name].dims == (segment_dim,)
    assert "orig_index" in out.as_dataset(copy="none").coords
    assert out.as_dataset(copy="none").coords["orig_index"].dims == (segment_dim, "sample")
    np.testing.assert_array_equal(
        out.as_dataset(copy="none")["orig_index"].isel({segment_dim: 0}).values,
        np.asarray([1, 2, -1, -1], dtype="int64"),
    )


def test_event_during_007_segments_layout_on_empty_contract() -> None:
    """ID: EVENT_DURING_007_segments_layout_on_empty_contract."""
    ao = _ao_series(values=[0.0, 0.0, 0.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "gt", 1.0)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments", on_empty="empty"))
    segment_dim = _segment_dim(out)
    assert out.as_dataset(copy="none").sizes[segment_dim] == 0
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
    roles = stream.as_dataset(copy="none").attrs["tal"]["core"]["roles"]
    assert str(roles["sequence_dim"]) in stream.as_dataset(copy="none").dims


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
    assert out.as_dataset(copy="none").sizes[segment_dim] == 2
    assert hasattr(out.as_dataset(copy="none")["value"].data, "chunks")


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
    assert out.as_dataset(copy="none").sizes["trial"] == 0
    assert out.as_dataset(copy="none").sizes[segment_dim] == 0
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["trial"].values, np.asarray([], dtype="int64"))


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
    np.testing.assert_array_equal(
        seg_out.as_dataset(copy="none")["segment_start_index"].isel(trial=0).values,
        np.asarray([0, 2, 4], dtype="int64"),
    )
    np.testing.assert_array_equal(
        seg_out.as_dataset(copy="none")["segment_end_index"].isel(trial=0).values,
        np.asarray([0, 2, 4], dtype="int64"),
    )
    expected_mask = np.asarray([[0.0, np.nan, 0.0, np.nan, 0.0, np.nan, np.nan]], dtype="float64")
    np.testing.assert_allclose(mask_out.as_dataset(copy="none")["value"].values, expected_mask, equal_nan=True)


def test_event_hard_016_during_segments_finalize_owner_preserves_param_and_validity() -> None:
    """ID: EVENT_HARD_016_during_segments_finalize_owner_preserves_param_and_validity."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0], time=[0.0, 1.0, 2.0, 3.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments"))
    core = out.as_dataset(copy="none").attrs["tal"]["core"]
    assert core["param_coord"]["name"] == "time"
    validity = core["validity"]
    assert isinstance(validity, dict)
    assert str(validity["sequence_size_coord"]) in out.as_dataset(copy="none").coords


def test_event_hard_021_during_segments_bounded_orig_index_blockwise_parity() -> None:
    """ID: EVENT_HARD_021_during_segments_bounded_orig_index_blockwise_parity."""
    ao = _ao_series(values=[0.0, 1.0, 1.0, 0.0, 0.0], time=[0.0, 1.0, 2.0, 3.0, 4.0])
    cond = Condition.compare(Condition.var("value"), "gt", 0.5)
    out = ao.events.when(cond, opts=WhenOptions(layout="segments", max_segments=3))
    segment_dim = _segment_dim(out)
    np.testing.assert_array_equal(
        out.as_dataset(copy="none")["orig_index"].isel({segment_dim: 0}).values,
        np.asarray([1, 2, -1, -1, -1], dtype="int64"),
    )
    np.testing.assert_array_equal(
        out.as_dataset(copy="none")["orig_index"].isel({segment_dim: 1}).values,
        np.asarray([-1, -1, -1, -1, -1], dtype="int64"),
    )


@pytest.mark.parametrize("layout", ("segments", "stream"))
@pytest.mark.parametrize("case,lazy,validate", [
    ("shared", False, True), ("shared", True, True),
    ("shared", False, False), ("shared", True, False),
    ("ragged", False, True), ("ragged", True, False),
    ("clock_lazy", True, True), ("native", False, True), ("native", True, False),
    ("truncated", False, True), ("truncated", True, False),
    ("complement", False, True), ("complement", True, False),
    ("none", False, True), ("none", True, False),
    ("empty_sequence", False, True), ("empty_sequence", True, False),
    ("zero_batch", False, True), ("zero_batch", True, False),
    ("unbounded", False, True),
])
def test_tut_003_packed_parameter_values_follow_observations(layout, case, lazy, validate):
    """TUT-003 / Contracts 026/029: gather absolute clocks and keep bounded padding truthful."""
    from dask.callbacks import Callback

    from tal.core.schema_read import read_roles, read_sequence_size_coord_name

    times = np.array([0., .2, .7, 1.5, 2., 3., 4., 5.])
    values = np.array([0., 3., 4., 0., 0., 3., 4., 0.])
    dims, batches, coords = ("sample",), (), {"sample": np.arange(8), "time": ("sample", times)}
    groups = [[[1, 2], [5, 6]]]
    limit, inside = 2, True
    if case == "ragged":
        dims, batches = ("trial", "sample"), ("trial",)
        values = np.stack([values, values])
        coords.update(trial=["a", "b"], time=(("trial", "sample"), np.stack([times, times + 10.])),
                      length=("trial", [6, 3]))
        groups = [[[1, 2], [5]], [[1, 2]]]
    elif case == "truncated":
        limit, groups = 1, [[[1, 2]]]
    elif case == "complement":
        limit, inside, groups = 3, False, [[[0], [3, 4], [7]]]
    elif case == "none":
        values, groups = np.zeros(8), [[]]
    elif case == "empty_sequence":
        values, times, groups = np.empty(0), np.empty(0), [[]]
        coords = {"sample": np.empty(0, dtype=int), "time": ("sample", times)}
    elif case == "zero_batch":
        dims, batches, values, groups = ("trial", "sample"), ("trial",), np.empty((0, 8)), []
        coords.update(trial=np.empty(0, dtype=str), time=(("trial", "sample"), np.empty((0, 8))))
    elif case == "unbounded":
        limit = None
    raw = xr.Dataset({"value": (dims, values)}, coords=coords)
    raw = raw.assign_coords(tag=("sample", np.arange(len(times), dtype=float) + 100.), station="lab")
    raw["time"].attrs["units"] = "fixture convention"
    if case == "native":
        raw = raw.assign_coords(xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(10, 90, 10, dim="sample")))
    if lazy:
        if case == "clock_lazy":
            raw = raw.assign_coords(time=raw["time"].chunk({"sample": 2}))
        else:
            raw = raw.chunk({"sample": 2})
            if case == "ragged":
                raw = raw.assign_coords(length=coords["length"])
    source = AnalysisObject.from_data(raw, sequence_dim="sample", batch_dims=batches, param_coord="time",
                                      sequence_size_coord="length" if case == "ragged" else None, validate=validate)
    snapshot = source.as_dataset(copy="deep")
    condition = Condition.compare(Condition.var("value"), "gt", 2.)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.events.when(condition, opts=WhenOptions(layout=layout, max_segments=limit, inside=inside))
    assert tasks == []
    ds = result.as_dataset().compute(scheduler="synchronous")
    _, sequence, output_batches, cores = read_roles(ds)
    size_name = read_sequence_size_coord_name(ds)
    count_name = "length" if layout == "segments" and case == "ragged" else "segment_size" if layout == "segments" else "stream_size"
    assert size_name in (None, count_name)
    if size_name is None:
        assert result.as_dataset()[count_name].chunks is not None
    n = len(times)
    max_groups = limit if limit is not None else max((len(g) for g in groups), default=0)
    if layout == "segments":
        shape = ((len(groups),) if batches else ()) + (max_groups, n)
        assert output_batches[:-1] == batches and sequence == "sample"
        expected_index = np.full(shape, -1, dtype=int)
        lanes = expected_index if batches else expected_index[None, ...]
        for lane, episodes in enumerate(groups):
            for segment, indices in enumerate(episodes):
                lanes[lane, segment, :len(indices)] = indices
        expected_size = (expected_index >= 0).sum(axis=-1)
        assert ds.sizes[output_batches[-1]] == max_groups
        assert ds.xindexes["sample"].equals(snapshot.xindexes["sample"])
        if case == "native":
            assert isinstance(ds.xindexes["sample"], xr.indexes.RangeIndex)
    else:
        width = limit * n if limit is not None else max((sum(map(len, g)) for g in groups), default=0)
        shape = ((len(groups),) if batches else ()) + (width,)
        expected_index = np.full(shape, -1, dtype=int)
        lanes = expected_index if batches else expected_index[None, ...]
        for lane, episodes in enumerate(groups):
            indices = [i for segment in episodes for i in segment]
            lanes[lane, :len(indices)] = indices
        expected_size = (expected_index >= 0).sum(axis=-1)
        assert output_batches == batches and ds.sizes[sequence] == width
    np.testing.assert_array_equal(ds.orig_index, expected_index)
    np.testing.assert_array_equal(ds[count_name], expected_size)
    expected_clock = np.full(shape, np.nan)
    expected_values = np.full(shape, np.nan)
    expected_tags = np.full(shape, np.nan)
    clock_lanes = expected_clock if batches else expected_clock[None, ...]
    value_lanes = expected_values if batches else expected_values[None, ...]
    tag_lanes = expected_tags if batches else expected_tags[None, ...]
    for lane, index_lane in enumerate(lanes):
        keep = index_lane >= 0
        clock_lanes[lane][keep] = times[index_lane[keep]] + (10. * lane if case == "ragged" else 0.)
        value_lanes[lane][keep] = (values[lane] if batches else values)[index_lane[keep]]
        tag_lanes[lane][keep] = 100. + index_lane[keep]
    np.testing.assert_allclose(ds.value, expected_values)
    np.testing.assert_allclose(ds.time, expected_clock)
    np.testing.assert_allclose(ds.tag, expected_tags)
    assert ds.time.dims == ds.value.dims == ds.orig_index.dims
    assert cores == () and set(ds.data_vars) == {"value"}
    assert ds.station == "lab"
    if batches:
        np.testing.assert_array_equal(ds.trial, snapshot.trial)
    if layout == "stream":
        np.testing.assert_array_equal(ds.stream_segment_index < 0, expected_index < 0)
        assert np.isnan(ds.segment_start_time.where(ds.orig_index < 0)).all()
    xr.testing.assert_identical(source.as_dataset(), snapshot)


_TUT_AUDIT_CLOCK=np.array([0.,.2,.7,1.5,2.,3.,4.,5.])
_TUT_AUDIT_VALUES=np.array([0.,3.,4.,0.,0.,3.,4.,0.])

def _tut_audit_ao(ds,*,batch=(),sequence='sample',param='time',lazy=False):
    if lazy: ds=ds.chunk({sequence:2})
    return AnalysisObject.from_data(ds,sequence_dim=sequence,batch_dims=batch,param_coord=param)


def _tut_audit_series(*,axis_clock=False,n=8,range_index=False,lazy=False):
    coords={'sample':_TUT_AUDIT_CLOCK[:n] if axis_clock else np.arange(n)*10+10}
    if not axis_clock:coords['time']=('sample',_TUT_AUDIT_CLOCK[:n])
    ds=xr.Dataset({'value':('sample',_TUT_AUDIT_VALUES[:n])},coords=coords)
    if range_index:ds=ds.assign_coords(xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(10,10+10*n,10,dim='sample')))
    return _tut_audit_ao(ds,param='sample' if axis_clock else 'time',lazy=lazy)


def _tut_audit_when(ao,layout,limit):
    return ao.events.when(Condition.compare(Condition.var('value'),'gt',2.),opts=WhenOptions(layout=layout,max_segments=limit,eval=ConditionEvalOptions(coord_name=ao.as_dataset().attrs['tal']['core']['param_coord']['name'])))

@pytest.mark.parametrize('layout',['segments','stream'])
@pytest.mark.parametrize('lazy,limit',[(False,None),(False,2),(True,2)])
def test_tut_005_axis_parameter_packing(layout,lazy,limit):
    ao=_tut_audit_series(axis_clock=True,lazy=lazy)
    tasks=[]
    with Callback(pretask=lambda key,*_:tasks.append(key)):result=_tut_audit_when(ao,layout,limit)
    assert tasks==[]
    ds=result.as_dataset().compute();index=ds.orig_index.values
    expected=np.where(index>=0,_TUT_AUDIT_CLOCK[np.maximum(index,0)],np.nan)
    assert ds['sample'].dims==ds.orig_index.dims
    np.testing.assert_equal(ds['sample'],expected)

@pytest.mark.parametrize('n',[0,8])
@pytest.mark.parametrize('layout',['segments','stream'])
@pytest.mark.parametrize('limit',[None,2])
def test_tut_006_range_unbounded_packing(n,layout,limit):
    ao=_tut_audit_series(range_index=True,n=n)
    ds=_tut_audit_when(ao,layout,limit).as_dataset()
    if layout=='segments':assert isinstance(ds.xindexes['sample'],xr.indexes.RangeIndex)



@pytest.mark.parametrize("layout", ["segments", "stream"])
@pytest.mark.parametrize("case,lazy,limit,validate", [
    ("ragged", False, None, True), ("ragged", True, 2, False),
    ("complement", False, None, False), ("complement", True, 3, True),
    ("truncated", True, 1, False), ("none", False, None, True),
    ("none", True, 2, False), ("empty", True, 2, True),
    ("zero_batch", False, None, False), ("zero_batch", True, 2, True),
])
def test_tut_005_axis_clocks_ragged_padding_and_empty(layout, case, lazy, limit, validate):
    """TUT-005: consumed parameter indexes cannot return through provenance or stream padding."""
    n = 0 if case == "empty" else 8
    lanes = 0 if case == "zero_batch" else 2
    times = np.broadcast_to(_TUT_AUDIT_CLOCK[:n], (lanes, n)) + np.arange(lanes)[:, None] * 10.
    values = np.broadcast_to(_TUT_AUDIT_VALUES[:n], (lanes, n)).copy()
    if case == "none":
        values[:] = 0.
    coords = xr.Coordinates({"sample": xr.Variable(("trial", "sample"), times)}, indexes={})
    ds = xr.Dataset({"value": (("trial", "sample"), values)}, coords=coords)
    ds = ds.assign_coords(xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(lanes, dim="trial")))
    if lazy:
        ds = ds.chunk({"sample": 2})
    if case == "ragged":
        ds = ds.assign_coords(length=("trial", [6, 3]))
    source = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=("trial",), param_coord="sample",
                                     sequence_size_coord="length" if case == "ragged" else None, validate=validate)
    snapshot = source.as_dataset(copy="deep")
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.events.when(Condition.compare(Condition.var("value"), "gt", 2.),
            opts=WhenOptions(layout=layout, inside=case != "complement", max_segments=limit,
                             eval=ConditionEvalOptions(coord_name="sample")))
    assert tasks == []
    actual = result.as_dataset().compute(scheduler="synchronous")
    episodes = [[1, 2], [5, 6]]
    if case == "complement":
        episodes = [[0], [3, 4], [7]]
    elif case == "truncated":
        episodes = [[1, 2]]
    elif case in ("none", "empty", "zero_batch"):
        episodes = []
    groups = [episodes for _ in range(lanes)]
    if case == "ragged":
        groups = [[[1, 2], [5]], [[1, 2]]]
    expected_index = np.full(actual.orig_index.shape, -1, dtype="int64")
    for lane, lane_episodes in enumerate(groups):
        if layout == "segments":
            for segment, indices in enumerate(lane_episodes):
                expected_index[lane, segment, :len(indices)] = indices
        else:
            indices = [index for episode in lane_episodes for index in episode]
            expected_index[lane, :len(indices)] = indices
    np.testing.assert_array_equal(actual.orig_index, expected_index)
    expected_clock = np.full(expected_index.shape, np.nan)
    for lane in range(lanes):
        keep = expected_index[lane] >= 0
        expected_clock[lane][keep] = times[lane, expected_index[lane][keep]]
    np.testing.assert_allclose(actual["sample"], expected_clock)
    assert actual["sample"].dims == actual.value.dims == actual.orig_index.dims
    assert "sample" not in actual.xindexes and set(actual.data_vars) == {"value"}
    assert isinstance(actual.xindexes["trial"], xr.indexes.RangeIndex)
    assert actual.xindexes["trial"].equals(snapshot.xindexes["trial"])
    count_name = "length" if case == "ragged" and layout == "segments" else "segment_size" if layout == "segments" else "stream_size"
    np.testing.assert_array_equal(actual[count_name], (expected_index >= 0).sum(axis=-1))
    xr.testing.assert_identical(source.as_dataset(), snapshot)


@pytest.mark.parametrize("layout", ["segments", "stream"])
@pytest.mark.parametrize("axis,lazy,limit", [(False, False, None), (True, False, 2), (True, True, 2), (False, True, 2)])
def test_tut_005_006_sampled_indexes_are_consumed(layout, axis, lazy, limit):
    """TUT-005/006: sampled native indexes are consumed while complete batch groups survive."""
    raw = xr.Dataset({"value": (("trial", "sample"), np.tile(_TUT_AUDIT_VALUES, (2, 1)))},
                     coords={"time": ("sample", _TUT_AUDIT_CLOCK), "tag": ("sample", np.arange(8) + 100.)})
    raw = raw.assign_coords(xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(8, dim="sample"))).set_xindex("tag")
    raw = raw.assign_coords(xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(10, 30, 10, dim="trial")))
    source = _tut_audit_ao(raw, batch=("trial",), param="sample" if axis else "time", lazy=lazy)
    snapshot = source.as_dataset(copy="deep")
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = _tut_audit_when(source, layout, limit).as_dataset()
    assert tasks == []
    assert "tag" not in result.xindexes
    if layout == "segments" and not axis:
        assert isinstance(result.xindexes["sample"], xr.indexes.RangeIndex)
    else:
        assert "sample" not in result.xindexes
    assert result.xindexes["trial"].equals(snapshot.xindexes["trial"])
    actual = result.compute(scheduler="synchronous")
    expected = np.full(actual.orig_index.shape, -1, dtype="int64")
    if layout == "segments":
        expected[:, :, :2] = [[1, 2], [5, 6]]
    else:
        expected[:, :4] = [1, 2, 5, 6]
    np.testing.assert_array_equal(actual.orig_index, expected)
    np.testing.assert_allclose(actual.tag, np.where(expected >= 0, 100. + expected, np.nan))
    parameter = "sample" if axis else "time"
    recorded = np.arange(8, dtype=float) if axis else _TUT_AUDIT_CLOCK
    np.testing.assert_allclose(actual[parameter], np.where(expected >= 0, recorded[expected.clip(min=0)], np.nan))
    assert actual.tag.dims == actual[parameter].dims == actual.orig_index.dims
    assert set(actual.data_vars) == {"value"}
    xr.testing.assert_identical(source.as_dataset(), snapshot)
