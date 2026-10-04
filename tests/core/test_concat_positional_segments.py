"""Contract 013: segment positions are independent of sequence index labels."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisLayoutSpec, SequenceConcatOptions


def _segment(labels, times, lengths, *, lazy, trials=("a", "b")):
    values = np.broadcast_to(np.asarray(times) * 10, (2, len(times))).copy()
    ds = xr.Dataset(
        {"value": (("trial", "sample"), values)},
        coords={"trial": list(trials), "time": (("trial", "sample"), np.broadcast_to(times, values.shape)),
                "tag": (("trial", "sample"), values + 1), "count": ("trial", lengths)},
    )
    if labels is not None:
        ds = ds.assign_coords(sample=labels)
    if lazy:
        ds["value"] = ds["value"].chunk({"trial": 1, "sample": 2})
        ds["tag"] = ds["tag"].chunk({"trial": 1, "sample": 2})
    return AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",), param_coord="time", sequence_size_coord="count").wrap(ds)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("labels", [None, [2, 3, 4], [40, 30, 20], [0, 1, 2]])
@pytest.mark.parametrize("sort", [False, True])
def test_concat_positional_segments_ignore_sequence_labels(lazy, labels, sort):
    """ID: COMBINE_POSITIONAL_001; complete row-wise append and sorted oracles."""
    first = _segment([10, 11], [0., 2.] if sort else [0., 1.], [2, 0], lazy=lazy)
    second = _segment(labels, [1., 3., 4.] if sort else [2., 3., 4.], [3, 1], lazy=lazy)
    snapshots = [x.as_dataset() for x in (first, second)]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = first.combine.concat_sequence([second], opts=SequenceConcatOptions(overlap="sort" if sort else "error"))
    assert not tasks
    ds = result.as_dataset(copy="shallow").compute()
    for row, sizes in enumerate(([2, 3], [0, 1])):
        times = np.concatenate([src.time.isel(trial=row).data[:n] for src, n in zip(snapshots, sizes, strict=True)])
        if sort:
            times.sort(kind="stable")
        np.testing.assert_allclose(ds.time[row, :len(times)], times)
        np.testing.assert_allclose(ds.value[row, :len(times)], times * 10)
        np.testing.assert_allclose(ds.tag[row, :len(times)], times * 10 + 1)
        assert np.isnan(ds.value[row, len(times):]).all()
    np.testing.assert_array_equal(ds["count"], [5, 1])
    np.testing.assert_array_equal(ds["sample"], np.arange(5))
    for source, before in zip((first, second), snapshots, strict=True):
        xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_concat_positional_multibatch_outer_and_empty(lazy, empty):
    """ID: COMBINE_POSITIONAL_002; outer labels and empty segments retain topology."""
    first = _segment([5, 6], [0., 1.], [0, 0] if empty else [2, 0], lazy=lazy)
    second = _segment([12, 13], [2., 3.], [0, 0] if empty else [1, 2], lazy=lazy, trials=("b", "c"))
    inputs = []
    for value in (first, second):
        ds = value.as_dataset().reset_coords(["time", "count", "tag"]).expand_dims(sensor=["imu"]).set_coords(["time", "count", "tag"])
        inputs.append(AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("sensor", "trial"), param_coord="time", sequence_size_coord="count").wrap(ds.drop_attrs()))
    result = inputs[0].combine.concat_sequence(inputs[1:], opts=SequenceConcatOptions(batch_join="outer"))
    ds = result.as_dataset().compute()
    np.testing.assert_array_equal(ds.trial, ["a", "b", "c"])
    np.testing.assert_array_equal(ds["count"], [[0, 0, 0] if empty else [2, 1, 2]])
    assert ds.sizes["sample"] == (0 if empty else 2)


def _auxiliary_index_segment(times, ticks, *, kind, lazy, count, batched=False):
    coords = {"time": ("sample", times), "tick": ("sample", ticks),
              "alias": ("sample", np.asarray(ticks) + 100), "count": count}
    ds = xr.Dataset({"v": ("sample", np.asarray(times) * 10)}, coords=coords)
    if kind in {"both", "multiple"}:
        ds = ds.assign_coords(sample=np.asarray(ticks) + 1000)
    names = ["time"] if kind == "parameter" else ["tick", "alias"] if kind == "multiple" else ["tick"]
    for name in names:
        ds = ds.set_xindex(name)
    ds.tick.attrs["units"] = "ticks"
    if batched:
        ds = ds.expand_dims(trial=["a", "b"])
        ds = ds.assign_coords(count=("trial", [count, 0]))
    if lazy:
        ds["v"] = ds.v.chunk({"sample": 1}).variable
    return AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",) if batched else (),
                              param_coord="time", sequence_size_coord="count").wrap(ds)


@pytest.mark.parametrize("kind", ["auxiliary", "both", "multiple", "parameter"])
@pytest.mark.parametrize("lazy,empty,validate", [(False, False, True), (True, False, False), (True, True, True)])
@pytest.mark.parametrize("batched", [False, True])
def test_concat_consumes_auxiliary_sequence_indexes(kind, lazy, empty, validate, batched):
    """ID: COMBINE_POSITIONAL_003; indexed auxiliaries append as sample values."""
    first = _auxiliary_index_segment([0., 1.], [21, 20], kind=kind, lazy=lazy, count=0 if empty else 1, batched=batched)
    second = _auxiliary_index_segment([2., 3., 4.], [20, 21, 22], kind=kind, lazy=lazy, count=0 if empty else 2, batched=batched)
    before = [value.as_dataset() for value in (first, second)]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = first.combine.concat_sequence([second], validate=validate)
    assert not tasks
    ds = result.as_dataset().compute()
    assert set(ds.data_vars) == {"v"}
    assert set(ds.xindexes) == ({"trial", "sample"} if batched else {"sample"})
    values = ds.isel(trial=0, drop=True) if batched else ds
    np.testing.assert_array_equal(values.time, [] if empty else [0., 2., 3.])
    np.testing.assert_array_equal(values.tick, [] if empty else [21, 20, 21])
    np.testing.assert_array_equal(values.alias, [] if empty else [121, 120, 121])
    np.testing.assert_array_equal(values.v, [] if empty else [0., 20., 30.])
    np.testing.assert_array_equal(ds["sample"], np.arange(0 if empty else 3))
    assert values.tick.attrs["units"] == "ticks"
    assert ds.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "count"
    if batched:
        np.testing.assert_array_equal(ds["count"], [0 if empty else 3, 0])
        for name in ("v", "tick", "alias", "time"):
            assert np.isnan(ds[name].isel(trial=1)).all()
    for source, original in zip((first, second), before, strict=True):
        xr.testing.assert_identical(source.as_dataset(), original)


@pytest.mark.parametrize("lazy", [False, True])
def test_auxiliary_indexes_follow_stable_sort_and_batch_join(lazy):
    """ID: COMBINE_POSITIONAL_004; sampled labels follow sorting and batch joins."""
    first = _auxiliary_index_segment([0., 2.], [10, 11], kind="multiple", lazy=lazy, count=2)
    second = _auxiliary_index_segment([1., 2.], [12, 13], kind="parameter", lazy=lazy, count=2)
    sorted_ds = first.combine.concat_sequence([second], opts=SequenceConcatOptions(overlap="sort")).as_dataset().compute()
    np.testing.assert_array_equal(sorted_ds.time, [0., 1., 2., 2.])
    np.testing.assert_array_equal(sorted_ds.tick, [10, 12, 11, 13])
    a = _auxiliary_index_segment([0., 1.], [10, 11], kind="auxiliary", lazy=lazy, count=2, batched=True)
    b = _auxiliary_index_segment([2., 3.], [12, 13], kind="auxiliary", lazy=lazy, count=2, batched=True)
    b = type(b)(b.as_dataset().assign_coords(trial=["b", "c"]))
    ds = a.combine.concat_sequence([b], opts=SequenceConcatOptions(batch_join="outer")).as_dataset().compute()
    np.testing.assert_array_equal(ds.trial, ["a", "b", "c"])
    np.testing.assert_array_equal(ds["count"], [2, 2, 0])
    np.testing.assert_array_equal(ds.tick[:2], [[10, 11], [12, 13]])
