"""Contracts 061/063/091: validity cannot redefine the aligned payload topology."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as ScipyRotation

from tal.core import AnalysisLayoutSpec
from tal.spatial import LinearVelocity, Pose, Position, Rotation


def _rotation(times, *, batches=(), count=None, lazy=False):
    dims = (*batches, "sample", "quat")
    shape = (*([2] * len(batches)), len(times), 4)
    coords = {name: [100, 200] for name in batches}
    coords.update(sample=np.arange(len(times)), time=("sample", times), quat=list("xyzw"))
    if count is not None:
        coords["count"] = count
    ds = xr.Dataset({"q": (dims, np.broadcast_to(ScipyRotation.from_euler("z", .3).as_quat(), shape))}, coords=coords)
    if lazy:
        ds["q"] = ds.q.chunk({"sample": 2})
    return Rotation(AnalysisLayoutSpec(sequence_dim="sample", batch_dims=batches, core_dims=("quat",), param_coord="time", sequence_size_coord="count" if count is not None else None).wrap(ds))


def _position(times, *, batches=(), lazy=False):
    rotation = _rotation(times, batches=batches, lazy=lazy)
    ds = rotation.as_dataset().drop_dims("quat")
    shape = (*([2] * len(batches)), len(times), 3)
    ds["p"] = xr.DataArray(np.broadcast_to([1., 0., 0.], shape), dims=(*batches, "sample", "axis"))
    ds = ds.assign_coords(axis=list("xyz"))
    if lazy:
        ds["p"] = ds.p.chunk({"sample": 2})
    return Position(AnalysisLayoutSpec(sequence_dim="sample", batch_dims=batches, core_dims=("axis",), param_coord="time").wrap(ds.drop_attrs()))


@pytest.mark.parametrize("lazy,validate", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("collision", ["batch", "index"])
@pytest.mark.parametrize("family", ["rotation", "pose", "linear"])
def test_generated_size_cannot_replace_surviving_labels(lazy, validate, empty, collision, family):
    """ID: SPATIAL_VALIDITY_ALIGNMENT_001; generated counts cannot replace indexes/dims."""
    times = np.arange(0 if empty else 3, dtype=float)
    rotation = _rotation(times, count=len(times), lazy=lazy)
    position = _position(times, batches=("trial",), lazy=lazy)
    if collision == "batch":
        position = position.rename({"trial": "count"})
    else:
        ds = position.as_dataset().drop_vars("trial").assign_coords(count=("trial", [100, 200])).set_xindex("count")
        position = Position(ds)
    transform = rotation
    if family == "pose":
        transform = Pose.from_components(rotation, _position(times))
    if family == "linear":
        position = LinearVelocity(position)
    before = [obj.as_dataset() for obj in (transform, position)]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])), pytest.raises(ValueError, match=r"^spatial\.(rotation|pose)\.apply:.*count"):
        transform.apply(position, validate=validate)
    assert not tasks
    for obj, original in zip((transform, position), before, strict=True):
        xr.testing.assert_identical(obj.as_dataset(), original)


@pytest.mark.parametrize("lazy,validate", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("join", ["outer", "inner", "param"])
@pytest.mark.parametrize("batches", [("trial",), ("trial", "sensor")])
@pytest.mark.parametrize("empty", [False, True])
def test_partial_temporal_masks_follow_explicit_alignment(lazy, validate, join, batches, empty):
    """ID: SPATIAL_VALIDITY_ALIGNMENT_002; partial masks acquire actual operand axes."""
    times = np.array([] if empty else ([0., 1., 2.] if lazy else [-1., 0., 1.]))
    query = xr.DataArray(times, dims="sample")
    if lazy:
        query = query.chunk({"sample": 2})
    temporal = _rotation(np.arange(3.), count=3, lazy=lazy).param.at(query)
    combined = _rotation(times, batches=batches, lazy=lazy).compose(temporal)
    assert combined.as_dataset().valid.dims == ("sample",)
    # Explicit param alignment consumes index labels; keep those eager while
    # preserving deferred payload and validity from the query chain.
    if join == "param":
        combined = Rotation(combined.as_dataset().assign_coords(time=("sample", times)))
    left = combined.a(on="param", sequence_join=None) if join == "param" else combined.a(batch_join=join)
    right = Rotation(combined.as_dataset().assign_coords(sample=np.arange(len(times)) + 10)) if join == "param" else combined
    before = [obj.as_dataset() for obj in (left, right)]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = left.compose(right, validate=validate)
    assert not tasks
    ds = result.as_dataset().compute()
    expected_valid = np.broadcast_to(times >= 0, (*([2] * len(batches)), len(times)))
    np.testing.assert_array_equal(ds.valid.transpose(*batches, "sample"), expected_valid)
    expected = np.broadcast_to(ScipyRotation.from_euler("z", 1.2).as_matrix(), (*expected_valid.shape, 3, 3)).copy()
    expected[~expected_valid] = np.nan
    np.testing.assert_allclose(result.as_matrix().to_dataarray().compute().transpose(*batches, "sample", ...), expected, atol=1e-12)
    assert ds.attrs["tal"]["core"]["roles"]["batch_dims"] == list(batches)
    for dim in batches:
        np.testing.assert_array_equal(ds[dim], [100, 200])
    for obj, original in zip((left, right), before, strict=True):
        xr.testing.assert_identical(obj.as_dataset(), original)


@pytest.mark.parametrize("lazy,validate", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("family", ["position", "linear", "pose"])
def test_partial_masks_reach_adjacent_numerical_consumers(lazy, validate, family):
    """ID: SPATIAL_VALIDITY_ALIGNMENT_003; Pose/vector delegates share mask planning."""
    times = np.array([-1., 0., 1.])
    query = xr.DataArray(times, dims="sample")
    if lazy:
        query = query.chunk({"sample": 2})
    source_r = _rotation(np.arange(3.), count=3, lazy=lazy)
    batch_r = _rotation(times, batches=("trial",), lazy=lazy)
    if family == "pose":
        source = Pose.from_components(source_r, _position(np.arange(3.)))
        batch = Pose.from_components(batch_r, _position(times, batches=("trial",)))
        combined = batch.compose(source.param.at(query))
        target = combined
    else:
        combined = batch_r.compose(source_r.param.at(query))
        target = combined.apply(_position(times))
        if family == "linear":
            target = LinearVelocity(target)
    before = [obj.as_dataset() for obj in (combined, target)]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        selected = combined.a(batch_join="outer")
        result = selected.compose(target, validate=validate) if family == "pose" else selected.apply(target, validate=validate)
    assert not tasks
    ds = result.as_dataset().compute()
    np.testing.assert_array_equal(ds.valid.broadcast_like(ds[next(iter(ds.data_vars))].isel({dim: 0 for dim in ds.attrs["tal"]["core"]["roles"]["core_dims"]}, drop=True, missing_dims="ignore")).transpose("trial", "sample"), [[False, True, True]] * 2)
    position = result.decompose()[0] if family == "pose" else result
    expected = ScipyRotation.from_euler("z", 1.2).apply([1., 0., 0.])
    if family == "pose":
        expected = sum(ScipyRotation.from_euler("z", angle).apply([1., 0., 0.]) for angle in [0., .3, .6, .9])
    actual = position.to_dataarray().compute().transpose("trial", "sample", ...).data
    np.testing.assert_allclose(actual[:, 1:], np.broadcast_to(expected, (2, 2, 3)), atol=1e-12)
    assert np.isnan(actual[:, 0]).all()
    for obj, original in zip((combined, target), before, strict=True):
        xr.testing.assert_identical(obj.as_dataset(), original)


@pytest.mark.parametrize("join", ["outer", "inner"])
@pytest.mark.parametrize("reverse", [False, True])
def test_partial_mask_preserves_batch_join_domains(join, reverse):
    """ID: SPATIAL_VALIDITY_ALIGNMENT_004; coverage follows each joined batch label."""
    times = np.array([0., 1., 3.])
    temporal = _rotation(np.arange(3.), count=3, lazy=True).param.at(xr.DataArray(times, dims="sample").chunk({"sample": 2}))
    left = _rotation(times, batches=("trial", "sensor"), lazy=True).compose(temporal)
    right = Rotation(left.as_dataset().assign_coords(trial=[200, 300]))
    if reverse:
        left, right = right, left
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = left.a(batch_join=join).compose(right)
    assert not tasks
    ds = result.as_dataset().compute()
    labels = [100, 200, 300] if join == "outer" else [200]
    expected_valid = (np.array(labels)[:, None, None] == 200) & (np.arange(3)[None, None, :] < np.array([2, 2])[None, :, None])
    np.testing.assert_array_equal(ds.trial, labels)
    np.testing.assert_array_equal(ds.valid.transpose("trial", "sensor", "sample"), expected_valid)
    expected = np.broadcast_to(ScipyRotation.from_euler("z", 1.2).as_matrix(), (*expected_valid.shape, 3, 3)).copy()
    expected[~expected_valid] = np.nan
    np.testing.assert_allclose(result.as_matrix().to_dataarray().compute().transpose("trial", "sensor", "sample", ...), expected, atol=1e-12)


def test_nonconflicting_native_index_survives_mask_expansion():
    """ID: SPATIAL_VALIDITY_ALIGNMENT_005; mask expansion keeps native index identity."""
    times = np.arange(3.)
    temporal = _rotation(times, count=3, lazy=True).param.at(xr.DataArray(times, dims="sample").chunk({"sample": 2}))
    index = xr.indexes.RangeIndex.arange(100, 300, 100, dim="trial")
    batch = _rotation(times, batches=("trial",), lazy=True)
    ds = batch.compose(temporal).as_dataset().drop_vars("trial")
    combined = Rotation(ds.assign_coords(xr.Coordinates.from_xindex(index)))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = combined.a(sequence_join="outer").compose(combined)
    assert not tasks
    out = result.as_dataset()
    assert isinstance(out.xindexes["trial"], xr.indexes.RangeIndex)
    assert out.xindexes["trial"].equals(index)
    np.testing.assert_array_equal(out.compute().valid.transpose("trial", "sample"), np.ones((2, 3), dtype=bool))


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
def test_generated_temporal_mask_cannot_replace_surviving_labels(validate, indexed):
    """ID: SPATIAL_VALIDITY_ALIGNMENT_006; temporal masks protect the same namespace."""
    times = np.array([-1., 0., 1.])
    rotation = _rotation(np.arange(3.), count=3, lazy=True).param.at(times)
    position = _position(times, batches=("trial",), lazy=True)
    if indexed:
        ds = position.as_dataset().drop_vars("trial").assign_coords(valid=("trial", [100, 200])).set_xindex("valid")
        position = Position(ds)
    else:
        position = position.rename({"trial": "valid"})
    before = position.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])), pytest.raises(ValueError, match=r"^spatial.rotation.apply:.*valid"):
        rotation.apply(position, validate=validate)
    assert not tasks
    xr.testing.assert_identical(position.as_dataset(), before)
