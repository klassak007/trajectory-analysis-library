"""Contracts 061/063/072/091: output topology and validity belong to the operation."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as ScipyRotation

from tal.core import AnalysisLayoutSpec
from tal.spatial import AngularVelocity, LinearVelocity, Position, Rotation, Velocity


def _rotation(*, batched, count=None, lazy=False, n=3):
    dims = ("trial", "sample", "quat") if batched else ("sample", "quat")
    shape = (2, n, 1) if batched else (n, 1)
    q = ScipyRotation.from_euler("z", .3).as_quat()
    coords = {"sample": np.arange(n), "quat": list("xyzw"), "time": ("sample", np.arange(n, dtype=float))}
    if batched:
        coords["trial"] = ["a", "b"]
    if count is not None:
        coords["count"] = ("trial", count) if batched else count
    ds = xr.Dataset({"q": (dims, np.tile(q, shape))}, coords=coords)
    if lazy:
        ds["q"] = ds.q.chunk({"sample": 2})
    ao = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",) if batched else (), core_dims=("quat",), param_coord="time", sequence_size_coord="count" if count is not None else None).wrap(ds)
    return Rotation(ao)


def _position(rotation):
    ds = rotation.as_dataset().drop_dims("quat")
    shape = tuple(ds.sizes[d] for d in ("trial", "sample") if d in ds.dims)
    dims = tuple(d for d in ("trial", "sample") if d in ds.dims)
    ds["p"] = xr.DataArray(np.broadcast_to([1., 0., 0.], (*shape, 3)), dims=(*dims, "axis"))
    ds = ds.assign_coords(axis=list("xyz"))
    return Position(AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",) if "trial" in dims else (), core_dims=("axis",), param_coord="time", sequence_size_coord="count" if "count" in ds.coords else None).wrap(ds.drop_attrs()))


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("count", [None, 0, 1, 3])
def test_broadcast_compose_uses_output_batch_topology(lazy, validate, reverse, count):
    """ID: SPATIAL_VALIDITY_TOPOLOGY_001; prefix counts remain per output trial."""
    left = _rotation(batched=False, count=count, lazy=lazy)
    right = _rotation(batched=True, count=[3, 1], lazy=lazy)
    before = [left.as_dataset(), right.as_dataset()]
    operands = (right, left) if reverse else (left, right)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = operands[0].compose(operands[1], validate=validate)
        inverse = out.inverse(validate=validate)
    assert not tasks
    ds = out.as_dataset().compute()
    expected_count = np.minimum([3, 1], 3 if count is None else count)
    np.testing.assert_array_equal(ds["count"], expected_count)
    assert ds["count"].dims == ("trial",)
    source_ds = operands[0].as_dataset()
    if "count" in source_ds.coords and source_ds["count"].variable.equals(ds["count"].variable):
        xr.testing.assert_identical(ds["count"].variable, source_ds["count"].variable)
    assert ds.attrs["tal"]["core"]["roles"]["batch_dims"] == ["trial"]
    expected = np.tile(ScipyRotation.from_euler("z", .6).as_matrix(), (2, 3, 1, 1))
    valid = np.arange(3)[None, :] < expected_count[:, None]
    expected[~valid] = np.nan
    np.testing.assert_allclose(out.as_matrix().to_dataarray().compute().transpose("trial", "sample", ...), expected, atol=1e-12)
    np.testing.assert_allclose(inverse.as_matrix().to_dataarray().compute().transpose("trial", "sample", ...), expected.swapaxes(-2, -1), atol=1e-12)
    for obj, original in zip((left, right), before, strict=True):
        xr.testing.assert_identical(obj.as_dataset(), original)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("family", ["position", "linear", "velocity"])
@pytest.mark.parametrize("n", [0, 3])
def test_rotation_apply_broadcast_finalizes_target_topology(lazy, validate, family, n):
    """ID: SPATIAL_VALIDITY_TOPOLOGY_002; vector/composite targets inherit resolved batches."""
    rotation = _rotation(batched=True, count=[n, min(n, 1)], lazy=lazy, n=n)
    position = _position(_rotation(batched=False, count=min(n, 1), n=n))
    target = position if family == "position" else LinearVelocity(position)
    if family == "velocity":
        target = Velocity.from_linear_angular(target, AngularVelocity(position.rename({"p": "angular", "axis": "spin"})))
    before = target.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = rotation.apply(target, validate=validate)
    assert not tasks
    ds = out.as_dataset().compute()
    np.testing.assert_array_equal(ds["count"], [min(n, 1)] * 2)
    assert ds.attrs["tal"]["core"]["roles"]["batch_dims"] == ["trial"]
    values = out.linear() if family == "velocity" else out
    expected = np.tile([np.cos(.3), np.sin(.3), 0.], (2, n, 1))
    expected[:, 1:] = np.nan
    np.testing.assert_allclose(values.to_dataarray().compute().transpose("trial", "sample", ...), expected, atol=1e-12)
    xr.testing.assert_identical(target.as_dataset(), before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_temporal_coverage_from_either_operand_survives_chaining(lazy, validate, reverse, prefix):
    """ID: SPATIAL_VALIDITY_TOPOLOGY_003; non-prefix temporal masks survive arithmetic."""
    source = _rotation(batched=True, count=[3, 3], lazy=lazy)
    times = [0., 1., 3.] if prefix else [-1., 0., 1.]
    temporal = source.param.at(times)
    ordinary = Rotation(source.as_dataset().assign_coords(time=("sample", times)))
    pair = (temporal, ordinary) if reverse else (ordinary, temporal)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = pair[0].compose(pair[1], validate=validate)
        chained = out.inverse(validate=validate).as_matrix(validate=validate)
    assert not tasks
    ds = out.as_dataset().compute()
    valid = np.tile([True, True, False] if prefix else [False, True, True], (2, 1))
    np.testing.assert_array_equal(ds.valid.transpose("trial", "sample"), valid)
    expected = np.tile(ScipyRotation.from_euler("z", -.6).as_matrix(), (2, 3, 1, 1))
    expected[~valid] = np.nan
    np.testing.assert_allclose(chained.to_dataarray().compute().transpose("trial", "sample", ...), expected, atol=1e-12)
    # Parameter query reuse observes generated ownership instead of treating it as a caller label.
    assert ds.valid.attrs["tal_reserved_name"] == "valid"


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("operation", ["compose", "apply"])
@pytest.mark.parametrize("join", ["param", "outer"])
def test_coverage_follows_selected_alignment_key(lazy, operation, join):
    """ID: SPATIAL_VALIDITY_TOPOLOGY_004; payload and mask share key/join semantics."""
    left = _rotation(batched=False, count=2, lazy=lazy)
    right_ds = _rotation(batched=False, count=3, lazy=lazy).as_dataset()
    labels = [10, 11, 12] if join == "param" else [-1, 0, 1]
    right = Rotation(right_ds.assign_coords(sample=labels))
    if operation == "apply":
        left = _position(left)
    if join == "param":
        right = right.a(on="param", sequence_join=None)
    else:
        left = left.set_param_coord(name=None)
        right = right.set_param_coord(name=None).a(sequence_join="outer")
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = left.compose(right) if operation == "compose" else right.apply(left)
    assert not tasks
    ds = out.as_dataset().compute()
    if join == "param":
        np.testing.assert_array_equal(ds["count"], 2)
        np.testing.assert_array_equal(ds["sample"], [0, 1, 2])
    else:
        np.testing.assert_array_equal(ds.valid, [False, True, True, False])
        np.testing.assert_array_equal(ds["sample"], [-1, 0, 1, 2])
    if operation == "compose":
        values = out.inverse().as_matrix().to_dataarray().compute()
        expected = ScipyRotation.from_euler("z", -.6).as_matrix()
    else:
        values = out.to_dataarray().compute()
        expected = [np.cos(.3), np.sin(.3), 0.]
    valid_slice = slice(0, 2) if join == "param" else slice(1, 3)
    np.testing.assert_allclose(values.isel(sample=valid_slice), np.broadcast_to(expected, values.isel(sample=valid_slice).shape), atol=1e-12)


@pytest.mark.parametrize("validate", [False, True])
def test_deferred_temporal_coverage_survives_binary_result(validate):
    """ID: SPATIAL_VALIDITY_TOPOLOGY_005; lazy prefix uncertainty retains an owned mask."""
    source = _rotation(batched=False, count=3, lazy=True)
    query = xr.DataArray([0., 1., 3.], dims="sample").chunk({"sample": 2})
    left = source.param.at(query)
    right = Rotation(_rotation(batched=True, lazy=True).as_dataset().assign_coords(time=("sample", [0., 1., 3.])))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = right.compose(left, validate=validate)
        inverse = out.inverse(validate=validate)
    assert not tasks
    np.testing.assert_array_equal(out.as_dataset().compute().valid.broadcast_like(out.as_dataset().q.isel(quat=0, drop=True)).transpose("trial", "sample"), [[True, True, False]] * 2)
    assert np.isnan(inverse.to_dataarray().compute().isel(sample=2)).all()


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_validity_metadata_cannot_replace_output_payload(lazy, validate, empty):
    """ID: SPATIAL_VALIDITY_TOPOLOGY_006; metadata name claims fail before payload work."""
    n = 0 if empty else 3
    target = _position(_rotation(batched=False, n=n)).rename({"p": "count"})
    rotation = _rotation(batched=True, count=[n, min(n, 1)], lazy=lazy, n=n)
    before = target.as_dataset()
    tasks = []
    with (
        Callback(pretask=lambda *args: tasks.append(args[0])),
        pytest.raises(ValueError, match=r"^spatial.rotation.apply:.*protected output names.*count"),
    ):
        rotation.apply(target, validate=validate)
    assert not tasks
    xr.testing.assert_identical(target.as_dataset(), before)


@pytest.mark.parametrize("lazy,validate,representation", [(False, True, "components"), (True, False, "components"), (True, True, "matrix")])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("operation", ["compose", "apply"])
@pytest.mark.parametrize("count", [1, 3])
def test_pose_translation_uses_resolved_binary_coverage(lazy, validate, representation, reverse, operation, count):
    """ID: SPATIAL_VALIDITY_TOPOLOGY_007; Pose translation and orientation share coverage."""
    from tal.spatial import Pose

    left_r = _rotation(batched=False, count=count, lazy=lazy)
    right_r = _rotation(batched=True, count=[3, 1], lazy=lazy)
    left = Pose.from_components(left_r, _position(left_r)).to_rep(representation)
    right = Pose.from_components(right_r, _position(right_r)).to_rep(representation)
    if reverse:
        left, right = right, left
    before = [left.as_dataset(), right.as_dataset()]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = left.compose(right, validate=validate) if operation == "compose" else left.apply(right.decompose()[0], validate=validate)
    assert not tasks
    np.testing.assert_array_equal(result.as_dataset().compute()["count"], np.minimum([3, 1], count))
    position = result.decompose()[0] if operation == "compose" else result
    expected = np.tile([1. + np.cos(.3), np.sin(.3), 0.], (2, 3, 1))
    valid = np.arange(3)[None, :] < np.minimum([3, 1], count)[:, None]
    expected[~valid] = np.nan
    actual = position.to_dataarray().compute().transpose("trial", "sample", ...).data
    np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-12)
    # Matrix decomposition retains its established neutral padding; the public
    # numerical result itself must mask unreachable matrices/positions.
    payload = result.as_dataset().compute()[next(iter(result.as_dataset().data_vars))].transpose("trial", "sample", ...).data
    assert np.isnan(payload[~valid]).all()
    for value, original in zip((left, right), before, strict=True):
        xr.testing.assert_identical(value.as_dataset(), original)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("representation", ["components", "matrix"])
def test_pose_preserves_nonprefix_temporal_coverage(lazy, validate, representation):
    """ID: SPATIAL_VALIDITY_TOPOLOGY_008; conversion/inverse retain owned coverage."""
    from tal.spatial import Pose

    rotation = _rotation(batched=True, count=[3, 3], lazy=lazy)
    source = Pose.from_components(rotation, _position(rotation))
    sampled = source.param.at([-1., 0., 1.])
    before = sampled.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        prepared = sampled.to_rep(representation, validate=validate)
        result = prepared.inverse(validate=validate)
        restored = result.inverse(validate=validate).as_matrix(validate=validate)
        matrix = result.as_matrix(validate=validate)
    assert not tasks
    ds = matrix.as_dataset().compute()
    np.testing.assert_array_equal(ds.valid.broadcast_like(ds.pose_matrix.isel({dim: 0 for dim in ds.attrs["tal"]["core"]["roles"]["core_dims"]}, drop=True)).transpose("trial", "sample"), [[False, True, True]] * 2)
    expected = np.eye(4)
    rotation_matrix = ScipyRotation.from_euler("z", -.3).as_matrix()
    expected[:3, :3] = rotation_matrix
    expected[:3, 3] = -rotation_matrix @ [1., 0., 0.]
    actual = ds.pose_matrix.transpose("trial", "sample", ...).data
    np.testing.assert_allclose(actual[:, 1:], np.broadcast_to(expected, (2, 2, 4, 4)), atol=1e-12)
    assert np.isnan(actual[:, 0]).all()
    for payload in result.as_dataset().compute().data_vars.values():
        assert np.isnan(payload.isel(sample=0)).all()
    np.testing.assert_allclose(restored.as_dataset().compute().pose_matrix, sampled.as_matrix().as_dataset().compute().pose_matrix, atol=1e-12)
    xr.testing.assert_identical(sampled.as_dataset(), before)
