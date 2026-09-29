"""Contracts 005/014/061/077: positional work retains native index identity."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as ScipyRotation

from tal.core import (
    AnalysisLayoutSpec,
    AnalysisObject,
    ComponentExtractOptions,
    read_components,
)
from tal.spatial import LinearVelocity, Pose, Position, Rotation
from tests.core.test_spatial_validity_topology import _position, _rotation


class _PairTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size, calls, dim):
        self.calls = calls
        self.dim = dim
        super().__init__((dim, "alias"), {dim: size})

    def __deepcopy__(self, memo):
        return type(self)(self.dim_size[self.dim], self.calls, self.dim)

    def forward(self, dim_positions):
        self.calls.append("forward")
        positions = dim_positions[self.dim]
        return {self.dim: positions + 10., "alias": positions + 100.}

    def reverse(self, coord_labels):
        self.calls.append("reverse")
        return {self.dim: coord_labels[self.dim] - 10.}

    def equals(self, other, **kwargs):
        return isinstance(other, _PairTransform) and self.dim == other.dim and self.dim_size == other.dim_size


def _indexed(ds, *, dim, kind, calls):
    out = ds.drop_vars(dim, errors="ignore")
    if kind == "unindexed":
        return out
    index = (xr.indexes.RangeIndex.arange(ds.sizes[dim], dim=dim) if kind == "range"
             else xr.indexes.CoordinateTransformIndex(_PairTransform(ds.sizes[dim], calls, dim)))
    return out.assign_coords(xr.Coordinates.from_xindex(index))


def _assert_indexes(source, result, dim):
    names = tuple(source.xindexes.get_all_coords(dim)) if dim in source.xindexes else ()
    if not names:
        assert dim not in result.xindexes
    for name in names:
        assert type(result.xindexes[name]) is type(source.xindexes[name])
        assert result.xindexes[name].equals(source.xindexes[name])
    if names:
        assert tuple(result.xindexes.get_all_coords(dim)) == names


@pytest.mark.parametrize("kind", ["range", "transform", "unindexed"])
@pytest.mark.parametrize("lazy,validate,empty", [(False, True, False), (True, False, False), (True, True, True)])
@pytest.mark.parametrize("operation", ["inverse", "matrix", "compose", "apply", "pose_inverse", "pose_compose"])
def test_native_sequence_indexes_survive_spatial_validity(kind, lazy, validate, empty, operation):
    """ID: NATIVE_NUMERICAL_TOPOLOGY_001; native sequence labels never become positions."""
    n = 0 if empty else 3
    calls = []
    raw = _rotation(batched=True, count=[n, min(n, 1)], lazy=lazy, n=n)
    source = Rotation(_indexed(raw.as_dataset(), dim="sample", kind=kind, calls=calls))
    position = Position(_indexed(_position(raw).as_dataset(), dim="sample", kind=kind, calls=calls))
    pose = Pose.from_components(source, position) if operation.startswith("pose") else None
    before = source.as_dataset()
    calls.clear()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        if operation == "inverse":
            result = source.inverse(validate=validate).as_matrix()
        elif operation == "matrix":
            result = source.as_matrix(validate=validate)
        elif operation == "compose":
            result = source.compose(source, validate=validate).as_matrix()
        elif operation == "apply":
            result = source.apply(LinearVelocity(position), validate=validate)
        elif operation == "pose_inverse":
            result = pose.inverse(validate=validate).as_matrix()
        else:
            result = pose.compose(pose, validate=validate).as_matrix()
    assert not tasks and not calls
    ds = result.as_dataset()
    _assert_indexes(before, ds, "sample")
    _assert_indexes(before, source.as_dataset(), "sample")
    np.testing.assert_array_equal(ds["count"], [n, min(n, 1)])
    data = ds[next(iter(ds.data_vars))].transpose("trial", "sample", ...).data
    actual = data.compute() if lazy else data
    valid = np.arange(n)[None, :] < np.array([n, min(n, 1)])[:, None]
    angle = -.3 if "inverse" in operation else .6 if "compose" in operation else .3
    matrix = ScipyRotation.from_euler("z", angle).as_matrix()
    expected = matrix
    if operation == "apply":
        expected = matrix @ [1., 0., 0.]
    if operation.startswith("pose"):
        expected = np.eye(4)
        expected[:3, :3] = matrix
        expected[:3, 3] = -matrix @ [1., 0., 0.] if operation == "pose_inverse" else np.array([1., 0., 0.]) + ScipyRotation.from_euler("z", .3).apply([1., 0., 0.])
    np.testing.assert_allclose(actual[valid], np.broadcast_to(expected, actual[valid].shape), atol=1e-12)
    assert np.isnan(actual[~valid]).all()
    assert not calls


@pytest.mark.parametrize("kind", ["range", "transform", "unindexed"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("family", ["generic", "rotation"])
@pytest.mark.parametrize("reverse", [False, True])
def test_semantic_batch_expansion_carries_complete_indexes(kind, lazy, family, reverse):
    """ID: NATIVE_NUMERICAL_TOPOLOGY_002; semantic broadcast shares actual index groups."""
    calls = []
    if family == "rotation":
        left = _rotation(batched=False, lazy=lazy)
        ds = _indexed(_rotation(batched=True, lazy=lazy).as_dataset(), dim="trial", kind=kind, calls=calls)
        right = Rotation(ds)
    else:
        ds = _indexed(xr.Dataset({"v": (("trial", "sample"), [[1., 2.], [3., 4.]])}, coords={"trial": [0, 1], "sample": [0, 1]}), dim="trial", kind=kind, calls=calls)
        if lazy:
            ds["v"] = ds.v.chunk({"sample": 1}).variable
        right = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",)).wrap(ds)
        left = AnalysisLayoutSpec(sequence_dim="sample").wrap(xr.Dataset({"v": ("sample", [10., 20.])}, coords={"sample": [0, 1]}))
    if reverse:
        left, right = right, left
    calls.clear()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = left.compose(right).as_matrix() if family == "rotation" else left + right
    assert not calls and not tasks
    out = result.as_dataset()
    _assert_indexes(ds, out, "trial")
    data = out[next(iter(out.data_vars))].transpose("trial", "sample", ...).data
    actual = data.compute() if lazy else data
    expected = np.broadcast_to(ScipyRotation.from_euler("z", .6).as_matrix(), (2, 3, 3, 3)) if family == "rotation" else [[11., 22.], [13., 24.]]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert not calls


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("kind", ["range", "transform", "unindexed"])
def test_native_structural_masks_support_reducers_and_parameter_queries(lazy, kind):
    """ID: NATIVE_NUMERICAL_TOPOLOGY_003; shared masks serve core and typed consumers."""
    calls = []
    ds = xr.Dataset({"v": (("trial", "sample"), [[1., 2., 90.], [4., 90., 90.]])}, coords={"trial": [0, 1], "sample": [0, 1, 2], "time": ("sample", [0., 1., 2.]), "count": ("trial", [2, 1])})
    ds = _indexed(ds, dim="trial", kind=kind, calls=calls)
    if lazy:
        ds["v"] = ds.v.chunk({"sample": 2}).variable
    source = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",), param_coord="time", sequence_size_coord="count").wrap(ds)
    calls.clear()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        reduced = source.sum(dim="sample")
        sampled = source.param.at([0., .5])
    assert not calls and not tasks
    _assert_indexes(ds, sampled.as_dataset(), "trial")
    np.testing.assert_allclose(reduced.as_dataset().v.data.compute() if lazy else reduced.as_dataset().v.data, [3., 4.])
    np.testing.assert_allclose(sampled.as_dataset().v.data.compute() if lazy else sampled.as_dataset().v.data, [[1., 1.5], [4., np.nan]], equal_nan=True)


@pytest.mark.parametrize("kind", ["range", "transform", "unindexed"])
@pytest.mark.parametrize("lazy,empty", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("validate", [False, True])
def test_component_projection_preserves_native_validity(kind, lazy, empty, validate):
    """ID: NATIVE_NUMERICAL_TOPOLOGY_004; core projection never changes sequence rows."""
    n = 0 if empty else 3
    calls = []
    raw = _rotation(batched=True, count=[n, min(n, 1)], lazy=lazy, n=n)
    rotation = Rotation(_indexed(raw.as_dataset(), dim="sample", kind=kind, calls=calls))
    position = Position(_indexed(_position(raw).as_dataset(), dim="sample", kind=kind, calls=calls))
    source = Pose.from_components(rotation, position)
    tasks = []
    calls.clear()
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        part = source.components.extract(
            opts=ComponentExtractOptions(names=("position",), output_var="selected"), validate=validate,
        )["position"]
    assert not tasks and not calls
    ds = part.as_dataset()
    assert type(part) is AnalysisObject
    assert set(ds.data_vars) == {"selected"}
    assert read_components(part)["position"].var == "selected"
    assert ds.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "count"
    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "time"
    np.testing.assert_array_equal(ds["count"], [n, min(n, 1)])
    _assert_indexes(source.as_dataset(), ds, "sample")
    assert not calls


def test_unindexed_unknown_slice_still_prunes_validity():
    """ID: NATIVE_NUMERICAL_TOPOLOGY_005; projection intent does not relax slicing."""
    raw = _rotation(batched=False, count=1)
    source = Rotation(raw.as_dataset().drop_vars("sample"))
    result = source.isel(sample=[2, 1])
    assert result.as_dataset().attrs["tal"]["core"].get("validity") is None
