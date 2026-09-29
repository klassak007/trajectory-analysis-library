"""Contracts 061--063: shared auxiliary equality is deferred with lazy payloads."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as ScipyRotation

from tal.spatial import Pose
from tests.core.test_spatial_validity_topology import _position, _rotation


def _operands(family, empty, mismatch=False):
    query = xr.DataArray([] if empty else [0., 1., 2.], dims="sample").chunk({"sample": 2})
    raw = _rotation(batched=True, count=[3, 3], lazy=True)
    if family in {"pose", "pose_apply"}:
        raw = Pose.from_components(raw, _position(raw))
    left = raw.param.at(query)
    right = raw.param.at([] if empty else [0., 1., 2.])
    if mismatch:
        ds = right.as_dataset().assign_coords(time=("sample", [.25, 1., 2.]))
        right = type(right)(ds)
    return left, _position(right) if family in {"apply", "pose_apply"} else right


@pytest.mark.parametrize("family", ["rotation", "apply", "pose", "pose_apply"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("validate", [False, True])
def test_shared_lazy_time_coordinate_is_checked_without_planning_tasks(family, empty, validate):
    """ID: SPATIAL_LAZY_COORDINATES_001; equal duplicate coordinates remain lazy."""
    left, right = _operands(family, empty)
    before = [left.as_dataset(), right.as_dataset()]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = left.apply(right, validate=validate) if family in {"apply", "pose_apply"} else left.compose(right, validate=validate)
    assert not tasks
    ds = result.as_dataset().compute()
    np.testing.assert_array_equal(ds.time, [] if empty else [0., 1., 2.])
    if not empty and family == "rotation":
        expected = np.tile(ScipyRotation.from_euler("z", .6).as_matrix(), (2, 3, 1, 1))
        np.testing.assert_allclose(result.as_matrix().to_dataarray().compute().transpose("trial", "sample", ...), expected, atol=1e-12)
    if not empty and family == "apply":
        np.testing.assert_allclose(result.to_dataarray().compute().transpose("trial", "sample", ...), np.tile([np.cos(.3), np.sin(.3), 0.], (2, 3, 1)), atol=1e-12)
    if not empty and family == "pose_apply":
        np.testing.assert_allclose(result.to_dataarray().compute().transpose("trial", "sample", ...), np.tile([1. + np.cos(.3), np.sin(.3), 0.], (2, 3, 1)), atol=1e-12)
    if not empty and family == "pose":
        position, rotation = result.decompose()
        np.testing.assert_allclose(position.to_dataarray().compute().transpose("trial", "sample", ...), np.tile([1. + np.cos(.3), np.sin(.3), 0.], (2, 3, 1)), atol=1e-12)
        np.testing.assert_allclose(rotation.as_matrix().to_dataarray().compute().transpose("trial", "sample", ...), np.tile(ScipyRotation.from_euler("z", .6).as_matrix(), (2, 3, 1, 1)), atol=1e-12)
    for value, original in zip((left, right), before, strict=True):
        xr.testing.assert_identical(value.as_dataset(), original)


@pytest.mark.parametrize("family", ["rotation", "apply", "pose", "pose_apply"])
@pytest.mark.parametrize("part", ["payload", "coordinate"])
def test_conflicting_lazy_coordinate_fails_on_materialization(family, part):
    """ID: SPATIAL_LAZY_COORDINATES_002; either public materialization keeps equality checks."""
    left, right = _operands(family, False, mismatch=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = left.apply(right) if family in {"apply", "pose_apply"} else left.compose(right)
    assert not tasks
    ds = out.as_dataset()
    values = ds.time.data if part == "coordinate" else ds[next(iter(ds.data_vars))].data
    with pytest.raises(ValueError, match=r"spatial\..*: shared coordinate 'time' must agree exactly"):
        values.compute()


def test_lazy_component_validity_requires_agreement():
    """ID: SPATIAL_LAZY_COORDINATES_003; component assembly cannot choose a validity side."""
    query = xr.DataArray([0., 1., 2.], dims="sample").chunk({"sample": 1})
    position = _position(_rotation(batched=True, count=[1, 1], lazy=True)).param.at(query)
    rotation = _rotation(batched=True, count=[3, 3], lazy=True).param.at(query)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = Pose.from_components(rotation, position)
    assert not tasks
    with pytest.raises(ValueError, match=r"spatial.pose.from_components: shared coordinate 'valid' must agree exactly"):
        result.as_dataset().compute()
