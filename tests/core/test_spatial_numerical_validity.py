"""Contracts 061/063/088/129: validity bounds spatial numerical work."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as ScipyRotation

from tal.core import AnalysisLayoutSpec
from tal.frames import FrameGraph
from tal.spatial import (
    AngularVelocity,
    LinearVelocity,
    Pose,
    Position,
    Rotation,
    Velocity,
)


def _components(*, lazy=False, empty=False, bad=False):
    n = 0 if empty else 3
    counts = np.array([n, min(n, 1)])
    quaternion = np.tile(ScipyRotation.from_euler("x", .25).as_quat(), (2, n, 1))
    position = np.tile([1., 2., 3.], (2, n, 1))
    quaternion[1, 1:] = np.nan
    position[1, 1:] = np.nan
    if bad:
        quaternion[0, 0] = 0
    coords = {"trial": ["a", "b"], "sample": np.arange(n), "time": ("sample", np.arange(n, dtype=float)), "count": ("trial", counts)}
    outputs = []
    for cls, name, dim, labels, data in [(Position, "position", "axis", "xyz", position), (Rotation, "rotation", "quat", "xyzw", quaternion)]:
        ds = xr.Dataset({name: (("trial", "sample", dim), data)}, coords={**coords, dim: list(labels)})
        if lazy:
            ds[name] = ds[name].chunk({"trial": 1, "sample": 2})
        ao = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",), core_dims=(dim,), param_coord="time", sequence_size_coord="count").wrap(ds)
        outputs.append(cls(ao))
    return outputs


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("rep", ["quat", "matrix"])
@pytest.mark.parametrize("operation", ["inverse", "compose", "apply"])
def test_rotation_operations_respect_structural_padding(lazy, empty, rep, operation):
    """ID: SPATIAL_NUMERICAL_VALIDITY_001; independent rotation matrices and padding."""
    position, rotation = _components(lazy=lazy, empty=empty)
    rotation = rotation.to_rep(rep)
    before = rotation.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = rotation.compose(rotation) if operation == "compose" else rotation.apply(position) if operation == "apply" else rotation.inverse()
    assert not tasks
    computed = result.as_dataset().compute()
    if not empty:
        r = ScipyRotation.from_euler("x", .25)
        expected = (r * r).as_matrix() if operation == "compose" else r.inv().as_matrix()
        if operation == "apply":
            np.testing.assert_allclose(computed.position[0], np.tile(r.apply([1., 2., 3.]), (3, 1)))
        else:
            matrices = result.as_matrix().as_dataset().compute()
            np.testing.assert_allclose(matrices.rotation[0], np.tile(expected, (3, 1, 1)), atol=1e-14)
        assert np.isnan(computed[next(iter(computed.data_vars))][1, 1:]).all()
    np.testing.assert_array_equal(computed["count"], before["count"])
    xr.testing.assert_identical(rotation.as_dataset(), before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("operation", ["inverse", "compose", "apply", "as_matrix"])
def test_reached_bad_quaternion_still_fails(lazy, operation):
    """ID: SPATIAL_NUMERICAL_VALIDITY_002; masks do not suppress reached failures."""
    position, rotation = _components(lazy=lazy, bad=True)
    with pytest.raises(ValueError, match="quaternion|norm|kernel"):
        result = rotation.compose(rotation) if operation == "compose" else rotation.apply(position) if operation == "apply" else getattr(rotation, operation)()
        result.as_dataset().compute()


def _graph():
    graph = FrameGraph()
    time = np.array([0., 1., 2.])
    angles = .3 + .1 * time
    source = xr.Dataset({"p": (("sample", "axis"), np.zeros((3, 3))), "q": (("sample", "quat"), ScipyRotation.from_euler("z", angles[:, None]).as_quat())},
                        coords={"sample": np.arange(3), "time": ("sample", time), "axis": list("xyz"), "quat": list("xyzw")})
    p = Position(AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",), param_coord="time").wrap(source[["p"]]))
    q = Rotation(AnalysisLayoutSpec(sequence_dim="sample", core_dims=("quat",), param_coord="time").wrap(source[["q"]]))
    Pose.from_components(q, p, parent="world", child="deck", graph=graph).register()
    Pose.from_components(q, p, parent="deck", child="camera", graph=graph).register()
    return graph, angles


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("family", ["position", "rotation", "pose", "linear", "velocity"])
@pytest.mark.parametrize("validate", [False, True])
def test_ragged_expressions_share_validity_boundary(lazy, family, validate):
    """ID: SPATIAL_NUMERICAL_VALIDITY_003; multi-edge basis changes preserve relation."""
    position, rotation = _components(lazy=lazy)
    graph, angles = _graph()
    candidates = {"position": position, "rotation": rotation, "pose": Pose.from_components(rotation, position),
                  "linear": LinearVelocity(position), "velocity": Velocity.from_linear_angular(LinearVelocity(position), AngularVelocity(position.rename({"position": "angular", "axis": "spin"})))}
    value = candidates[family]
    source = type(value)(value, parent="world", child="body", graph=graph)
    before = source.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = source.express_in("camera", validate=validate)
    assert not tasks
    assert type(out) is type(source)
    assert out.frames.ids() == source.frames.ids() and out.graph is graph
    ds = out.as_dataset().compute()
    np.testing.assert_array_equal(ds["count"], [3, 1])
    values = out.decompose()[0] if family == "pose" else out.linear() if family == "velocity" else out
    if family != "rotation":
        expected = ScipyRotation.from_euler("z", (-2 * angles)[:, None]).apply(np.tile([1., 2., 3.], (3, 1)))
        np.testing.assert_allclose(values.to_dataarray().compute()[0], expected, atol=1e-12)
    else:
        basis = ScipyRotation.from_euler("z", (-2 * angles)[:, None])
        expected = (basis.inv() * ScipyRotation.from_euler("x", .25) * basis).as_matrix()
        np.testing.assert_allclose(out.as_matrix().to_dataarray().compute()[0], expected, atol=1e-12)
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("count_name", ["count", "other_count"])
def test_binary_spatial_coverage_keeps_truthful_sizes(lazy, validate, count_name):
    """ID: SPATIAL_NUMERICAL_VALIDITY_004; output uses intersected valid prefixes."""
    position, rotation = _components(lazy=lazy)
    ds = rotation.as_dataset().assign_coords(count=("trial", [2, 0]))
    limited = Rotation(ds).rename({"count": count_name}) if count_name != "count" else Rotation(ds)
    before = limited.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        outputs = [rotation.compose(limited, validate=validate), limited.apply(position, validate=validate)]
    assert not tasks
    for output in outputs:
        result = output.as_dataset().compute()
        np.testing.assert_array_equal(result["count"], [2, 0])
        values = result[next(iter(result.data_vars))].transpose("trial", "sample", ...)
        assert np.isnan(values[0, 2:]).all() and np.isnan(values[1]).all()
        assert result.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "count"
    xr.testing.assert_identical(limited.as_dataset(), before)
