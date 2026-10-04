from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as SciRotation

from tal import AnalysisObject
from tal.core.schema_read import read_roles
from tal.frames import FrameGraph
from tal.spatial import (
    AngularVelocity,
    LinearVelocity,
    Pose,
    Position,
    Rotation,
    Velocity,
    solve_rotation_path_transform,
)
from tal.spatial.metadata import get_expressed_in


def _leaf(kind, values, *, name, dim, times, lazy=False):
    labels = list("xyzw" if values.shape[-1] == 4 else "xyz")
    data = da.from_array(values, chunks=(2, values.shape[-1])) if lazy else values
    ds = xr.Dataset(
        {name: (("sample", dim), data)},
        coords={"sample": np.arange(len(times)), dim: labels, "time": ("sample", times)},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(dim,), param_coord="time")
    return kind(ao)


def _source(kind, times, *, lazy, graph):
    xyz = np.tile([1.0, 2.0, 3.0], (len(times), 1))
    quat = np.tile(SciRotation.from_euler("x", 35, degrees=True).as_quat(), (len(times), 1))
    position = _leaf(Position, xyz, name="offset", dim="cartesian", times=times, lazy=lazy)
    rotation = _leaf(Rotation, quat, name="attitude", dim="quaternion", times=times, lazy=lazy)
    if kind == "position":
        source = position
    elif kind.startswith("rotation"):
        source = rotation.as_matrix() if kind.endswith("matrix") else rotation
    elif kind.startswith("pose"):
        source = Pose.from_components(rotation, position)
        if kind.endswith("matrix"):
            source = source.as_matrix()
    else:
        linear = _leaf(LinearVelocity, xyz, name="velocity", dim="linear_axis", times=times, lazy=lazy)
        angular = _leaf(AngularVelocity, xyz, name="spin", dim="angular_axis", times=times, lazy=lazy)
        source = linear if kind == "linear" else Velocity.from_linear_angular(linear, angular)
    if kind.endswith("matrix"):
        core_dims = read_roles(source.as_dataset())[3]
        source = source.rename(dict(zip(core_dims, ("matrix_row", "matrix_col"), strict=True)))
    ds = source.as_dataset(copy="shallow").assign_coords(label=("sample", [f"q{i}" for i in range(len(times))]))
    return type(source)(ds, parent="sensor", child="probe", graph=graph)


def _edge(*, lazy):
    times = np.array([0.0, 1.0])
    quat = SciRotation.from_euler("z", [[0], [90]], degrees=True).as_quat()
    rotation = _leaf(Rotation, quat, name="rotation", dim="quat", times=times, lazy=lazy)
    position = _leaf(Position, np.zeros((2, 3)), name="position", dim="axis", times=times, lazy=lazy)
    return rotation, Pose.from_components(rotation, position)


def _assert_numeric(out, kind, times):
    basis = SciRotation.from_euler("z", 90 * times[:, None], degrees=True).as_matrix()
    vectors = np.einsum("nij,j->ni", basis, [1.0, 2.0, 3.0])
    orientation = SciRotation.from_euler("x", 35, degrees=True).as_matrix()
    matrices = basis.transpose(0, 2, 1) @ orientation @ basis
    if kind in {"position", "linear"}:
        np.testing.assert_allclose(next(iter(out.as_dataset().data_vars.values())), vectors, atol=1e-12)
    elif kind.startswith("rotation"):
        np.testing.assert_allclose(next(iter(out.as_matrix().as_dataset().data_vars.values())), matrices, atol=1e-12)
    elif kind.startswith("pose"):
        position, rotation = out.decompose()
        np.testing.assert_allclose(next(iter(position.as_dataset().data_vars.values())), vectors, atol=1e-12)
        np.testing.assert_allclose(next(iter(rotation.as_matrix().as_dataset().data_vars.values())), matrices, atol=1e-12)
    else:
        for component in (out.linear(), out.angular()):
            np.testing.assert_allclose(next(iter(component.as_dataset().data_vars.values())), vectors, atol=1e-12)


@pytest.mark.parametrize("kind", ["position", "rotation", "rotation_matrix", "pose", "pose_matrix", "linear", "velocity"])
@pytest.mark.parametrize("lazy,validate", [(False, True), (True, False)])
@pytest.mark.parametrize("empty", [False, True])
def test_spatial_core_expression_output_001_dynamic_declarations(kind, lazy, validate, empty):
    """ID: SPATIAL_CORE_EXPRESSION_OUTPUT_001_intermediate_and_final_declarations."""
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    graph.get_or_create_frame("sensor", parent=world)
    times = np.array([]) if empty else np.array([0.0, 0.5, 1.0])
    source = _source(kind, times, lazy=lazy, graph=graph)
    rotation, pose = _edge(lazy=lazy)
    snapshots = [value.as_dataset(copy="deep") for value in (source, rotation, pose)]
    calls = []

    def resolver(child, parent):
        calls.append((child.id, parent.id))
        return pose if kind.startswith("pose") else rotation

    key = "edge_pose_fn" if kind.startswith("pose") else "edge_rotation_fn"
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = source.express_in("world", **{key: resolver}, validate=validate)
    assert tasks == []
    assert calls == [("sensor", "world")]
    assert out.graph is graph
    assert out.frames.ids() == ("sensor", "probe")
    actual = out.as_dataset(copy="none")
    before = snapshots[0]
    assert get_expressed_in(actual, owner="test") == "world"
    assert set(actual.data_vars) == set(before.data_vars)
    assert dict(actual.sizes) == dict(before.sizes)
    assert read_roles(actual) == read_roles(before)
    xr.testing.assert_identical(actual.coords["label"], before.coords["label"])
    xr.testing.assert_identical(actual.coords["time"], before.coords["time"])
    _assert_numeric(out, kind, times)
    for value, snapshot in zip((source, rotation, pose), snapshots, strict=True):
        xr.testing.assert_identical(value.as_dataset(copy="none"), snapshot)


@pytest.mark.parametrize("lazy", [False, True])
def test_spatial_core_expression_output_002_direct_query_coordinates(lazy):
    """ID: SPATIAL_CORE_EXPRESSION_OUTPUT_002_direct_rotation_namespace."""
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    graph.get_or_create_frame("sensor", parent=world)
    rotation, _ = _edge(lazy=lazy)
    query = xr.DataArray([0.25, 0.75], dims="sample", coords={"label": ("sample", ["a", "b"])})
    out = solve_rotation_path_transform("sensor", "world", graph=graph, edge_rotation_fn=lambda *_: rotation, query=query)
    np.testing.assert_allclose(
        out.as_matrix().as_dataset()["rotation"],
        SciRotation.from_euler("z", [[22.5], [67.5]], degrees=True).as_matrix(), atol=1e-12,
    )
    np.testing.assert_array_equal(out.as_dataset().label, ["a", "b"])
    assert out.graph is graph
    for name in ("rotation", "quat"):
        bad = query.assign_coords({name: ("sample", ["a", "b"])})
        with pytest.raises(ValueError, match="spatial.path_solve.rotation"):
            solve_rotation_path_transform("sensor", "world", graph=graph, edge_rotation_fn=lambda *_: rotation, query=bad)
