from __future__ import annotations

import operator
from typing import Any

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation

from tal import AnalysisObject
from tal.core import ParamEvalOptions
from tal.frames import FrameGraph
from tal.spatial import (
    PathSolveOptions,
    Pose,
    Position,
    Rotation,
    solve_pose_path_transform,
    solve_rotation_path_transform,
)
from tal.spatial.metadata import get_pose_rep, get_rotation_rep
from tal.spatial.temporal import PoseTemporalOptions, RotationTemporalOptions
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames
from tests._path_options_helpers import (
    BAD_PATH_OPTIONS,
    FalseyPathSolveOptions,
    forbid_path_resolution,
)

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")

_PATH_SOLVERS = [
    pytest.param(solve_rotation_path_transform, "edge_rotation_fn", "rotation", id="rotation"),
    pytest.param(solve_pose_path_transform, "edge_pose_fn", "pose", id="pose"),
]


def _quat(axis: str, degrees: float) -> np.ndarray:
    return SciRotation.from_euler(axis, degrees, degrees=True).as_quat().astype(float)


def _rotation_from_quat(values: np.ndarray, *, sequence_dim: str = "sample") -> Rotation:
    arr = xr.DataArray(
        values,
        dims=(sequence_dim, "quat"),
        coords={sequence_dim: list(range(values.shape[0])), "quat": list(_QUAT)},
        name="rotation",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="rotation"), sequence_dim=sequence_dim, core_dims=("quat",), validate=True)
    return Rotation(ao)


def _position_from_xyz(values: np.ndarray, *, sequence_dim: str = "sample") -> Position:
    arr = xr.DataArray(
        values,
        dims=(sequence_dim, "axis"),
        coords={sequence_dim: list(range(values.shape[0])), "axis": list(_XYZ)},
        name="position",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="position"), sequence_dim=sequence_dim, core_dims=("axis",), validate=True)
    return Position(ao)


def _pose_from_translation_and_quat(translation: np.ndarray, quat: np.ndarray) -> Pose:
    position = _position_from_xyz(translation)
    rotation = _rotation_from_quat(quat)
    return Pose.from_components(rotation, position, validate=True)


def _quat_equivalent(lhs: np.ndarray, rhs: np.ndarray, *, atol: float = 1e-6) -> bool:
    return bool(np.allclose(lhs, rhs, atol=atol, rtol=0.0) or np.allclose(lhs, -rhs, atol=atol, rtol=0.0))


@pytest.mark.parametrize(("solver", "resolver_arg", "kind"), _PATH_SOLVERS)
@pytest.mark.parametrize("opts", BAD_PATH_OPTIONS)
@pytest.mark.parametrize("dst", ["sensor", "missing"])
def test_spatial_hard_127h_001_functional_options_precede_resolution(
    solver, resolver_arg, kind, opts, dst, monkeypatch,
) -> None:
    """ID: SPATIAL_HARD_127H_001_functional_options_precede_resolution."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        graph.get_or_create_frame("sensor", parent=world)
        probe = forbid_path_resolution(monkeypatch, graph)
        with pytest.raises(TypeError) as exc_info:
            solver("sensor", dst, opts=opts, **{resolver_arg: probe})
    assert str(exc_info.value) == f"spatial.path_solve.{kind}: opts must be PathSolveOptions or None."
    assert probe.events == []


@pytest.mark.parametrize(("solver", "resolver_arg", "kind"), _PATH_SOLVERS)
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("frame_endpoints", [False, True])
def test_spatial_core_127h_001_functional_options_preserve_defaults_and_graph_policy(
    solver, resolver_arg, kind, identity, frame_endpoints,
) -> None:
    """ID: SPATIAL_CORE_127H_001_functional_options_preserve_defaults_and_graph_policy."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    rotation = _rotation_from_quat(np.asarray([_quat("z", 20.0)]))
    edge = rotation if kind == "rotation" else Pose.from_components(
        rotation, _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]])), validate=True,
    )
    calls = []

    def resolver(child, parent):
        calls.append((child.id, parent.id))
        return edge

    dst = sensor if identity else world
    endpoints = (sensor, dst) if frame_endpoints else (sensor.id, dst.id)
    with graph:
        expected = solver(*endpoints, **{resolver_arg: resolver})
        for options in (None, PathSolveOptions()):
            out = solver(*endpoints, opts=options, **{resolver_arg: resolver})
            xr.testing.assert_identical(out.as_dataset(copy="none"), expected.as_dataset(copy="none"))
    with FrameGraph():
        if frame_endpoints:
            out = solver(*endpoints, opts=None, **{resolver_arg: resolver})
            xr.testing.assert_identical(out.as_dataset(copy="none"), expected.as_dataset(copy="none"))
        for options in (PathSolveOptions(graph=graph), FalseyPathSolveOptions(graph=graph)):
            out = solver(sensor.id, dst.id, opts=options, **{resolver_arg: resolver})
            xr.testing.assert_identical(out.as_dataset(copy="none"), expected.as_dataset(copy="none"))
    if identity:
        assert calls == []
    else:
        assert calls and set(calls) == {("sensor", "world")}


@pytest.mark.parametrize(("solver", "resolver_arg", "kind"), _PATH_SOLVERS)
def test_spatial_perf_127h_001_valid_options_preserve_lazy_path_solving(solver, resolver_arg, kind) -> None:
    """ID: SPATIAL_PERF_127H_001_valid_options_preserve_lazy_path_solving."""
    from dask.callbacks import Callback

    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
    rotation = _rotation_from_quat(np.asarray([_quat("z", 20.0)]))
    edge = rotation if kind == "rotation" else Pose.from_components(
        rotation, _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]])), validate=True,
    )
    edge = type(edge)(edge.as_dataset(copy="none").chunk({"sample": 1}))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = solver(sensor, world, opts=FalseyPathSolveOptions(graph=graph), **{resolver_arg: lambda *_: edge})
    assert tasks == []
    assert any(var.chunks is not None for var in out.as_dataset(copy="none").data_vars.values())


def test_spatial_core_069_solve_rotation_path_transform_identity_src_eq_dst_deterministic() -> None:
    """ID: SPATIAL_CORE_069_solve_rotation_path_transform_identity_src_eq_dst_deterministic."""
    graph = FrameGraph()
    with graph:
        camera = graph.get_or_create_frame("camera")
        out = solve_rotation_path_transform(camera, camera, edge_rotation_fn=lambda *_: None)
    assert isinstance(out, Rotation)
    assert get_rotation_rep(out.as_dataset(copy="none"), owner="test") == "quat"
    assert get_frames(out.as_dataset(copy="none")) == ("camera", "camera")
    quat = out.as_dataset(copy="none")["rotation"].isel(sample=0).values
    assert _quat_equivalent(quat, np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float))


def test_spatial_core_070_solve_rotation_path_transform_chain_deterministic_and_frame_truthful() -> None:
    """ID: SPATIAL_CORE_070_solve_rotation_path_transform_chain_deterministic_and_frame_truthful."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_sb = frame_retag(_rotation_from_quat(np.asarray([[_quat("z", 90.0)]]).reshape(1, 4)), parent="body", child="sensor", validate=True)
        edge_bw = frame_retag(_rotation_from_quat(np.asarray([[_quat("x", 90.0)]]).reshape(1, 4)), parent="world", child="body", validate=True)
        edge_map = {("sensor", "body"): edge_sb, ("body", "world"): edge_bw}
        out = solve_rotation_path_transform(sensor, world, edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)])
    expected = SciRotation.from_quat(_quat("x", 90.0)).as_matrix() @ SciRotation.from_quat(_quat("z", 90.0)).as_matrix()
    assert get_frames(out.as_dataset(copy="none")) == ("world", "sensor")
    out_matrix = out.as_matrix(validate=True).as_dataset(copy="none")["rotation"].isel(sample=0).values
    np.testing.assert_allclose(out_matrix, expected, atol=1e-6, rtol=0.0)


def test_spatial_core_071_solve_pose_path_transform_identity_src_eq_dst_deterministic() -> None:
    """ID: SPATIAL_CORE_071_solve_pose_path_transform_identity_src_eq_dst_deterministic."""
    graph = FrameGraph()
    with graph:
        tool = graph.get_or_create_frame("tool")
        out = solve_pose_path_transform(tool, tool, edge_pose_fn=lambda *_: None)
    assert isinstance(out, Pose)
    assert get_pose_rep(out.as_dataset(copy="none"), owner="test") == "components"
    assert get_frames(out.as_dataset(copy="none")) == ("tool", "tool")
    matrix = out.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].isel(sample=0).values
    np.testing.assert_allclose(matrix, np.eye(4, dtype=float), atol=1e-6, rtol=0.0)


def test_spatial_core_072_solve_pose_path_transform_chain_deterministic_and_frame_truthful() -> None:
    """ID: SPATIAL_CORE_072_solve_pose_path_transform_chain_deterministic_and_frame_truthful."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_sb = frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("z", 90.0)], dtype=float),
            ),
            parent="body",
            child="sensor",
            validate=True,
        )
        edge_bw = frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.0, 2.0, 0.0]], dtype=float),
                np.asarray([_quat("x", 90.0)], dtype=float),
            ),
            parent="world",
            child="body",
            validate=True,
        )
        edge_map = {("sensor", "body"): edge_sb, ("body", "world"): edge_bw}
        out = solve_pose_path_transform(sensor, world, edge_pose_fn=lambda child, parent: edge_map[(child.id, parent.id)])
    h_sb = edge_sb.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].isel(sample=0).values
    h_bw = edge_bw.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].isel(sample=0).values
    expected = h_bw @ h_sb
    assert get_frames(out.as_dataset(copy="none")) == ("world", "sensor")
    out_matrix = out.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].isel(sample=0).values
    np.testing.assert_allclose(out_matrix, expected, atol=1e-6, rtol=0.0)


def test_spatial_core_073_path_solver_accepts_frame_and_string_endpoints_with_explicit_graph_resolution() -> None:
    """ID: SPATIAL_CORE_073_path_solver_accepts_frame_and_string_endpoints_with_explicit_graph_resolution."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_map = {
            ("sensor", "body"): _rotation_from_quat(np.asarray([_quat("z", 15.0)], dtype=float)),
            ("body", "world"): _rotation_from_quat(np.asarray([_quat("x", 20.0)], dtype=float)),
        }
        resolver = lambda child, parent: edge_map[(child.id, parent.id)]
        out_ff = solve_rotation_path_transform(sensor, world, edge_rotation_fn=resolver)
        out_fs = solve_rotation_path_transform(sensor, "world", edge_rotation_fn=resolver)
        out_sf = solve_rotation_path_transform("sensor", world, edge_rotation_fn=resolver)
        out_ss = solve_rotation_path_transform(
            "sensor",
            "world",
            edge_rotation_fn=resolver,
            opts=PathSolveOptions(graph=graph),
        )
    mat_ff = out_ff.as_matrix(validate=True).as_dataset(copy="none")["rotation"].isel(sample=0).values
    mat_fs = out_fs.as_matrix(validate=True).as_dataset(copy="none")["rotation"].isel(sample=0).values
    mat_sf = out_sf.as_matrix(validate=True).as_dataset(copy="none")["rotation"].isel(sample=0).values
    mat_ss = out_ss.as_matrix(validate=True).as_dataset(copy="none")["rotation"].isel(sample=0).values
    np.testing.assert_allclose(mat_ff, mat_fs, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(mat_ff, mat_sf, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(mat_ff, mat_ss, atol=1e-6, rtol=0.0)


def test_spatial_core_074_path_solver_reuses_frames_find_path_fold_path_without_local_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: SPATIAL_CORE_074_path_solver_reuses_frames_find_path_fold_path_without_local_traversal."""
    import tal.spatial.ops.path_solve_ops as ops

    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_map = {
            ("sensor", "body"): _rotation_from_quat(np.asarray([_quat("z", 5.0)], dtype=float)),
            ("body", "world"): _rotation_from_quat(np.asarray([_quat("x", 7.0)], dtype=float)),
        }
        calls = {"find": 0, "fold": 0}
        original_find = ops.find_path
        original_fold = ops.fold_path

        def _wrapped_find(*args, **kwargs):
            calls["find"] += 1
            return original_find(*args, **kwargs)

        def _wrapped_fold(*args, **kwargs):
            calls["fold"] += 1
            return original_fold(*args, **kwargs)

        monkeypatch.setattr(ops, "find_path", _wrapped_find)
        monkeypatch.setattr(ops, "fold_path", _wrapped_fold)
        _ = solve_rotation_path_transform(sensor, world, edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)])
    assert calls["find"] == 1
    assert calls["fold"] == 1


def test_spatial_hard_076_path_solver_rejects_cross_graph_or_disconnected_endpoints_fail_closed() -> None:
    """ID: SPATIAL_HARD_076_path_solver_rejects_cross_graph_or_disconnected_endpoints_fail_closed."""
    g1 = FrameGraph()
    g2 = FrameGraph()
    with g1:
        world = g1.get_or_create_frame("world")
        sensor = g1.get_or_create_frame("sensor", parent=world)
    with g2:
        map_root = g2.get_or_create_frame("map")
    with pytest.raises(ValueError, match="different FrameGraph"):
        solve_rotation_path_transform(sensor, map_root, edge_rotation_fn=lambda *_: _rotation_from_quat(np.asarray([_quat("z", 1.0)], dtype=float)))

    g3 = FrameGraph()
    with g3:
        a = g3.get_or_create_frame("a")
        b = g3.get_or_create_frame("b", parent=a)
        c = g3.get_or_create_frame("c")
        d = g3.get_or_create_frame("d", parent=c)
        with pytest.raises(ValueError, match="no path"):
            solve_rotation_path_transform(b, d, edge_rotation_fn=lambda *_: _rotation_from_quat(np.asarray([_quat("z", 1.0)], dtype=float)))


def test_spatial_hard_077_path_solver_rejects_missing_or_noncoercible_edge_payload_fail_closed() -> None:
    """ID: SPATIAL_HARD_077_path_solver_rejects_missing_or_noncoercible_edge_payload_fail_closed."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        with pytest.raises(ValueError, match="edge resolver must return Rotation-coercible payload"):
            solve_rotation_path_transform(sensor, world, edge_rotation_fn=lambda *_: None)
        with pytest.raises(ValueError, match="edge resolver must return Rotation-coercible payload"):
            solve_rotation_path_transform(sensor, world, edge_rotation_fn=lambda *_: object())


def test_spatial_hard_078_path_solver_rejects_edge_frame_tag_conflict_fail_closed() -> None:
    """ID: SPATIAL_HARD_078_path_solver_rejects_edge_frame_tag_conflict_fail_closed."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        bad = frame_retag(_rotation_from_quat(np.asarray([_quat("z", 30.0)], dtype=float)), parent="map", child="imu", validate=True)
        edge_map = {
            ("sensor", "body"): bad,
            ("body", "world"): _rotation_from_quat(np.asarray([_quat("x", 20.0)], dtype=float)),
        }
        with pytest.raises(ValueError, match="framed edge payload must match"):
            solve_rotation_path_transform(sensor, world, edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)])


def test_spatial_hard_079_path_solver_alignment_or_non_core_topology_conflicts_fail_closed() -> None:
    """ID: SPATIAL_HARD_079_path_solver_alignment_or_non_core_topology_conflicts_fail_closed."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_map = {
            ("sensor", "body"): _rotation_from_quat(np.asarray([_quat("z", 10.0)], dtype=float), sequence_dim="sample"),
            ("body", "world"): _rotation_from_quat(np.asarray([_quat("x", 20.0)], dtype=float), sequence_dim="time"),
        }
        with pytest.raises(ValueError, match="matching sequence_dim"):
            solve_rotation_path_transform(sensor, world, edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)])


def test_spatial_hard_080_path_solver_public_type_boundary_no_raw_runtime_exception_leakage() -> None:
    """ID: SPATIAL_HARD_080_path_solver_public_type_boundary_no_raw_runtime_exception_leakage."""
    with pytest.raises(TypeError, match="src must be Frame or non-empty string frame id"):
        solve_rotation_path_transform(123, "world", edge_rotation_fn=lambda *_: None)  # type: ignore[arg-type]

    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
        with pytest.raises(TypeError, match="edge_rotation_fn must be callable"):
            solve_rotation_path_transform(sensor, world, edge_rotation_fn=123)  # type: ignore[arg-type]


def test_spatial_hard_081_path_solver_dask_lazy_kernel_failure_preserves_path_owner_context() -> None:
    """ID: SPATIAL_HARD_081_path_solver_dask_lazy_kernel_failure_preserves_path_owner_context."""
    dask_array = pytest.importorskip("dask.array")
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        valid = _rotation_from_quat(dask_array.from_array(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float), chunks=(1, 4)))
        invalid = _rotation_from_quat(dask_array.from_array(np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=float), chunks=(1, 4)))
        edge_map = {
            ("sensor", "body"): valid,
            ("body", "world"): invalid,
        }
        out = solve_rotation_path_transform(sensor, world, edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)])
    with pytest.raises(ValueError) as exc_info:
        out.as_dataset(copy="none")["rotation"].compute()
    message = str(exc_info.value)
    assert "spatial.path_solve.rotation" in message
    assert "spatial.rotation.kernel" not in message


@pytest.mark.parametrize(
    ("solver", "resolver_arg", "owner", "edge_value"),
    [
        (
            solve_rotation_path_transform,
            "edge_rotation_fn",
            "spatial.path_solve.rotation",
            _rotation_from_quat(np.asarray([_quat("z", 10.0)], dtype=float)),
        ),
        (
            solve_pose_path_transform,
            "edge_pose_fn",
            "spatial.path_solve.pose",
            _pose_from_translation_and_quat(
                np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("z", 10.0)], dtype=float),
            ),
        ),
    ],
)
def test_spatial_hard_082_path_solver_wraps_missing_edge_key_resolver_error_with_owner_context(
    solver,
    resolver_arg: str,
    owner: str,
    edge_value,
) -> None:
    """ID: SPATIAL_HARD_082_path_solver_wraps_missing_edge_key_resolver_error_with_owner_context."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_map = {
            ("sensor", "body"): edge_value,
        }
        resolver = lambda child, parent: edge_map[(child.id, parent.id)]
        with pytest.raises(ValueError) as exc_info:
            solver(sensor, world, **{resolver_arg: resolver})
    message = str(exc_info.value)
    assert owner in message
    assert "failed for edge" in message
    assert "KeyError" not in message


@pytest.mark.parametrize(
    ("solver", "resolver_arg", "owner"),
    [
        (solve_rotation_path_transform, "edge_rotation_fn", "spatial.path_solve.rotation"),
        (solve_pose_path_transform, "edge_pose_fn", "spatial.path_solve.pose"),
    ],
)
def test_spatial_hard_083_path_solver_wraps_resolver_signature_error_with_owner_context(
    solver,
    resolver_arg: str,
    owner: str,
) -> None:
    """ID: SPATIAL_HARD_083_path_solver_wraps_resolver_signature_error_with_owner_context."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)

        def bad_resolver(child):
            return child

        with pytest.raises(TypeError) as exc_info:
            solver(sensor, world, **{resolver_arg: bad_resolver})
    assert type(exc_info.value) is TypeError
    assert type(exc_info.value.__cause__) is TypeError
    message = str(exc_info.value)
    assert owner in message
    assert "callable(child, parent)" in message
    assert "required positional argument" not in message


@pytest.mark.parametrize(
    ("solver", "resolver_arg", "owner"),
    [
        (solve_rotation_path_transform, "edge_rotation_fn", "spatial.path_solve.rotation"),
        (solve_pose_path_transform, "edge_pose_fn", "spatial.path_solve.pose"),
    ],
)
def test_spatial_hard_084_path_solver_classifies_resolver_internal_typeerror_as_runtime_owner_prefixed_valueerror(
    solver,
    resolver_arg: str,
    owner: str,
) -> None:
    """ID: SPATIAL_HARD_084_path_solver_classifies_resolver_internal_typeerror_as_runtime_owner_prefixed_valueerror."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)

        def bad_runtime_resolver(child, parent):
            return "x" + 1

        with pytest.raises(ValueError) as exc_info:
            solver(sensor, world, **{resolver_arg: bad_runtime_resolver})
    message = str(exc_info.value)
    assert owner in message
    assert "failed for edge" in message
    assert "callable(child, parent)" not in message


@pytest.mark.parametrize(
    ("solver", "resolver_arg", "owner"),
    [
        (solve_rotation_path_transform, "edge_rotation_fn", "spatial.path_solve.rotation"),
        (solve_pose_path_transform, "edge_pose_fn", "spatial.path_solve.pose"),
    ],
)
def test_spatial_hard_085_path_solver_uninspectable_resolver_signature_misuse_preserves_owner_prefixed_typeerror(
    solver,
    resolver_arg: str,
    owner: str,
) -> None:
    """ID: SPATIAL_HARD_085_path_solver_uninspectable_resolver_signature_misuse_preserves_owner_prefixed_typeerror."""
    graph = FrameGraph()

    class UninspectableBadResolver:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, child):
            return child

    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        with pytest.raises(TypeError) as exc_info:
            solver(sensor, world, **{resolver_arg: UninspectableBadResolver()})
    assert type(exc_info.value) is TypeError
    assert type(exc_info.value.__cause__) is TypeError
    message = str(exc_info.value)
    assert owner in message
    assert "callable(child, parent)" in message
    assert "failed for edge" not in message


@pytest.mark.parametrize(
    ("solver", "resolver_arg", "owner"),
    [
        (solve_rotation_path_transform, "edge_rotation_fn", "spatial.path_solve.rotation"),
        (solve_pose_path_transform, "edge_pose_fn", "spatial.path_solve.pose"),
    ],
)
def test_spatial_hard_086_path_solver_inspectable_c_callable_runtime_typeerror_classified_as_owner_prefixed_runtime_valueerror(
    solver,
    resolver_arg: str,
    owner: str,
) -> None:
    """ID: SPATIAL_HARD_086_path_solver_inspectable_c_callable_runtime_typeerror_classified_as_owner_prefixed_runtime_valueerror."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        with pytest.raises(ValueError) as exc_info:
            solver(sensor, world, **{resolver_arg: operator.add})
    message = str(exc_info.value)
    assert owner in message
    assert "failed for edge" in message
    assert "callable(child, parent)" not in message


def _dynamic_pose(
    translation: np.ndarray,
    angles: list[float],
    *,
    param_name: str = "time",
    param: list[float],
) -> Pose:
    pose = _pose_from_translation_and_quat(
        translation,
        np.asarray([_quat("z", angle) for angle in angles], dtype=float),
    )
    ds = pose.as_dataset(copy="none").assign_coords({param_name: ("sample", param)})
    return Pose(ds).set_param_coord(name=param_name, validate=True)


def _batched_dynamic_pose(
    *,
    batch_dim: str = "trial",
    labels: tuple[str, str] = ("a", "b"),
) -> Pose:
    pose = _dynamic_pose(
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        [0.0, 0.0],
        param=[0.0, 10.0],
    )
    ds = pose.as_dataset(copy="none").expand_dims({batch_dim: labels})
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(batch_dim,),
        core_dims=("axis", "quat"),
        param_coord="time",
    )
    return Pose(ao)


def test_spatial_core_129d_001_direct_pose_path_uses_explicit_query_grid() -> None:
    """ID: SPATIAL_CORE_129D_001_direct_pose_path_uses_explicit_query_grid."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    edge = frame_retag(
        _dynamic_pose(
            np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            [0.0, 90.0],
            param=[0.0, 10.0],
        ),
        parent="world",
        child="sensor",
        validate=True,
    )
    out = solve_pose_path_transform(
        sensor,
        world,
        edge_pose_fn=lambda *_: edge,
        graph=graph,
        query=[0.0, 5.0, 10.0],
        opts=PathSolveOptions(temporal=PoseTemporalOptions()),
    )
    ds = out.as_dataset(copy="none")
    assert ds.attrs["tal"]["core"]["roles"]["sequence_dim"] == "query"
    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "query"
    np.testing.assert_allclose(ds["position"].sel(axis="x"), [0.0, 5.0, 10.0])


def test_spatial_core_129d_002_position_to_frame_infers_caller_grid() -> None:
    """ID: SPATIAL_CORE_129D_002_position_to_frame_infers_caller_grid."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        graph.get_or_create_frame("sensor", parent=world)
    edge = frame_retag(
        _dynamic_pose(
            np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            [0.0, 0.0],
            param=[0.0, 10.0],
        ),
        parent="world",
        child="sensor",
        validate=True,
    )
    source = frame_retag(
        _position_from_xyz(np.zeros((3, 3))),
        parent="sensor",
        child="probe",
        validate=True,
    )
    source_ds = source.as_dataset(copy="none").assign_coords(t=("sample", [0.0, 5.0, 10.0]))
    source = Position(source_ds).set_param_coord(name="t", validate=True)
    out = source.to_frame(
        world,
        edge_pose_fn=lambda *_: edge,
        opts=PathSolveOptions(graph=graph),
    )
    ds = out.as_dataset(copy="none")
    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "t"
    np.testing.assert_allclose(ds["t"], [0.0, 5.0, 10.0])
    np.testing.assert_allclose(ds["position"].sel(axis="x"), [0.0, 5.0, 10.0])


def _static_pose(x: float = 0.0) -> Pose:
    rotation = Rotation(
        AnalysisObject.from_data(
            xr.DataArray(
                [0.0, 0.0, 0.0, 1.0],
                dims="quat",
                coords={"quat": list(_QUAT)},
                name="rotation",
            ),
            core_dims=("quat",),
        )
    )
    position = Position(
        AnalysisObject.from_data(
            xr.DataArray(
                [x, 0.0, 0.0],
                dims="axis",
                coords={"axis": list(_XYZ)},
                name="position",
            ),
            core_dims=("axis",),
        )
    )
    return Pose.from_components(rotation, position, validate=True)


def test_spatial_core_129d_003_static_direct_query_broadcasts_without_inference() -> None:
    """ID: SPATIAL_CORE_129D_003_static_direct_query_broadcasts_without_inference."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    edge = frame_retag(_static_pose(2.0), parent="world", child="sensor", validate=True)
    out = solve_pose_path_transform(
        sensor,
        world,
        edge_pose_fn=lambda *_: edge,
        graph=graph,
        query=np.asarray([2.0, 4.0]),
    )
    ds = out.as_dataset(copy="none")
    assert ds.sizes["query"] == 2
    np.testing.assert_allclose(ds["query"], [2.0, 4.0])
    np.testing.assert_allclose(ds["position"].sel(axis="x"), [2.0, 2.0])


def test_spatial_core_129d_004_rotation_query_uses_slerp_default() -> None:
    """ID: SPATIAL_CORE_129D_004_rotation_query_uses_slerp_default."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    edge = _rotation_from_quat(np.asarray([_quat("z", 0.0), _quat("z", 90.0)]))
    edge_ds = edge.as_dataset(copy="none").assign_coords(t=("sample", [0.0, 10.0]))
    edge = frame_retag(
        Rotation(edge_ds).set_param_coord(name="t", validate=True),
        parent="world",
        child="sensor",
        validate=True,
    )
    out = solve_rotation_path_transform(
        sensor,
        world,
        edge_rotation_fn=lambda *_: edge,
        graph=graph,
        query=[5.0],
    )
    expected = SciRotation.from_euler("z", 45.0, degrees=True).as_matrix()
    actual = out.as_matrix().as_dataset(copy="none")["rotation"].isel(query=0)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0.0)


def test_spatial_core_129d_005_mixed_rate_edges_evaluate_before_composition() -> None:
    """ID: SPATIAL_CORE_129D_005_mixed_rate_edges_evaluate_before_composition."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
    sensor_body = frame_retag(
        _dynamic_pose(
            np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            [0.0, 0.0],
            param=[0.0, 10.0],
        ),
        parent="body",
        child="sensor",
        validate=True,
    )
    body_world = frame_retag(
        _dynamic_pose(
            np.zeros((2, 3)),
            [0.0, 90.0],
            param_name="clock",
            param=[0.0, 20.0],
        ),
        parent="world",
        child="body",
        validate=True,
    )
    edges = {("sensor", "body"): sensor_body, ("body", "world"): body_world}
    out = solve_pose_path_transform(
        sensor,
        world,
        edge_pose_fn=lambda child, parent: edges[(child.id, parent.id)],
        graph=graph,
        query=[10.0],
    )
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["position"].isel(query=0),
        [np.sqrt(50.0), np.sqrt(50.0), 0.0],
        atol=1e-6,
        rtol=0.0,
    )


def test_spatial_core_129d_006_datetime_provider_and_query() -> None:
    """ID: SPATIAL_CORE_129D_006_datetime_provider_and_query."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    times = np.asarray(["2026-01-01", "2026-01-03"], dtype="datetime64[D]")
    edge = frame_retag(
        _dynamic_pose(
            np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            [0.0, 0.0],
            param=list(times),
        ),
        parent="world",
        child="sensor",
        validate=True,
    )
    out = solve_pose_path_transform(
        sensor,
        world,
        edge_pose_fn=lambda *_: edge,
        graph=graph,
        query=[np.datetime64("2026-01-02")],
    )
    np.testing.assert_allclose(out.as_dataset(copy="none")["position"].sel(axis="x"), [5.0])


def test_spatial_hard_129d_001_query_classification_and_coverage_fail_closed() -> None:
    """ID: SPATIAL_HARD_129D_001_query_classification_and_coverage_fail_closed."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
    dynamic = frame_retag(
        _dynamic_pose(np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), [0.0, 0.0], param=[0.0, 2.0]),
        parent="body",
        child="sensor",
        validate=True,
    )
    exact = frame_retag(
        _pose_from_translation_and_quat(np.zeros((1, 3)), np.asarray([_quat("z", 0.0)])),
        parent="world",
        child="body",
        validate=True,
    )
    edges = {("sensor", "body"): dynamic, ("body", "world"): exact}
    resolver = lambda child, parent: edges[(child.id, parent.id)]
    with pytest.raises(ValueError, match="requires explicit query"):
        solve_pose_path_transform(sensor, body, edge_pose_fn=resolver, graph=graph)
    with pytest.raises(ValueError, match="exact providers"):
        solve_pose_path_transform(sensor, world, edge_pose_fn=resolver, graph=graph, query=[1.0])
    with pytest.raises(ValueError, match="closed parameter domain"):
        solve_pose_path_transform(sensor, body, edge_pose_fn=resolver, graph=graph, query=[-1.0])
    with pytest.raises(ValueError, match="closed parameter domain"):
        solve_pose_path_transform(sensor, body, edge_pose_fn=resolver, graph=graph, query=[np.nan])


def test_spatial_core_129d_007_identity_ignores_unused_query_and_temporal_fields() -> None:
    """ID: SPATIAL_CORE_129D_007_identity_ignores_unused_query_and_temporal_fields."""
    graph = FrameGraph()
    with graph:
        sensor = graph.get_or_create_frame("sensor")
        body = graph.get_or_create_frame("body", parent=sensor)
    opts = PathSolveOptions(graph=graph, temporal=object())
    out = solve_pose_path_transform(sensor, sensor, query=object(), opts=opts)
    assert out.as_dataset(copy="none").sizes["sample"] == 1
    with pytest.raises(TypeError, match="opts.temporal"):
        solve_pose_path_transform(body, sensor, query=[0.0], opts=opts)


def test_spatial_perf_129d_001_dynamic_payload_remains_lazy_during_planning() -> None:
    """ID: SPATIAL_PERF_129D_001_dynamic_payload_remains_lazy_during_planning."""
    from dask.callbacks import Callback

    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    edge = _dynamic_pose(
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        [0.0, 90.0],
        param=[0.0, 10.0],
    )
    edge = Pose(edge.as_dataset(copy="none").chunk({"sample": 1}))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        out = solve_pose_path_transform(
            sensor,
            world,
            edge_pose_fn=lambda *_: edge,
            graph=graph,
            query=[0.0, 5.0, 10.0],
        )
    assert tasks  # strict coverage may realize only the chunked parameter coordinate
    assert any(var.chunks is not None for var in out.as_dataset(copy="none").data_vars.values())


def test_spatial_core_129d_009_direct_batched_query_preserves_query_topology() -> None:
    """ID: SPATIAL_CORE_129D_009_direct_batched_query_preserves_query_topology."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    edge = frame_retag(
        _batched_dynamic_pose(),
        parent="world",
        child="sensor",
        validate=True,
    )
    query = xr.DataArray(
        [[0.0, 5.0], [5.0, 10.0]],
        dims=("trial", "when"),
        coords={"trial": ["a", "b"], "when": [10, 20]},
    )
    out = solve_pose_path_transform(
        sensor,
        world,
        edge_pose_fn=lambda *_: edge,
        graph=graph,
        query=query,
    )
    ds = out.as_dataset(copy="none")
    assert ds.attrs["tal"]["core"]["roles"]["sequence_dim"] == "query"
    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "query_value"
    expected_query = query.rename({"when": "query"}).rename("query_value")
    expected_query = expected_query.assign_coords(query_value=expected_query)
    xr.testing.assert_identical(ds["query_value"], expected_query)
    np.testing.assert_allclose(
        ds["position"].sel(axis="x"),
        [[0.0, 5.0], [5.0, 10.0]],
    )


def test_spatial_hard_129d_002_direct_batch_indexes_must_match_exactly() -> None:
    """ID: SPATIAL_HARD_129D_002_direct_batch_indexes_must_match_exactly."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    edge = frame_retag(
        _batched_dynamic_pose(),
        parent="world",
        child="sensor",
        validate=True,
    )
    query = xr.DataArray(
        [[0.0], [10.0]],
        dims=("trial", "when"),
        coords={"trial": ["b", "a"], "when": [0]},
    )
    with pytest.raises(ValueError, match="provider batch index must exactly match"):
        solve_pose_path_transform(
            sensor,
            world,
            edge_pose_fn=lambda *_: edge,
            graph=graph,
            query=query,
        )


def test_spatial_hard_129d_005_direct_provider_batches_cannot_form_cartesian_product() -> None:
    """ID: SPATIAL_HARD_129D_005_direct_provider_batches_cannot_form_cartesian_product."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
    edges = {
        ("sensor", "body"): frame_retag(
            _batched_dynamic_pose(batch_dim="trial"),
            parent="body",
            child="sensor",
            validate=True,
        ),
        ("body", "world"): frame_retag(
            _batched_dynamic_pose(batch_dim="run"),
            parent="world",
            child="body",
            validate=True,
        ),
    }
    with pytest.raises(ValueError, match="would create Cartesian expansion"):
        solve_pose_path_transform(
            sensor,
            world,
            edge_pose_fn=lambda child, parent: edges[(child.id, parent.id)],
            graph=graph,
            query=[5.0],
        )


def test_spatial_core_129d_011_temporal_options_map_provider_policy_only() -> None:
    """ID: SPATIAL_CORE_129D_011_temporal_options_map_provider_policy_only."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    edge = _dynamic_pose(
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        [0.0, 90.0],
        param=[100.0, 200.0],
    )
    edge_ds = edge.as_dataset(copy="none").assign_coords(clock=("sample", [0.0, 20.0]))
    edge = frame_retag(Pose(edge_ds), parent="world", child="sensor", validate=True)
    temporal = PoseTemporalOptions(
        position_opts=ParamEvalOptions(method="nearest", query_dim="when"),
        rotation_opts=RotationTemporalOptions(method="nearest", query_dim="when"),
        on="clock",
    )
    out = solve_pose_path_transform(
        sensor,
        world,
        edge_pose_fn=lambda *_: edge,
        graph=graph,
        query=[8.0, 12.0],
        opts=PathSolveOptions(temporal=temporal),
    )
    ds = out.as_dataset(copy="none")
    assert ds.attrs["tal"]["core"]["roles"]["sequence_dim"] == "when"
    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "when"
    np.testing.assert_allclose(ds["when"], [8.0, 12.0])
    np.testing.assert_allclose(ds["position"].sel(axis="x"), [0.0, 10.0])


def test_spatial_hard_129d_006_temporal_options_preflight_precedes_provider_callback() -> None:
    """ID: SPATIAL_HARD_129D_006_temporal_options_preflight_precedes_provider_callback."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    calls: list[tuple[str, str]] = []

    def resolver(child, parent):
        calls.append((child.id, parent.id))
        return _static_pose()

    temporal = PoseTemporalOptions(
        position_opts=ParamEvalOptions(query_dim="position_query"),
        rotation_opts=RotationTemporalOptions(query_dim="rotation_query"),
    )
    with pytest.raises(ValueError, match="query_dim.*must match"):
        solve_pose_path_transform(
            sensor,
            world,
            edge_pose_fn=resolver,
            graph=graph,
            query=[0.0],
            opts=PathSolveOptions(temporal=temporal),
        )
    assert calls == []


def test_spatial_hard_129d_007_direct_query_axis_collision_is_owned() -> None:
    """ID: SPATIAL_HARD_129D_007_direct_query_axis_collision_is_owned."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    position = Position(
        AnalysisObject.from_data(
            xr.DataArray(
                [1.0, 0.0, 0.0],
                dims="query",
                coords={"query": list(_XYZ)},
                name="position",
            ),
            core_dims=("query",),
        )
    )
    edge = Pose.from_components(
        _rotation_from_quat(np.asarray([_quat("z", 0.0)])).isel(sample=0),
        position,
    )
    with pytest.raises(
        ValueError,
        match="spatial.path_solve.pose: query_dim 'query' collides",
    ):
        solve_pose_path_transform(
            sensor,
            world,
            edge_pose_fn=lambda *_: edge,
            graph=graph,
            query=[0.0, 1.0],
        )


def test_spatial_perf_129d_002_mixed_dynamic_path_builds_lazy_graph() -> None:
    """ID: SPATIAL_PERF_129D_002_mixed_dynamic_path_builds_lazy_graph."""
    from dask.callbacks import Callback

    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
    edges = {
        ("sensor", "body"): frame_retag(
            Pose(
                _dynamic_pose(
                    np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    [0.0, 0.0],
                    param=[0.0, 1.0],
                ).as_dataset(copy="none").chunk({"sample": 1})
            ),
            parent="body",
            child="sensor",
        ),
        ("body", "world"): frame_retag(
            Pose(
                _dynamic_pose(
                    np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    [0.0, 45.0],
                    param=[0.0, 1.0],
                ).as_dataset(copy="none").chunk({"sample": 1})
            ),
            parent="world",
            child="body",
        ),
    }
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        out = solve_pose_path_transform(
            sensor,
            world,
            edge_pose_fn=lambda child, parent: edges[(child.id, parent.id)],
            graph=graph,
            query=np.linspace(0.0, 1.0, 5),
        )
    assert tasks
    assert all(variable.chunks is not None for variable in out.as_dataset(copy="none").data_vars.values())
    query = np.linspace(0.0, 1.0, 5)
    world_from_body = edges[("body", "world")].param.at(query, validate=False)
    body_from_sensor = edges[("sensor", "body")].param.at(query, validate=False)
    reference = world_from_body.compose(body_from_sensor, validate=False)
    out_tasks = sum(
        len(variable.data.__dask_graph__())
        for variable in out.as_dataset(copy="none").data_vars.values()
    )
    reference_tasks = sum(
        len(variable.data.__dask_graph__())
        for variable in reference.as_dataset(copy="none").data_vars.values()
    )
    assert out_tasks <= reference_tasks


def _direct_query_graph() -> tuple[FrameGraph, object, object]:
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    return graph, sensor, world


def _solve_direct_pose(
    edge: Pose,
    query: object,
    *,
    temporal: PoseTemporalOptions | None = None,
) -> Pose:
    graph, sensor, world = _direct_query_graph()
    tagged = frame_retag(edge, parent="world", child="sensor", validate=True)
    return solve_pose_path_transform(
        sensor,
        world,
        edge_pose_fn=lambda *_: tagged,
        graph=graph,
        query=query,
        opts=None if temporal is None else PathSolveOptions(temporal=temporal),
    )


def _solve_direct_rotation(
    edge: Rotation,
    query: object,
    *,
    temporal: PoseTemporalOptions | None = None,
) -> Rotation:
    graph, sensor, world = _direct_query_graph()
    tagged = frame_retag(edge, parent="world", child="sensor", validate=True)
    return solve_rotation_path_transform(
        sensor,
        world,
        edge_rotation_fn=lambda *_: tagged,
        graph=graph,
        query=query,
        opts=None if temporal is None else PathSolveOptions(temporal=temporal),
    )


def _replace_trial_coordinates(value: Pose, coords: xr.Coordinates) -> Pose:
    ds = value.as_dataset(copy="none")
    if "trial" in ds.xindexes:
        ds = ds.drop_indexes("trial")
    ds = ds.drop_vars("trial", errors="ignore").assign_coords(coords)
    return Pose(ds)


class _DirectQueryTransform(xr.indexes.CoordinateTransform):
    def __init__(self, calls: list[str], *, offset: float = 0.5) -> None:
        self.calls = calls
        self.offset = offset
        super().__init__(("trial",), {"trial": 2})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("forward")
        return {"trial": dim_positions["trial"] + self.offset}

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("reverse")
        return {"trial": coord_labels["trial"] - self.offset}

    def equals(self, other: object, **kwargs: object) -> bool:
        _ = kwargs
        return isinstance(other, _DirectQueryTransform) and self.offset == other.offset


def _direct_query_index_coordinates(
    kind: str,
    *,
    calls: list[str],
) -> tuple[xr.Coordinates, str]:
    if kind == "range":
        index = xr.indexes.RangeIndex.arange(2, coord_name="trial", dim="trial")
        return xr.Coordinates.from_xindex(index), "trial"
    if kind == "transform":
        index = xr.indexes.CoordinateTransformIndex(_DirectQueryTransform(calls))
        return xr.Coordinates.from_xindex(index), "trial"
    coords = xr.Dataset(coords={"label": ("trial", ["a", "b"])}).set_xindex("label").coords
    return xr.Coordinates(coords), "label"


@pytest.mark.parametrize("provider_kind", ["static", "dynamic", "batch-subset"])
@pytest.mark.parametrize("kind", ["range", "transform", "renamed-pandas"])
def test_spatial_core_129d_012_direct_query_owns_batch_topology(
    kind: str,
    provider_kind: str,
) -> None:
    """ID: SPATIAL_CORE_129D_012_direct_query_owns_batch_topology."""
    calls: list[str] = []
    coords, index_name = _direct_query_index_coordinates(kind, calls=calls)
    if provider_kind == "static":
        edge = _static_pose(3.0)
    elif provider_kind == "dynamic":
        edge = _dynamic_pose(
            np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            [0.0, 0.0],
            param=[0.0, 10.0],
        )
    else:
        edge = _replace_trial_coordinates(_batched_dynamic_pose(), coords)
    source = edge.as_dataset(copy="deep")
    query_coords = coords.assign(run=("run", ["r0", "r1"]), when=("when", [10, 20]))
    query = xr.DataArray(
        np.asarray([[[0.0, 5.0], [5.0, 10.0]], [[2.0, 4.0], [6.0, 8.0]]]),
        dims=("run", "trial", "when"),
        coords=query_coords,
    )
    calls.clear()

    out = _solve_direct_pose(edge, query)
    ds = out.as_dataset(copy="none")

    assert ds.attrs["tal"]["core"]["roles"] == {
        "batch_dims": ["run", "trial"],
        "core_dims": ["axis", "quat"],
        "sequence_dim": "query",
    }
    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "query_value"
    assert ds["position"].sel(axis="x").dims == ("run", "trial", "query")
    expected = xr.full_like(query, 3.0) if provider_kind == "static" else query
    np.testing.assert_allclose(ds["position"].sel(axis="x"), expected)
    assert type(ds.xindexes[index_name]) is type(query.xindexes[index_name])
    assert ds.xindexes[index_name].equals(query.xindexes[index_name])
    assert calls == []
    xr.testing.assert_identical(edge.as_dataset(copy="none"), source)

    rotation = edge.decompose(validate=False)[1]
    rotation_ds = _solve_direct_rotation(rotation, query).as_dataset(copy="none")
    assert rotation_ds["rotation"].dims == ("run", "trial", "query", "quat")
    assert type(rotation_ds.xindexes[index_name]) is type(query.xindexes[index_name])
    assert rotation_ds.xindexes[index_name].equals(query.xindexes[index_name])
    assert calls == []


@pytest.mark.parametrize("renamed", ["sequence", "parameter"])
def test_spatial_core_129d_013_direct_query_namespace_isolation(renamed: str) -> None:
    """ID: SPATIAL_CORE_129D_013_direct_query_namespace_isolation."""
    edge = _dynamic_pose(
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        [0.0, 0.0],
        param=[0.0, 2.0],
    )
    edge = edge.rename({"sample" if renamed == "sequence" else "time": "query"})
    edge = Pose(edge.as_dataset(copy="none").assign_coords(query_value=123))
    query = xr.DataArray(
        [[0.0, 1.0]],
        dims=("trial", "when"),
        coords={"trial": ["a"], "when": [5, 6]},
    )

    ds = _solve_direct_pose(edge, query).as_dataset(copy="none")

    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "query_value_"
    assert ds.coords["query_value"].item() == 123
    np.testing.assert_allclose(ds.coords["query_value_"], query)
    np.testing.assert_allclose(ds["position"].sel(axis="x"), query)

    if renamed == "sequence":
        collision = Pose(_static_pose(1.0).as_dataset(copy="none").assign(query=42.0))
        with pytest.raises(ValueError, match="query_dim 'query' collides"):
            _solve_direct_pose(collision, query)
        with pytest.raises(ValueError, match="query_dim 'query' collides"):
            _solve_direct_pose(_static_pose(1.0), query.assign_coords(query=42.0))


def test_spatial_hard_129d_008_provider_only_batch_policy() -> None:
    """ID: SPATIAL_HARD_129D_008_provider_only_batch_policy."""
    batched = _batched_dynamic_pose()
    with pytest.raises(ValueError, match="provider-only batch dims.*trial"):
        _solve_direct_pose(batched, [0.0, 1.0])

    singleton = Pose(batched.as_dataset(copy="none").isel(trial=slice(0, 1)))
    out = _solve_direct_pose(singleton, [0.0, 1.0])
    assert "trial" not in out.as_dataset(copy="none").dims

    mismatch = xr.DataArray(
        [[0.0], [1.0]],
        dims=("trial", "when"),
        coords={"trial": ["b", "a"], "when": [0]},
    )
    with pytest.raises(ValueError, match="provider batch index must exactly match"):
        _solve_direct_pose(batched, mismatch)

    duplicate_query = xr.DataArray(
        [[0.0], [1.0]],
        dims=("trial", "when"),
        coords={"trial": ["a", "a"], "when": [0]},
    )
    with pytest.raises(ValueError, match="labels along 'trial' must be unique"):
        _solve_direct_pose(_static_pose(1.0), duplicate_query)

    provider_coords = xr.Dataset(
        coords={"trial": ["a", "b"], "label": ("trial", [10, 20])}
    ).set_xindex("label").coords
    indexed = _replace_trial_coordinates(batched, xr.Coordinates(provider_coords))
    matching_query = xr.DataArray(
        [[0.0], [1.0]],
        dims=("trial", "when"),
        coords=provider_coords.assign(when=("when", [0])),
    )
    assert _solve_direct_pose(indexed, matching_query).as_dataset(copy="none").sizes["trial"] == 2
    with pytest.raises(ValueError, match="relevant coordinate topology differs"):
        _solve_direct_pose(indexed, matching_query.drop_indexes("label").drop_vars("label"))

    for kind in ("transform", "renamed-pandas"):
        provider_calls: list[str] = []
        provider_index, _ = _direct_query_index_coordinates(kind, calls=provider_calls)
        if kind == "transform":
            query_calls: list[str] = []
            query_index = xr.Coordinates.from_xindex(
                xr.indexes.CoordinateTransformIndex(
                    _DirectQueryTransform(query_calls, offset=1.5)
                )
            )
        else:
            query_calls = []
            query_index = xr.Coordinates(
                xr.Dataset(coords={"label": ("trial", ["b", "a"])}).set_xindex("label").coords
            )
        native_provider = _replace_trial_coordinates(batched, provider_index)
        native_query = xr.DataArray(
            [[0.0], [1.0]],
            dims=("trial", "when"),
            coords=query_index.assign(when=("when", [0])),
        )
        with pytest.raises(ValueError, match="provider batch index must exactly match"):
            _solve_direct_pose(native_provider, native_query)
        assert provider_calls == []
        assert query_calls == []


def _dask_graph_keys(value: Pose) -> set[object]:
    keys: set[object] = set()
    for variable in value.as_dataset(copy="none").data_vars.values():
        graph = getattr(variable.data, "__dask_graph__", lambda: None)()
        if graph is not None:
            keys.update(graph.keys())
    return keys


def test_spatial_perf_129d_003_direct_query_planning_skips_payload_execution() -> None:
    """ID: SPATIAL_PERF_129D_003_direct_query_planning_skips_payload_execution."""
    from dask.callbacks import Callback

    edge = _dynamic_pose(
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        [0.0, 0.0],
        param=[0.0, 2.0],
    )
    edge_ds = edge.as_dataset(copy="none").copy()
    edge_ds["position"] = edge_ds["position"].chunk({"sample": 1})
    edge_ds["rotation"] = edge_ds["rotation"].chunk({"sample": 1})
    edge = Pose(edge_ds)
    query = xr.DataArray(
        [[0.0, 1.0], [1.0, 2.0]],
        dims=("trial", "when"),
        coords={"trial": ["a", "b"], "when": [0, 1]},
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        out = _solve_direct_pose(edge, query)

    aligned_ds = edge.as_dataset(copy="none").expand_dims({"trial": query.coords["trial"]})
    aligned = Pose(aligned_ds).set_roles(batch_dims=("trial",), validate=True)
    internal_query = query.rename({"when": "__tal_path_query"})
    temporal = PoseTemporalOptions(
        position_opts=ParamEvalOptions(query_dim="__tal_path_query"),
        rotation_opts=RotationTemporalOptions(query_dim="__tal_path_query"),
    )
    reference = aligned.param.at(internal_query, opts=temporal, validate=False)
    reference_ds = reference.as_dataset(copy="none")
    reference_order = (
        "trial",
        "sample",
        *(dim for dim in reference_ds.dims if dim not in {"trial", "sample"}),
    )
    reference = Pose(reference_ds.transpose(*reference_order))

    assert tasks == []
    assert _dask_graph_keys(out) == _dask_graph_keys(reference)
    assert all(var.chunks is not None for var in out.as_dataset(copy="none").data_vars.values())


@pytest.mark.parametrize("kind", ["pose", "rotation"])
def test_spatial_core_129d_014_effective_direct_query_namespace(kind: str) -> None:
    """ID: SPATIAL_CORE_129D_014_effective_direct_query_namespace."""
    edge = _dynamic_pose(
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        [0.0, 0.0],
        param=[0.0, 2.0],
    )
    provider = edge if kind == "pose" else edge.decompose(validate=False)[1]
    before = provider.as_dataset(copy="deep")
    query = xr.DataArray(
        [[0.0, 1.0]],
        dims=("sample", "query_value"),
        coords={
            "sample": ["trial-0"],
            "query_value": [10, 20],
            "time": ("sample", [7]),
        },
    )
    solve = _solve_direct_pose if kind == "pose" else _solve_direct_rotation

    ds = solve(
        provider,
        query,
        temporal=PoseTemporalOptions(on="time"),
    ).as_dataset(copy="none")

    assert ds.attrs["tal"]["core"]["roles"]["batch_dims"] == ["sample"]
    assert ds.attrs["tal"]["core"]["param_coord"]["name"] == "query_value"
    assert ds.coords["time"].dims == ("sample",)
    np.testing.assert_allclose(ds.coords["query_value"], query)
    xr.testing.assert_identical(provider.as_dataset(copy="none"), before)


@pytest.mark.parametrize("kind", ["pose", "rotation"])
@pytest.mark.parametrize(
    "collision",
    ["core-dimension", "data-variable", "core-coordinate", "batch-coordinate"],
)
def test_spatial_hard_129d_009_query_namespace_preflight_is_owned(
    kind: str,
    collision: str,
) -> None:
    """ID: SPATIAL_HARD_129D_009_query_namespace_preflight_is_owned."""
    from dask.callbacks import Callback

    pose = (
        _batched_dynamic_pose()
        if collision == "batch-coordinate"
        else _static_pose(1.0)
    )
    if collision == "batch-coordinate":
        pose = Pose(
            pose.as_dataset(copy="none").assign_coords(aux=("trial", [1, 2]))
        )
    edge = pose if kind == "pose" else pose.decompose(validate=False)[1]
    variable = "position" if kind == "pose" else "rotation"
    core_dim = "axis" if kind == "pose" else "quat"
    if collision == "core-dimension":
        query = xr.DataArray(
            np.zeros((edge.as_dataset(copy="none").sizes[core_dim], 1)),
            dims=(core_dim, "when"),
        )
    elif collision == "batch-coordinate":
        query = xr.DataArray(
            np.zeros((2, 1)),
            dims=("trial", "when"),
            coords={"trial": ["a", "b"], "aux": ("trial", [3, 4])},
        )
    else:
        coord = variable if collision == "data-variable" else core_dim
        query = xr.DataArray(
            np.zeros((1, 1)),
            dims=("trial", "when"),
            coords={coord: ("trial", [1])},
        )
    edge_ds = edge.as_dataset(copy="none").copy()
    for name in edge_ds.data_vars:
        edge_ds[name] = edge_ds[name].chunk()
    edge = type(edge)(edge_ds)
    solve = _solve_direct_pose if kind == "pose" else _solve_direct_rotation
    tasks: list[object] = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(
            ValueError,
            match=rf"spatial.path_solve.{kind}: query batch topology collides",
        ),
    ):
        solve(edge, query)
    assert tasks == []


def test_spatial_perf_129d_004_corrective_preflight_skips_payload_execution() -> None:
    """ID: SPATIAL_PERF_129D_004_corrective_preflight_skips_payload_execution."""
    from dask.callbacks import Callback

    edge = _batched_dynamic_pose().rename({"trial": "query"})
    edge_ds = edge.as_dataset(copy="none").isel(query=slice(0, 1)).copy()
    for name in edge_ds.data_vars:
        edge_ds[name] = edge_ds[name].chunk()
    edge = Pose(edge_ds)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        out = _solve_direct_pose(edge, [0.0, 1.0])

    assert tasks == []
    assert "query" in out.as_dataset(copy="none").dims
    assert all(
        variable.chunks is not None
        for variable in out.as_dataset(copy="none").data_vars.values()
    )
