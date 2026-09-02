from __future__ import annotations

import operator
import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation

from tal import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import PathSolveOptions, Pose, Position, Rotation, solve_pose_path_transform, solve_rotation_path_transform
from tal.spatial.metadata import get_pose_rep, get_rotation_rep
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


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
