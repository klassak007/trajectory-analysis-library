from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation

from tal import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import (
    PathSolveOptions,
    Pose,
    Position,
    Rotation,
    solve_pose_path_transform,
    solve_rotation_path_transform,
)
from tal.spatial.metadata import (
    get_expressed_in,
    get_pose_rep,
    get_position_rep,
    get_rotation_rep,
    set_expressed_in,
)
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames
from tests._path_options_helpers import (
    DELEGATOR_BAD_PATH_OPTIONS,
    FalseyPathSolveOptions,
    ResolutionProbe,
    forbid_path_resolution,
)

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")

_CONFIGURATION_PATH_APIS = [
    pytest.param("rotation", "solve_path_transform", "edge_rotation_fn", id="rotation-solve"),
    pytest.param("pose", "solve_path_transform", "edge_pose_fn", id="pose-solve"),
    pytest.param("position", "to_frame", "edge_pose_fn", id="position-to-frame"),
    pytest.param("position", "express_in", "edge_rotation_fn", id="position-express-in"),
    pytest.param("rotation", "express_in", "edge_rotation_fn", id="rotation-express-in"),
    pytest.param("pose", "express_in", "edge_pose_fn", id="pose-express-in"),
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
    return Pose.from_components(_rotation_from_quat(quat), _position_from_xyz(translation), validate=True)


def _with_sample_and_param(value, *, sample: list[int], param_name: str, param: list[float]):
    ds = value.as_dataset(copy="none").assign_coords(sample=sample, **{param_name: ("sample", param)})
    return value.__class__(ds).set_param_coord(name=param_name, validate=False)


def _configuration_path_source(kind):
    position = _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]]))
    rotation = _rotation_from_quat(np.asarray([_quat("x", 15.0)]))
    source = {"position": position, "rotation": rotation}.get(kind)
    if source is None:
        source = Pose.from_components(rotation, position, validate=True)
    return frame_retag(source, parent="sensor", child="probe", validate=True)


def _configuration_path_call(source, operation, dst, **kwargs):
    if operation == "solve_path_transform":
        return type(source).solve_path_transform("sensor", dst, **kwargs)
    return getattr(source, operation)(dst, **kwargs)


@pytest.mark.parametrize(("kind", "operation", "resolver_arg"), _CONFIGURATION_PATH_APIS)
@pytest.mark.parametrize("opts", DELEGATOR_BAD_PATH_OPTIONS)
@pytest.mark.parametrize("validate", [False, True])
def test_spatial_hard_127h_002_configuration_options_precede_resolution(
    kind, operation, resolver_arg, opts, validate, monkeypatch,
) -> None:
    """ID: SPATIAL_HARD_127H_002_configuration_options_precede_resolution."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        graph.get_or_create_frame("sensor", parent=world)
        source = _configuration_path_source(kind)
        before = source.as_dataset(copy="deep")
        probe = forbid_path_resolution(monkeypatch, graph)
        with pytest.raises(TypeError) as exc_info:
            _configuration_path_call(
                source,
                operation,
                "sensor",
                opts=opts,
                validate=validate,
                **{resolver_arg: probe},
            )
    assert str(exc_info.value) == f"spatial.{kind}.{operation}: opts must be PathSolveOptions or None."
    assert probe.events == []
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize(("kind", "operation", "resolver_arg"), _CONFIGURATION_PATH_APIS)
@pytest.mark.parametrize("identity", [False, True])
def test_spatial_core_127h_002_configuration_options_preserve_results(kind, operation, resolver_arg, identity) -> None:
    """ID: SPATIAL_CORE_127H_002_configuration_options_preserve_results."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        graph.get_or_create_frame("sensor", parent=world)
    source = _configuration_path_source(kind)
    rotation = _rotation_from_quat(np.asarray([_quat("z", 20.0)]))
    edge = rotation if resolver_arg == "edge_rotation_fn" else Pose.from_components(
        rotation, _position_from_xyz(np.asarray([[0.5, 0.0, 0.0]])), validate=True,
    )
    calls = []

    def resolver(child, parent):
        calls.append((child.id, parent.id))
        return edge

    dst = "sensor" if identity else "world"
    kwargs = {resolver_arg: resolver, "validate": False}
    with graph:
        expected = _configuration_path_call(source, operation, dst, **kwargs)
        for options in (None, PathSolveOptions()):
            out = _configuration_path_call(source, operation, dst, opts=options, **kwargs)
            xr.testing.assert_identical(out.as_dataset(copy="none"), expected.as_dataset(copy="none"))
    with FrameGraph():
        for options in (PathSolveOptions(graph=graph), FalseyPathSolveOptions(graph=graph)):
            out = _configuration_path_call(source, operation, dst, opts=options, **kwargs)
            xr.testing.assert_identical(out.as_dataset(copy="none"), expected.as_dataset(copy="none"))
    if identity:
        assert calls == []
    else:
        assert calls and set(calls) == {("sensor", "world")}


@pytest.mark.parametrize("kind", ["position", "rotation", "pose"])
def test_spatial_core_127h_003_configuration_identity_keeps_field_checks_deferred(kind) -> None:
    """ID: SPATIAL_CORE_127H_003_configuration_identity_keeps_field_checks_deferred."""
    source = _configuration_path_source(kind)
    opts = PathSolveOptions(strict=False, kinematics_support=object())
    probe = ResolutionProbe()
    resolver_arg = "edge_pose_fn" if kind == "pose" else "edge_rotation_fn"
    expected = source.express_in("sensor", **{resolver_arg: probe})
    out = source.express_in("sensor", opts=opts, validate=False, **{resolver_arg: probe})
    assert probe.events == []
    xr.testing.assert_identical(out.as_dataset(copy="none"), expected.as_dataset(copy="none"))


def test_spatial_core_075_position_to_frame_identity_parent_eq_dst_deterministic() -> None:
    """ID: SPATIAL_CORE_075_position_to_frame_identity_parent_eq_dst_deterministic."""
    graph = FrameGraph()
    with graph:
        _ = graph.get_or_create_frame("world")
        _ = graph.get_or_create_frame("tool")
    position = frame_retag(
        _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="tool",
        validate=True,
    )
    calls = {"count": 0}

    def resolver(*_args):
        calls["count"] += 1
        raise AssertionError("identity path should not call resolver")

    out = position.to_frame("world", edge_pose_fn=resolver, opts=PathSolveOptions(graph=graph), validate=True)
    assert calls["count"] == 0
    assert get_frames(out.as_dataset(copy="none")) == ("world", "tool")
    assert get_position_rep(out.as_dataset(copy="none"), owner="test") == "cart"
    np.testing.assert_allclose(out.as_dataset(copy="none")["position"].values, position.as_dataset(copy="none")["position"].values, atol=1e-9, rtol=0.0)


def test_spatial_core_076_position_to_frame_chain_matches_c1_plus_apply_reference() -> None:
    """ID: SPATIAL_CORE_076_position_to_frame_chain_matches_c1_plus_apply_reference."""
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
        resolver = lambda child, parent: edge_map[(child.id, parent.id)]
        position = frame_retag(
            _position_from_xyz(np.asarray([[0.5, 0.5, 0.5]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        opts = PathSolveOptions(graph=graph)
        out = position.to_frame("world", edge_pose_fn=resolver, opts=opts, validate=True)
        reference_pose = solve_pose_path_transform("sensor", "world", edge_pose_fn=resolver, opts=opts)
        reference = reference_pose.apply(position, validate=True)
    assert get_frames(out.as_dataset(copy="none")) == ("world", "probe")
    np.testing.assert_allclose(out.as_dataset(copy="none")["position"].values, reference.as_dataset(copy="none")["position"].values, atol=1e-6, rtol=0.0)


def test_spatial_core_077_rotation_class_solve_path_transform_wrapper_parity_with_c1_function_api() -> None:
    """ID: SPATIAL_CORE_077_rotation_class_solve_path_transform_wrapper_parity_with_c1_function_api."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_map = {
            ("sensor", "body"): _rotation_from_quat(np.asarray([_quat("z", 20.0)], dtype=float)),
            ("body", "world"): _rotation_from_quat(np.asarray([_quat("x", 15.0)], dtype=float)),
        }
        resolver = lambda child, parent: edge_map[(child.id, parent.id)]
        opts = PathSolveOptions(graph=graph)
        out_class = Rotation.solve_path_transform(sensor, world, edge_rotation_fn=resolver, opts=opts, validate=True)
        out_func = solve_rotation_path_transform(sensor, world, edge_rotation_fn=resolver, opts=opts)
    assert get_rotation_rep(out_class.as_dataset(copy="none"), owner="test") == "quat"
    assert get_frames(out_class.as_dataset(copy="none")) == get_frames(out_func.as_dataset(copy="none"))
    np.testing.assert_allclose(
        out_class.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values,
        out_func.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values,
        atol=1e-6,
        rtol=0.0,
    )


def test_spatial_core_078_pose_class_solve_path_transform_wrapper_parity_with_c1_function_api() -> None:
    """ID: SPATIAL_CORE_078_pose_class_solve_path_transform_wrapper_parity_with_c1_function_api."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_map = {
            ("sensor", "body"): _pose_from_translation_and_quat(
                np.asarray([[1.0, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("z", 10.0)], dtype=float),
            ),
            ("body", "world"): _pose_from_translation_and_quat(
                np.asarray([[0.0, 1.0, 0.0]], dtype=float),
                np.asarray([_quat("x", 20.0)], dtype=float),
            ),
        }
        resolver = lambda child, parent: edge_map[(child.id, parent.id)]
        opts = PathSolveOptions(graph=graph)
        out_class = Pose.solve_path_transform(sensor, world, edge_pose_fn=resolver, opts=opts, validate=True)
        out_func = solve_pose_path_transform(sensor, world, edge_pose_fn=resolver, opts=opts)
    assert get_pose_rep(out_class.as_dataset(copy="none"), owner="test") == "components"
    assert get_frames(out_class.as_dataset(copy="none")) == get_frames(out_func.as_dataset(copy="none"))
    np.testing.assert_allclose(
        out_class.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].values,
        out_func.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].values,
        atol=1e-6,
        rtol=0.0,
    )


def test_spatial_core_079_position_to_frame_accepts_frame_and_string_dst_with_explicit_graph_policy() -> None:
    """ID: SPATIAL_CORE_079_position_to_frame_accepts_frame_and_string_dst_with_explicit_graph_policy."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        position = frame_retag(
            _position_from_xyz(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        edge_map = {
            ("sensor", "body"): _pose_from_translation_and_quat(
                np.asarray([[0.0, 0.0, 1.0]], dtype=float),
                np.asarray([_quat("z", 5.0)], dtype=float),
            ),
            ("body", "world"): _pose_from_translation_and_quat(
                np.asarray([[0.0, 1.0, 0.0]], dtype=float),
                np.asarray([_quat("x", 6.0)], dtype=float),
            ),
        }
        resolver = lambda child, parent: edge_map[(child.id, parent.id)]
        opts = PathSolveOptions(graph=graph)
        out_frame = position.to_frame(world, edge_pose_fn=resolver, opts=opts, validate=True)
        out_str = position.to_frame("world", edge_pose_fn=resolver, opts=opts, validate=True)
    np.testing.assert_allclose(out_frame.as_dataset(copy="none")["position"].values, out_str.as_dataset(copy="none")["position"].values, atol=1e-6, rtol=0.0)
    assert get_frames(out_frame.as_dataset(copy="none")) == ("world", "probe")
    assert get_frames(out_str.as_dataset(copy="none")) == ("world", "probe")


def test_bcast_hard_041_path_solve_and_frame_api_behavior_unchanged() -> None:
    """ID: BCAST_HARD_041_path_solve_and_frame_api_behavior_unchanged."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        position = frame_retag(
            _position_from_xyz(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        edge_sb = _pose_from_translation_and_quat(
            np.asarray([[0.0, 0.0, 1.0]], dtype=float),
            np.asarray([_quat("z", 5.0)], dtype=float),
        )
        edge_bw = _pose_from_translation_and_quat(
            np.asarray([[0.0, 1.0, 0.0]], dtype=float),
            np.asarray([_quat("x", 6.0)], dtype=float),
        )
        edge_plain = {
            ("sensor", "body"): edge_sb,
            ("body", "world"): edge_bw,
        }
        edge_intent = {
            ("sensor", "body"): edge_sb.a(on="sequence").b(),
            ("body", "world"): edge_bw.a(on="sequence").b(),
        }
        opts = PathSolveOptions(graph=graph)
        out_plain = position.to_frame(
            "world",
            edge_pose_fn=lambda child, parent: edge_plain[(child.id, parent.id)],
            opts=opts,
            validate=True,
        )
        out_intent = position.to_frame(
            "world",
            edge_pose_fn=lambda child, parent: edge_intent[(child.id, parent.id)],
            opts=opts,
            validate=True,
        )
    np.testing.assert_allclose(out_plain.as_dataset(copy="none")["position"].values, out_intent.as_dataset(copy="none")["position"].values, atol=1e-6, rtol=0.0)
    assert get_frames(out_plain.as_dataset(copy="none")) == get_frames(out_intent.as_dataset(copy="none"))


def test_spatial_hard_087_position_to_frame_rejects_unframed_input_without_parent_fail_closed() -> None:
    """ID: SPATIAL_HARD_087_position_to_frame_rejects_unframed_input_without_parent_fail_closed."""
    position = _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]], dtype=float))
    with pytest.raises(ValueError, match="spatial.position.to_frame"):
        position.to_frame("world", edge_pose_fn=lambda *_: None)


def test_spatial_hard_088_position_to_frame_rejects_cross_graph_or_unregistered_dst_fail_closed() -> None:
    """ID: SPATIAL_HARD_088_position_to_frame_rejects_cross_graph_or_unregistered_dst_fail_closed."""
    g1 = FrameGraph()
    g2 = FrameGraph()
    with g1:
        world = g1.get_or_create_frame("world")
        sensor = g1.get_or_create_frame("sensor", parent=world)
        position = frame_retag(_position_from_xyz(np.asarray([[0.0, 0.0, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True)
    with g2:
        map_frame = g2.get_or_create_frame("map")
    with pytest.raises(ValueError, match="different FrameGraph"):
        position.to_frame(map_frame, edge_pose_fn=lambda *_: _pose_from_translation_and_quat(np.zeros((1, 3)), np.asarray([_quat("z", 1.0)])), opts=PathSolveOptions(graph=g1))
    with pytest.raises(ValueError, match="not registered in graph"):
        position.to_frame("missing", edge_pose_fn=lambda *_: _pose_from_translation_and_quat(np.zeros((1, 3)), np.asarray([_quat("z", 1.0)])), opts=PathSolveOptions(graph=g1))


def test_spatial_hard_089_position_to_frame_rejects_missing_or_noncoercible_edge_pose_payload_fail_closed() -> None:
    """ID: SPATIAL_HARD_089_position_to_frame_rejects_missing_or_noncoercible_edge_pose_payload_fail_closed."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
        position = frame_retag(_position_from_xyz(np.asarray([[1.0, 0.0, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True)
        opts = PathSolveOptions(graph=graph)
        with pytest.raises(ValueError, match="spatial.position.to_frame"):
            position.to_frame("world", edge_pose_fn=lambda *_: None, opts=opts)
        with pytest.raises(ValueError, match="Pose-coercible payload"):
            position.to_frame("world", edge_pose_fn=lambda *_: object(), opts=opts)


def test_spatial_hard_090_position_to_frame_public_boundary_no_raw_runtime_exception_leakage() -> None:
    """ID: SPATIAL_HARD_090_position_to_frame_public_boundary_no_raw_runtime_exception_leakage."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
        position = frame_retag(_position_from_xyz(np.asarray([[1.0, 0.0, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True)
        opts = PathSolveOptions(graph=graph)
        with pytest.raises(TypeError, match="spatial.position.to_frame"):
            position.to_frame(123, edge_pose_fn=lambda *_: None, opts=opts)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="spatial.position.to_frame"):
            position.to_frame("world", edge_pose_fn=123, opts=opts)  # type: ignore[arg-type]


def test_spatial_hard_091_rotation_pose_class_wrapper_public_boundary_no_raw_runtime_exception_leakage() -> None:
    """ID: SPATIAL_HARD_091_rotation_pose_class_wrapper_public_boundary_no_raw_runtime_exception_leakage."""
    with pytest.raises(TypeError, match="spatial.rotation.solve_path_transform"):
        Rotation.solve_path_transform(123, "world", edge_rotation_fn=lambda *_: None)  # type: ignore[arg-type]
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    sensor = graph.get_or_create_frame("sensor", parent=world)
    with pytest.raises(TypeError, match="spatial.pose.solve_path_transform"):
        Pose.solve_path_transform(sensor, world, graph=graph, edge_pose_fn=123)  # type: ignore[arg-type]


def test_spatial_hard_092_position_to_frame_dask_lazy_kernel_failure_preserves_c2_owner_context() -> None:
    """ID: SPATIAL_HARD_092_position_to_frame_dask_lazy_kernel_failure_preserves_c2_owner_context."""
    dask_array = pytest.importorskip("dask.array")
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
        position_values = dask_array.from_array(np.asarray([[1.0, 2.0, 3.0]], dtype=float), chunks=(1, 3))
        position = frame_retag(_position_from_xyz(position_values), parent="sensor", child="probe", validate=True)
        translation = _position_from_xyz(
            dask_array.from_array(np.asarray([[0.0, 0.0, 0.0]], dtype=float), chunks=(1, 3))
        )
        invalid_quat = _rotation_from_quat(
            dask_array.from_array(np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=float), chunks=(1, 4))
        )
        edge_pose = frame_retag(Pose.from_components(invalid_quat, translation, validate=True), parent="world", child="sensor", validate=True)
        out = position.to_frame(
            "world",
            edge_pose_fn=lambda *_: edge_pose,
            opts=PathSolveOptions(graph=graph),
            validate=True,
        )
    with pytest.raises(ValueError) as exc_info:
        out.as_dataset(copy="none")["position"].compute()
    message = str(exc_info.value)
    assert "spatial.position.to_frame" in message
    assert "spatial.pose.apply" not in message
    assert "spatial.pose.kernel" not in message


def test_spatial_core_129b_011_position_identity_skips_unused_resolver() -> None:
    """ID: SPATIAL_CORE_129B_011_position_identity_skips_unused_resolver."""
    position = frame_retag(
        _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="tool",
        validate=True,
    )
    result = position.to_frame("world", edge_pose_fn=123, validate=True)  # type: ignore[arg-type]
    xr.testing.assert_identical(
        result.as_dataset(copy="none"),
        position.as_dataset(copy="none"),
    )


def test_spatial_core_129b_012_position_identity_skips_strict_policy() -> None:
    """ID: SPATIAL_CORE_129B_012_position_identity_skips_strict_policy."""
    position = frame_retag(
        _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="tool",
        validate=True,
    )
    result = position.to_frame(
        "world",
        edge_pose_fn=lambda *_: None,
        opts=PathSolveOptions(strict=False),
        validate=True,
    )
    xr.testing.assert_identical(
        result.as_dataset(copy="none"),
        position.as_dataset(copy="none"),
    )


def test_spatial_hard_095_position_to_frame_identity_request_rejects_explicit_graph_mismatch_even_when_dst_id_matches_parent() -> None:
    """ID: SPATIAL_HARD_095_position_to_frame_identity_request_rejects_explicit_graph_mismatch_even_when_dst_id_matches_parent."""
    g1 = FrameGraph()
    g2 = FrameGraph()
    with g1:
        _ = g1.get_or_create_frame("world")
    with g2:
        world_other_graph = g2.get_or_create_frame("world")
    position = frame_retag(
        _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="tool",
        validate=True,
    )
    with pytest.raises(ValueError) as exc_info:
        position.to_frame(
            world_other_graph,
            edge_pose_fn=lambda *_: None,
            opts=PathSolveOptions(graph=g1),
            validate=True,
        )
    message = str(exc_info.value)
    assert "spatial.position.to_frame" in message
    assert "different FrameGraph" in message


def test_spatial_core_c5_001_express_in_changes_representation_without_changing_relation() -> None:
    """ID: SPATIAL_CORE_C5_001_express_in_changes_representation_without_changing_relation."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        edge_map = {
            ("sensor", "body"): _rotation_from_quat(np.asarray([_quat("z", 20.0)], dtype=float)),
            ("body", "world"): _rotation_from_quat(np.asarray([_quat("x", 10.0)], dtype=float)),
        }
        position = frame_retag(
            _position_from_xyz(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        out = position.express_in(
            "world",
            edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)],
            opts=PathSolveOptions(graph=graph),
            validate=True,
        )
        basis = solve_rotation_path_transform(
            "sensor",
            "world",
            edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)],
            opts=PathSolveOptions(graph=graph),
        )
    assert get_frames(out.as_dataset(copy="none")) == ("sensor", "probe")
    assert get_expressed_in(out.as_dataset(copy="none"), owner="test") == "world"
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    expected = basis_m @ position.as_dataset(copy="none")["position"].values[0]
    np.testing.assert_allclose(out.as_dataset(copy="none")["position"].values[0], expected, atol=1e-6, rtol=0.0)


def test_spatial_hard_c5_006_express_in_identity_short_circuit_uses_resolved_expressed_in_not_parent() -> None:
    """ID: SPATIAL_HARD_C5_006_express_in_identity_short_circuit_uses_resolved_expressed_in_not_parent."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
        map_frame = graph.get_or_create_frame("map", parent=world)
        edge_map = {
            ("map", "world"): _rotation_from_quat(np.asarray([_quat("z", 5.0)], dtype=float)),
            ("world", "map"): _rotation_from_quat(np.asarray([_quat("z", 5.0)], dtype=float)),
            ("sensor", "world"): _rotation_from_quat(np.asarray([_quat("x", 12.0)], dtype=float)),
        }
        base = frame_retag(
            _position_from_xyz(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        source = Position(
            set_expressed_in(
                base.as_dataset(copy="none"),
                expressed_in="map",
                validate=False,
                owner="test",
            )
        )

    calls = {"count": 0}

    def resolver(child, parent):
        calls["count"] += 1
        return edge_map[(child.id, parent.id)]

    out_identity = source.express_in(
        "map",
        edge_rotation_fn=resolver,
        opts=PathSolveOptions(graph=graph),
        validate=True,
    )
    assert calls["count"] == 0
    assert get_expressed_in(out_identity.as_dataset(copy="none"), owner="test") == "map"
    np.testing.assert_allclose(
        out_identity.as_dataset(copy="none")["position"].values,
        source.as_dataset(copy="none")["position"].values,
        atol=1e-9,
        rtol=0.0,
    )

    out_non_identity = source.express_in(
        "sensor",
        edge_rotation_fn=resolver,
        opts=PathSolveOptions(graph=graph),
        validate=True,
    )
    assert calls["count"] > 0
    assert get_expressed_in(out_non_identity.as_dataset(copy="none"), owner="test") == "sensor"


def test_spatial_core_c5_003_to_frame_and_express_in_semantics_are_distinct() -> None:
    """ID: SPATIAL_CORE_C5_003_to_frame_and_express_in_semantics_are_distinct."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        rotation_edges = {
            ("sensor", "body"): _rotation_from_quat(np.asarray([_quat("z", 25.0)], dtype=float)),
            ("body", "world"): _rotation_from_quat(np.asarray([_quat("x", 15.0)], dtype=float)),
        }
        pose_edges = {
            ("sensor", "body"): _pose_from_translation_and_quat(
                np.asarray([[0.2, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("z", 25.0)], dtype=float),
            ),
            ("body", "world"): _pose_from_translation_and_quat(
                np.asarray([[0.0, 0.3, 0.0]], dtype=float),
                np.asarray([_quat("x", 15.0)], dtype=float),
            ),
        }
        position = frame_retag(
            _position_from_xyz(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        opts = PathSolveOptions(graph=graph)
        rel_changed = position.to_frame(
            "world",
            edge_pose_fn=lambda child, parent: pose_edges[(child.id, parent.id)],
            opts=opts,
            validate=True,
        )
        rep_changed = position.express_in(
            "world",
            edge_rotation_fn=lambda child, parent: rotation_edges[(child.id, parent.id)],
            opts=opts,
            validate=True,
        )
    assert get_frames(rel_changed.as_dataset(copy="none")) == ("world", "probe")
    assert get_frames(rep_changed.as_dataset(copy="none")) == ("sensor", "probe")
    assert get_expressed_in(rep_changed.as_dataset(copy="none"), owner="test") == "world"


def test_spatial_core_c5_004_configuration_express_in_supports_third_frame_representation_with_path_support() -> None:
    """ID: SPATIAL_CORE_C5_004_configuration_express_in_supports_third_frame_representation_with_path_support."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        map_frame = graph.get_or_create_frame("map", parent=world)
        body = graph.get_or_create_frame("body", parent=map_frame)
        camera = graph.get_or_create_frame("camera", parent=body)
        edge_map = {
            ("camera", "body"): _rotation_from_quat(np.asarray([_quat("z", 5.0)], dtype=float)),
            ("body", "map"): _rotation_from_quat(np.asarray([_quat("x", 10.0)], dtype=float)),
            ("map", "world"): _rotation_from_quat(np.asarray([_quat("y", 15.0)], dtype=float)),
        }
        rotation = frame_retag(
            _rotation_from_quat(np.asarray([_quat("z", 35.0)], dtype=float)),
            parent="camera",
            child="sensor",
            validate=True,
        )
        out = rotation.express_in(
            "world",
            edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)],
            opts=PathSolveOptions(graph=graph),
            validate=True,
        )
        basis = solve_rotation_path_transform(
            "camera",
            "world",
            edge_rotation_fn=lambda child, parent: edge_map[(child.id, parent.id)],
            opts=PathSolveOptions(graph=graph),
        )
    assert get_frames(out.as_dataset(copy="none")) == ("camera", "sensor")
    assert get_expressed_in(out.as_dataset(copy="none"), owner="test") == "world"
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    src_m = rotation.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    expected = basis_m.T @ src_m @ basis_m
    out_m = out.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    np.testing.assert_allclose(out_m, expected, atol=1e-6, rtol=0.0)


def test_spatial_core_c5_006_pose_express_in_rotates_translation_without_origin_offset() -> None:
    """ID: SPATIAL_CORE_C5_006_pose_express_in_rotates_translation_without_origin_offset."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        camera = graph.get_or_create_frame("camera", parent=body)
        edge_map = {
            ("camera", "body"): _pose_from_translation_and_quat(
                np.asarray([[0.2, -0.1, 0.0]], dtype=float),
                np.asarray([_quat("z", 10.0)], dtype=float),
            ),
            ("body", "world"): _pose_from_translation_and_quat(
                np.asarray([[0.0, 0.4, -0.2]], dtype=float),
                np.asarray([_quat("x", 25.0)], dtype=float),
            ),
        }
        pose = frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.4, -0.3, 0.1]], dtype=float),
                np.asarray([_quat("y", 35.0)], dtype=float),
            ),
            parent="camera",
            child="tool",
            validate=True,
        )
        opts = PathSolveOptions(graph=graph)
        out = pose.express_in(
            "world",
            edge_pose_fn=lambda child, parent: edge_map[(child.id, parent.id)],
            opts=opts,
            validate=True,
        )
        basis = solve_pose_path_transform(
            "camera",
            "world",
            edge_pose_fn=lambda child, parent: edge_map[(child.id, parent.id)],
            opts=opts,
        )
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].values[0]
    src_m = pose.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].values[0]
    basis_r = basis_m[:3, :3]
    src_r = src_m[:3, :3]
    src_t = src_m[:3, 3]
    expected = np.eye(4, dtype=float)
    expected[:3, :3] = basis_r.T @ src_r @ basis_r
    expected[:3, 3] = basis_r @ src_t
    out_m = out.as_matrix(validate=True).as_dataset(copy="none")["pose_matrix"].values[0]
    np.testing.assert_allclose(out_m, expected, atol=1e-6, rtol=0.0)
    assert get_frames(out.as_dataset(copy="none")) == ("camera", "tool")
    assert get_expressed_in(out.as_dataset(copy="none"), owner="test") == "world"


def test_spatial_core_c8_001_frame_aware_ops_default_to_sequence_alignment_with_exact_join() -> None:
    """ID: SPATIAL_CORE_C8_001_frame_aware_ops_default_to_sequence_alignment_with_exact_join."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
        source = frame_retag(
            _position_from_xyz(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        source = _with_sample_and_param(source, sample=[0], param_name="tau", param=[10.0])
        edge_pose = frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.5, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("z", 90.0)], dtype=float),
            ),
            parent="world",
            child="sensor",
            validate=True,
        )
        edge_pose = _with_sample_and_param(edge_pose, sample=[1], param_name="tau", param=[10.0])
        opts = PathSolveOptions(graph=graph)
        with pytest.raises(ValueError, match="spatial.position.to_frame"):
            source.to_frame("world", edge_pose_fn=lambda *_: edge_pose, opts=opts, validate=True)


def test_spatial_core_c8_002_frame_aware_ops_support_explicit_param_alignment_when_valid() -> None:
    """ID: SPATIAL_CORE_C8_002_frame_aware_ops_support_explicit_param_alignment_when_valid."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
        source = frame_retag(
            _position_from_xyz(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        source = _with_sample_and_param(source, sample=[0], param_name="tau", param=[10.0])
        edge_pose = frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.5, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("z", 90.0)], dtype=float),
            ),
            parent="world",
            child="sensor",
            validate=True,
        )
        edge_pose = _with_sample_and_param(edge_pose, sample=[1], param_name="tau", param=[10.0])
        opts = PathSolveOptions(graph=graph)
        out = source.a(on="param", sequence_join=None).to_frame(
            "world",
            edge_pose_fn=lambda *_: edge_pose,
            opts=opts,
            validate=True,
        )
    np.testing.assert_allclose(out.as_dataset(copy="none")["position"].values[0], np.asarray([0.5, 1.0, 0.0]), atol=1e-6, rtol=0.0)
    assert get_frames(out.as_dataset(copy="none")) == ("world", "probe")


def test_spatial_core_c8_003_output_frame_metadata_derived_from_operation_not_alignment_side_effects() -> None:
    """ID: SPATIAL_CORE_C8_003_output_frame_metadata_derived_from_operation_not_alignment_side_effects."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
        source = frame_retag(
            _position_from_xyz(np.asarray([[0.0, 1.0, 0.0]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
        source = _with_sample_and_param(source, sample=[5], param_name="tau", param=[2.0])
        edge_rotation = frame_retag(
            _rotation_from_quat(np.asarray([_quat("x", 45.0)], dtype=float)),
            parent="world",
            child="sensor",
            validate=True,
        )
        edge_rotation = _with_sample_and_param(edge_rotation, sample=[5], param_name="tau", param=[2.0])
        opts = PathSolveOptions(graph=graph)
        out_sequence = source.express_in("world", edge_rotation_fn=lambda *_: edge_rotation, opts=opts, validate=True)
        out_param = source.a(on="param", sequence_join=None).express_in(
            "world",
            edge_rotation_fn=lambda *_: edge_rotation,
            opts=opts,
            validate=True,
        )
    assert get_frames(out_sequence.as_dataset(copy="none")) == ("sensor", "probe")
    assert get_frames(out_param.as_dataset(copy="none")) == ("sensor", "probe")
    assert get_expressed_in(out_sequence.as_dataset(copy="none"), owner="test") == "world"
    assert get_expressed_in(out_param.as_dataset(copy="none"), owner="test") == "world"
    np.testing.assert_allclose(out_sequence.as_dataset(copy="none")["position"].values, out_param.as_dataset(copy="none")["position"].values, atol=1e-6, rtol=0.0)
