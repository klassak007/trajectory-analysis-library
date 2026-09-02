from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation

from tal import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    get_edge_motion_class,
    get_frame_inertial_status,
    KinematicsPathSupportOptions,
    LinearAcceleration,
    LinearVelocity,
    PathSolveOptions,
    Pose,
    Position,
    Rotation,
    set_edge_motion_class,
    set_frame_inertial_status,
    Velocity,
    solve_rotation_path_transform,
)
from tal.spatial.metadata import (
    get_acceleration_rep,
    get_expressed_in,
    get_instantaneous_inertial,
    get_velocity_rep,
    set_expressed_in,
    set_instantaneous_inertial,
)
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames


_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


def _quat(axis: str, degrees: float) -> np.ndarray:
    return SciRotation.from_euler(axis, degrees, degrees=True).as_quat().astype(float)


def _position(values: np.ndarray) -> Position:
    arr = xr.DataArray(
        values,
        dims=("sample", "axis"),
        coords={"sample": list(range(values.shape[0])), "axis": list(_XYZ)},
        name="position",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="position"), sequence_dim="sample", core_dims=("axis",), validate=True)
    return Position(ao)


def _rotation(values: np.ndarray) -> Rotation:
    arr = xr.DataArray(
        values,
        dims=("sample", "quat"),
        coords={"sample": list(range(values.shape[0])), "quat": list(_QUAT)},
        name="rotation",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="rotation"), sequence_dim="sample", core_dims=("quat",), validate=True)
    return Rotation(ao)


def _linear_velocity(values: np.ndarray) -> LinearVelocity:
    arr = xr.DataArray(
        values,
        dims=("sample", "lin_axis"),
        coords={"sample": list(range(values.shape[0])), "lin_axis": list(_XYZ)},
        name="linear_velocity",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="linear_velocity"),
        sequence_dim="sample",
        core_dims=("lin_axis",),
        validate=True,
    )
    return LinearVelocity(ao)


def _angular_velocity(values: np.ndarray) -> AngularVelocity:
    arr = xr.DataArray(
        values,
        dims=("sample", "ang_axis"),
        coords={"sample": list(range(values.shape[0])), "ang_axis": list(_XYZ)},
        name="angular_velocity",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="angular_velocity"),
        sequence_dim="sample",
        core_dims=("ang_axis",),
        validate=True,
    )
    return AngularVelocity(ao)


def _linear_acceleration(values: np.ndarray) -> LinearAcceleration:
    arr = xr.DataArray(
        values,
        dims=("sample", "lin_axis"),
        coords={"sample": list(range(values.shape[0])), "lin_axis": list(_XYZ)},
        name="linear_acceleration",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="linear_acceleration"),
        sequence_dim="sample",
        core_dims=("lin_axis",),
        validate=True,
    )
    return LinearAcceleration(ao)


def _angular_acceleration(values: np.ndarray) -> AngularAcceleration:
    arr = xr.DataArray(
        values,
        dims=("sample", "ang_axis"),
        coords={"sample": list(range(values.shape[0])), "ang_axis": list(_XYZ)},
        name="angular_acceleration",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="angular_acceleration"),
        sequence_dim="sample",
        core_dims=("ang_axis",),
        validate=True,
    )
    return AngularAcceleration(ao)


def _pose_from_translation_and_quat(translation: np.ndarray, quat: np.ndarray) -> Pose:
    return Pose.from_components(_rotation(quat), _position(translation), validate=True)


def _single_var_values(value) -> np.ndarray:
    var_name = next(iter(value.as_dataset(copy="none").data_vars))
    return np.asarray(value.as_dataset(copy="none")[var_name].values, dtype=float)


def _with_sample_and_param(value, *, sample: list[int], param_name: str, param: list[float]):
    ds = value.as_dataset(copy="none").assign_coords(sample=sample, **{param_name: ("sample", param)})
    return value.__class__(ds).set_param_coord(name=param_name, validate=False)


def _build_graph_and_edges() -> tuple[FrameGraph, object, object, PathSolveOptions]:
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        set_frame_inertial_status(world, "inertial")
        set_frame_inertial_status(body, "inertial")
        set_frame_inertial_status(sensor, "inertial")
        set_edge_motion_class(body, "static", world)
        set_edge_motion_class(sensor, "static", body)
        rot_edges = {
            ("sensor", "body"): frame_retag(
                _rotation(np.asarray([_quat("z", 15.0)], dtype=float)),
                parent="body",
                child="sensor",
                validate=True,
            ),
            ("body", "world"): frame_retag(
                _rotation(np.asarray([_quat("x", 20.0)], dtype=float)),
                parent="world",
                child="body",
                validate=True,
            ),
        }
        pose_edges = {
            ("sensor", "body"): frame_retag(
                _pose_from_translation_and_quat(
                    np.asarray([[0.2, 0.0, 0.0]], dtype=float),
                    np.asarray([_quat("z", 15.0)], dtype=float),
                ),
                parent="body",
                child="sensor",
                validate=True,
            ),
            ("body", "world"): frame_retag(
                _pose_from_translation_and_quat(
                    np.asarray([[0.0, 0.3, 0.0]], dtype=float),
                    np.asarray([_quat("x", 20.0)], dtype=float),
                ),
                parent="world",
                child="body",
                validate=True,
            ),
        }
    return (
        graph,
        lambda child, parent: rot_edges[(child.id, parent.id)],
        lambda child, parent: pose_edges[(child.id, parent.id)],
        PathSolveOptions(graph=graph),
    )


def _framed_velocity_family(rep: str = "components") -> Velocity:
    value = Velocity.from_linear_angular(
        frame_retag(_linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True),
        frame_retag(_angular_velocity(np.asarray([[0.0, 0.5, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True),
        validate=True,
    )
    if rep == "vector6":
        value = value.to_rep("vector6", validate=True)
    return value


def _framed_acceleration_family(rep: str = "components") -> Acceleration:
    value = Acceleration.from_linear_angular(
        frame_retag(_linear_acceleration(np.asarray([[0.1, 0.0, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True),
        frame_retag(_angular_acceleration(np.asarray([[0.0, 0.2, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True),
        validate=True,
    )
    if rep == "vector6":
        value = value.to_rep("vector6", validate=True)
    return value


def test_spatial_core_088_kinematics_to_frame_methods_exist_on_all_c3_target_types() -> None:
    """ID: SPATIAL_CORE_088_kinematics_to_frame_methods_exist_on_all_c3_target_types."""
    for cls in (LinearVelocity, AngularVelocity, Velocity, LinearAcceleration, AngularAcceleration, Acceleration):
        assert hasattr(cls, "to_frame")


def test_spatial_core_090_kinematics_to_frame_components_paths_preserve_kind_frames_and_registry_truth() -> None:
    """ID: SPATIAL_CORE_090_kinematics_to_frame_components_paths_preserve_kind_frames_and_registry_truth."""
    graph, rot_fn, pose_fn, opts = _build_graph_and_edges()
    _ = graph
    velocity = _framed_velocity_family(rep="components")
    acceleration = _framed_acceleration_family(rep="components")
    v_out = velocity.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)
    a_out = acceleration.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)
    assert get_frames(v_out.as_dataset(copy="none")) == ("world", "probe")
    assert get_frames(a_out.as_dataset(copy="none")) == ("world", "probe")
    assert get_velocity_rep(v_out.as_dataset(copy="none"), owner="test") == "components"
    assert get_acceleration_rep(a_out.as_dataset(copy="none"), owner="test") == "components"


def test_spatial_core_091_kinematics_to_frame_vector6_paths_preserve_input_rep_and_canonical_ordering() -> None:
    """ID: SPATIAL_CORE_091_kinematics_to_frame_vector6_paths_preserve_input_rep_and_canonical_ordering."""
    _, _, pose_fn, opts = _build_graph_and_edges()
    velocity = _framed_velocity_family(rep="vector6")
    acceleration = _framed_acceleration_family(rep="vector6")
    v_out = velocity.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)
    a_out = acceleration.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)
    assert get_velocity_rep(v_out.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_acceleration_rep(a_out.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_frames(v_out.as_dataset(copy="none")) == ("world", "probe")
    assert get_frames(a_out.as_dataset(copy="none")) == ("world", "probe")


def test_spatial_core_092_kinematics_to_frame_numeric_outputs_match_rotation_reference_for_linear_angular_and_family_paths() -> None:
    """ID: SPATIAL_CORE_092_kinematics_to_frame_numeric_outputs_match_rotation_reference_for_linear_angular_and_family_paths."""
    _, rot_fn, pose_fn, opts = _build_graph_and_edges()
    basis = solve_rotation_path_transform("sensor", "world", edge_rotation_fn=rot_fn, opts=opts)
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]

    lin_src = frame_retag(_linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True)
    ang_src = frame_retag(_angular_velocity(np.asarray([[0.0, 0.5, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True)
    lin_out = lin_src.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)
    ang_out = ang_src.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)

    lin_expected = basis_m @ np.asarray([1.0, 0.0, 0.0], dtype=float)
    ang_expected = basis_m @ np.asarray([0.0, 0.5, 0.0], dtype=float)
    np.testing.assert_allclose(_single_var_values(lin_out)[0], lin_expected, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_single_var_values(ang_out)[0], ang_expected, atol=1e-6, rtol=0.0)

    vel_components = _framed_velocity_family(rep="components")
    vel_vector6 = _framed_velocity_family(rep="vector6")
    vel_components_out = vel_components.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)
    vel_vector6_out = vel_vector6.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)
    np.testing.assert_allclose(
        _single_var_values(vel_components_out.linear(validate=True))[0],
        lin_expected,
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        _single_var_values(vel_components_out.angular(validate=True))[0],
        ang_expected,
        atol=1e-6,
        rtol=0.0,
    )
    vel_vector6_expected = np.concatenate([lin_expected, ang_expected])[None, :]
    np.testing.assert_allclose(_single_var_values(vel_vector6_out.as_vector6(validate=True)), vel_vector6_expected, atol=1e-6, rtol=0.0)


def test_spatial_core_093_kinematics_to_frame_non_identity_uses_expressed_in_source_basis() -> None:
    """ID: SPATIAL_CORE_093_kinematics_to_frame_non_identity_uses_expressed_in_source_basis."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        map_frame = graph.get_or_create_frame("map", parent=world)
        set_frame_inertial_status(world, "inertial")
        set_frame_inertial_status(body, "inertial")
        set_frame_inertial_status(sensor, "inertial")
        set_frame_inertial_status(map_frame, "inertial")
        set_edge_motion_class(body, "static", world)
        set_edge_motion_class(sensor, "static", body)
        set_edge_motion_class(map_frame, "static", world)
    rot_edges = {
        ("sensor", "body"): frame_retag(
            _rotation(np.asarray([_quat("z", 15.0)], dtype=float)),
            parent="body",
            child="sensor",
            validate=True,
        ),
        ("body", "world"): frame_retag(
            _rotation(np.asarray([_quat("x", 20.0)], dtype=float)),
            parent="world",
            child="body",
            validate=True,
        ),
        ("map", "world"): frame_retag(
            _rotation(np.asarray([_quat("y", -35.0)], dtype=float)),
            parent="world",
            child="map",
            validate=True,
        ),
    }
    pose_edges = {
        ("sensor", "body"): frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.0, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("z", 15.0)], dtype=float),
            ),
            parent="body",
            child="sensor",
            validate=True,
        ),
        ("body", "world"): frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.0, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("x", 20.0)], dtype=float),
            ),
            parent="world",
            child="body",
            validate=True,
        ),
        ("map", "world"): frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.0, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("y", -35.0)], dtype=float),
            ),
            parent="world",
            child="map",
            validate=True,
        ),
    }
    rot_fn = lambda child, parent: rot_edges[(child.id, parent.id)]
    pose_fn = lambda child, parent: pose_edges[(child.id, parent.id)]
    opts = PathSolveOptions(graph=graph)

    source = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="sensor",
        child="probe",
        validate=True,
    )
    source = LinearVelocity(
        set_expressed_in(
            source.as_dataset(copy="none"),
            expressed_in="map",
            validate=False,
            owner="test",
        )
    )
    out = source.to_frame("world", edge_pose_fn=pose_fn, opts=opts, validate=True)

    to_dst = solve_rotation_path_transform("sensor", "world", edge_rotation_fn=rot_fn, opts=opts)
    to_parent = solve_rotation_path_transform("map", "sensor", edge_rotation_fn=rot_fn, opts=opts)
    to_dst_m = to_dst.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    to_parent_m = to_parent.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    expected = to_dst_m @ to_parent_m @ np.asarray([1.0, 0.0, 0.0], dtype=float)
    parent_only = to_dst_m @ np.asarray([1.0, 0.0, 0.0], dtype=float)

    np.testing.assert_allclose(_single_var_values(out)[0], expected, atol=1e-6, rtol=0.0)
    assert not np.allclose(_single_var_values(out)[0], parent_only, atol=1e-6, rtol=0.0)
    assert get_frames(out.as_dataset(copy="none")) == ("world", "probe")


def test_spatial_core_094_position_to_frame_semantic_broadcast_static_edge_pose_regression() -> None:
    """ID: SPATIAL_CORE_924_position_to_frame_semantic_broadcast_static_edge_pose_regression."""
    ship_arr = xr.DataArray(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        dims=("sample", "axis"),
        coords={
            "sample": [0, 1, 2],
            "axis": list(_XYZ),
            "time_s": ("sample", [0.0, 1.0, 2.0]),
        },
        name="position",
    )
    drone_arr = xr.DataArray(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [4.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        dims=("sample", "axis"),
        coords={
            "sample": [0, 1, 2, 3, 4],
            "axis": list(_XYZ),
            "time_s": ("sample", [0.0, 0.5, 1.0, 1.5, 2.0]),
        },
        name="position",
    )
    ship_ao = AnalysisObject.from_data(
        ship_arr.to_dataset(name="position"),
        sequence_dim="sample",
        core_dims=("axis",),
        param_coord="time_s",
        validate=True,
    )
    drone_ao = AnalysisObject.from_data(
        drone_arr.to_dataset(name="position"),
        sequence_dim="sample",
        core_dims=("axis",),
        param_coord="time_s",
        validate=True,
    )
    synced_ship = ship_ao.param.interp_like(drone_ao, on="time_s", validate=True)
    ship_pos = frame_retag(Position(synced_ship), parent="world", child="ship", validate=True)
    drone_pos = frame_retag(Position(drone_ao), parent="world", child="drone", validate=True)

    identity_rot = Rotation(
        AnalysisObject.from_data(
            xr.DataArray(
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float),
                dims=("quat",),
                coords={"quat": list(_QUAT)},
                name="rotation",
            ).to_dataset(name="rotation"),
            core_dims=("quat",),
            validate=True,
        )
    )
    ship_edge_pose = Pose.from_components(
        frame_retag(identity_rot, parent="world", child="ship", validate=True),
        ship_pos,
        validate=True,
    )

    with FrameGraph() as graph:
        world = graph.get_or_create_frame("world")
        _ = graph.get_or_create_frame("ship", parent=world)
        opts = PathSolveOptions(graph=graph)
        out = drone_pos.to_frame(
            "ship",
            edge_pose_fn=lambda child, parent: ship_edge_pose,
            opts=opts,
            validate=True,
        )

    assert get_frames(out.as_dataset(copy="none")) == ("ship", "drone")
    expected = drone_pos.as_dataset(copy="none")["position"] - ship_pos.as_dataset(copy="none")["position"]
    np.testing.assert_allclose(out.as_dataset(copy="none")["position"].values, expected.values, atol=1e-6, rtol=0.0)


def test_spatial_core_c5_002_kinematic_express_in_preserves_instantaneous_inertial_role_set() -> None:
    """ID: SPATIAL_CORE_C5_002_kinematic_express_in_preserves_instantaneous_inertial_role_set."""
    _, rot_fn, _, opts = _build_graph_and_edges()
    velocity = _framed_velocity_family(rep="vector6")
    acceleration = _framed_acceleration_family(rep="components")
    velocity_ds = set_instantaneous_inertial(
        velocity.as_dataset(copy="none"),
        instantaneous_inertial={"parent", "child"},
        validate=False,
        owner="test",
    )
    acceleration_ds = set_instantaneous_inertial(
        acceleration.as_dataset(copy="none"),
        instantaneous_inertial={"child"},
        validate=False,
        owner="test",
    )
    velocity = Velocity(velocity_ds)
    acceleration = Acceleration(acceleration_ds)

    v_out = velocity.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)
    a_out = acceleration.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)

    assert get_frames(v_out.as_dataset(copy="none")) == ("sensor", "probe")
    assert get_frames(a_out.as_dataset(copy="none")) == ("sensor", "probe")
    assert get_expressed_in(v_out.as_dataset(copy="none"), owner="test") == "world"
    assert get_expressed_in(a_out.as_dataset(copy="none"), owner="test") == "world"
    assert get_instantaneous_inertial(v_out.as_dataset(copy="none"), owner="test") == frozenset({"parent", "child"})
    assert get_instantaneous_inertial(a_out.as_dataset(copy="none"), owner="test") == frozenset({"child"})
    assert get_velocity_rep(v_out.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_acceleration_rep(a_out.as_dataset(copy="none"), owner="test") == "components"


def test_spatial_core_c5_005_express_in_is_basis_only_for_vector_like_quantities() -> None:
    """ID: SPATIAL_CORE_C5_005_express_in_is_basis_only_for_vector_like_quantities."""
    _, rot_fn, _, opts = _build_graph_and_edges()
    basis = solve_rotation_path_transform("sensor", "world", edge_rotation_fn=rot_fn, opts=opts)
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]

    lin_src = frame_retag(_linear_acceleration(np.asarray([[0.1, 0.0, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True)
    ang_src = frame_retag(_angular_acceleration(np.asarray([[0.0, 0.2, 0.0]], dtype=float)), parent="sensor", child="probe", validate=True)
    lin_out = lin_src.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)
    ang_out = ang_src.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)

    lin_expected = basis_m @ np.asarray([0.1, 0.0, 0.0], dtype=float)
    ang_expected = basis_m @ np.asarray([0.0, 0.2, 0.0], dtype=float)
    np.testing.assert_allclose(_single_var_values(lin_out)[0], lin_expected, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_single_var_values(ang_out)[0], ang_expected, atol=1e-6, rtol=0.0)

    acc_components = _framed_acceleration_family(rep="components")
    acc_vector6 = _framed_acceleration_family(rep="vector6")
    acc_components_out = acc_components.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)
    acc_vector6_out = acc_vector6.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)
    np.testing.assert_allclose(
        _single_var_values(acc_components_out.linear(validate=True))[0],
        lin_expected,
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        _single_var_values(acc_components_out.angular(validate=True))[0],
        ang_expected,
        atol=1e-6,
        rtol=0.0,
    )
    acc_vector6_expected = np.concatenate([lin_expected, ang_expected])[None, :]
    np.testing.assert_allclose(_single_var_values(acc_vector6_out.as_vector6(validate=True)), acc_vector6_expected, atol=1e-6, rtol=0.0)


def test_spatial_core_c5_007_corotating_representation_change_does_not_zero_angular_rate() -> None:
    """ID: SPATIAL_CORE_C5_007_corotating_representation_change_does_not_zero_angular_rate."""
    _, rot_fn, _, opts = _build_graph_and_edges()
    source = frame_retag(
        _angular_velocity(np.asarray([[0.3, 0.0, -0.2]], dtype=float)),
        parent="sensor",
        child="probe",
        validate=True,
    )
    out = source.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)
    basis = solve_rotation_path_transform("sensor", "world", edge_rotation_fn=rot_fn, opts=opts)
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    expected = basis_m @ np.asarray([0.3, 0.0, -0.2], dtype=float)
    np.testing.assert_allclose(_single_var_values(out)[0], expected, atol=1e-6, rtol=0.0)
    assert np.linalg.norm(_single_var_values(out)[0]) > 0.0


def test_spatial_hard_c5_004_express_in_does_not_rewrite_inertial_roles() -> None:
    """ID: SPATIAL_HARD_C5_004_express_in_does_not_rewrite_inertial_roles."""
    _, rot_fn, _, opts = _build_graph_and_edges()
    velocity = _framed_velocity_family(rep="components")
    velocity = Velocity(
        set_instantaneous_inertial(
            velocity.as_dataset(copy="none"),
            instantaneous_inertial={"parent"},
            validate=False,
            owner="test",
        )
    )
    out = velocity.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)
    assert get_instantaneous_inertial(out.as_dataset(copy="none"), owner="test") == frozenset({"parent"})


def test_spatial_hard_c5_005_express_in_cannot_inertialize_representation_role() -> None:
    """ID: SPATIAL_HARD_C5_005_express_in_cannot_inertialize_representation_role."""
    _, rot_fn, _, opts = _build_graph_and_edges()
    velocity = _framed_velocity_family(rep="components")
    out = velocity.express_in("world", edge_rotation_fn=rot_fn, opts=opts, validate=True)
    assert get_instantaneous_inertial(out.as_dataset(copy="none"), owner="test") == frozenset()


def test_spatial_hard_105_kinematics_to_frame_rejects_unframed_source_without_parent_fail_closed() -> None:
    """ID: SPATIAL_HARD_105_kinematics_to_frame_rejects_unframed_source_without_parent_fail_closed."""
    velocity = _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float))
    with pytest.raises(ValueError, match="spatial.linear_velocity.to_frame"):
        velocity.to_frame("world", edge_pose_fn=lambda *_: None)


def test_spatial_hard_106_kinematics_to_frame_rejects_cross_graph_or_unregistered_dst_fail_closed() -> None:
    """ID: SPATIAL_HARD_106_kinematics_to_frame_rejects_cross_graph_or_unregistered_dst_fail_closed."""
    g1 = FrameGraph()
    g2 = FrameGraph()
    with g1:
        world = g1.get_or_create_frame("world")
        sensor = g1.get_or_create_frame("sensor", parent=world)
        source = frame_retag(
            _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            parent="sensor",
            child="probe",
            validate=True,
        )
    with g2:
        map_frame = g2.get_or_create_frame("map")
    with pytest.raises(ValueError, match="different FrameGraph"):
        source.to_frame(
            map_frame,
            edge_pose_fn=lambda *_: _pose_from_translation_and_quat(np.zeros((1, 3)), np.asarray([_quat("z", 1.0)])),
            opts=PathSolveOptions(graph=g1),
        )


def test_spatial_hard_c5_001_express_in_fails_closed_on_missing_path_support() -> None:
    """ID: SPATIAL_HARD_C5_001_express_in_fails_closed_on_missing_path_support."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        _ = graph.get_or_create_frame("sensor", parent=world)
    source = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="sensor",
        child="probe",
        validate=True,
    )
    with pytest.raises(ValueError, match="not registered in graph|no path"):
        source.express_in(
            "missing",
            edge_rotation_fn=lambda *_: _rotation(np.asarray([_quat("z", 1.0)], dtype=float)),
            opts=PathSolveOptions(graph=graph),
        )


def test_spatial_hard_107_kinematics_identity_paths_short_circuit_before_solver() -> None:
    """ID: SPATIAL_HARD_107_kinematics_identity_paths_short_circuit_before_solver."""
    import tal.spatial.path_solve as path_solve

    graph = FrameGraph()
    with graph:
        _ = graph.get_or_create_frame("world")
        _ = graph.get_or_create_frame("map")
    opts = PathSolveOptions(graph=graph)
    source = frame_retag(
        _framed_velocity_family(rep="components"),
        parent="world",
        child="probe",
        validate=True,
    )
    source = Velocity(
        set_expressed_in(
            source.as_dataset(copy="none"),
            expressed_in="map",
            validate=False,
            owner="test",
        )
    )

    original_pose_solver = path_solve._solve_pose_path_transform_with_owner
    original_rot_solver = path_solve._solve_rotation_path_transform_with_owner

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("identity frame requests must short-circuit before path solver")

    path_solve._solve_pose_path_transform_with_owner = _raise_if_called
    path_solve._solve_rotation_path_transform_with_owner = _raise_if_called
    try:
        to_out = source.to_frame("world", edge_pose_fn=lambda *_: None, opts=opts, validate=True)
        expr_out = source.express_in("map", edge_rotation_fn=lambda *_: None, opts=opts, validate=True)
    finally:
        path_solve._solve_pose_path_transform_with_owner = original_pose_solver
        path_solve._solve_rotation_path_transform_with_owner = original_rot_solver

    assert get_frames(to_out.as_dataset(copy="none")) == ("world", "probe")
    assert get_frames(expr_out.as_dataset(copy="none")) == ("world", "probe")
    assert get_expressed_in(to_out.as_dataset(copy="none"), owner="test") == "map"
    assert get_expressed_in(expr_out.as_dataset(copy="none"), owner="test") == "map"


def _build_dynamic_support_case() -> tuple[FrameGraph, object, object, object, object, PathSolveOptions]:
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        map_frame = graph.get_or_create_frame("map", parent=world)
        probe = graph.get_or_create_frame("probe", parent=world)
        set_frame_inertial_status(world, "inertial")
        set_frame_inertial_status(map_frame, "non_inertial")
        set_frame_inertial_status(probe, "inertial")
        set_edge_motion_class(map_frame, "dynamic", world)
        rot_edges = {
            ("map", "world"): frame_retag(
                _rotation(np.asarray([_quat("z", 90.0)], dtype=float)),
                parent="world",
                child="map",
                validate=True,
            )
        }
        pose_edges = {
            ("map", "world"): frame_retag(
                _pose_from_translation_and_quat(
                    np.asarray([[0.0, 0.0, 0.0]], dtype=float),
                    np.asarray([_quat("z", 90.0)], dtype=float),
                ),
                parent="world",
                child="map",
                validate=True,
            )
        }
        edge_velocity = frame_retag(
            _framed_velocity_family(rep="components"),
            parent="world",
            child="map",
            validate=True,
        )
        edge_velocity = Velocity(
            set_expressed_in(edge_velocity.as_dataset(copy="none"), expressed_in="world", validate=False, owner="test")
        )
        lin = edge_velocity.linear(validate=True)
        ang = edge_velocity.angular(validate=True)
        lin_ds = lin.as_dataset(copy="none").copy()
        ang_ds = ang.as_dataset(copy="none").copy()
        lin_name = next(iter(lin_ds.data_vars))
        ang_name = next(iter(ang_ds.data_vars))
        lin_ds[lin_name] = xr.DataArray(
            np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            dims=("sample", "lin_axis"),
            coords={"sample": [0], "lin_axis": list(_XYZ)},
        )
        ang_ds[ang_name] = xr.DataArray(
            np.asarray([[0.0, 0.0, 0.5]], dtype=float),
            dims=("sample", "ang_axis"),
            coords={"sample": [0], "ang_axis": list(_XYZ)},
        )
        edge_velocity = Velocity.from_linear_angular(LinearVelocity(lin_ds), AngularVelocity(ang_ds), validate=True)
        edge_velocity = frame_retag(edge_velocity, parent="world", child="map", validate=True)
        edge_acceleration = frame_retag(
            _framed_acceleration_family(rep="components"),
            parent="world",
            child="map",
            validate=True,
        )
        edge_acceleration = Acceleration(
            set_expressed_in(edge_acceleration.as_dataset(copy="none"), expressed_in="world", validate=False, owner="test")
        )
    support = KinematicsPathSupportOptions(
        edge_velocity_fn=lambda child, parent: edge_velocity if (child.id, parent.id) == ("map", "world") else None,
        edge_acceleration_fn=(
            lambda child, parent: edge_acceleration if (child.id, parent.id) == ("map", "world") else None
        ),
    )
    return (
        graph,
        lambda child, parent: rot_edges[(child.id, parent.id)],
        lambda child, parent: pose_edges[(child.id, parent.id)],
        lambda child, parent: edge_velocity if (child.id, parent.id) == ("map", "world") else None,
        lambda child, parent: edge_acceleration if (child.id, parent.id) == ("map", "world") else None,
        PathSolveOptions(graph=graph, kinematics_support=support),
    )


def _build_c8_param_alignment_case() -> tuple[LinearVelocity, object, object, PathSolveOptions]:
    graph, rot_fn, pose_fn, _, _, _ = _build_dynamic_support_case()
    source = frame_retag(
        _linear_velocity(np.asarray([[2.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    source = _with_sample_and_param(source, sample=[0], param_name="tau", param=[10.0])
    edge_velocity = frame_retag(_framed_velocity_family(rep="components"), parent="world", child="map", validate=True)
    edge_velocity = Velocity(
        set_expressed_in(edge_velocity.as_dataset(copy="none"), expressed_in="world", validate=False, owner="test")
    )
    edge_velocity = _with_sample_and_param(edge_velocity, sample=[1], param_name="tau", param=[10.0])

    def _pose_with_param(child, parent):
        pose = pose_fn(child, parent)
        return _with_sample_and_param(pose, sample=[0], param_name="tau", param=[10.0])

    support = KinematicsPathSupportOptions(
        edge_velocity_fn=lambda child, parent: edge_velocity if (child.id, parent.id) == ("map", "world") else None
    )
    return source, _pose_with_param, rot_fn, PathSolveOptions(graph=graph, kinematics_support=support)


def test_spatial_core_c7_001_kinematics_to_frame_succeeds_with_explicit_supported_path_motion() -> None:
    """ID: SPATIAL_CORE_C7_001_kinematics_to_frame_succeeds_with_explicit_supported_path_motion."""
    _, rot_fn, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = frame_retag(
        _linear_velocity(np.asarray([[2.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    out = source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    basis = solve_rotation_path_transform("world", "map", edge_rotation_fn=rot_fn, opts=opts)
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    expected = basis_m @ np.asarray([1.0, 0.0, 0.0], dtype=float)
    np.testing.assert_allclose(_single_var_values(out)[0], expected, atol=1e-6, rtol=0.0)
    assert get_frames(out.as_dataset(copy="none")) == ("map", "probe")


def test_spatial_core_c7_002_kinematics_to_frame_preserves_relation_representation_and_kind_truthfulness() -> None:
    """ID: SPATIAL_CORE_C7_002_kinematics_to_frame_preserves_relation_representation_and_kind_truthfulness."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = _framed_velocity_family(rep="components")
    source = frame_retag(source, parent="world", child="probe", validate=True)
    source = Velocity(
        set_expressed_in(
            source.as_dataset(copy="none"),
            expressed_in="world",
            validate=False,
            owner="test",
        )
    )
    out = source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    assert get_frames(out.as_dataset(copy="none")) == ("map", "probe")
    assert get_velocity_rep(out.as_dataset(copy="none"), owner="test") == "components"
    assert get_expressed_in(out.as_dataset(copy="none"), owner="test") == "map"


def test_spatial_core_c7_003_velocity_acceleration_vector6_paths_preserve_rep_after_canonical_internal_execution() -> None:
    """ID: SPATIAL_CORE_C7_003_velocity_acceleration_vector6_paths_preserve_rep_after_canonical_internal_execution."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    vel = frame_retag(_framed_velocity_family(rep="vector6"), parent="world", child="probe", validate=True)
    acc = frame_retag(_framed_acceleration_family(rep="vector6"), parent="world", child="probe", validate=True)
    vel_out = vel.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    acc_out = acc.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    assert get_velocity_rep(vel_out.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_acceleration_rep(acc_out.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_frames(vel_out.as_dataset(copy="none")) == ("map", "probe")
    assert get_frames(acc_out.as_dataset(copy="none")) == ("map", "probe")


def test_spatial_core_c7_004_kinematic_same_endpoint_inputs_support_parent_or_child_inertial_roles() -> None:
    """ID: SPATIAL_CORE_C7_004_kinematic_same_endpoint_inputs_support_parent_or_child_inertial_roles."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = frame_retag(
        _angular_velocity(np.asarray([[0.0, 0.0, 0.2]], dtype=float)),
        parent="world",
        child="world",
        validate=True,
    )
    source = AngularVelocity(
        set_instantaneous_inertial(
            source.as_dataset(copy="none"),
            instantaneous_inertial={"parent", "child"},
            validate=False,
            owner="test",
        )
    )
    out = source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    assert get_frames(out.as_dataset(copy="none")) == ("map", "world")


def test_spatial_core_c7_005_to_frame_retargets_parent_while_preserving_child() -> None:
    """ID: SPATIAL_CORE_C7_005_to_frame_retargets_parent_while_preserving_child."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    out = source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    assert get_frames(out.as_dataset(copy="none")) == ("map", "probe")


def test_spatial_core_c7_006_corotating_parent_retarget_can_zero_relative_angular_rate() -> None:
    """ID: SPATIAL_CORE_C7_006_corotating_parent_retarget_can_zero_relative_angular_rate."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = frame_retag(
        _angular_velocity(np.asarray([[0.0, 0.0, 0.5]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    out = source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    np.testing.assert_allclose(_single_var_values(out)[0], np.asarray([0.0, 0.0, 0.0]), atol=1e-6, rtol=0.0)


def test_spatial_core_c7_007_to_frame_dst_frame_with_opts_none_uses_dst_bound_graph() -> None:
    """ID: SPATIAL_CORE_C7_007_to_frame_dst_frame_with_opts_none_uses_dst_bound_graph."""
    target_graph = FrameGraph()
    with target_graph:
        world = target_graph.get_or_create_frame("world")
        sensor = target_graph.get_or_create_frame("sensor", parent=world)
        map_frame = target_graph.get_or_create_frame("map", parent=world)
        set_edge_motion_class(sensor, "static", world)
        set_edge_motion_class(map_frame, "static", world)

    active_graph = FrameGraph()
    with active_graph:
        _ = active_graph.get_or_create_frame("active_world")

    rot_edges = {
        ("sensor", "world"): frame_retag(
            _rotation(np.asarray([_quat("z", 45.0)], dtype=float)),
            parent="world",
            child="sensor",
            validate=True,
        ),
        ("map", "world"): frame_retag(
            _rotation(np.asarray([_quat("x", 30.0)], dtype=float)),
            parent="world",
            child="map",
            validate=True,
        ),
    }
    pose_edges = {
        ("sensor", "world"): frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.0, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("z", 45.0)], dtype=float),
            ),
            parent="world",
            child="sensor",
            validate=True,
        ),
        ("map", "world"): frame_retag(
            _pose_from_translation_and_quat(
                np.asarray([[0.0, 0.0, 0.0]], dtype=float),
                np.asarray([_quat("x", 30.0)], dtype=float),
            ),
            parent="world",
            child="map",
            validate=True,
        ),
    }
    rot_fn = lambda child, parent: rot_edges[(child.id, parent.id)]
    pose_fn = lambda child, parent: pose_edges[(child.id, parent.id)]

    source = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="sensor",
        child="probe",
        validate=True,
    )
    with active_graph:
        out = source.to_frame(map_frame, edge_pose_fn=pose_fn, opts=None, validate=True)

    basis = solve_rotation_path_transform("sensor", map_frame, edge_rotation_fn=rot_fn, opts=None)
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    expected = basis_m @ np.asarray([1.0, 0.0, 0.0], dtype=float)
    np.testing.assert_allclose(_single_var_values(out)[0], expected, atol=1e-6, rtol=0.0)
    assert get_frames(out.as_dataset(copy="none")) == ("map", "probe")


def test_spatial_core_c8_004_kinematic_coupling_supports_explicit_param_primary_key_with_sequence_label_mismatch() -> None:
    """ID: SPATIAL_CORE_C8_004_kinematic_coupling_supports_explicit_param_primary_key_with_sequence_label_mismatch."""
    source, pose_fn, rot_fn, opts = _build_c8_param_alignment_case()
    out = source.a(on="param", sequence_join=None).to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    basis = solve_rotation_path_transform("world", "map", edge_rotation_fn=rot_fn, opts=opts)
    basis_m = basis.as_matrix(validate=True).as_dataset(copy="none")["rotation"].values[0]
    expected = basis_m @ np.asarray([1.0, 0.0, 0.0], dtype=float)
    np.testing.assert_allclose(_single_var_values(out)[0], expected, atol=1e-6, rtol=0.0)
    assert get_frames(out.as_dataset(copy="none")) == ("map", "probe")


def test_spatial_hard_c8_001_frame_validation_runs_before_alignment_and_broadcast() -> None:
    """ID: SPATIAL_HARD_C8_001_frame_validation_runs_before_alignment_and_broadcast."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = _with_sample_and_param(
        _linear_velocity(np.asarray([[2.0, 0.0, 0.0]], dtype=float)),
        sample=[0],
        param_name="tau",
        param=[10.0],
    )
    with pytest.raises(ValueError, match="parent frame id is required"):
        source.a(on="param", sequence_join=None).to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)


def test_spatial_hard_c8_002_param_alignment_rejects_missing_or_ambiguous_primary_key() -> None:
    """ID: SPATIAL_HARD_C8_002_param_alignment_rejects_missing_or_ambiguous_primary_key."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = frame_retag(
        _linear_velocity(np.asarray([[2.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    with pytest.raises(ValueError, match="param"):
        source.a(on="param", sequence_join=None).to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)


def test_spatial_hard_c7_001_kinematics_to_frame_fails_closed_when_path_support_is_insufficient() -> None:
    """ID: SPATIAL_HARD_C7_001_kinematics_to_frame_fails_closed_when_path_support_is_insufficient."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        sensor = graph.get_or_create_frame("sensor", parent=world)
    source = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="sensor",
        child="probe",
        validate=True,
    )
    edge_pose = lambda child, parent: frame_retag(
        _pose_from_translation_and_quat(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([_quat("z", 1.0)])),
        parent=parent.id,
        child=child.id,
        validate=True,
    )
    with pytest.raises(ValueError, match="unknown edge motion class"):
        source.to_frame("world", edge_pose_fn=edge_pose, opts=PathSolveOptions(graph=graph), validate=True)


def test_spatial_hard_c7_002_kinematics_to_frame_does_not_silently_differentiate_path_edges() -> None:
    """ID: SPATIAL_HARD_C7_002_kinematics_to_frame_does_not_silently_differentiate_path_edges."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        map_frame = graph.get_or_create_frame("map", parent=world)
        set_edge_motion_class(map_frame, "dynamic", world)
    source = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    pose_fn = lambda child, parent: frame_retag(
        _pose_from_translation_and_quat(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([_quat("z", 1.0)])),
        parent=parent.id,
        child=child.id,
        validate=True,
    )
    opts = PathSolveOptions(graph=graph, kinematics_support=KinematicsPathSupportOptions())
    with pytest.raises(ValueError, match="edge_velocity_fn is required"):
        source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)


def test_spatial_hard_c7_003_kinematics_to_frame_does_not_infer_hidden_rigid_or_motion_relations() -> None:
    """ID: SPATIAL_HARD_C7_003_kinematics_to_frame_does_not_infer_hidden_rigid_or_motion_relations."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = frame_retag(
        _linear_velocity(np.asarray([[2.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    bad_opts = PathSolveOptions(
        graph=opts.graph,
        kinematics_support=KinematicsPathSupportOptions(edge_velocity_fn=lambda *_: None),
    )
    with pytest.raises(ValueError, match="edge_velocity_fn"):
        source.to_frame("map", edge_pose_fn=pose_fn, opts=bad_opts, validate=True)


def test_spatial_hard_c7_004_to_frame_fails_closed_when_required_inertial_role_support_is_missing() -> None:
    """ID: SPATIAL_HARD_C7_004_to_frame_fails_closed_when_required_inertial_role_support_is_missing."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        map_frame = graph.get_or_create_frame("map", parent=world)
        set_frame_inertial_status(world, "non_inertial")
        set_frame_inertial_status(map_frame, "non_inertial")
        set_edge_motion_class(map_frame, "dynamic", world)
    source = frame_retag(
        _angular_velocity(np.asarray([[0.0, 0.0, 0.5]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    source = AngularVelocity(
        set_instantaneous_inertial(
            source.as_dataset(copy="none"),
            instantaneous_inertial={"parent"},
            validate=False,
            owner="test",
        )
    )
    pose_fn = lambda child, parent: frame_retag(
        _pose_from_translation_and_quat(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([_quat("z", 1.0)])),
        parent=parent.id,
        child=child.id,
        validate=True,
    )
    edge_velocity = frame_retag(_framed_velocity_family(rep="components"), parent="world", child="map", validate=True)
    support = KinematicsPathSupportOptions(edge_velocity_fn=lambda *_: edge_velocity)
    with pytest.raises(ValueError, match="required inertial support"):
        source.to_frame("map", edge_pose_fn=pose_fn, opts=PathSolveOptions(graph=graph, kinematics_support=support), validate=True)


def test_spatial_hard_c7_005_to_frame_does_not_gate_on_expressed_in_inertial_status() -> None:
    """ID: SPATIAL_HARD_C7_005_to_frame_does_not_gate_on_expressed_in_inertial_status."""
    _, _, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = frame_retag(
        _angular_velocity(np.asarray([[0.0, 0.0, 0.5]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    source = AngularVelocity(
        set_instantaneous_inertial(
            set_expressed_in(source.as_dataset(copy="none"), expressed_in="map", validate=False, owner="test"),
            instantaneous_inertial={"parent"},
            validate=False,
            owner="test",
        )
    )
    out = source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    assert get_frames(out.as_dataset(copy="none")) == ("map", "probe")


def test_spatial_hard_c7_006_express_in_style_basis_change_is_not_used_to_satisfy_to_frame_semantics() -> None:
    """ID: SPATIAL_HARD_C7_006_express_in_style_basis_change_is_not_used_to_satisfy_to_frame_semantics."""
    _, rot_fn, pose_fn, _, _, opts = _build_dynamic_support_case()
    source = frame_retag(
        _angular_velocity(np.asarray([[0.0, 0.0, 0.5]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    expr = source.express_in("map", edge_rotation_fn=rot_fn, opts=opts, validate=True)
    rel = source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    assert not np.allclose(_single_var_values(expr), _single_var_values(rel), atol=1e-6, rtol=0.0)


def test_spatial_hard_c7_007_kinematic_coupling_exact_alignment_mismatch_fails_closed() -> None:
    """ID: SPATIAL_HARD_C7_007_kinematic_coupling_exact_alignment_mismatch_fails_closed."""
    graph, _, pose_fn, _, _, _ = _build_dynamic_support_case()
    source = frame_retag(
        _linear_velocity(np.asarray([[2.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    edge_velocity = frame_retag(_framed_velocity_family(rep="components"), parent="world", child="map", validate=True)
    edge_velocity = Velocity(
        set_expressed_in(edge_velocity.as_dataset(copy="none"), expressed_in="world", validate=False, owner="test")
    )
    edge_velocity = Velocity(edge_velocity.as_dataset(copy="none").assign_coords(sample=[1]))
    support = KinematicsPathSupportOptions(
        edge_velocity_fn=lambda child, parent: edge_velocity if (child.id, parent.id) == ("map", "world") else None
    )
    opts = PathSolveOptions(graph=graph, kinematics_support=support)
    with pytest.raises(ValueError, match="strict exact policy|exact"):
        source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)


def test_spatial_hard_c7_008_invalid_kinematics_support_type_fails_with_owner_prefixed_typeerror() -> None:
    """ID: SPATIAL_HARD_C7_008_invalid_kinematics_support_type_fails_with_owner_prefixed_typeerror."""
    graph, _, pose_fn, _, _, _ = _build_dynamic_support_case()
    source = frame_retag(
        _linear_velocity(np.asarray([[2.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="probe",
        validate=True,
    )
    opts = PathSolveOptions(graph=graph, kinematics_support=object())
    with pytest.raises(TypeError, match="opts.kinematics_support must be KinematicsPathSupportOptions or None"):
        source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)


def test_spatial_hard_c7_009_to_frame_explicit_opts_graph_conflict_with_dst_frame_fails_closed() -> None:
    """ID: SPATIAL_HARD_C7_009_to_frame_explicit_opts_graph_conflict_with_dst_frame_fails_closed."""
    graph_a = FrameGraph()
    graph_b = FrameGraph()
    with graph_a:
        world_a = graph_a.get_or_create_frame("world")
        _ = graph_a.get_or_create_frame("sensor", parent=world_a)
    with graph_b:
        world_b = graph_b.get_or_create_frame("world")
        map_b = graph_b.get_or_create_frame("map", parent=world_b)

    source = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="sensor",
        child="probe",
        validate=True,
    )
    opts = PathSolveOptions(graph=graph_a)
    edge_pose = lambda child, parent: frame_retag(
        _pose_from_translation_and_quat(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([_quat("z", 1.0)])),
        parent=parent.id,
        child=child.id,
        validate=True,
    )
    with pytest.raises(ValueError, match="different FrameGraph"):
        source.to_frame(map_b, edge_pose_fn=edge_pose, opts=opts, validate=True)


def test_spatial_hard_c8_003_frame_aware_alignment_does_not_invoke_hidden_interpolation_owners() -> None:
    """ID: SPATIAL_HARD_C8_003_frame_aware_alignment_does_not_invoke_hidden_interpolation_owners."""
    import tal.core.param_engine.map_apply as map_apply
    import tal.core.param_engine.map_build as map_build

    source, pose_fn, _rot_fn, opts = _build_c8_param_alignment_case()
    original_build = map_build.build_param_map
    original_apply = map_apply.apply_param_map

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("frame-aware C8 alignment must not route through interpolation owners")

    map_build.build_param_map = _raise_if_called
    map_apply.apply_param_map = _raise_if_called
    try:
        out = source.a(on="param", sequence_join=None).to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
    finally:
        map_build.build_param_map = original_build
        map_apply.apply_param_map = original_apply
    assert get_frames(out.as_dataset(copy="none")) == ("map", "probe")


def test_spatial_hard_c8_004_kinematic_coupling_default_sequence_primary_key_fails_on_sequence_label_mismatch_even_when_param_labels_match() -> None:
    """ID: SPATIAL_HARD_C8_004_kinematic_coupling_default_sequence_primary_key_fails_on_sequence_label_mismatch_even_when_param_labels_match."""
    source, pose_fn, _rot_fn, opts = _build_c8_param_alignment_case()
    with pytest.raises(ValueError, match="strict exact policy|exact"):
        source.to_frame("map", edge_pose_fn=pose_fn, opts=opts, validate=True)
