from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import tal.spatial.ops.rotation_apply_ops as rotation_apply_ops
from tal import AnalysisObject
from tal.core.schema_errors import SchemaError
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Pose,
    Position,
    Rotation,
    Velocity,
)
from tal.spatial.metadata import get_acceleration_rep, get_position_rep, get_velocity_rep, set_rotation_rep
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


def _position(values: np.ndarray, *, sequence_dim: str = "sample") -> Position:
    arr = xr.DataArray(
        values,
        dims=(sequence_dim, "axis"),
        coords={sequence_dim: list(range(values.shape[0])), "axis": list(_XYZ)},
        name="position",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="position"),
        sequence_dim=sequence_dim,
        core_dims=("axis",),
        validate=True,
    )
    return Position(ao)


def _position_missing_sequence(values: np.ndarray) -> Position:
    arr = xr.DataArray(
        values,
        dims=("axis",),
        coords={"axis": list(_XYZ)},
        name="position",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="position").assign_coords(sample=("sample", [0])),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=False,
    )
    return Position(ao)


def _rotation_quat(values: np.ndarray, *, sequence_dim: str = "sample") -> Rotation:
    arr = xr.DataArray(
        values,
        dims=(sequence_dim, "quat"),
        coords={sequence_dim: list(range(values.shape[0])), "quat": list(_QUAT)},
        name="rotation",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="rotation"),
        sequence_dim=sequence_dim,
        core_dims=("quat",),
        validate=True,
    )
    return Rotation(ao)


def _rotation_matrix(values: np.ndarray, *, sequence_dim: str = "sample") -> Rotation:
    arr = xr.DataArray(
        values,
        dims=(sequence_dim, "row", "col"),
        coords={sequence_dim: list(range(values.shape[0])), "row": list(_XYZ), "col": list(_XYZ)},
        name="rotation",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="rotation"),
        sequence_dim=sequence_dim,
        core_dims=("row", "col"),
        validate=True,
    )
    ds = set_rotation_rep(ao.as_dataset(copy="none"), rep="matrix", validate=False, owner="test")
    return Rotation(ds)


def _linear_velocity(values: np.ndarray) -> LinearVelocity:
    arr = xr.DataArray(
        values,
        dims=("sample", "linear_axis"),
        coords={"sample": list(range(values.shape[0])), "linear_axis": list(_XYZ)},
        name="linear_velocity",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="linear_velocity"),
        sequence_dim="sample",
        core_dims=("linear_axis",),
        validate=True,
    )
    return LinearVelocity(ao)


def _angular_velocity(values: np.ndarray) -> AngularVelocity:
    arr = xr.DataArray(
        values,
        dims=("sample", "angular_axis"),
        coords={"sample": list(range(values.shape[0])), "angular_axis": list(_XYZ)},
        name="angular_velocity",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="angular_velocity"),
        sequence_dim="sample",
        core_dims=("angular_axis",),
        validate=True,
    )
    return AngularVelocity(ao)


def _linear_acceleration(values: np.ndarray) -> LinearAcceleration:
    arr = xr.DataArray(
        values,
        dims=("sample", "linear_axis"),
        coords={"sample": list(range(values.shape[0])), "linear_axis": list(_XYZ)},
        name="linear_acceleration",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="linear_acceleration"),
        sequence_dim="sample",
        core_dims=("linear_axis",),
        validate=True,
    )
    return LinearAcceleration(ao)


def _angular_acceleration(values: np.ndarray) -> AngularAcceleration:
    arr = xr.DataArray(
        values,
        dims=("sample", "angular_axis"),
        coords={"sample": list(range(values.shape[0])), "angular_axis": list(_XYZ)},
        name="angular_acceleration",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="angular_acceleration"),
        sequence_dim="sample",
        core_dims=("angular_axis",),
        validate=True,
    )
    return AngularAcceleration(ao)


def _pose_components(rotation_values: np.ndarray, translation_values: np.ndarray) -> Pose:
    rotation = _rotation_quat(rotation_values)
    translation = _position(translation_values)
    return Pose.from_components(rotation, translation, validate=True)


def _pose_matrix(values: np.ndarray) -> Pose:
    arr = xr.DataArray(
        values,
        dims=("sample", "row", "col"),
        coords={"sample": list(range(values.shape[0])), "row": list(_QUAT), "col": list(_QUAT)},
        name="pose_matrix",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="pose_matrix"),
        sequence_dim="sample",
        core_dims=("row", "col"),
        validate=True,
    )
    return Pose.from_matrix(ao, validate=True)


def _set_bad_frames(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy(deep=True)
    tal = dict(out.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    ext["frames"] = {"parent": "world", "child": "body", "extra": "bad"}
    tal["ext"] = ext
    out.attrs["tal"] = tal
    return out


def test_spatial_core_053_rotation_apply_position_local_same_context_deterministic() -> None:
    """ID: SPATIAL_CORE_053_rotation_apply_position_local_same_context_deterministic."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float))
    pos = _position(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float))
    out = rot.apply(pos, validate=True)

    np.testing.assert_allclose(
        out.as_dataset(copy="none")["position"].values,
        np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        atol=1e-6,
    )


def test_spatial_core_054_pose_apply_position_local_same_context_deterministic() -> None:
    """ID: SPATIAL_CORE_054_pose_apply_position_local_same_context_deterministic."""
    pose = _pose_components(
        np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        np.asarray([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=float),
    )
    target = _position(np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float))
    out = pose.apply(target, validate=True)

    np.testing.assert_allclose(
        out.as_dataset(copy="none")["position"].values,
        np.asarray([[1.0, 3.0, 3.0], [3.0, 1.0, 1.0]], dtype=float),
        atol=1e-6,
    )


def test_spatial_core_055_rotation_apply_linear_and_angular_kinematics_deterministic() -> None:
    """ID: SPATIAL_CORE_055_rotation_apply_linear_and_angular_kinematics_deterministic."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float))
    linear = _linear_velocity(np.asarray([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=float))
    angular = _angular_velocity(np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float))

    linear_out = rot.apply(linear, validate=True)
    angular_out = rot.apply(angular, validate=True)

    np.testing.assert_allclose(
        linear_out.as_dataset(copy="none")["linear_velocity"].values,
        np.asarray([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]], dtype=float),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        angular_out.as_dataset(copy="none")["angular_velocity"].values,
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        atol=1e-6,
    )


def test_spatial_core_056_pose_apply_spatial_velocity_and_acceleration_deterministic() -> None:
    """ID: SPATIAL_CORE_056_pose_apply_spatial_velocity_and_acceleration_deterministic."""
    pose = _pose_components(
        np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        np.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=float),
    )
    rotation = pose.decompose(validate=True)[1]
    vel = Velocity.from_linear_angular(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float)),
        _angular_velocity(np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)),
        validate=True,
    )
    acc = Acceleration.from_linear_angular(
        _linear_acceleration(np.asarray([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)),
        _angular_acceleration(np.asarray([[0.0, 0.2, 0.0], [0.0, 0.0, 0.3]], dtype=float)),
        validate=True,
    )

    vel_out = pose.apply(vel, validate=True)
    acc_out = pose.apply(acc, validate=True)
    vel_expected = rotation.apply(vel, validate=True)
    acc_expected = rotation.apply(acc, validate=True)

    np.testing.assert_allclose(vel_out.linear().as_dataset(copy="none")["linear_velocity"].values, vel_expected.linear().as_dataset(copy="none")["linear_velocity"].values)
    np.testing.assert_allclose(vel_out.angular().as_dataset(copy="none")["angular_velocity"].values, vel_expected.angular().as_dataset(copy="none")["angular_velocity"].values)
    np.testing.assert_allclose(acc_out.linear().as_dataset(copy="none")["linear_acceleration"].values, acc_expected.linear().as_dataset(copy="none")["linear_acceleration"].values)
    np.testing.assert_allclose(acc_out.angular().as_dataset(copy="none")["angular_acceleration"].values, acc_expected.angular().as_dataset(copy="none")["angular_acceleration"].values)


def test_spatial_core_057_apply_mixed_rep_transform_executes_canonical_path_and_preserves_target_policy() -> None:
    """ID: SPATIAL_CORE_057_apply_mixed_rep_transform_executes_canonical_path_and_preserves_target_policy."""
    theta = np.pi / 2.0
    rot_matrix = _rotation_matrix(
        np.asarray(
            [
                [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]],
                np.eye(3, dtype=float),
            ],
            dtype=float,
        )
    )
    pos = _position(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float))
    out_rot = rot_matrix.apply(pos, validate=True)
    assert isinstance(out_rot, Position)
    assert get_position_rep(out_rot.as_dataset(copy="none"), owner="test") == "cart"

    pose_matrix = _pose_matrix(
        np.asarray(
            [
                [[np.cos(theta), -np.sin(theta), 0.0, 1.0], [np.sin(theta), np.cos(theta), 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
                np.eye(4, dtype=float),
            ],
            dtype=float,
        )
    )
    out_pose = pose_matrix.apply(pos, validate=True)
    assert isinstance(out_pose, Position)
    assert get_position_rep(out_pose.as_dataset(copy="none"), owner="test") == "cart"


def test_spatial_core_067_rotation_pose_apply_accept_vector6_spatial_targets_via_canonical_components_path() -> None:
    """ID: SPATIAL_CORE_067_rotation_pose_apply_accept_vector6_spatial_targets_via_canonical_components_path."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float))
    pose = _pose_components(
        np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        np.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=float),
    )
    vel_vec6 = Velocity.from_linear_angular(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float)),
        _angular_velocity(np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)),
        validate=True,
    ).as_vector6(validate=True)
    acc_vec6 = Acceleration.from_linear_angular(
        _linear_acceleration(np.asarray([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)),
        _angular_acceleration(np.asarray([[0.0, 0.2, 0.0], [0.0, 0.0, 0.3]], dtype=float)),
        validate=True,
    ).as_vector6(validate=True)

    rot_vel = rot.apply(vel_vec6, validate=True)
    pose_vel = pose.apply(vel_vec6, validate=True)
    rot_acc = rot.apply(acc_vec6, validate=True)
    pose_acc = pose.apply(acc_vec6, validate=True)

    assert get_velocity_rep(rot_vel.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_velocity_rep(pose_vel.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_acceleration_rep(rot_acc.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_acceleration_rep(pose_acc.as_dataset(copy="none"), owner="test") == "vector6"

    vel_var = next(iter(rot_vel.as_dataset(copy="none").data_vars))
    acc_var = next(iter(rot_acc.as_dataset(copy="none").data_vars))
    assert vel_var == "velocity"
    assert acc_var == "acceleration"
    assert "datavar" not in rot_vel.as_dataset(copy="none").data_vars
    assert "datavar" not in rot_acc.as_dataset(copy="none").data_vars
    rot_vel_expected = rot.apply(vel_vec6.as_components(validate=True), validate=True).as_vector6(validate=True)
    pose_vel_expected = pose.apply(vel_vec6.as_components(validate=True), validate=True).as_vector6(validate=True)
    rot_acc_expected = rot.apply(acc_vec6.as_components(validate=True), validate=True).as_vector6(validate=True)
    pose_acc_expected = pose.apply(acc_vec6.as_components(validate=True), validate=True).as_vector6(validate=True)
    np.testing.assert_allclose(rot_vel.as_dataset(copy="none")[vel_var].values, rot_vel_expected.as_dataset(copy="none")[vel_var].values, atol=1e-6)
    np.testing.assert_allclose(pose_vel.as_dataset(copy="none")[vel_var].values, pose_vel_expected.as_dataset(copy="none")[vel_var].values, atol=1e-6)
    np.testing.assert_allclose(rot_acc.as_dataset(copy="none")[acc_var].values, rot_acc_expected.as_dataset(copy="none")[acc_var].values, atol=1e-6)
    np.testing.assert_allclose(pose_acc.as_dataset(copy="none")[acc_var].values, pose_acc_expected.as_dataset(copy="none")[acc_var].values, atol=1e-6)


def test_spatial_core_058_apply_frame_policy_one_framed_inherits_and_tip_tail_chain_deterministic() -> None:
    """ID: SPATIAL_CORE_058_apply_frame_policy_one_framed_inherits_and_tip_tail_chain_deterministic."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    pos = _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float))

    transform_framed = frame_retag(rot, parent="world", child="body", validate=True)
    one_framed = transform_framed.apply(pos, validate=True)
    assert get_frames(one_framed.as_dataset(copy="none")) == ("world", "body")

    target_framed = frame_retag(pos, parent="body", child="tip", validate=True)
    target_only = rot.apply(target_framed, validate=True)
    assert get_frames(target_only.as_dataset(copy="none")) == ("body", "tip")

    tip_tail = transform_framed.apply(target_framed, validate=True)
    assert get_frames(tip_tail.as_dataset(copy="none")) == ("world", "tip")


def test_spatial_core_059_pose_apply_pose_inverse_roundtrip_identity_for_position() -> None:
    """ID: SPATIAL_CORE_059_pose_apply_pose_inverse_roundtrip_identity_for_position."""
    pose = _pose_components(
        np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
    )
    target = _position(np.asarray([[1.0, -1.0, 2.0], [0.5, 0.2, -0.1]], dtype=float))

    forward = pose.apply(target, validate=True)
    backward = pose.inverse(validate=True).apply(forward, validate=True)
    np.testing.assert_allclose(backward.as_dataset(copy="none")["position"].values, target.as_dataset(copy="none")["position"].values, atol=1e-6)


def test_spatial_core_060_apply_exact_non_core_alignment_behavior_deterministic() -> None:
    """ID: SPATIAL_CORE_060_apply_exact_non_core_alignment_behavior_deterministic."""
    values = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    pos = _position(values)
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float))
    out = rot.apply(pos, validate=True)
    np.testing.assert_allclose(out.as_dataset(copy="none")["position"].values, values)


def test_spatial_hard_056_rotation_apply_rejects_unsupported_target_type_fail_closed() -> None:
    """ID: SPATIAL_HARD_056_rotation_apply_rejects_unsupported_target_type_fail_closed."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    with pytest.raises(TypeError, match="spatial\\.rotation\\.apply"):
        rot.apply(object(), validate=True)


def test_spatial_hard_057_pose_apply_rejects_unsupported_target_type_fail_closed() -> None:
    """ID: SPATIAL_HARD_057_pose_apply_rejects_unsupported_target_type_fail_closed."""
    pose = _pose_components(
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float),
        np.asarray([[0.0, 0.0, 0.0]], dtype=float),
    )
    with pytest.raises(TypeError, match="spatial\\.pose\\.apply"):
        pose.apply(object(), validate=True)


def test_spatial_hard_058_apply_framed_chain_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_058_apply_framed_chain_mismatch_fail_closed."""
    rot = frame_retag(_rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)), parent="world", child="body", validate=True)
    pos = frame_retag(_position(np.asarray([[1.0, 0.0, 0.0]], dtype=float)), parent="map", child="tip", validate=True)
    with pytest.raises(ValueError, match="spatial\\.rotation\\.apply"):
        rot.apply(pos, validate=True)


def test_spatial_hard_059_apply_non_core_dim_name_topology_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_059_apply_non_core_dim_name_topology_mismatch_fail_closed."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float), sequence_dim="sample")
    pos = _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float), sequence_dim="time")
    with pytest.raises(ValueError, match="spatial\\.rotation\\.apply"):
        rot.apply(pos, validate=True)


def test_spatial_hard_060_apply_extra_unmatched_non_core_dims_fail_closed() -> None:
    """ID: SPATIAL_HARD_060_apply_extra_unmatched_non_core_dims_fail_closed."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    pos = _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float))
    ds = pos.as_dataset(copy="none").expand_dims(trial=[0])
    pos_extra = Position._from_unvalidated(ds)
    with pytest.raises(ValueError, match="spatial\\.rotation\\.apply"):
        rot.apply(pos_extra, validate=True)


def test_spatial_hard_061_apply_malformed_frame_schema_fail_closed() -> None:
    """ID: SPATIAL_HARD_061_apply_malformed_frame_schema_fail_closed."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    rot._bind_dataset(_set_bad_frames(rot.as_dataset(copy="none")))
    pos = _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float))
    with pytest.raises(SchemaError, match="tal\\.ext\\.frames\\.extra"):
        rot.apply(pos, validate=True)


def test_spatial_hard_062_apply_fail_closed_on_malformed_target_internal_state() -> None:
    """ID: SPATIAL_HARD_062_apply_fail_closed_on_malformed_target_internal_state."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    malformed_target = _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float))
    malformed_target._bind_dataset(malformed_target.as_dataset(copy="none").isel(axis=slice(0, 2)))
    with pytest.raises(ValueError, match="spatial\\.rotation\\.apply"):
        rot.apply(malformed_target, validate=True)


def test_spatial_hard_063_pose_apply_spatial6_translation_coupling_not_introduced_in_b4() -> None:
    """ID: SPATIAL_HARD_063_pose_apply_spatial6_translation_coupling_not_introduced_in_b4."""
    pose = _pose_components(
        np.asarray([[0.0, 0.0, 0.70710678, 0.70710678]], dtype=float),
        np.asarray([[10.0, 20.0, 30.0]], dtype=float),
    )
    vel = Velocity.from_linear_angular(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        _angular_velocity(np.asarray([[0.0, 1.0, 0.0]], dtype=float)),
        validate=True,
    )
    rot = pose.decompose(validate=True)[1]
    pose_out = pose.apply(vel, validate=True)
    rot_out = rot.apply(vel, validate=True)
    np.testing.assert_allclose(pose_out.linear().as_dataset(copy="none")["linear_velocity"].values, rot_out.linear().as_dataset(copy="none")["linear_velocity"].values)
    np.testing.assert_allclose(pose_out.angular().as_dataset(copy="none")["angular_velocity"].values, rot_out.angular().as_dataset(copy="none")["angular_velocity"].values)


def test_spatial_hard_064_apply_public_boundary_no_raw_runtime_exception_leakage() -> None:
    """ID: SPATIAL_HARD_064_apply_public_boundary_no_raw_runtime_exception_leakage."""
    da = pytest.importorskip("dask.array")
    target = _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float))

    bad_rot_ds = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=float)).as_dataset(copy="none").copy(deep=True)
    bad_rot_ds["rotation"] = xr.DataArray(
        da.from_array(bad_rot_ds["rotation"].values, chunks=(1, 4)),
        dims=bad_rot_ds["rotation"].dims,
        coords=bad_rot_ds["rotation"].coords,
    )
    bad_rot = Rotation._from_unvalidated(bad_rot_ds)
    out_rot = bad_rot.apply(target, validate=True)
    with pytest.raises(ValueError) as rot_exc:
        out_rot.as_dataset(copy="none")["position"].compute()
    assert "spatial.rotation.apply" in str(rot_exc.value)
    assert "spatial.rotation.kernel" not in str(rot_exc.value)
    assert "AttributeError" not in str(rot_exc.value)
    assert "KeyError" not in str(rot_exc.value)

    bad_pose = Pose.from_components(bad_rot, _position(np.asarray([[0.0, 0.0, 0.0]], dtype=float)), validate=False)
    out_pose = bad_pose.apply(target, validate=True)
    with pytest.raises(ValueError) as pose_exc:
        out_pose.as_dataset(copy="none")["position"].compute()
    assert "spatial.pose.apply" in str(pose_exc.value)
    assert "spatial.pose.kernel" not in str(pose_exc.value)
    assert "AttributeError" not in str(pose_exc.value)
    assert "KeyError" not in str(pose_exc.value)


def test_spatial_hard_065_pose_apply_spatial_target_dask_lazy_delegated_rotation_failure_preserves_pose_owner_context() -> None:
    """ID: SPATIAL_HARD_065_pose_apply_spatial_target_dask_lazy_delegated_rotation_failure_preserves_pose_owner_context."""
    da = pytest.importorskip("dask.array")
    bad_rot_ds = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=float)).as_dataset(copy="none").copy(deep=True)
    bad_rot_ds["rotation"] = xr.DataArray(
        da.from_array(bad_rot_ds["rotation"].values, chunks=(1, 4)),
        dims=bad_rot_ds["rotation"].dims,
        coords=bad_rot_ds["rotation"].coords,
    )
    bad_rot = Rotation._from_unvalidated(bad_rot_ds)
    bad_pose = Pose.from_components(bad_rot, _position(np.asarray([[0.0, 0.0, 0.0]], dtype=float)), validate=False)
    targets = [
        Velocity.from_linear_angular(
            _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            _angular_velocity(np.asarray([[0.0, 1.0, 0.0]], dtype=float)),
            validate=True,
        ),
        Acceleration.from_linear_angular(
            _linear_acceleration(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            _angular_acceleration(np.asarray([[0.0, 1.0, 0.0]], dtype=float)),
            validate=True,
        ),
    ]
    for target in targets:
        out = bad_pose.apply(target, validate=True)
        linear_ds = out.linear().as_dataset(copy="none")
        linear_var = next(iter(linear_ds.data_vars))
        with pytest.raises(ValueError) as exc:
            linear_ds[linear_var].compute()
        msg = str(exc.value)
        assert "spatial.pose.apply" in msg
        assert "spatial.rotation.apply" not in msg
        assert "spatial.rotation.kernel" not in msg


def test_spatial_hard_073_vector6_apply_paths_fail_closed_without_raw_runtime_exception_leakage() -> None:
    """ID: SPATIAL_HARD_073_vector6_apply_paths_fail_closed_without_raw_runtime_exception_leakage."""
    da = pytest.importorskip("dask.array")
    bad_rot_ds = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=float)).as_dataset(copy="none").copy(deep=True)
    bad_rot_ds["rotation"] = xr.DataArray(
        da.from_array(bad_rot_ds["rotation"].values, chunks=(1, 4)),
        dims=bad_rot_ds["rotation"].dims,
        coords=bad_rot_ds["rotation"].coords,
    )
    bad_rot = Rotation._from_unvalidated(bad_rot_ds)
    bad_pose = Pose.from_components(bad_rot, _position(np.asarray([[0.0, 0.0, 0.0]], dtype=float)), validate=False)
    targets = [
        Velocity.from_linear_angular(
            _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            _angular_velocity(np.asarray([[0.0, 1.0, 0.0]], dtype=float)),
            validate=True,
        ).as_vector6(validate=True),
        Acceleration.from_linear_angular(
            _linear_acceleration(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
            _angular_acceleration(np.asarray([[0.0, 1.0, 0.0]], dtype=float)),
            validate=True,
        ).as_vector6(validate=True),
    ]

    for target in targets:
        rot_out = bad_rot.apply(target, validate=True)
        rot_var = next(iter(rot_out.as_dataset(copy="none").data_vars))
        with pytest.raises(ValueError) as rot_exc:
            rot_out.as_dataset(copy="none")[rot_var].compute()
        rot_msg = str(rot_exc.value)
        assert "spatial.rotation.apply" in rot_msg
        assert "spatial.rotation.kernel" not in rot_msg
        assert "AttributeError" not in rot_msg
        assert "KeyError" not in rot_msg

        pose_out = bad_pose.apply(target, validate=True)
        pose_var = next(iter(pose_out.as_dataset(copy="none").data_vars))
        with pytest.raises(ValueError) as pose_exc:
            pose_out.as_dataset(copy="none")[pose_var].compute()
        pose_msg = str(pose_exc.value)
        assert "spatial.pose.apply" in pose_msg
        assert "spatial.rotation.apply" not in pose_msg
        assert "spatial.rotation.kernel" not in pose_msg
        assert "AttributeError" not in pose_msg
        assert "KeyError" not in pose_msg


def test_topo_core_007_apply_paths_use_core_topology_touchpoint() -> None:
    """ID: TOPO_CORE_007_apply_paths_use_core_topology_touchpoint."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float))
    pose = _pose_components(
        np.asarray([[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        np.asarray([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=float),
    )
    target = _position(np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float))
    target_transposed = Position(target.as_dataset(copy="none").transpose("axis", "sample"))
    rot_out = rot.apply(target_transposed, validate=True)
    pose_out = pose.apply(target_transposed, validate=True)
    rot_expected = rot.apply(target, validate=True)
    pose_expected = pose.apply(target, validate=True)
    np.testing.assert_allclose(rot_out.as_dataset(copy="none")["position"].values, rot_expected.as_dataset(copy="none")["position"].values, atol=1e-6)
    np.testing.assert_allclose(pose_out.as_dataset(copy="none")["position"].values, pose_expected.as_dataset(copy="none")["position"].values, atol=1e-6)


def test_bcast_core_008_spatial_optin_broadcast_path_uses_shared_core_policy() -> None:
    """ID: BCAST_CORE_008_spatial_optin_broadcast_path_uses_shared_core_policy."""
    rot = _rotation_quat(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
    )
    target = _position_missing_sequence(np.asarray([1.0, 2.0, 3.0], dtype=float))
    out = rot.b().apply(target, validate=True)
    assert out.as_dataset(copy="none")["position"].sizes["sample"] == 2
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["position"].values,
        np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=float),
    )


def test_bcast_core_045_spatial_default_semantic_index_broadcast_enabled_for_approved_families() -> None:
    """ID: BCAST_CORE_045_spatial_default_semantic_index_broadcast_enabled_for_approved_families."""
    rot = _rotation_quat(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
    )
    target = _position_missing_sequence(np.asarray([1.0, 2.0, 3.0], dtype=float))
    out = rot.apply(target, validate=True)
    assert out.as_dataset(copy="none")["position"].sizes["sample"] == 2
    np.testing.assert_allclose(
        out.as_dataset(copy="none")["position"].values,
        np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=float),
    )


def test_bcast_core_047_spatial_apply_family_supports_missing_semantic_dim_materialization_after_frame_check() -> None:
    """ID: BCAST_CORE_047_spatial_apply_family_supports_missing_semantic_dim_materialization_after_frame_check."""
    pose = _pose_components(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float),
    )
    target = _position_missing_sequence(np.asarray([1.0, 2.0, 3.0], dtype=float))
    out = pose.apply(target, validate=True)
    assert out.as_dataset(copy="none")["position"].sizes["sample"] == 2


def test_bcast_core_051_spatial_param_key_alignment_supported_after_frame_precheck() -> None:
    """ID: BCAST_CORE_051_spatial_param_key_alignment_supported_after_frame_precheck."""
    rot_arr = xr.DataArray(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        dims=("sample", "quat"),
        coords={"sample": [0, 1], "quat": list(_QUAT), "time_s": ("sample", [0.0, 0.5])},
        name="rotation",
    )
    rot = Rotation(
        AnalysisObject.from_data(
            rot_arr.to_dataset(name="rotation"),
            sequence_dim="sample",
            core_dims=("quat",),
            param_coord="time_s",
            validate=True,
        )
    )
    target_arr = xr.DataArray(
        np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        dims=("sample", "axis"),
        coords={"sample": ["a", "b"], "axis": list(_XYZ), "time_s": ("sample", [0.0, 0.5])},
        name="position",
    )
    target = Position(
        AnalysisObject.from_data(
            target_arr.to_dataset(name="position"),
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
            validate=True,
        )
    )
    with pytest.raises(
        ValueError,
        match=r"^spatial\.rotation\.apply: rotation apply requires exact 'sample' labels under sequence/batch policy\.",
    ):
        rot.apply(target, validate=True)
    out = rot.a(on="param", sequence_join=None).apply(target, validate=True)
    assert tuple(out.as_dataset(copy="none")["position"].coords["sample"].values.tolist()) == ("a", "b")
    np.testing.assert_allclose(out.as_dataset(copy="none")["position"].coords["time_s"].values, np.asarray([0.0, 0.5]))


def test_bcast_core_046_spatial_frame_precheck_runs_before_alignment_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: BCAST_CORE_046_spatial_frame_precheck_runs_before_alignment_resolution."""
    rot = frame_retag(
        _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    pos = frame_retag(
        _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="map",
        child="tip",
        validate=True,
    )

    def _boom(*_args: object, **_kwargs: object):
        raise AssertionError("topology selection should not run before frame precheck")

    monkeypatch.setattr(rotation_apply_ops, "select_topology_policy_with_intents", _boom)
    with pytest.raises(ValueError, match="transform.child == target.parent"):
        rot.apply(pos, validate=True)


def test_bcast_hard_035_frame_mismatch_not_rescuable_by_alignment_or_broadcast_policy() -> None:
    """ID: BCAST_HARD_035_frame_mismatch_not_rescuable_by_alignment_or_broadcast_policy."""
    rot = frame_retag(
        _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    ).a(on="param", sequence_join=None).b()
    pos = frame_retag(
        _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="map",
        child="tip",
        validate=True,
    ).a(on="param", sequence_join=None).b()
    with pytest.raises(ValueError, match="transform.child == target.parent"):
        rot.apply(pos, validate=True)


def test_bcast_hard_039_param_key_alignment_rejected_when_frame_preconditions_fail() -> None:
    """ID: BCAST_HARD_039_param_key_alignment_rejected_when_frame_preconditions_fail."""
    rot = frame_retag(
        _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    pos = frame_retag(
        _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="map",
        child="tip",
        validate=True,
    )
    with pytest.raises(ValueError, match="transform.child == target.parent"):
        rot.a(on="param", sequence_join=None).apply(pos.a(on="param", sequence_join=None), validate=True)


def test_bcast_hard_040_spatial_param_key_ambiguity_fails_closed() -> None:
    """ID: BCAST_HARD_040_spatial_param_key_ambiguity_fails_closed."""
    rot_arr = xr.DataArray(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        dims=("sample", "quat"),
        coords={"sample": [0, 1], "quat": list(_QUAT), "time_s": ("sample", [0.0, 0.0])},
        name="rotation",
    )
    target_arr = xr.DataArray(
        np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        dims=("sample", "axis"),
        coords={"sample": [0, 1], "axis": list(_XYZ), "time_s": ("sample", [0.0, 0.0])},
        name="position",
    )
    rot = Rotation(
        AnalysisObject.from_data(
            rot_arr.to_dataset(name="rotation"),
            sequence_dim="sample",
            core_dims=("quat",),
            param_coord="time_s",
            validate=True,
        )
    )
    target = Position(
        AnalysisObject.from_data(
            target_arr.to_dataset(name="position"),
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
            validate=True,
        )
    )
    with pytest.raises(ValueError, match="requires unique param labels"):
        rot.a(on="param", sequence_join=None).apply(target.a(on="param", sequence_join=None), validate=True)


def test_bcast_hard_042_owner_prefixed_error_boundaries_preserved_in_spatial_e3c_paths() -> None:
    """ID: BCAST_HARD_042_owner_prefixed_error_boundaries_preserved_in_spatial_e3c_paths."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    target = _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float))
    bad_target = Position(target.as_dataset(copy="none").assign_coords(sample=[10]))
    with pytest.raises(ValueError) as exc:
        rot.a(on="sequence", sequence_join="exact").apply(bad_target, validate=True)
    assert "spatial.rotation.apply" in str(exc.value)


def test_bcast_hard_043_no_implicit_spatial_interpolation_in_alignment_broadcast_paths() -> None:
    """ID: BCAST_HARD_043_no_implicit_spatial_interpolation_in_alignment_broadcast_paths."""
    rot_arr = xr.DataArray(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        dims=("sample", "quat"),
        coords={"sample": [0, 1], "quat": list(_QUAT), "time_s": ("sample", [0.0, 0.5])},
        name="rotation",
    )
    target_arr = xr.DataArray(
        np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        dims=("sample", "axis"),
        coords={"sample": ["a", "b"], "axis": list(_XYZ), "time_s": ("sample", [0.0, 0.6])},
        name="position",
    )
    rot = Rotation(
        AnalysisObject.from_data(
            rot_arr.to_dataset(name="rotation"),
            sequence_dim="sample",
            core_dims=("quat",),
            param_coord="time_s",
            validate=True,
        )
    )
    target = Position(
        AnalysisObject.from_data(
            target_arr.to_dataset(name="position"),
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
            validate=True,
        )
    )
    with pytest.raises(ValueError, match="requires exact 'time_s' labels"):
        rot.a(on="param", sequence_join=None).apply(target.a(on="param", sequence_join=None), validate=True)


def test_bcast_core_009_no_frame_or_rep_semantics_changed_by_b_helper() -> None:
    """ID: BCAST_CORE_009_no_frame_or_rep_semantics_changed_by_b_helper."""
    rot = frame_retag(
        _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    target = frame_retag(
        _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="body",
        child="tip",
        validate=True,
    )
    strict = rot.apply(target, validate=True)
    semantic = rot.b().apply(target, validate=True)
    assert get_frames(strict.as_dataset(copy="none")) == get_frames(semantic.as_dataset(copy="none"))
    assert get_position_rep(strict.as_dataset(copy="none"), owner="test") == get_position_rep(
        semantic.as_dataset(copy="none"),
        owner="test",
    )
    np.testing.assert_allclose(
        strict.as_dataset(copy="none")["position"].values,
        semantic.as_dataset(copy="none")["position"].values,
    )


def test_bcast_hard_006_spatial_frame_mismatch_paths_unchanged_under_b() -> None:
    """ID: BCAST_HARD_006_spatial_frame_mismatch_paths_unchanged_under_b."""
    rot = frame_retag(
        _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    pos = frame_retag(
        _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="map",
        child="tip",
        validate=True,
    )
    with pytest.raises(ValueError, match=r"^spatial\.rotation\.apply:"):
        rot.b().apply(pos, validate=True)


def test_bcast_hard_007_representation_conflict_paths_unchanged_under_b() -> None:
    """ID: BCAST_HARD_007_representation_conflict_paths_unchanged_under_b."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    bad_ds = rot.as_dataset(copy="none").copy(deep=True)
    tal = dict(bad_ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial = dict(ext.get("spatial", {}))
    representation = dict(spatial.get("representation", {}))
    representation["rep"] = "bad"
    spatial["representation"] = representation
    ext["spatial"] = spatial
    tal["ext"] = ext
    bad_ds.attrs["tal"] = tal
    bad_rot = Rotation(rot.as_dataset(copy="none"))
    bad_rot._data = bad_ds
    with pytest.raises(ValueError, match=r"unsupported rotation representation"):
        _ = bad_rot.b()


def test_bcast_hard_036_representation_conflict_paths_unchanged_under_e3c_rollout() -> None:
    """ID: BCAST_HARD_036_representation_conflict_paths_unchanged_under_e3c_rollout."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float))
    bad_ds = rot.as_dataset(copy="none").copy(deep=True)
    tal = dict(bad_ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial = dict(ext.get("spatial", {}))
    representation = dict(spatial.get("representation", {}))
    representation["rep"] = "bad"
    spatial["representation"] = representation
    ext["spatial"] = spatial
    tal["ext"] = ext
    bad_ds.attrs["tal"] = tal
    bad_rot = Rotation(rot.as_dataset(copy="none"))
    bad_rot._data = bad_ds
    target = _position(np.asarray([[1.0, 0.0, 0.0]], dtype=float))
    with pytest.raises(ValueError, match="unsupported rotation representation"):
        bad_rot.a(on="sequence").apply(target, validate=True)


def test_bcast_hard_005_no_raw_runtime_exception_leakage_in_b_paths() -> None:
    """ID: BCAST_HARD_005_no_raw_runtime_exception_leakage_in_b_paths."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float))
    target = _position(np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float))
    bad_target = Position(target.as_dataset(copy="none").assign_coords(sample=[10, 11]))
    with pytest.raises(ValueError) as exc:
        rot.b().apply(bad_target, validate=True)
    message = str(exc.value)
    assert "spatial.rotation.apply" in message
    assert "KeyError" not in message
    assert "AttributeError" not in message


def test_topo_hard_005_no_raw_runtime_exception_leakage_after_migration() -> None:
    """ID: TOPO_HARD_005_no_raw_runtime_exception_leakage_after_migration."""
    rot = _rotation_quat(np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float))
    target = _position(np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float))
    target_ds = target.as_dataset(copy="none").assign_coords(sample=[10, 11])
    bad_target = Position(target_ds)
    with pytest.raises(ValueError, match=r"^spatial\.rotation\.apply:"):
        rot.apply(bad_target, validate=True)
