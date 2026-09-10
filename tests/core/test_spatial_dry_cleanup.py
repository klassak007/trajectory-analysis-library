from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tal import AnalysisObject
from tal import ufuncs as tal_ufuncs
from tal.core import (
    ComponentRegistryOptions,
    ComponentSpec,
    define_components,
    read_components,
)
from tal.core.orchestration.runtime_checks import (
    resolve_single_numeric_var_single_core_dim,
)
from tal.core.schema import set_param_coord
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.frames import FrameGraph
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
from tal.spatial.metadata import (
    get_acceleration_rep,
    get_pose_rep,
    get_position_rep,
    get_rotation_rep,
    get_velocity_rep,
)
from tal.spatial.policies.wrap import wrap_as, wrap_like
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


def _vector3_dataset(*, var_name: str, core_dim: str, values: np.ndarray | None = None, sequence_dim: str = "sample") -> xr.Dataset:
    payload = values if values is not None else np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    arr = xr.DataArray(
        payload,
        dims=(sequence_dim, core_dim),
        coords={sequence_dim: [0, 1], core_dim: list(_XYZ)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name=var_name), sequence_dim=sequence_dim, core_dims=(core_dim,), validate=True)
    return ao.as_dataset(copy="none").copy(deep=True)


def _quat_dataset(*, sequence_dim: str = "sample", labels: tuple[str, str, str, str] = _QUAT) -> xr.Dataset:
    arr = xr.DataArray(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]], dtype=float),
        dims=(sequence_dim, "quat"),
        coords={sequence_dim: [0, 1], "quat": list(labels)},
        name="rotation",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="rotation"), sequence_dim=sequence_dim, core_dims=("quat",), validate=True)
    return ao.as_dataset(copy="none").copy(deep=True)


def _position(*, sequence_dim: str = "sample", values: np.ndarray | None = None) -> Position:
    return Position(_vector3_dataset(var_name="position", core_dim="axis", values=values, sequence_dim=sequence_dim))


def _rotation(*, sequence_dim: str = "sample", labels: tuple[str, str, str, str] = _QUAT) -> Rotation:
    return Rotation(_quat_dataset(sequence_dim=sequence_dim, labels=labels))


def _pose(*, parent: str | None = None, child: str | None = None, sequence_dim: str = "sample") -> Pose:
    pos = _position(sequence_dim=sequence_dim)
    rot = _rotation(sequence_dim=sequence_dim)
    if parent is not None and child is not None:
        pos = frame_retag(pos, parent=parent, child=child, validate=True)
        rot = frame_retag(rot, parent=parent, child=child, validate=True)
    return Pose.from_components(rot, pos, validate=True)


def _linear_velocity(*, sequence_dim: str = "sample") -> LinearVelocity:
    return LinearVelocity(_vector3_dataset(var_name="linear_velocity", core_dim="linear_axis", sequence_dim=sequence_dim))


def _angular_velocity(*, sequence_dim: str = "sample") -> AngularVelocity:
    return AngularVelocity(_vector3_dataset(var_name="angular_velocity", core_dim="angular_axis", sequence_dim=sequence_dim))


def _linear_acceleration(*, sequence_dim: str = "sample") -> LinearAcceleration:
    return LinearAcceleration(_vector3_dataset(var_name="linear_acceleration", core_dim="linear_axis", sequence_dim=sequence_dim))


def _angular_acceleration(*, sequence_dim: str = "sample") -> AngularAcceleration:
    return AngularAcceleration(_vector3_dataset(var_name="angular_acceleration", core_dim="angular_axis", sequence_dim=sequence_dim))


def _interp_like_vector3_ao(
    *,
    var_name: str,
    core_dim: str,
    low_payload: np.ndarray,
    high_payload: np.ndarray,
) -> AnalysisObject:
    low = xr.DataArray(
        low_payload,
        dims=("sample", core_dim),
        coords={
            "sample": [0, 1, 2],
            core_dim: list(_XYZ),
            "time_s": ("sample", [0.0, 1.0, 2.0]),
        },
        name=var_name,
    )
    high = xr.DataArray(
        high_payload,
        dims=("sample", core_dim),
        coords={
            "sample": [0, 1, 2, 3, 4],
            core_dim: list(_XYZ),
            "time_s": ("sample", [0.0, 0.5, 1.0, 1.5, 2.0]),
        },
        name=var_name,
    )
    low_ao = AnalysisObject.from_data(
        low.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=(core_dim,),
        param_coord="time_s",
        validate=True,
    )
    high_ao = AnalysisObject.from_data(
        high.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=(core_dim,),
        param_coord="time_s",
        validate=True,
    )
    return low_ao.param.interp_like(high_ao, on="time_s", validate=True)


def test_spatial_core_080_velocity_acceleration_family_parity_through_shared_owner() -> None:
    """ID: SPATIAL_CORE_080_velocity_acceleration_family_parity_through_shared_owner."""
    vel = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    acc = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)
    assert get_velocity_rep(vel.as_dataset(copy="none"), owner="test") == "components"
    assert get_acceleration_rep(acc.as_dataset(copy="none"), owner="test") == "components"
    assert set(read_components(vel).keys()) == {"linear", "angular"}
    assert set(read_components(acc).keys()) == {"linear", "angular"}
    assert isinstance(vel.linear(validate=True), LinearVelocity)
    assert isinstance(vel.angular(validate=True), AngularVelocity)
    assert isinstance(acc.linear(validate=True), LinearAcceleration)
    assert isinstance(acc.angular(validate=True), AngularAcceleration)


def test_spatial_core_081_frame_policy_parity_across_compose_apply_and_intent() -> None:
    """ID: SPATIAL_CORE_081_frame_policy_parity_across_compose_apply_and_intent."""
    rot_out = frame_retag(_rotation(), parent="world", child="a", validate=True).compose(
        frame_retag(_rotation(), parent="a", child="b", validate=True),
        validate=True,
    )
    pose_out = _pose(parent="world", child="a").compose(_pose(parent="a", child="b"), validate=True)
    pos_out = frame_retag(_position(values=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])), parent="world", child="a", validate=True) + frame_retag(
        _position(values=np.asarray([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])),
        parent="a",
        child="b",
        validate=True,
    )
    assert get_frames(rot_out.as_dataset(copy="none")) == ("world", "b")
    assert get_frames(pose_out.as_dataset(copy="none")) == ("world", "b")
    assert get_frames(pos_out.as_dataset(copy="none")) == ("world", "b")


def test_spatial_core_082_single_var_single_core_shared_resolver_parity() -> None:
    """ID: SPATIAL_CORE_082_single_var_single_core_shared_resolver_parity."""
    pos_var, pos_core = resolve_single_numeric_var_single_core_dim(_position().as_dataset(copy="none"), owner="test", what="position")
    rot_var, rot_core = resolve_single_numeric_var_single_core_dim(_rotation().as_dataset(copy="none"), owner="test", what="rotation")
    assert pos_var == "position"
    assert pos_core == "axis"
    assert rot_var == "rotation"
    assert rot_core == "quat"


def test_spatial_core_083_paired_component_assembly_parity_for_pose_and_kinematics() -> None:
    """ID: SPATIAL_CORE_083_paired_component_assembly_parity_for_pose_and_kinematics."""
    pose = _pose(parent="world", child="body")
    vel = Velocity.from_linear_angular(
        frame_retag(_linear_velocity(), parent="world", child="body", validate=True),
        frame_retag(_angular_velocity(), parent="world", child="body", validate=True),
        validate=True,
    )
    assert set(read_components(pose).keys()) == {"position", "rotation"}
    assert set(read_components(vel).keys()) == {"linear", "angular"}
    pos, rot = pose.decompose(validate=True)
    assert list(pos.as_dataset(copy="none").data_vars) == ["position"]
    assert list(rot.as_dataset(copy="none").data_vars) == ["rotation"]
    assert "datavar" not in pos.as_dataset(copy="none").data_vars
    assert "datavar" not in rot.as_dataset(copy="none").data_vars


def test_spatial_core_084_conversion_finalization_parity_rotation_pose_vector6() -> None:
    """ID: SPATIAL_CORE_084_conversion_finalization_parity_rotation_pose_vector6."""
    rot = frame_retag(_rotation(), parent="world", child="body", validate=True).as_matrix(validate=True)
    pose = _pose(parent="world", child="body").as_matrix(validate=True)
    vel = frame_retag(Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True), parent="world", child="body", validate=True)
    vel_v6 = vel.as_vector6(validate=True)
    assert get_rotation_rep(rot.as_dataset(copy="none"), owner="test") == "matrix"
    assert get_pose_rep(pose.as_dataset(copy="none"), owner="test") == "matrix"
    assert get_velocity_rep(vel_v6.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_frames(rot.as_dataset(copy="none")) == ("world", "body")
    assert get_frames(pose.as_dataset(copy="none")) == ("world", "body")
    assert get_frames(vel_v6.as_dataset(copy="none")) == ("world", "body")


def test_spatial_core_085_alignment_topology_parity_after_shared_alignment_owner() -> None:
    """ID: SPATIAL_CORE_085_alignment_topology_parity_after_shared_alignment_owner."""
    with pytest.raises(ValueError, match="spatial.rotation.compose: compose requires matching sequence_dim"):
        _rotation(sequence_dim="sample").compose(_rotation(sequence_dim="time"), validate=True)
    with pytest.raises(ValueError, match="spatial.pose.compose: pose compose requires matching sequence_dim"):
        _pose(sequence_dim="sample").compose(_pose(sequence_dim="time"), validate=True)


def test_spatial_core_086_wrap_helper_parity_validate_and_nonvalidate_paths() -> None:
    """ID: SPATIAL_CORE_086_wrap_helper_parity_validate_and_nonvalidate_paths."""
    rot = _rotation()
    wrapped_validate = wrap_as(Rotation, rot.as_dataset(copy="none"), validate=True)
    wrapped_like = wrap_like(rot, rot.as_dataset(copy="none"), validate=False)
    assert isinstance(wrapped_validate, Rotation)
    assert isinstance(wrapped_like, Rotation)


def test_spatial_core_087_spatial_conversion_apply_compose_public_behavior_smoke() -> None:
    """ID: SPATIAL_CORE_087_spatial_conversion_apply_compose_public_behavior_smoke."""
    rot = _rotation()
    pose = _pose()
    target = _position(values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert isinstance(rot.as_matrix(validate=True).as_quat(validate=True), Rotation)
    assert isinstance(rot.compose(rot.inverse(validate=True), validate=True), Rotation)
    assert isinstance(pose.as_matrix(validate=True).as_components(validate=True), Pose)
    assert isinstance(pose.compose(pose.inverse(validate=True), validate=True), Pose)
    assert isinstance(pose.apply(target, validate=True), Position)


def test_spatial_hard_096_no_raw_runtime_leakage_after_cleanup_owner_reuse() -> None:
    """ID: SPATIAL_HARD_096_no_raw_runtime_leakage_after_cleanup_owner_reuse."""
    with pytest.raises(TypeError, match="spatial.pose.apply"):
        _pose().apply(object(), validate=True)


def test_spatial_hard_097_framed_chain_mismatch_remains_fail_closed_with_owner_context() -> None:
    """ID: SPATIAL_HARD_097_framed_chain_mismatch_remains_fail_closed_with_owner_context."""
    left = frame_retag(_rotation(), parent="world", child="a", validate=True)
    right = frame_retag(_rotation(), parent="map", child="b", validate=True)
    with pytest.raises(ValueError, match="spatial.rotation.compose"):
        left.compose(right, validate=True)


def test_spatial_hard_098_non_core_dim_mismatch_remains_fail_closed() -> None:
    """ID: SPATIAL_HARD_098_non_core_dim_mismatch_remains_fail_closed."""
    with pytest.raises(ValueError, match="matching sequence_dim"):
        _pose(sequence_dim="sample").compose(_pose(sequence_dim="time"), validate=True)


def test_spatial_hard_099_invalid_label_and_core_layout_checks_remain_fail_closed() -> None:
    """ID: SPATIAL_HARD_099_invalid_label_and_core_layout_checks_remain_fail_closed."""
    with pytest.raises(ValueError, match="spatial.rotation.__init__"):
        _ = _rotation(labels=("a", "b", "c", "d"))


def test_spatial_hard_100_kinematics_family_wrong_operand_type_boundaries_are_deterministic() -> None:
    """ID: SPATIAL_HARD_100_kinematics_family_wrong_operand_type_boundaries_are_deterministic."""
    with pytest.raises(TypeError, match="spatial.velocity.from_linear_angular"):
        _ = Velocity.from_linear_angular(object(), _angular_velocity(), validate=True)


def test_spatial_hard_101_paired_component_optional_name_mismatch_fails_closed() -> None:
    """ID: SPATIAL_HARD_101_paired_component_optional_name_mismatch_fails_closed."""
    linear_ds = _linear_velocity().as_dataset(copy="none").assign_coords({"tau": ("sample", [0.0, 1.0])})
    angular_ds = _angular_velocity().as_dataset(copy="none").assign_coords({"sigma": ("sample", [0.0, 1.0])})
    linear = LinearVelocity(set_param_coord(linear_ds, name="tau", validate=False))
    angular = AngularVelocity(set_param_coord(angular_ds, name="sigma", validate=False))
    with pytest.raises(ValueError, match="spatial.velocity.from_linear_angular"):
        _ = Velocity.from_linear_angular(linear, angular, validate=True)


def test_bcast_core_051_velocity_from_linear_angular_one_sided_static_component_adopts_dynamic_series_roles() -> None:
    """ID: BCAST_CORE_901_velocity_from_linear_angular_one_sided_static_component_adopts_dynamic_series_roles."""
    linear_arr = xr.DataArray(
        np.asarray(
            [
                [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
            ],
            dtype=float,
        ),
        dims=("sample", "trial", "linear_axis"),
        coords={
            "sample": [0, 1],
            "trial": ["t0", "t1"],
            "linear_axis": list(_XYZ),
            "time_s": ("sample", [0.0, 0.5]),
            "sample_size": ("trial", [2, 2]),
        },
        name="linear_velocity",
    )
    angular_arr = xr.DataArray(
        np.asarray([0.0, 1.0, 0.0], dtype=float),
        dims=("angular_axis",),
        coords={"angular_axis": list(_XYZ)},
        name="angular_velocity",
    )
    linear = LinearVelocity(
        AnalysisObject.from_data(
            linear_arr.to_dataset(name="linear_velocity"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("linear_axis",),
            param_coord="time_s",
            sequence_size_coord="sample_size",
            validate=True,
        )
    )
    angular = AngularVelocity(
        AnalysisObject.from_data(
            angular_arr.to_dataset(name="angular_velocity"),
            core_dims=("angular_axis",),
            validate=True,
        )
    )
    out = Velocity.from_linear_angular(linear, angular, validate=True)
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ("linear_axis", "angular_axis")
    assert read_param_coord_name(out.as_dataset(copy="none")) == "time_s"
    assert read_sequence_size_coord_name(out.as_dataset(copy="none")) == "sample_size"
    np.testing.assert_allclose(
        out.angular(validate=True).as_dataset(copy="none")["angular_velocity"].transpose("sample", "trial", "angular_axis").values,
        np.broadcast_to(np.asarray([0.0, 1.0, 0.0], dtype=float), (2, 2, 3)),
        atol=1e-6,
    )


def test_bcast_core_052_acceleration_from_linear_angular_one_sided_static_component_adopts_dynamic_series_roles() -> None:
    """ID: BCAST_CORE_902_acceleration_from_linear_angular_one_sided_static_component_adopts_dynamic_series_roles."""
    linear_arr = xr.DataArray(
        np.asarray(
            [
                [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]],
                [[0.4, 0.5, 0.6], [4.0, 5.0, 6.0]],
            ],
            dtype=float,
        ),
        dims=("sample", "trial", "linear_axis"),
        coords={
            "sample": [0, 1],
            "trial": ["t0", "t1"],
            "linear_axis": list(_XYZ),
        },
        name="linear_acceleration",
    )
    angular_arr = xr.DataArray(
        np.asarray([0.0, 0.5, 0.0], dtype=float),
        dims=("angular_axis",),
        coords={"angular_axis": list(_XYZ)},
        name="angular_acceleration",
    )
    linear = LinearAcceleration(
        AnalysisObject.from_data(
            linear_arr.to_dataset(name="linear_acceleration"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("linear_axis",),
            validate=True,
        )
    )
    angular = AngularAcceleration(
        AnalysisObject.from_data(
            angular_arr.to_dataset(name="angular_acceleration"),
            core_dims=("angular_axis",),
            validate=True,
        )
    )
    out = Acceleration.from_linear_angular(linear, angular, validate=True)
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ("linear_axis", "angular_axis")
    np.testing.assert_allclose(
        out.angular(validate=True).as_dataset(copy="none")["angular_acceleration"].transpose("sample", "trial", "angular_axis").values,
        np.broadcast_to(np.asarray([0.0, 0.5, 0.0], dtype=float), (2, 2, 3)),
        atol=1e-6,
    )


def test_spatial_core_120_velocity_merge_tolerates_reserved_valid_coord_attr_drift() -> None:
    """ID: SPATIAL_CORE_925_velocity_merge_tolerates_reserved_valid_coord_attr_drift."""
    linear_ao = _interp_like_vector3_ao(
        var_name="linear_velocity",
        core_dim="linear_axis",
        low_payload=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        high_payload=np.asarray(
            [[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=float,
        ),
    )
    linear = LinearVelocity(linear_ao)
    angular_ds = linear_ao.as_dataset(copy="none").rename({"linear_velocity": "angular_velocity", "linear_axis": "angular_axis"}).copy(deep=True)
    assert "valid" in angular_ds.coords
    assert linear_ao.as_dataset(copy="none").coords["valid"].attrs
    angular_ds.coords["valid"].attrs = {}
    angular = AngularVelocity(
        AnalysisObject.from_data(
            angular_ds,
            sequence_dim="sample",
            core_dims=("angular_axis",),
            param_coord="time_s",
            validate=True,
        )
    )
    out = Velocity.from_linear_angular(linear, angular, validate=True)
    assert read_param_coord_name(out.as_dataset(copy="none")) == "time_s"
    np.testing.assert_allclose(
        out.linear(validate=True).as_dataset(copy="none")["linear_velocity"].values,
        linear.as_dataset(copy="none")["linear_velocity"].values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out.angular(validate=True).as_dataset(copy="none")["angular_velocity"].values,
        angular.as_dataset(copy="none")["angular_velocity"].values,
        atol=1e-6,
    )


def test_spatial_core_121_acceleration_merge_tolerates_reserved_valid_coord_attr_drift() -> None:
    """ID: SPATIAL_CORE_926_acceleration_merge_tolerates_reserved_valid_coord_attr_drift."""
    linear_ao = _interp_like_vector3_ao(
        var_name="linear_acceleration",
        core_dim="linear_axis",
        low_payload=np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=float),
        high_payload=np.asarray(
            [[0.1, 0.0, 0.0], [0.15, 0.0, 0.0], [0.2, 0.0, 0.0], [0.25, 0.0, 0.0], [0.3, 0.0, 0.0]],
            dtype=float,
        ),
    )
    linear = LinearAcceleration(linear_ao)
    angular_ds = linear_ao.as_dataset(copy="none").rename(
        {"linear_acceleration": "angular_acceleration", "linear_axis": "angular_axis"}
    ).copy(deep=True)
    assert "valid" in angular_ds.coords
    assert linear_ao.as_dataset(copy="none").coords["valid"].attrs
    angular_ds.coords["valid"].attrs = {}
    angular = AngularAcceleration(
        AnalysisObject.from_data(
            angular_ds,
            sequence_dim="sample",
            core_dims=("angular_axis",),
            param_coord="time_s",
            validate=True,
        )
    )
    out = Acceleration.from_linear_angular(linear, angular, validate=True)
    assert read_param_coord_name(out.as_dataset(copy="none")) == "time_s"
    np.testing.assert_allclose(
        out.linear(validate=True).as_dataset(copy="none")["linear_acceleration"].values,
        linear.as_dataset(copy="none")["linear_acceleration"].values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out.angular(validate=True).as_dataset(copy="none")["angular_acceleration"].values,
        angular.as_dataset(copy="none")["angular_acceleration"].values,
        atol=1e-6,
    )


def test_spatial_hard_101b_velocity_sequence_size_coord_mismatch_fails_closed() -> None:
    """ID: SPATIAL_HARD_922_velocity_sequence_size_coord_mismatch_fails_closed."""
    linear_arr = xr.DataArray(
        np.asarray(
            [
                [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
            ],
            dtype=float,
        ),
        dims=("sample", "trial", "linear_axis"),
        coords={
            "sample": [0, 1],
            "trial": ["t0", "t1"],
            "linear_axis": list(_XYZ),
            "sample_size_a": ("trial", [2, 2]),
        },
        name="linear_velocity",
    )
    angular_arr = xr.DataArray(
        np.asarray(
            [
                [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            ],
            dtype=float,
        ),
        dims=("sample", "trial", "angular_axis"),
        coords={
            "sample": [0, 1],
            "trial": ["t0", "t1"],
            "angular_axis": list(_XYZ),
            "sample_size_b": ("trial", [2, 2]),
        },
        name="angular_velocity",
    )
    linear = LinearVelocity(
        AnalysisObject.from_data(
            linear_arr.to_dataset(name="linear_velocity"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("linear_axis",),
            sequence_size_coord="sample_size_a",
            validate=True,
        )
    )
    angular = AngularVelocity(
        AnalysisObject.from_data(
            angular_arr.to_dataset(name="angular_velocity"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("angular_axis",),
            sequence_size_coord="sample_size_b",
            validate=True,
        )
    )
    with pytest.raises(ValueError, match="spatial.velocity.from_linear_angular: sequence_size_coord mismatch"):
        _ = Velocity.from_linear_angular(linear, angular, validate=True)


def test_spatial_hard_102_conversion_allocator_collision_fallback_is_deterministic() -> None:
    """ID: SPATIAL_HARD_102_conversion_allocator_collision_fallback_is_deterministic."""
    ds = _rotation().as_dataset(copy="none")
    for coord in ("row", "rot_row", "matrix_row", "col", "rot_col", "matrix_col"):
        ds = ds.assign_coords({coord: 0})
    matrix = Rotation(ds).as_matrix(validate=True)
    _, _, _, core_dims = read_roles(matrix.as_dataset(copy="none"))
    assert core_dims == ("row_2", "col_2")


def test_spatial_hard_103_vector6_component_registry_truthfulness_preserved() -> None:
    """ID: SPATIAL_HARD_103_vector6_component_registry_truthfulness_preserved."""
    vel = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    acc = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)
    vel_v6 = vel.as_vector6(validate=True)
    acc_v6 = acc.as_vector6(validate=True)
    assert read_components(vel_v6) == {}
    assert read_components(acc_v6) == {}
    assert set(read_components(vel_v6.as_components(validate=True)).keys()) == {"linear", "angular"}
    assert set(read_components(acc_v6.as_components(validate=True)).keys()) == {"linear", "angular"}


def test_spatial_hard_104_stale_helper_modules_removed_regression_lock() -> None:
    """ID: SPATIAL_HARD_104_stale_helper_modules_removed_regression_lock."""
    assert not Path("tal/spatial/apply_shared.py").exists()
    assert not Path("tal/spatial/pose_shared.py").exists()


def test_spatial_core_095_pose_and_kinematics_component_spec_shared_owner_supports_renamed_component_vars() -> None:
    """ID: SPATIAL_CORE_095_pose_and_kinematics_component_spec_shared_owner_supports_renamed_component_vars."""
    pose_ds = _pose().as_dataset(copy="none").rename({"position": "pos_payload", "rotation": "rot_payload"})
    pose_source = AnalysisObject._from_validated(pose_ds)
    pose_registry = {
        "position": ComponentSpec(core_dim="axis", labels=_XYZ, var="pos_payload"),
        "rotation": ComponentSpec(core_dim="quat", labels=_QUAT, var="rot_payload"),
    }
    pose_with_registry = define_components(
        pose_source,
        opts=ComponentRegistryOptions(registry=pose_registry, replace=True),
        validate=False,
    )
    pose = Pose(pose_with_registry.as_dataset(copy="none"))
    assert read_components(pose)["position"].var == "pos_payload"
    assert read_components(pose)["rotation"].var == "rot_payload"

    vel_ds = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True).as_dataset(copy="none")
    vel_ds = vel_ds.rename({"linear_velocity": "lin_payload", "angular_velocity": "ang_payload"})
    vel_source = AnalysisObject._from_validated(vel_ds)
    vel_registry = {
        "linear": ComponentSpec(core_dim="linear_axis", labels=_XYZ, var="lin_payload"),
        "angular": ComponentSpec(core_dim="angular_axis", labels=_XYZ, var="ang_payload"),
    }
    vel_with_registry = define_components(
        vel_source,
        opts=ComponentRegistryOptions(registry=vel_registry, replace=True),
        validate=False,
    )
    velocity = Velocity(vel_with_registry.as_dataset(copy="none"))
    assert read_components(velocity)["linear"].var == "lin_payload"
    assert read_components(velocity)["angular"].var == "ang_payload"


@pytest.mark.parametrize(
    ("builder", "registry"),
    [
        (
            lambda: _pose().as_dataset(copy="none"),
            {
                "position": ComponentSpec(core_dim="axis", labels=_XYZ, var=None),
                "rotation": ComponentSpec(core_dim="quat", labels=_QUAT, var="rotation"),
            },
        ),
        (
            lambda: Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True).as_dataset(copy="none"),
            {
                "linear": ComponentSpec(core_dim="linear_axis", labels=_XYZ, var=None),
                "angular": ComponentSpec(core_dim="angular_axis", labels=_XYZ, var="angular_velocity"),
            },
        ),
    ],
)
def test_spatial_hard_113_pose_and_kinematics_components_require_spec_var_in_registry(
    builder,
    registry: dict[str, ComponentSpec],
) -> None:
    """ID: SPATIAL_HARD_113_pose_and_kinematics_components_require_spec_var_in_registry."""
    source = AnalysisObject._from_validated(builder())
    rewritten = define_components(
        source,
        opts=ComponentRegistryOptions(registry=registry, replace=True),
        validate=False,
    )
    if "position" in registry:
        owner = "spatial.pose.__init__"
        ctor = Pose
    else:
        owner = "spatial.velocity.__init__"
        ctor = Velocity
    with pytest.raises(ValueError, match=owner):
        ctor(rewritten.as_dataset(copy="none"))
    with pytest.raises(ValueError, match="must declare spec.var"):
        ctor(rewritten.as_dataset(copy="none"))


def test_spatial_core_096_rotation_pose_conversion_allocator_and_finalize_shared_owner_parity() -> None:
    """ID: SPATIAL_CORE_096_rotation_pose_conversion_allocator_and_finalize_shared_owner_parity."""
    rot_ds = _rotation().as_dataset(copy="none")
    for coord in ("row", "rot_row", "matrix_row", "col", "rot_col", "matrix_col"):
        rot_ds = rot_ds.assign_coords({coord: 0})
    rot_matrix = Rotation(rot_ds).as_matrix(validate=True)
    _, _, _, rot_core_dims = read_roles(rot_matrix.as_dataset(copy="none"))
    assert rot_core_dims == ("row_2", "col_2")

    pose_ds = _pose().as_dataset(copy="none")
    for coord in ("row", "pose_row", "col", "pose_col"):
        pose_ds = pose_ds.assign_coords({coord: 0})
    pose_matrix = Pose(pose_ds).as_matrix(validate=True)
    _, _, _, pose_core_dims = read_roles(pose_matrix.as_dataset(copy="none"))
    assert pose_core_dims == ("pose_row_2", "pose_col_2")


def test_spatial_hard_117_spatial_kinematics_components_module_removed_no_reintroduction() -> None:
    """ID: SPATIAL_HARD_117_spatial_kinematics_components_module_removed_no_reintroduction."""
    assert not Path("tal/spatial/kinematics_components.py").exists()


def test_spatial_core_097_public_spatial_surface_remains_flat_after_internal_relayout() -> None:
    """ID: SPATIAL_CORE_097_public_spatial_surface_remains_flat_after_internal_relayout."""
    from tal.spatial import (
        Acceleration,
        AngularAcceleration,
        AngularVelocity,
        LinearAcceleration,
        LinearVelocity,
        PathSolveOptions,
        Pose,
        Position,
        Rotation,
        Velocity,
        solve_pose_path_transform,
        solve_rotation_path_transform,
    )

    assert Position.__name__ == "Position"
    assert Rotation.__name__ == "Rotation"
    assert Pose.__name__ == "Pose"
    assert Velocity.__name__ == "Velocity"
    assert Acceleration.__name__ == "Acceleration"
    assert LinearVelocity.__name__ == "LinearVelocity"
    assert AngularVelocity.__name__ == "AngularVelocity"
    assert LinearAcceleration.__name__ == "LinearAcceleration"
    assert AngularAcceleration.__name__ == "AngularAcceleration"
    assert callable(solve_pose_path_transform)
    assert callable(solve_rotation_path_transform)
    assert PathSolveOptions.__name__ == "PathSolveOptions"


def test_spatial_core_098_metadata_package_relayout_preserves_rep_and_role_accessors() -> None:
    """ID: SPATIAL_CORE_098_metadata_package_relayout_preserves_rep_and_role_accessors."""
    from tal.spatial import metadata as spatial_metadata

    pose = _pose(parent="world", child="body")
    rotation = _rotation()
    velocity = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True).as_vector6(validate=True)
    assert spatial_metadata.get_pose_rep(pose.as_dataset(copy="none"), owner="test") == "components"
    assert spatial_metadata.get_rotation_rep(rotation.as_dataset(copy="none"), owner="test") == "quat"
    assert spatial_metadata.get_velocity_rep(velocity.as_dataset(copy="none"), owner="test") == "vector6"
    assert spatial_metadata.validate_spatial_roles is not None


def test_spatial_core_099_ops_and_kernels_relocation_preserves_transform_behavior() -> None:
    """ID: SPATIAL_CORE_099_ops_and_kernels_relocation_preserves_transform_behavior."""
    rot = _rotation()
    pos = _position(values=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float))
    out = rot.apply(pos, validate=True)
    assert isinstance(out, Position)
    assert get_position_rep(out.as_dataset(copy="none"), owner="test") == "cart"


def test_spatial_core_100_kinematics_subpackage_relayout_preserves_vector6_bridge_behavior() -> None:
    """ID: SPATIAL_CORE_100_kinematics_subpackage_relayout_preserves_vector6_bridge_behavior."""
    vel = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    acc = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)
    vel_v6 = vel.as_vector6(validate=True)
    acc_v6 = acc.as_vector6(validate=True)
    assert get_velocity_rep(vel_v6.as_dataset(copy="none"), owner="test") == "vector6"
    assert get_acceleration_rep(acc_v6.as_dataset(copy="none"), owner="test") == "vector6"
    assert isinstance(vel_v6.as_components(validate=True), Velocity)
    assert isinstance(acc_v6.as_components(validate=True), Acceleration)


def test_spatial_core_101_internal_subpackage_modules_importable_at_new_locations() -> None:
    """ID: SPATIAL_CORE_101_internal_subpackage_modules_importable_at_new_locations."""
    modules = [
        "tal.spatial.metadata.facade",
        "tal.spatial.policies.frame",
        "tal.spatial.ops.pose_ops",
        "tal.spatial.kernels.rotation_kernels",
        "tal.spatial.kinematics.family",
        "tal.spatial.conversion.finalize",
    ]
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_spatial_core_102_tal_ufuncs_root_facade_behavior_remains_stable() -> None:
    """ID: SPATIAL_CORE_102_tal_ufuncs_root_facade_behavior_remains_stable."""
    ds = _vector3_dataset(var_name="signal", core_dim="axis")
    ao = AnalysisObject._from_validated(ds)
    out = tal_ufuncs.sin(ao)
    assert isinstance(out, AnalysisObject)
    np.testing.assert_allclose(out.as_dataset(copy="none")["signal"].values, np.sin(ds["signal"].values))


def test_spatial_hard_118_removed_flat_internal_module_paths_are_not_importable() -> None:
    """ID: SPATIAL_HARD_118_removed_flat_internal_module_paths_are_not_importable."""
    removed_modules = [
        "tal.spatial.pose_ops",
        "tal.spatial.pose_apply_ops",
        "tal.spatial.rotation_apply_ops",
        "tal.spatial.path_solve_ops",
        "tal.spatial.frame_api_ops",
        "tal.spatial.frame_policies",
        "tal.spatial.wrap",
    ]
    for module_name in removed_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_spatial_hard_119_owner_prefixed_apply_boundaries_preserved_after_internal_relayout() -> None:
    """ID: SPATIAL_HARD_119_owner_prefixed_apply_boundaries_preserved_after_internal_relayout."""
    with pytest.raises(TypeError, match="spatial.rotation.apply"):
        _rotation().apply(object(), validate=True)
    with pytest.raises(TypeError, match="spatial.pose.apply"):
        _pose().apply(object(), validate=True)


def test_spatial_hard_121_owner_prefixed_frame_path_solver_errors_preserved_after_ops_relayout() -> None:
    """ID: SPATIAL_HARD_121_owner_prefixed_frame_path_solver_errors_preserved_after_ops_relayout."""
    graph = FrameGraph()
    parent = graph.get_or_create_frame("a")
    child = graph.get_or_create_frame("b", parent=parent)
    with pytest.raises(TypeError, match="spatial.pose.solve_path_transform"):
        Pose.solve_path_transform(parent, child, graph=graph, edge_pose_fn=object(), validate=True)
    with pytest.raises(TypeError, match="spatial.rotation.solve_path_transform"):
        Rotation.solve_path_transform(parent, child, graph=graph, edge_rotation_fn=object(), validate=True)


def test_spatial_hard_122_removed_flat_internal_module_files_remain_absent() -> None:
    """ID: SPATIAL_HARD_122_removed_flat_internal_module_files_remain_absent."""
    removed_paths = [
        "tal/spatial/metadata.py",
        "tal/spatial/metadata_common.py",
        "tal/spatial/metadata_roles.py",
        "tal/spatial/metadata_representation.py",
        "tal/spatial/frame_policies.py",
        "tal/spatial/intent.py",
        "tal/spatial/runtime_checks.py",
        "tal/spatial/wrap.py",
        "tal/spatial/pose_ops.py",
        "tal/spatial/pose_apply_ops.py",
        "tal/spatial/rotation_apply_ops.py",
        "tal/spatial/frame_api_ops.py",
        "tal/spatial/path_solve_ops.py",
        "tal/spatial/pose_kernels.py",
        "tal/spatial/pose_apply_kernels.py",
        "tal/spatial/rotation_kernels.py",
        "tal/spatial/rotation_compose_kernels.py",
        "tal/spatial/rotation_apply_kernels.py",
        "tal/spatial/kinematics_family.py",
        "tal/spatial/kinematics_vector6_ops.py",
        "tal/spatial/kinematics_vector6_kernels.py",
        "tal/spatial/paired_components.py",
        "tal/spatial/conversion_finalize.py",
    ]
    for rel_path in removed_paths:
        assert not Path(rel_path).exists()


def test_spatial_hard_123_tal_ufuncs_root_facade_remains_thin_and_core_delegating() -> None:
    """ID: SPATIAL_HARD_123_tal_ufuncs_root_facade_remains_thin_and_core_delegating."""
    text = Path("tal/ufuncs.py").read_text(encoding="utf-8")
    assert "from tal.core.ufunc_ops import" in text
    assert "apply_unary_ufunc(" in text
    assert "apply_binary_ufunc(" in text
    assert "tal.spatial" not in text
