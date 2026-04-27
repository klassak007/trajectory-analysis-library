from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import tal.spatial.kinematics.family as family_module

from tal import AnalysisObject
from tal.core import ParamEvalOptions
from tal.core import read_components
from tal.core.schema_errors import SchemaError
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.linalg import Array
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Position,
    Velocity,
)
from tal.spatial.metadata import (
    get_acceleration_rep,
    get_angular_acceleration_rep,
    get_angular_velocity_rep,
    get_kinematics_kind,
    get_linear_acceleration_rep,
    get_linear_velocity_rep,
    get_position_intent,
    get_velocity_rep,
)
from tal.spatial.kernels.kinematics_temporal_kernels import cumulative_simpson_kernel
from tal.spatial.temporal.options import (
    KinematicsDerivativeOptions,
    KinematicsIntegralOptions,
    KinematicsSmoothingOptions,
)
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames

_XYZ = ("x", "y", "z")


def _vector3_dataset(
    *,
    var_name: str,
    core_dim: str = "axis",
    values: np.ndarray | None = None,
    samples: tuple[int, int] = (0, 1),
) -> xr.Dataset:
    if values is None:
        values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    arr = xr.DataArray(
        values,
        dims=("sample", core_dim),
        coords={"sample": list(samples), core_dim: list(_XYZ)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=(core_dim,),
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _batched_vector3_dataset(
    *,
    var_name: str,
    core_dim: str,
    include_trial_dim: bool,
) -> xr.Dataset:
    if include_trial_dim:
        values = np.asarray(
            [
                [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
            ],
            dtype=float,
        )
        arr = xr.DataArray(
            values,
            dims=("sample", "trial", core_dim),
            coords={
                "sample": [0, 1],
                "trial": ["t0", "t1"],
                core_dim: list(_XYZ),
                "time_s": ("sample", [0.0, 0.5]),
                "sample_size": ("trial", [2, 2]),
            },
            name=var_name,
        )
    else:
        values = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=float,
        )
        arr = xr.DataArray(
            values,
            dims=("sample", core_dim),
            coords={
                "sample": [0, 1],
                core_dim: list(_XYZ),
                "time_s": ("sample", [0.0, 0.5]),
            },
            name=var_name,
        )
    ds = arr.to_dataset(name=var_name)
    if not include_trial_dim:
        ds = ds.assign_coords(
            trial=("trial", ["t0", "t1"]),
            sample_size=("trial", [2, 2]),
        )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(core_dim,),
        param_coord="time_s",
        sequence_size_coord="sample_size",
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _temporal_vector3_dataset(
    *,
    var_name: str,
    core_dim: str,
    values: np.ndarray,
    param: np.ndarray,
) -> xr.Dataset:
    arr = xr.DataArray(
        values,
        dims=("sample", core_dim),
        coords={
            "sample": list(range(values.shape[0])),
            core_dim: list(_XYZ),
            "time_s": ("sample", param),
            "group_size": xr.DataArray(np.asarray(values.shape[0], dtype="int64"), dims=()),
        },
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=(core_dim,),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _as_dataarray_with_schema(ds: xr.Dataset, *, var_name: str) -> xr.DataArray:
    da = ds[var_name].copy(deep=True)
    da.attrs["tal"] = ds.attrs["tal"]
    return da


def _set_rep(ds: xr.Dataset, rep: object) -> xr.Dataset:
    out = ds.copy(deep=True)
    tal = dict(out.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial = dict(ext.get("spatial", {}))
    representation = dict(spatial.get("representation", {}))
    representation["rep"] = rep
    spatial["representation"] = representation
    ext["spatial"] = spatial
    tal["ext"] = ext
    out.attrs["tal"] = tal
    return out


def _set_roles(ds: xr.Dataset, roles: object) -> xr.Dataset:
    out = ds.copy(deep=True)
    tal = dict(out.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial = dict(ext.get("spatial", {}))
    spatial["roles"] = roles
    ext["spatial"] = spatial
    tal["ext"] = ext
    out.attrs["tal"] = tal
    return out


def _set_bad_frames(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy(deep=True)
    tal = dict(out.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    ext["frames"] = {"parent": "world", "child": "body", "extra": "bad"}
    tal["ext"] = ext
    out.attrs["tal"] = tal
    return out


def _linear_velocity() -> LinearVelocity:
    return LinearVelocity(_vector3_dataset(var_name="linear_velocity", core_dim="linear_axis"))


def _angular_velocity() -> AngularVelocity:
    return AngularVelocity(_vector3_dataset(var_name="angular_velocity", core_dim="angular_axis"))


def _linear_acceleration() -> LinearAcceleration:
    return LinearAcceleration(_vector3_dataset(var_name="linear_acceleration", core_dim="linear_axis"))


def _angular_acceleration() -> AngularAcceleration:
    return AngularAcceleration(_vector3_dataset(var_name="angular_acceleration", core_dim="angular_axis"))


def test_spatial_core_024_linear_velocity_constructor_accepts_ao_dataset_dataarray_deterministically() -> None:
    """ID: SPATIAL_CORE_024_linear_velocity_constructor_accepts_ao_dataset_dataarray_deterministically."""
    ds = _vector3_dataset(var_name="linear_velocity")
    ao = AnalysisObject._from_validated(ds)
    da = _as_dataarray_with_schema(ds, var_name="linear_velocity")

    from_ao = LinearVelocity(ao)
    from_ds = LinearVelocity(ds)
    from_da = LinearVelocity(da)

    assert isinstance(from_ao, LinearVelocity)
    assert isinstance(from_ds, LinearVelocity)
    assert isinstance(from_da, LinearVelocity)
    assert get_linear_velocity_rep(from_ds.unsafe_data, owner="test") == "cart"
    assert get_kinematics_kind(from_ds.unsafe_data, owner="test") == "linear_velocity"


def test_spatial_core_025_angular_velocity_constructor_accepts_ao_dataset_dataarray_deterministically() -> None:
    """ID: SPATIAL_CORE_025_angular_velocity_constructor_accepts_ao_dataset_dataarray_deterministically."""
    ds = _vector3_dataset(var_name="angular_velocity")
    ao = AnalysisObject._from_validated(ds)
    da = _as_dataarray_with_schema(ds, var_name="angular_velocity")

    from_ao = AngularVelocity(ao)
    from_ds = AngularVelocity(ds)
    from_da = AngularVelocity(da)

    assert isinstance(from_ao, AngularVelocity)
    assert isinstance(from_ds, AngularVelocity)
    assert isinstance(from_da, AngularVelocity)
    assert get_angular_velocity_rep(from_ds.unsafe_data, owner="test") == "cart"
    assert get_kinematics_kind(from_ds.unsafe_data, owner="test") == "angular_velocity"


def test_spatial_core_026_velocity_spatial6_from_linear_angular_composition_boundary_deterministic() -> None:
    """ID: SPATIAL_CORE_026_velocity_spatial6_from_linear_angular_composition_boundary_deterministic."""
    linear = frame_retag(_linear_velocity(), parent="world", child="body", validate=True)
    angular = frame_retag(_angular_velocity(), parent="world", child="body", validate=True)

    vel = Velocity.from_linear_angular(linear, angular, validate=True)

    assert isinstance(vel, Velocity)
    assert get_velocity_rep(vel.unsafe_data, owner="test") == "components"
    assert get_kinematics_kind(vel.unsafe_data, owner="test") == "velocity"
    assert set(read_components(vel).keys()) == {"linear", "angular"}


def test_spatial_core_027_linear_acceleration_constructor_accepts_ao_dataset_dataarray_deterministically() -> None:
    """ID: SPATIAL_CORE_027_linear_acceleration_constructor_accepts_ao_dataset_dataarray_deterministically."""
    ds = _vector3_dataset(var_name="linear_acceleration")
    ao = AnalysisObject._from_validated(ds)
    da = _as_dataarray_with_schema(ds, var_name="linear_acceleration")

    from_ao = LinearAcceleration(ao)
    from_ds = LinearAcceleration(ds)
    from_da = LinearAcceleration(da)

    assert isinstance(from_ao, LinearAcceleration)
    assert isinstance(from_ds, LinearAcceleration)
    assert isinstance(from_da, LinearAcceleration)
    assert get_linear_acceleration_rep(from_ds.unsafe_data, owner="test") == "cart"
    assert get_kinematics_kind(from_ds.unsafe_data, owner="test") == "linear_acceleration"


def test_spatial_core_028_angular_acceleration_constructor_accepts_ao_dataset_dataarray_deterministically() -> None:
    """ID: SPATIAL_CORE_028_angular_acceleration_constructor_accepts_ao_dataset_dataarray_deterministically."""
    ds = _vector3_dataset(var_name="angular_acceleration")
    ao = AnalysisObject._from_validated(ds)
    da = _as_dataarray_with_schema(ds, var_name="angular_acceleration")

    from_ao = AngularAcceleration(ao)
    from_ds = AngularAcceleration(ds)
    from_da = AngularAcceleration(da)

    assert isinstance(from_ao, AngularAcceleration)
    assert isinstance(from_ds, AngularAcceleration)
    assert isinstance(from_da, AngularAcceleration)
    assert get_angular_acceleration_rep(from_ds.unsafe_data, owner="test") == "cart"
    assert get_kinematics_kind(from_ds.unsafe_data, owner="test") == "angular_acceleration"


def test_spatial_core_029_acceleration_spatial6_from_linear_angular_composition_boundary_deterministic() -> None:
    """ID: SPATIAL_CORE_029_acceleration_spatial6_from_linear_angular_composition_boundary_deterministic."""
    linear = frame_retag(_linear_acceleration(), parent="world", child="body", validate=True)
    angular = frame_retag(_angular_acceleration(), parent="world", child="body", validate=True)

    acc = Acceleration.from_linear_angular(linear, angular, validate=True)

    assert isinstance(acc, Acceleration)
    assert get_acceleration_rep(acc.unsafe_data, owner="test") == "components"
    assert get_kinematics_kind(acc.unsafe_data, owner="test") == "acceleration"
    assert set(read_components(acc).keys()) == {"linear", "angular"}


def test_spatial_core_032_velocity_from_linear_angular_one_framed_operand_inherits_frame_tags() -> None:
    """ID: SPATIAL_CORE_032_velocity_from_linear_angular_one_framed_operand_inherits_frame_tags."""
    linear_framed = frame_retag(_linear_velocity(), parent="world", child="body", validate=True)
    angular_unframed = _angular_velocity()
    vel_linear_framed = Velocity.from_linear_angular(linear_framed, angular_unframed, validate=True)
    assert get_frames(vel_linear_framed.unsafe_data) == ("world", "body")
    assert set(read_components(vel_linear_framed).keys()) == {"linear", "angular"}

    linear_unframed = _linear_velocity()
    angular_framed = frame_retag(_angular_velocity(), parent="map", child="imu", validate=True)
    vel_angular_framed = Velocity.from_linear_angular(linear_unframed, angular_framed, validate=True)
    assert get_frames(vel_angular_framed.unsafe_data) == ("map", "imu")
    assert set(read_components(vel_angular_framed).keys()) == {"linear", "angular"}


def test_spatial_core_033_acceleration_from_linear_angular_one_framed_operand_inherits_frame_tags() -> None:
    """ID: SPATIAL_CORE_033_acceleration_from_linear_angular_one_framed_operand_inherits_frame_tags."""
    linear_framed = frame_retag(_linear_acceleration(), parent="world", child="body", validate=True)
    angular_unframed = _angular_acceleration()
    acc_linear_framed = Acceleration.from_linear_angular(linear_framed, angular_unframed, validate=True)
    assert get_frames(acc_linear_framed.unsafe_data) == ("world", "body")
    assert set(read_components(acc_linear_framed).keys()) == {"linear", "angular"}

    linear_unframed = _linear_acceleration()
    angular_framed = frame_retag(_angular_acceleration(), parent="odom", child="sensor", validate=True)
    acc_angular_framed = Acceleration.from_linear_angular(linear_unframed, angular_framed, validate=True)
    assert get_frames(acc_angular_framed.unsafe_data) == ("odom", "sensor")
    assert set(read_components(acc_angular_framed).keys()) == {"linear", "angular"}


def test_bcast_core_054_kinematics_component_assembly_preserves_declared_param_and_sequence_size_roles_after_alignment() -> None:
    """ID: BCAST_CORE_054_kinematics_component_assembly_preserves_declared_param_and_sequence_size_roles_after_alignment."""
    linear = LinearVelocity(
        _batched_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            include_trial_dim=True,
        )
    )
    angular = AngularVelocity(
        _batched_vector3_dataset(
            var_name="angular_velocity",
            core_dim="angular_axis",
            include_trial_dim=False,
        )
    )
    out = Velocity.from_linear_angular(linear, angular, validate=True)
    assert read_param_coord_name(out.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(out.unsafe_data) == "sample_size"
    angular_spec = read_components(out)["angular"]
    assert angular_spec.var is not None
    assert out.unsafe_data[angular_spec.var].sizes["trial"] == 2


def test_bcast_hard_046_component_assembly_frame_mismatch_fails_before_topology_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: BCAST_HARD_046_component_assembly_frame_mismatch_fails_before_topology_selection."""
    linear = frame_retag(_linear_velocity(), parent="world", child="body", validate=True)
    angular = frame_retag(_angular_velocity(), parent="map", child="imu", validate=True)

    def _boom(*_args: object, **_kwargs: object):
        raise AssertionError("topology selection should not run before frame precheck")

    monkeypatch.setattr(family_module, "select_topology_policy_with_intents", _boom)
    with pytest.raises(ValueError, match="frame tags must match exactly"):
        Velocity.from_linear_angular(linear, angular, validate=True)


def test_spatial_core_030_kinematics_type_routing_deterministic() -> None:
    """ID: SPATIAL_CORE_030_kinematics_type_routing_deterministic."""
    vel = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    acc = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)

    vel_linear = vel.linear(validate=True)
    vel_angular = vel.angular(validate=True)
    acc_linear = acc.linear(validate=True)
    acc_angular = acc.angular(validate=True)

    assert isinstance(vel_linear, LinearVelocity)
    assert isinstance(vel_angular, AngularVelocity)
    assert isinstance(acc_linear, LinearAcceleration)
    assert isinstance(acc_angular, AngularAcceleration)


def test_spatial_core_031_kinematics_representation_policy_alignment_locked() -> None:
    """ID: SPATIAL_CORE_031_kinematics_representation_policy_alignment_locked."""
    vel = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    acc = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)

    assert get_linear_velocity_rep(_linear_velocity().unsafe_data, owner="test") == "cart"
    assert get_angular_velocity_rep(_angular_velocity().unsafe_data, owner="test") == "cart"
    assert get_velocity_rep(vel.unsafe_data, owner="test") == "components"
    assert get_linear_acceleration_rep(_linear_acceleration().unsafe_data, owner="test") == "cart"
    assert get_angular_acceleration_rep(_angular_acceleration().unsafe_data, owner="test") == "cart"
    assert get_acceleration_rep(acc.unsafe_data, owner="test") == "components"


def test_spatial_hard_025_linear_velocity_constructor_type_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_025_linear_velocity_constructor_type_mismatch_fail_closed."""
    with pytest.raises(TypeError, match="spatial\\.linear_velocity\\.__init__"):
        LinearVelocity(object())


def test_spatial_hard_026_angular_velocity_constructor_type_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_026_angular_velocity_constructor_type_mismatch_fail_closed."""
    with pytest.raises(TypeError, match="spatial\\.angular_velocity\\.__init__"):
        AngularVelocity(object())


def test_spatial_hard_027_velocity_spatial6_constructor_semantic_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_027_velocity_spatial6_constructor_semantic_mismatch_fail_closed."""
    vel = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    wrong_rep = _set_rep(vel.unsafe_data, "cart")
    with pytest.raises(ValueError, match="unsupported velocity representation"):
        Velocity(wrong_rep)

    wrong_kind = _set_roles(vel.unsafe_data, {"kinematics_kind": "linear_velocity"})
    with pytest.raises(ValueError, match="kinematics_kind must be 'velocity'"):
        Velocity(wrong_kind)


def test_spatial_hard_028_linear_acceleration_constructor_type_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_028_linear_acceleration_constructor_type_mismatch_fail_closed."""
    with pytest.raises(TypeError, match="spatial\\.linear_acceleration\\.__init__"):
        LinearAcceleration(object())


def test_spatial_hard_029_angular_acceleration_constructor_type_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_029_angular_acceleration_constructor_type_mismatch_fail_closed."""
    with pytest.raises(TypeError, match="spatial\\.angular_acceleration\\.__init__"):
        AngularAcceleration(object())


def test_spatial_hard_030_acceleration_spatial6_constructor_semantic_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_030_acceleration_spatial6_constructor_semantic_mismatch_fail_closed."""
    acc = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)
    wrong_rep = _set_rep(acc.unsafe_data, "cart")
    with pytest.raises(ValueError, match="unsupported acceleration representation"):
        Acceleration(wrong_rep)

    wrong_kind = _set_roles(acc.unsafe_data, {"kinematics_kind": "angular_acceleration"})
    with pytest.raises(ValueError, match="kinematics_kind must be 'acceleration'"):
        Acceleration(wrong_kind)


def test_spatial_hard_115_velocity_component_var_missing_declared_non_core_dims_rejected_by_core_component_runtime_checks() -> None:
    """ID: SPATIAL_HARD_115_velocity_component_var_missing_declared_non_core_dims_rejected_by_core_component_runtime_checks."""
    ds = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True).unsafe_data
    linear_spec = read_components(Velocity(ds))["linear"]
    linear_values = ds[linear_spec.var].isel(sample=0).values
    ds = ds.drop_vars(linear_spec.var)
    ds[linear_spec.var] = xr.DataArray(
        linear_values,
        dims=(linear_spec.core_dim,),
        coords={linear_spec.core_dim: list(_XYZ)},
    )
    with pytest.raises(ValueError, match="spatial.velocity.__init__"):
        Velocity(ds)
    with pytest.raises(ValueError, match="missing required dims"):
        Velocity(ds)


def test_spatial_hard_031_kinematics_constructor_rejects_malformed_frames_block() -> None:
    """ID: SPATIAL_HARD_031_kinematics_constructor_rejects_malformed_frames_block."""
    ds = _set_bad_frames(_vector3_dataset(var_name="linear_velocity"))
    with pytest.raises(SchemaError, match="tal.ext.frames.extra"):
        LinearVelocity(ds)


def test_spatial_hard_032_kinematics_constructor_rejects_malformed_spatial_roles_shape() -> None:
    """ID: SPATIAL_HARD_032_kinematics_constructor_rejects_malformed_spatial_roles_shape."""
    ds_non_mapping = _set_roles(_vector3_dataset(var_name="linear_velocity"), 123)
    with pytest.raises(ValueError, match="tal.ext.spatial.roles must be a mapping"):
        LinearVelocity(ds_non_mapping)

    ds_non_string_key = _set_roles(_vector3_dataset(var_name="linear_velocity"), {1: "bad"})
    with pytest.raises(ValueError, match="tal.ext.spatial.roles keys must be strings"):
        LinearVelocity(ds_non_string_key)


def test_spatial_core_105_temporal_vector_like_nearest_and_linear_methods_deterministic() -> None:
    """ID: SPATIAL_CORE_105_temporal_vector_like_nearest_and_linear_methods_deterministic."""
    ds = _vector3_dataset(var_name="linear_velocity", core_dim="axis")
    ds = ds.assign_coords(time_s=("sample", [0.0, 1.0]))
    lin = LinearVelocity(
        AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
            validate=True,
        )
    )
    nearest = lin.param.at([0.49], opts=ParamEvalOptions(method="nearest"))
    linear = lin.param.at([0.5], opts=ParamEvalOptions(method="linear"))
    assert isinstance(nearest, LinearVelocity)
    assert isinstance(linear, LinearVelocity)
    np.testing.assert_allclose(nearest.unsafe_data["linear_velocity"].isel(sample=0).values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(linear.unsafe_data["linear_velocity"].isel(sample=0).values, [2.5, 3.5, 4.5])


def test_spatial_core_107_temporal_vector_like_batch_isolation_no_cross_batch_interpolation() -> None:
    """ID: SPATIAL_CORE_107_temporal_vector_like_batch_isolation_no_cross_batch_interpolation."""
    lin = LinearVelocity(
        _batched_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            include_trial_dim=True,
        )
    )
    query = xr.DataArray(
        np.asarray([[0.25, 0.5], [0.75, 0.25]], dtype="float64"),
        dims=("trial", "query"),
        coords={"trial": ["t0", "t1"], "query": [0, 1]},
    )
    out = lin.param.at(query, opts=ParamEvalOptions(method="linear", query_dim="query"))
    assert isinstance(out, LinearVelocity)
    t0 = out.unsafe_data["linear_velocity"].sel(trial="t0").isel(sample=0).values
    t1 = out.unsafe_data["linear_velocity"].sel(trial="t1").isel(sample=0).values
    assert not np.allclose(t0, t1)


def test_spatial_hard_131_temporal_vector_like_no_hidden_batch_broadcast_bleed() -> None:
    """ID: SPATIAL_HARD_131_temporal_vector_like_no_hidden_batch_broadcast_bleed."""
    lin = LinearVelocity(
        _batched_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            include_trial_dim=True,
        )
    )
    query = xr.DataArray(
        np.asarray([[0.5], [0.5]], dtype="float64"),
        dims=("trial", "query"),
        coords={"trial": ["t0", "t1"], "query": [0]},
    )
    out = lin.param.at(query, opts=ParamEvalOptions(method="linear", query_dim="query"))
    row0 = out.unsafe_data["linear_velocity"].sel(trial="t0").isel(sample=0).values
    row1 = out.unsafe_data["linear_velocity"].sel(trial="t1").isel(sample=0).values
    assert not np.allclose(row0, row1)


def test_spatial_core_129_temporal_vector_like_interp_uses_specified_param_coord_as_primary_domain_key() -> None:
    """ID: SPATIAL_CORE_129_temporal_vector_like_interp_uses_specified_param_coord_as_primary_domain_key."""
    ds = _vector3_dataset(var_name="linear_velocity", core_dim="axis")
    ds = ds.assign_coords(
        time_s=("sample", [0.0, 1.0]),
        alt_time=("sample", [0.0, 10.0]),
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
    lin = LinearVelocity(ao)
    out_default = lin.param.at([0.5], on="time_s", opts=ParamEvalOptions(method="linear"))
    out_alt = lin.param.at([5.0], on="alt_time", opts=ParamEvalOptions(method="linear"))
    np.testing.assert_allclose(
        out_default.unsafe_data["linear_velocity"].isel(sample=0).values,
        out_alt.unsafe_data["linear_velocity"].isel(sample=0).values,
    )


def test_spatial_core_120_kinematics_derivative_baseline_acceleration_from_velocity_boundary() -> None:
    """ID: SPATIAL_CORE_120_kinematics_derivative_baseline_acceleration_from_velocity_boundary."""
    linear = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        )
    )
    angular = AngularVelocity(
        _temporal_vector3_dataset(
            var_name="angular_velocity",
            core_dim="angular_axis",
            values=np.asarray([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        )
    )
    linear_out = linear.differentiate(validate=True)
    angular_out = angular.differentiate(validate=True)
    assert isinstance(linear_out, LinearAcceleration)
    assert isinstance(angular_out, AngularAcceleration)


def test_spatial_core_121_kinematics_integral_baseline_position_from_velocity_boundary() -> None:
    """ID: SPATIAL_CORE_121_kinematics_integral_baseline_position_from_velocity_boundary."""
    linear = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        )
    )
    out = linear.integrate(validate=True)
    assert isinstance(out, Position)
    assert out.unsafe_data.sizes["sample"] == linear.unsafe_data.sizes["sample"]


def test_spatial_core_151_kinematics_integral_simpson_same_length_baseline_and_trailing_invalid_nan() -> None:
    """ID: SPATIAL_CORE_151_kinematics_integral_simpson_same_length_baseline_and_trailing_invalid_nan."""
    values = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [9.9, 9.9, 9.9]],
        ],
        dtype="float64",
    )
    ds = xr.Dataset(
        data_vars={"linear_velocity": (("sample", "trial", "linear_axis"), values)},
        coords={
            "sample": [0, 1, 2, 3, 4],
            "trial": ["t0", "t1"],
            "linear_axis": list(_XYZ),
            "time_s": ("sample", [0.0, 1.0, 2.0, 3.0, 4.0]),
            "sample_size": ("trial", [5, 3]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("linear_axis",),
        param_coord="time_s",
        sequence_size_coord="sample_size",
        validate=True,
    )
    out = LinearVelocity(ao).integrate(
        opts=KinematicsIntegralOptions(method="simpson", initial_value=2.0),
        validate=True,
    )
    assert isinstance(out, Position)
    assert out.unsafe_data.sizes["sample"] == 5
    np.testing.assert_allclose(
        out.unsafe_data["linear_velocity"].isel(sample=0).values,
        np.asarray([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype="float64"),
    )
    assert np.isnan(out.unsafe_data["linear_velocity"].sel(trial="t1").isel(sample=4)).all()


def test_spatial_core_154_kinematics_integral_simpson_honors_valid_prefix_with_padded_tail_coords() -> None:
    """ID: SPATIAL_CORE_154_kinematics_integral_simpson_honors_valid_prefix_with_padded_tail_coords."""
    values = np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
        dtype="float64",
    )
    tails = (np.asarray([2.0, 2.0], dtype="float64"), np.asarray([1.5, 1.0], dtype="float64"))
    for tail in tails:
        param = np.concatenate([np.asarray([0.0, 1.0, 2.0], dtype="float64"), tail])
        ds = xr.Dataset(
            data_vars={"linear_velocity": (("sample", "linear_axis"), values)},
            coords={
                "sample": [0, 1, 2, 3, 4],
                "linear_axis": list(_XYZ),
                "time_s": ("sample", param),
                "group_size": xr.DataArray(np.asarray(3, dtype="int64"), dims=()),
            },
        )
        ao = AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            core_dims=("linear_axis",),
            param_coord="time_s",
            sequence_size_coord="group_size",
            validate=True,
        )
        out = LinearVelocity(ao).integrate(
            opts=KinematicsIntegralOptions(method="simpson", initial_value=5.0),
            validate=True,
        )
        assert out.unsafe_data.sizes["sample"] == 5
        np.testing.assert_allclose(
            out.unsafe_data["linear_velocity"].isel(sample=0).values,
            np.asarray([5.0, 5.0, 5.0], dtype="float64"),
        )
        assert np.isnan(out.unsafe_data["linear_velocity"].isel(sample=3)).all()
        assert np.isnan(out.unsafe_data["linear_velocity"].isel(sample=4)).all()


def test_spatial_core_122_kinematics_integral_baseline_velocity_from_acceleration_boundary() -> None:
    """ID: SPATIAL_CORE_122_kinematics_integral_baseline_velocity_from_acceleration_boundary."""
    linear = LinearAcceleration(
        _temporal_vector3_dataset(
            var_name="linear_acceleration",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        )
    )
    angular = AngularAcceleration(
        _temporal_vector3_dataset(
            var_name="angular_acceleration",
            core_dim="angular_axis",
            values=np.asarray([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        )
    )
    linear_out = linear.integrate(validate=True)
    angular_out = angular.integrate(validate=True)
    assert isinstance(linear_out, LinearVelocity)
    assert isinstance(angular_out, AngularVelocity)


def test_spatial_core_123_kinematics_methods_support_nonuniform_param_spacing_baseline() -> None:
    """ID: SPATIAL_CORE_123_kinematics_methods_support_nonuniform_param_spacing_baseline."""
    linear = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 0.5, 2.0], dtype="float64"),
        )
    )
    out = linear.differentiate(validate=True)
    np.testing.assert_allclose(out.unsafe_data["linear_velocity"].isel(linear_axis=0).values, [0.5, 0.5, 2.5])


def test_spatial_core_125_kinematics_frame_kind_rep_truthfulness_preserved_through_derivative_integral() -> None:
    """ID: SPATIAL_CORE_125_kinematics_frame_kind_rep_truthfulness_preserved_through_derivative_integral."""
    linear = frame_retag(
        LinearVelocity(
            _temporal_vector3_dataset(
                var_name="linear_velocity",
                core_dim="linear_axis",
                values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype="float64"),
                param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
            )
        ),
        parent="world",
        child="body",
        validate=True,
    )
    acc = linear.differentiate(validate=True)
    pos = linear.integrate(validate=True)
    assert get_frames(acc.unsafe_data) == ("world", "body")
    assert get_frames(pos.unsafe_data) == ("world", "body")
    assert get_kinematics_kind(acc.unsafe_data, owner="test") == "linear_acceleration"
    assert get_linear_acceleration_rep(acc.unsafe_data, owner="test") == "cart"
    assert get_position_intent(pos.unsafe_data, owner="test") == "delta"


def test_spatial_core_126_kinematics_temporal_validity_and_sequence_size_metadata_truthful() -> None:
    """ID: SPATIAL_CORE_126_kinematics_temporal_validity_and_sequence_size_metadata_truthful."""
    linear = LinearVelocity(
        _batched_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            include_trial_dim=True,
        )
    )
    out = linear.differentiate(validate=True)
    assert read_param_coord_name(out.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(out.unsafe_data) == "sample_size"
    np.testing.assert_array_equal(out.unsafe_data.coords["sample_size"].values, [2, 2])


def test_spatial_core_127_kinematics_temporal_dask_lazy_boundary_preserved() -> None:
    """ID: SPATIAL_CORE_127_kinematics_temporal_dask_lazy_boundary_preserved."""
    dask_array = pytest.importorskip("dask.array")
    values = dask_array.from_array(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype="float64"),
        chunks=(3, 3),
    )
    ds = xr.Dataset(
        data_vars={"linear_velocity": (("sample", "axis"), values)},
        coords={"sample": [0, 1, 2], "axis": ["x", "y", "z"], "time_s": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), param_coord="time_s", validate=True)
    out = LinearVelocity(ao).differentiate(validate=True)
    assert getattr(out.unsafe_data["linear_velocity"].data, "chunks", None) is not None


def test_spatial_core_128_kinematics_batch_isolation_no_cross_batch_transport_bleed() -> None:
    """ID: SPATIAL_CORE_128_kinematics_batch_isolation_no_cross_batch_transport_bleed."""
    linear = LinearVelocity(
        _batched_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            include_trial_dim=True,
        )
    )
    out = linear.differentiate(validate=True)
    row0 = out.unsafe_data["linear_velocity"].sel(trial="t0").isel(sample=0).values
    row1 = out.unsafe_data["linear_velocity"].sel(trial="t1").isel(sample=0).values
    assert not np.allclose(row0, row1)


def test_spatial_core_142_linear_velocity_integrate_outputs_baseline_relative_position_semantics() -> None:
    """ID: SPATIAL_CORE_142_linear_velocity_integrate_outputs_baseline_relative_position_semantics."""
    linear = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        )
    )
    out = linear.integrate(validate=True)
    assert isinstance(out, Position)
    assert get_position_intent(out.unsafe_data, owner="test") == "delta"
    np.testing.assert_allclose(out.unsafe_data["linear_velocity"].isel(linear_axis=0).values, [0.0, 1.0, 2.0])


def test_spatial_hard_141_kinematics_integral_requires_numeric_monotonic_param_domain() -> None:
    """ID: SPATIAL_HARD_141_kinematics_integral_requires_numeric_monotonic_param_domain."""
    linear = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 2.0, 1.0], dtype="float64"),
        )
    )
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.integrate"):
        _ = linear.integrate()


def test_spatial_hard_152_kinematics_temporal_param_key_paths_do_not_fallback_to_sequence_label_matching() -> None:
    """ID: SPATIAL_HARD_152_kinematics_temporal_param_key_paths_do_not_fallback_to_sequence_label_matching."""
    ds = _temporal_vector3_dataset(
        var_name="linear_velocity",
        core_dim="linear_axis",
        values=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="float64"),
        param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
    )
    ds = ds.assign_coords(alt_time=("sample", [0.0, 10.0, 20.0]))
    linear = LinearVelocity(ds)
    out_default = linear.differentiate(on="time_s", validate=True)
    out_alt = linear.differentiate(on="alt_time", validate=True)
    np.testing.assert_allclose(
        out_default.unsafe_data["linear_velocity"].isel(linear_axis=0).values,
        10.0 * out_alt.unsafe_data["linear_velocity"].isel(linear_axis=0).values,
    )


def test_spatial_core_157_d6_velocity_wrapper_differentiate_and_smooth_preserve_rep_and_frames() -> None:
    """ID: SPATIAL_CORE_157_d6_velocity_wrapper_differentiate_and_smooth_preserve_rep_and_frames."""
    base_linear = frame_retag(
        LinearVelocity(
            _temporal_vector3_dataset(
                var_name="linear_velocity",
                core_dim="linear_axis",
                values=np.asarray(
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                    dtype="float64",
                ),
                param=np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64"),
            )
        ),
        parent="world",
        child="body",
        validate=True,
    )
    base_angular = frame_retag(
        AngularVelocity(
            _temporal_vector3_dataset(
                var_name="angular_velocity",
                core_dim="angular_axis",
                values=np.asarray(
                    [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    dtype="float64",
                ),
                param=np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64"),
            )
        ),
        parent="world",
        child="body",
        validate=True,
    )
    for rep in ("components", "vector6"):
        velocity = Velocity.from_linear_angular(base_linear, base_angular, validate=True).to_rep(rep, validate=True)
        differentiated = velocity.differentiate(validate=True)
        smoothed = velocity.smooth(validate=True)
        assert isinstance(differentiated, Acceleration)
        assert isinstance(smoothed, Velocity)
        assert get_acceleration_rep(differentiated.unsafe_data, owner="test") == rep
        assert get_velocity_rep(smoothed.unsafe_data, owner="test") == rep
        assert get_frames(differentiated.unsafe_data) == ("world", "body")
        assert get_frames(smoothed.unsafe_data) == ("world", "body")
        assert get_kinematics_kind(differentiated.unsafe_data, owner="test") == "acceleration"
        assert get_kinematics_kind(smoothed.unsafe_data, owner="test") == "velocity"


def test_spatial_core_158_d6_acceleration_wrapper_integrate_and_smooth_preserve_rep_and_frames() -> None:
    """ID: SPATIAL_CORE_158_d6_acceleration_wrapper_integrate_and_smooth_preserve_rep_and_frames."""
    base_linear = frame_retag(
        LinearAcceleration(
            _temporal_vector3_dataset(
                var_name="linear_acceleration",
                core_dim="linear_axis",
                values=np.asarray(
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                    dtype="float64",
                ),
                param=np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64"),
            )
        ),
        parent="world",
        child="body",
        validate=True,
    )
    base_angular = frame_retag(
        AngularAcceleration(
            _temporal_vector3_dataset(
                var_name="angular_acceleration",
                core_dim="angular_axis",
                values=np.asarray(
                    [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    dtype="float64",
                ),
                param=np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64"),
            )
        ),
        parent="world",
        child="body",
        validate=True,
    )
    for rep in ("components", "vector6"):
        acceleration = Acceleration.from_linear_angular(base_linear, base_angular, validate=True).to_rep(rep, validate=True)
        integrated = acceleration.integrate(validate=True)
        smoothed = acceleration.smooth(validate=True)
        assert isinstance(integrated, Velocity)
        assert isinstance(smoothed, Acceleration)
        assert get_velocity_rep(integrated.unsafe_data, owner="test") == rep
        assert get_acceleration_rep(smoothed.unsafe_data, owner="test") == rep
        assert get_frames(integrated.unsafe_data) == ("world", "body")
        assert get_frames(smoothed.unsafe_data) == ("world", "body")
        assert get_kinematics_kind(integrated.unsafe_data, owner="test") == "velocity"
        assert get_kinematics_kind(smoothed.unsafe_data, owner="test") == "acceleration"


def test_spatial_core_143_kinematics_typed_smoothing_boundaries_present() -> None:
    """ID: SPATIAL_CORE_143_kinematics_typed_smoothing_boundaries_present."""
    assert hasattr(Position, "smooth")
    assert hasattr(LinearVelocity, "smooth")
    assert hasattr(AngularVelocity, "smooth")
    assert hasattr(LinearAcceleration, "smooth")
    assert hasattr(AngularAcceleration, "smooth")


def test_spatial_core_177_kinematics_vector_like_norm_magnitude_value_parity() -> None:
    """ID: SPATIAL_CORE_177_kinematics_vector_like_norm_magnitude_value_parity."""
    linear = LinearVelocity(
        _vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]], dtype="float64"),
        )
    )
    angular = AngularVelocity(
        _vector3_dataset(
            var_name="angular_velocity",
            core_dim="angular_axis",
            values=np.asarray([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]], dtype="float64"),
        )
    )
    lin_l2 = linear.norm(ord=2)
    ang_l1 = angular.norm(ord=1)

    assert isinstance(lin_l2, Array)
    assert isinstance(ang_l1, Array)
    np.testing.assert_allclose(lin_l2.unsafe_data["datavar"].values, np.asarray([5.0, 3.0], dtype="float64"))
    np.testing.assert_allclose(ang_l1.unsafe_data["datavar"].values, np.asarray([3.0, 15.0], dtype="float64"))
    xr.testing.assert_identical(lin_l2.unsafe_data, linear.magnitude().unsafe_data)
    xr.testing.assert_identical(angular.norm(ord=2).unsafe_data, angular.magnitude().unsafe_data)


def test_spatial_core_178_kinematics_acceleration_magnitude_preserves_scalar_topology_and_metadata() -> None:
    """ID: SPATIAL_CORE_178_kinematics_acceleration_magnitude_preserves_scalar_topology_and_metadata."""
    linear = LinearAcceleration(
        _batched_vector3_dataset(
            var_name="linear_acceleration",
            core_dim="linear_axis",
            include_trial_dim=True,
        )
    )
    angular = AngularAcceleration(
        _batched_vector3_dataset(
            var_name="angular_acceleration",
            core_dim="angular_axis",
            include_trial_dim=True,
        )
    )
    linear_mag = linear.magnitude()
    angular_norm = angular.norm(ord=2)

    assert isinstance(linear_mag, Array)
    assert isinstance(angular_norm, Array)
    assert tuple(linear_mag.unsafe_data.data_vars) == ("datavar",)
    declared, sequence_dim, batch_dims, core_dims = read_roles(linear_mag.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ()
    assert read_param_coord_name(linear_mag.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(linear_mag.unsafe_data) == "sample_size"
    expected_angular = xr.apply_ufunc(
        np.linalg.norm,
        angular.unsafe_data["angular_acceleration"],
        input_core_dims=[["angular_axis"]],
        output_core_dims=[[]],
        vectorize=False,
        dask="allowed",
        kwargs={"ord": 2, "axis": -1},
    ).rename("datavar")
    xr.testing.assert_allclose(angular_norm.unsafe_data["datavar"], expected_angular)
    xr.testing.assert_identical(angular_norm.unsafe_data, angular.magnitude().unsafe_data)


def test_spatial_hard_164_kinematics_temporal_placeholder_methods_fail_closed() -> None:
    """ID: SPATIAL_HARD_164_kinematics_temporal_placeholder_methods_fail_closed."""
    linear = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        )
    )
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.differentiate"):
        _ = linear.differentiate(opts=KinematicsDerivativeOptions(method="finite_difference", order=2))
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.integrate"):
        _ = linear.integrate(opts=KinematicsIntegralOptions(method="romberg"))


def test_spatial_hard_166_kinematics_local_derivative_fails_closed_on_unsupported_method_or_order() -> None:
    """ID: SPATIAL_HARD_166_kinematics_local_derivative_fails_closed_on_unsupported_method_or_order."""
    linear = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        )
    )
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.differentiate"):
        _ = linear.differentiate(
            opts=KinematicsDerivativeOptions(method="local_poly", order=2, edge_mode="partial_renorm")
        )
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.smooth"):
        _ = linear.smooth(opts=KinematicsSmoothingOptions(method="local_poly", window=3, poly_order=3))


def test_spatial_hard_171_kinematics_integral_simpson_requires_left_packed_min_three_valid_samples() -> None:
    """ID: SPATIAL_HARD_171_kinematics_integral_simpson_requires_left_packed_min_three_valid_samples."""
    short = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0], dtype="float64"),
        )
    )
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.integrate"):
        _ = short.integrate(opts=KinematicsIntegralOptions(method="simpson"), validate=True)
    values = np.asarray([[[1.0], [2.0], [3.0], [4.0]]], dtype="float64")
    param = np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype="float64")
    non_left_packed = np.asarray([[True, False, True, True]], dtype=bool)
    with pytest.raises(ValueError, match="left-packed"):
        _ = cumulative_simpson_kernel(values, param, non_left_packed, initial_value=0.0)
