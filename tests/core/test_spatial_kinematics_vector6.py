from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from tal import AnalysisObject
from tal.core import read_components
from tal.core.schema_errors import SchemaError
from tal.core.schema_read import read_roles
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Velocity,
)
from tal.spatial.kinematics.vector6_ops import vector6_labels
from tal.spatial.metadata import get_acceleration_rep, get_kinematics_kind, get_velocity_rep
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames

_XYZ = ("x", "y", "z")


def _vector3_dataset(
    *,
    var_name: str,
    core_dim: str,
    values: np.ndarray,
    with_batch: bool = False,
) -> xr.Dataset:
    if with_batch:
        dims = ("sample", "sensor", core_dim)
        coords = {"sample": [0, 1], "sensor": ["a", "b"], core_dim: list(_XYZ)}
    else:
        dims = ("sample", core_dim)
        coords = {"sample": [0, 1], core_dim: list(_XYZ)}
    arr = xr.DataArray(values, dims=dims, coords=coords, name=var_name)
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        batch_dims=("sensor",) if with_batch else (),
        core_dims=(core_dim,),
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _linear_velocity(*, with_batch: bool = False) -> LinearVelocity:
    values = (
        np.asarray(
            [
                [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=float,
        )
        if with_batch
        else np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    )
    return LinearVelocity(_vector3_dataset(var_name="linear_velocity", core_dim="linear_axis", values=values, with_batch=with_batch))


def _angular_velocity(*, with_batch: bool = False) -> AngularVelocity:
    values = (
        np.asarray(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
                [[0.0, 1.0, 0.0], [0.0, 0.2, 0.0]],
            ],
            dtype=float,
        )
        if with_batch
        else np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)
    )
    return AngularVelocity(_vector3_dataset(var_name="angular_velocity", core_dim="angular_axis", values=values, with_batch=with_batch))


def _linear_acceleration(*, with_batch: bool = False) -> LinearAcceleration:
    values = (
        np.asarray(
            [
                [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                [[0.0, 0.2, 0.0], [0.0, 0.1, 0.0]],
            ],
            dtype=float,
        )
        if with_batch
        else np.asarray([[1.0, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=float)
    )
    return LinearAcceleration(
        _vector3_dataset(var_name="linear_acceleration", core_dim="linear_axis", values=values, with_batch=with_batch)
    )


def _angular_acceleration(*, with_batch: bool = False) -> AngularAcceleration:
    values = (
        np.asarray(
            [
                [[0.0, 0.0, 0.2], [0.0, 0.0, 0.1]],
                [[0.0, 0.3, 0.0], [0.0, 0.4, 0.0]],
            ],
            dtype=float,
        )
        if with_batch
        else np.asarray([[0.0, 0.0, 0.2], [0.0, 0.3, 0.0]], dtype=float)
    )
    return AngularAcceleration(
        _vector3_dataset(var_name="angular_acceleration", core_dim="angular_axis", values=values, with_batch=with_batch)
    )


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


def _vector6_dataset(*, var_name: str, core_dim: str, values: np.ndarray) -> xr.Dataset:
    arr = xr.DataArray(
        values,
        dims=("sample", core_dim),
        coords={"sample": [0, 1], core_dim: list(vector6_labels())},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(core_dim,),
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def test_spatial_core_061_velocity_vector6_constructor_and_to_rep_boundary_deterministic() -> None:
    """ID: SPATIAL_CORE_061_velocity_vector6_constructor_and_to_rep_boundary_deterministic."""
    velocity = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    as_vector6 = velocity.as_vector6(validate=True)
    assert get_velocity_rep(as_vector6.unsafe_data, owner="test") == "vector6"
    vec_dim = read_roles(as_vector6.unsafe_data)[3][0]
    assert tuple(as_vector6.unsafe_data.get_index(vec_dim).tolist()) == vector6_labels()
    restored = Velocity.from_vector6(as_vector6.unsafe_data, validate=True).to_rep("components", validate=True)
    np.testing.assert_allclose(restored.linear().unsafe_data["linear_velocity"].values, velocity.linear().unsafe_data["linear_velocity"].values)
    np.testing.assert_allclose(restored.angular().unsafe_data["angular_velocity"].values, velocity.angular().unsafe_data["angular_velocity"].values)


def test_spatial_core_062_acceleration_vector6_constructor_and_to_rep_boundary_deterministic() -> None:
    """ID: SPATIAL_CORE_062_acceleration_vector6_constructor_and_to_rep_boundary_deterministic."""
    acceleration = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)
    as_vector6 = acceleration.as_vector6(validate=True)
    assert get_acceleration_rep(as_vector6.unsafe_data, owner="test") == "vector6"
    vec_dim = read_roles(as_vector6.unsafe_data)[3][0]
    assert tuple(as_vector6.unsafe_data.get_index(vec_dim).tolist()) == vector6_labels()
    restored = Acceleration.from_vector6(as_vector6.unsafe_data, validate=True).to_rep("components", validate=True)
    np.testing.assert_allclose(
        restored.linear().unsafe_data["linear_acceleration"].values,
        acceleration.linear().unsafe_data["linear_acceleration"].values,
    )
    np.testing.assert_allclose(
        restored.angular().unsafe_data["angular_acceleration"].values,
        acceleration.angular().unsafe_data["angular_acceleration"].values,
    )


def test_spatial_core_063_velocity_components_vector6_roundtrip_preserves_frames_roles_semantics() -> None:
    """ID: SPATIAL_CORE_063_velocity_components_vector6_roundtrip_preserves_frames_roles_semantics."""
    velocity = frame_retag(
        Velocity.from_linear_angular(_linear_velocity(with_batch=True), _angular_velocity(with_batch=True), validate=True),
        parent="world",
        child="body",
        validate=True,
    )
    vector6 = velocity.as_vector6(validate=True)
    restored = vector6.as_components(validate=True)
    assert get_frames(vector6.unsafe_data) == ("world", "body")
    assert get_frames(restored.unsafe_data) == ("world", "body")
    assert read_roles(vector6.unsafe_data)[1:3] == read_roles(velocity.unsafe_data)[1:3]
    np.testing.assert_allclose(restored.linear().unsafe_data["linear_velocity"].values, velocity.linear().unsafe_data["linear_velocity"].values)
    np.testing.assert_allclose(restored.angular().unsafe_data["angular_velocity"].values, velocity.angular().unsafe_data["angular_velocity"].values)


def test_spatial_core_064_acceleration_components_vector6_roundtrip_preserves_frames_roles_semantics() -> None:
    """ID: SPATIAL_CORE_064_acceleration_components_vector6_roundtrip_preserves_frames_roles_semantics."""
    acceleration = frame_retag(
        Acceleration.from_linear_angular(_linear_acceleration(with_batch=True), _angular_acceleration(with_batch=True), validate=True),
        parent="map",
        child="imu",
        validate=True,
    )
    vector6 = acceleration.as_vector6(validate=True)
    restored = vector6.as_components(validate=True)
    assert get_frames(vector6.unsafe_data) == ("map", "imu")
    assert get_frames(restored.unsafe_data) == ("map", "imu")
    assert read_roles(vector6.unsafe_data)[1:3] == read_roles(acceleration.unsafe_data)[1:3]
    np.testing.assert_allclose(
        restored.linear().unsafe_data["linear_acceleration"].values,
        acceleration.linear().unsafe_data["linear_acceleration"].values,
    )
    np.testing.assert_allclose(
        restored.angular().unsafe_data["angular_acceleration"].values,
        acceleration.angular().unsafe_data["angular_acceleration"].values,
    )


def test_spatial_core_065_velocity_linear_angular_extract_deterministic_from_vector6_rep() -> None:
    """ID: SPATIAL_CORE_065_velocity_linear_angular_extract_deterministic_from_vector6_rep."""
    velocity = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True).as_vector6(validate=True)
    np.testing.assert_allclose(
        velocity.linear(validate=True).unsafe_data["linear_velocity"].values,
        _linear_velocity().unsafe_data["linear_velocity"].values,
    )
    np.testing.assert_allclose(
        velocity.angular(validate=True).unsafe_data["angular_velocity"].values,
        _angular_velocity().unsafe_data["angular_velocity"].values,
    )


def test_spatial_core_066_acceleration_linear_angular_extract_deterministic_from_vector6_rep() -> None:
    """ID: SPATIAL_CORE_066_acceleration_linear_angular_extract_deterministic_from_vector6_rep."""
    acceleration = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True).as_vector6(validate=True)
    np.testing.assert_allclose(
        acceleration.linear(validate=True).unsafe_data["linear_acceleration"].values,
        _linear_acceleration().unsafe_data["linear_acceleration"].values,
    )
    np.testing.assert_allclose(
        acceleration.angular(validate=True).unsafe_data["angular_acceleration"].values,
        _angular_acceleration().unsafe_data["angular_acceleration"].values,
    )


def test_spatial_core_068_vector6_conversion_preserves_kinematics_kind_and_sequence_batch_layout() -> None:
    """ID: SPATIAL_CORE_068_vector6_conversion_preserves_kinematics_kind_and_sequence_batch_layout."""
    velocity = Velocity.from_linear_angular(_linear_velocity(with_batch=True), _angular_velocity(with_batch=True), validate=True)
    vector6 = velocity.as_vector6(validate=True)
    assert get_kinematics_kind(vector6.unsafe_data, owner="test") == "velocity"
    assert read_roles(vector6.unsafe_data)[1:3] == read_roles(velocity.unsafe_data)[1:3]

    acceleration = Acceleration.from_linear_angular(
        _linear_acceleration(with_batch=True),
        _angular_acceleration(with_batch=True),
        validate=True,
    )
    vector6_acc = acceleration.as_vector6(validate=True)
    assert get_kinematics_kind(vector6_acc.unsafe_data, owner="test") == "acceleration"
    assert read_roles(vector6_acc.unsafe_data)[1:3] == read_roles(acceleration.unsafe_data)[1:3]


def test_spatial_hard_066_velocity_to_rep_rejects_unsupported_target_rep() -> None:
    """ID: SPATIAL_HARD_066_velocity_to_rep_rejects_unsupported_target_rep."""
    velocity = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    with pytest.raises(ValueError, match="unsupported velocity representation"):
        velocity.to_rep("cart", validate=True)  # type: ignore[arg-type]


def test_spatial_hard_067_acceleration_to_rep_rejects_unsupported_target_rep() -> None:
    """ID: SPATIAL_HARD_067_acceleration_to_rep_rejects_unsupported_target_rep."""
    acceleration = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)
    with pytest.raises(ValueError, match="unsupported acceleration representation"):
        acceleration.to_rep("cart", validate=True)  # type: ignore[arg-type]


def test_spatial_hard_068_spatial6_vector6_constructor_rejects_invalid_core_length_or_labels() -> None:
    """ID: SPATIAL_HARD_068_spatial6_vector6_constructor_rejects_invalid_core_length_or_labels."""
    velocity = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True).as_vector6(validate=True)
    vec_dim = read_roles(velocity.unsafe_data)[3][0]

    bad_length = velocity.unsafe_data.isel({vec_dim: slice(0, 5)})
    with pytest.raises(ValueError, match="must have length 6"):
        Velocity.from_vector6(bad_length, validate=True)

    bad_labels = velocity.unsafe_data.assign_coords({vec_dim: ["x", "y", "z", "a", "b", "c"]})
    with pytest.raises(ValueError, match="labels must equal"):
        Velocity.from_vector6(bad_labels, validate=True)


def test_spatial_hard_069_velocity_as_vector6_clears_component_registry_truthfully() -> None:
    """ID: SPATIAL_HARD_069_velocity_as_vector6_clears_component_registry_truthfully."""
    velocity = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True)
    vector6 = velocity.as_vector6(validate=True)
    assert read_components(vector6) == {}


def test_spatial_hard_070_acceleration_as_vector6_clears_component_registry_truthfully() -> None:
    """ID: SPATIAL_HARD_070_acceleration_as_vector6_clears_component_registry_truthfully."""
    acceleration = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True)
    vector6 = acceleration.as_vector6(validate=True)
    assert read_components(vector6) == {}


def test_spatial_hard_071_spatial6_as_components_from_vector6_restores_truthful_component_registry() -> None:
    """ID: SPATIAL_HARD_071_spatial6_as_components_from_vector6_restores_truthful_component_registry."""
    velocity = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True).as_vector6(validate=True)
    acceleration = Acceleration.from_linear_angular(_linear_acceleration(), _angular_acceleration(), validate=True).as_vector6(validate=True)
    vel_components = velocity.as_components(validate=True)
    acc_components = acceleration.as_components(validate=True)
    assert set(read_components(vel_components).keys()) == {"linear", "angular"}
    assert set(read_components(acc_components).keys()) == {"linear", "angular"}


def test_spatial_hard_072_vector6_conversion_fail_closed_on_malformed_frames_or_roles() -> None:
    """ID: SPATIAL_HARD_072_vector6_conversion_fail_closed_on_malformed_frames_or_roles."""
    velocity = Velocity.from_linear_angular(_linear_velocity(), _angular_velocity(), validate=True).as_vector6(validate=True)
    with pytest.raises(SchemaError, match="tal\\.ext\\.frames\\.extra"):
        Velocity.from_vector6(_set_bad_frames(velocity.unsafe_data), validate=True)

    with pytest.raises(ValueError, match="tal.ext.spatial.roles must be a mapping"):
        Velocity(_set_roles(velocity.unsafe_data, 123))
    with pytest.raises(ValueError, match="tal.ext.spatial.roles keys must be strings"):
        Velocity(_set_roles(velocity.unsafe_data, {1: "bad"}))


@pytest.mark.parametrize(
    ("linear_cls", "angular_cls", "spatial_cls", "linear_var", "angular_var"),
    [
        (LinearVelocity, AngularVelocity, Velocity, "linear_velocity", "angular_velocity"),
        (LinearAcceleration, AngularAcceleration, Acceleration, "linear_acceleration", "angular_acceleration"),
    ],
)
def test_spatial_hard_074_vector6_pack_dask_lazy_dtype_metadata_preserved(
    linear_cls: type[LinearVelocity] | type[LinearAcceleration],
    angular_cls: type[AngularVelocity] | type[AngularAcceleration],
    spatial_cls: type[Velocity] | type[Acceleration],
    linear_var: str,
    angular_var: str,
) -> None:
    """ID: SPATIAL_HARD_074_vector6_pack_dask_lazy_dtype_metadata_preserved."""
    dask_array = pytest.importorskip("dask.array")
    linear_values = dask_array.from_array(np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64), chunks=(1, 3))
    angular_values = dask_array.from_array(np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.int64), chunks=(1, 3))
    linear = linear_cls(_vector3_dataset(var_name=linear_var, core_dim="linear_axis", values=linear_values))
    angular = angular_cls(_vector3_dataset(var_name=angular_var, core_dim="angular_axis", values=angular_values))
    vector6 = spatial_cls.from_linear_angular(linear, angular, validate=True).as_vector6(validate=True)
    vector_var = next(iter(vector6.unsafe_data.data_vars))
    expected_dtype = np.result_type(linear_values.dtype, angular_values.dtype)
    assert np.dtype(vector6.unsafe_data[vector_var].dtype) == np.dtype(expected_dtype)
    computed = vector6.unsafe_data[vector_var].compute()
    assert np.dtype(computed.dtype) == np.dtype(expected_dtype)


@pytest.mark.parametrize(
    "spatial_cls",
    [Velocity, Acceleration],
)
def test_spatial_hard_075_vector6_unpack_dask_lazy_dtype_metadata_preserved_and_no_complex_cast_warning(
    spatial_cls: type[Velocity] | type[Acceleration],
) -> None:
    """ID: SPATIAL_HARD_075_vector6_unpack_dask_lazy_dtype_metadata_preserved_and_no_complex_cast_warning."""
    dask_array = pytest.importorskip("dask.array")
    values = dask_array.from_array(
        np.asarray(
            [
                [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j, 5.0 + 5.0j, 6.0 + 6.0j],
                [7.0 + 7.0j, 8.0 + 8.0j, 9.0 + 9.0j, 10.0 + 10.0j, 11.0 + 11.0j, 12.0 + 12.0j],
            ],
            dtype=np.complex128,
        ),
        chunks=(1, 6),
    )
    vector6 = spatial_cls.from_vector6(_vector6_dataset(var_name="spatial6", core_dim="vector_axis", values=values), validate=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        linear = vector6.linear(validate=True)
        angular = vector6.angular(validate=True)
        linear_var = next(iter(linear.unsafe_data.data_vars))
        angular_var = next(iter(angular.unsafe_data.data_vars))
        assert np.dtype(linear.unsafe_data[linear_var].dtype) == np.dtype(np.complex128)
        assert np.dtype(angular.unsafe_data[angular_var].dtype) == np.dtype(np.complex128)
        linear.unsafe_data[linear_var].compute()
        angular.unsafe_data[angular_var].compute()
    assert not any(w.category.__name__ == "ComplexWarning" for w in caught)


def test_topo_core_008_vector6_pack_path_uses_core_topology_touchpoint() -> None:
    """ID: TOPO_CORE_008_vector6_pack_path_uses_core_topology_touchpoint."""
    linear = _linear_velocity(with_batch=True)
    angular = _angular_velocity(with_batch=True)
    linear_transposed = LinearVelocity(linear.unsafe_data.transpose("sensor", "sample", "linear_axis"))
    angular_transposed = AngularVelocity(angular.unsafe_data.transpose("angular_axis", "sensor", "sample"))
    out = Velocity.from_linear_angular(linear_transposed, angular_transposed, validate=True).as_vector6(validate=True)
    expected = Velocity.from_linear_angular(linear, angular, validate=True).as_vector6(validate=True)
    out_var = next(iter(out.unsafe_data.data_vars))
    out_da = out.unsafe_data[out_var].transpose(*expected.unsafe_data[out_var].dims)
    np.testing.assert_allclose(out_da.values, expected.unsafe_data[out_var].values, atol=1e-6)
