from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal import AnalysisObject
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.spatial import (
    AO_TEMPORAL_KIND_VALUES,
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Position,
    Velocity,
    differentiate,
    integrate,
    smooth,
)
from tal.spatial.metadata import (
    KINEMATICS_KIND_VALUES,
    get_acceleration_rep,
    get_angular_acceleration_rep,
    get_angular_velocity_rep,
    get_kinematics_kind,
    get_linear_acceleration_rep,
    get_linear_velocity_rep,
    get_position_rep,
    get_velocity_rep,
)

_XYZ = ("x", "y", "z")


def _temporal_vector3_dataset(*, var_name: str, core_dim: str, values: np.ndarray, param: np.ndarray) -> xr.Dataset:
    arr = xr.DataArray(
        values,
        dims=("sample", core_dim),
        coords={"sample": list(range(values.shape[0])), core_dim: list(_XYZ), "time_s": ("sample", param)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=(core_dim,),
        param_coord="time_s",
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _typed_sources() -> dict[str, object]:
    param = np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64")
    pos = Position(
        _temporal_vector3_dataset(
            var_name="position",
            core_dim="axis",
            values=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    lv = LinearVelocity(
        _temporal_vector3_dataset(
            var_name="linear_velocity",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    av = AngularVelocity(
        _temporal_vector3_dataset(
            var_name="angular_velocity",
            core_dim="angular_axis",
            values=np.asarray([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    la = LinearAcceleration(
        _temporal_vector3_dataset(
            var_name="linear_acceleration",
            core_dim="linear_axis",
            values=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    aa = AngularAcceleration(
        _temporal_vector3_dataset(
            var_name="angular_acceleration",
            core_dim="angular_axis",
            values=np.asarray([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="float64"),
            param=param,
        )
    )
    vel = Velocity.from_linear_angular(lv, av, validate=True)
    acc = Acceleration.from_linear_angular(la, aa, validate=True)
    return {
        "position": pos,
        "linear_velocity": lv,
        "angular_velocity": av,
        "velocity": vel,
        "linear_acceleration": la,
        "angular_acceleration": aa,
        "acceleration": acc,
    }


def _spatial_rep(ds: xr.Dataset) -> str:
    kind = get_kinematics_kind(ds, owner="test")
    if kind is None:
        return get_position_rep(ds, owner="test")
    if kind == "linear_velocity":
        return get_linear_velocity_rep(ds, owner="test")
    if kind == "angular_velocity":
        return get_angular_velocity_rep(ds, owner="test")
    if kind == "velocity":
        return get_velocity_rep(ds, owner="test")
    if kind == "linear_acceleration":
        return get_linear_acceleration_rep(ds, owner="test")
    if kind == "angular_acceleration":
        return get_angular_acceleration_rep(ds, owner="test")
    if kind == "acceleration":
        return get_acceleration_rep(ds, owner="test")
    raise AssertionError(f"unsupported spatial kind for test parity: {kind!r}")


def test_spatial_core_155_d6_ao_temporal_dispatch_uses_single_canonical_allowed_kind_constant() -> None:
    """ID: SPATIAL_CORE_155_d6_ao_temporal_dispatch_uses_single_canonical_allowed_kind_constant."""
    assert AO_TEMPORAL_KIND_VALUES == ("position", *KINEMATICS_KIND_VALUES)


def test_spatial_core_156_d6_ao_temporal_surface_delegates_to_existing_typed_or_family_owners() -> None:
    """ID: SPATIAL_CORE_156_d6_ao_temporal_surface_delegates_to_existing_typed_or_family_owners."""
    sources = _typed_sources()
    for kind, source in sources.items():
        typed_out = smooth(source, validate=True, target_cls=source.__class__)
        assert isinstance(typed_out, source.__class__)
        ao = AnalysisObject._from_validated(source.unsafe_data)
        ao_out = smooth(ao, kind=kind, validate=True)
        assert isinstance(ao_out, AnalysisObject)
    diff_expected = {
        "position": LinearVelocity,
        "linear_velocity": LinearAcceleration,
        "angular_velocity": AngularAcceleration,
        "velocity": Acceleration,
    }
    for kind, expected_cls in diff_expected.items():
        out = differentiate(sources[kind], validate=True, target_cls=expected_cls)
        assert isinstance(out, expected_cls)
    int_expected = {
        "linear_velocity": Position,
        "linear_acceleration": LinearVelocity,
        "angular_acceleration": AngularVelocity,
        "acceleration": Velocity,
    }
    for kind, expected_cls in int_expected.items():
        out = integrate(sources[kind], validate=True, target_cls=expected_cls)
        assert isinstance(out, expected_cls)
    assert not hasattr(Velocity, "integrate")
    assert not hasattr(Acceleration, "differentiate")


def test_spatial_hard_173_d6_ao_temporal_kind_validation_fails_closed_on_unsupported_or_ambiguous_kind() -> None:
    """ID: SPATIAL_HARD_173_d6_ao_temporal_kind_validation_fails_closed_on_unsupported_or_ambiguous_kind."""
    src = _typed_sources()["linear_velocity"]
    ao = AnalysisObject._from_validated(src.unsafe_data)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.smooth"):
        _ = smooth(ao, validate=True)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.smooth"):
        _ = smooth(ao, kind="rotation", validate=True)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.smooth"):
        _ = smooth(src, kind="angular_velocity", validate=True)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.integrate"):
        _ = integrate(_typed_sources()["position"], validate=True)
    with pytest.raises(ValueError, match="spatial\\.temporal\\.differentiate"):
        _ = differentiate(src, validate=True, target_cls=AngularAcceleration)


def test_spatial_core_159_d6_ao_default_target_cls_none_returns_analysis_object_with_schema_role_continuity() -> None:
    """ID: SPATIAL_CORE_159_d6_ao_default_target_cls_none_returns_analysis_object_with_schema_role_continuity."""
    sources = _typed_sources()
    op_matrix = (
        (
            differentiate,
            {
                "position": LinearVelocity,
                "linear_velocity": LinearAcceleration,
                "angular_velocity": AngularAcceleration,
                "velocity": Acceleration,
            },
        ),
        (
            integrate,
            {
                "linear_velocity": Position,
                "linear_acceleration": LinearVelocity,
                "angular_acceleration": AngularVelocity,
                "acceleration": Velocity,
            },
        ),
        (smooth, {kind: source.__class__ for kind, source in sources.items()}),
    )
    for op, expected in op_matrix:
        for kind, typed_target in expected.items():
            typed_source = sources[kind]
            typed_out = op(typed_source, validate=True, target_cls=typed_target)
            ao_source = AnalysisObject._from_validated(typed_source.unsafe_data)
            ao_out = op(ao_source, kind=kind, validate=True, target_cls=None)
            assert type(ao_out) is AnalysisObject
            assert read_roles(ao_out.unsafe_data) == read_roles(typed_out.unsafe_data)
            assert read_param_coord_name(ao_out.unsafe_data) == read_param_coord_name(typed_out.unsafe_data)
            assert read_sequence_size_coord_name(ao_out.unsafe_data) == read_sequence_size_coord_name(typed_out.unsafe_data)
            assert get_kinematics_kind(ao_out.unsafe_data, owner="test") == get_kinematics_kind(
                typed_out.unsafe_data, owner="test"
            )
            assert _spatial_rep(ao_out.unsafe_data) == _spatial_rep(typed_out.unsafe_data)
            xr.testing.assert_identical(ao_out.unsafe_data, typed_out.unsafe_data)
