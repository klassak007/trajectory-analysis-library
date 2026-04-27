from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal import AnalysisObject
from tal.spatial import LinearVelocity, Pose, Position, Rotation
from tal.spatial.metadata import (
    get_expressed_in,
    get_instantaneous_inertial,
    set_expressed_in,
    set_instantaneous_inertial,
)
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames


_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


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
        dims=("sample", "axis"),
        coords={"sample": list(range(values.shape[0])), "axis": list(_XYZ)},
        name="linear_velocity",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="linear_velocity"), sequence_dim="sample", core_dims=("axis",), validate=True)
    return LinearVelocity(ao)


def test_spatial_core_c4_001_relation_semantics_fields_roundtrip_for_typed_families() -> None:
    """ID: SPATIAL_CORE_C4_001_relation_semantics_fields_roundtrip_for_typed_families."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    ds = set_expressed_in(
        velocity.unsafe_data,
        expressed_in="map",
        validate=False,
        owner="test",
    )
    ds = set_instantaneous_inertial(
        ds,
        instantaneous_inertial={"parent"},
        validate=False,
        owner="test",
    )
    out = LinearVelocity(ds)
    assert get_expressed_in(out.unsafe_data, owner="test") == "map"
    assert get_instantaneous_inertial(out.unsafe_data, owner="test") == frozenset({"parent"})


def test_spatial_core_c4_002_missing_expressed_in_defaults_to_canonical_parent_representation() -> None:
    """ID: SPATIAL_CORE_C4_002_missing_expressed_in_defaults_to_canonical_parent_representation."""
    position = frame_retag(
        _position(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="sensor",
        validate=True,
    )
    assert get_expressed_in(position.unsafe_data, owner="test") == "world"


def test_spatial_core_c4_003_kinematic_instantaneous_inertial_roles_roundtrip() -> None:
    """ID: SPATIAL_CORE_C4_003_kinematic_instantaneous_inertial_roles_roundtrip."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    ds = set_instantaneous_inertial(
        velocity.unsafe_data,
        instantaneous_inertial={"parent", "child"},
        validate=False,
        owner="test",
    )
    out = LinearVelocity(ds)
    assert get_instantaneous_inertial(out.unsafe_data, owner="test") == frozenset({"parent", "child"})


def test_spatial_core_c4_004_missing_instantaneous_inertial_defaults_to_empty_frozenset() -> None:
    """ID: SPATIAL_CORE_C4_004_missing_instantaneous_inertial_defaults_to_empty_frozenset."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    assert get_instantaneous_inertial(velocity.unsafe_data, owner="test") == frozenset()


def test_spatial_core_c4_005_configuration_expressed_in_supports_third_frame_not_limited_to_parent_child() -> None:
    """ID: SPATIAL_CORE_C4_005_configuration_expressed_in_supports_third_frame_not_limited_to_parent_child."""
    pose = Pose.from_components(
        frame_retag(_rotation(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)), parent="robot", child="camera", validate=True),
        frame_retag(_position(np.asarray([[0.0, 0.0, 0.0]], dtype=float)), parent="robot", child="camera", validate=True),
        validate=True,
    )
    ds = set_expressed_in(
        pose.unsafe_data,
        expressed_in="world",
        validate=False,
        owner="test",
    )
    out = Pose(ds)
    assert get_expressed_in(out.unsafe_data, owner="test") == "world"


def test_spatial_core_c4_006_kinematic_parent_eq_child_nontrivial_values_are_allowed_by_contract() -> None:
    """ID: SPATIAL_CORE_C4_006_kinematic_parent_eq_child_nontrivial_values_are_allowed_by_contract."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[1.0, -2.0, 0.5]], dtype=float)),
        parent="body",
        child="body",
        validate=True,
    )
    out = LinearVelocity(velocity.unsafe_data)
    np.testing.assert_allclose(out.unsafe_data["linear_velocity"].values, velocity.unsafe_data["linear_velocity"].values)


def test_spatial_core_c4_007_expressed_in_is_basis_only_and_does_not_change_relation_identity() -> None:
    """ID: SPATIAL_CORE_C4_007_expressed_in_is_basis_only_and_does_not_change_relation_identity."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="sensor",
        validate=True,
    )
    ds = set_expressed_in(
        velocity.unsafe_data,
        expressed_in="map",
        validate=False,
        owner="test",
    )
    out = LinearVelocity(ds)
    assert get_frames(out.unsafe_data) == ("world", "sensor")
    assert get_expressed_in(out.unsafe_data, owner="test") == "map"


def test_spatial_hard_c4_001_configuration_types_reject_instantaneous_inertial_metadata() -> None:
    """ID: SPATIAL_HARD_C4_001_configuration_types_reject_instantaneous_inertial_metadata."""
    position = frame_retag(
        _position(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="sensor",
        validate=True,
    )
    ds = set_instantaneous_inertial(
        position.unsafe_data,
        instantaneous_inertial={"parent"},
        validate=False,
        owner="test",
    )
    with pytest.raises(ValueError, match="kinematics-only"):
        Position(ds)


def test_spatial_hard_c4_004_expressed_in_is_rejected_as_instantaneous_inertial_role() -> None:
    """ID: SPATIAL_HARD_C4_004_expressed_in_is_rejected_as_instantaneous_inertial_role."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    with pytest.raises(ValueError, match="not a valid instantaneous_inertial role"):
        set_instantaneous_inertial(
            velocity.unsafe_data,
            instantaneous_inertial={"expressed_in"},
            validate=False,
            owner="test",
        )


def test_spatial_hard_c4_005_invalid_instantaneous_inertial_role_fails_closed() -> None:
    """ID: SPATIAL_HARD_C4_005_invalid_instantaneous_inertial_role_fails_closed."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    with pytest.raises(ValueError, match="must be in"):
        set_instantaneous_inertial(
            velocity.unsafe_data,
            instantaneous_inertial={"observer"},
            validate=False,
            owner="test",
        )
