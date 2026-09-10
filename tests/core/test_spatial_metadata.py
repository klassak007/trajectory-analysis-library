from __future__ import annotations

import inspect

import numpy as np
import pytest
import xarray as xr

import tal.spatial.metadata as spatial_metadata
import tal.spatial.metadata.representation as representation_metadata
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


def _metadata_only_dataset() -> xr.Dataset:
    return xr.Dataset()


def _dataset_with_rep_block(block: dict[str, object]) -> xr.Dataset:
    return xr.Dataset(attrs={"tal": {"ext": {"spatial": {"representation": block}}}})


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


@pytest.mark.parametrize(
    ("getter_name", "expected"),
    [
        ("get_position_rep", "cart"),
        ("get_rotation_rep", "quat"),
        ("get_linear_velocity_rep", "cart"),
        ("get_angular_velocity_rep", "cart"),
        ("get_velocity_rep", "components"),
        ("get_linear_acceleration_rep", "cart"),
        ("get_angular_acceleration_rep", "cart"),
        ("get_acceleration_rep", "components"),
    ],
)
def test_spatial_hard_187_rep_getter_factory_default_parity(getter_name: str, expected: str) -> None:
    """ID: SPATIAL_HARD_187_rep_getter_factory_default_parity."""
    getter = getattr(representation_metadata, getter_name)
    ds = _metadata_only_dataset()
    assert getter(ds, owner="test") == expected
    assert ds.attrs == {}


@pytest.mark.parametrize(
    "ds",
    [
        _metadata_only_dataset(),
        _dataset_with_rep_block({}),
    ],
)
def test_spatial_hard_188_rep_getter_factory_required_pose_parity(ds: xr.Dataset) -> None:
    """ID: SPATIAL_HARD_188_rep_getter_factory_required_pose_parity."""
    with pytest.raises(
        ValueError,
        match=r"test: tal\.ext\.spatial\.representation\.rep must be explicitly set for pose",
    ):
        representation_metadata.get_pose_rep(ds, owner="test")


@pytest.mark.parametrize(
    ("setter_name", "getter_name", "rep", "expected"),
    [
        ("set_position_rep", "get_position_rep", " cart ", "cart"),
        ("set_rotation_rep", "get_rotation_rep", " matrix ", "matrix"),
        ("set_pose_rep", "get_pose_rep", " components ", "components"),
        ("set_linear_velocity_rep", "get_linear_velocity_rep", " cart ", "cart"),
        ("set_angular_velocity_rep", "get_angular_velocity_rep", " cart ", "cart"),
        ("set_velocity_rep", "get_velocity_rep", " vector6 ", "vector6"),
        ("set_linear_acceleration_rep", "get_linear_acceleration_rep", " cart ", "cart"),
        ("set_angular_acceleration_rep", "get_angular_acceleration_rep", " cart ", "cart"),
        ("set_acceleration_rep", "get_acceleration_rep", " vector6 ", "vector6"),
    ],
)
def test_spatial_hard_189_rep_setter_factory_roundtrip_parity(
    setter_name: str,
    getter_name: str,
    rep: str,
    expected: str,
) -> None:
    """ID: SPATIAL_HARD_189_rep_setter_factory_roundtrip_parity."""
    setter = getattr(representation_metadata, setter_name)
    getter = getattr(representation_metadata, getter_name)
    tagged = setter(_metadata_only_dataset(), rep=rep, validate=False, owner="test")
    assert getter(tagged, owner="test") == expected


@pytest.mark.parametrize(
    ("setter_name", "bad_rep", "expected"),
    [
        ("set_rotation_rep", "axis_angle", r"test: unsupported rotation representation 'axis_angle'"),
        ("set_velocity_rep", "cart", r"test: unsupported velocity representation 'cart'"),
        ("set_acceleration_rep", "cart", r"test: unsupported acceleration representation 'cart'"),
    ],
)
def test_spatial_hard_190_rep_setter_factory_validation_error_parity(
    setter_name: str,
    bad_rep: str,
    expected: str,
) -> None:
    """ID: SPATIAL_HARD_190_rep_setter_factory_validation_error_parity."""
    setter = getattr(representation_metadata, setter_name)
    with pytest.raises(ValueError, match=expected):
        setter(_metadata_only_dataset(), rep=bad_rep, validate=False, owner="test")


def test_spatial_hard_191_rep_factory_exports_and_introspection_are_stable() -> None:
    """ID: SPATIAL_HARD_191_rep_factory_exports_and_introspection_are_stable."""
    expected_exports = [
        "get_acceleration_rep",
        "get_angular_acceleration_rep",
        "get_angular_velocity_rep",
        "get_linear_acceleration_rep",
        "get_linear_velocity_rep",
        "get_pose_rep",
        "get_position_rep",
        "get_rotation_rep",
        "get_velocity_rep",
        "set_acceleration_rep",
        "set_angular_acceleration_rep",
        "set_angular_velocity_rep",
        "set_linear_acceleration_rep",
        "set_linear_velocity_rep",
        "set_pose_rep",
        "set_position_rep",
        "set_rotation_rep",
        "set_velocity_rep",
    ]
    assert representation_metadata.__all__ == expected_exports
    for name in expected_exports:
        func = getattr(representation_metadata, name)
        assert getattr(spatial_metadata, name) is func
        assert func.__name__ == name
        assert func.__qualname__ == name
        assert func.__module__ == "tal.spatial.metadata.representation"
        assert "representation" in (func.__doc__ or "")

    getter_params = list(inspect.signature(representation_metadata.get_position_rep).parameters.values())
    setter_params = list(inspect.signature(representation_metadata.set_position_rep).parameters.values())
    assert [(param.name, param.kind) for param in getter_params] == [
        ("ds", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("owner", inspect.Parameter.KEYWORD_ONLY),
    ]
    assert [(param.name, param.kind) for param in setter_params] == [
        ("ds", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("rep", inspect.Parameter.KEYWORD_ONLY),
        ("validate", inspect.Parameter.KEYWORD_ONLY),
        ("owner", inspect.Parameter.KEYWORD_ONLY),
    ]


def test_spatial_core_c4_001_relation_semantics_fields_roundtrip_for_typed_families() -> None:
    """ID: SPATIAL_CORE_C4_001_relation_semantics_fields_roundtrip_for_typed_families."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    ds = set_expressed_in(
        velocity.as_dataset(copy="none"),
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
    assert get_expressed_in(out.as_dataset(copy="none"), owner="test") == "map"
    assert get_instantaneous_inertial(out.as_dataset(copy="none"), owner="test") == frozenset({"parent"})


def test_spatial_core_c4_002_missing_expressed_in_defaults_to_canonical_parent_representation() -> None:
    """ID: SPATIAL_CORE_C4_002_missing_expressed_in_defaults_to_canonical_parent_representation."""
    position = frame_retag(
        _position(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="sensor",
        validate=True,
    )
    assert get_expressed_in(position.as_dataset(copy="none"), owner="test") == "world"


def test_spatial_core_c4_003_kinematic_instantaneous_inertial_roles_roundtrip() -> None:
    """ID: SPATIAL_CORE_C4_003_kinematic_instantaneous_inertial_roles_roundtrip."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    ds = set_instantaneous_inertial(
        velocity.as_dataset(copy="none"),
        instantaneous_inertial={"parent", "child"},
        validate=False,
        owner="test",
    )
    out = LinearVelocity(ds)
    assert get_instantaneous_inertial(out.as_dataset(copy="none"), owner="test") == frozenset({"parent", "child"})
    relation = out.as_dataset(copy="none").attrs["tal"]["ext"]["spatial"]["relation"]
    assert "expressed_in" not in relation


def test_spatial_core_c4_004_missing_instantaneous_inertial_defaults_to_empty_frozenset() -> None:
    """ID: SPATIAL_CORE_C4_004_missing_instantaneous_inertial_defaults_to_empty_frozenset."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[0.1, 0.2, 0.3]], dtype=float)),
        parent="world",
        child="body",
        validate=True,
    )
    assert get_instantaneous_inertial(velocity.as_dataset(copy="none"), owner="test") == frozenset()


def test_spatial_core_c4_005_configuration_expressed_in_supports_third_frame_not_limited_to_parent_child() -> None:
    """ID: SPATIAL_CORE_C4_005_configuration_expressed_in_supports_third_frame_not_limited_to_parent_child."""
    pose = Pose.from_components(
        frame_retag(_rotation(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float)), parent="robot", child="camera", validate=True),
        frame_retag(_position(np.asarray([[0.0, 0.0, 0.0]], dtype=float)), parent="robot", child="camera", validate=True),
        validate=True,
    )
    ds = set_expressed_in(
        pose.as_dataset(copy="none"),
        expressed_in="world",
        validate=False,
        owner="test",
    )
    out = Pose(ds)
    assert get_expressed_in(out.as_dataset(copy="none"), owner="test") == "world"


def test_spatial_core_c4_006_kinematic_parent_eq_child_nontrivial_values_are_allowed_by_contract() -> None:
    """ID: SPATIAL_CORE_C4_006_kinematic_parent_eq_child_nontrivial_values_are_allowed_by_contract."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[1.0, -2.0, 0.5]], dtype=float)),
        parent="body",
        child="body",
        validate=True,
    )
    out = LinearVelocity(velocity.as_dataset(copy="none"))
    np.testing.assert_allclose(out.as_dataset(copy="none")["linear_velocity"].values, velocity.as_dataset(copy="none")["linear_velocity"].values)


def test_spatial_core_c4_007_expressed_in_is_basis_only_and_does_not_change_relation_identity() -> None:
    """ID: SPATIAL_CORE_C4_007_expressed_in_is_basis_only_and_does_not_change_relation_identity."""
    velocity = frame_retag(
        _linear_velocity(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="sensor",
        validate=True,
    )
    ds = set_expressed_in(
        velocity.as_dataset(copy="none"),
        expressed_in="map",
        validate=False,
        owner="test",
    )
    out = LinearVelocity(ds)
    assert get_frames(out.as_dataset(copy="none")) == ("world", "sensor")
    assert get_expressed_in(out.as_dataset(copy="none"), owner="test") == "map"


def test_spatial_hard_c4_001_configuration_types_reject_instantaneous_inertial_metadata() -> None:
    """ID: SPATIAL_HARD_C4_001_configuration_types_reject_instantaneous_inertial_metadata."""
    position = frame_retag(
        _position(np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
        parent="world",
        child="sensor",
        validate=True,
    )
    ds = set_instantaneous_inertial(
        position.as_dataset(copy="none"),
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
            velocity.as_dataset(copy="none"),
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
            velocity.as_dataset(copy="none"),
            instantaneous_inertial={"observer"},
            validate=False,
            owner="test",
        )
