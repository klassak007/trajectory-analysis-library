from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal import AnalysisObject
from tal.core.param_ops.types import ParamEvalOptions
from tal.core.schema_errors import SchemaError
from tal.core.schema_read import read_param_coord_name, read_roles
from tal.linalg import Array
from tal.spatial import Rotation
from tal.spatial.metadata import get_rotation_rep
from tal.spatial.temporal.options import RotationTemporalOptions
from tal.utils.frame_schema import get_frames, set_frames


def _rotation_dataset_quat(
    *,
    var_name: str = "rotation",
    labels: tuple[str, str, str, str] = ("x", "y", "z", "w"),
    values: np.ndarray | None = None,
) -> xr.Dataset:
    if values is None:
        values = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.70710678, 0.70710678],
            ],
            dtype=float,
        )
    arr = xr.DataArray(
        values,
        dims=("sample", "quat"),
        coords={"sample": [0, 1], "quat": list(labels)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=("quat",),
        validate=True,
    )
    return ao.as_dataset(copy="none").copy(deep=True)


def _rotation_dataset_quat_with_sequence_dim(sequence_dim: str) -> xr.Dataset:
    values = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.70710678, 0.70710678],
        ],
        dtype=float,
    )
    arr = xr.DataArray(
        values,
        dims=(sequence_dim, "quat"),
        coords={sequence_dim: [0, 1], "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="rotation"),
        sequence_dim=sequence_dim,
        core_dims=("quat",),
        validate=True,
    )
    return ao.as_dataset(copy="none").copy(deep=True)


def _rotation_dataset_matrix(
    *,
    var_name: str = "rotation",
    labels: tuple[str, str, str] = ("x", "y", "z"),
    values: np.ndarray | None = None,
) -> xr.Dataset:
    if values is None:
        values = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=float,
        )
    arr = xr.DataArray(
        values,
        dims=("sample", "row", "col"),
        coords={"sample": [0, 1], "row": list(labels), "col": list(labels)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim="sample",
        core_dims=("row", "col"),
        validate=True,
    )
    return ao.as_dataset(copy="none").copy(deep=True)


def _rotation_dataset_temporal(
    *,
    rep: str = "quat",
    angles_deg: tuple[float, float] = (0.0, 120.0),
) -> xr.Dataset:
    theta0, theta1 = np.deg2rad(angles_deg[0]), np.deg2rad(angles_deg[1])
    quat_values = np.array(
        [
            [0.0, 0.0, np.sin(theta0 / 2.0), np.cos(theta0 / 2.0)],
            [0.0, 0.0, np.sin(theta1 / 2.0), np.cos(theta1 / 2.0)],
        ],
        dtype=float,
    )
    arr = xr.DataArray(
        quat_values,
        dims=("sample", "quat"),
        coords={
            "sample": [0, 1],
            "quat": ["x", "y", "z", "w"],
            "time_s": ("sample", [0.0, 1.0]),
            "alt_time": ("sample", [10.0, 20.0]),
        },
        name="rotation",
    )
    ds = AnalysisObject.from_data(
        arr.to_dataset(name="rotation"),
        sequence_dim="sample",
        core_dims=("quat",),
        param_coord="time_s",
        validate=True,
    ).as_dataset(copy="none")
    if rep == "matrix":
        ds = Rotation(ds).as_matrix(validate=False).as_dataset(copy="none")
        ds = AnalysisObject._from_validated(ds).set_param_coord(name="time_s", validate=False).as_dataset(copy="none")
        ds = ds.assign_coords(alt_time=("sample", [10.0, 20.0]))
    return ds.copy(deep=True)


def _as_dataarray_with_schema(ds: xr.Dataset, *, var_name: str = "rotation") -> xr.DataArray:
    da = ds[var_name].copy(deep=True)
    da.attrs["tal"] = ds.attrs["tal"]
    return da


def _set_rep(ds: xr.Dataset, rep: object) -> xr.Dataset:
    tal = dict(ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial = dict(ext.get("spatial", {}))
    representation = dict(spatial.get("representation", {}))
    representation["rep"] = rep
    spatial["representation"] = representation
    ext["spatial"] = spatial
    tal["ext"] = ext
    ds.attrs["tal"] = tal
    return ds


def _set_roles(ds: xr.Dataset, roles: object) -> xr.Dataset:
    tal = dict(ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial = dict(ext.get("spatial", {}))
    spatial["roles"] = roles
    ext["spatial"] = spatial
    tal["ext"] = ext
    ds.attrs["tal"] = tal
    return ds


def _assert_quat_equivalent(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-6) -> None:
    lhs = actual.reshape((-1, 4))
    rhs = expected.reshape((-1, 4))
    assert lhs.shape == rhs.shape
    for idx in range(lhs.shape[0]):
        if np.allclose(lhs[idx], rhs[idx], atol=atol, rtol=0.0):
            continue
        if np.allclose(lhs[idx], -rhs[idx], atol=atol, rtol=0.0):
            continue
        raise AssertionError(f"quaternion mismatch at row {idx}: {lhs[idx]!r} vs {rhs[idx]!r}")


def test_spatial_core_014_rotation_constructor_accepts_ao_dataset_dataarray_deterministically() -> None:
    """ID: SPATIAL_CORE_014_rotation_constructor_accepts_ao_dataset_dataarray_deterministically."""
    ds = _rotation_dataset_quat()
    ao = AnalysisObject._from_validated(ds)
    da = _as_dataarray_with_schema(ds)

    from_ao = Rotation(ao)
    from_ds = Rotation(ds)
    from_da = Rotation(da)

    assert isinstance(from_ao, Rotation)
    assert isinstance(from_ds, Rotation)
    assert isinstance(from_da, Rotation)
    assert get_rotation_rep(from_ao.as_dataset(copy="none"), owner="test") == "quat"
    assert get_rotation_rep(from_ds.as_dataset(copy="none"), owner="test") == "quat"
    assert get_rotation_rep(from_da.as_dataset(copy="none"), owner="test") == "quat"


def test_spatial_core_015_rotation_representation_tag_boundary_deterministic() -> None:
    """ID: SPATIAL_CORE_015_rotation_representation_tag_boundary_deterministic."""
    ds_missing_rep = _rotation_dataset_quat()
    out = Rotation(ds_missing_rep)
    assert get_rotation_rep(out.as_dataset(copy="none"), owner="test") == "quat"

    ds_matrix = _set_rep(_rotation_dataset_matrix(), "matrix")
    assert get_rotation_rep(Rotation(ds_matrix).as_dataset(copy="none"), owner="test") == "matrix"

    for rep in ("cart", "rotvec", "euler"):
        ds = _set_rep(_rotation_dataset_quat(), rep)
        with pytest.raises(ValueError, match="unsupported rotation representation"):
            Rotation(ds)


def test_spatial_core_016_rotation_canonical_operation_representation_architecture_locked() -> None:
    """ID: SPATIAL_CORE_016_rotation_canonical_operation_representation_architecture_locked."""
    rot = Rotation(_rotation_dataset_quat())
    assert isinstance(rot, Rotation)
    assert Rotation.CANONICAL_REP == "quat"
    assert Rotation.QUAT_LABELS == ("x", "y", "z", "w")
    assert Rotation.MATRIX_LABELS == ("x", "y", "z")
    assert callable(getattr(Rotation, "compose"))
    assert callable(getattr(Rotation, "inverse"))
    assert not hasattr(Rotation, "rotate")


def test_spatial_core_179_rotation_norm_magnitude_angle_values_and_rep_independence() -> None:
    """ID: SPATIAL_CORE_179_rotation_norm_magnitude_angle_values_and_rep_independence."""
    values = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float64",
    )
    arr = xr.DataArray(
        values,
        dims=("sample", "quat"),
        coords={"sample": [0, 1, 2], "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    quat_ds = AnalysisObject.from_data(
        arr.to_dataset(name="rotation"),
        sequence_dim="sample",
        core_dims=("quat",),
        validate=True,
    ).as_dataset(copy="none")
    quat_rot = Rotation(quat_ds)
    matrix_rot = quat_rot.as_matrix(validate=True)

    out_norm = quat_rot.norm()
    out_mag = quat_rot.magnitude()
    out_matrix_mag = matrix_rot.magnitude()
    expected = np.asarray([0.0, np.pi / 2.0, np.pi], dtype="float64")

    assert isinstance(out_norm, Array)
    assert isinstance(out_mag, Array)
    assert isinstance(out_matrix_mag, Array)
    np.testing.assert_allclose(out_norm.as_dataset(copy="none")["datavar"].values, expected, atol=1e-6)
    np.testing.assert_allclose(out_mag.as_dataset(copy="none")["datavar"].values, expected, atol=1e-6)
    np.testing.assert_allclose(out_matrix_mag.as_dataset(copy="none")["datavar"].values, expected, atol=1e-6)
    xr.testing.assert_identical(out_norm.as_dataset(copy="none"), out_mag.as_dataset(copy="none"))


def test_spatial_core_180_rotation_magnitude_preserves_scalar_sequence_topology() -> None:
    """ID: SPATIAL_CORE_180_rotation_magnitude_preserves_scalar_sequence_topology."""
    rot = Rotation(_rotation_dataset_temporal(rep="matrix"))
    out = rot.magnitude()

    assert isinstance(out, Array)
    assert tuple(out.as_dataset(copy="none").data_vars) == ("datavar",)
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ()
    assert core_dims == ()
    assert read_param_coord_name(out.as_dataset(copy="none")) == "time_s"


def test_spatial_hard_181_rotation_magnitude_fail_closed_on_quaternion_layout_violation() -> None:
    """ID: SPATIAL_HARD_181_rotation_magnitude_fail_closed_on_quaternion_layout_violation."""
    rot = Rotation(_rotation_dataset_quat())
    rot.as_dataset(copy="none")["rotation"] = xr.DataArray(
        np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype="float64"),
        dims=("sample", "quat_component"),
        coords={"sample": [0, 1], "quat_component": ["x", "y", "z"]},
        name="rotation",
    )
    with pytest.raises(ValueError, match="spatial\\.rotation\\.magnitude"):
        _ = rot.magnitude()


def test_spatial_hard_012_rotation_invalid_representation_tag_fail_closed() -> None:
    """ID: SPATIAL_HARD_012_rotation_invalid_representation_tag_fail_closed."""
    ds_non_numeric = _rotation_dataset_quat(
        values=np.array([["a", "b", "c", "d"], ["e", "f", "g", "h"]], dtype=object)
    )
    with pytest.raises(ValueError, match="must be numeric"):
        Rotation(ds_non_numeric)

    ds_multi_var = _rotation_dataset_quat().assign(extra=("sample", [1.0, 2.0]))
    with pytest.raises(ValueError, match="exactly one data variable"):
        Rotation(ds_multi_var)

    ds_bad_labels = _rotation_dataset_quat(labels=("x", "y", "z", "q"))
    with pytest.raises(ValueError, match="core labels must equal"):
        Rotation(ds_bad_labels)

    ds_bad_core_len = _rotation_dataset_quat().isel(quat=slice(0, 3))
    with pytest.raises(ValueError, match="must have length 4"):
        Rotation(ds_bad_core_len)

    ds_bad_frames = _rotation_dataset_quat()
    tal = dict(ds_bad_frames.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    ext["frames"] = {"parent": "world", "child": "body", "extra": "bad"}
    tal["ext"] = ext
    ds_bad_frames.attrs["tal"] = tal
    with pytest.raises(SchemaError, match="tal.ext.frames.extra"):
        Rotation(ds_bad_frames)


def test_spatial_hard_013_rotation_constructor_type_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_013_rotation_constructor_type_mismatch_fail_closed."""
    with pytest.raises(TypeError, match="spatial.rotation.__init__"):
        Rotation(object())


def test_spatial_hard_014_rotation_constructor_rejects_non_mapping_spatial_roles_block() -> None:
    """ID: SPATIAL_HARD_014_rotation_constructor_rejects_non_mapping_spatial_roles_block."""
    ds = _set_roles(_rotation_dataset_quat(), 123)
    with pytest.raises(ValueError, match="tal.ext.spatial.roles must be a mapping"):
        Rotation(ds)


def test_spatial_hard_015_rotation_constructor_rejects_non_string_spatial_roles_keys() -> None:
    """ID: SPATIAL_HARD_015_rotation_constructor_rejects_non_string_spatial_roles_keys."""
    ds = _set_roles(_rotation_dataset_quat(), {1: "bad"})
    with pytest.raises(ValueError, match="tal.ext.spatial.roles keys must be strings"):
        Rotation(ds)


def test_spatial_core_017_rotation_constructor_accepts_unknown_string_spatial_roles_keys() -> None:
    """ID: SPATIAL_CORE_017_rotation_constructor_accepts_unknown_string_spatial_roles_keys."""
    ds = _set_roles(_rotation_dataset_quat(), {"rotation_hint": "quat"})
    rot = Rotation(ds)
    assert isinstance(rot, Rotation)


def test_spatial_core_034_rotation_constructor_accepts_matrix_rep_layout_deterministically() -> None:
    """ID: SPATIAL_CORE_034_rotation_constructor_accepts_matrix_rep_layout_deterministically."""
    ds = _set_rep(_rotation_dataset_matrix(), "matrix")
    ao = AnalysisObject._from_validated(ds)
    da = _as_dataarray_with_schema(ds)

    from_ao = Rotation(ao)
    from_ds = Rotation(ds)
    from_da = Rotation(da)

    assert get_rotation_rep(from_ao.as_dataset(copy="none"), owner="test") == "matrix"
    assert get_rotation_rep(from_ds.as_dataset(copy="none"), owner="test") == "matrix"
    assert get_rotation_rep(from_da.as_dataset(copy="none"), owner="test") == "matrix"


def test_spatial_core_035_rotation_to_rep_quat_to_matrix_deterministic() -> None:
    """ID: SPATIAL_CORE_035_rotation_to_rep_quat_to_matrix_deterministic."""
    rot = Rotation(_rotation_dataset_quat())
    out = rot.as_matrix()

    assert get_rotation_rep(out.as_dataset(copy="none"), owner="test") == "matrix"
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ()
    assert core_dims == ("row", "col")
    assert tuple(out.as_dataset(copy="none").get_index("row").tolist()) == ("x", "y", "z")
    assert tuple(out.as_dataset(copy="none").get_index("col").tolist()) == ("x", "y", "z")

    expected = _rotation_dataset_matrix()["rotation"].values
    assert np.allclose(out.as_dataset(copy="none")["rotation"].values, expected, atol=1e-6, rtol=0.0)


def test_spatial_core_036_rotation_to_rep_matrix_to_quat_deterministic() -> None:
    """ID: SPATIAL_CORE_036_rotation_to_rep_matrix_to_quat_deterministic."""
    rot = Rotation(_set_rep(_rotation_dataset_matrix(), "matrix"))
    out = rot.as_quat()

    assert get_rotation_rep(out.as_dataset(copy="none"), owner="test") == "quat"
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ()
    assert core_dims == ("quat",)
    assert tuple(out.as_dataset(copy="none").get_index("quat").tolist()) == ("x", "y", "z", "w")

    expected = _rotation_dataset_quat()["rotation"].values
    _assert_quat_equivalent(out.as_dataset(copy="none")["rotation"].values, expected, atol=1e-6)


def test_spatial_core_037_rotation_conversion_roundtrip_quat_matrix_within_tolerance() -> None:
    """ID: SPATIAL_CORE_037_rotation_conversion_roundtrip_quat_matrix_within_tolerance."""
    rot = Rotation(_rotation_dataset_quat())
    roundtrip = rot.as_matrix().as_quat()
    _assert_quat_equivalent(roundtrip.as_dataset(copy="none")["rotation"].values, rot.as_dataset(copy="none")["rotation"].values, atol=1e-6)


def test_spatial_core_038_rotation_conversion_preserves_frames_roles_and_sequence_layout() -> None:
    """ID: SPATIAL_CORE_038_rotation_conversion_preserves_frames_roles_and_sequence_layout."""
    arr = xr.DataArray(
        np.array(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.25881905, 0.96592583]],
                [[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 1.0, 0.0]],
            ],
            dtype=float,
        ),
        dims=("trial", "sample", "quat"),
        coords={"trial": ["t0", "t1"], "sample": [0, 1], "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="rotation"), sequence_dim="sample", batch_dims=("trial",), core_dims=("quat",), validate=True)
    ds = set_frames(ao.as_dataset(copy="none"), parent="world", child="body", validate=False)
    ds = _set_roles(ds, {"rotation_hint": "quat"})

    rot = Rotation(ds)
    matrix_rot = rot.as_matrix()
    quat_rot = matrix_rot.as_quat()

    assert get_frames(matrix_rot.as_dataset(copy="none")) == ("world", "body")
    assert get_frames(quat_rot.as_dataset(copy="none")) == ("world", "body")

    _, seq_matrix, batch_matrix, core_matrix = read_roles(matrix_rot.as_dataset(copy="none"))
    _, seq_quat, batch_quat, core_quat = read_roles(quat_rot.as_dataset(copy="none"))
    assert seq_matrix == "sample"
    assert batch_matrix == ("trial",)
    assert core_matrix == ("row", "col")
    assert seq_quat == "sample"
    assert batch_quat == ("trial",)
    assert core_quat == ("quat",)

    roles_matrix = matrix_rot.as_dataset(copy="none").attrs["tal"]["ext"]["spatial"]["roles"]
    roles_quat = quat_rot.as_dataset(copy="none").attrs["tal"]["ext"]["spatial"]["roles"]
    assert roles_matrix["rotation_hint"] == "quat"
    assert roles_quat["rotation_hint"] == "quat"


def test_spatial_hard_033_rotation_constructor_matrix_rep_rejects_invalid_shape_or_labels() -> None:
    """ID: SPATIAL_HARD_033_rotation_constructor_matrix_rep_rejects_invalid_shape_or_labels."""
    ds_bad_len = _set_rep(_rotation_dataset_matrix().isel(row=slice(0, 2)), "matrix")
    with pytest.raises(ValueError, match="must both have length 3"):
        Rotation(ds_bad_len)

    ds_bad_labels = _set_rep(_rotation_dataset_matrix(labels=("x", "y", "q")), "matrix")
    with pytest.raises(ValueError, match="labels must equal"):
        Rotation(ds_bad_labels)


def test_spatial_hard_034_rotation_matrix_to_quat_rejects_non_orthonormal_matrix() -> None:
    """ID: SPATIAL_HARD_034_rotation_matrix_to_quat_rejects_non_orthonormal_matrix."""
    values = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
            [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    rot = Rotation(_set_rep(_rotation_dataset_matrix(values=values), "matrix"))
    with pytest.raises(ValueError, match="spatial.rotation.kernel.matrix_to_quat"):
        rot.as_quat()


def test_spatial_hard_035_rotation_to_rep_rejects_unsupported_target_rep() -> None:
    """ID: SPATIAL_HARD_035_rotation_to_rep_rejects_unsupported_target_rep."""
    rot = Rotation(_rotation_dataset_quat())
    with pytest.raises(ValueError, match="unsupported target rotation representation"):
        rot.to_rep("rotvec")  # type: ignore[arg-type]


def test_spatial_hard_036_rotation_conversion_type_boundary_no_raw_runtime_exceptions() -> None:
    """ID: SPATIAL_HARD_036_rotation_conversion_type_boundary_no_raw_runtime_exceptions."""
    rot = Rotation(_rotation_dataset_quat())
    with pytest.raises(TypeError, match="rep must be a string"):
        rot.to_rep(123)  # type: ignore[arg-type]

    bad_matrix = Rotation(
        _set_rep(
            _rotation_dataset_matrix(
                values=np.array(
                    [
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
                        [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
                    ],
                    dtype=float,
                )
            ),
            "matrix",
        )
    )
    with pytest.raises(ValueError, match="spatial.rotation.kernel.matrix_to_quat"):
        bad_matrix.to_rep("quat")


def test_spatial_core_039_rotation_compose_quat_tip_tail_chain_deterministic() -> None:
    """ID: SPATIAL_CORE_039_rotation_compose_quat_tip_tail_chain_deterministic."""
    s = np.sqrt(0.5)
    left_ds = _rotation_dataset_quat(values=np.array([[0.0, 0.0, s, s], [0.0, 0.0, s, s]], dtype=float))
    right_ds = _rotation_dataset_quat(values=np.array([[0.0, 0.0, s, s], [0.0, 0.0, s, s]], dtype=float))
    left = Rotation(set_frames(left_ds, parent="world", child="a", validate=False))
    right = Rotation(set_frames(right_ds, parent="a", child="b", validate=False))

    out = left.compose(right)
    assert get_rotation_rep(out.as_dataset(copy="none"), owner="test") == "quat"
    assert get_frames(out.as_dataset(copy="none")) == ("world", "b")

    expected = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float)
    _assert_quat_equivalent(out.as_dataset(copy="none")["rotation"].values, expected, atol=1e-6)


def test_spatial_core_040_rotation_compose_mixed_rep_executes_via_quat_and_returns_left_rep() -> None:
    """ID: SPATIAL_CORE_040_rotation_compose_mixed_rep_executes_via_quat_and_returns_left_rep."""
    left_matrix = Rotation(_set_rep(_rotation_dataset_matrix(), "matrix"))
    right_quat = Rotation(_rotation_dataset_quat())
    out_matrix = left_matrix.compose(right_quat)
    assert get_rotation_rep(out_matrix.as_dataset(copy="none"), owner="test") == "matrix"
    expected_matrix = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    assert np.allclose(out_matrix.as_dataset(copy="none")["rotation"].values, expected_matrix, atol=1e-6, rtol=0.0)

    left_quat = Rotation(_rotation_dataset_quat())
    right_matrix = Rotation(_set_rep(_rotation_dataset_matrix(), "matrix"))
    out_quat = left_quat.compose(right_matrix)
    assert get_rotation_rep(out_quat.as_dataset(copy="none"), owner="test") == "quat"
    expected_quat = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], dtype=float)
    _assert_quat_equivalent(out_quat.as_dataset(copy="none")["rotation"].values, expected_quat, atol=1e-6)


def test_spatial_core_041_rotation_inverse_quat_roundtrip_identity_deterministic() -> None:
    """ID: SPATIAL_CORE_041_rotation_inverse_quat_roundtrip_identity_deterministic."""
    rot = Rotation(_rotation_dataset_quat())
    identity = rot.compose(rot.inverse())
    assert get_rotation_rep(identity.as_dataset(copy="none"), owner="test") == "quat"
    expected = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    _assert_quat_equivalent(identity.as_dataset(copy="none")["rotation"].values, expected, atol=1e-6)


def test_spatial_core_042_rotation_inverse_matrix_roundtrip_identity_deterministic() -> None:
    """ID: SPATIAL_CORE_042_rotation_inverse_matrix_roundtrip_identity_deterministic."""
    rot = Rotation(_set_rep(_rotation_dataset_matrix(), "matrix"))
    identity = rot.compose(rot.inverse())
    assert get_rotation_rep(identity.as_dataset(copy="none"), owner="test") == "matrix"
    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    assert np.allclose(identity.as_dataset(copy="none")["rotation"].values, expected, atol=1e-6, rtol=0.0)


def test_spatial_core_043_rotation_compose_frame_policy_one_framed_inherits_tags() -> None:
    """ID: SPATIAL_CORE_043_rotation_compose_frame_policy_one_framed_inherits_tags."""
    base_left = Rotation(_rotation_dataset_quat())
    base_right = Rotation(_rotation_dataset_quat())
    left_framed = Rotation(set_frames(base_left.as_dataset(copy="none"), parent="world", child="body", validate=False))
    right_framed = Rotation(set_frames(base_right.as_dataset(copy="none"), parent="map", child="sensor", validate=False))

    out_left = left_framed.compose(base_right)
    assert get_frames(out_left.as_dataset(copy="none")) == ("world", "body")

    out_right = base_left.compose(right_framed)
    assert get_frames(out_right.as_dataset(copy="none")) == ("map", "sensor")


def test_spatial_core_044_rotation_inverse_swaps_frame_tags_deterministic() -> None:
    """ID: SPATIAL_CORE_044_rotation_inverse_swaps_frame_tags_deterministic."""
    rot = Rotation(set_frames(_rotation_dataset_quat(), parent="world", child="body", validate=False))
    inv = rot.inverse()
    assert get_frames(inv.as_dataset(copy="none")) == ("body", "world")


def test_bcast_core_048_spatial_compose_family_supports_missing_semantic_dim_materialization_after_frame_check() -> None:
    """ID: BCAST_CORE_048_spatial_compose_family_supports_missing_semantic_dim_materialization_after_frame_check."""
    left_arr = xr.DataArray(
        np.asarray(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=float,
        ),
        dims=("trial", "sample", "quat"),
        coords={"trial": ["a", "b"], "sample": [0, 1], "quat": list(_rotation_dataset_quat()["rotation"].coords["quat"].values)},
        name="rotation",
    )
    left = Rotation(
        set_frames(
            AnalysisObject.from_data(
                left_arr.to_dataset(name="rotation"),
                sequence_dim="sample",
                batch_dims=("trial",),
                core_dims=("quat",),
                validate=True,
            ).as_dataset(copy="none"),
            parent="world",
            child="a",
            validate=False,
        )
    )
    right_arr = xr.DataArray(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
        dims=("sample", "quat"),
        coords={"sample": [0, 1], "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    right = Rotation(
        set_frames(
            AnalysisObject.from_data(
                right_arr.to_dataset(name="rotation").assign_coords(trial=("trial", ["a", "b"])),
                sequence_dim="sample",
                batch_dims=("trial",),
                core_dims=("quat",),
                validate=False,
            ).as_dataset(copy="none"),
            parent="a",
            child="b",
            validate=False,
        )
    )
    out = left.compose(right, validate=True)
    assert out.as_dataset(copy="none")["rotation"].sizes["trial"] == 2
    assert get_frames(out.as_dataset(copy="none")) == ("world", "b")


def test_spatial_hard_037_rotation_compose_framed_chain_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_037_rotation_compose_framed_chain_mismatch_fail_closed."""
    left = Rotation(set_frames(_rotation_dataset_quat(), parent="world", child="a", validate=False))
    right = Rotation(set_frames(_rotation_dataset_quat(), parent="b", child="c", validate=False))
    with pytest.raises(ValueError, match="left.child == right.parent"):
        left.compose(right)


def test_bcast_hard_037_tip_tail_compose_mismatch_fails_before_alignment() -> None:
    """ID: BCAST_HARD_037_tip_tail_compose_mismatch_fails_before_alignment."""
    left = Rotation(set_frames(_rotation_dataset_quat(), parent="world", child="a", validate=False)).a(
        on="param",
        sequence_join=None,
    )
    right = Rotation(set_frames(_rotation_dataset_quat(), parent="b", child="c", validate=False)).b()
    with pytest.raises(ValueError, match="left.child == right.parent"):
        left.compose(right, validate=True)


def test_spatial_hard_038_rotation_compose_type_boundary_no_raw_runtime_exceptions() -> None:
    """ID: SPATIAL_HARD_038_rotation_compose_type_boundary_no_raw_runtime_exceptions."""
    left = Rotation(_rotation_dataset_quat())
    with pytest.raises(TypeError, match="spatial.rotation.compose"):
        left.compose(object())  # type: ignore[arg-type]


def test_spatial_hard_039_rotation_inverse_fail_closed_on_malformed_internal_state() -> None:
    """ID: SPATIAL_HARD_039_rotation_inverse_fail_closed_on_malformed_internal_state."""
    rot = Rotation(_rotation_dataset_quat())
    rot.as_dataset(copy="none")["rotation"].data[0, :] = 0.0
    with pytest.raises(ValueError, match="spatial.rotation.kernel.inverse"):
        rot.inverse()


def test_spatial_hard_040_rotation_compose_malformed_frame_schema_fail_closed() -> None:
    """ID: SPATIAL_HARD_040_rotation_compose_malformed_frame_schema_fail_closed."""
    left = Rotation(_rotation_dataset_quat())
    bad = _rotation_dataset_quat()
    tal = dict(bad.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    ext["frames"] = {"parent": "world", "child": "body", "extra": "bad"}
    tal["ext"] = ext
    bad.attrs["tal"] = tal
    with pytest.raises(SchemaError, match="tal.ext.frames.extra"):
        left.compose(bad)


def test_spatial_hard_041_rotation_compose_rejects_non_core_dim_name_topology_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_041_rotation_compose_rejects_non_core_dim_name_topology_mismatch_fail_closed."""
    left = Rotation(_rotation_dataset_quat_with_sequence_dim("sample"))
    right = Rotation(_rotation_dataset_quat_with_sequence_dim("time"))
    with pytest.raises(ValueError, match="spatial.rotation.compose: compose requires matching sequence_dim"):
        left.compose(right)


def test_bcast_core_050_rotation_compose_static_dynamic_adopts_batch_semantics() -> None:
    """ID: BCAST_CORE_900_rotation_compose_static_dynamic_adopts_batch_semantics."""
    left_arr = xr.DataArray(
        np.array(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]],
            ],
            dtype=float,
        ),
        dims=("trial", "sample", "quat"),
        coords={"trial": ["a", "b"], "sample": [0, 1], "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    left = Rotation(
        AnalysisObject.from_data(
            left_arr.to_dataset(name="rotation"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("quat",),
            validate=True,
        )
    )
    right = Rotation(_rotation_dataset_quat())
    out = left.compose(right, validate=True)
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ("quat",)
    assert out.as_dataset(copy="none")["rotation"].sizes["trial"] == 2

    trial_coord = left.as_dataset(copy="none").coords["trial"]
    right_batched_arr = right.as_dataset(copy="none")["rotation"].expand_dims(trial=trial_coord).transpose("trial", "sample", "quat")
    right_batched = Rotation(
        AnalysisObject.from_data(
            right_batched_arr.to_dataset(name="rotation"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("quat",),
            validate=True,
        )
    )
    expected = left.compose(right_batched, validate=True)
    np.testing.assert_allclose(out.as_dataset(copy="none")["rotation"].values, expected.as_dataset(copy="none")["rotation"].values, atol=1e-7)


def test_spatial_hard_042_rotation_compose_rejects_conflicting_non_empty_batch_dims_fail_closed() -> None:
    """ID: SPATIAL_HARD_042_rotation_compose_rejects_conflicting_non_empty_batch_dims_fail_closed."""
    left_arr = xr.DataArray(
        np.array(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]],
            ],
            dtype=float,
        ),
        dims=("trial", "sample", "quat"),
        coords={"trial": ["a", "b"], "sample": [0, 1], "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    right_arr = xr.DataArray(
        np.array(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]],
            ],
            dtype=float,
        ),
        dims=("run", "sample", "quat"),
        coords={"run": ["r0", "r1"], "sample": [0, 1], "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    left = Rotation(
        AnalysisObject.from_data(
            left_arr.to_dataset(name="rotation"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("quat",),
            validate=True,
        )
    )
    right = Rotation(
        AnalysisObject.from_data(
            right_arr.to_dataset(name="rotation"),
            sequence_dim="sample",
            batch_dims=("run",),
            core_dims=("quat",),
            validate=True,
        )
    )
    with pytest.raises(ValueError, match="spatial.rotation.compose: compose requires matching batch_dims"):
        left.compose(right)


def test_spatial_hard_043_rotation_compose_frame_mismatch_short_circuits_before_kernel_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: SPATIAL_HARD_043_rotation_compose_frame_mismatch_short_circuits_before_kernel_conversion."""
    left = Rotation(set_frames(_rotation_dataset_quat(), parent="world", child="a", validate=False))
    right = Rotation(set_frames(_rotation_dataset_quat(), parent="b", child="c", validate=False))

    def _boom(*args: object, **kwargs: object) -> Rotation:
        raise RuntimeError("compose should not reach as_quat when frame chain is invalid")

    monkeypatch.setattr(Rotation, "as_quat", _boom)
    with pytest.raises(ValueError, match="left.child == right.parent"):
        left.compose(right)


def test_topo_core_005_rotation_compose_uses_core_topology_touchpoint() -> None:
    """ID: TOPO_CORE_005_rotation_compose_uses_core_topology_touchpoint."""
    left = Rotation(_rotation_dataset_quat())
    right = Rotation(_rotation_dataset_quat().transpose("quat", "sample"))
    out = left.compose(right, validate=True)
    expected = left.compose(Rotation(_rotation_dataset_quat()), validate=True)
    np.testing.assert_allclose(out.as_dataset(copy="none")["rotation"].values, expected.as_dataset(copy="none")["rotation"].values, atol=1e-6)


def test_spatial_core_111_rotation_interpolation_uses_quaternion_canonical_path() -> None:
    """ID: SPATIAL_CORE_111_rotation_interpolation_uses_quaternion_canonical_path."""
    rot = Rotation(_rotation_dataset_temporal(rep="matrix", angles_deg=(0.0, 120.0)))
    out = rot.param.at([0.5], validate=True)
    assert get_rotation_rep(out.as_dataset(copy="none"), owner="test") == "matrix"
    mat = out.as_dataset(copy="none")["rotation"].values[0]
    eye = np.eye(3, dtype=float)
    np.testing.assert_allclose(mat.T @ mat, eye, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(mat), 1.0, atol=1e-6)


def test_spatial_core_112_rotation_slerp_and_nearest_linear_boundaries_deterministic() -> None:
    """ID: SPATIAL_CORE_112_rotation_slerp_and_nearest_linear_boundaries_deterministic."""
    rot = Rotation(_rotation_dataset_temporal(angles_deg=(10.0, 170.0)))
    nearest = rot.param.at([0.49], opts=RotationTemporalOptions(method="nearest"), validate=True)
    linear = rot.param.at([0.25], opts=RotationTemporalOptions(method="linear"), validate=True)
    slerp = rot.param.at([0.25], opts=RotationTemporalOptions(method="slerp"), validate=True)
    for out in (nearest, linear, slerp):
        quat = out.as_quat(validate=True).as_dataset(copy="none")["rotation"].values[0]
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)
    assert not np.allclose(
        linear.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        slerp.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        atol=1e-5,
        rtol=0.0,
    )


def test_spatial_core_130_rotation_pose_interp_uses_specified_param_coord_as_primary_domain_key() -> None:
    """ID: SPATIAL_CORE_130_rotation_pose_interp_uses_specified_param_coord_as_primary_domain_key."""
    rot = Rotation(_rotation_dataset_temporal(angles_deg=(0.0, 90.0)))
    out_time = rot.param.at([0.5], on="time_s", opts=RotationTemporalOptions(method="slerp"), validate=True)
    out_alt = rot.param.at([15.0], on="alt_time", opts=RotationTemporalOptions(method="slerp"), validate=True)
    np.testing.assert_allclose(
        out_time.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        out_alt.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        atol=1e-6,
    )


def test_spatial_core_136_rotation_param_default_uses_typed_preferred_interpolator() -> None:
    """ID: SPATIAL_CORE_136_rotation_param_default_uses_typed_preferred_interpolator."""
    rot = Rotation(_rotation_dataset_temporal(angles_deg=(10.0, 170.0)))
    out_default = rot.param.at([0.5], validate=True)
    out_slerp = rot.param.at([0.5], opts=RotationTemporalOptions(method="slerp"), validate=True)
    np.testing.assert_allclose(
        out_default.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        out_slerp.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        atol=1e-6,
    )


def test_spatial_core_138_rotation_public_slerp_method_is_explicit_query_delegator() -> None:
    """ID: SPATIAL_CORE_138_rotation_public_slerp_method_is_explicit_query_delegator."""
    rot = Rotation(_rotation_dataset_temporal(angles_deg=(0.0, 120.0)))
    via_method = rot.slerp([0.5], validate=True)
    via_param = rot.param.at([0.5], opts=RotationTemporalOptions(method="slerp"), validate=True)
    np.testing.assert_allclose(
        via_method.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        via_param.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        atol=1e-6,
    )


def test_spatial_hard_157_typed_rotation_pose_defaults_do_not_change_plain_ao_param_defaults() -> None:
    """ID: SPATIAL_HARD_157_typed_rotation_pose_defaults_do_not_change_plain_ao_param_defaults."""
    ds = _rotation_dataset_temporal(angles_deg=(10.0, 170.0))
    ao = AnalysisObject._from_validated(ds)
    rot = Rotation(ds)
    ao_out = ao.param.at([0.5], opts=ParamEvalOptions(method="linear"), validate=True)
    rot_out = rot.param.at([0.5], validate=True)
    assert not np.allclose(
        ao_out.as_dataset(copy="none")["rotation"].values,
        rot_out.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        atol=1e-5,
        rtol=0.0,
    )


def test_spatial_hard_158_rotation_linear_mode_is_euclidean_while_slerp_is_manifold_safe() -> None:
    """ID: SPATIAL_HARD_158_rotation_linear_mode_is_euclidean_while_slerp_is_manifold_safe."""
    quat_values = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    arr = xr.DataArray(
        quat_values,
        dims=("sample", "quat"),
        coords={"sample": [0, 1], "quat": ["x", "y", "z", "w"], "time_s": ("sample", [0.0, 1.0])},
        name="rotation",
    )
    ds = AnalysisObject.from_data(
        arr.to_dataset(name="rotation"),
        sequence_dim="sample",
        core_dims=("quat",),
        param_coord="time_s",
        validate=True,
    ).as_dataset(copy="none")
    rot = Rotation(ds)
    out_linear = rot.param.at([0.25], opts=RotationTemporalOptions(method="linear"), validate=True)
    out_slerp = rot.param.at([0.25], opts=RotationTemporalOptions(method="slerp"), validate=True)
    assert not np.allclose(
        out_linear.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        out_slerp.as_quat(validate=True).as_dataset(copy="none")["rotation"].values,
        atol=1e-5,
        rtol=0.0,
    )


def test_spatial_hard_159_rotation_temporal_rejects_auxiliary_payload_vars_without_silent_drop() -> None:
    """ID: SPATIAL_HARD_159_rotation_temporal_rejects_auxiliary_payload_vars_without_silent_drop."""
    rot = Rotation(_rotation_dataset_temporal(rep="matrix", angles_deg=(0.0, 120.0)))
    base_var = str(next(iter(rot.as_dataset(copy="none").data_vars)))
    rot.as_dataset(copy="none")["aux"] = rot.as_dataset(copy="none")[base_var].isel(row=0, col=0, drop=True).copy(deep=True)

    with pytest.raises(ValueError) as exc_info:
        rot.param.at([0.5], validate=True)
    assert "spatial.rotation.param.at:" in str(exc_info.value)
    assert "auxiliary payload vars are not supported" in str(exc_info.value)
    assert "spatial.rotation.to_rep" not in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        rot.param.resample_to([0.25, 0.75], validate=True)
    assert "spatial.rotation.param.resample_to:" in str(exc_info.value)
    assert "auxiliary payload vars are not supported" in str(exc_info.value)
    assert "spatial.rotation.to_rep" not in str(exc_info.value)
