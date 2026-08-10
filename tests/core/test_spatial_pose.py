from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import tal.spatial.pose as pose_module

from tal import AnalysisObject
from tal.core import ComponentRegistryOptions, ComponentSpec, define_components, extract_components, read_components
from tal.core.param_ops.types import ParamEvalOptions
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.core.schema_errors import SchemaError
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation
from tal.spatial.metadata import get_pose_rep, get_position_rep, get_rotation_rep, set_pose_rep
from tal.spatial.temporal.options import PoseTemporalOptions, RotationTemporalOptions
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames, set_frames

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


def _position_dataset(
    *,
    var_name: str = "position",
    samples: tuple[int, int] = (0, 1),
    sequence_dim: str = "sample",
) -> xr.Dataset:
    values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    arr = xr.DataArray(
        values,
        dims=(sequence_dim, "axis"),
        coords={sequence_dim: list(samples), "axis": list(_XYZ)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim=sequence_dim,
        core_dims=("axis",),
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _rotation_dataset(
    *,
    var_name: str = "rotation",
    samples: tuple[int, int] = (0, 1),
    sequence_dim: str = "sample",
) -> xr.Dataset:
    values = np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]], dtype=float)
    arr = xr.DataArray(
        values,
        dims=(sequence_dim, "quat"),
        coords={sequence_dim: list(samples), "quat": list(_QUAT)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name=var_name),
        sequence_dim=sequence_dim,
        core_dims=("quat",),
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _batched_position_dataset() -> xr.Dataset:
    values = np.asarray(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
        ],
        dtype=float,
    )
    arr = xr.DataArray(
        values,
        dims=("sample", "trial", "axis"),
        coords={
            "sample": [0, 1],
            "trial": ["t0", "t1"],
            "axis": list(_XYZ),
            "time_s": ("sample", [0.0, 0.5]),
            "sample_size": ("trial", [2, 2]),
        },
        name="position",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(name="position"),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="time_s",
        sequence_size_coord="sample_size",
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _batched_rotation_dataset(*, include_trial_dim: bool) -> xr.Dataset:
    if include_trial_dim:
        values = np.asarray(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.70710678, 0.70710678]],
            ],
            dtype=float,
        )
        arr = xr.DataArray(
            values,
            dims=("sample", "trial", "quat"),
            coords={
                "sample": [0, 1],
                "trial": ["t0", "t1"],
                "quat": list(_QUAT),
                "time_s": ("sample", [0.0, 0.5]),
                "sample_size": ("trial", [2, 2]),
            },
            name="rotation",
        )
    else:
        values = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.70710678, 0.70710678],
            ],
            dtype=float,
        )
        arr = xr.DataArray(
            values,
            dims=("sample", "quat"),
            coords={
                "sample": [0, 1],
                "quat": list(_QUAT),
                "time_s": ("sample", [0.0, 0.5]),
            },
            name="rotation",
        )
    ds = arr.to_dataset(name="rotation")
    if not include_trial_dim:
        ds = ds.assign_coords(
            trial=("trial", ["t0", "t1"]),
            sample_size=("trial", [2, 2]),
        )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("quat",),
        param_coord="time_s",
        sequence_size_coord="sample_size",
        validate=True,
    )
    return ao.unsafe_data.copy(deep=True)


def _matrix_dataset(*, var_name: str = "pose_matrix") -> xr.Dataset:
    values = np.zeros((2, 4, 4), dtype=float)
    values[0] = np.eye(4, dtype=float)
    values[0, :3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=float)
    theta = np.pi / 2.0
    values[1] = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0, 4.0],
            [np.sin(theta), np.cos(theta), 0.0, 5.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    arr = xr.DataArray(
        values,
        dims=("sample", "row", "col"),
        coords={"sample": [0, 1], "row": list(_QUAT), "col": list(_QUAT)},
        name=var_name,
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name=var_name), sequence_dim="sample", core_dims=("row", "col"), validate=True)
    return set_pose_rep(ao.unsafe_data, rep="matrix", validate=False, owner="test")


def _as_dataarray_with_schema(ds: xr.Dataset, *, var_name: str) -> xr.DataArray:
    da = ds[var_name].copy(deep=True)
    da.attrs["tal"] = ds.attrs["tal"]
    return da


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


def _component_pose() -> Pose:
    pos = Position(_position_dataset())
    rot = Rotation(_rotation_dataset())
    return Pose.from_components(rot, pos, validate=True)


def _component_pose_with_frames(parent: str, child: str) -> Pose:
    pos = frame_retag(Position(_position_dataset()), parent=parent, child=child, validate=True)
    rot = frame_retag(Rotation(_rotation_dataset()), parent=parent, child=child, validate=True)
    return Pose.from_components(rot, pos, validate=True)


def _matrix_payload(pose: Pose) -> np.ndarray:
    matrix_pose = pose.as_matrix(validate=True)
    var_name = str(next(iter(matrix_pose.unsafe_data.data_vars)))
    return matrix_pose.unsafe_data[var_name].values


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


def _pose_temporal_dataset(*, batched: bool = False) -> xr.Dataset:
    if batched:
        position = xr.DataArray(
            np.asarray(
                [
                    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                    [[2.0, 0.0, 0.0], [14.0, 0.0, 0.0]],
                ],
                dtype=float,
            ),
            dims=("sample", "trial", "axis"),
            coords={
                "sample": [0, 1],
                "trial": ["t0", "t1"],
                "axis": list(_XYZ),
                "time_s": ("sample", [0.0, 1.0]),
                "alt_time": ("sample", [10.0, 20.0]),
            },
            name="position",
        )
        rotation = xr.DataArray(
            np.asarray(
                [
                    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.70710678, 0.70710678], [0.0, 0.0, 0.70710678, -0.70710678]],
                ],
                dtype=float,
            ),
            dims=("sample", "trial", "quat"),
            coords={
                "sample": [0, 1],
                "trial": ["t0", "t1"],
                "quat": list(_QUAT),
                "time_s": ("sample", [0.0, 1.0]),
                "alt_time": ("sample", [10.0, 20.0]),
            },
            name="rotation",
        )
        pos_ds = AnalysisObject.from_data(
            position.to_dataset(name="position"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("axis",),
            param_coord="time_s",
            validate=True,
        ).unsafe_data
        rot_ds = AnalysisObject.from_data(
            rotation.to_dataset(name="rotation"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("quat",),
            param_coord="time_s",
            validate=True,
        ).unsafe_data
    else:
        position = xr.DataArray(
            np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            dims=("sample", "axis"),
            coords={
                "sample": [0, 1],
                "axis": list(_XYZ),
                "time_s": ("sample", [0.0, 1.0]),
                "alt_time": ("sample", [10.0, 20.0]),
            },
            name="position",
        )
        rotation = xr.DataArray(
            np.asarray(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.70710678, 0.70710678],
                ],
                dtype=float,
            ),
            dims=("sample", "quat"),
            coords={
                "sample": [0, 1],
                "quat": list(_QUAT),
                "time_s": ("sample", [0.0, 1.0]),
                "alt_time": ("sample", [10.0, 20.0]),
            },
            name="rotation",
        )
        pos_ds = AnalysisObject.from_data(
            position.to_dataset(name="position"),
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
            validate=True,
        ).unsafe_data
        rot_ds = AnalysisObject.from_data(
            rotation.to_dataset(name="rotation"),
            sequence_dim="sample",
            core_dims=("quat",),
            param_coord="time_s",
            validate=True,
        ).unsafe_data
    pos = frame_retag(Position(pos_ds), parent="world", child="body", validate=True)
    rot = frame_retag(Rotation(rot_ds), parent="world", child="body", validate=True)
    return Pose.from_components(rot, pos, validate=True).unsafe_data


def test_spatial_core_018_pose_constructor_accepts_ao_dataset_dataarray_deterministically() -> None:
    """ID: SPATIAL_CORE_018_pose_constructor_accepts_ao_dataset_dataarray_deterministically."""
    component_pose = _component_pose()
    from_components_ao = Pose(AnalysisObject._from_validated(component_pose.unsafe_data))
    from_components_ds = Pose(component_pose.unsafe_data)

    matrix_ds = _matrix_dataset(var_name="matrix")
    from_matrix_ao = Pose(AnalysisObject._from_validated(matrix_ds))
    from_matrix_ds = Pose(matrix_ds)
    from_matrix_da = Pose(_as_dataarray_with_schema(matrix_ds, var_name="matrix"))

    assert isinstance(from_components_ao, Pose)
    assert isinstance(from_components_ds, Pose)
    assert isinstance(from_matrix_ao, Pose)
    assert isinstance(from_matrix_ds, Pose)
    assert isinstance(from_matrix_da, Pose)


def test_spatial_core_019_pose_layout_rep_boundary_matrix_vs_components_deterministic() -> None:
    """ID: SPATIAL_CORE_019_pose_layout_rep_boundary_matrix_vs_components_deterministic."""
    component_pose = _component_pose()
    matrix_pose = Pose(_matrix_dataset())

    assert get_pose_rep(component_pose.unsafe_data, owner="test") == "components"
    assert get_pose_rep(matrix_pose.unsafe_data, owner="test") == "matrix"
    assert set(read_components(component_pose).keys()) == {"position", "rotation"}
    assert read_components(matrix_pose) == {}


def test_spatial_core_020_pose_components_storage_defaults_to_cart_quat() -> None:
    """ID: SPATIAL_CORE_020_pose_components_storage_defaults_to_cart_quat."""
    pose = _component_pose()
    pos, rot = pose.decompose(validate=True)

    assert isinstance(pos, Position)
    assert isinstance(rot, Rotation)
    assert get_position_rep(pos.unsafe_data, owner="test") == "cart"
    assert get_rotation_rep(rot.unsafe_data, owner="test") == "quat"


def test_spatial_core_021_pose_matrix_storage_decompose_default_is_cart_quat() -> None:
    """ID: SPATIAL_CORE_021_pose_matrix_storage_decompose_default_is_cart_quat."""
    pose = Pose(_matrix_dataset())
    pos, rot = pose.decompose(validate=True)

    assert get_position_rep(pos.unsafe_data, owner="test") == "cart"
    assert get_rotation_rep(rot.unsafe_data, owner="test") == "quat"
    assert list(pos.unsafe_data.data_vars) == ["position"]
    assert list(rot.unsafe_data.data_vars) == ["rotation"]
    assert "datavar" not in pos.unsafe_data.data_vars
    assert "datavar" not in rot.unsafe_data.data_vars
    np.testing.assert_allclose(pos.unsafe_data["position"].values, np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float))
    np.testing.assert_allclose(
        rot.unsafe_data["rotation"].values,
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]], dtype=float),
        atol=1e-6,
    )


def test_spatial_core_022_pose_component_boundary_reuses_phase7_5_owners() -> None:
    """ID: SPATIAL_CORE_022_pose_component_boundary_reuses_phase7_5_owners."""
    pose = _component_pose()
    registry = read_components(pose)

    assert registry == {
        "position": ComponentSpec(core_dim="axis", labels=_XYZ, var="position"),
        "rotation": ComponentSpec(core_dim="quat", labels=_QUAT, var="rotation"),
    }
    extracted = extract_components(AnalysisObject._from_validated(pose.unsafe_data), validate=True)
    assert set(extracted.keys()) == {"position", "rotation"}


def test_spatial_core_023_pose_frame_metadata_boundary_reuses_phase7_owners() -> None:
    """ID: SPATIAL_CORE_023_pose_frame_metadata_boundary_reuses_phase7_owners."""
    pos = frame_retag(Position(_position_dataset()), parent="world", child="body", validate=True)
    rot = frame_retag(Rotation(_rotation_dataset()), parent="world", child="body", validate=True)

    pose = Pose.from_components(rot, pos, validate=True)
    assert get_frames(pose.unsafe_data) == ("world", "body")

    pos_out, rot_out = pose.decompose(validate=True)
    assert get_frames(pos_out.unsafe_data) == ("world", "body")
    assert get_frames(rot_out.unsafe_data) == ("world", "body")


def test_spatial_hard_016_pose_constructor_type_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_016_pose_constructor_type_mismatch_fail_closed."""
    with pytest.raises(TypeError, match="spatial.pose.__init__"):
        Pose(object())


def test_spatial_hard_017_pose_constructor_rejects_unknown_pose_layout_rep() -> None:
    """ID: SPATIAL_HARD_017_pose_constructor_rejects_unknown_pose_layout_rep."""
    ds = _matrix_dataset()
    tal = dict(ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    spatial = dict(ext.get("spatial", {}))
    representation = dict(spatial.get("representation", {}))
    representation["rep"] = "bad"
    spatial["representation"] = representation
    ext["spatial"] = spatial
    tal["ext"] = ext
    ds.attrs["tal"] = tal

    with pytest.raises(ValueError, match="unsupported pose representation"):
        Pose(ds)


def test_spatial_hard_018_pose_matrix_constructor_rejects_malformed_se3_shape_or_labels() -> None:
    """ID: SPATIAL_HARD_018_pose_matrix_constructor_rejects_malformed_se3_shape_or_labels."""
    bad_len = _matrix_dataset().isel(row=slice(0, 3))
    with pytest.raises(ValueError, match="must both have length 4"):
        Pose(bad_len)

    bad_labels = _matrix_dataset().assign_coords(row=["x", "y", "z", "q"])
    with pytest.raises(ValueError, match="labels must equal"):
        Pose(bad_labels)


def test_spatial_hard_019_pose_no_hidden_graph_mutation_on_constructor_paths() -> None:
    """ID: SPATIAL_HARD_019_pose_no_hidden_graph_mutation_on_constructor_paths."""
    graph = FrameGraph()
    with graph:
        pos = frame_retag(Position(_position_dataset()), parent="world", child="body", validate=True)
        rot = frame_retag(Rotation(_rotation_dataset()), parent="world", child="body", validate=True)
        before = tuple(sorted(graph._frames.keys()))

        _ = Pose.from_components(rot, pos, validate=True)
        matrix = set_frames(_matrix_dataset(), parent="world", child="body", validate=False)
        _ = Pose.from_matrix(matrix, validate=True)

        after = tuple(sorted(graph._frames.keys()))
        assert before == after


def test_spatial_hard_020_pose_constructor_rejects_non_mapping_spatial_roles_block() -> None:
    """ID: SPATIAL_HARD_020_pose_constructor_rejects_non_mapping_spatial_roles_block."""
    ds = _set_roles(_matrix_dataset(), 123)
    with pytest.raises(ValueError, match="tal.ext.spatial.roles must be a mapping"):
        Pose(ds)


def test_spatial_hard_021_pose_constructor_rejects_non_string_spatial_roles_keys() -> None:
    """ID: SPATIAL_HARD_021_pose_constructor_rejects_non_string_spatial_roles_keys."""
    ds = _set_roles(_matrix_dataset(), {1: "bad"})
    with pytest.raises(ValueError, match="tal.ext.spatial.roles keys must be strings"):
        Pose(ds)


def test_spatial_hard_022_pose_matrix_constructor_rejects_matrix_var_missing_declared_core_dims() -> None:
    """ID: SPATIAL_HARD_022_pose_matrix_constructor_rejects_matrix_var_missing_declared_core_dims."""
    ds = _matrix_dataset()
    values = ds["pose_matrix"].sel(col="w").values
    ds = ds.drop_vars("pose_matrix")
    ds["pose_matrix"] = xr.DataArray(values, dims=("sample", "row"), coords={"sample": [0, 1], "row": list(_QUAT)})

    with pytest.raises(ValueError, match="missing required dims"):
        Pose(ds)


def test_spatial_hard_023_pose_from_components_rejects_coordinate_mismatch_no_outer_align() -> None:
    """ID: SPATIAL_HARD_023_pose_from_components_rejects_coordinate_mismatch_no_outer_align."""
    rot = Rotation(_rotation_dataset(samples=(10, 11)))
    pos = Position(_position_dataset(samples=(0, 1)))

    with pytest.raises(ValueError, match=r"requires exact 'sample' labels under sequence/batch policy"):
        Pose.from_components(rot, pos, validate=True)


def test_spatial_hard_114_pose_component_var_missing_declared_non_core_dims_rejected_by_core_component_runtime_checks() -> None:
    """ID: SPATIAL_HARD_114_pose_component_var_missing_declared_non_core_dims_rejected_by_core_component_runtime_checks."""
    ds = _component_pose().unsafe_data
    pos_values = ds["position"].isel(sample=0).values
    ds = ds.drop_vars("position")
    ds["position"] = xr.DataArray(pos_values, dims=("axis",), coords={"axis": list(_XYZ)})
    with pytest.raises(ValueError, match="spatial.pose.__init__"):
        Pose(ds)
    with pytest.raises(ValueError, match="missing required dims"):
        Pose(ds)


def test_spatial_hard_024_pose_from_matrix_clears_preexisting_component_registry_metadata() -> None:
    """ID: SPATIAL_HARD_024_pose_from_matrix_clears_preexisting_component_registry_metadata."""
    source = AnalysisObject._from_validated(_matrix_dataset())
    with_registry = define_components(
        source,
        opts=ComponentRegistryOptions(
            registry={"legacy_matrix": ComponentSpec(core_dim="row", labels=_QUAT, var="pose_matrix")},
            replace=True,
        ),
        validate=True,
    )

    pose = Pose.from_matrix(with_registry, validate=True)
    assert get_pose_rep(pose.unsafe_data, owner="test") == "matrix"
    assert read_components(pose) == {}
    np.testing.assert_allclose(pose.unsafe_data["pose_matrix"].values, with_registry.unsafe_data["pose_matrix"].values)
    assert pose.unsafe_data["pose_matrix"].dims == with_registry.unsafe_data["pose_matrix"].dims


@pytest.mark.parametrize(
    "rotation",
    [
        pytest.param(np.diag([2.0, 1.0, 1.0]), id="scale"),
        pytest.param(np.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), id="shear"),
        pytest.param(np.diag([-1.0, 1.0, 1.0]), id="reflection"),
    ],
)
def test_spatial_hard_192_pose_matrix_rejects_non_rigid_rotation_blocks(rotation: np.ndarray) -> None:
    """ID: SPATIAL_HARD_192_pose_matrix_rejects_non_rigid_rotation_blocks."""
    ds = _matrix_dataset()
    ds["pose_matrix"].data[0, :3, :3] = rotation
    with pytest.raises(ValueError, match="spatial.pose.from_matrix"):
        Pose.from_matrix(ds, validate=True)
    with pytest.raises(ValueError, match="spatial.pose.__init__"):
        Pose(ds)

    near = _matrix_dataset()
    near["pose_matrix"].data[0, 0, 0] += 1.0e-7
    near["pose_matrix"].data[0, 3, 0] = 5.0e-7
    Pose.from_matrix(near, validate=True)


@pytest.mark.parametrize(
    "case",
    ("nan", "inf", "complex", "bottom_x", "bottom_y", "bottom_z", "bottom_w"),
)
def test_spatial_hard_193_pose_matrix_rejects_nonfinite_values_and_malformed_bottom_rows(case: str) -> None:
    """ID: SPATIAL_HARD_193_pose_matrix_rejects_nonfinite_values_and_malformed_bottom_rows."""
    ds = _matrix_dataset()
    if case == "complex":
        ds["pose_matrix"] = ds["pose_matrix"].astype(np.complex128)
        ds["pose_matrix"].data[0, 0, 3] += 1j
    elif case == "nan":
        ds["pose_matrix"].data[0, 0, 3] = np.nan
    elif case == "inf":
        ds["pose_matrix"].data[0, 1, 1] = np.inf
    else:
        index = {"bottom_x": 0, "bottom_y": 1, "bottom_z": 2, "bottom_w": 3}[case]
        ds["pose_matrix"].data[0, 3, index] = 0.25 if index < 3 else 0.75
    with pytest.raises(ValueError, match="spatial.pose.from_matrix"):
        Pose.from_matrix(ds, validate=True)


def test_spatial_hard_194_pose_matrix_conversion_revalidates_unvalidated_inputs() -> None:
    """ID: SPATIAL_HARD_194_pose_matrix_conversion_revalidates_unvalidated_inputs."""
    ds = _matrix_dataset()
    ds["pose_matrix"].data[0, 3, 0] = 0.5
    pose = Pose.from_matrix(ds, validate=False)

    with pytest.raises(ValueError, match="spatial.pose.decompose"):
        pose.decompose(validate=True)
    with pytest.raises(ValueError, match="spatial.pose.to_rep"):
        pose.as_components(validate=True)
    with pytest.raises(ValueError, match="spatial.pose.to_rep"):
        pose.as_matrix(validate=True)
    with pytest.raises(ValueError, match="Pose._from_validated"):
        pose.rename({"sample": "step"}, validate=True)


def test_spatial_hard_195_pose_matrix_validation_ignores_structural_padding() -> None:
    """ID: SPATIAL_HARD_195_pose_matrix_validation_ignores_structural_padding."""
    values = np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy()
    values[0, 2] = 0.0
    values[1, 1:] = 0.0
    arr = xr.DataArray(
        values,
        dims=("trial", "sample", "row", "col"),
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "row": list(_QUAT),
            "col": list(_QUAT),
            "sample_size": ("trial", [2, 1]),
        },
        name="pose_matrix",
    )
    ao = AnalysisObject.from_data(
        arr.to_dataset(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        sequence_size_coord="sample_size",
        validate=True,
    )
    ds = set_pose_rep(ao.unsafe_data, rep="matrix", validate=False, owner="test")
    pose = Pose.from_matrix(ds, validate=True)
    np.testing.assert_array_equal(pose.unsafe_data["pose_matrix"].values, values)
    converted = pose.as_components(validate=True)
    assert read_sequence_size_coord_name(converted.unsafe_data) == "sample_size"
    np.testing.assert_array_equal(converted.unsafe_data.coords["sample_size"], [2, 1])

    invariant = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    invariant[2] = 0.0
    invariant_arr = xr.DataArray(
        invariant,
        dims=("sample", "row", "col"),
        coords={
            "sample": [0, 1, 2],
            "row": list(_QUAT),
            "col": list(_QUAT),
        },
        name="pose_matrix",
    )
    invariant_ds_raw = invariant_arr.to_dataset().assign_coords(
        trial=("trial", ["a", "b"]),
        sample_size=("trial", [2, 1]),
    )
    invariant_ao = AnalysisObject.from_data(
        invariant_ds_raw,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        sequence_size_coord="sample_size",
        validate=True,
    )
    invariant_ds = set_pose_rep(invariant_ao.unsafe_data, rep="matrix", validate=False, owner="test")
    Pose.from_matrix(invariant_ds, validate=True)
    invariant_ds["pose_matrix"].data[1] = 0.0
    with pytest.raises(ValueError, match="spatial.pose.from_matrix"):
        Pose.from_matrix(invariant_ds, validate=True)

    sequence_invariant = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    sequence_invariant[0] = 0.0
    sequence_invariant_arr = xr.DataArray(
        sequence_invariant,
        dims=("trial", "row", "col"),
        coords={
            "trial": ["a", "b"],
            "row": list(_QUAT),
            "col": list(_QUAT),
        },
        name="pose_matrix",
    )
    sequence_invariant_raw = sequence_invariant_arr.to_dataset().assign_coords(
        sample=("sample", [0, 1, 2]),
        sample_size=("trial", [0, 1]),
    )
    sequence_invariant_ao = AnalysisObject.from_data(
        sequence_invariant_raw,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        sequence_size_coord="sample_size",
        validate=True,
    )
    sequence_invariant_ds = set_pose_rep(
        sequence_invariant_ao.unsafe_data,
        rep="matrix",
        validate=False,
        owner="test",
    )
    Pose.from_matrix(sequence_invariant_ds, validate=True)
    sequence_invariant_ds["pose_matrix"].data[1] = 0.0
    with pytest.raises(ValueError, match="spatial.pose.from_matrix"):
        Pose.from_matrix(sequence_invariant_ds, validate=True)


def test_spatial_perf_001_pose_matrix_validation_preserves_dask_laziness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_PERF_001_pose_matrix_validation_preserves_dask_laziness."""
    da = pytest.importorskip("dask.array")
    ds = _matrix_dataset()
    expected = ds["pose_matrix"].values.copy()
    ds["pose_matrix"].attrs["matrix_kind"] = "rigid"
    ds["pose_matrix"] = xr.DataArray(
        da.from_array(expected, chunks=(1, 2, 2)),
        dims=ds["pose_matrix"].dims,
        coords=ds["pose_matrix"].coords,
        attrs=ds["pose_matrix"].attrs,
        name="pose_matrix",
    )
    with monkeypatch.context() as guarded:
        guarded.setattr(da.Array, "compute", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager")))
        pose = Pose.from_matrix(ds, validate=True)
        assert getattr(pose.unsafe_data["pose_matrix"].data, "chunks", None) is not None
        assert pose.unsafe_data["pose_matrix"].dims == ds["pose_matrix"].dims
        assert pose.unsafe_data["pose_matrix"].attrs["matrix_kind"] == "rigid"
    np.testing.assert_allclose(pose.unsafe_data["pose_matrix"].compute(), expected)

    bad = ds.copy(deep=True)
    bad_values = expected.copy()
    bad_values[0, 0, 0] = 2.0
    bad["pose_matrix"].data = da.from_array(bad_values, chunks=(1, 2, 2))
    with monkeypatch.context() as guarded:
        guarded.setattr(da.Array, "compute", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager")))
        bad_pose = Pose.from_matrix(bad, validate=True)
    with pytest.raises(ValueError, match="spatial.pose.from_matrix"):
        bad_pose.unsafe_data["pose_matrix"].compute()


def test_spatial_core_045_pose_to_rep_components_to_matrix_deterministic() -> None:
    """ID: SPATIAL_CORE_045_pose_to_rep_components_to_matrix_deterministic."""
    pose = _component_pose()
    matrix_pose = pose.to_rep("matrix", validate=True)

    assert isinstance(matrix_pose, Pose)
    assert get_pose_rep(matrix_pose.unsafe_data, owner="test") == "matrix"
    assert read_components(matrix_pose) == {}
    np.testing.assert_allclose(_matrix_payload(matrix_pose), _matrix_payload(pose), atol=1e-8)


def test_spatial_core_046_pose_to_rep_matrix_to_components_deterministic() -> None:
    """ID: SPATIAL_CORE_046_pose_to_rep_matrix_to_components_deterministic."""
    pose = Pose(_matrix_dataset())
    components_pose = pose.to_rep("components", validate=True)

    assert get_pose_rep(components_pose.unsafe_data, owner="test") == "components"
    assert set(read_components(components_pose).keys()) == {"position", "rotation"}
    np.testing.assert_allclose(_matrix_payload(components_pose), _matrix_payload(pose), atol=1e-6)


def test_spatial_core_047_pose_conversion_roundtrip_components_matrix_preserves_frames_roles() -> None:
    """ID: SPATIAL_CORE_047_pose_conversion_roundtrip_components_matrix_preserves_frames_roles."""
    pose = _component_pose_with_frames("world", "body")
    roundtrip = pose.as_matrix(validate=True).as_components(validate=True)

    assert get_frames(roundtrip.unsafe_data) == ("world", "body")
    assert get_pose_rep(roundtrip.unsafe_data, owner="test") == "components"
    np.testing.assert_allclose(_matrix_payload(roundtrip), _matrix_payload(pose), atol=1e-6)


def test_spatial_core_048_pose_compose_tip_tail_chain_deterministic() -> None:
    """ID: SPATIAL_CORE_048_pose_compose_tip_tail_chain_deterministic."""
    left = _component_pose_with_frames("world", "body")
    right = _component_pose_with_frames("body", "sensor")
    composed = left.compose(right, validate=True)

    expected = np.matmul(_matrix_payload(right), _matrix_payload(left))
    np.testing.assert_allclose(_matrix_payload(composed), expected, atol=1e-6)
    assert get_frames(composed.unsafe_data) == ("world", "sensor")


def test_spatial_core_049_pose_compose_mixed_rep_executes_via_canonical_split_and_returns_left_rep() -> None:
    """ID: SPATIAL_CORE_049_pose_compose_mixed_rep_executes_via_canonical_split_and_returns_left_rep."""
    left = _component_pose_with_frames("world", "body").as_matrix(validate=True)
    right = _component_pose_with_frames("body", "sensor")
    composed = left.compose(right, validate=True)

    assert get_pose_rep(composed.unsafe_data, owner="test") == "matrix"
    expected = np.matmul(_matrix_payload(right), _matrix_payload(left))
    np.testing.assert_allclose(_matrix_payload(composed), expected, atol=1e-6)


def test_spatial_core_050_pose_inverse_components_roundtrip_identity_deterministic() -> None:
    """ID: SPATIAL_CORE_050_pose_inverse_components_roundtrip_identity_deterministic."""
    pose = _component_pose_with_frames("world", "body")
    identity = pose.compose(pose.inverse(validate=True), validate=True).as_matrix(validate=True)

    expected = np.broadcast_to(np.eye(4, dtype=float), _matrix_payload(identity).shape)
    np.testing.assert_allclose(_matrix_payload(identity), expected, atol=1e-6)


def test_spatial_core_051_pose_inverse_matrix_roundtrip_identity_deterministic() -> None:
    """ID: SPATIAL_CORE_051_pose_inverse_matrix_roundtrip_identity_deterministic."""
    pose = Pose(_matrix_dataset()).as_matrix(validate=True)
    identity = pose.compose(pose.inverse(validate=True), validate=True).as_matrix(validate=True)

    expected = np.broadcast_to(np.eye(4, dtype=float), _matrix_payload(identity).shape)
    np.testing.assert_allclose(_matrix_payload(identity), expected, atol=1e-6)


def test_spatial_core_052_pose_compose_frame_policy_one_framed_inherits_tags() -> None:
    """ID: SPATIAL_CORE_052_pose_compose_frame_policy_one_framed_inherits_tags."""
    framed = _component_pose_with_frames("world", "body")
    unframed = _component_pose()
    composed = framed.compose(unframed, validate=True)

    assert get_frames(composed.unsafe_data) == ("world", "body")


def test_spatial_hard_044_pose_to_rep_rejects_unsupported_target_rep() -> None:
    """ID: SPATIAL_HARD_044_pose_to_rep_rejects_unsupported_target_rep."""
    pose = _component_pose()
    with pytest.raises(ValueError, match="spatial.pose.to_rep"):
        pose.to_rep("bad", validate=True)  # type: ignore[arg-type]


def test_spatial_hard_045_pose_compose_framed_chain_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_045_pose_compose_framed_chain_mismatch_fail_closed."""
    left = _component_pose_with_frames("world", "a")
    right = _component_pose_with_frames("b", "c")
    with pytest.raises(ValueError, match="left.child == right.parent"):
        left.compose(right, validate=True)


def test_spatial_hard_046_pose_compose_non_core_dim_name_topology_mismatch_fail_closed() -> None:
    """ID: SPATIAL_HARD_046_pose_compose_non_core_dim_name_topology_mismatch_fail_closed."""
    left = Pose.from_components(
        Rotation(_rotation_dataset(sequence_dim="sample")),
        Position(_position_dataset(sequence_dim="sample")),
        validate=True,
    )
    right = Pose.from_components(
        Rotation(_rotation_dataset(sequence_dim="time")),
        Position(_position_dataset(sequence_dim="time")),
        validate=True,
    )
    with pytest.raises(ValueError, match="spatial.pose.compose: pose compose requires matching sequence_dim"):
        left.compose(right, validate=True)


def test_spatial_hard_047_pose_compose_type_boundary_no_raw_runtime_exceptions() -> None:
    """ID: SPATIAL_HARD_047_pose_compose_type_boundary_no_raw_runtime_exceptions."""
    pose = _component_pose()
    with pytest.raises(TypeError, match="spatial.pose.compose"):
        pose.compose(object(), validate=True)


def test_spatial_hard_048_pose_inverse_fail_closed_on_malformed_internal_state() -> None:
    """ID: SPATIAL_HARD_048_pose_inverse_fail_closed_on_malformed_internal_state."""
    pose = _component_pose()
    rotation_spec = read_components(pose)["rotation"]
    assert rotation_spec.var is not None
    pose.unsafe_data[rotation_spec.var].data[0, :] = 0.0
    with pytest.raises(ValueError, match="spatial.pose.inverse"):
        pose.inverse(validate=True)


def test_spatial_hard_049_pose_compose_malformed_frame_schema_fail_closed() -> None:
    """ID: SPATIAL_HARD_049_pose_compose_malformed_frame_schema_fail_closed."""
    left = _component_pose()
    right_ds = left.unsafe_data.copy(deep=True)
    tal = dict(right_ds.attrs["tal"])
    ext = dict(tal.get("ext", {}))
    ext["frames"] = {"parent": "world", "child": "body", "extra": "bad"}
    tal["ext"] = ext
    right_ds.attrs["tal"] = tal
    with pytest.raises(SchemaError, match="tal.ext.frames.extra"):
        left.compose(right_ds, validate=True)


def test_spatial_hard_050_pose_to_rep_matrix_clears_stale_component_registry_truthfully() -> None:
    """ID: SPATIAL_HARD_050_pose_to_rep_matrix_clears_stale_component_registry_truthfully."""
    pose = _component_pose()
    matrix_pose = pose.to_rep("matrix", validate=True)
    assert get_pose_rep(matrix_pose.unsafe_data, owner="test") == "matrix"
    assert read_components(matrix_pose) == {}


def test_bcast_core_049_spatial_component_assembly_uses_frame_shared_precheck_then_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: BCAST_CORE_049_spatial_component_assembly_uses_frame_shared_precheck_then_alignment."""
    rotation = frame_retag(
        Rotation(_rotation_dataset()),
        parent="world",
        child="body",
        validate=True,
    )
    position = frame_retag(
        Position(_position_dataset()),
        parent="map",
        child="tool",
        validate=True,
    )

    def _boom(*_args: object, **_kwargs: object):
        raise AssertionError("topology selection should not run before frame precheck")

    monkeypatch.setattr(pose_module, "select_topology_policy_with_intents", _boom)
    with pytest.raises(ValueError, match="frame tags must match exactly"):
        Pose.from_components(rotation, position, validate=True)


def test_bcast_core_053_pose_component_assembly_preserves_declared_param_and_sequence_size_roles_after_alignment() -> None:
    """ID: BCAST_CORE_053_pose_component_assembly_preserves_declared_param_and_sequence_size_roles_after_alignment."""
    rotation = Rotation(_batched_rotation_dataset(include_trial_dim=False))
    position = Position(_batched_position_dataset())
    out = Pose.from_components(rotation, position, validate=True)
    assert read_param_coord_name(out.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(out.unsafe_data) == "sample_size"
    rotation_spec = read_components(out)["rotation"]
    assert rotation_spec.var is not None
    assert out.unsafe_data[rotation_spec.var].sizes["trial"] == 2


def test_bcast_core_054_pose_compose_static_dynamic_mix_adopts_dynamic_series_roles() -> None:
    """ID: BCAST_CORE_903_pose_compose_static_dynamic_mix_adopts_dynamic_series_roles."""
    left = Pose.from_components(
        Rotation(_batched_rotation_dataset(include_trial_dim=True)),
        Position(_batched_position_dataset()),
        validate=True,
    )
    right = Pose.from_components(
        Rotation(_rotation_dataset()),
        Position(_position_dataset()),
        validate=True,
    )
    out = left.compose(right, validate=True)
    declared, sequence_dim, batch_dims, _ = read_roles(out.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    out_pos, out_rot = out.decompose(validate=True)
    assert out_pos.unsafe_data["position"].sizes["trial"] == 2
    assert out_rot.unsafe_data["rotation"].sizes["trial"] == 2


def test_spatial_core_118_pose_inverse_preserves_declared_batch_only_roles() -> None:
    """ID: SPATIAL_CORE_900_pose_inverse_preserves_declared_batch_only_roles."""
    pos_arr = xr.DataArray(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        dims=("trial", "axis"),
        coords={"trial": ["t0", "t1"], "axis": list(_XYZ)},
        name="position",
    )
    rot_arr = xr.DataArray(
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]], dtype=float),
        dims=("trial", "quat"),
        coords={"trial": ["t0", "t1"], "quat": list(_QUAT)},
        name="rotation",
    )
    pose = Pose.from_components(
        Rotation(
            AnalysisObject.from_data(
                rot_arr.to_dataset(name="rotation"),
                batch_dims=("trial",),
                core_dims=("quat",),
                validate=True,
            )
        ),
        Position(
            AnalysisObject.from_data(
                pos_arr.to_dataset(name="position"),
                batch_dims=("trial",),
                core_dims=("axis",),
                validate=True,
            )
        ),
        validate=True,
    )
    inv = pose.inverse(validate=True)
    declared, sequence_dim, batch_dims, core_dims = read_roles(inv.unsafe_data)
    assert declared is True
    assert sequence_dim is None
    assert batch_dims == ("trial",)
    assert core_dims == ("axis", "quat")
    assert read_param_coord_name(inv.unsafe_data) is None
    assert read_sequence_size_coord_name(inv.unsafe_data) is None

    ident = pose.compose(inv, validate=True).as_matrix(validate=True).unsafe_data["pose_matrix"].values
    expected = np.broadcast_to(np.eye(4, dtype=float), ident.shape)
    np.testing.assert_allclose(ident, expected, atol=1e-6)


def test_spatial_core_119_pose_inverse_succeeds_with_reserved_valid_coord_attr_drift() -> None:
    """ID: SPATIAL_CORE_923_pose_inverse_succeeds_with_reserved_valid_coord_attr_drift."""
    ship_arr = xr.DataArray(
        np.asarray(
            [
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [14.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [18.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        dims=("sample", "trial", "axis"),
        coords={
            "sample": [0, 1, 2],
            "trial": ["t0", "t1"],
            "axis": list(_XYZ),
            "time": ("sample", [0.0, 1.0, 2.0]),
        },
        name="position",
    )
    drone_arr = xr.DataArray(
        np.asarray(
            [
                [[0.0, 1.0, 0.0], [20.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.0], [22.0, 1.0, 0.0]],
                [[2.0, 1.0, 0.0], [24.0, 1.0, 0.0]],
                [[3.0, 1.0, 0.0], [26.0, 1.0, 0.0]],
                [[4.0, 1.0, 0.0], [28.0, 1.0, 0.0]],
            ],
            dtype=float,
        ),
        dims=("sample", "trial", "axis"),
        coords={
            "sample": [0, 1, 2, 3, 4],
            "trial": ["t0", "t1"],
            "axis": list(_XYZ),
            "time": ("sample", [0.0, 0.5, 1.0, 1.5, 2.0]),
        },
        name="position",
    )
    ship_ao = AnalysisObject.from_data(
        ship_arr.to_dataset(name="position"),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="time",
        validate=True,
    )
    drone_ao = AnalysisObject.from_data(
        drone_arr.to_dataset(name="position"),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="time",
        validate=True,
    )
    synced_ship = ship_ao.param.interp_like(drone_ao, on="time", validate=True)
    ship_pos = frame_retag(Position(synced_ship), parent="world", child="ship", validate=True)
    assert "valid" in ship_pos.unsafe_data.coords
    assert ship_pos.unsafe_data.coords["valid"].attrs

    identity_rot = frame_retag(
        Rotation(
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
        ),
        parent="world",
        child="ship",
        validate=True,
    )

    pose = Pose.from_components(identity_rot, ship_pos, validate=True)
    inv = pose.inverse(validate=True)
    declared, sequence_dim, batch_dims, core_dims = read_roles(inv.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ("axis", "quat")
    assert read_param_coord_name(inv.unsafe_data) == "time"
    assert read_sequence_size_coord_name(inv.unsafe_data) is None

    ident = pose.compose(inv, validate=True).as_matrix(validate=True).unsafe_data["pose_matrix"].values
    expected = np.broadcast_to(np.eye(4, dtype=float), ident.shape)
    np.testing.assert_allclose(ident, expected, atol=1e-6)


def test_spatial_hard_118_pose_from_components_conflicting_declared_sequence_dims_fail_closed() -> None:
    """ID: SPATIAL_HARD_920_pose_from_components_conflicting_declared_sequence_dims_fail_closed."""
    with pytest.raises(ValueError, match="spatial.pose.from_components: pose component assembly requires matching sequence_dim"):
        Pose.from_components(
            Rotation(_rotation_dataset(sequence_dim="sample")),
            Position(_position_dataset(sequence_dim="time")),
            validate=True,
        )


def test_spatial_hard_119_pose_from_components_conflicting_declared_batch_dims_fail_closed() -> None:
    """ID: SPATIAL_HARD_921_pose_from_components_conflicting_declared_batch_dims_fail_closed."""
    rot_arr = xr.DataArray(
        np.asarray(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]],
            ],
            dtype=float,
        ),
        dims=("sample", "run", "quat"),
        coords={"sample": [0, 1], "run": ["r0", "r1"], "quat": list(_QUAT)},
        name="rotation",
    )
    pos_arr = xr.DataArray(
        np.asarray(
            [
                [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
            ],
            dtype=float,
        ),
        dims=("sample", "trial", "axis"),
        coords={"sample": [0, 1], "trial": ["t0", "t1"], "axis": list(_XYZ)},
        name="position",
    )
    rotation = Rotation(
        AnalysisObject.from_data(
            rot_arr.to_dataset(name="rotation"),
            sequence_dim="sample",
            batch_dims=("run",),
            core_dims=("quat",),
            validate=True,
        )
    )
    position = Position(
        AnalysisObject.from_data(
            pos_arr.to_dataset(name="position"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("axis",),
            validate=True,
        )
    )
    with pytest.raises(ValueError, match="spatial.pose.from_components: pose component assembly requires matching batch_dims"):
        Pose.from_components(rotation, position, validate=True)


def test_spatial_core_117_rotation_pose_alignment_uses_shared_param_runtime_and_no_batch_bleed() -> None:
    """ID: SPATIAL_CORE_117_rotation_pose_alignment_uses_shared_param_runtime_and_no_batch_bleed."""
    pose = Pose(_pose_temporal_dataset(batched=True))
    out = pose.param.at([0.5], validate=True)
    pos, rot = out.decompose(validate=True)
    pos_values = pos.unsafe_data["position"].transpose("sample", "trial", "axis").values
    np.testing.assert_allclose(pos_values[0, 0], np.asarray([1.0, 0.0, 0.0], dtype=float), atol=1e-6)
    np.testing.assert_allclose(pos_values[0, 1], np.asarray([12.0, 0.0, 0.0], dtype=float), atol=1e-6)
    quat = rot.as_quat(validate=True).unsafe_data["rotation"].transpose("sample", "trial", "quat").values
    assert quat.shape[1] == 2
    assert not np.allclose(quat[0, 0, :], quat[0, 1, :], atol=1e-6, rtol=0.0)


def test_spatial_core_137_pose_param_default_uses_split_typed_preferred_interpolator() -> None:
    """ID: SPATIAL_CORE_137_pose_param_default_uses_split_typed_preferred_interpolator."""
    pose = Pose(_pose_temporal_dataset())
    explicit = PoseTemporalOptions(
        position_opts=ParamEvalOptions(method="linear"),
        rotation_opts=RotationTemporalOptions(method="slerp"),
    )
    out_default_at = pose.param.at([0.5], validate=True)
    out_default_resample = pose.param.resample_to([0.5], validate=True)
    out_explicit_at = pose.param.at([0.5], opts=explicit, validate=True)
    pos_default, rot_default = out_default_at.decompose(validate=True)
    pos_explicit, rot_explicit = out_explicit_at.decompose(validate=True)
    np.testing.assert_allclose(
        pos_default.unsafe_data["position"].values,
        pos_explicit.unsafe_data["position"].values,
        atol=1e-6,
    )
    _assert_quat_equivalent(
        rot_default.as_quat(validate=True).unsafe_data["rotation"].values,
        rot_explicit.as_quat(validate=True).unsafe_data["rotation"].values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _matrix_payload(out_default_resample),
        _matrix_payload(out_default_at),
        atol=1e-6,
    )
    assert get_pose_rep(out_default_at.unsafe_data, owner="test") == get_pose_rep(pose.unsafe_data, owner="test")
    assert get_frames(out_default_at.unsafe_data) == get_frames(pose.unsafe_data)


def test_spatial_core_139_pose_temporal_components_preserves_auxiliary_payload_vars() -> None:
    """ID: SPATIAL_CORE_139_pose_temporal_components_preserves_auxiliary_payload_vars."""
    ds = _pose_temporal_dataset().copy(deep=True)
    ds["temp"] = xr.DataArray(
        np.asarray([10.0, 20.0], dtype=float),
        dims=("sample",),
        coords={"sample": ds.coords["sample"]},
    )
    pose = Pose(ds)
    out_at = pose.param.at([0.5], validate=True)
    out_resample = pose.param.resample_to([0.25, 0.75], validate=True)

    assert {"position", "rotation", "temp"}.issubset(set(out_at.unsafe_data.data_vars))
    assert {"position", "rotation", "temp"}.issubset(set(out_resample.unsafe_data.data_vars))
    np.testing.assert_allclose(out_at.unsafe_data["temp"].values, np.asarray([15.0]), atol=1e-6)
    np.testing.assert_allclose(out_resample.unsafe_data["temp"].values, np.asarray([12.5, 17.5]), atol=1e-6)


def test_spatial_core_140_pose_temporal_matrix_preserves_source_matrix_var_name() -> None:
    """ID: SPATIAL_CORE_140_pose_temporal_matrix_preserves_source_matrix_var_name."""
    ds = _matrix_dataset(var_name="tf_world_body").assign_coords(time_s=("sample", [0.0, 1.0]))
    pose = Pose(ds).set_param_coord(name="time_s", validate=False)
    out_at = pose.param.at([0.5], validate=True)
    out_resample = pose.param.resample_to([0.25, 0.75], validate=True)

    assert list(out_at.unsafe_data.data_vars) == ["tf_world_body"]
    assert list(out_resample.unsafe_data.data_vars) == ["tf_world_body"]
    assert "pose_matrix" not in out_at.unsafe_data.data_vars
    assert "pose_matrix" not in out_resample.unsafe_data.data_vars


def test_spatial_core_141_pose_temporal_matrix_preserves_auxiliary_numeric_payload_vars() -> None:
    """ID: SPATIAL_CORE_141_pose_temporal_matrix_preserves_auxiliary_numeric_payload_vars."""
    ds = _matrix_dataset(var_name="tf_world_body").assign_coords(time_s=("sample", [0.0, 1.0]))
    pose = Pose(ds).set_param_coord(name="time_s", validate=False)
    pose.unsafe_data["temp"] = xr.DataArray(
        np.asarray([10.0, 20.0], dtype=float),
        dims=("sample",),
        coords={"sample": pose.unsafe_data.coords["sample"]},
    )
    out_at = pose.param.at([0.5], validate=False)
    out_resample = pose.param.resample_to([0.25, 0.75], validate=False)

    assert {"tf_world_body", "temp"}.issubset(set(out_at.unsafe_data.data_vars))
    assert {"tf_world_body", "temp"}.issubset(set(out_resample.unsafe_data.data_vars))
    np.testing.assert_allclose(out_at.unsafe_data["temp"].values, np.asarray([15.0]), atol=1e-6)
    np.testing.assert_allclose(out_resample.unsafe_data["temp"].values, np.asarray([12.5, 17.5]), atol=1e-6)
    assert "pose_matrix" not in out_at.unsafe_data.data_vars
    assert "pose_matrix" not in out_resample.unsafe_data.data_vars


def test_spatial_hard_160_pose_temporal_matrix_payload_resolution_fails_closed_on_ambiguous_candidates() -> None:
    """ID: SPATIAL_HARD_160_pose_temporal_matrix_payload_resolution_fails_closed_on_ambiguous_candidates."""
    ds = _matrix_dataset(var_name="tf_world_body").assign_coords(time_s=("sample", [0.0, 1.0]))
    pose = Pose(ds).set_param_coord(name="time_s", validate=False)
    pose.unsafe_data["tf_alt"] = pose.unsafe_data["tf_world_body"].copy(deep=True)

    with pytest.raises(ValueError) as exc_info:
        pose.param.at([0.5], validate=False)
    message = str(exc_info.value)
    assert "spatial.pose.param.at:" in message
    assert "ambiguous" in message


def test_spatial_hard_161_pose_temporal_matrix_aux_rebind_is_validate_false_only_and_preserve_path() -> None:
    """ID: SPATIAL_HARD_161_pose_temporal_matrix_aux_rebind_is_validate_false_only_and_preserve_path."""
    ds = _matrix_dataset(var_name="tf_world_body").assign_coords(time_s=("sample", [0.0, 1.0]))
    pose = Pose(ds).set_param_coord(name="time_s", validate=False)
    pose.unsafe_data["temp"] = xr.DataArray(
        np.asarray([10.0, 20.0], dtype=float),
        dims=("sample",),
        coords={"sample": pose.unsafe_data.coords["sample"]},
    )
    out_at = pose.param.at([0.5], validate=False)
    out_resample = pose.param.resample_to([0.25, 0.75], validate=False)
    assert {"tf_world_body", "temp"}.issubset(set(out_at.unsafe_data.data_vars))
    assert {"tf_world_body", "temp"}.issubset(set(out_resample.unsafe_data.data_vars))
    with pytest.raises(ValueError) as at_error:
        pose.param.at([0.5], validate=True)
    with pytest.raises(ValueError) as resample_error:
        pose.param.resample_to([0.25, 0.75], validate=True)
    assert "spatial.pose.param.at:" in str(at_error.value)
    assert "spatial.pose.param.resample_to:" in str(resample_error.value)


def test_spatial_hard_136_rotation_pose_interp_fail_closed_on_unsupported_method() -> None:
    """ID: SPATIAL_HARD_136_rotation_pose_interp_fail_closed_on_unsupported_method."""
    pose = Pose(_pose_temporal_dataset())
    with pytest.raises(ValueError, match="spatial.pose.param.at: opts.method must be one of"):
        pose.param.at([0.5], opts=ParamEvalOptions(method="cubic"), validate=True)


def test_spatial_hard_137_rotation_pose_interp_owner_prefixed_lazy_error_boundary() -> None:
    """ID: SPATIAL_HARD_137_rotation_pose_interp_owner_prefixed_lazy_error_boundary."""
    pose = Pose(_pose_temporal_dataset())
    opts = PoseTemporalOptions(on="time_s")
    with pytest.raises(ValueError, match="spatial.pose.param.at: conflicting on= value"):
        pose.param.at([0.5], on="alt_time", opts=opts, validate=True)


def test_spatial_hard_052_pose_compose_kernel_failure_wrapped_with_operation_owner_context() -> None:
    """ID: SPATIAL_HARD_052_pose_compose_kernel_failure_wrapped_with_operation_owner_context."""
    left = _component_pose()
    right = _component_pose()
    rotation_spec = read_components(right)["rotation"]
    assert rotation_spec.var is not None
    right.unsafe_data[rotation_spec.var].data[:, :] = 0.0

    with pytest.raises(ValueError) as exc_info:
        left.compose(right, validate=True)

    message = str(exc_info.value)
    assert "spatial.pose.compose" in message
    assert "spatial.pose.kernel" not in message


def test_spatial_hard_053_pose_to_rep_matrix_kernel_failure_wrapped_with_operation_owner_context() -> None:
    """ID: SPATIAL_HARD_053_pose_to_rep_matrix_kernel_failure_wrapped_with_operation_owner_context."""
    pose = _component_pose()
    rotation_spec = read_components(pose)["rotation"]
    assert rotation_spec.var is not None
    pose.unsafe_data[rotation_spec.var].data[:, :] = 0.0

    with pytest.raises(ValueError) as exc_info:
        pose.to_rep("matrix", validate=True)

    message = str(exc_info.value)
    assert "spatial.pose.to_rep" in message
    assert "spatial.pose.kernel" not in message


def test_spatial_hard_054_pose_decompose_matrix_kernel_failure_wrapped_with_decompose_owner_context() -> None:
    """ID: SPATIAL_HARD_054_pose_decompose_matrix_kernel_failure_wrapped_with_decompose_owner_context."""
    matrix_ds = _matrix_dataset()
    matrix_ds["pose_matrix"].data[:, :3, :3] = 0.0
    pose = Pose.from_matrix(matrix_ds, validate=False)

    with pytest.raises(ValueError) as exc_info:
        pose.decompose(validate=True)

    message = str(exc_info.value)
    assert "spatial.pose.decompose" in message
    assert "spatial.pose.kernel" not in message


def test_spatial_hard_055_pose_decompose_matrix_dask_lazy_kernel_failure_wrapped_with_decompose_owner_context() -> None:
    """ID: SPATIAL_HARD_055_pose_decompose_matrix_dask_lazy_kernel_failure_wrapped_with_decompose_owner_context."""
    da = pytest.importorskip("dask.array")
    matrix_ds = _matrix_dataset()
    matrix_vals = matrix_ds["pose_matrix"].values.copy()
    matrix_vals[:, :3, :3] = 0.0
    matrix_ds["pose_matrix"] = xr.DataArray(
        da.from_array(matrix_vals, chunks=(1, 4, 4)),
        dims=matrix_ds["pose_matrix"].dims,
        coords=matrix_ds["pose_matrix"].coords,
    )
    pose = Pose.from_matrix(matrix_ds, validate=False)

    _, rot = pose.decompose(validate=True)
    rot_var = str(next(iter(rot.unsafe_data.data_vars)))
    with pytest.raises(ValueError) as exc_info:
        rot.unsafe_data[rot_var].compute()

    message = str(exc_info.value)
    assert "spatial.pose.decompose" in message
    assert "spatial.pose.kernel" not in message


def test_spatial_hard_116_pose_decompose_matrix_non_trailing_core_dims_preserve_semantic_core_binding() -> None:
    """ID: SPATIAL_HARD_116_pose_decompose_matrix_non_trailing_core_dims_preserve_semantic_core_binding."""
    matrix_ds = _matrix_dataset()
    matrix_ds["pose_matrix"] = matrix_ds["pose_matrix"].transpose("row", "sample", "col")
    pose = Pose(matrix_ds)

    position, rotation = pose.decompose(validate=True)
    _, pos_seq, _, pos_core = read_roles(position.unsafe_data)
    _, rot_seq, _, rot_core = read_roles(rotation.unsafe_data)

    assert pos_seq == "sample"
    assert pos_core == ("row",)
    assert rot_seq == "sample"
    assert rot_core == ("quat",)


def test_topo_core_006_pose_compose_uses_core_topology_touchpoint() -> None:
    """ID: TOPO_CORE_006_pose_compose_uses_core_topology_touchpoint."""
    left = _component_pose()
    right = Pose(_component_pose().unsafe_data.transpose("axis", "quat", "sample"))
    out = left.compose(right, validate=True)
    expected = left.compose(_component_pose(), validate=True)
    np.testing.assert_allclose(
        out.decompose(validate=True)[0].unsafe_data["position"].values,
        expected.decompose(validate=True)[0].unsafe_data["position"].values,
        atol=1e-6,
    )
