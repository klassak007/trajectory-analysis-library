from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal import AnalysisObject
from tal.core.param_engine import map_apply as map_apply_module
from tal.core.param_engine import map_build as map_build_module
from tal.core.schema_read import read_param_coord_name, read_sequence_size_coord_name
from tal.spatial import (
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Position,
)
from tal.spatial.metadata import (
    get_kinematics_kind,
    get_linear_velocity_rep,
    get_position_intent,
)
from tal.spatial.temporal.options import KinematicsDerivativeOptions, KinematicsSmoothingOptions
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames

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


def _linear_velocity(values: np.ndarray, param: np.ndarray) -> LinearVelocity:
    return LinearVelocity(
        _temporal_vector3_dataset(var_name="linear_velocity", core_dim="linear_axis", values=values, param=param)
    )


def test_spatial_core_144_kinematics_smoothing_preserves_frame_kind_rep_truthfulness() -> None:
    """ID: SPATIAL_CORE_144_kinematics_smoothing_preserves_frame_kind_rep_truthfulness."""
    linear = frame_retag(
        _linear_velocity(
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype="float64"),
            np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64"),
        ),
        parent="world",
        child="body",
        validate=True,
    )
    out = linear.smooth(validate=True)
    assert isinstance(out, LinearVelocity)
    assert get_frames(out.unsafe_data) == ("world", "body")
    assert get_kinematics_kind(out.unsafe_data, owner="test") == "linear_velocity"
    assert get_linear_velocity_rep(out.unsafe_data, owner="test") == "cart"

    pos = Position(
        _temporal_vector3_dataset(
            var_name="position",
            core_dim="axis",
            values=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype="float64"),
            param=np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64"),
        )
    ).as_delta(validate=True)
    smoothed_pos = pos.smooth(validate=True)
    assert get_position_intent(smoothed_pos.unsafe_data, owner="test") == "delta"


def test_spatial_core_145_kinematics_smoothing_uses_specified_param_coord_as_primary_domain_key() -> None:
    """ID: SPATIAL_CORE_145_kinematics_smoothing_uses_specified_param_coord_as_primary_domain_key."""
    ds = _temporal_vector3_dataset(
        var_name="linear_velocity",
        core_dim="linear_axis",
        values=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [9.0, 0.0, 0.0], [16.0, 0.0, 0.0]]),
        param=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64"),
    )
    ds = ds.assign_coords(alt_time=("sample", [0.0, 0.1, 0.2, 2.0, 10.0]))
    linear = LinearVelocity(ds)
    opts = KinematicsSmoothingOptions(method="local_poly", window=5, poly_order=1)
    out_default = linear.smooth(on="time_s", opts=opts, validate=True)
    out_alt = linear.smooth(on="alt_time", opts=opts, validate=True)
    assert not np.allclose(
        out_default.unsafe_data["linear_velocity"].values, out_alt.unsafe_data["linear_velocity"].values
    )


def test_spatial_core_146_kinematics_local_derivative_optin_present_without_default_expansion() -> None:
    """ID: SPATIAL_CORE_146_kinematics_local_derivative_optin_present_without_default_expansion."""
    linear = _linear_velocity(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [9.0, 0.0, 0.0], [16.0, 0.0, 0.0]]),
        np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64"),
    )
    default = linear.differentiate(validate=True)
    explicit = linear.differentiate(opts=KinematicsDerivativeOptions(method="finite_difference"), validate=True)
    local_poly = linear.differentiate(
        opts=KinematicsDerivativeOptions(method="local_poly", edge_mode="partial_renorm", order=1, window=5),
        validate=True,
    )
    np.testing.assert_allclose(default.unsafe_data["linear_velocity"].values, explicit.unsafe_data["linear_velocity"].values)
    assert isinstance(local_poly, LinearAcceleration)


def test_spatial_core_147_kinematics_local_stencil_methods_support_nonuniform_param_spacing() -> None:
    """ID: SPATIAL_CORE_147_kinematics_local_stencil_methods_support_nonuniform_param_spacing."""
    linear = _linear_velocity(
        np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [1.2, 0.0, 0.0], [3.5, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        np.asarray([0.0, 0.4, 1.1, 2.7, 4.0], dtype="float64"),
    )
    smoothed = linear.smooth(opts=KinematicsSmoothingOptions(method="local_poly", window=5, poly_order=2), validate=True)
    derived = linear.differentiate(
        opts=KinematicsDerivativeOptions(method="local_poly", edge_mode="partial_renorm", order=1, window=5),
        validate=True,
    )
    assert np.isfinite(smoothed.unsafe_data["linear_velocity"].values[:4]).all()
    assert np.isfinite(derived.unsafe_data["linear_velocity"].values[:4]).all()


def test_spatial_core_152_kinematics_smoothing_gaussian_uses_param_distance_weighting() -> None:
    """ID: SPATIAL_CORE_152_kinematics_smoothing_gaussian_uses_param_distance_weighting."""
    ds = _temporal_vector3_dataset(
        var_name="linear_velocity",
        core_dim="linear_axis",
        values=np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        param=np.asarray([0.0, 0.1, 10.0, 10.1, 10.2], dtype="float64"),
    )
    linear = LinearVelocity(ds)
    gaussian = linear.smooth(
        opts=KinematicsSmoothingOptions(method="gaussian", window=5, sigma=1.0),
        validate=True,
    )
    moving = linear.smooth(
        opts=KinematicsSmoothingOptions(method="moving_average", window=5),
        validate=True,
    )
    center_gaussian = float(gaussian.unsafe_data["linear_velocity"].isel(sample=2, linear_axis=0))
    center_moving = float(moving.unsafe_data["linear_velocity"].isel(sample=2, linear_axis=0))
    assert center_gaussian < 1.0
    assert center_moving > 15.0


def test_spatial_core_153_kinematics_smoothing_gaussian_respects_selected_on_domain() -> None:
    """ID: SPATIAL_CORE_153_kinematics_smoothing_gaussian_respects_selected_on_domain."""
    ds = _temporal_vector3_dataset(
        var_name="linear_velocity",
        core_dim="linear_axis",
        values=np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        param=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64"),
    )
    ds = ds.assign_coords(alt_time=("sample", [0.0, 0.1, 10.0, 10.1, 10.2]))
    linear = LinearVelocity(ds)
    opts = KinematicsSmoothingOptions(method="gaussian", window=5, sigma=1.0)
    out_default = linear.smooth(on="time_s", opts=opts, validate=True)
    out_alt = linear.smooth(on="alt_time", opts=opts, validate=True)
    center_default = float(out_default.unsafe_data["linear_velocity"].isel(sample=2, linear_axis=0))
    center_alt = float(out_alt.unsafe_data["linear_velocity"].isel(sample=2, linear_axis=0))
    assert center_default > 15.0
    assert center_alt < 1.0


def test_spatial_core_148_kinematics_smoothing_validity_and_sequence_size_metadata_truthful() -> None:
    """ID: SPATIAL_CORE_148_kinematics_smoothing_validity_and_sequence_size_metadata_truthful."""
    values = np.asarray(
        [
            [[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]],
            [[1.0, 2.0, 3.0], [40.0, 50.0, 60.0]],
            [[2.0, 3.0, 4.0], [70.0, 80.0, 90.0]],
            [[3.0, 4.0, 5.0], [100.0, 110.0, 120.0]],
        ],
        dtype="float64",
    )
    ds = xr.Dataset(
        data_vars={"linear_velocity": (("sample", "trial", "linear_axis"), values)},
        coords={
            "sample": [0, 1, 2, 3],
            "trial": ["t0", "t1"],
            "linear_axis": list(_XYZ),
            "time_s": ("sample", [0.0, 1.0, 2.0, 3.0]),
            "sample_size": ("trial", [4, 2]),
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
    out = LinearVelocity(ao).smooth(validate=True)
    assert read_param_coord_name(out.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(out.unsafe_data) == "sample_size"
    assert np.isnan(out.unsafe_data["linear_velocity"].sel(trial="t1").isel(sample=3)).all()


def test_spatial_core_149_kinematics_smoothing_dask_lazy_boundary_preserved() -> None:
    """ID: SPATIAL_CORE_149_kinematics_smoothing_dask_lazy_boundary_preserved."""
    dask_array = pytest.importorskip("dask.array")
    values = dask_array.from_array(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype="float64"),
        chunks=(4, 3),
    )
    ds = xr.Dataset(
        data_vars={"linear_velocity": (("sample", "linear_axis"), values)},
        coords={"sample": [0, 1, 2, 3], "linear_axis": list(_XYZ), "time_s": ("sample", [0.0, 1.0, 2.0, 3.0])},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("linear_axis",), param_coord="time_s", validate=True)
    out = LinearVelocity(ao).smooth(validate=True)
    assert getattr(out.unsafe_data["linear_velocity"].data, "chunks", None) is not None


def test_spatial_core_150_kinematics_smoothing_batch_isolation_no_cross_batch_bleed() -> None:
    """ID: SPATIAL_CORE_150_kinematics_smoothing_batch_isolation_no_cross_batch_bleed."""
    values = np.asarray(
        [
            [[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]],
            [[1.0, 2.0, 3.0], [11.0, 21.0, 31.0]],
            [[4.0, 5.0, 6.0], [14.0, 24.0, 34.0]],
        ],
        dtype="float64",
    )
    ds = xr.Dataset(
        data_vars={"linear_velocity": (("sample", "trial", "linear_axis"), values)},
        coords={"sample": [0, 1, 2], "trial": ["t0", "t1"], "linear_axis": list(_XYZ), "time_s": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("linear_axis",), param_coord="time_s", validate=True
    )
    out = LinearVelocity(ao).smooth(validate=True)
    row0 = out.unsafe_data["linear_velocity"].sel(trial="t0").isel(sample=1).values
    row1 = out.unsafe_data["linear_velocity"].sel(trial="t1").isel(sample=1).values
    assert not np.allclose(row0, row1)


def test_spatial_hard_165_kinematics_smoothing_requires_numeric_monotonic_param_domain() -> None:
    """ID: SPATIAL_HARD_165_kinematics_smoothing_requires_numeric_monotonic_param_domain."""
    linear = _linear_velocity(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64"),
        np.asarray([0.0, 2.0, 1.0], dtype="float64"),
    )
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.smooth"):
        _ = linear.smooth()


def test_spatial_hard_168_kinematics_smoothing_param_key_paths_do_not_fallback_to_sequence_labels() -> None:
    """ID: SPATIAL_HARD_168_kinematics_smoothing_param_key_paths_do_not_fallback_to_sequence_labels."""
    ds = _temporal_vector3_dataset(
        var_name="linear_velocity",
        core_dim="linear_axis",
        values=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="float64"),
        param=np.asarray([0.0, 1.0, 2.0], dtype="float64"),
    )
    ds = ds.assign_coords(alt_time=("sample", [0.0, 2.0, 1.0]))
    linear = LinearVelocity(ds)
    _ = linear.smooth(on="time_s", validate=True)
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.smooth"):
        _ = linear.smooth(on="alt_time", validate=True)


def test_spatial_hard_167_kinematics_smoothing_paths_do_not_invoke_hidden_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_HARD_167_kinematics_smoothing_paths_do_not_invoke_hidden_interpolation."""

    def _boom(*_args: object, **_kwargs: object):
        raise AssertionError("interpolation map owners must not be called by smoothing paths")

    monkeypatch.setattr(map_build_module, "build_param_map", _boom)
    monkeypatch.setattr(map_apply_module, "apply_param_map", _boom)
    linear = _linear_velocity(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype="float64"),
        np.asarray([0.0, 1.0, 2.0], dtype="float64"),
    )
    _ = linear.smooth(validate=True)


def test_spatial_hard_169_kinematics_smoothing_owner_prefixed_lazy_error_boundary_preserved() -> None:
    """ID: SPATIAL_HARD_169_kinematics_smoothing_owner_prefixed_lazy_error_boundary_preserved."""
    linear = _linear_velocity(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype="float64"),
        np.asarray([0.0, 1.0], dtype="float64"),
    )
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.smooth"):
        _ = linear.smooth(opts=KinematicsSmoothingOptions(window=4))


def test_spatial_hard_172_kinematics_smoothing_gaussian_sigma_must_be_positive_finite() -> None:
    """ID: SPATIAL_HARD_172_kinematics_smoothing_gaussian_sigma_must_be_positive_finite."""
    linear = _linear_velocity(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype="float64"),
        np.asarray([0.0, 1.0, 2.0], dtype="float64"),
    )
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.smooth"):
        _ = linear.smooth(opts=KinematicsSmoothingOptions(method="gaussian", sigma=0.0))
    with pytest.raises(ValueError, match="spatial\\.linear_velocity\\.smooth"):
        _ = linear.smooth(opts=KinematicsSmoothingOptions(method="gaussian", sigma=float("inf")))


def test_spatial_hard_170_kinematics_smoothing_no_translational_transport_coupling_introduced_in_d4() -> None:
    """ID: SPATIAL_HARD_170_kinematics_smoothing_no_translational_transport_coupling_introduced_in_d4."""
    linear = _linear_velocity(
        np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype="float64"),
        np.asarray([0.0, 1.0, 2.0], dtype="float64"),
    )
    out = linear.smooth(validate=True)
    assert set(out.unsafe_data.data_vars) == {"linear_velocity"}
    assert out.unsafe_data["linear_velocity"].dims == linear.unsafe_data["linear_velocity"].dims
