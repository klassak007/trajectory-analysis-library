from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import tal.spatial.ops.rotation_reduce_ops as rotation_reduce_ops
from tal.core.analysis_object import AnalysisObject
from tal.spatial import Rotation


def _rotation_quat_ao(values: np.ndarray) -> Rotation:
    arr = xr.DataArray(
        values,
        dims=("sample", "quat"),
        coords={"sample": np.arange(values.shape[0], dtype=int), "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    ds = AnalysisObject.from_data(
        arr.to_dataset(name="rotation"),
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("quat",),
        validate=True,
    ).unsafe_data
    return Rotation(ds)


def _rotation_quat_batched_ao(values: np.ndarray) -> Rotation:
    arr = xr.DataArray(
        values,
        dims=("trial", "sample", "quat"),
        coords={
            "trial": np.arange(values.shape[0], dtype=int),
            "sample": np.arange(values.shape[1], dtype=int),
            "quat": ["x", "y", "z", "w"],
        },
        name="rotation",
    )
    ds = AnalysisObject.from_data(
        arr.to_dataset(name="rotation"),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("quat",),
        validate=True,
    ).unsafe_data
    return Rotation(ds)


def _assert_quat_equivalent(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-6) -> None:
    if np.allclose(actual, expected, atol=atol, rtol=0.0):
        return
    if np.allclose(actual, -expected, atol=atol, rtol=0.0):
        return
    raise AssertionError(f"quaternion mismatch: actual={actual!r}, expected={expected!r}")


def test_spatial_core_p9c_001_rotation_mean_delegates_to_domain_local_rotation_reduce_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_CORE_P9C_001_rotation_mean_delegates_to_domain_local_rotation_reduce_owner."""
    rot = _rotation_quat_ao(np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]], dtype=float))
    called = {"value": False}

    def _fake_rotation_mean(*args, **kwargs):
        called["value"] = True
        return args[0]

    monkeypatch.setattr(rotation_reduce_ops, "rotation_mean", _fake_rotation_mean)
    out = rot.mean(dim="sample", validate=True)
    assert called["value"]
    assert isinstance(out, Rotation)


def test_spatial_core_p9c_002_rotation_weighted_mean_and_na_policy_semantics_remain_deterministic() -> None:
    """ID: SPATIAL_CORE_P9C_002_rotation_weighted_mean_and_na_policy_semantics_remain_deterministic."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    q180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    rot = _rotation_quat_ao(np.stack([q0, q90, q180], axis=0))

    out = rot.mean(dim="sample", weights=np.array([1.0, 1.0, 0.0], dtype=float), validate=True)
    actual = out.as_quat(validate=True).unsafe_data["rotation"].to_numpy()
    expected = np.array([0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)], dtype=float)
    _assert_quat_equivalent(actual, expected)

    bad = _rotation_quat_ao(np.stack([q0, np.array([np.nan, np.nan, np.nan, np.nan]), q180], axis=0))
    out_bad = bad.mean(dim="sample", skipna=False, validate=True)
    assert np.isnan(out_bad.unsafe_data["rotation"].to_numpy()).all()


def test_spatial_core_p9c_003_typed_reducers_preserve_typed_outputs_when_invariants_hold() -> None:
    """ID: SPATIAL_CORE_P9C_003_typed_reducers_preserve_typed_outputs_when_invariants_hold."""
    rot = _rotation_quat_ao(np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]], dtype=float))
    out = rot.mean(dim="sample", validate=True)
    assert isinstance(out, Rotation)


def test_spatial_core_p9c_004_rotation_weighted_mean_batched_reduction_is_shape_safe() -> None:
    """ID: SPATIAL_CORE_P9C_004_rotation_weighted_mean_batched_reduction_is_shape_safe."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    q180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    values = np.array([[q0, q90, q0], [q0, q0, q180]], dtype=float)
    rot = _rotation_quat_batched_ao(values)

    sample_weights = np.array([1.0, 2.0, 1.0], dtype=float)
    reduced_sample = rot.mean(dim="sample", weights=sample_weights, validate=True)
    for trial in range(values.shape[0]):
        expected = _rotation_quat_ao(values[trial]).mean(dim="sample", weights=sample_weights, validate=True)
        _assert_quat_equivalent(
            reduced_sample.unsafe_data["rotation"].isel(trial=trial).to_numpy(),
            expected.unsafe_data["rotation"].to_numpy(),
        )

    trial_weights = np.array([1.0, 2.0], dtype=float)
    reduced_trial = rot.mean(dim="trial", weights=trial_weights, validate=True)
    for sample in range(values.shape[1]):
        expected = _rotation_quat_ao(values[:, sample, :]).mean(dim="sample", weights=trial_weights, validate=True)
        _assert_quat_equivalent(
            reduced_trial.unsafe_data["rotation"].isel(sample=sample).to_numpy(),
            expected.unsafe_data["rotation"].to_numpy(),
        )


def test_spatial_hard_p9c_001_rotation_weight_mapping_missing_reduced_dim_fails_closed() -> None:
    """ID: SPATIAL_HARD_P9C_001_rotation_weight_mapping_missing_reduced_dim_fails_closed."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q0]], dtype=float))
    with pytest.raises(ValueError, match="spatial\\.rotation\\.mean: weights mapping must include every reduced dim"):
        _ = rot.mean(dim="sample", weights={"trial": np.array([1.0, 1.0], dtype=float)}, validate=True)


def test_spatial_hard_p9c_002_rotation_ndarray_weights_multi_dim_fail_closed() -> None:
    """ID: SPATIAL_HARD_P9C_002_rotation_ndarray_weights_multi_dim_fail_closed."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q0]], dtype=float))
    with pytest.raises(ValueError, match="spatial\\.rotation\\.mean: ndarray weights are only valid for single-dim reduction"):
        _ = rot.mean(dim=("trial", "sample"), weights=np.array([1.0, 2.0], dtype=float), validate=True)


def test_spatial_core_p9c_005_rotation_multi_dim_dataarray_weights_supported_in_one_pass() -> None:
    """ID: SPATIAL_CORE_P9C_005_rotation_multi_dim_dataarray_weights_supported_in_one_pass."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    q180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    values = np.array([[q0, q90, q0], [q0, q0, q180]], dtype=float)
    rot = _rotation_quat_batched_ao(values)
    weight_da = xr.DataArray(
        np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 3.0]], dtype=float),
        dims=("trial", "sample"),
        coords={"trial": rot.unsafe_data.coords["trial"], "sample": rot.unsafe_data.coords["sample"]},
    )
    out = rot.mean(dim=("trial", "sample"), weights=weight_da, validate=True)

    flat_rot = _rotation_quat_ao(values.reshape(-1, 4))
    flat_weight = weight_da.to_numpy().reshape(-1)
    expected = flat_rot.mean(dim="sample", weights=flat_weight, validate=True)
    _assert_quat_equivalent(out.unsafe_data["rotation"].to_numpy(), expected.unsafe_data["rotation"].to_numpy())


def test_spatial_core_p9c_006_rotation_multi_dim_mapping_and_dataarray_weights_are_equivalent() -> None:
    """ID: SPATIAL_CORE_P9C_006_rotation_multi_dim_mapping_and_dataarray_weights_are_equivalent."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    q180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    values = np.array([[q0, q90, q0], [q0, q0, q180]], dtype=float)
    rot = _rotation_quat_batched_ao(values)

    trial_w = np.array([1.0, 2.0], dtype=float)
    sample_w = np.array([1.0, 3.0, 2.0], dtype=float)
    map_out = rot.mean(
        dim=("trial", "sample"),
        weights={"trial": trial_w, "sample": sample_w},
        validate=True,
    )
    dense_weight = xr.DataArray(
        trial_w[:, None] * sample_w[None, :],
        dims=("trial", "sample"),
        coords={"trial": rot.unsafe_data.coords["trial"], "sample": rot.unsafe_data.coords["sample"]},
    )
    dense_out = rot.mean(dim=("trial", "sample"), weights=dense_weight, validate=True)
    _assert_quat_equivalent(map_out.unsafe_data["rotation"].to_numpy(), dense_out.unsafe_data["rotation"].to_numpy())


def test_spatial_core_p9c_007_rotation_reduce_prunes_reduced_dims_from_output_topology_metadata() -> None:
    """ID: SPATIAL_CORE_P9C_007_rotation_reduce_prunes_reduced_dims_from_output_topology_metadata."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q0]], dtype=float))

    sample_reduced = rot.mean(dim="sample", validate=True)
    core_block = sample_reduced.unsafe_data.attrs.get("tal", {}).get("core", {})
    assert "sample" not in sample_reduced.unsafe_data.dims
    assert sample_reduced.unsafe_data["rotation"].dims == ("trial", "quat")
    assert core_block.get("roles") == {"batch_dims": [], "core_dims": []}
    assert core_block.get("param_coord") is None
    assert core_block.get("validity") is None

    trial_reduced = rot.mean(dim="trial", validate=True)
    roles = trial_reduced.unsafe_data.attrs.get("tal", {}).get("core", {}).get("roles")
    assert "trial" not in trial_reduced.unsafe_data.dims
    assert trial_reduced.unsafe_data["rotation"].dims == ("sample", "quat")
    assert roles == {"sequence_dim": "sample", "batch_dims": [], "core_dims": ["quat"]}


def test_spatial_core_p9c_008_rotation_chained_reduce_over_remaining_non_core_dim_succeeds_after_sequence_clear() -> None:
    """ID: SPATIAL_CORE_P9C_008_rotation_chained_reduce_over_remaining_non_core_dim_succeeds_after_sequence_clear."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    q180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q180]], dtype=float))
    first = rot.mean(dim="sample", validate=True)
    second = first.mean(dim="trial", validate=True)
    assert isinstance(second, Rotation)
    assert second.unsafe_data["rotation"].dims == ("quat",)
    assert np.isfinite(second.unsafe_data["rotation"].to_numpy()).all()


def test_spatial_hard_p9c_006_rotation_component_dim_guard_persists_after_sequence_clear() -> None:
    """ID: SPATIAL_HARD_P9C_006_rotation_component_dim_guard_persists_after_sequence_clear."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q0]], dtype=float))
    first = rot.mean(dim="sample", validate=True)

    with pytest.raises(
        ValueError,
        match="spatial\\.rotation\\.mean: explicit reduction over required component dims is not allowed",
    ):
        _ = first.mean(dim="quat", validate=True)
    with pytest.raises(
        ValueError,
        match="spatial\\.rotation\\.mean: explicit reduction over required component dims is not allowed",
    ):
        _ = first.mean(dim=("trial", "quat"), validate=True)

    default_reduced = first.mean(validate=True)
    assert isinstance(default_reduced, Rotation)
    assert default_reduced.unsafe_data["rotation"].dims == ("quat",)


def test_spatial_core_p9c_009_rotation_matrix_rep_mean_reducer_succeeds() -> None:
    """ID: SPATIAL_CORE_P9C_009_rotation_matrix_rep_mean_reducer_succeeds."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    q180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q180]], dtype=float)).as_matrix(validate=True)
    first = rot.mean(dim="sample", validate=True)
    second = first.mean(dim="trial", validate=True)
    assert isinstance(first, Rotation)
    assert isinstance(second, Rotation)
    assert "sample" not in first.unsafe_data.dims
    assert second.unsafe_data["rotation"].dims == ("quat",)
    assert np.isfinite(second.unsafe_data["rotation"].to_numpy()).all()


def test_spatial_core_p9c_010_rotation_inherited_non_owned_reducers_demote_to_analysisobject() -> None:
    """ID: SPATIAL_CORE_P9C_010_rotation_inherited_non_owned_reducers_demote_to_analysisobject."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q0]], dtype=float))
    any_out = rot.any(dim="sample", validate=True)
    sum_out = rot.sum(dim="sample", validate=True)
    assert type(any_out) is AnalysisObject
    assert type(sum_out) is AnalysisObject
    assert not isinstance(any_out, Rotation)
    assert not isinstance(sum_out, Rotation)
    assert any_out.unsafe_data["rotation"].dtype.kind == "b"


def test_spatial_hard_p9c_007_rotation_matrix_component_dims_fail_closed() -> None:
    """ID: SPATIAL_HARD_P9C_007_rotation_matrix_component_dims_fail_closed."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q0]], dtype=float)).as_matrix(validate=True)
    with pytest.raises(
        ValueError,
        match="spatial\\.rotation\\.mean: explicit reduction over required component dims is not allowed",
    ):
        _ = rot.mean(dim="row", validate=True)
    with pytest.raises(
        ValueError,
        match="spatial\\.rotation\\.mean: explicit reduction over required component dims is not allowed",
    ):
        _ = rot.mean(dim=("sample", "col"), validate=True)


def test_spatial_hard_p9c_003_rotation_repeat_reduce_on_removed_dim_fails_closed() -> None:
    """ID: SPATIAL_HARD_P9C_003_rotation_repeat_reduce_on_removed_dim_fails_closed."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q0]], dtype=float))
    first = rot.mean(dim="sample", validate=True)
    with pytest.raises(ValueError, match="spatial\\.rotation\\.mean: dim entries must reference dataset dims"):
        _ = first.mean(dim="sample", validate=True)


def test_spatial_hard_p9c_005_rotation_roleless_ambiguous_quat_dim_resolution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_HARD_P9C_005_rotation_roleless_ambiguous_quat_dim_resolution_fails_closed."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    rot = _rotation_quat_batched_ao(np.array([[q0, q90], [q0, q0]], dtype=float)).mean(dim="sample", validate=True)
    ambiguous = xr.Dataset(
        data_vars={"rotation": (("trial", "quat", "alt_quat"), np.zeros((2, 4, 4), dtype=float))},
        coords={
            "trial": np.array([0, 1], dtype=int),
            "quat": np.array(["x", "y", "z", "w"], dtype=object),
            "alt_quat": np.array([0, 1, 2, 3], dtype=int),
        },
    )

    monkeypatch.setattr(Rotation, "as_quat", lambda self, validate=False: SimpleNamespace(unsafe_data=ambiguous))
    with pytest.raises(ValueError, match="spatial\\.rotation\\.mean: Rotation.mean without declared sequence roles requires exactly one length-4 quaternion dim"):
        _ = rot.mean(dim="trial", validate=True)


def test_reduce_hard_p9c_002_typed_component_dim_reduction_fails_closed() -> None:
    """ID: REDUCE_HARD_P9C_002_typed_component_dim_reduction_fails_closed."""
    rot = _rotation_quat_ao(np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]], dtype=float))
    with pytest.raises(ValueError, match="component dims"):
        _ = rot.mean(dim="quat")


def test_spatial_hard_p9c_004_rotation_sequence_size_resolver_errors_are_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_HARD_P9C_004_rotation_sequence_size_resolver_errors_are_not_swallowed."""
    rot = _rotation_quat_ao(np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.70710678, 0.70710678]], dtype=float))

    def _boom(_ds):
        raise RuntimeError("resolver boom")

    monkeypatch.setattr(rotation_reduce_ops, "read_sequence_size_coord_name", _boom)
    with pytest.raises(RuntimeError, match="resolver boom"):
        _ = rot.mean(dim="sample", validate=True)
