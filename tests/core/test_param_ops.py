import importlib
from pathlib import Path
import numpy as np
import pytest
import warnings
import xarray as xr

from tal.core import AnalysisObject, ParamEvalOptions, ParamSelectOptions
from tal.spatial import Position


def _ao_unbatched() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 10.0, 20.0, 30.0])},
        coords={"sample": [0, 1, 2, 3], "tau": ("sample", [0.0, 1.0, 1.0, 2.0])},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")


def _ao_unbatched_valid() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 10.0, 20.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "group_size": xr.DataArray(np.asarray(3, dtype="int64"), dims=()),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )


def _ao_batched() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2, 3],
            "phase": (("trial", "sample"), [[0.0, 0.5, 1.0, 1.5], [0.0, 1.0, np.nan, np.nan]]),
            "group_size": ("trial", [4, 2]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )


def _ao_invalid_interior() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [5.0, 6.0, 7.0, 8.0])},
        coords={"sample": [0, 1, 2, 3], "tau": ("sample", [0.0, np.nan, 2.0, 3.0])},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")


def _ao_with_aux_sequence_coord() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 10.0, 20.0])},
        coords={
            "sample": [0, 1, 2],
            "phase": ("sample", [100.0, 101.0, 102.0]),
            "aux": ("sample", [7.0, 8.0, 9.0]),
        },
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="phase")


def _ao_with_string_sequence_var() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 10.0, 20.0]),
            "label": (("sample",), ["a", "b", "c"]),
        },
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")


def _ao_multi_batch(
    *,
    trial_labels: tuple[str, ...] = ("a", "b"),
    sensor_labels: tuple[str, ...] = ("s0", "s1"),
    offset: float = 0.0,
) -> AnalysisObject:
    base = np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64")
    values = np.zeros((len(trial_labels), len(sensor_labels), base.size), dtype="float64")
    phase = np.zeros_like(values)
    for i in range(len(trial_labels)):
        for j in range(len(sensor_labels)):
            values[i, j] = base + offset + i * 10.0 + j
            phase[i, j] = base
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), values)},
        coords={
            "trial": list(trial_labels),
            "sensor": list(sensor_labels),
            "sample": np.arange(base.size, dtype="int64"),
            "phase": (("trial", "sensor", "sample"), phase),
            "group_size": (("trial", "sensor"), np.full((len(trial_labels), len(sensor_labels)), base.size)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )


def test_param_ops_001_index_scalar_and_vector_shapes() -> None:
    """ID: PARAM_OPS_001_index_scalar_and_vector_shapes."""
    ao = _ao_unbatched()
    scalar = ao.param.index(0.2)
    vector = ao.param.index([0.2, 1.9])
    assert scalar.dims == ()
    assert int(scalar.values) == 0
    assert tuple(vector.dims) == ("query",)
    np.testing.assert_array_equal(vector.values, [0, 3])


def test_param_ops_002_sel_point_nearest_preserves_sample_index() -> None:
    """ID: PARAM_OPS_002_sel_point_nearest_preserves_sample_index."""
    ao = _ao_unbatched()
    out = ao.param.sel([0.1, 1.9])
    assert tuple(out.data["value"].dims) == ("sample",)
    np.testing.assert_array_equal(out.data.coords["sample_index"].values, [0, 3])
    np.testing.assert_array_equal(out.data.coords["valid"].values, [True, True])


def test_param_ops_003_sel_slice_packed_updates_validity() -> None:
    """ID: PARAM_OPS_003_sel_slice_packed_updates_validity."""
    ao = _ao_batched()
    out = ao.param.sel(slice(0.25, 1.0), opts=ParamSelectOptions(layout="packed"))
    core = out.data.attrs["tal"]["core"]
    assert core["validity"]["sequence_size_coord"] == "group_size"
    np.testing.assert_array_equal(out.data.coords["group_size"].values, [2, 1])
    np.testing.assert_array_equal(out.data.coords["valid"].sel(trial="a").values, [True, True, False, False])
    np.testing.assert_array_equal(out.data.coords["valid"].sel(trial="b").values, [True, False, False, False])


def test_param_ops_004_sel_slice_padded_masks_coords() -> None:
    """ID: PARAM_OPS_004_sel_slice_padded_masks_coords."""
    ao = _ao_batched()
    out = ao.param.sel(slice(0.25, 1.0), opts=ParamSelectOptions(layout="padded"))
    assert out.data.sizes["sample"] == 2
    np.testing.assert_array_equal(out.data.coords["group_size"].values, [2, 1])
    assert bool(out.data.coords["valid"].sel(trial="b", sample=1).item()) is False
    assert int(out.data.coords["sample_index"].sel(trial="b", sample=1).item()) == -1
    assert np.isnan(out.data["value"].sel(trial="b", sample=1).item())


def test_param_ops_005_at_linear_duplicate_invalid_default() -> None:
    """ID: PARAM_OPS_005_at_linear_duplicate_invalid_default."""
    ao = _ao_unbatched()
    out = ao.param.at([1.0], opts=ParamEvalOptions(method="linear"))
    assert np.isnan(out.data["value"].isel(sample=0).item())


def test_param_ops_006_at_linear_duplicate_left_right_policies() -> None:
    """ID: PARAM_OPS_006_at_linear_duplicate_left_right_policies."""
    ao = _ao_unbatched()
    left = ao.param.at([1.0], opts=ParamEvalOptions(method="linear", duplicate_policy="left"))
    right = ao.param.at([1.0], opts=ParamEvalOptions(method="linear", duplicate_policy="right"))
    assert float(left.data["value"].isel(sample=0).item()) == 10.0
    assert float(right.data["value"].isel(sample=0).item()) == 20.0


def test_param_ops_007_at_duplicate_raise_policy_errors() -> None:
    """ID: PARAM_OPS_007_at_duplicate_raise_policy_errors."""
    ao = _ao_unbatched()
    with pytest.raises(ValueError) as err:
        ao.param.at([1.0], opts=ParamEvalOptions(method="linear", duplicate_policy="raise"))
    assert "duplicate parameter bracket" in str(err.value)


def test_param_ops_008_resample_to_sequence_grid_updates_validity() -> None:
    """ID: PARAM_OPS_008_resample_to_sequence_grid_updates_validity."""
    ao = _ao_batched()
    grid = xr.DataArray([0.0, 0.5, 1.0, 1.5], dims=("sample",))
    out = ao.param.resample_to(grid)
    core = out.data.attrs["tal"]["core"]
    assert core["validity"]["sequence_size_coord"] == "group_size"
    np.testing.assert_array_equal(out.data.coords["group_size"].values, [4, 3])


def test_param_ops_009_resample_to_stacked_query_drops_validity() -> None:
    """ID: PARAM_OPS_009_resample_to_stacked_query_drops_validity."""
    ao = _ao_batched()
    grid = xr.DataArray(
        np.asarray(
            [
                [[0.0, 0.5], [1.0, 1.5]],
                [[0.0, 1.0], [np.nan, np.nan]],
            ],
            dtype="float64",
        ),
        dims=("trial", "q", "extra"),
        coords={"trial": ["a", "b"], "q": [0, 1], "extra": [0, 1]},
    )
    out = ao.param.resample_to(grid)
    assert "validity" not in out.data.attrs["tal"]["core"]


def test_param_ops_010_interp_like_inner_left_batch_join() -> None:
    """ID: PARAM_OPS_010_interp_like_inner_left_batch_join."""
    ao = _ao_batched()
    other_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[100.0, 101.0], [200.0, 201.0]])},
        coords={
            "trial": ["b", "c"],
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0], [10.0, 11.0]]),
            "group_size": ("trial", [2, 2]),
        },
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )
    inner = ao.param.interp_like(other, batch_join="inner")
    left = ao.param.interp_like(other, batch_join="left")
    assert list(inner.data.coords["trial"].values) == ["b"]
    assert list(left.data.coords["trial"].values) == ["a", "b"]
    assert np.isnan(left.data["value"].sel(trial="a").values).all()


def test_param_ops_011_slice_masks_invalid_interior_samples() -> None:
    """ID: PARAM_OPS_011_slice_masks_invalid_interior_samples."""
    ao = _ao_invalid_interior()
    out = ao.param.sel(slice(0.0, 3.0), opts=ParamSelectOptions(layout="packed"))
    np.testing.assert_array_equal(out.data.coords["valid"].values, [True, False, True, True])
    assert np.isnan(out.data["value"].isel(sample=1).item())


def test_param_ops_012_sel_scalar_point_collapses_singleton_query_dim() -> None:
    """ID: PARAM_OPS_012_sel_scalar_point_collapses_singleton_query_dim."""
    ao = _ao_unbatched()
    out = ao.param.sel(1.9)
    assert tuple(out.data["value"].dims) == ()
    assert float(out.data["value"].item()) == 30.0
    assert "sample" not in out.data.dims


def test_param_ops_013_point_invalid_query_does_not_leak_index0_value() -> None:
    """ID: PARAM_OPS_013_point_invalid_query_does_not_leak_index0_value."""
    ao = _ao_unbatched()
    out = ao.param.sel([np.nan, 1.9])
    np.testing.assert_array_equal(out.data.coords["valid"].values, [False, True])
    np.testing.assert_array_equal(out.data.coords["sample_index"].values, [-1, 3])
    assert np.isnan(out.data["value"].isel(sample=0).item())
    assert float(out.data["value"].isel(sample=1).item()) == 30.0


def test_param_ops_014_slice_step_rejected() -> None:
    """ID: PARAM_OPS_014_slice_step_rejected."""
    ao = _ao_unbatched()
    with pytest.raises(ValueError) as err:
        ao.param.sel(slice(0.0, 2.0, 0.5))
    assert "slice step is not supported" in str(err.value)


def test_param_ops_015_non_numeric_sequence_policy_consistent_point_and_slice() -> None:
    """ID: PARAM_OPS_015_non_numeric_sequence_policy_consistent_point_and_slice."""
    ao = _ao_with_string_sequence_var()
    with pytest.raises(TypeError) as err_point:
        ao.param.sel([0.2, 1.8])
    assert "non-numeric sequence variable" in str(err_point.value)
    with pytest.raises(TypeError) as err_slice:
        ao.param.sel(slice(0.0, 2.0))
    assert "non-numeric sequence variable" in str(err_slice.value)


def test_param_ops_016_slice_path_emits_no_index_rename_warning() -> None:
    """ID: PARAM_OPS_016_slice_path_emits_no_index-rename_warning."""
    ao = _ao_batched()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ao.param.sel(slice(0.25, 1.0), opts=ParamSelectOptions(layout="padded"))
    messages = [str(w.message) for w in caught]
    assert not any("does not create an index anymore" in msg for msg in messages)


def test_param_ops_017_non_left_packed_validity_drops_sequence_size_coord() -> None:
    """ID: PARAM_OPS_017_non_left_packed_validity_drops_sequence_size_coord."""
    ao = _ao_unbatched_valid()
    sel_out = ao.param.sel([np.nan, 1.0])
    at_out = ao.param.at([np.nan, 1.0])
    for out in (sel_out, at_out):
        core = out.data.attrs["tal"]["core"]
        assert "validity" not in core
        assert "group_size" not in out.data.coords


def test_param_ops_018_slice_invalid_sequence_coords_masked() -> None:
    """ID: PARAM_OPS_018_slice_invalid_sequence_coords_masked."""
    ao = _ao_batched()
    out = ao.param.sel(slice(0.25, 1.0), opts=ParamSelectOptions(layout="padded"))
    assert bool(out.data.coords["valid"].sel(trial="b", sample=1).item()) is False
    assert np.isnan(out.data.coords["phase"].sel(trial="b", sample=1).item())


def test_param_ops_019_point_invalid_sequence_coords_masked() -> None:
    """ID: PARAM_OPS_019_point_invalid_sequence_coords_masked."""
    ao = _ao_with_aux_sequence_coord()
    out = ao.param.sel([np.nan, 101.0])
    np.testing.assert_array_equal(out.data.coords["valid"].values, [False, True])
    assert np.isnan(out.data.coords["phase"].isel(sample=0).item())
    assert np.isnan(out.data.coords["aux"].isel(sample=0).item())


def test_param_ops_020_query_scalar_coord_collision_sanitized() -> None:
    """ID: PARAM_OPS_020_query_scalar_coord_collision_sanitized."""
    ao_u = _ao_unbatched()
    q_sample = xr.DataArray([0.0, 2.0], dims=("query",), coords={"sample": 0})
    ao_u.param.sel(q_sample)
    ao_u.param.at(q_sample)
    ao_u.param.resample_to(q_sample)

    ao_b = _ao_batched()
    q_trial = xr.DataArray([0.25, 1.0], dims=("query",), coords={"trial": "a"})
    ao_b.param.sel(q_trial)
    ao_b.param.at(q_trial)
    ao_b.param.resample_to(q_trial)


def test_param_ops_021_query_index_labels_do_not_realign_attached_param() -> None:
    """ID: PARAM_OPS_021_query_index_labels_do_not_realign_attached_param."""
    ao = _ao_unbatched()
    q_labeled = xr.DataArray([0.0, 2.0], dims=("query",), coords={"query": [10, 20]})
    for op in (ao.param.sel, ao.param.at, ao.param.resample_to):
        out = op(q_labeled)
        np.testing.assert_allclose(out.data.coords["tau"].values, [0.0, 2.0])
    q_reordered = xr.DataArray([2.0, 0.0], dims=("query",), coords={"query": [1, 0]})
    out_reordered = ao.param.at(q_reordered)
    np.testing.assert_allclose(out_reordered.data.coords["tau"].values, [2.0, 0.0])


def test_param_ops_022_duplicate_query_labels_fail_fast() -> None:
    """ID: PARAM_OPS_022_duplicate_query_labels_fail_fast."""
    ao = _ao_unbatched()
    q_dup = xr.DataArray([0.0, 2.0], dims=("query",), coords={"query": [0, 0]})
    for op in (ao.param.sel, ao.param.at, ao.param.resample_to):
        with pytest.raises(ValueError) as err:
            op(q_dup)
        assert "labels along 'query' must be unique" in str(err.value)


def test_param_ops_023_interp_like_duplicate_source_batch_labels_fail_fast() -> None:
    """ID: PARAM_OPS_023_interp_like_duplicate_source_batch_labels_fail_fast."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={
            "trial": ["a", "a"],
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
        },
    )
    src = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    other = _ao_batched()
    with pytest.raises(ValueError) as err:
        src.param.interp_like(other, batch_join="inner")
    assert "labels along 'trial' must be unique" in str(err.value)


def test_param_ops_024_query_dim_collision_with_dataset_dim_fails_fast() -> None:
    """ID: PARAM_OPS_024_query_dim_collision_with_dataset_dim_fails_fast."""
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "aux": (("query",), [100.0, 200.0]),
        },
        coords={"sample": [0, 1, 2], "query": ["q0", "q1"], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        ao.param.at([0.0, 2.0])
    assert "query_dim 'query' collides with existing non-sequence dim" in str(err.value)


def test_param_ops_025_slice_temp_dim_collision_avoided() -> None:
    """ID: PARAM_OPS_025_slice_temp_dim_collision_avoided."""
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "extra": (("sample__slice",), [10.0, 20.0]),
        },
        coords={
            "sample": [0, 1, 2],
            "sample__slice": [0, 1],
            "tau": ("sample", [0.0, 1.0, 2.0]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out = ao.param.sel(slice(0.0, 1.0))
    assert tuple(out.data["extra"].dims) == ("sample__slice",)
    np.testing.assert_array_equal(out.data["extra"].values, [10.0, 20.0])


def test_param_ops_026_reserved_name_collision_user_valid_rejected() -> None:
    """ID: PARAM_OPS_026_reserved_name_collision_user_valid_rejected."""
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "valid": (("sample",), [5.0, 6.0, 7.0]),
        },
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        ao.param.at([0.5, 1.5])
    assert "reserved metadata name collision" in str(err.value)


def test_param_ops_027_reserved_name_collision_param_valid_rejected() -> None:
    """ID: PARAM_OPS_027_reserved_name_collision_param_valid_rejected."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 1, 2], "valid": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="valid",
    )
    with pytest.raises(ValueError) as err:
        ao.param.at([0.5, 1.5])
    assert "reserved metadata name collision" in str(err.value)


def test_param_ops_028_reserved_name_collision_param_sample_index_rejected() -> None:
    """ID: PARAM_OPS_028_reserved_name_collision_param_sample_index_rejected."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 1, 2], "sample_index": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="sample_index",
    )
    with pytest.raises(ValueError) as err:
        ao.param.sel([0.4, 1.6])
    assert "reserved metadata name collision" in str(err.value)


def test_param_ops_029_sel_point_empty_sequence_returns_all_invalid() -> None:
    """ID: PARAM_OPS_029_sel_point_empty_sequence_returns_all_invalid."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray([], dtype="float64"))},
        coords={
            "sample": np.asarray([], dtype="int64"),
            "tau": ("sample", np.asarray([], dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out = ao.param.sel([0.0])
    assert tuple(out.data["value"].dims) == ("sample",)
    assert out.data.sizes["sample"] == 1
    assert bool(out.data.coords["valid"].isel(sample=0).item()) is False
    assert int(out.data.coords["sample_index"].isel(sample=0).item()) == -1
    assert np.isnan(out.data["value"].isel(sample=0).item())


def test_param_ops_030_non_numeric_query_rejected_across_param_apis() -> None:
    """ID: PARAM_OPS_030_non_numeric_query_rejected_across_param_apis."""
    ao = _ao_unbatched()
    for op in (
        lambda: ao.param.index(["x"]),
        lambda: ao.param.sel(["x"]),
        lambda: ao.param.at(["x"]),
        lambda: ao.param.resample_to(["x"]),
    ):
        with pytest.raises(ValueError) as err:
            op()
        assert "query values must be numeric" in str(err.value)
    other = xr.DataArray(["x"], dims=("sample",), name="tau")
    with pytest.raises(ValueError) as err_interp:
        ao.param.interp_like(other)
    assert "query values must be numeric" in str(err_interp.value)


def test_param_ops_031_reserved_metadata_guard_allows_tal_chaining() -> None:
    """ID: PARAM_OPS_031_reserved_metadata_guard_allows_tal_chaining."""
    ao = _ao_unbatched()
    out_at = ao.param.at([0.5, 1.5])
    chained_at = out_at.param.at([0.75])
    assert float(chained_at.data["value"].isel(sample=0).item()) == 10.0
    out_sel = ao.param.sel([0.2, 1.8])
    chained_sel = out_sel.param.sel([1.8])
    assert float(chained_sel.data["value"].isel(sample=0).item()) == 30.0


def test_param_ops_032_reserved_metadata_guard_rejects_user_data_var_collisions() -> None:
    """ID: PARAM_OPS_032_reserved_metadata_guard_rejects_user_data_var_collisions."""
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "sample_index": (("sample",), [9, 9, 9]),
        },
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        ao.param.sel([0.2, 1.8])
    assert "reserved metadata name collision" in str(err.value)


def test_param_ops_038_reserved_metadata_guard_rejects_unowned_bool_int_coords() -> None:
    """ID: PARAM_OPS_038_reserved_metadata_guard_rejects_unowned_bool_int_coords."""
    ds_valid = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "valid": ("sample", [True, False, True]),
        },
    )
    ao_valid = AnalysisObject.from_data(
        ds_valid,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err_valid:
        ao_valid.param.at([0.5, 1.5])
    assert "reserved metadata name collision" in str(err_valid.value)

    ds_index = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "sample_index": ("sample", [0, 1, 2]),
        },
    )
    ao_index = AnalysisObject.from_data(
        ds_index,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err_index:
        ao_index.param.sel([0.4, 1.6])
    assert "reserved metadata name collision" in str(err_index.value)


def test_param_ops_039_interp_like_inner_mixed_null_representation_fail_tal_valueerror() -> None:
    """ID: PARAM_OPS_039_interp_like_inner_mixed_null_representation_fail_tal_valueerror."""
    ds_nan = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": np.asarray([np.nan], dtype="float64"),
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0]]),
        },
    )
    ds_none = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": np.asarray([None], dtype=object),
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0]]),
        },
    )
    src = AnalysisObject.from_data(
        ds_nan,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    other = AnalysisObject.from_data(
        ds_none,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    with pytest.raises(ValueError) as err:
        src.param.interp_like(other, batch_join="inner")
    assert "batch labels are not representable" in str(err.value)


def test_param_ops_040_query_dim_invalid_type_or_empty_rejected() -> None:
    """ID: PARAM_OPS_040_query_dim_invalid_type_or_empty_rejected."""
    ao = _ao_unbatched()
    with pytest.raises(ValueError) as err_type:
        ao.param.at([0.0, 1.0], opts=ParamEvalOptions(query_dim=1))  # type: ignore[arg-type]
    assert "query_dim" in str(err_type.value)
    with pytest.raises(ValueError) as err_none:
        ao.param.sel([0.0, 1.0], opts=ParamSelectOptions(query_dim=None))  # type: ignore[arg-type]
    assert "query_dim" in str(err_none.value)
    with pytest.raises(ValueError) as err_empty:
        ao.param.index([0.0, 1.0], opts=ParamSelectOptions(query_dim=""))
    assert "query_dim" in str(err_empty.value)


def test_param_ops_041_query_dim_reserved_names_rejected_across_ops() -> None:
    """ID: PARAM_OPS_041_query_dim_reserved_names_rejected_across_ops."""
    ao = _ao_unbatched()
    with pytest.raises(ValueError) as err_at:
        ao.param.at([0.0, 1.0], opts=ParamEvalOptions(query_dim="valid"))
    assert "reserved" in str(err_at.value)
    with pytest.raises(ValueError) as err_sel:
        ao.param.sel([0.0, 1.0], opts=ParamSelectOptions(query_dim="sample_index"))
    assert "reserved" in str(err_sel.value)
    with pytest.raises(ValueError) as err_resample:
        ao.param.resample_to([0.0, 1.0], opts=ParamEvalOptions(query_dim="valid"))
    assert "reserved" in str(err_resample.value)


def test_param_ops_042_reserved_metadata_spoof_attrs_rejected() -> None:
    """ID: PARAM_OPS_042_reserved_metadata_spoof_attrs_rejected."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "valid": ("sample", [True, False, True]),
        },
    )
    ds.coords["valid"].attrs["tal_reserved_owner"] = "param_ops"
    ds.coords["valid"].attrs["tal_reserved_name"] = "valid"
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        ao.param.at([0.5, 1.5])
    assert "reserved metadata name collision" in str(err.value)


def test_param_ops_033_interp_like_inner_keeps_matching_null_labels() -> None:
    """ID: PARAM_OPS_033_interp_like_inner_keeps_matching_null_labels."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": [np.nan],
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0]]),
        },
    )
    src = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    other = AnalysisObject.from_data(
        ds.copy(deep=True),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    out = src.param.interp_like(other, batch_join="inner")
    assert out.data.sizes["trial"] == 1
    assert bool(np.isnan(out.data.coords["trial"].values[0]))


def test_param_ops_034_query_dim_collision_with_scalar_coord_or_data_var_fails_fast() -> None:
    """ID: PARAM_OPS_034_query_dim_collision_with_scalar_coord_or_data_var_fails_fast."""
    ds_coord = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0]), "query": 99},
    )
    ao_coord = AnalysisObject.from_data(
        ds_coord,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err_coord:
        ao_coord.param.at([0.0, 1.0])
    assert "collides with existing coordinate name" in str(err_coord.value)

    ds_var = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "query": (("sample",), [5.0, 6.0, 7.0]),
        },
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ao_var = AnalysisObject.from_data(
        ds_var,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err_var:
        ao_var.param.index([0.0, 1.0])
    assert "collides with existing data variable name" in str(err_var.value)


def test_param_ops_035_index_result_no_malformed_scalar_query_coord() -> None:
    """ID: PARAM_OPS_035_index_result_no_malformed_scalar_query_coord."""
    ao = _ao_unbatched()
    out = ao.param.index([0.0, 2.0])
    assert tuple(out.dims) == ("query",)
    assert tuple(out.coords["query"].dims) == ("query",)
    np.testing.assert_array_equal(out.coords["query"].values, [0, 1])


def test_param_ops_036_slice_temp_dim_avoids_coord_and_data_var_collisions() -> None:
    """ID: PARAM_OPS_036_slice_temp_dim_avoids_coord_and_data_var_collisions."""
    ds_coord = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0]), "sample__slice": 123},
    )
    ao_coord = AnalysisObject.from_data(
        ds_coord,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out_coord = ao_coord.param.sel(slice(0.0, 1.0))
    assert out_coord.data.sizes["sample"] == 3
    assert int(out_coord.data.coords["sample__slice"].item()) == 123

    ds_var = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "sample__slice": xr.DataArray(np.asarray(5.0), dims=()),
        },
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ao_var = AnalysisObject.from_data(
        ds_var,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out_var = ao_var.param.sel(slice(0.0, 1.0))
    assert tuple(out_var.data["sample__slice"].dims) == ()
    assert float(out_var.data["sample__slice"].item()) == 5.0


def test_param_ops_037_non_numeric_query_and_query_namespace_errors_are_tal_owned() -> None:
    """ID: PARAM_OPS_037_non_numeric_query_and_query_namespace_errors_are_tal_owned."""
    ao = _ao_unbatched()
    with pytest.raises(ValueError) as err_num:
        ao.param.at(["x"])
    assert "query values must be numeric" in str(err_num.value)

    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0]), "query": 7},
    )
    ao_collision = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err_collision:
        ao_collision.param.at([0.0, 1.0])
    msg = str(err_collision.value)
    assert "collides with existing coordinate name" in msg
    assert "already exists as a scalar variable" not in msg


def test_param_ops_043_reserved_token_survives_public_copy_roundtrip() -> None:
    """ID: PARAM_OPS_043_reserved_token_survives_public_copy_roundtrip."""
    ao = _ao_unbatched()
    out = ao.param.at([0.25, 0.75])
    copy_ao = AnalysisObject(out.data)
    chained = copy_ao.param.at([0.5])
    assert "valid" in chained.data.coords
    assert bool(chained.data.coords["valid"].dtype == bool)


def test_param_ops_044_sel_slice_non_numeric_bounds_rejected_tal_valueerror() -> None:
    """ID: PARAM_OPS_044_sel_slice_non_numeric_bounds_rejected_tal_valueerror."""
    ao = _ao_unbatched()
    with pytest.raises(ValueError) as err_start:
        ao.param.sel(slice("x", 1.0))
    assert "slice.start" in str(err_start.value)
    with pytest.raises(ValueError) as err_stop:
        ao.param.sel(slice(0.0, "y"))
    assert "slice.stop" in str(err_stop.value)


def test_param_ops_045_select_layout_invalid_rejected_point_path() -> None:
    """ID: PARAM_OPS_045_select_layout_invalid_rejected_point_path."""
    ao = _ao_unbatched()
    with pytest.raises(ValueError) as err:
        ao.param.sel([0.5], opts=ParamSelectOptions(layout="garbage"))  # type: ignore[arg-type]
    assert "opts.layout" in str(err.value)


def test_param_ops_046_dask_sel_indexers_materialized_no_chunked_indexer_error() -> None:
    """ID: PARAM_OPS_046_dask_sel_indexers_materialized_no_chunked_indexer_error."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out = ao.param.sel([0.5, 1.5])
    np.testing.assert_allclose(out.data["value"].values, [0.0, 10.0])


def test_param_ops_047_index_rejects_invalid_select_method_and_layout() -> None:
    """ID: PARAM_OPS_047_index_rejects_invalid_select_method_and_layout."""
    ao = _ao_unbatched()
    with pytest.raises(ValueError) as err_method:
        ao.param.index([0.5], opts=ParamSelectOptions(method="linear"))  # type: ignore[arg-type]
    assert "opts.method" in str(err_method.value)
    with pytest.raises(ValueError) as err_layout:
        ao.param.index([0.5], opts=ParamSelectOptions(layout="garbage"))  # type: ignore[arg-type]
    assert "opts.layout" in str(err_layout.value)


def test_param_ops_048_param_accessor_opts_type_checked() -> None:
    """ID: PARAM_OPS_048_param_accessor_opts_type_checked."""
    ao = _ao_unbatched()
    with pytest.raises(TypeError) as err_sel:
        ao.param.sel([0.5], opts={"layout": "packed"})  # type: ignore[arg-type]
    assert "ParamSelectOptions" in str(err_sel.value)
    with pytest.raises(TypeError) as err_at:
        ao.param.at([0.5], opts={"method": "nearest"})  # type: ignore[arg-type]
    assert "ParamEvalOptions" in str(err_at.value)
    with pytest.raises(TypeError) as err_idx:
        ao.param.index([0.5], opts={"query_dim": "q"})  # type: ignore[arg-type]
    assert "ParamSelectOptions" in str(err_idx.value)


def test_param_ops_049_interp_like_invalid_ao_like_rejected_typeerror() -> None:
    """ID: PARAM_OPS_049_interp_like_invalid_ao_like_rejected_typeerror."""

    class _Bad:
        unsafe_data = "not-a-dataset"

    ao = _ao_unbatched()
    with pytest.raises(TypeError) as err:
        ao.param.interp_like(_Bad())  # type: ignore[arg-type]
    assert "AnalysisObject, xarray.Dataset, or xarray.DataArray" in str(err.value)


def test_param_ops_056_interp_like_duck_typed_impostor_dataset_rejected() -> None:
    """ID: PARAM_OPS_056_interp_like_duck_typed_impostor_dataset_rejected."""

    class _Impostor:
        @property
        def unsafe_data(self) -> xr.Dataset:
            return _ao_unbatched().unsafe_data

    ao = _ao_unbatched()
    with pytest.raises(TypeError) as err:
        ao.param.interp_like(_Impostor())  # type: ignore[arg-type]
    assert "AnalysisObject, xarray.Dataset, or xarray.DataArray" in str(err.value)


def test_param_ops_050_reserved_dim_name_collision_rejected() -> None:
    """ID: PARAM_OPS_050_reserved_dim_name_collision_rejected."""
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "other": (("valid",), [10.0, 11.0]),
        },
        coords={
            "sample": [0, 1, 2],
            "valid": ["v0", "v1"],
            "tau": ("sample", [0.0, 1.0, 2.0]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("valid",),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        ao.param.sel([0.5, 1.5])
    assert "reserved metadata name collision" in str(err.value)


def test_param_ops_051_eval_preflights_sample_index_collision() -> None:
    """ID: PARAM_OPS_051_eval_preflights_sample_index_collision."""
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "sample_index": (("sample",), [5, 6, 7]),
        },
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        ao.param.at([0.5, 1.5])
    assert "reserved metadata name collision" in str(err.value)


def test_param_ops_052_batched_dask_slice_packed_no_chunked_indexer_error() -> None:
    """ID: PARAM_OPS_052_batched_dask_slice_packed_no_chunked_indexer_error."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]), chunks=(1, 2)),
            )
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]), chunks=(1, 2)),
            ),
            "group_size": ("trial", [3, 3]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )
    out = ao.param.sel(slice(0.5, 1.5), opts=ParamSelectOptions(layout="packed"))
    assert out.data.sizes["sample"] == 3
    np.testing.assert_array_equal(out.data.coords["group_size"].values, [1, 1])


def test_param_ops_053_batched_dask_slice_padded_no_chunked_indexer_error() -> None:
    """ID: PARAM_OPS_053_batched_dask_slice_padded_no_chunked_indexer_error."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]), chunks=(1, 2)),
            )
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]), chunks=(1, 2)),
            ),
            "group_size": ("trial", [3, 3]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )
    out = ao.param.sel(slice(0.5, 1.5), opts=ParamSelectOptions(layout="padded"))
    assert out.data.sizes["sample"] == 1
    np.testing.assert_array_equal(out.data.coords["group_size"].values, [1, 1])


def test_param_ops_054_multi_batch_interp_like_supported() -> None:
    """ID: PARAM_OPS_054_multi_batch_interp_like_supported."""
    left = _ao_multi_batch(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    right = _ao_multi_batch(trial_labels=("b", "c"), sensor_labels=("s0", "s1"), offset=100.0)
    out = left.param.interp_like(right, batch_join="inner")
    assert {"trial", "sensor", "sample"} <= set(out.data["value"].dims)
    assert list(out.data.coords["trial"].values) == ["b"]
    assert list(out.data.coords["sensor"].values) == ["s0", "s1"]


def test_orch_topo_parity_002_interp_like_multi_batch_behavior_parity() -> None:
    """ID: ORCH_TOPO_PARITY_002_interp_like_multi_batch_behavior_parity."""
    left = _ao_multi_batch(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    right = _ao_multi_batch(trial_labels=("b", "a"), sensor_labels=("s1", "s0"), offset=100.0)
    out = left.param.interp_like(right, batch_join="inner")
    assert set(out.data["value"].dims) == {"trial", "sensor", "sample"}
    assert list(out.data.coords["trial"].values) == ["a", "b"]
    assert list(out.data.coords["sensor"].values) == ["s0", "s1"]


def test_orch_finalize_parity_002_interp_like_restored_path_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ORCH_FINALIZE_PARITY_002_interp_like_restored_path_stable."""
    interp_mod = importlib.import_module("tal.core.param_ops.interp_like")
    calls = {"restore_and_finalize": 0}
    orig_restore = interp_mod.restore_and_finalize

    def _count_restore(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["restore_and_finalize"] += 1
        return orig_restore(*args, **kwargs)

    monkeypatch.setattr(interp_mod, "restore_and_finalize", _count_restore)

    left = _ao_multi_batch(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    right = _ao_multi_batch(trial_labels=("b", "a"), sensor_labels=("s1", "s0"), offset=100.0)
    out = left.param.interp_like(right, batch_join="inner")
    assert set(out.data["value"].dims) == {"trial", "sensor", "sample"}
    assert list(out.data.coords["trial"].values) == ["a", "b"]
    assert list(out.data.coords["sensor"].values) == ["s0", "s1"]
    assert calls["restore_and_finalize"] >= 1


def test_param_ops_055_left_packed_dask_validity_no_eager_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_OPS_055_left_packed_dask_validity_no_eager_compute."""
    da = pytest.importorskip("dask.array")
    validity_layout_mod = importlib.import_module("tal.core.validity_layout")
    valid = xr.DataArray(
        da.from_array(np.asarray([True, True, False], dtype=bool), chunks=2),
        dims=("sample",),
    )

    def _boom_compute(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected eager compute")

    monkeypatch.setattr(da.Array, "compute", _boom_compute, raising=True)
    assert validity_layout_mod.is_left_packed_mask(valid, sequence_dim="sample") is False


def test_param_ops_057_sel_point_chunked_indexers_no_dataarray_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_OPS_057_sel_point_chunked_indexers_no_dataarray_compute."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )

    def _boom_compute(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected DataArray.compute call")

    monkeypatch.setattr(xr.DataArray, "compute", _boom_compute, raising=True)
    out = ao.param.sel([0.5, 1.5])
    assert tuple(out.unsafe_data["value"].dims) == ("sample",)


def test_param_ops_058_sel_slice_chunked_indexers_no_dataarray_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_OPS_058_sel_slice_chunked_indexers_no_dataarray_compute."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]), chunks=(1, 2)),
            )
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]), chunks=(1, 2)),
            ),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )

    def _boom_compute(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected DataArray.compute call")

    monkeypatch.setattr(xr.DataArray, "compute", _boom_compute, raising=True)
    out = ao.param.sel(slice(0.25, 1.5), opts=ParamSelectOptions(layout="packed"))
    assert tuple(out.unsafe_data["value"].dims) == ("trial", "sample")


def test_param_ops_059_sel_slice_padded_chunked_explicit_scalar_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_OPS_059_sel_slice_padded_chunked_explicit_scalar_boundary."""
    da = pytest.importorskip("dask.array")
    select_mod = importlib.import_module("tal.core.param_ops.select")
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]), chunks=(1, 2)),
            )
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]), chunks=(1, 2)),
            ),
            "group_size": ("trial", [3, 3]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )
    calls = {"n": 0}
    original = select_mod.scalar_int_boundary

    def _count_scalar_boundary(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(select_mod, "scalar_int_boundary", _count_scalar_boundary)
    out = ao.param.sel(slice(0.5, 1.5), opts=ParamSelectOptions(layout="padded"))
    assert out.data.sizes["sample"] == 1
    assert calls["n"] == 1


def test_param_ops_060_sel_slice_packed_chunked_no_scalar_boundary_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_OPS_060_sel_slice_packed_chunked_no_scalar_boundary_compute."""
    da = pytest.importorskip("dask.array")
    select_mod = importlib.import_module("tal.core.param_ops.select")
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]), chunks=(1, 2)),
            )
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (
                ("trial", "sample"),
                da.from_array(np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]), chunks=(1, 2)),
            ),
            "group_size": ("trial", [3, 3]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )

    def _boom_scalar_boundary(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected scalar boundary compute")

    monkeypatch.setattr(select_mod, "scalar_int_boundary", _boom_scalar_boundary)
    out = ao.param.sel(slice(0.5, 1.5), opts=ParamSelectOptions(layout="packed"))
    assert out.data.sizes["sample"] == 3


def test_param_ops_061_index_reserved_metadata_collision_guard_parity() -> None:
    """ID: PARAM_OPS_061_index_reserved_metadata_collision_guard_parity."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "valid": ("sample", [True, True, False]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        ao.param.index([0.5, 1.5])
    assert "reserved metadata name collision" in str(err.value)


def test_param_ops_062_param_ops_do_not_mutate_source_reserved_coord_attrs() -> None:
    """ID: PARAM_OPS_062_param_ops_do_not_mutate_source_reserved_coord_attrs."""
    seed = _ao_unbatched().param.sel([0.2, 1.8])
    valid_before = dict(seed.unsafe_data.coords["valid"].attrs)
    sample_before = dict(seed.unsafe_data.coords["sample_index"].attrs)
    seed.param.sel([0.2])
    seed.param.at([0.2])
    seed.param.resample_to([0.2])
    assert dict(seed.unsafe_data.coords["valid"].attrs) == valid_before
    assert dict(seed.unsafe_data.coords["sample_index"].attrs) == sample_before


def test_orch_parity_002_param_interp_like_behavior_parity_after_migration() -> None:
    """ID: ORCH_PARITY_002_param_interp_like_behavior_parity_after_migration."""
    test_orch_topo_parity_002_interp_like_multi_batch_behavior_parity()


def test_spatial_core_133_temporal_vector_like_method_execution_paths_use_blockwise_vectorize_false() -> None:
    """ID: SPATIAL_CORE_133_temporal_vector_like_method_execution_paths_use_blockwise_vectorize_false."""
    text = Path("tal/core/param_engine/map_apply.py").read_text(encoding="utf-8")
    gather_body = text.split("def gather_along_sequence(", 1)[1].split("\ndef ", 1)[0]
    assert "vectorize=False" in gather_body
    assert "vectorize=True" not in gather_body


def test_spatial_hard_154_temporal_vector_like_rejects_hidden_or_explicit_python_row_loop_execution_paths() -> None:
    """ID: SPATIAL_HARD_154_temporal_vector_like_rejects_hidden_or_explicit_python_row_loop_execution_paths."""
    text = Path("tal/core/param_engine/map_apply.py").read_text(encoding="utf-8")
    apply_body = text.split("def apply_param_map(", 1)[1].split("\ndef ", 1)[0]
    assert "vectorize=True" not in apply_body
    assert "for i in range(" not in apply_body


def test_param_ops_inherited_spatial_accessor_matches_ao_behavior_parity() -> None:
    """ID: PARAM_OPS_064_inherited_spatial_accessor_matches_ao_behavior_parity."""
    ds = xr.Dataset(
        data_vars={"position": (("sample", "axis"), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        coords={"sample": [0, 1], "axis": ["x", "y", "z"], "time_s": ("sample", [0.0, 1.0])},
    )
    base = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=("axis",),
        param_coord="time_s",
    )
    pos = Position(base)
    ao_out = base.param.at([0.5], opts=ParamEvalOptions(method="linear"))
    pos_out = pos.param.at([0.5], opts=ParamEvalOptions(method="linear"))
    np.testing.assert_allclose(ao_out.unsafe_data["position"].values, pos_out.unsafe_data["position"].values)
    assert isinstance(pos_out, Position)
