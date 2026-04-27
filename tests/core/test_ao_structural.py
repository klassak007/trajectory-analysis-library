import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, SchemaError, merge_schema


def _ao_structured() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample", "axis"), np.arange(12).reshape(2, 3, 2))},
        coords={
            "trial": [0, 1],
            "sample": [0, 1, 2],
            "axis": ["x", "y"],
            "phase": (("trial", "sample"), np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])),
            "group_size": ("trial", [3, 2]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="phase",
        sequence_size_coord="group_size",
    )


def _ao_with_scalar_validity() -> AnalysisObject:
    ds = _ao_structured().data.isel(trial=0, drop=True)
    ds = ds.assign_coords(group_size=xr.DataArray(np.asarray(3), dims=()))
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
        param_coord="phase",
        sequence_size_coord="group_size",
    )


def _ao_with_stale_optional_blocks() -> AnalysisObject:
    stale = merge_schema(
        _ao_structured().data,
        {
            "core": {
                "param_coord": {"name": "missing_param"},
                "validity": {"sequence_size_coord": "missing_size", "layout": "left_packed"},
            }
        },
        validate=False,
    )
    return AnalysisObject._from_unvalidated(stale)


def test_ao_struct_001_isel_preserves_bound_schema() -> None:
    """ID: AO_STRUCT_001_isel_preserves_bound_schema."""
    ao = _ao_structured()
    out = ao.isel(sample=slice(0, 2))
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }
    assert core["param_coord"]["name"] == "phase"
    assert core["validity"]["sequence_size_coord"] == "group_size"


def test_ao_struct_002_isel_drop_batch_prunes_batch_dims() -> None:
    """ID: AO_STRUCT_002_isel_drop_batch_prunes_batch_dims."""
    ao = _ao_structured()
    out = ao.isel(trial=0, drop=True)
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": [],
        "core_dims": ["axis"],
    }
    assert core["param_coord"]["name"] == "phase"
    assert "validity" not in core


def test_ao_struct_003_isel_drop_sequence_preserves_core_batch_roles_and_prunes_sequence_blocks() -> None:
    """ID: AO_STRUCT_003_isel_drop_sequence_preserves_core_batch_roles_and_prunes_sequence_blocks."""
    ao = _ao_structured()
    out = ao.isel(sample=0, drop=True)
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }
    assert "param_coord" not in core
    assert "validity" not in core


def test_ao_struct_004_rename_updates_roles_param_validity() -> None:
    """ID: AO_STRUCT_004_rename_updates_roles_param_validity."""
    ao = _ao_structured()
    out = ao.rename({"sample": "step", "trial": "run", "phase": "tau", "group_size": "n"})
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "step",
        "batch_dims": ["run"],
        "core_dims": ["axis"],
    }
    assert core["param_coord"]["name"] == "tau"
    assert core["validity"]["sequence_size_coord"] == "n"


def test_ao_struct_005_drop_vars_clears_optional_blocks() -> None:
    """ID: AO_STRUCT_005_drop_vars_clears_optional_blocks."""
    ao = _ao_structured()
    out = ao.drop_vars(["phase", "group_size"])
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }
    assert "param_coord" not in core
    assert "validity" not in core


def test_ao_struct_006_transpose_preserves_schema() -> None:
    """ID: AO_STRUCT_006_transpose_preserves_schema."""
    ao = _ao_structured()
    out = ao.transpose("axis", "trial", "sample")
    roles = out.data.attrs["tal"]["core"]["roles"]
    assert roles == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }
    assert tuple(out.data["value"].dims) == ("axis", "trial", "sample")


def test_ao_struct_007_sel_preserves_bound_schema() -> None:
    """ID: AO_STRUCT_007_sel_preserves_bound_schema."""
    ao = _ao_structured()
    out = ao.sel(sample=slice(0, 1))
    roles = out.data.attrs["tal"]["core"]["roles"]
    assert roles == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }


def test_ao_struct_008_where_preserves_bound_schema() -> None:
    """ID: AO_STRUCT_008_where_preserves_bound_schema."""
    ao = _ao_structured()
    cond = ao.data["value"] >= 0
    out = ao.where(cond, drop=False)
    roles = out.data.attrs["tal"]["core"]["roles"]
    assert roles == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }


def test_ao_struct_009_invalid_roles_validate_true_raises() -> None:
    """ID: AO_STRUCT_009_invalid_roles_validate_true_raises."""
    ao = _ao_structured()
    invalid = merge_schema(ao.data, {"core": {"roles": "bad"}}, validate=False)
    bad_ao = AnalysisObject._from_unvalidated(invalid)
    with pytest.raises(SchemaError) as err:
        bad_ao.isel(sample=slice(0, 2), validate=True)
    assert err.value.code == "schema.roles.not_mapping"
    assert err.value.path == "tal.core.roles"


def test_ao_struct_010_invalid_roles_validate_false_is_permissive() -> None:
    """ID: AO_STRUCT_010_invalid_roles_validate_false_is_permissive."""
    ao = _ao_structured()
    invalid = merge_schema(ao.data, {"core": {"roles": "bad"}}, validate=False)
    bad_ao = AnalysisObject._from_unvalidated(invalid)
    out = bad_ao.isel(sample=slice(0, 2), validate=False)
    assert out.data.attrs["tal"]["core"]["roles"] == "bad"


def test_ao_struct_011_sequence_drop_validate_true_rejects_unknown_tal_key() -> None:
    """ID: AO_STRUCT_011_sequence_drop_validate_true_rejects_unknown_tal_key."""
    ao = _ao_structured()
    invalid = merge_schema(ao.data, {"unknown": 123}, validate=False)
    bad_ao = AnalysisObject._from_unvalidated(invalid)
    with pytest.raises(SchemaError) as err:
        bad_ao.isel(sample=0, drop=True, validate=True)
    assert err.value.code == "schema.unknown_key"
    assert err.value.path == "tal.unknown"


def test_ao_struct_param_012_rename_param_coord_dim_break_prunes_param_coord() -> None:
    """ID: AO_STRUCT_PARAM_012_rename_param_coord_dim_break_prunes_param_coord."""
    ao = _ao_structured()
    ds = ao.data.assign_coords(phase_bad=ao.data.coords["phase"].transpose("sample", "trial"))
    stale = merge_schema(ds, {"core": {"param_coord": {"name": "phase_bad"}}}, validate=False)
    bad_ao = AnalysisObject._from_unvalidated(stale)
    out = bad_ao.rename({"axis": "axis2"})
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis2"],
    }
    assert "param_coord" not in core
    assert core["validity"]["sequence_size_coord"] == "group_size"


def test_ao_struct_param_013_transpose_param_coord_noncanonical_order_prunes_param_coord() -> None:
    """ID: AO_STRUCT_PARAM_013_transpose_param_coord_noncanonical_order_prunes_param_coord."""
    ao = _ao_structured()
    out = ao.transpose("sample", "trial", "axis")
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }
    assert "param_coord" not in core
    assert core["validity"]["sequence_size_coord"] == "group_size"


def test_ao_struct_param_014_drop_vars_sequence_size_prunes_validity() -> None:
    """ID: AO_STRUCT_PARAM_014_drop_vars_sequence_size_prunes_validity."""
    ao = _ao_structured()
    out = ao.drop_vars("group_size")
    core = out.data.attrs["tal"]["core"]
    assert core["param_coord"]["name"] == "phase"
    assert "validity" not in core


def test_ao_struct_param_015_where_drop_true_prunes_stale_optional_blocks() -> None:
    """ID: AO_STRUCT_PARAM_015_where_drop_true_prunes_stale_optional_blocks."""
    bad_ao = _ao_with_stale_optional_blocks()
    out = bad_ao.where(bad_ao.data["value"] >= 0, drop=True)
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }
    assert "param_coord" not in core
    assert "validity" not in core


def test_ao_struct_param_016_validate_false_still_prunes_stale_optional_blocks_when_roles_resolved() -> None:
    """ID: AO_STRUCT_PARAM_016_validate_false_still_prunes_stale_optional_blocks_when_roles_resolved."""
    bad_ao = _ao_with_stale_optional_blocks()
    out = bad_ao.isel(sample=slice(0, 2), validate=False)
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }
    assert "param_coord" not in core
    assert "validity" not in core


def test_ao_struct_param_017_batch_squeeze_scalar_validity_preserve_or_prune_is_deterministic() -> None:
    """ID: AO_STRUCT_PARAM_017_batch_squeeze_scalar_validity_preserve_or_prune_is_deterministic."""
    grouped = _ao_structured()
    squeezed = grouped.isel(trial=0, drop=True)
    squeezed_core = squeezed.data.attrs["tal"]["core"]
    assert squeezed_core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": [],
        "core_dims": ["axis"],
    }
    assert "validity" not in squeezed_core

    unbatched = _ao_with_scalar_validity()
    out = unbatched.isel(sample=slice(0, 2))
    core = out.data.attrs["tal"]["core"]
    assert core["roles"] == {
        "sequence_dim": "sample",
        "batch_dims": [],
        "core_dims": ["axis"],
    }
    assert core["validity"] == {"sequence_size_coord": "group_size", "layout": "left_packed"}


def test_ao_struct_018_isel_prefix_truncation_clamps_validity_sizes() -> None:
    """ID: AO_STRUCT_018_isel_prefix_truncation_clamps_validity_sizes."""
    ao = _ao_structured()
    out = ao.isel(sample=slice(0, 2))
    core = out.data.attrs["tal"]["core"]
    assert core["validity"] == {"sequence_size_coord": "group_size", "layout": "left_packed"}
    np.testing.assert_array_equal(out.data.coords["group_size"].values, np.asarray([2, 2], dtype=np.int64))


def test_ao_struct_019_isel_nonprefix_selection_drops_validity() -> None:
    """ID: AO_STRUCT_019_isel_nonprefix_selection_drops_validity."""
    ao = _ao_structured()
    out = ao.isel(sample=[0, 2])
    core = out.data.attrs["tal"]["core"]
    assert "validity" not in core


def test_ao_struct_020_where_drop_true_nonprefix_drops_validity() -> None:
    """ID: AO_STRUCT_020_where_drop_true_nonprefix_drops_validity."""
    ao = _ao_structured()
    cond = xr.DataArray(np.asarray([True, False, True]), dims=("sample",), coords={"sample": [0, 1, 2]})
    out = ao.where(cond, drop=True)
    core = out.data.attrs["tal"]["core"]
    assert "validity" not in core


def test_ao_struct_021_scalar_validity_prefix_truncation_clamps_size() -> None:
    """ID: AO_STRUCT_021_scalar_validity_prefix_truncation_clamps_size."""
    ao = _ao_with_scalar_validity()
    out = ao.isel(sample=slice(0, 2))
    core = out.data.attrs["tal"]["core"]
    assert core["validity"] == {"sequence_size_coord": "group_size", "layout": "left_packed"}
    assert int(out.data.coords["group_size"].item()) == 2
