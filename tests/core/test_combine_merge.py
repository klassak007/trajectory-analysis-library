import importlib
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, MergeOptions, ParamPrealignOptions, ParamSyncOptions
from tal.core.schema_errors import SchemaError

merge_mod = importlib.import_module("tal.core.combine_ops.merge")


def _ao_grouped(
    *,
    trial_labels: tuple[str, ...],
    tau_rows: list[list[float]],
    values: list[list[float]],
    var_name: str,
    size_name: str = "group_size",
) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={var_name: (("trial", "sample"), np.asarray(values, dtype="float64"))},
        coords={
            "trial": list(trial_labels),
            "sample": [0, 1, 2],
            "tau": (("trial", "sample"), np.asarray(tau_rows, dtype="float64")),
            size_name: ("trial", [3 for _ in trial_labels]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord=size_name,
    )


def _ao_static(name: str, value: float) -> AnalysisObject:
    return AnalysisObject(xr.Dataset(data_vars={name: xr.DataArray(value)}))


def test_combine_merge_001_merge_exact_sequence_and_inner_batch_default() -> None:
    """ID: COMBINE_MERGE_001_merge_exact_sequence_and_inner_batch_default."""
    right = _ao_grouped(
        trial_labels=("b", "c"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        var_name="right_value",
        size_name="n_valid",
    )
    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        var_name="left_value",
        size_name="n_valid",
    )
    out = left.combine.merge([right])
    assert list(out.data.coords["trial"].values) == ["b"]
    assert set(out.data.data_vars) == {"left_value", "right_value"}
    assert out.data.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "n_valid"


def test_combine_merge_002_merge_outer_fill_value_mapping() -> None:
    """ID: COMBINE_MERGE_002_merge_outer_fill_value_mapping."""
    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        var_name="left_value",
    )
    right = _ao_grouped(
        trial_labels=("b", "c"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        var_name="right_value",
    )
    out = left.combine.merge(
        [right],
        opts=MergeOptions(
            batch_join="outer",
            sequence_join="exact",
            compat="override",
            outer_fill_value={"left_value": -1.0, "right_value": -2.0},
        ),
    )
    np.testing.assert_allclose(out.data["left_value"].sel(trial="c").values, [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(out.data["right_value"].sel(trial="a").values, [-2.0, -2.0, -2.0])


def test_combine_merge_003_merge_with_param_prealign_delegates_to_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: COMBINE_MERGE_003_merge_with_param_prealign_delegates_to_sync."""
    left = _ao_grouped(
        trial_labels=("a",),
        tau_rows=[[0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0]],
        var_name="left_value",
    )
    right = _ao_grouped(
        trial_labels=("a",),
        tau_rows=[[0.0, 1.0, 2.0]],
        values=[[10.0, 11.0, 12.0]],
        var_name="right_value",
    )
    calls: list[int] = []

    def _count_sync(aos, *, on=None, grid=None, opts=None):  # type: ignore[no-untyped-def]
        calls.append(len(aos))
        return list(aos)

    monkeypatch.setattr(merge_mod, "synchronize_param", _count_sync)
    out = left.combine.merge(
        [right],
        opts=MergeOptions(
            param_prealign=ParamPrealignOptions(sync_opts=ParamSyncOptions(join="inner", how="nearest"))
        ),
    )
    assert calls == [2]
    assert set(out.data.data_vars) == {"left_value", "right_value"}


def test_combine_merge_004_schema_optional_blocks_pruned_when_untruthful() -> None:
    """ID: COMBINE_MERGE_004_schema_optional_blocks_pruned_when_untruthful."""
    dynamic = _ao_grouped(
        trial_labels=("a",),
        tau_rows=[[0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0]],
        var_name="value",
    )
    static = _ao_static("offset", 5.0)
    out = dynamic.combine.merge([static], opts=MergeOptions(batch_join="inner", sequence_join="exact", compat="override"))
    core = out.data.attrs["tal"]["core"]
    assert "param_coord" not in core
    assert "validity" not in core


def test_combine_merge_005_static_scalar_merge_without_sequence_dim() -> None:
    """ID: COMBINE_MERGE_005_static_scalar_merge_without_sequence_dim."""
    left = _ao_static("a", 1.0)
    right = _ao_static("b", 2.0)
    out = left.combine.merge([right])
    assert set(out.data.data_vars) == {"a", "b"}
    assert out.data.attrs["tal"]["core"]["roles"] == {"batch_dims": [], "core_dims": []}


def test_orch_finalize_parity_004_combine_merge_finalize_path_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ORCH_FINALIZE_PARITY_004_combine_merge_finalize_path_stable."""
    combine_finalize_mod = importlib.import_module("tal.core.combine_ops.finalize")
    calls = {"finalize_with_schema": 0}
    orig_finalize_with_schema = combine_finalize_mod.finalize_with_schema

    def _count_finalize_with_schema(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["finalize_with_schema"] += 1
        return orig_finalize_with_schema(*args, **kwargs)

    monkeypatch.setattr(combine_finalize_mod, "finalize_with_schema", _count_finalize_with_schema)

    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        var_name="left_value",
    )
    right = _ao_grouped(
        trial_labels=("b", "c"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        var_name="right_value",
    )
    out = left.combine.merge([right], opts=MergeOptions(batch_join="inner", sequence_join="exact"))
    assert set(out.data.data_vars) == {"left_value", "right_value"}
    assert calls["finalize_with_schema"] >= 1


def test_combine_merge_006_outer_fill_ignored_for_non_outer_join() -> None:
    """ID: COMBINE_MERGE_006_outer_fill_ignored_for_non_outer_join."""
    ds_left = xr.Dataset(
        data_vars={"a": (("sample",), [1.0, np.nan, 3.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ds_right = xr.Dataset(
        data_vars={"b": (("sample",), [10.0, 11.0, 12.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    left = AnalysisObject.from_data(ds_left, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    right = AnalysisObject.from_data(ds_right, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    out = left.combine.merge(
        [right],
        opts=MergeOptions(batch_join="inner", sequence_join="exact", outer_fill_value={"a": -99.0}),
    )
    np.testing.assert_allclose(out.data["a"].values, [1.0, np.nan, 3.0], equal_nan=True)


def test_combine_merge_007_outer_fill_applies_only_outer_introduced_cells() -> None:
    """ID: COMBINE_MERGE_007_outer_fill_applies_only_outer_introduced_cells."""
    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, np.nan, 2.0], [10.0, 11.0, 12.0]],
        var_name="left_value",
    )
    right = _ao_grouped(
        trial_labels=("b", "c"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        var_name="right_value",
    )
    out = left.combine.merge(
        [right],
        opts=MergeOptions(batch_join="outer", sequence_join="exact", outer_fill_value={"left_value": -1.0}),
    )
    np.testing.assert_allclose(out.data["left_value"].sel(trial="c").values, [-1.0, -1.0, -1.0])
    assert np.isnan(out.data["left_value"].sel(trial="a", sample=1).item())


def test_combine_merge_008_core_dims_deterministic_independent_of_input_order() -> None:
    """ID: COMBINE_MERGE_008_core_dims_deterministic_independent_of_input_order."""
    ds = xr.Dataset(
        data_vars={"value": (("sample", "axis"), np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float64"))},
        coords={"sample": [0, 1], "axis": ["x", "y"], "tau": ("sample", [0.0, 1.0])},
    )
    with_core = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
        param_coord="tau",
    )
    without_core = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out_a = with_core.combine.merge([without_core], validate=False)
    out_b = without_core.combine.merge([with_core], validate=False)
    assert out_a.data.attrs["tal"]["core"]["roles"]["core_dims"] == ["axis"]
    assert out_b.data.attrs["tal"]["core"]["roles"]["core_dims"] == ["axis"]


def test_combine_merge_009_malformed_roles_fail_closed_schema_error() -> None:
    """ID: COMBINE_MERGE_009_malformed_roles_fail_closed_schema_error."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0])},
        coords={"sample": [0, 1], "tau": ("sample", [0.0, 1.0])},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    malformed = ao.merge_schema(
        {"core": {"roles": {"sequence_dim": "sample", "batch_dims": "trial", "core_dims": []}}},
        validate=False,
    )
    with pytest.raises(SchemaError) as err:
        malformed.combine.merge([ao], validate=False)
    assert err.value.code == "schema.roles.batch_dims.invalid"


def test_combine_merge_010_outer_fill_preserves_invariant_payload() -> None:
    """ID: COMBINE_MERGE_010_outer_fill_preserves_invariant_payload."""
    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        var_name="left_value",
    )
    right = _ao_static("bias", 5.0)
    out = left.combine.merge(
        [right],
        opts=MergeOptions(
            batch_join="outer",
            sequence_join="exact",
            compat="override",
            outer_fill_value={"bias": -99.0},
        ),
    )
    np.testing.assert_allclose(out.data["bias"].values, [5.0, 5.0])


def test_combine_merge_011_override_preserves_right_only_shared_coords() -> None:
    """ID: COMBINE_MERGE_011_override_preserves_right_only_shared_coords."""
    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        var_name="left_value",
    )
    right = _ao_grouped(
        trial_labels=("b", "c"),
        tau_rows=[[10.0, 11.0, 12.0], [10.0, 11.0, 12.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        var_name="right_value",
    )
    out = left.combine.merge(
        [right],
        opts=MergeOptions(batch_join="outer", sequence_join="exact", compat="override"),
    )
    np.testing.assert_allclose(out.data.coords["tau"].sel(trial="c").values, [10.0, 11.0, 12.0])
    assert int(out.data.coords["group_size"].sel(trial="c").item()) == 3


def test_combine_merge_012_override_preserves_right_only_same_name_vars() -> None:
    """ID: COMBINE_MERGE_012_override_preserves_right_only_same_name_vars."""
    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        var_name="value",
    )
    right = _ao_grouped(
        trial_labels=("b", "c"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        var_name="value",
    )
    out = left.combine.merge(
        [right],
        opts=MergeOptions(batch_join="outer", sequence_join="exact", compat="override"),
    )
    np.testing.assert_allclose(out.data["value"].sel(trial="c").values, [200.0, 201.0, 202.0])
    np.testing.assert_allclose(out.data["value"].sel(trial="a").values, [0.0, 1.0, 2.0])


def test_combine_merge_013_invalid_sequence_size_prunes_validity_metadata() -> None:
    """ID: COMBINE_MERGE_013_invalid_sequence_size_prunes_validity_metadata."""
    left = _ao_grouped(
        trial_labels=("a",),
        tau_rows=[[0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0]],
        var_name="left_value",
    )
    right_ds = xr.Dataset(
        data_vars={"right_value": (("trial", "sample"), [[10.0, 11.0, 12.0]])},
        coords={
            "trial": ["b"],
            "sample": [0, 1, 2],
            "tau": (("trial", "sample"), [[0.0, 1.0, 2.0]]),
            "group_size": ("trial", [np.nan]),
        },
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
        validate=False,
    )
    with pytest.raises(SchemaError) as err:
        left.combine.merge(
            [right],
            opts=MergeOptions(batch_join="outer", sequence_join="exact", compat="override"),
            validate=False,
        )
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"


def test_combine_merge_014_no_conflicts_chunked_overlap_fails_fast() -> None:
    """ID: COMBINE_MERGE_014_no_conflicts_chunked_overlap_fails_fast."""
    da = pytest.importorskip("dask.array")
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2))},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([10.0, 11.0, 12.0]), chunks=2))},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    with pytest.raises(ValueError) as err:
        left.combine.merge([right], opts=MergeOptions(compat="no_conflicts"))
    assert "does not support overlapping chunked variables/coords" in str(err.value)


def test_combine_dry_003_sequence_size_derivation_single_owner_parity() -> None:
    """ID: COMBINE_DRY_003_sequence_size_derivation_single_owner_parity."""
    combine_finalize_text = Path("tal/core/combine_ops/finalize.py").read_text(encoding="utf-8")
    param_finalize_text = Path("tal/core/param_ops/finalize.py").read_text(encoding="utf-8")
    validity_finalize_text = Path("tal/core/validity_finalize.py").read_text(encoding="utf-8")
    assert "def _sequence_size_from_valid(" not in combine_finalize_text
    assert "def assign_sequence_size_from_valid_mask(" not in combine_finalize_text
    assert "def assign_sequence_size_from_valid_mask(" not in param_finalize_text
    assert "def assign_sequence_size_from_valid_mask(" in validity_finalize_text
