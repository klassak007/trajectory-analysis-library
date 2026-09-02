from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import tal.core.group_ops.foundation as foundation_module
import tal.core.group_ops.runtime_plan as runtime_plan_module
from tal.core.analysis_object import AnalysisObject
from tal.core.group_ops import GroupByOptions, GroupMaterializeOptions, GroupingFoundationOptions
from tal.core.schema_read import read_sequence_size_coord_name


def _grouping_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (("trial", "sample"), np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=float)),
            "flag": (("trial", "sample"), np.array([[True, False, True], [True, False, False]], dtype=bool)),
        },
        coords={
            "trial": np.array(["t0", "t1"], dtype=object),
            "sample": np.array([0, 1, 2], dtype=int),
            "time_s": (("trial", "sample"), np.array([[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=float)),
            "label": (("trial", "sample"), np.array([["A", "C", "A"], ["A", "B", "B"]], dtype=object)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
    )


def _all_invalid_group_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"signal": (("sample",), np.array([1.0, np.nan, 3.0], dtype=float))},
        coords={
            "sample": np.array([0, 1, 2], dtype=int),
            "label": (("sample",), np.array(["A", "N", "A"], dtype=object)),
        },
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), validate=True)


def _structurally_padded_grouping_ao(*, invalid_label: object = "A") -> AnalysisObject:
    labels = np.full((2, 3), "A", dtype=object)
    labels[0, 2] = invalid_label
    ds = xr.Dataset(
        data_vars={
            "signal": (
                ("trial", "sample"),
                np.array([[1.0, 2.0, 999.0], [10.0, 20.0, 30.0]], dtype=float),
            ),
        },
        coords={
            "trial": np.array(["t0", "t1"], dtype=object),
            "sample": np.array([0, 1, 2], dtype=int),
            "label": (("trial", "sample"), labels),
            "group_size": (("trial",), np.array([2, 3], dtype=np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="group_size",
        validate=True,
    )


def test_group_perf_p9d_001_groupby_defers_dense_structural_mask_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: GROUP_PERF_P9D_001_groupby_defers_dense_structural_mask_to_runtime."""
    runtime_owner = runtime_plan_module.resolve_validated_structural_mask_base
    runtime_calls = 0

    def reject_foundation_mask(*_args: object, **_kwargs: object) -> xr.DataArray:
        raise AssertionError("groupby construction must not build a dense structural mask without active NA policy work")

    def counted_runtime_mask(
        ds: xr.Dataset,
        *,
        sequence_dim: str | None,
        sequence_size_coord: str | None,
    ) -> xr.DataArray | None:
        nonlocal runtime_calls
        runtime_calls += 1
        return runtime_owner(
            ds,
            sequence_dim=sequence_dim,
            sequence_size_coord=sequence_size_coord,
        )

    monkeypatch.setattr(foundation_module, "resolve_validated_structural_mask_base", reject_foundation_mask)
    monkeypatch.setattr(runtime_plan_module, "resolve_validated_structural_mask_base", counted_runtime_mask)
    grouped = _structurally_padded_grouping_ao().group.groupby("label")
    assert runtime_calls == 0
    _ = grouped._resolve_plan(owner="grouping.performance")
    assert runtime_calls == 1


def test_group_hard_p9d_001_grouped_reducers_preserve_ragged_tail_exclusion_semantics() -> None:
    """ID: GROUP_HARD_P9D_001_grouped_reducers_preserve_ragged_tail_exclusion_semantics."""
    grouped = _grouping_ao().group.groupby("label")
    count_out = grouped.count(validate=True).as_dataset(copy="none")["signal"]
    mean_out = grouped.mean(skipna=False, validate=True).as_dataset(copy="none")["signal"]
    np.testing.assert_array_equal(
        count_out.sel(group_key=np.array(["A", "C", "B"], dtype=object)).to_numpy(),
        np.array([3, 1, 2], dtype=np.int64),
    )
    np.testing.assert_allclose(
        mean_out.sel(group_key=np.array(["A", "C", "B"], dtype=object)).to_numpy(),
        np.array([14.0, 11.0, 21.5], dtype=float),
    )


def test_group_hard_p9d_002_all_invalid_present_groups_remain_present_with_invalid_reducer_outcomes() -> None:
    """ID: GROUP_HARD_P9D_002_all_invalid_present_groups_remain_present_with_invalid_reducer_outcomes."""
    out = _all_invalid_group_ao().group.groupby("label").mean(validate=True).as_dataset(copy="none")
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array(["A", "N"], dtype=object))
    np.testing.assert_allclose(out["signal"].sel(group_key="A").to_numpy(), np.array(2.0, dtype=float))
    assert np.isnan(out["signal"].sel(group_key="N").to_numpy()).all()


def test_group_hard_p9d_003_no_member_groups_absent_by_default_and_present_when_explicitly_included() -> None:
    """ID: GROUP_HARD_P9D_003_no_member_groups_absent_by_default_and_present_when_explicitly_included."""
    grouped = _grouping_ao().group.groupby_bins(
        "time_s",
        bins=np.array([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=float),
        labels=["low", "mid", "high", "very_high"],
    )
    default_out = grouped.count(validate=True).as_dataset(copy="none")
    include_out = grouped.count(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=True),
        validate=True,
    ).as_dataset(copy="none")
    assert "very_high" not in set(default_out.coords["group_key"].to_numpy().tolist())
    assert "very_high" in set(include_out.coords["group_key"].to_numpy().tolist())
    assert float(include_out["signal"].sel(group_key="very_high").to_numpy()) == 0.0


def test_group_hard_p9d_004_global_grouping_excludes_structural_padding_from_layouts_and_reducers() -> None:
    """ID: GROUP_HARD_P9D_004_global_grouping_excludes_structural_padding_from_layouts_and_reducers."""
    grouped = _structurally_padded_grouping_ao().group.groupby("label")
    padded = grouped.padded(validate=True).as_dataset(copy="none")
    stacked = grouped.stacked(validate=True).as_dataset(copy="none")
    size_name = read_sequence_size_coord_name(padded)
    assert size_name is not None
    np.testing.assert_array_equal(padded.coords[size_name].to_numpy(), np.array([5], dtype=np.int64))
    expected = np.array([1.0, 2.0, 10.0, 20.0, 30.0], dtype=float)
    np.testing.assert_allclose(padded["signal"].sel(group_key="A").to_numpy(), expected)
    np.testing.assert_allclose(stacked["signal"].to_numpy(), expected)
    assert int(grouped.count(validate=True).as_dataset(copy="none")["signal"].sel(group_key="A")) == 5
    assert float(grouped.sum(validate=True).as_dataset(copy="none")["signal"].sel(group_key="A")) == 63.0
    assert float(grouped.mean(validate=True).as_dataset(copy="none")["signal"].sel(group_key="A")) == 12.6


def test_group_hard_p9d_005_preserve_batch_grouping_excludes_structural_padding() -> None:
    """ID: GROUP_HARD_P9D_005_preserve_batch_grouping_excludes_structural_padding."""
    grouped = _structurally_padded_grouping_ao().group.groupby(
        "label",
        opts=GroupByOptions(preserve_batch=True),
    )
    padded = grouped.padded(validate=True).as_dataset(copy="none")["signal"].sel(group_key="A")
    stacked = grouped.stacked(validate=True).as_dataset(copy="none")["signal"]
    expected_rows = np.array([[1.0, 2.0, np.nan], [10.0, 20.0, 30.0]], dtype=float)
    np.testing.assert_allclose(padded.to_numpy(), expected_rows, equal_nan=True)
    np.testing.assert_allclose(stacked.to_numpy(), expected_rows, equal_nan=True)
    count = grouped.count(validate=True).as_dataset(copy="none")["signal"].sel(group_key="A")
    summed = grouped.sum(validate=True).as_dataset(copy="none")["signal"].sel(group_key="A")
    mean = grouped.mean(validate=True).as_dataset(copy="none")["signal"].sel(group_key="A")
    np.testing.assert_array_equal(count.to_numpy(), np.array([2, 3], dtype=np.int64))
    np.testing.assert_allclose(summed.to_numpy(), np.array([3.0, 60.0], dtype=float))
    np.testing.assert_allclose(mean.to_numpy(), np.array([1.5, 20.0], dtype=float))


@pytest.mark.parametrize("policy", ["error", "drop", "group"])
@pytest.mark.parametrize("invalid_label", [pytest.param("PAD", id="non_na"), pytest.param(np.nan, id="na")])
def test_group_hard_p9d_006_all_na_policies_ignore_structural_padding(
    policy: str,
    invalid_label: object,
) -> None:
    """ID: GROUP_HARD_P9D_006_all_na_policies_ignore_structural_padding."""
    na_group_label = "NA" if policy == "group" else None
    grouped = _structurally_padded_grouping_ao(invalid_label=invalid_label).group.groupby(
        "label",
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(
                na_key_policy=policy,
                na_group_label=na_group_label,
            )
        ),
    )
    out = grouped.count(validate=True).as_dataset(copy="none")
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array(["A"], dtype=object))
    assert int(out["signal"].sel(group_key="A")) == 5


def test_group_hard_p9d_007_group_na_collision_check_ignores_structural_padding() -> None:
    """ID: GROUP_HARD_P9D_007_group_na_collision_check_ignores_structural_padding."""
    ao = _structurally_padded_grouping_ao(invalid_label="NA")
    labels = ao.as_dataset(copy="none").coords["label"].copy(data=np.array([["A", np.nan, "NA"], ["A", "A", "A"]], dtype=object))
    regrouped = AnalysisObject.from_data(
        ao.as_dataset(copy="none").assign_coords(label=labels),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="group_size",
        validate=True,
    )
    out = regrouped.group.groupby(
        "label",
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(
                na_key_policy="group",
                na_group_label="NA",
            )
        ),
    ).count(validate=True).as_dataset(copy="none")
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array(["A", "NA"], dtype=object))
    np.testing.assert_array_equal(out["signal"].to_numpy(), np.array([4, 1], dtype=np.int64))
