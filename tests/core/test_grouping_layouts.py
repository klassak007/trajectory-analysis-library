from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.group_ops import GroupByOptions, GroupMaterializeOptions, GroupingFoundationOptions
from tal.core.schema_read import read_roles, read_sequence_size_coord_name


def _grouping_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (("trial", "sample"), np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=float)),
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


def _grouping_bins_with_na_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (("sample",), np.array([10.0, 11.0, 12.0], dtype=float)),
        },
        coords={
            "sample": np.array([0, 1, 2], dtype=int),
            "time_s": (("sample",), np.array([0.2, np.nan, 1.2], dtype=float)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time_s",
    )


def test_group_core_p9b_002_padded_layout_preserves_ragged_schema_truthfulness() -> None:
    """ID: GROUP_CORE_P9B_002_padded_layout_preserves_ragged_schema_truthfulness."""
    ao = _grouping_ao()
    out = ao.group.groupby("label").padded().unsafe_data
    declared, sequence_dim, batch_dims, core_dims = read_roles(out)
    assert declared
    assert sequence_dim == "sample"
    assert batch_dims == ("group_key",)
    assert core_dims == ()
    assert list(out.coords["group_key"].to_numpy()) == ["A", "C", "B"]
    size_name = read_sequence_size_coord_name(out)
    assert size_name is not None
    assert tuple(out.coords[size_name].dims) == ("group_key",)
    np.testing.assert_array_equal(out.coords[size_name].to_numpy(), np.array([3, 1, 2], dtype=np.int64))
    signal = out["signal"].to_numpy()
    np.testing.assert_allclose(signal[0], np.array([10.0, 12.0, 20.0], dtype=float), equal_nan=True)
    np.testing.assert_allclose(signal[1], np.array([11.0, np.nan, np.nan], dtype=float), equal_nan=True)
    np.testing.assert_allclose(signal[2], np.array([21.0, 22.0, np.nan], dtype=float), equal_nan=True)


def test_group_core_p9b_003_stacked_layout_is_deterministic_and_schema_safe() -> None:
    """ID: GROUP_CORE_P9B_003_stacked_layout_is_deterministic_and_schema_safe."""
    ao = _grouping_ao()
    out = ao.group.groupby("label").stacked().unsafe_data
    declared, sequence_dim, batch_dims, core_dims = read_roles(out)
    assert declared
    assert sequence_dim == "group_member"
    assert batch_dims == ()
    assert core_dims == ()
    assert read_sequence_size_coord_name(out) is None
    assert tuple(out.coords["group_key"].dims) == ("group_member",)
    assert tuple(out.coords["sequence_index"].dims) == ("group_member",)
    assert tuple(out.coords["trial"].dims) == ("group_member",)
    np.testing.assert_array_equal(
        out.coords["group_key"].to_numpy(),
        np.array(["A", "A", "A", "C", "B", "B"], dtype=object),
    )
    np.testing.assert_array_equal(
        out.coords["sequence_index"].to_numpy(),
        np.array([0, 2, 0, 1, 1, 2], dtype=float),
    )
    np.testing.assert_array_equal(
        out.coords["trial"].to_numpy(),
        np.array(["t0", "t0", "t1", "t0", "t1", "t1"], dtype=object),
    )
    np.testing.assert_allclose(
        out["signal"].to_numpy(),
        np.array([10.0, 12.0, 20.0, 11.0, 21.0, 22.0], dtype=float),
    )


def test_group_core_p9b_005_group_and_member_ordering_are_stable_and_deterministic() -> None:
    """ID: GROUP_CORE_P9B_005_group_and_member_ordering_are_stable_and_deterministic."""
    ao = _grouping_ao()
    out = ao.group.groupby_bins("time_s", bins=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float)).padded().unsafe_data
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array([0.0, 1.0, 2.0], dtype=float))
    signal = out["signal"].to_numpy()
    np.testing.assert_allclose(signal[0], np.array([11.0, 21.0], dtype=float), equal_nan=True)
    np.testing.assert_allclose(signal[1], np.array([12.0, 22.0], dtype=float), equal_nan=True)
    np.testing.assert_allclose(signal[2], np.array([10.0, 20.0], dtype=float), equal_nan=True)


def test_group_core_p9b_009_padded_sequence_coord_uses_member_rank_not_source_labels() -> None:
    """ID: GROUP_CORE_P9B_009_padded_sequence_coord_uses_member_rank_not_source_labels."""
    source = _grouping_ao().unsafe_data
    ao = AnalysisObject.from_data(
        source.assign_coords({"sample": np.array([10, 20, 30], dtype=int)}),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
    )
    out = ao.group.groupby("label").padded().unsafe_data
    np.testing.assert_array_equal(out.coords["sample"].to_numpy(), np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_allclose(
        out["signal"].sel(group_key="A").to_numpy(),
        np.array([10.0, 12.0, 20.0], dtype=float),
        equal_nan=True,
    )


def test_group_core_p9b_010_groupby_bins_custom_labels_preserve_interval_order() -> None:
    """ID: GROUP_CORE_P9B_010_groupby_bins_custom_labels_preserve_interval_order."""
    ao = _grouping_ao()
    out = ao.group.groupby_bins(
        "time_s",
        bins=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float),
        labels=["low", "mid", "high"],
    ).padded().unsafe_data
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array(["low", "mid", "high"], dtype=object))


def test_group_core_p9b_011_include_empty_groups_controls_bin_universe_for_padded_layout() -> None:
    """ID: GROUP_CORE_P9B_011_include_empty_groups_controls_bin_universe_for_padded_layout."""
    ao = _grouping_ao()
    bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=float)
    labels = ["low", "mid", "high", "very_high"]
    grouped = ao.group.groupby_bins("time_s", bins=bins, labels=labels)
    keep_empty = grouped.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=True)
    ).unsafe_data
    drop_empty = grouped.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False)
    ).unsafe_data
    np.testing.assert_array_equal(
        keep_empty.coords["group_key"].to_numpy(),
        np.array(["low", "mid", "high", "very_high"], dtype=object),
    )
    assert np.isnan(keep_empty["signal"].sel(group_key="very_high").to_numpy()).all()
    np.testing.assert_array_equal(
        drop_empty.coords["group_key"].to_numpy(),
        np.array(["low", "mid", "high"], dtype=object),
    )
    grouped_preserve = ao.group.groupby_bins(
        "time_s",
        bins=bins,
        labels=labels,
        opts=GroupByOptions(preserve_batch=True),
    )
    keep_empty_batch = grouped_preserve.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=True),
        validate=True,
    ).unsafe_data
    np.testing.assert_array_equal(
        keep_empty_batch.coords["group_key"].to_numpy(),
        np.array(["low", "mid", "high", "very_high"], dtype=object),
    )
    assert np.isnan(keep_empty_batch["signal"].sel(group_key="very_high").to_numpy()).all()


def test_group_core_p9b_012_groupby_bins_group_na_retains_na_label_and_members() -> None:
    """ID: GROUP_CORE_P9B_012_groupby_bins_group_na_retains_na_label_and_members."""
    ao = _grouping_bins_with_na_ao()
    grouped = ao.group.groupby_bins(
        "time_s",
        bins=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        labels=["low", "mid", "high"],
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(
                na_key_policy="group",
                na_group_label="NA",
            )
        ),
    )
    out_drop = grouped.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False)
    ).unsafe_data
    out_keep = grouped.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=True)
    ).unsafe_data
    np.testing.assert_array_equal(
        out_drop.coords["group_key"].to_numpy(),
        np.array(["low", "mid", "NA"], dtype=object),
    )
    np.testing.assert_array_equal(
        out_keep.coords["group_key"].to_numpy(),
        np.array(["low", "mid", "high", "NA"], dtype=object),
    )
    np.testing.assert_allclose(
        out_drop["signal"].sel(group_key="NA").to_numpy(),
        np.array([11.0], dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out_keep["signal"].sel(group_key="NA").to_numpy(),
        np.array([11.0], dtype=float),
        equal_nan=True,
    )


def test_group_core_p9b_013_bin_domain_labels_precede_observed_non_domain_labels() -> None:
    """ID: GROUP_CORE_P9B_013_bin_domain_labels_precede_observed_non_domain_labels."""
    ao = _grouping_bins_with_na_ao()
    grouped = ao.group.groupby_bins(
        "time_s",
        bins=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(
                na_key_policy="group",
                na_group_label="NA",
            )
        ),
    )
    out = grouped.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False)
    ).unsafe_data
    np.testing.assert_array_equal(
        out.coords["group_key"].to_numpy(),
        np.array([0.0, 1.0, "NA"], dtype=object),
    )


def test_group_hard_p9b_004_include_empty_groups_false_excludes_globally_empty_bins() -> None:
    """ID: GROUP_HARD_P9B_004_include_empty_groups_false_excludes_globally_empty_bins."""
    ao = _grouping_ao()
    bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=float)
    labels = ["low", "mid", "high", "very_high"]
    out = ao.group.groupby_bins(
        "time_s",
        bins=bins,
        labels=labels,
        opts=GroupByOptions(preserve_batch=True),
    ).padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False)
    ).unsafe_data
    assert "very_high" not in set(out.coords["group_key"].to_numpy().tolist())


def test_group_hard_p9b_005_include_empty_groups_false_keeps_observed_na_group_label() -> None:
    """ID: GROUP_HARD_P9B_005_include_empty_groups_false_keeps_observed_na_group_label."""
    ao = _grouping_bins_with_na_ao()
    grouped = ao.group.groupby_bins(
        "time_s",
        bins=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        labels=["low", "mid", "high"],
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(
                na_key_policy="group",
                na_group_label="NA",
            )
        ),
    )
    out = grouped.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False)
    ).unsafe_data
    values = out.coords["group_key"].to_numpy().tolist()
    assert "high" not in values
    assert "NA" in values


def test_group_core_p9b_006_generated_grouping_names_are_stable_by_contract() -> None:
    """ID: GROUP_CORE_P9B_006_generated_grouping_names_are_stable_by_contract."""
    ao = _grouping_ao()
    out = ao.group.groupby("label").stacked().unsafe_data
    assert "group_key" in out.coords
    assert "group_member" in out.dims
    assert "sequence_index" in out.coords


def test_group_core_p9b_007_preserve_batch_padded_uses_global_group_key_union() -> None:
    """ID: GROUP_CORE_P9B_007_preserve_batch_padded_uses_global_group_key_union."""
    ao = _grouping_ao()
    out = ao.group.groupby("label", opts=GroupByOptions(preserve_batch=True)).padded().unsafe_data
    declared, sequence_dim, batch_dims, _ = read_roles(out)
    assert declared
    assert sequence_dim == "sample"
    assert batch_dims == ("trial", "group_key")
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array(["A", "C", "B"], dtype=object))
    size_name = read_sequence_size_coord_name(out)
    assert size_name is not None
    np.testing.assert_array_equal(
        out.coords[size_name].to_numpy(),
        np.array([[2, 1, 0], [1, 0, 2]], dtype=np.int64),
    )
    trial1_group_c = out["signal"].sel(trial="t1", group_key="C").to_numpy()
    assert np.isnan(trial1_group_c).all()


def test_group_core_p9b_008_preserve_batch_stacked_keeps_batch_dims_and_provenance() -> None:
    """ID: GROUP_CORE_P9B_008_preserve_batch_stacked_keeps_batch_dims_and_provenance."""
    ao = _grouping_ao()
    out = ao.group.groupby("label", opts=GroupByOptions(preserve_batch=True)).stacked().unsafe_data
    declared, sequence_dim, batch_dims, _ = read_roles(out)
    assert declared
    assert sequence_dim == "group_member"
    assert batch_dims == ("trial",)
    assert tuple(out.coords["group_key"].dims) == ("trial", "group_member")
    assert tuple(out.coords["sequence_index"].dims) == ("trial", "group_member")
    np.testing.assert_array_equal(
        out.coords["group_key"].sel(trial="t0").to_numpy(),
        np.array(["A", "A", "C"], dtype=object),
    )
    np.testing.assert_array_equal(
        out.coords["group_key"].sel(trial="t1").to_numpy(),
        np.array(["A", "B", "B"], dtype=object),
    )
    np.testing.assert_allclose(
        out["signal"].sel(trial="t0").to_numpy(),
        np.array([10.0, 12.0, 11.0], dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out["signal"].sel(trial="t1").to_numpy(),
        np.array([20.0, 21.0, 22.0], dtype=float),
        equal_nan=True,
    )


def test_group_hard_p9b_002_grouped_surface_does_not_invoke_hidden_interpolation_owners() -> None:
    """ID: GROUP_HARD_P9B_002_grouped_surface_does_not_invoke_hidden_interpolation_owners."""
    text = (Path("tal/core/group_ops/materialize.py")).read_text(encoding="utf-8")
    assert "build_param_map" not in text
    assert "apply_param_map" not in text
