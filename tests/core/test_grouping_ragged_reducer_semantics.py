from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.group_ops import GroupMaterializeOptions


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


def test_group_hard_p9d_001_grouped_reducers_preserve_ragged_tail_exclusion_semantics() -> None:
    """ID: GROUP_HARD_P9D_001_grouped_reducers_preserve_ragged_tail_exclusion_semantics."""
    grouped = _grouping_ao().group.groupby("label")
    count_out = grouped.count(validate=True).unsafe_data["signal"]
    mean_out = grouped.mean(skipna=False, validate=True).unsafe_data["signal"]
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
    out = _all_invalid_group_ao().group.groupby("label").mean(validate=True).unsafe_data
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
    default_out = grouped.count(validate=True).unsafe_data
    include_out = grouped.count(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=True),
        validate=True,
    ).unsafe_data
    assert "very_high" not in set(default_out.coords["group_key"].to_numpy().tolist())
    assert "very_high" in set(include_out.coords["group_key"].to_numpy().tolist())
    assert float(include_out["signal"].sel(group_key="very_high").to_numpy()) == 0.0
