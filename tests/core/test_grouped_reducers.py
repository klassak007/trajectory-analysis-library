from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.group_ops import GroupByOptions, GroupMaterializeOptions
from tal.spatial import Rotation


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


def _rotation_grouping() -> Rotation:
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q90 = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=float)
    q180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    values = np.array([[q0, q90, q0], [q0, q0, q180]], dtype=float)
    ds = xr.Dataset(
        data_vars={"rotation": (("trial", "sample", "quat"), values)},
        coords={
            "trial": np.array(["t0", "t1"], dtype=object),
            "sample": np.array([0, 1, 2], dtype=int),
            "quat": np.array(["x", "y", "z", "w"], dtype=object),
            "label": (("trial", "sample"), np.array([["A", "C", "A"], ["A", "B", "B"]], dtype=object)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("quat",),
        validate=True,
    )
    return Rotation(ao.as_dataset(copy="none"))


def _windowed_grouping_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (
                ("trial", "event", "tau"),
                np.asarray(
                    [
                        [[1.0, 2.0, 3.0], [1.2, 2.2, 3.2]],
                        [[1.5, 2.5, 3.5], [1.7, 2.7, 3.7]],
                        [[4.0, 5.0, 6.0], [4.2, 5.2, 6.2]],
                        [[4.5, 5.5, 6.5], [4.7, 5.7, 6.7]],
                    ],
                    dtype=float,
                ),
            )
        },
        coords={
            "trial": np.asarray(["t0", "t1", "t2", "t3"], dtype=object),
            "event": np.asarray([0, 1], dtype=np.int64),
            "tau": np.asarray([-1.0, 0.0, 1.0], dtype=float),
            "window_tau": ("tau", np.asarray([-1.0, 0.0, 1.0], dtype=float)),
            "outcome": ("trial", np.asarray(["intercept", "intercept", "miss", "miss"], dtype=object)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="tau",
        batch_dims=("trial", "event"),
        core_dims=(),
        param_coord="window_tau",
    )


@pytest.mark.parametrize("name", ["mean", "sum", "std", "var", "median", "min", "max", "count", "any", "all"])
def test_group_core_p9d_001_grouped_reducer_surface_methods_exist_and_are_invocable(name: str) -> None:
    """ID: GROUP_CORE_P9D_001_grouped_reducer_surface_methods_exist_and_are_invocable."""
    grouped = _grouping_ao().group.groupby("label")
    out = getattr(grouped, name)(validate=True)
    assert isinstance(out, AnalysisObject)


def test_group_core_p9d_002_grouped_dim_none_reduces_member_axis_and_preserves_group_axis() -> None:
    """ID: GROUP_CORE_P9D_002_grouped_dim_none_reduces_member_axis_and_preserves_group_axis."""
    out = _grouping_ao().group.groupby("label").mean(validate=True).as_dataset(copy="none")
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array(["A", "C", "B"], dtype=object))
    np.testing.assert_allclose(out["signal"].to_numpy(), np.array([14.0, 11.0, 21.5], dtype=float))
    assert out["signal"].dims == ("group_key",)


def test_group_core_p9d_003_grouped_explicit_dim_override_matches_canonical_padded_reference() -> None:
    """ID: GROUP_CORE_P9D_003_grouped_explicit_dim_override_matches_canonical_padded_reference."""
    grouped = _grouping_ao().group.groupby("label")
    out = grouped.sum(dim="group_key", validate=True).as_dataset(copy="none")
    reference = grouped.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False),
        validate=True,
    ).sum(dim="group_key", validate=True).as_dataset(copy="none")
    assert out.identical(reference)


def test_group_core_p9d_004_stacked_layout_requests_normalize_to_canonical_padded_compute() -> None:
    """ID: GROUP_CORE_P9D_004_stacked_layout_requests_normalize_to_canonical_padded_compute."""
    grouped = _grouping_ao().group.groupby("label")
    padded = grouped.mean(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False),
        validate=True,
    ).as_dataset(copy="none")
    stacked = grouped.mean(
        opts=GroupMaterializeOptions(layout="stacked", include_empty_groups=False),
        validate=True,
    ).as_dataset(copy="none")
    assert padded.identical(stacked)

    stacked_member = grouped.sum(
        dim="group_member",
        opts=GroupMaterializeOptions(layout="stacked", include_empty_groups=False),
        validate=True,
    ).as_dataset(copy="none")
    padded_member = grouped.sum(
        dim="sample",
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False),
        validate=True,
    ).as_dataset(copy="none")
    assert stacked_member.identical(padded_member)


def test_group_core_p9d_005_grouped_reducer_default_excludes_no_member_groups_and_override_includes() -> None:
    """ID: GROUP_CORE_P9D_005_grouped_reducer_default_excludes_no_member_groups_and_override_includes."""
    grouped = _grouping_ao().group.groupby_bins(
        "time_s",
        bins=np.array([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=float),
        labels=["low", "mid", "high", "very_high"],
    )
    default_out = grouped.mean(validate=True).as_dataset(copy="none")
    include_out = grouped.mean(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=True),
        validate=True,
    ).as_dataset(copy="none")
    np.testing.assert_array_equal(
        default_out.coords["group_key"].to_numpy(),
        np.array(["low", "mid", "high"], dtype=object),
    )
    np.testing.assert_array_equal(
        include_out.coords["group_key"].to_numpy(),
        np.array(["low", "mid", "high", "very_high"], dtype=object),
    )
    assert np.isnan(include_out["signal"].sel(group_key="very_high").to_numpy()).all()


def test_group_core_p9d_006_grouped_rotation_mean_preserves_typed_and_non_owned_ops_demote() -> None:
    """ID: GROUP_CORE_P9D_006_grouped_rotation_mean_preserves_typed_and_non_owned_ops_demote."""
    grouped = _rotation_grouping().group.groupby("label")
    mean_out = grouped.mean(validate=True)
    sum_out = grouped.sum(validate=True)
    any_out = grouped.any(validate=True)
    assert isinstance(mean_out, Rotation)
    assert type(sum_out) is AnalysisObject
    assert type(any_out) is AnalysisObject
    assert any_out.as_dataset(copy="none")["rotation"].dtype.kind == "b"


def test_group_core_p9d_007_grouped_weighted_supported_ops_match_ao_and_unsupported_fail_closed() -> None:
    """ID: GROUP_CORE_P9D_007_grouped_weighted_supported_ops_match_ao_and_unsupported_fail_closed."""
    grouped = _grouping_ao().group.groupby("label")
    weights = np.array([1.0, 2.0, 1.0], dtype=float)
    mean_out = grouped.mean(weights=weights, validate=True).as_dataset(copy="none")
    sum_out = grouped.sum(weights=weights, validate=True).as_dataset(copy="none")
    padded = grouped.padded(
        opts=GroupMaterializeOptions(layout="padded", include_empty_groups=False),
        validate=True,
    )
    mean_ref = padded.mean(dim="sample", weights=weights, validate=True).as_dataset(copy="none")
    sum_ref = padded.sum(dim="sample", weights=weights, validate=True).as_dataset(copy="none")
    assert mean_out.identical(mean_ref)
    assert sum_out.identical(sum_ref)
    with pytest.raises(ValueError, match="weights are supported only"):
        _ = grouped.var(weights=weights, validate=True)


def test_group_core_p9d_008_windowed_grouping_supports_primary_batch_key_with_preserve_batch_mean() -> None:
    """ID: GROUP_CORE_P9D_008_windowed_grouping_supports_primary_batch_key_with_preserve_batch_mean."""
    ao = _windowed_grouping_ao()
    out = ao.group.groupby("outcome", opts=GroupByOptions(preserve_batch=True)).mean(dim="trial", validate=True).as_dataset(copy="none")
    assert out["signal"].dims == ("event", "group_key", "tau")
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.asarray(["intercept", "miss"], dtype=object))
