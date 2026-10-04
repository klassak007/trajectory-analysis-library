from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal.core import (
    AnalysisObject,
    BatchGroupReduceOptions,
    GroupByOptions,
    GroupMaterializeOptions,
)
from tal.core.group_ops import GroupingFoundationOptions


def _grouping_source(topology: str) -> AnalysisObject:
    if topology == "batch":
        ds = xr.Dataset(
            {"value": ("trial", [1.0, 2.0])},
            coords={"trial": [0, 1], "score": ("trial", [0.25, 1.25])},
        )
        return AnalysisObject.from_data(ds, batch_dims=("trial",), core_dims=())
    ds = xr.Dataset(
        {"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": [0],
            "sample": [0, 1],
            "score": (("trial", "sample"), [[0.25, 1.25]]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
    )


@pytest.mark.parametrize("topology", ["sequence", "batch"])
@pytest.mark.parametrize(
    "labels",
    [
        (float("nan"), np.float64("nan")),
        (np.datetime64("NaT", "ns"), np.datetime64("NaT", "D")),
        (np.timedelta64("NaT", "ns"), np.timedelta64("NaT", "D")),
        (pd.NaT, np.datetime64("NaT", "ns")),
        ((float("nan"), "x"), (np.datetime64("NaT", "ns"), "x")),
    ],
)
def test_duplicate_missing_bin_labels_share_canonical_identity(
    topology: str,
    labels: tuple[object, object],
) -> None:
    """ID: GROUP_LABEL_057_duplicate_missing_bin_labels_fail_closed."""
    with pytest.raises(ValueError, match="labels must be unique"):
        _grouping_source(topology).group.groupby_bins(
            "score",
            [0.0, 1.0, 2.0],
            labels=labels,
        )


def _sequence_bin_na_source() -> AnalysisObject:
    return AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[10.0, 20.0]])},
            coords={
                "trial": ["run"],
                "sample": [0, 1],
                "score": (("trial", "sample"), [[1.5, np.nan]]),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
    )


@pytest.mark.parametrize("preserve_batch", [False, True])
def test_sequence_empty_domain_and_observed_na_share_one_group(
    preserve_batch: bool,
) -> None:
    """ID: GROUP_LABEL_058_sequence_empty_domain_uses_canonical_lookup."""
    domain_nat = np.datetime64("NaT", "ns")
    observed_nat = np.timedelta64("NaT", "ns")
    grouped = _sequence_bin_na_source().group.groupby_bins(
        "score",
        [0.0, 1.0, 2.0],
        labels=(domain_nat, "high"),
        opts=GroupByOptions(
            preserve_batch=preserve_batch,
            foundation_opts=GroupingFoundationOptions(
                na_key_policy="group",
                na_group_label=observed_nat,
            ),
        ),
    )
    actual = grouped.padded(
        opts=GroupMaterializeOptions(include_empty_groups=True)
    ).as_dataset(copy="none")
    labels = actual.coords["group_key"].data
    assert labels.shape == (2,)
    assert bool(pd.isna(labels[0]))
    assert labels[1] == "high"
    expected = np.asarray([[20.0], [10.0]])
    if preserve_batch:
        expected = np.asarray([[[20.0, np.nan], [10.0, np.nan]]])
    np.testing.assert_allclose(actual["value"], expected)


def test_batch_empty_domain_and_observed_na_share_one_group() -> None:
    """ID: GROUP_LABEL_060_batch_empty_domain_uses_canonical_lookup."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("trial", [10.0, 20.0])},
            coords={"trial": [0, 1], "score": ("trial", [1.5, np.nan])},
        ),
        batch_dims=("trial",),
        core_dims=(),
    )
    grouped = source.group.groupby_bins(
        "score",
        [0.0, 1.0, 2.0],
        labels=(np.datetime64("NaT", "ns"), "high"),
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(
                na_key_policy="group",
                na_group_label=pd.NaT,
            )
        ),
    )
    actual = grouped.mean(
        opts=BatchGroupReduceOptions(include_empty_groups=True)
    ).as_dataset(copy="none")
    labels = actual.coords["group_key"].data
    assert labels.shape == (2,)
    assert bool(pd.isna(labels[0]))
    assert labels[1] == "high"
    np.testing.assert_allclose(actual["value"], [20.0, 10.0])


@pytest.mark.parametrize("preserve_batch", [False, True])
def test_sequence_tuple_na_label_collision_uses_canonical_identity(
    preserve_batch: bool,
) -> None:
    """ID: GROUP_LABEL_059_tuple_na_label_collision_is_fail_closed."""
    keys = np.empty(2, dtype=object)
    keys[0] = (np.datetime64("NaT", "ns"), "x")
    keys[1] = np.nan
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[1.0, 2.0]])},
            coords={
                "trial": [0],
                "sample": [0, 1],
                "kind": (("trial", "sample"), keys[None, :]),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
    )
    with pytest.raises(ValueError, match="na_group_label=.*collides"):
        source.group.groupby(
            "kind",
            opts=GroupByOptions(
                preserve_batch=preserve_batch,
                foundation_opts=GroupingFoundationOptions(
                    na_key_policy="group",
                    na_group_label=(pd.NaT, "x"),
                ),
            ),
        )


@pytest.mark.parametrize("layout", ["padded", "stacked", "mean"])
@pytest.mark.parametrize("missing", [pd.NA, (pd.NA, "tag")])
def test_preserved_bin_reordering_uses_canonical_missing_identity(layout: str, missing: object) -> None:
    """ID: GROUP_LABEL_070_preserved_bin_order_handles_pandas_missing_labels."""
    source = _sequence_bin_na_source()
    other = "high" if missing is pd.NA else ("high", "tag")
    grouped = source.group.groupby_bins(
        "score", [0.0, 1.0, 2.0], labels=(missing, other),
        opts=GroupByOptions(
            preserve_batch=True,
            foundation_opts=GroupingFoundationOptions(na_key_policy="group", na_group_label=missing),
        ),
    )
    actual = getattr(grouped, layout)().as_dataset(copy="none")
    if layout == "mean" or layout == "stacked":
        np.testing.assert_allclose(actual["value"], [[20.0, 10.0]])
    else:
        np.testing.assert_allclose(actual["value"], [[[20.0, np.nan], [10.0, np.nan]]])
