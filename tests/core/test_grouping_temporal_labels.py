from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import (
    AnalysisObject,
    BatchGroupReduceOptions,
    GroupByOptions,
    GroupMaterializeOptions,
)


def _temporal_labels(kind: str) -> np.ndarray:
    if kind == "datetime":
        return np.asarray(["2020-01-02", "2020-01-01"], dtype="datetime64[ns]")
    return np.asarray([2, 1], dtype="timedelta64[ns]")


def _temporal_source(kind: str, topology: str) -> AnalysisObject:
    labels = _temporal_labels(kind)[[0, 1, 0]]
    ds = xr.Dataset(
        {"value": ("trial", [1.0, 2.0, 3.0])},
        coords={
            "trial": [0, 1, 2],
            "when": ("trial", labels),
            "kind": ("trial", ["x", "y", "x"]),
            "score": ("trial", [0.2, 1.2, 0.4]),
        },
    )
    if topology == "batch":
        return AnalysisObject.from_data(ds, batch_dims=("trial",), core_dims=())
    ds = ds.rename(trial="sample").expand_dims(trial=["run"])
    ds = ds.assign_coords({
        name: (("trial", "sample"), ds[name].data[None, :])
        for name in ("when", "kind", "score")
    })
    return AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",), core_dims=()
    )


@pytest.mark.parametrize("kind", ["datetime", "timedelta"])
@pytest.mark.parametrize("topology", ["batch", "sequence", "preserved"])
@pytest.mark.parametrize("multi_key", [False, True])
def test_temporal_grouping_labels_preserve_identity_and_order(
    kind: str, topology: str, multi_key: bool
) -> None:
    """ID: GROUP_LABEL_041_temporal_identity_across_topologies."""
    source = _temporal_source(kind, topology)
    key = ("when", "kind") if multi_key else "when"
    actual = source.group.groupby(
        key, preserve_batch=topology == "preserved"
    ).mean().as_dataset(copy="none")
    expected_labels = _temporal_labels(kind)
    if multi_key:
        actual_labels = actual.coords["group_key"].data
        assert actual_labels.tolist() == list(zip(expected_labels, ("x", "y"), strict=True))
        assert all(isinstance(label[0], (np.datetime64, np.timedelta64)) for label in actual_labels)
    else:
        np.testing.assert_array_equal(actual.coords["group_key"], expected_labels)
    np.testing.assert_allclose(actual["value"], [[2.0, 2.0]] if topology == "preserved" else [2.0, 2.0])


@pytest.mark.parametrize("kind", ["datetime", "timedelta"])
@pytest.mark.parametrize("topology", ["batch", "sequence", "preserved"])
def test_temporal_bin_labels_share_declared_and_observed_identity(
    kind: str, topology: str
) -> None:
    """ID: GROUP_LABEL_042_temporal_bin_order_across_topologies."""
    source = _temporal_source(kind, topology)
    labels = _temporal_labels(kind)
    grouped = source.group.groupby_bins("score", [0.0, 1.0, 2.0], labels=labels)
    if topology == "preserved":
        grouped = source.group.groupby_bins(
            "score", [0.0, 1.0, 2.0], labels=labels,
            opts=GroupByOptions(preserve_batch=True),
        )
    opts = BatchGroupReduceOptions() if topology == "batch" else GroupMaterializeOptions()
    actual = grouped.mean(opts=opts).as_dataset(copy="none")
    np.testing.assert_array_equal(actual.coords["group_key"], labels)
    np.testing.assert_allclose(actual["value"], [[2.0, 2.0]] if topology == "preserved" else [2.0, 2.0])
