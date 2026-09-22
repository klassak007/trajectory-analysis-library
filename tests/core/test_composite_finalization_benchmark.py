from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from benchmarks.bench_composite_finalization import (
    DEFAULT_CONFIG,
    CompositeBenchmarkConfig,
    _validate,
    benchmark_report,
    composite_fixture,
)


def test_composite_benchmark_protocol_validates_routes_and_task_growth() -> None:
    config = CompositeBenchmarkConfig(
        trials=4,
        source_samples=17,
        query_samples=33,
        trial_chunk=1,
        source_chunk=8,
        query_chunk=16,
    )

    report = benchmark_report(config, warmups=0, repeats=1)

    assert set(report["eager"]) == {
        "pose_temporal",
        "pose_components",
        "paired_kinematics",
    }
    for evidence in report["partition_scaling"].values():
        assert evidence["one_partition"] > 0
        assert evidence["four_partitions"] <= 5 * evidence["one_partition"]


def test_composite_benchmark_projects_frozen_ragged_validity() -> None:
    fixture = composite_fixture(DEFAULT_CONFIG, lazy=False)
    expected = np.where(np.arange(DEFAULT_CONFIG.trials) % 2 == 0, 1_025, 769)

    for value in (
        fixture.linear_velocity,
        fixture.angular_velocity,
        fixture.linear_acceleration,
        fixture.angular_acceleration,
    ):
        np.testing.assert_array_equal(
            value.as_dataset(copy="none")["group_size"].data,
            expected,
        )


def _metadata_pair() -> tuple[xr.Dataset, xr.Dataset]:
    coordinates = xr.Coordinates.from_xindex(
        xr.indexes.RangeIndex.arange(2, dim="row")
    )
    reference = xr.Dataset(
        {"value": ("row", [1.0, 2.0], {"units": "m"})},
        coords=coordinates,
        attrs={
            "tal": {"core": {"roles": {"batch_dims": ["row"]}}},
            "ordinary": {"fixture": "frozen"},
        },
    )
    return reference.copy(deep=True), reference


def test_composite_benchmark_rejects_metadata_and_index_drift() -> None:
    observed, reference = _metadata_pair()
    observed.attrs["tal"]["core"]["roles"] = {"core_dims": ["row"]}
    with pytest.raises(AssertionError, match="Dataset metadata"):
        _validate(observed, (reference,))

    observed, reference = _metadata_pair()
    observed.attrs["ordinary"]["fixture"] = "changed"
    with pytest.raises(AssertionError, match="Dataset metadata"):
        _validate(observed, (reference,))

    observed, reference = _metadata_pair()
    observed["value"].attrs["units"] = "cm"
    with pytest.raises(AssertionError, match="metadata changed for 'value'"):
        _validate(observed, (reference,))

    observed, reference = _metadata_pair()
    observed["value"].encoding["benchmark"] = "changed"
    with pytest.raises(AssertionError, match="metadata changed for 'value'"):
        _validate(observed, (reference,))

    _, reference = _metadata_pair()
    observed = xr.Dataset(
        {"value": ("row", [1.0, 2.0], {"units": "m"})},
        coords={"row": [0, 1]},
        attrs=reference.attrs,
    )
    with pytest.raises(AssertionError, match="index type changed"):
        _validate(observed, (reference,))
