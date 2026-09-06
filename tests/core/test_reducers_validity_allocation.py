from __future__ import annotations

import gc
import tracemalloc

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import AnalysisObject


def _static_ragged_source(samples: int) -> AnalysisObject:
    trials = 1024
    return AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("trial", np.ones(trials))},
            coords={
                "trial": np.arange(trials), "sample": np.arange(samples),
                "group_size": ("trial", np.full(trials, samples, dtype=np.int64)),
            },
        ),
        sequence_dim="sample", batch_dims=("trial",), core_dims=(),
        sequence_size_coord="group_size",
    )


def _reduction_peak(source: AnalysisObject) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        result = source.mean(dim="trial")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.as_dataset(copy="none")["value"].item() == 1.0
    return peak


def test_static_ragged_reduction_allocation_does_not_scale_as_row_grid() -> None:
    """ID: REDUCE_ALLOC_049_static_reductions_avoid_cartesian_validity_masks."""
    small = _static_ragged_source(32)
    large = _static_ragged_source(8192)
    small.mean(dim="trial")  # Warm library caches outside either measured call.
    small_peak = _reduction_peak(small)
    large_peak = _reduction_peak(large)
    assert large_peak - small_peak < 2 * 1024**2, (small_peak, large_peak)


def test_static_ragged_reduction_preserves_unrelated_lazy_data() -> None:
    """ID: REDUCE_ALLOC_050_static_reductions_preserve_unrelated_laziness."""
    calls: list[str] = []

    @delayed
    def values(name: str, data: np.ndarray) -> np.ndarray:
        calls.append(name)
        return data

    ds = xr.Dataset(
        {"value": ("trial", da.from_delayed(values("payload", np.ones(2)), shape=(2,), dtype=float))},
        coords={
            "trial": [0, 1], "sample": [0, 1, 2], "group_size": ("trial", [3, 2]),
            "aux": ("sample", da.from_delayed(values("coordinate", np.arange(3.0)), shape=(3,), dtype=float)),
        },
    )
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), sequence_size_coord="group_size"
    )
    actual = source.mean(dim="trial").as_dataset(copy="none")
    assert calls == []
    assert isinstance(actual["value"].data, da.Array)
    assert isinstance(actual.coords["aux"].data, da.Array)
    assert actual.sizes == {"sample": 3}
    assert actual["value"].compute().item() == 1.0
    assert calls == ["payload"]


@pytest.mark.parametrize("skipna", [False, True])
def test_sequence_dependent_reduction_still_masks_invalid_tail(skipna: bool) -> None:
    """ID: REDUCE_ALLOC_051_active_sequence_data_retains_validity_masking."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[1.0, 999.0, 999.0], [3.0, np.nan, 999.0]])},
            coords={"trial": [0, 1], "sample": [0, 1, 2], "group_size": ("trial", [1, 2])},
        ),
        sequence_dim="sample", batch_dims=("trial",), core_dims=(), sequence_size_coord="group_size",
    )
    actual = source.mean(dim="sample", skipna=skipna).as_dataset(copy="none")["value"]
    np.testing.assert_allclose(actual, [1.0, 3.0 if skipna else np.nan])
    # Reducing batch rather than sequence must still exclude every invalid tail.
    across_trials = source.mean(dim="trial", skipna=skipna).as_dataset(copy="none")["value"]
    np.testing.assert_allclose(across_trials, [2.0, np.nan, np.nan])
