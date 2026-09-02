from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, SchemaError
from tal.io.adapter_finalize import _finalize_owned_adapter_dataset


def _adapter_dataset(*, size: int = 2) -> tuple[xr.Dataset, tuple[np.ndarray, ...]]:
    values = np.asarray([[1.0, 2.0, np.nan]])
    times = np.asarray([[10, 11, 0]], dtype=np.int64)
    sizes = np.asarray([size], dtype=np.int64)
    ds = xr.Dataset(
        {"value": (("trial", "sample"), values)},
        coords={
            "trial": np.asarray(["run"], dtype=object),
            "sample": np.arange(3, dtype=np.int64),
            "time": (("trial", "sample"), times),
            "sequence_size": ("trial", sizes),
        },
    )
    return ds, (values, times, sizes)


def _finalize(ds: xr.Dataset, *, validate: bool) -> AnalysisObject:
    return _finalize_owned_adapter_dataset(
        ds,
        batch_dim="trial",
        sequence_dim="sample",
        size_name="sequence_size",
        param_name="time",
        validate=validate,
        owner="tal.io.read_csv_logs",
    )


def test_io_perf_p10b_010_owned_adapter_finalize_reuses_fresh_buffers() -> None:
    """ID: IO_PERF_P10B_010_owned_adapter_finalize_reuses_fresh_buffers."""
    ds, (values, times, sizes) = _adapter_dataset()

    out = _finalize(ds, validate=True)

    assert np.shares_memory(values, out.as_dataset(copy="none")["value"].data)
    assert np.shares_memory(times, out.as_dataset(copy="none").coords["time"].data)
    assert np.shares_memory(sizes, out.as_dataset(copy="none").coords["sequence_size"].data)

    external, (external_values, _, _) = _adapter_dataset()
    isolated = AnalysisObject(external)
    assert not np.shares_memory(external_values, isolated.as_dataset(copy="none")["value"].data)


def test_io_core_p10b_024_owned_adapter_finalize_preserves_validate_policy_and_owner() -> None:
    """ID: IO_CORE_P10B_024_owned_adapter_finalize_preserves_validate_policy_and_owner."""
    invalid, _ = _adapter_dataset(size=4)

    unchecked = _finalize(invalid, validate=False)
    assert unchecked.as_dataset(copy="none").coords["sequence_size"].item() == 4

    with pytest.raises(
        ValueError,
        match="tal.io.read_csv_logs: failed finalizing adapter dataset schema",
    ) as error:
        _finalize(invalid, validate=True)
    assert isinstance(error.value.__cause__, SchemaError)
