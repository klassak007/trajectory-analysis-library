from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import validity_values
from tal.core.analysis_object import AnalysisObject
from tal.core.reducer_ops.validity import resolve_structural_valid_mask


def _ragged_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (("trial", "sample"), np.array([[1.0, np.nan, 5.0], [2.0, 3.0, np.nan]], dtype=float)),
            "energy": (("trial", "sample"), np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=float)),
            "flag": (("trial", "sample"), np.array([[True, False, True], [False, False, True]], dtype=bool)),
        },
        coords={
            "trial": np.array(["t0", "t1"], dtype=object),
            "sample": np.array([0, 1, 2], dtype=int),
            "time_s": (("trial", "sample"), np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=float)),
            "group_size": (("trial",), np.array([2, 1], dtype=np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
        sequence_size_coord="group_size",
    )


def test_reduce_core_p9c_002_ragged_tail_is_excluded_independent_of_skipna() -> None:
    """ID: REDUCE_CORE_P9C_002_ragged_tail_is_excluded_independent_of_skipna."""
    ao = _ragged_ao()
    out_true = ao.sum(dim="sample", skipna=True, validate=True).as_dataset(copy="none")
    out_false = ao.sum(dim="sample", skipna=False, validate=True).as_dataset(copy="none")
    np.testing.assert_allclose(out_true["energy"].to_numpy(), np.array([30.0, 40.0], dtype=float))
    np.testing.assert_allclose(out_false["energy"].to_numpy(), np.array([30.0, 40.0], dtype=float))


def test_reduce_hard_p9c_001_skipna_false_poisons_only_on_valid_prefix_missingness() -> None:
    """ID: REDUCE_HARD_P9C_001_skipna_false_poisons_only_on_valid_prefix_missingness."""
    ao = _ragged_ao()
    out = ao.mean(dim="sample", skipna=False, validate=True).as_dataset(copy="none")
    np.testing.assert_allclose(out["signal"].to_numpy(), np.array([np.nan, 2.0], dtype=float), equal_nan=True)


def test_reduce_core_p9c_004_count_any_all_apply_invalid_tail_neutral_semantics() -> None:
    """ID: REDUCE_CORE_P9C_004_count_any_all_apply_invalid_tail_neutral_semantics."""
    ao = _ragged_ao()
    count_out = ao.count(dim="sample", validate=True).as_dataset(copy="none")
    any_out = ao.any(dim="sample", validate=True).as_dataset(copy="none")
    all_out = ao.all(dim="sample", validate=True).as_dataset(copy="none")

    np.testing.assert_array_equal(count_out["signal"].to_numpy(), np.array([1, 1], dtype=np.int64))
    np.testing.assert_array_equal(count_out["energy"].to_numpy(), np.array([2, 1], dtype=np.int64))
    np.testing.assert_array_equal(any_out["flag"].to_numpy(), np.array([True, False], dtype=bool))
    np.testing.assert_array_equal(all_out["flag"].to_numpy(), np.array([False, False], dtype=bool))


def test_reduce_perf_p9c_009_sequence_size_is_normalized_once_per_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: REDUCE_PERF_P9C_009_sequence_size_is_normalized_once_per_dataset."""
    ao = _ragged_ao()
    original = validity_values.normalize_sequence_size_values
    calls = 0

    def counted_normalize(size: xr.DataArray, *, sequence_len: int) -> xr.DataArray:
        nonlocal calls
        calls += 1
        return original(size, sequence_len=sequence_len)

    monkeypatch.setattr(validity_values, "normalize_sequence_size_values", counted_normalize)
    ao.sum(dim="sample", skipna=True, validate=True)
    assert calls == 1


@pytest.mark.parametrize(
    "size",
    [
        pytest.param(np.asarray(1 + 2j, dtype=np.complex128), id="complex"),
        pytest.param(np.asarray(True, dtype=bool), id="bool"),
        pytest.param(np.asarray(1, dtype="timedelta64[s]"), id="timedelta"),
    ],
)
def test_reduce_hard_p9c_008_invalid_sequence_size_dtype_is_not_silently_unmasked(
    size: np.ndarray,
) -> None:
    """ID: REDUCE_HARD_P9C_008_invalid_sequence_size_dtype_is_not_silently_unmasked."""
    ds = xr.Dataset(
        {"signal": ("sample", np.asarray([1.0, 100.0]))},
        coords={"sample": [0, 1], "group_size": xr.DataArray(size)},
    )
    with pytest.raises(ValueError, match="real numeric count dtype"):
        resolve_structural_valid_mask(
            ds,
            sequence_dim="sample",
            sequence_size_coord="group_size",
            var=ds["signal"],
            owner="test.reducer",
        )
