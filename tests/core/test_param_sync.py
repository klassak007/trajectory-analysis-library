import importlib
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import xarray as xr

from tal.core import AnalysisObject, ParamSyncOptions, synchronize, synchronize_param


def _ao_sync_left() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 10.0, 20.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")


def _ao_sync_right() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [100.0, 200.0, 300.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [1.0, 2.0, 3.0])},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")


def _integer_sync_ao(param: np.ndarray, values: list[float]) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": ("sample", np.asarray(values, dtype="float64"))},
        coords={
            "sample": np.arange(param.size, dtype="int64"),
            "tau": ("sample", param),
        },
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")


def _ao_sync_left_valid() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 10.0, 20.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "group_size": xr.DataArray(np.asarray(3, dtype="int64"), dims=()),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )


def _ao_sync_right_valid() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [100.0, 200.0, 300.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [1.0, 2.0, 3.0]),
            "group_size": xr.DataArray(np.asarray(3, dtype="int64"), dims=()),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )


def _ao_batch_a() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 10.0, 20.0], [1.0, 11.0, 21.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (("trial", "sample"), [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            "group_size": ("trial", [3, 3]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )


def _ao_batch_b() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[100.0, 110.0, 120.0], [200.0, 210.0, 220.0]])},
        coords={
            "trial": ["b", "c"],
            "sample": [0, 1, 2],
            "phase": (("trial", "sample"), [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]),
            "group_size": ("trial", [3, 3]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )


def _ao_batch_sample_invariant(offset: float) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype="float64") + offset)},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": ("sample", [0.0, 1.0, 2.0]),
            "group_size": ("trial", [3, 3]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )


def _ao_batch_sample_invariant_no_validity(labels: list[str], offset: float) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("trial", "sample"),
                np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype="float64") + offset,
            )
        },
        coords={
            "trial": labels,
            "sample": [0, 1, 2],
            "phase": ("sample", [0.0, 1.0, 2.0]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )


def _ao_batch_with_labels(labels: np.ndarray, *, offset: float = 0.0) -> AnalysisObject:
    n = int(labels.size)
    values = np.arange(n * 2, dtype="float64").reshape(n, 2) + float(offset)
    phase = np.tile(np.asarray([0.0, 1.0], dtype="float64"), (n, 1))
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), values)},
        coords={
            "trial": labels,
            "sample": [0, 1],
            "phase": (("trial", "sample"), phase),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )


def _assert_nullable_trial_labels(values: np.ndarray) -> None:
    labels = list(values)
    assert len(labels) == 2
    assert pd.isna(labels[0])
    assert labels[1] == "a"


def _ao_multi_batch_sync(
    *,
    trial_labels: tuple[str, ...] = ("a", "b"),
    sensor_labels: tuple[str, ...] = ("s0", "s1"),
    offset: float = 0.0,
) -> AnalysisObject:
    base = np.asarray([0.0, 1.0, 2.0], dtype="float64")
    values = np.zeros((len(trial_labels), len(sensor_labels), base.size), dtype="float64")
    phase = np.zeros_like(values)
    for i in range(len(trial_labels)):
        for j in range(len(sensor_labels)):
            values[i, j] = base + offset + i * 10.0 + j
            phase[i, j] = base
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), values)},
        coords={
            "trial": list(trial_labels),
            "sensor": list(sensor_labels),
            "sample": np.arange(base.size, dtype="int64"),
            "phase": (("trial", "sensor", "sample"), phase),
            "group_size": (("trial", "sensor"), np.full((len(trial_labels), len(sensor_labels)), base.size)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="phase",
        sequence_size_coord="group_size",
    )


def test_param_sync_011_join_outer_inner_domain_exact() -> None:
    """ID: PARAM_SYNC_011_join_outer_inner_domain_exact."""
    a = _ao_sync_left()
    b = _ao_sync_right()
    outer = synchronize_param([a, b], opts=ParamSyncOptions(join="outer", how="nearest"))
    inner = synchronize_param([a, b], opts=ParamSyncOptions(join="inner", how="nearest"))
    domain = synchronize_param([a, b], opts=ParamSyncOptions(join="domain", how="nearest"))
    np.testing.assert_allclose(outer[0].as_dataset().coords["tau"].values, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(inner[0].as_dataset().coords["tau"].values, [1.0, 2.0])
    np.testing.assert_allclose(domain[0].as_dataset().coords["tau"].values, [1.0, 2.0])
    with pytest.raises(ValueError) as err:
        synchronize_param([a, b], opts=ParamSyncOptions(join="exact", how="nearest"))
    assert "join='exact'" in str(err.value)


def test_param_sync_046_join_tolerance_edge_parity() -> None:
    """ID: PARAM_SYNC_046_join_tolerance_edge_parity."""
    left = _ao_sync_left()
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [100.0, 200.0, 300.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.05, 1.05, 3.0])},
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    outer = synchronize_param([left, right], opts=ParamSyncOptions(join="outer", how="nearest", tol=0.1))
    inner = synchronize_param([left, right], opts=ParamSyncOptions(join="inner", how="nearest", tol=0.1))
    domain = synchronize_param([left, right], opts=ParamSyncOptions(join="domain", how="nearest", tol=0.1))
    np.testing.assert_allclose(outer[0].as_dataset().coords["tau"].values, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(inner[0].as_dataset().coords["tau"].values, [0.0, 1.0])
    np.testing.assert_allclose(domain[0].as_dataset().coords["tau"].values, [1.0, 2.0])


def test_param_sync_047_join_exact_with_tolerance_accepts_near_equal_rows() -> None:
    """ID: PARAM_SYNC_047_join_exact_with_tolerance_accepts_near_equal_rows."""
    left = _ao_sync_left()
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [100.0, 200.0, 300.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.05, 1.05, 2.05])},
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out = synchronize_param([left, right], opts=ParamSyncOptions(join="exact", how="nearest", tol=0.1))
    np.testing.assert_allclose(out[0].as_dataset().coords["tau"].values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(out[1].as_dataset().coords["tau"].values, [0.0, 1.0, 2.0])


def test_param_sync_048_outer_join_keeps_last_kept_tolerance_points() -> None:
    """ID: PARAM_SYNC_048_outer_join_keeps_last_kept_tolerance_points."""
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), [1.0])},
        coords={"sample": [0], "tau": ("sample", [0.92260597])},
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [10.0, 20.0, 30.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [2.03273864, 2.3821086, 2.55788419])},
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    out = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(join="outer", how="nearest", tol=0.4332844209688609),
    )
    np.testing.assert_allclose(out[0].as_dataset().coords["tau"].values, [0.92260597, 2.03273864, 2.55788419])


def test_param_sync_012_how_fill_tolerance_mask_behavior() -> None:
    """ID: PARAM_SYNC_012_how_fill_tolerance_mask_behavior."""
    a = _ao_sync_left()
    b = _ao_sync_right()
    out = synchronize_param(
        [a, b],
        opts=ParamSyncOptions(join="left", how="fill", tol=0.1, fill_value=-999.0),
    )
    vals = out[1].as_dataset()["value"].values
    np.testing.assert_allclose(vals, [-999.0, 100.0, 200.0])


def test_param_sync_013_batch_join_inner_outer_exact() -> None:
    """ID: PARAM_SYNC_013_batch_join_inner_outer_exact."""
    a = _ao_batch_a()
    b = _ao_batch_b()
    inner = synchronize_param([a, b], opts=ParamSyncOptions(batch_join="inner", join="left", how="nearest"))
    outer = synchronize_param([a, b], opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"))
    assert list(inner[0].as_dataset().coords["trial"].values) == ["b"]
    assert list(outer[0].as_dataset().coords["trial"].values) == ["a", "b", "c"]
    with pytest.raises(ValueError) as err:
        synchronize_param([a, b], opts=ParamSyncOptions(batch_join="exact", join="left", how="nearest"))
    assert "batch_join='exact'" in str(err.value)


def test_param_sync_014_batch_join_and_join_composed() -> None:
    """ID: PARAM_SYNC_014_batch_join_and_join_composed."""
    a = _ao_batch_a()
    b = _ao_batch_b()
    out = synchronize(
        [a, b],
        opts=ParamSyncOptions(batch_join="outer", join="inner", how="nearest"),
    )
    labels = list(out[0].as_dataset().coords["trial"].values)
    assert labels == ["a", "b", "c"]
    sizes = out[0].as_dataset().coords["group_size"]
    assert int(sizes.sel(trial="a").item()) == 0
    assert int(sizes.sel(trial="c").item()) == 0
    assert int(sizes.sel(trial="b").item()) > 0


def test_param_sync_015_sample_invariant_batched_rows_do_not_crash() -> None:
    """ID: PARAM_SYNC_015_sample_invariant_batched_rows_do_not_crash."""
    a = _ao_batch_sample_invariant(0.0)
    b = _ao_batch_sample_invariant(100.0)
    out = synchronize_param([a, b], opts=ParamSyncOptions(join="inner", how="nearest"))
    assert len(out) == 2
    np.testing.assert_allclose(out[0].as_dataset().coords["phase"].sel(trial="a").values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(out[1].as_dataset().coords["phase"].sel(trial="b").values, [0.0, 1.0, 2.0])


def test_param_sync_016_invalid_sync_options_fail_fast() -> None:
    """ID: PARAM_SYNC_016_invalid_sync_options_fail_fast."""
    a = _ao_sync_left()
    b = _ao_sync_right()
    with pytest.raises(ValueError) as err_join:
        synchronize_param([a, b], grid=np.asarray([0.0, 1.0]), opts=ParamSyncOptions(join="bogus"))  # type: ignore[arg-type]
    assert "opts.join must be one of" in str(err_join.value)
    with pytest.raises(ValueError) as err_how:
        synchronize_param([a, b], grid=np.asarray([0.0, 1.0]), opts=ParamSyncOptions(how="bogus"))  # type: ignore[arg-type]
    assert "opts.how must be one of" in str(err_how.value)
    with pytest.raises(ValueError) as err_batch:
        synchronize_param([a, b], grid=np.asarray([0.0, 1.0]), opts=ParamSyncOptions(batch_join="bogus"))  # type: ignore[arg-type]
    assert "opts.batch_join must be one of" in str(err_batch.value)


def test_param_sync_017_fill_updates_valid_and_sequence_size() -> None:
    """ID: PARAM_SYNC_017_fill_updates_valid_and_sequence_size."""
    a = _ao_sync_left_valid()
    b = _ao_sync_right_valid()
    out = synchronize_param(
        [a, b],
        opts=ParamSyncOptions(join="left", how="fill", tol=0.1, fill_value=-999.0),
    )
    vals = out[1].as_dataset()["value"].values
    valid = out[1].as_dataset().coords["valid"].values
    np.testing.assert_allclose(vals, [-999.0, 100.0, 200.0])
    np.testing.assert_array_equal(valid, [False, True, True])
    assert "group_size" not in out[1].as_dataset().coords


def test_param_sync_018_outer_batch_join_sample_invariant_missing_rows_invalid() -> None:
    """ID: PARAM_SYNC_018_outer_batch_join_sample_invariant_missing_rows_invalid."""
    left = _ao_batch_sample_invariant_no_validity(["a", "b"], 0.0)
    right = _ao_batch_sample_invariant_no_validity(["b", "c"], 100.0)
    out_left, out_right = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
    )
    assert "trial" in out_left.as_dataset().coords["valid"].dims
    assert bool(out_left.as_dataset()["value"].sel(trial="c").isnull().all().item()) is True
    np.testing.assert_array_equal(out_left.as_dataset().coords["valid"].sel(trial="c").values, [False, False, False])
    np.testing.assert_array_equal(out_right.as_dataset().coords["valid"].sel(trial="a").values, [False, False, False])


def test_param_sync_019_fill_non_left_packed_drops_validity() -> None:
    """ID: PARAM_SYNC_019_fill_non_left_packed_drops_validity."""
    a = _ao_sync_left_valid()
    b = _ao_sync_right_valid()
    out = synchronize_param(
        [a, b],
        opts=ParamSyncOptions(join="left", how="fill", tol=0.1, fill_value=-999.0),
    )
    core = out[1].as_dataset().attrs["tal"]["core"]
    assert "validity" not in core
    assert "group_size" not in out[1].as_dataset().coords


def test_param_sync_020_fill_left_packed_retains_validity() -> None:
    """ID: PARAM_SYNC_020_fill_left_packed_retains_validity."""
    a = _ao_sync_left_valid()
    b = _ao_sync_right_valid()
    out = synchronize_param(
        [a, b],
        opts=ParamSyncOptions(join="right", how="fill", tol=0.1, fill_value=-999.0),
    )
    core = out[0].as_dataset().attrs["tal"]["core"]
    assert "validity" in core
    np.testing.assert_array_equal(out[0].as_dataset().coords["valid"].values, [True, True, False])
    assert int(out[0].as_dataset().coords["group_size"].item()) == 2


def test_param_sync_021_duplicate_batch_labels_fail_fast() -> None:
    """ID: PARAM_SYNC_021_duplicate_batch_labels_fail_fast."""
    ao = _ao_batch_a()
    q = xr.DataArray(
        np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype="float64"),
        dims=("trial", "query"),
        coords={"trial": ["a", "a"], "query": [0, 1]},
    )
    with pytest.raises(ValueError) as err:
        synchronize_param([ao], grid=q, opts=ParamSyncOptions(join="left", how="nearest"))
    assert "labels along 'trial' must be unique" in str(err.value)


def test_param_sync_022_duplicate_source_batch_labels_fail_fast() -> None:
    """ID: PARAM_SYNC_022_duplicate_source_batch_labels_fail_fast."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={
            "trial": ["a", "a"],
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    with pytest.raises(ValueError) as err:
        synchronize_param([ao], opts=ParamSyncOptions(join="left", how="nearest"))
    assert "labels along 'trial' must be unique" in str(err.value)


def test_param_sync_023_outer_batch_join_mixed_label_domains_rejected() -> None:
    """ID: PARAM_SYNC_023_outer_batch_join_mixed_label_domains_rejected."""
    ds_int = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={
            "trial": [1, 2],
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
        },
    )
    ds_str = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[10.0, 11.0], [12.0, 13.0]])},
        coords={
            "trial": ["1", "2"],
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
        },
    )
    ao_int = AnalysisObject.from_data(
        ds_int,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    ao_str = AnalysisObject.from_data(
        ds_str,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    with pytest.raises(ValueError) as err:
        synchronize_param(
            [ao_int, ao_str],
            opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
        )
    assert "incompatible mixed batch label domains" in str(err.value)


def test_param_sync_024_sync_query_dim_override_resolves_namespace_collision() -> None:
    """ID: PARAM_SYNC_024_sync_query_dim_override_resolves_namespace_collision."""
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0, 2.0]),
            "aux": (("query",), [10.0, 20.0]),
        },
        coords={"sample": [0, 1, 2], "query": ["q0", "q1"], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out = synchronize_param(
        [ao],
        opts=ParamSyncOptions(join="left", how="nearest", query_dim="q_sync"),
    )[0]
    np.testing.assert_allclose(out.as_dataset()["value"].values, [0.0, 1.0, 2.0])


def test_param_sync_025_null_batch_labels_single_input_outer_identity() -> None:
    """ID: PARAM_SYNC_025_null_batch_labels_single_input_outer_identity."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": np.asarray([np.datetime64("NaT", "ns")], dtype="datetime64[ns]"),
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0]]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    out = synchronize_param(
        [ao],
        opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
    )[0]
    assert bool(np.isnat(out.as_dataset().coords["trial"].values[0]))
    np.testing.assert_allclose(out.as_dataset()["value"].values, [[1.0, 2.0]])
    np.testing.assert_array_equal(out.as_dataset().coords["valid"].values, [[True, True]])


def test_param_sync_026_null_batch_labels_single_input_inner_exact_identity() -> None:
    """ID: PARAM_SYNC_026_null_batch_labels_single_input_inner_exact_identity."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": np.asarray([np.datetime64("NaT", "ns")], dtype="datetime64[ns]"),
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0]]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    out_inner = synchronize_param(
        [ao],
        opts=ParamSyncOptions(batch_join="inner", join="left", how="nearest"),
    )[0]
    out_exact = synchronize_param(
        [ao],
        opts=ParamSyncOptions(batch_join="exact", join="left", how="nearest"),
    )[0]
    np.testing.assert_allclose(out_inner.as_dataset()["value"].values, [[1.0, 2.0]])
    np.testing.assert_allclose(out_exact.as_dataset()["value"].values, [[1.0, 2.0]])


def test_param_sync_027_outer_nan_sample_invariant_validity_not_false_missing() -> None:
    """ID: PARAM_SYNC_027_outer_nan_sample_invariant_validity_not_false_missing."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": np.asarray([np.nan], dtype="float64"),
            "sample": [0, 1],
            "phase": ("sample", [0.0, 1.0]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    out = synchronize_param(
        [ao],
        opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
    )[0]
    np.testing.assert_array_equal(out.as_dataset().coords["valid"].values, [[True, True]])
    np.testing.assert_allclose(out.as_dataset()["value"].values, [[1.0, 2.0]])


def test_param_sync_028_outer_mixed_none_nonnull_domains_fail_deterministically() -> None:
    """ID: PARAM_SYNC_028_outer_mixed_none_nonnull_domains_fail_deterministically."""
    ds_a = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0]])},
        coords={
            "trial": np.asarray(["a"], dtype=object),
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0]]),
        },
    )
    ds_b = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[10.0, 11.0]])},
        coords={
            "trial": np.asarray([None], dtype=object),
            "sample": [0, 1],
            "phase": (("trial", "sample"), [[0.0, 1.0]]),
        },
    )
    ao_a = AnalysisObject.from_data(
        ds_a,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    ao_b = AnalysisObject.from_data(
        ds_b,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="phase",
    )
    with pytest.raises(ValueError) as err:
        synchronize_param(
            [ao_a, ao_b],
            opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
        )
    assert "incompatible mixed batch label domains" in str(err.value)


def test_param_sync_029_outer_single_input_mixed_null_nonnull_identity() -> None:
    """ID: PARAM_SYNC_029_outer_single_input_mixed_null_nonnull_identity."""
    ao = _ao_batch_with_labels(np.asarray([None, "a"], dtype=object))
    out = synchronize_param(
        [ao],
        opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
    )[0]
    _assert_nullable_trial_labels(out.as_dataset().coords["trial"].values)
    np.testing.assert_allclose(out.as_dataset()["value"].values, ao.as_dataset()["value"].values)
    np.testing.assert_array_equal(out.as_dataset().coords["valid"].values, np.ones((2, 2), dtype=bool))


def test_param_sync_030_outer_multi_input_aligned_mixed_null_nonnull_allowed() -> None:
    """ID: PARAM_SYNC_030_outer_multi_input_aligned_mixed_null_nonnull_allowed."""
    labels = np.asarray([None, "a"], dtype=object)
    left = _ao_batch_with_labels(labels, offset=0.0)
    right = _ao_batch_with_labels(labels, offset=100.0)
    out_left, out_right = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
    )
    _assert_nullable_trial_labels(out_left.as_dataset().coords["trial"].values)
    _assert_nullable_trial_labels(out_right.as_dataset().coords["trial"].values)
    np.testing.assert_allclose(out_left.as_dataset()["value"].values, left.as_dataset()["value"].values)
    np.testing.assert_allclose(out_right.as_dataset()["value"].values, right.as_dataset()["value"].values)


def test_param_sync_031_inner_exact_mixed_null_representation_fail_tal_valueerror() -> None:
    """ID: PARAM_SYNC_031_inner_exact_mixed_null_representation_fail_tal_valueerror."""
    ao_nan = _ao_batch_with_labels(np.asarray([np.nan], dtype="float64"))
    ao_none = _ao_batch_with_labels(np.asarray([None], dtype=object))
    for mode in ("inner", "exact"):
        with pytest.raises(ValueError) as err:
            synchronize_param(
                [ao_nan, ao_none],
                opts=ParamSyncOptions(batch_join=mode, join="left", how="nearest"),
            )
        assert "batch labels are not representable" in str(err.value)


def test_param_sync_032_sync_query_dim_invalid_or_reserved_rejected() -> None:
    """ID: PARAM_SYNC_032_sync_query_dim_invalid_or_reserved_rejected."""
    ao = _ao_sync_left()
    with pytest.raises(ValueError) as err_type:
        synchronize_param([ao], opts=ParamSyncOptions(query_dim=1))  # type: ignore[arg-type]
    assert "query_dim" in str(err_type.value)
    with pytest.raises(ValueError) as err_empty:
        synchronize_param([ao], opts=ParamSyncOptions(query_dim=""))
    assert "query_dim" in str(err_empty.value)
    with pytest.raises(ValueError) as err_reserved:
        synchronize_param([ao], opts=ParamSyncOptions(query_dim="valid"))
    assert "reserved" in str(err_reserved.value)


def test_param_sync_033_fill_value_invalid_type_rejected() -> None:
    """ID: PARAM_SYNC_033_fill_value_invalid_type_rejected."""
    left = _ao_sync_left()
    right = _ao_sync_right()
    with pytest.raises(ValueError) as err:
        synchronize_param(
            [left, right],
            opts=ParamSyncOptions(join="left", how="fill", fill_value={"a": 1}),  # type: ignore[arg-type]
        )
    assert "fill_value" in str(err.value)


def test_param_sync_034_fill_value_string_rejected_no_object_dtype_pollution() -> None:
    """ID: PARAM_SYNC_034_fill_value_string_rejected_no_object_dtype_pollution."""
    left = _ao_sync_left()
    right = _ao_sync_right()
    with pytest.raises(ValueError) as err:
        synchronize_param(
            [left, right],
            opts=ParamSyncOptions(join="left", how="fill", fill_value="x"),  # type: ignore[arg-type]
        )
    assert "fill_value" in str(err.value)


def test_param_sync_035_tol_invalid_type_rejected_tal_owned() -> None:
    """ID: PARAM_SYNC_035_tol_invalid_type_rejected_tal_owned."""
    ao = _ao_sync_left()
    with pytest.raises(ValueError) as err:
        synchronize_param([ao], opts=ParamSyncOptions(tol="x"))  # type: ignore[arg-type]
    assert "opts.tol" in str(err.value)
    assert "numeric scalar" in str(err.value)


def test_param_sync_036_fill_value_ignored_when_how_not_fill() -> None:
    """ID: PARAM_SYNC_036_fill_value_ignored_when_how_not_fill."""
    left = _ao_sync_left()
    right = _ao_sync_right()
    out = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(join="left", how="interp", fill_value="bad"),  # type: ignore[arg-type]
    )
    assert len(out) == 2
    np.testing.assert_allclose(out[0].as_dataset().coords["tau"].values, [0.0, 1.0, 2.0])


def test_param_sync_037_outer_permuted_mixed_null_nonnull_allowed() -> None:
    """ID: PARAM_SYNC_037_outer_permuted_mixed_null_nonnull_allowed."""
    left = _ao_batch_with_labels(np.asarray([None, "a"], dtype=object), offset=0.0)
    right = _ao_batch_with_labels(np.asarray(["a", None], dtype=object), offset=100.0)
    out_left, out_right = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
    )
    _assert_nullable_trial_labels(out_left.as_dataset().coords["trial"].values)
    _assert_nullable_trial_labels(out_right.as_dataset().coords["trial"].values)


def test_param_sync_038_dask_sync_eval_path_no_chunked_indexer_error() -> None:
    """ID: PARAM_SYNC_038_dask_sync_eval_path_no_chunked_indexer_error."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2)),
        },
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    out = synchronize_param([ao], opts=ParamSyncOptions(join="left", how="nearest"))[0]
    np.testing.assert_allclose(out.as_dataset()["value"].values, [0.0, 10.0, 20.0])


def test_param_sync_039_synchronize_param_input_type_checked() -> None:
    """ID: PARAM_SYNC_039_synchronize_param_input_type_checked."""
    with pytest.raises(TypeError) as err:
        synchronize_param([object()])  # type: ignore[list-item]
    assert "AnalysisObject, xr.Dataset, or xr.DataArray" in str(err.value)

    ao = _ao_sync_left()
    out_ds = synchronize_param([ao.as_dataset()], opts=ParamSyncOptions(join="left", how="nearest"))
    assert len(out_ds) == 1
    da_input = ao.as_dataset()["value"].copy(deep=True)
    da_input.attrs["tal"] = ao.as_dataset().attrs["tal"]
    out_da = synchronize_param(
        [da_input],
        opts=ParamSyncOptions(join="left", how="nearest"),
    )
    assert len(out_da) == 1


def test_param_sync_045_synchronize_param_impostor_unsafe_data_rejected_typeerror() -> None:
    """ID: PARAM_SYNC_045_synchronize_param_impostor_unsafe_data_rejected_typeerror."""

    class _Impostor:
        @property
        def unsafe_data(self) -> xr.Dataset:
            return _ao_sync_left().as_dataset(copy="none")

    with pytest.raises(TypeError) as err:
        synchronize_param([_Impostor()])  # type: ignore[list-item]
    assert "AnalysisObject, xr.Dataset, or xr.DataArray" in str(err.value)


def test_param_sync_040_multi_batch_sync_supported() -> None:
    """ID: PARAM_SYNC_040_multi_batch_sync_supported."""
    left = _ao_multi_batch_sync(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    right = _ao_multi_batch_sync(trial_labels=("b", "c"), sensor_labels=("s0", "s1"), offset=100.0)
    out_left, out_right = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="inner", join="left", how="nearest"),
    )
    assert {"trial", "sensor", "sample"} <= set(out_left.as_dataset()["value"].dims)
    assert {"trial", "sensor", "sample"} <= set(out_right.as_dataset()["value"].dims)
    assert list(out_left.as_dataset().coords["trial"].values) == ["b"]
    assert list(out_right.as_dataset().coords["trial"].values) == ["b"]
    assert list(out_left.as_dataset().coords["sensor"].values) == ["s0", "s1"]
    out_left_outer, out_right_outer = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="outer", join="left", how="nearest"),
    )
    assert list(out_left_outer.as_dataset().coords["trial"].values) == ["a", "b", "c"]
    assert list(out_right_outer.as_dataset().coords["trial"].values) == ["a", "b", "c"]
    assert list(out_left_outer.as_dataset().coords["sensor"].values) == ["s0", "s1"]
    assert list(out_right_outer.as_dataset().coords["sensor"].values) == ["s0", "s1"]


def test_param_sync_041_sync_module_option_validation_centralized_behavior_parity() -> None:
    """ID: PARAM_SYNC_041_sync_module_option_validation_centralized_behavior_parity."""
    ao = _ao_sync_left()
    with pytest.raises(ValueError) as err_tol:
        synchronize_param([ao], opts=ParamSyncOptions(tol=-1.0))
    assert "opts.tol" in str(err_tol.value)
    with pytest.raises(ValueError) as err_fill:
        synchronize_param([ao], opts=ParamSyncOptions(how="fill", fill_value="bad"))  # type: ignore[arg-type]
    assert "fill_value" in str(err_fill.value)
    out = synchronize_param([ao], opts=ParamSyncOptions(how="nearest", fill_value="bad"))  # type: ignore[arg-type]
    assert len(out) == 1


def test_param_sync_042_multi_batch_interp_like_uses_shared_flatten_contract() -> None:
    """ID: PARAM_SYNC_042_multi_batch_interp_like_uses_shared_flatten_contract."""
    left = _ao_multi_batch_sync(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    right = _ao_multi_batch_sync(trial_labels=("b", "a"), sensor_labels=("s1", "s0"), offset=50.0)
    out = left.param.interp_like(right, batch_join="inner")
    assert set(out.as_dataset()["value"].dims) == {"trial", "sensor", "sample"}
    assert list(out.as_dataset().coords["trial"].values) == ["a", "b"]
    assert list(out.as_dataset().coords["sensor"].values) == ["s0", "s1"]


def test_orch_topo_parity_001_sync_multi_batch_behavior_parity() -> None:
    """ID: ORCH_TOPO_PARITY_001_sync_multi_batch_behavior_parity."""
    left = _ao_multi_batch_sync(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    right = _ao_multi_batch_sync(trial_labels=("b", "a"), sensor_labels=("s1", "s0"), offset=25.0)
    out = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="inner", join="left", how="nearest"),
    )
    assert len(out) == 2
    assert set(out[0].as_dataset()["value"].dims) == {"trial", "sensor", "sample"}
    assert list(out[0].as_dataset().coords["trial"].values) == ["a", "b"]
    assert list(out[0].as_dataset().coords["sensor"].values) == ["s0", "s1"]


def test_orch_finalize_parity_001_sync_identity_and_restored_paths_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ORCH_FINALIZE_PARITY_001_sync_identity_and_restored_paths_stable."""
    sync_mod = importlib.import_module("tal.core.param_ops.sync")
    counts = {"finalize_like": 0, "restore_and_finalize": 0}
    orig_finalize = sync_mod.finalize_like
    orig_restore = sync_mod.restore_and_finalize

    def _count_finalize(*args, **kwargs):  # type: ignore[no-untyped-def]
        counts["finalize_like"] += 1
        return orig_finalize(*args, **kwargs)

    def _count_restore(*args, **kwargs):  # type: ignore[no-untyped-def]
        counts["restore_and_finalize"] += 1
        return orig_restore(*args, **kwargs)

    monkeypatch.setattr(sync_mod, "finalize_like", _count_finalize)
    monkeypatch.setattr(sync_mod, "restore_and_finalize", _count_restore)

    single = _ao_sync_left()
    out_single = synchronize_param([single], opts=ParamSyncOptions(join="left", how="nearest"))
    np.testing.assert_allclose(out_single[0].as_dataset()["value"].values, single.as_dataset()["value"].values)

    left = _ao_multi_batch_sync(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    right = _ao_multi_batch_sync(trial_labels=("b", "a"), sensor_labels=("s1", "s0"), offset=50.0)
    out_multi = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="inner", join="left", how="nearest"),
    )
    assert len(out_multi) == 2
    assert set(out_multi[0].as_dataset()["value"].dims) == {"trial", "sensor", "sample"}
    assert counts["finalize_like"] >= 1
    assert counts["restore_and_finalize"] >= 1


def test_param_sync_043_no_stale_single_batch_error_text_paths() -> None:
    """ID: PARAM_SYNC_043_no_stale_single_batch_error_text_paths."""
    path = Path("tal/core/param_ops/interp_like.py")
    text = path.read_text(encoding="utf-8")
    assert "currently supports one batch dimension" not in text


def test_param_sync_044_single_input_identity_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_SYNC_044_single_input_identity_fast_path."""
    sync_mod = importlib.import_module("tal.core.param_ops.sync")
    ao = _ao_sync_left()

    def _boom_sync_one(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected sync execution")

    monkeypatch.setattr(sync_mod, "_sync_one", _boom_sync_one)
    out = synchronize_param([ao], opts=ParamSyncOptions(join="left", how="nearest"))
    assert len(out) == 1
    np.testing.assert_allclose(out[0].as_dataset()["value"].values, ao.as_dataset()["value"].values)
    np.testing.assert_array_equal(out[0].as_dataset().coords["valid"].values, [True, True, True])


def test_param_sync_049_auto_join_chunked_inputs_without_grid_fails_fast() -> None:
    """ID: PARAM_SYNC_049_auto_join_chunked_inputs_without_grid_fails_fast."""
    da = pytest.importorskip("dask.array")
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([100.0, 200.0, 300.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([1.0, 2.0, 3.0]), chunks=2)),
        },
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    with pytest.raises(ValueError) as err:
        synchronize_param([left, right], opts=ParamSyncOptions(join="outer", how="nearest"))
    assert "pass an explicit grid" in str(err.value)


def test_param_sync_050_explicit_grid_with_chunked_inputs_allowed() -> None:
    """ID: PARAM_SYNC_050_explicit_grid_with_chunked_inputs_allowed."""
    da = pytest.importorskip("dask.array")
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([100.0, 200.0, 300.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([1.0, 2.0, 3.0]), chunks=2)),
        },
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    out = synchronize_param(
        [left, right],
        grid=xr.DataArray(np.asarray([0.0, 1.0, 2.0, 3.0]), dims=("sample",)),
        opts=ParamSyncOptions(join="outer", how="nearest"),
    )
    assert len(out) == 2
    np.testing.assert_allclose(out[0].as_dataset().coords["tau"].values, [0.0, 1.0, 2.0, 3.0])


def test_param_sync_051_single_owner_chunked_precheck_is_sync_resolve_target_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_SYNC_051_single_owner_chunked_precheck_is_sync_resolve_target_grid."""
    da = pytest.importorskip("dask.array")
    sync_mod = importlib.import_module("tal.core.param_ops.sync")
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([100.0, 200.0, 300.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([1.0, 2.0, 3.0]), chunks=2)),
        },
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")

    def _raise_precheck(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("synchronize_param: single-owner chunked precheck sentinel")

    monkeypatch.setattr(sync_mod, "require_unchunked_auto_grid_sources", _raise_precheck)
    with pytest.raises(ValueError) as err:
        synchronize_param([left, right], opts=ParamSyncOptions(join="outer", how="nearest"))
    assert "single-owner chunked precheck sentinel" in str(err.value)


def test_param_sync_052_single_input_chunked_auto_join_allowed() -> None:
    """ID: PARAM_SYNC_052_single_input_chunked_auto_join_allowed."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    out_outer = synchronize_param([ao], opts=ParamSyncOptions(join="outer", how="nearest"))[0]
    out_inner = synchronize_param([ao], opts=ParamSyncOptions(join="inner", how="nearest"))[0]
    np.testing.assert_allclose(out_outer.as_dataset().coords["tau"].values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(out_inner.as_dataset().coords["tau"].values, [0.0, 1.0, 2.0])


def test_param_sync_053_autogrid_batched_join_is_row_local_and_has_no_cross_batch_bleed() -> None:
    """ID: PARAM_SYNC_053_autogrid_batched_join_is_row_local_and_has_no_cross_batch_bleed."""
    left_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (("trial", "sample"), [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]),
            "group_size": ("trial", [3, 3]),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (("trial", "sample"), [[1.0, 2.0, 3.0], [20.0, 21.0, 22.0]]),
            "group_size": ("trial", [3, 3]),
        },
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="phase", sequence_size_coord="group_size")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="phase", sequence_size_coord="group_size")
    out_left, _ = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="inner", join="domain", how="nearest"),
    )
    phase = out_left.as_dataset().coords["phase"]
    row_a = phase.sel(trial="a").values
    row_b = phase.sel(trial="b").values
    np.testing.assert_allclose(row_a[np.isfinite(row_a)], [1.0, 2.0])
    assert np.isfinite(row_b).sum() == 0
    sizes = out_left.as_dataset().coords["group_size"]
    assert int(sizes.sel(trial="a")) == 2
    assert int(sizes.sel(trial="b")) == 0


def test_param_sync_054_autogrid_sample_invariant_param_coord_broadcast_is_batch_safe() -> None:
    """ID: PARAM_SYNC_054_autogrid_sample_invariant_param_coord_broadcast_is_batch_safe."""
    left_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2, 3],
            "phase": ("sample", [0.0, 1.0, 2.0, 3.0]),
            "group_size": ("trial", [4, 1]),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[100.0, 101.0, 102.0, 103.0], [200.0, 201.0, 202.0, 203.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2, 3],
            "phase": ("sample", [0.0, 1.0, 2.0, 3.0]),
            "group_size": ("trial", [3, 2]),
        },
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="phase", sequence_size_coord="group_size")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="phase", sequence_size_coord="group_size")
    out_left, _ = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="inner", join="outer", how="nearest"),
    )
    phase = out_left.as_dataset().coords["phase"]
    row_a = phase.sel(trial="a").values
    row_b = phase.sel(trial="b").values
    np.testing.assert_allclose(row_a[np.isfinite(row_a)], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(row_b[np.isfinite(row_b)], [0.0, 1.0])
    sizes = out_left.as_dataset().coords["group_size"]
    assert int(sizes.sel(trial="a")) == 4
    assert int(sizes.sel(trial="b")) == 2


def test_param_sync_055_autogrid_join_width_and_nan_tail_packing_are_deterministic() -> None:
    """ID: PARAM_SYNC_055_autogrid_join_width_and_nan_tail_packing_are_deterministic."""
    left_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": ("sample", [0.0, 1.0, 2.0]),
            "group_size": ("trial", [3, 1]),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": ("sample", [0.0, 2.0, 4.0]),
            "group_size": ("trial", [3, 1]),
        },
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="phase", sequence_size_coord="group_size")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="phase", sequence_size_coord="group_size")
    out_left, _ = synchronize_param(
        [left, right],
        opts=ParamSyncOptions(batch_join="inner", join="outer", how="nearest"),
    )
    phase = out_left.as_dataset().coords["phase"]
    assert phase.shape == (2, 4)
    row_b = phase.sel(trial="b").values
    np.testing.assert_allclose(row_b[0], 0.0)
    assert np.isnan(row_b[1:]).all()
    sizes = out_left.as_dataset().coords["group_size"]
    assert int(sizes.sel(trial="a")) == 4
    assert int(sizes.sel(trial="b")) == 1


def test_param_sync_056_unsafe_large_integer_autogrid_fails_closed() -> None:
    """ID: PARAM_SYNC_056_unsafe_large_integer_autogrid_fails_closed."""
    base = 2**53
    left = _integer_sync_ao(np.asarray([base, base + 1], dtype="int64"), [1.0, 2.0])
    right = _integer_sync_ao(np.asarray([base + 2, base + 3], dtype="int64"), [3.0, 4.0])
    with pytest.raises(ValueError, match="synthesized numeric grid would convert integer value"):
        synchronize_param(
            [left, right],
            opts=ParamSyncOptions(join="outer", how="nearest"),
        )


def test_param_sync_057_explicit_large_integer_grid_and_tolerance_are_exact() -> None:
    """ID: PARAM_SYNC_057_explicit_large_integer_grid_and_tolerance_are_exact."""
    maximum = np.iinfo(np.uint64).max
    source = _integer_sync_ao(np.asarray([0, maximum], dtype="uint64"), [10.0, 20.0])
    grid = np.asarray([maximum - 1, maximum], dtype="uint64")
    [out] = synchronize_param(
        [source],
        grid=grid,
        opts=ParamSyncOptions(join="override", how="fill", tol=np.uint64(1), fill_value=-1),
    )
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["tau"].values, grid)
    np.testing.assert_allclose(out.as_dataset(copy="none")["value"].values, [20.0, 20.0])
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["valid"].values, [True, True])


def test_param_sync_058_unsafe_mixed_grid_tolerance_fails_closed() -> None:
    """ID: PARAM_SYNC_058_unsafe_mixed_grid_tolerance_fails_closed."""
    runtime = importlib.import_module("tal.core.param_ops.sync_runtime")
    base = 2**53
    with pytest.raises(ValueError, match="mixed integer/float tolerance comparison"):
        runtime._numeric_within_tolerance_block(
            np.asarray([base + 3], dtype="int64"),
            np.asarray([float(base + 2)], dtype="float64"),
            np.asarray([True], dtype=bool),
            tol=1,
        )

    source = _integer_sync_ao(np.asarray([base + 3], dtype="int64"), [7.0])
    with pytest.raises(ValueError, match="integer parameter value.*cannot be represented exactly as float64"):
        synchronize_param(
            [source],
            grid=np.asarray([float(base + 2)], dtype="float64"),
            opts=ParamSyncOptions(join="override", how="fill", tol=1, fill_value=-1),
        )


def test_orch_lazy_parity_001_sync_auto_grid_chunked_failfast_stable() -> None:
    """ID: ORCH_LAZY_PARITY_001_sync_auto_grid_chunked_failfast_stable."""
    test_param_sync_049_auto_join_chunked_inputs_without_grid_fails_fast()


def test_orch_lazy_parity_002_sync_explicit_grid_chunked_allowed_stable() -> None:
    """ID: ORCH_LAZY_PARITY_002_sync_explicit_grid_chunked_allowed_stable."""
    test_param_sync_050_explicit_grid_with_chunked_inputs_allowed()


def test_orch_parity_001_param_sync_behavior_parity_after_migration() -> None:
    """ID: ORCH_PARITY_001_param_sync_behavior_parity_after_migration."""
    test_orch_topo_parity_001_sync_multi_batch_behavior_parity()
