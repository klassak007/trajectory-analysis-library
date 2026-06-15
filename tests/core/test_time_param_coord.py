import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal.core import (
    AnalysisObject,
    ParamEvalOptions,
    ParamSelectOptions,
    ParamSyncOptions,
    SchemaError,
    synchronize,
    synchronize_param,
)
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.param_engine import ParamMapOptions, build_param_bounds_map, build_param_map, normalize_query_grid
from tal.core.param_ops.sync_runtime import align_contexts_batch


def _dt(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]")


def _datetime_ao(
    *,
    times: np.ndarray | None = None,
    values: list[float] | None = None,
) -> AnalysisObject:
    coord = times if times is not None else _dt(["2026-01-01T00:00:00", "2026-01-01T00:00:10", "2026-01-01T00:00:20"])
    payload = values if values is not None else [0.0, 10.0, 40.0]
    ds = xr.Dataset(
        {"value": ("sample", np.asarray(payload, dtype="float64"))},
        coords={"sample": np.arange(len(payload), dtype="int64"), "time": ("sample", coord)},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), param_coord="time")


def _datetime_batched_ao(label: str, times: list[str], values: list[float]) -> AnalysisObject:
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.asarray([values], dtype="float64"))},
        coords={
            "trial": [label],
            "sample": np.arange(len(values), dtype="int64"),
            "time": (("trial", "sample"), [_dt(times)]),
        },
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=(), param_coord="time")


def _numeric_ao() -> AnalysisObject:
    ds = xr.Dataset(
        {"value": ("sample", [0.0, 10.0, 20.0])},
        coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), param_coord="time")


def test_time_core_t1_001_datetime64_param_coord_schema_validates() -> None:
    """ID: TIME_CORE_T1_001_datetime64_param_coord_schema_validates."""
    ao = _datetime_ao()
    assert np.issubdtype(ao.unsafe_data.coords["time"].dtype, np.datetime64)
    assert ao.unsafe_data.attrs["tal"]["core"]["param_coord"]["name"] == "time"


def test_time_core_t1_002_numeric_param_behavior_remains_numeric() -> None:
    """ID: TIME_CORE_T1_002_numeric_param_behavior_remains_numeric."""
    ctx = resolve_param_runtime_context(_numeric_ao(), on="time")
    assert ctx.param_kind == "numeric"
    np.testing.assert_array_equal(_numeric_ao().param.index([0.1, 1.9]).values, [0, 2])


def test_time_core_t1_003_param_runtime_records_datetime64_kind() -> None:
    """ID: TIME_CORE_T1_003_param_runtime_records_param_kind."""
    ctx = resolve_param_runtime_context(_datetime_ao(), on="time")
    assert ctx.param_kind == "datetime64"


def test_time_core_t1_004_datetime64_param_selection_accepts_datetime_queries() -> None:
    """ID: TIME_CORE_T1_004_datetime64_param_selection_accepts_datetime_queries."""
    ao = _datetime_ao()
    q0 = np.datetime64("2026-01-01T00:00:03", "ns")
    q1 = pd.Timestamp("2026-01-01T00:00:18")
    np.testing.assert_array_equal(ao.param.index([q0, q1], on="time").values, [0, 2])
    selected = ao.param.sel(xr.DataArray([q0, q1], dims=("query",)), on="time", opts=ParamSelectOptions())
    np.testing.assert_allclose(selected.unsafe_data["value"].values, [0.0, 40.0])


def test_time_core_t1_005_datetime64_param_interpolation_uses_local_deltas() -> None:
    """ID: TIME_CORE_T1_005_datetime64_param_interpolation_uses_local_deltas."""
    base = np.datetime64("2262-04-11T00:00:00.000000000", "ns")
    ao = _datetime_ao(times=base + np.asarray([0, 1, 2], dtype="timedelta64[ns]"), values=[0.0, 1.0, 2.0])
    out = ao.param.at([base + np.timedelta64(1, "ns")], on="time", opts=ParamEvalOptions(method="linear"))
    assert float(out.unsafe_data["value"].item()) == 1.0


def test_time_core_t1_006_datetime64_query_normalization_owner() -> None:
    """ID: TIME_CORE_T1_006_datetime64_query_normalization_owner."""
    query = normalize_query_grid(
        [pd.Timestamp("2026-01-01T00:00:00"), np.datetime64("2026-01-01T00:00:01", "ns")],
        query_dim="query",
        param_kind="datetime64",
    )
    assert str(query.values.dtype) == "datetime64[ns]"
    with pytest.raises(ValueError, match="datetime64 param queries must be datetime-like"):
        normalize_query_grid([1.0, 2.0], query_dim="query", param_kind="datetime64")


def test_time_core_t1_007_datetime64_sync_accepts_timedelta_tolerance() -> None:
    """ID: TIME_CORE_T1_007_datetime64_sync_accepts_timedelta_tolerance."""
    left = _datetime_ao(times=_dt(["2026-01-01T00:00:00", "2026-01-01T00:00:10"]), values=[0.0, 10.0])
    right = _datetime_ao(times=_dt(["2026-01-01T00:00:05", "2026-01-01T00:00:10"]), values=[5.0, 10.0])
    out = synchronize_param(
        [left, right],
        on="time",
        opts=ParamSyncOptions(join="outer", how="fill", tol=dt.timedelta(seconds=0)),
    )
    np.testing.assert_array_equal(
        out[0].unsafe_data.coords["time"].values,
        _dt(["2026-01-01T00:00:00", "2026-01-01T00:00:05", "2026-01-01T00:00:10"]),
    )
    np.testing.assert_allclose(out[0].unsafe_data["value"].values, [0.0, np.nan, 10.0])
    np.testing.assert_allclose(out[1].unsafe_data["value"].values, [np.nan, 5.0, 10.0])


def test_time_core_t1_008_datetime64_entrypoints_share_param_kind() -> None:
    """ID: TIME_CORE_T1_008_datetime64_entrypoints_share_param_kind."""
    source = _datetime_ao()
    target = _datetime_ao(
        times=_dt(["2026-01-01T00:00:00", "2026-01-01T00:00:20"]),
        values=[100.0, 200.0],
    )
    grid = [np.datetime64("2026-01-01T00:00:00", "ns"), np.datetime64("2026-01-01T00:00:20", "ns")]
    np.testing.assert_allclose(source.param.resample_to(grid, on="time").unsafe_data["value"].values, [0.0, 40.0])
    np.testing.assert_allclose(source.param.interp_like(target, on="time").unsafe_data["value"].values, [0.0, 40.0])
    np.testing.assert_allclose(
        synchronize([source], on="time", grid=grid, opts=ParamSyncOptions(join="override"))[0].unsafe_data["value"],
        [0.0, 40.0],
    )


def test_time_hard_t1_001_object_datetime_param_coord_rejected() -> None:
    """ID: TIME_HARD_T1_001_object_datetime_param_coord_rejected."""
    values = np.asarray([dt.datetime(2026, 1, 1), "not-a-datetime64"], dtype=object)
    ds = xr.Dataset(
        {"value": ("sample", [0.0, 1.0])},
        coords={"sample": [0, 1], "time": ("sample", values)},
    )
    with pytest.raises(SchemaError) as err:
        AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), param_coord="time")
    assert err.value.code == "schema.param_coord.dtype.invalid"


def test_time_hard_t1_002_datetime64_numeric_tolerance_mismatch_fails_closed() -> None:
    """ID: TIME_HARD_T1_002_datetime64_numeric_tolerance_mismatch_fails_closed."""
    with pytest.raises(ValueError, match="nonzero numeric opts.tol"):
        synchronize_param(
            [_datetime_ao()],
            on="time",
            grid=[np.datetime64("2026-01-01T00:00:00", "ns")],
            opts=ParamSyncOptions(join="override", tol=1.0),
        )


def test_time_hard_t1_003_numeric_sync_rejects_timedelta_tolerance() -> None:
    """ID: TIME_HARD_T1_003_numeric_sync_rejects_timedelta_tolerance."""
    with pytest.raises(ValueError, match="timedelta opts.tol is only valid for datetime64"):
        synchronize_param([_numeric_ao()], opts=ParamSyncOptions(join="left", tol=np.timedelta64(1, "s")))


def test_time_hard_t1_004_datetime64_nat_queries_are_invalid_maps() -> None:
    """ID: TIME_HARD_T1_004_datetime64_nat_queries_are_invalid_maps."""
    ctx = resolve_param_runtime_context(_datetime_ao(), on="time")
    query = normalize_query_grid([np.datetime64("NaT", "ns")], query_dim="query", param_kind=ctx.param_kind)
    pmap = build_param_map(
        param=ctx.spec.coord,
        query=query.values,
        sequence_dim=ctx.sequence_dim,
        query_dim=query.query_dim,
        valid_mask=ctx.valid_mask,
        options=ParamMapOptions(method="nearest"),
        param_kind=ctx.param_kind,
    )
    np.testing.assert_array_equal(pmap.valid.values, [False])


def test_time_hard_t1_005_mixed_param_kind_sync_fails_closed() -> None:
    """ID: TIME_HARD_T1_005_mixed_param_kind_sync_fails_closed."""
    with pytest.raises(ValueError, match="same param coordinate kind"):
        synchronize_param([_datetime_ao(), _numeric_ao()], on="time", opts=ParamSyncOptions(join="outer"))


def test_time_hard_t1_006_datetime64_slice_bounds_and_open_bounds() -> None:
    """ID: TIME_HARD_T1_006_datetime64_slice_bounds_and_open_bounds."""
    ctx = resolve_param_runtime_context(_datetime_ao(), on="time")
    lower = build_param_bounds_map(
        param=ctx.spec.coord,
        start=np.datetime64("2026-01-01T00:00:05", "ns"),
        stop=None,
        sequence_dim=ctx.sequence_dim,
        valid_mask=ctx.valid_mask,
        param_kind=ctx.param_kind,
    )
    upper = build_param_bounds_map(
        param=ctx.spec.coord,
        start=None,
        stop=np.datetime64("2026-01-01T00:00:10", "ns"),
        sequence_dim=ctx.sequence_dim,
        valid_mask=ctx.valid_mask,
        param_kind=ctx.param_kind,
    )
    assert int(lower.i0.item()) == 1
    assert int(lower.i1.item()) == 3
    assert int(upper.i0.item()) == 0
    assert int(upper.i1.item()) == 2


def test_time_hard_t1_007_datetime64_outer_batch_sync_uses_nat_reindex_fill() -> None:
    """ID: TIME_HARD_T1_007_datetime64_outer_batch_sync_uses_nat_reindex_fill."""
    left = _datetime_batched_ao(
        "a",
        ["2026-01-01T00:00:00", "2026-01-01T00:00:10"],
        [0.0, 10.0],
    )
    right = _datetime_batched_ao(
        "b",
        ["2026-01-01T00:00:05", "2026-01-01T00:00:15"],
        [5.0, 15.0],
    )
    left_ctx = resolve_param_runtime_context(left, on="time")
    right_ctx = resolve_param_runtime_context(right, on="time")
    aligned_left, aligned_right = align_contexts_batch([left_ctx, right_ctx], mode="outer")
    assert np.isnat(aligned_left.spec.coord.sel(trial="b").values).all()
    assert np.isnat(aligned_right.spec.coord.sel(trial="a").values).all()
    np.testing.assert_array_equal(aligned_left.valid_mask.sel(trial="b").values, [False, False])
    np.testing.assert_array_equal(aligned_right.valid_mask.sel(trial="a").values, [False, False])

    out_left, out_right = synchronize_param(
        [left, right],
        on="time",
        opts=ParamSyncOptions(batch_join="outer", join="outer", how="nearest"),
    )
    assert list(out_left.unsafe_data.coords["trial"].values) == ["a", "b"]
    assert list(out_right.unsafe_data.coords["trial"].values) == ["a", "b"]
    assert str(out_left.unsafe_data.coords["time"].dtype) == "datetime64[ns]"
    assert str(out_right.unsafe_data.coords["time"].dtype) == "datetime64[ns]"
    np.testing.assert_array_equal(out_left.unsafe_data.coords["valid"].sel(trial="b").values, [False, False])
    np.testing.assert_array_equal(out_right.unsafe_data.coords["valid"].sel(trial="a").values, [False, False])
