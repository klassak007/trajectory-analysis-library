from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

import tal.core.group_ops.runtime_plan as runtime_plan_module
from tal.core.analysis_object import AnalysisObject
from tal.core.group_ops import (
    GroupByOptions,
    GroupingBinSpec,
    GroupingFoundationOptions,
    resolve_grouping_foundation_context,
)
from tal.core.group_ops.accessor import GroupedView
from tal.core.group_ops.runtime_plan import resolve_grouped_runtime_plan


def _grouping_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (("trial", "sample"), np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=float)),
        },
        coords={
            "trial": np.array(["t0", "t1"], dtype=object),
            "sample": np.array([0, 1, 2], dtype=int),
            "outcome": ("trial", np.array(["ok", "fail"], dtype=object)),
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


def _external_key(
    ao: AnalysisObject,
    values: xr.DataArray | np.ndarray,
    *,
    name: str = "ext_key",
) -> xr.DataArray:
    ds = ao.as_dataset(copy="none")
    return xr.DataArray(
        values,
        dims=("trial", "sample"),
        coords={"trial": ds.coords["trial"], "sample": ds.coords["sample"]},
        name=name,
    )


def test_group_core_p9b_001_groupby_surface_produces_deterministic_grouped_wrappers() -> None:
    """ID: GROUP_CORE_P9B_001_groupby_surface_produces_deterministic_grouped_wrappers."""
    ao = _grouping_ao()
    left = ao.group.groupby("label")
    right = ao.group.groupby("label")
    assert isinstance(left, GroupedView)
    assert isinstance(right, GroupedView)
    left_ds = left.padded().as_dataset(copy="none")
    right_ds = right.padded().as_dataset(copy="none")
    assert left_ds.identical(right_ds)


def test_group_core_p9b_004_empty_group_selection_returns_empty_grouped_output() -> None:
    """ID: GROUP_CORE_P9B_004_empty_group_selection_returns_empty_grouped_output."""
    ao = _grouping_ao()
    na_key = xr.DataArray(
        np.full((2, 3), np.nan, dtype=float),
        dims=("trial", "sample"),
        coords={"trial": ao.as_dataset(copy="none").coords["trial"], "sample": ao.as_dataset(copy="none").coords["sample"]},
        name="all_na_key",
    )
    grouped = ao.group.groupby(
        na_key,
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(na_key_policy="drop"),
        ),
    )
    out = grouped.stacked().as_dataset(copy="none")
    assert int(out.sizes["group_member"]) == 0


def test_group_hard_p9b_001_grouped_surface_fails_closed_on_alignment_or_key_mismatch() -> None:
    """ID: GROUP_HARD_P9B_001_grouped_surface_fails_closed_on_alignment_or_key_mismatch."""
    ao = _grouping_ao()
    bad = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]], dtype=float),
        dims=("trial", "sample"),
        coords={"trial": ao.as_dataset(copy="none").coords["trial"], "sample": np.array([0, 1, 99], dtype=int)},
        name="bad_key",
    )
    with pytest.raises(ValueError, match="exact|aligned|labels"):
        ao.group.groupby(bad)


def test_group_hard_p9b_003_analysis_object_does_not_gain_parallel_duplicate_grouping_api_surface() -> None:
    """ID: GROUP_HARD_P9B_003_analysis_object_does_not_gain_parallel_duplicate_grouping_api_surface."""
    ao = _grouping_ao()
    assert hasattr(ao, "group")
    assert not hasattr(ao, "groupby")
    assert not hasattr(ao, "groupby_bins")


def test_group_core_p9b_014_chunked_groupby_construction_is_lazy_until_materialization_boundary() -> None:
    """ID: GROUP_CORE_P9B_014_chunked_groupby_construction_is_lazy_until_materialization_boundary."""
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    ao = _grouping_ao()
    key = _external_key(
        ao,
        da.from_array(
            np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 2.0]], dtype=float),
            chunks=(1, 3),
        ),
        name="chunked_lazy_key",
    )
    calls = {"count": 0}

    class _TaskCounter(Callback):
        def _pretask(self, key, dsk, state):  # type: ignore[override]
            calls["count"] += 1

    with _TaskCounter():
        grouped = ao.group.groupby(key)
    assert calls["count"] == 0

    calls["count"] = 0
    with _TaskCounter():
        _ = grouped.padded()
    assert calls["count"] > 0


def test_group_core_p9b_015_chunked_default_na_error_without_na_succeeds_at_materialization() -> None:
    """ID: GROUP_CORE_P9B_015_chunked_default_na_error_without_na_succeeds_at_materialization."""
    da = pytest.importorskip("dask.array")
    ao = _grouping_ao()
    key = _external_key(
        ao,
        da.from_array(
            np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 2.0]], dtype=float),
            chunks=(1, 3),
        ),
        name="chunked_no_na_key",
    )
    out = ao.group.groupby(key).padded().as_dataset(copy="none")
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array([1.0, 2.0], dtype=float))


def test_group_hard_p9b_006_chunked_default_na_error_with_na_fails_at_materialization_boundary() -> None:
    """ID: GROUP_HARD_P9B_006_chunked_default_na_error_with_na_fails_at_materialization_boundary."""
    da = pytest.importorskip("dask.array")
    ao = _grouping_ao()
    key = _external_key(
        ao,
        da.from_array(
            np.array([[1.0, np.nan, 2.0], [2.0, 1.0, 2.0]], dtype=float),
            chunks=(1, 3),
        ),
        name="chunked_na_key",
    )
    grouped = ao.group.groupby(key)
    with pytest.raises(ValueError, match="na_key_policy='error'"):
        _ = grouped.padded()


@pytest.mark.parametrize(
    "labels",
    (
        ("A", "A", "B"),
        (float("nan"), float("nan"), "B"),
    ),
)
def test_group_hard_p9b_007_duplicate_bin_labels_fail_closed_at_groupby_construction(
    labels: tuple[object, object, object],
) -> None:
    """ID: GROUP_HARD_P9B_007_duplicate_bin_labels_fail_closed_at_groupby_construction."""
    ao = _grouping_ao()
    with pytest.raises(ValueError, match="labels must be unique|duplicate label"):
        _ = ao.group.groupby_bins(
            "time_s",
            bins=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float),
            labels=labels,
        )


@pytest.mark.parametrize(
    "domain_order",
    (
        ("dup", "dup", "unique"),
        (float("nan"), float("nan"), "unique"),
    ),
)
def test_group_hard_p9b_008_runtime_defensive_guard_rejects_duplicate_domain_order_labels(
    domain_order: tuple[object, object, object],
) -> None:
    """ID: GROUP_HARD_P9B_008_runtime_defensive_guard_rejects_duplicate_domain_order_labels."""
    ao = _grouping_ao()
    context = resolve_grouping_foundation_context(
        ao,
        GroupingBinSpec(
            source="time_s",
            bins=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float),
            labels=["low", "mid", "high"],
        ),
        owner="grouping.test",
    )
    bad_key = replace(context.keys[0], domain_order=domain_order)
    bad_context = replace(context, keys=(bad_key,))
    with pytest.raises(ValueError, match="domain_order contains duplicate label"):
        _ = resolve_grouped_runtime_plan(bad_context, opts=GroupByOptions(), owner="grouping.runtime")


def test_group_core_p9b_016_runtime_owner_reuse_probe_avoids_dense_reference_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: GROUP_CORE_P9B_016_runtime_owner_reuse_probe_avoids_dense_reference_allocation."""
    ao = _grouping_ao()
    grouped = ao.group.groupby("label")

    def _raise_dense_zeros(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("runtime owner-reuse probe must not allocate dense np.zeros payloads")

    monkeypatch.setattr(runtime_plan_module.np, "zeros", _raise_dense_zeros)
    _ = grouped._resolve_plan(owner="grouping.runtime")


def test_group_core_p9b_017_transposed_grouping_key_dims_are_accepted_by_name_alignment() -> None:
    """ID: GROUP_CORE_P9B_017_transposed_grouping_key_dims_are_accepted_by_name_alignment."""
    ao = _grouping_ao()
    ds = ao.as_dataset(copy="none")
    transposed_key = xr.DataArray(
        np.array([["A", "A"], ["C", "B"], ["A", "B"]], dtype=object),
        dims=("sample", "trial"),
        coords={"trial": ds.coords["trial"], "sample": ds.coords["sample"]},
        name="label_transposed",
    )
    out = ao.group.groupby(transposed_key).padded().as_dataset(copy="none")
    np.testing.assert_array_equal(
        out.coords["group_key"].to_numpy(),
        np.array(["A", "C", "B"], dtype=object),
    )


def test_group_hard_p9b_009_group_dim_collision_with_source_namespace_fails_closed() -> None:
    """ID: GROUP_HARD_P9B_009_group_dim_collision_with_source_namespace_fails_closed."""
    ao = _grouping_ao()
    grouped = ao.group.groupby(
        "label",
        opts=GroupByOptions(preserve_batch=True, group_dim="trial"),
    )
    with pytest.raises(ValueError, match="opts.group_dim='trial'|collides with source namespace"):
        _ = grouped.padded()


def test_group_hard_p9b_010_sequence_index_coord_collision_with_source_namespace_fails_closed() -> None:
    """ID: GROUP_HARD_P9B_010_sequence_index_coord_collision_with_source_namespace_fails_closed."""
    ao = _grouping_ao()
    grouped = ao.group.groupby(
        "label",
        opts=GroupByOptions(preserve_batch=True, sequence_index_coord="trial"),
    )
    with pytest.raises(
        ValueError,
        match="opts.sequence_index_coord='trial'|collides with source namespace",
    ):
        _ = grouped.stacked()


def test_group_hard_p9b_011_runtime_plan_unhashable_observed_labels_fail_closed() -> None:
    """ID: GROUP_HARD_P9B_011_runtime_plan_unhashable_observed_labels_fail_closed."""
    ao = _grouping_ao()
    ds = ao.as_dataset(copy="none")
    values = np.empty((2, 3), dtype=object)
    values[0, :] = ([1], [2], [3])
    values[1, :] = ([1], [2], [3])
    key = xr.DataArray(
        values,
        dims=("trial", "sample"),
        coords={"trial": ds.coords["trial"], "sample": ds.coords["sample"]},
        name="unhashable_key",
    )
    grouped = ao.group.groupby(key)
    with pytest.raises(ValueError, match="unhashable group label|labels must be hashable"):
        _ = grouped.padded()


def test_groupby_bins_sequence_only_source_fails_closed_and_broadcasted_source_succeeds() -> None:
    ds = _grouping_ao().as_dataset(copy="none").assign_coords(time=("sample", np.array([0.0, 1.0, 2.0], dtype=float)))
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
    )
    with pytest.raises(ValueError, match="row dims|missing="):
        _ = ao.group.groupby_bins("time", bins=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float))

    time_row = xr.broadcast(ao.as_dataset(copy="none").coords["time"], ao.as_dataset(copy="none")["signal"])[0].rename("time_row")
    out = ao.group.groupby_bins(
        time_row,
        bins=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float),
    ).mean(validate=True).as_dataset(copy="none")
    assert int(out.sizes["group_key"]) == 3


def test_groupby_accepts_batch_only_coord_key() -> None:
    ao = _grouping_ao()
    out = ao.group.groupby("outcome").mean(validate=True).as_dataset(copy="none")
    np.testing.assert_array_equal(out.coords["group_key"].to_numpy(), np.array(["ok", "fail"], dtype=object))
