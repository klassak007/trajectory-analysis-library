from __future__ import annotations

import gc
import tracemalloc
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import AnalysisObject, GroupByOptions, GroupMaterializeOptions
from tal.core.group_ops import GroupingFoundationOptions


def _temporal_values(kind: str) -> np.ndarray:
    if kind == "datetime":
        return np.asarray(["2020-01-01", "NaT", "NaT"], dtype="datetime64[ns]")
    return np.asarray([1, "NaT", "NaT"], dtype="timedelta64[ns]")


def _sequence_with_temporal_na(kind: str) -> AnalysisObject:
    values = _temporal_values(kind)
    return AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[1.0, 2.0, 999.0]])},
            coords={
                "trial": ["run"],
                "sample": [0, 1, 2],
                "when": (("trial", "sample"), values[None, :]),
                "kind": (("trial", "sample"), [["x", "y", "ignored"]]),
                "group_size": ("trial", [2]),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="group_size",
    )


@pytest.mark.parametrize("kind", ["datetime", "timedelta"])
@pytest.mark.parametrize("preserve_batch", [False, True])
@pytest.mark.parametrize("multi_key", [False, True])
def test_sequence_na_groups_preserve_temporal_and_tuple_scalars(
    kind: str,
    preserve_batch: bool,
    multi_key: bool,
) -> None:
    """ID: GROUP_LABEL_052_na_group_scalar_parity."""
    source = _sequence_with_temporal_na(kind)
    missing = ("missing", 0)
    key = ("when", "kind") if multi_key else "when"
    grouped = source.group.groupby(
        key,
        opts=GroupByOptions(
            preserve_batch=preserve_batch,
            foundation_opts=GroupingFoundationOptions(
                na_key_policy="group",
                na_group_label=missing,
            ),
        ),
    )
    actual = grouped.mean().as_dataset(copy="none")
    first = _temporal_values(kind)[0]
    expected = ((first, "x"), (missing, "y")) if multi_key else (first, missing)
    assert actual.coords["group_key"].data.tolist() == list(expected)
    assert isinstance(actual.coords["group_key"].data[0][0] if multi_key else actual.coords["group_key"].data[0], (np.datetime64, np.timedelta64))
    expected_values = [[1.0, 2.0]] if preserve_batch else [1.0, 2.0]
    np.testing.assert_allclose(actual["value"], expected_values)


@pytest.mark.parametrize("preserve_batch", [False, True])
def test_sequence_na_group_collision_uses_scalar_label_identity(preserve_batch: bool) -> None:
    """ID: GROUP_LABEL_053_na_group_collision_scalar_identity."""
    source = _sequence_with_temporal_na("datetime")
    with pytest.raises(ValueError, match="na_group_label=.*collides"):
        source.group.groupby(
            "when",
            opts=GroupByOptions(
                preserve_batch=preserve_batch,
                foundation_opts=GroupingFoundationOptions(
                    na_key_policy="group",
                    na_group_label=np.datetime64("2020-01-01", "ns"),
                ),
            ),
        )


def _empty_sequence_source(batch_shape: tuple[int, ...], *, lazy: bool) -> AnalysisObject:
    batch_dims = ("trial",) if len(batch_shape) == 1 else ("trial", "lane")
    dims = batch_dims + ("sample",)
    shape = batch_shape + (3,)
    chunks = tuple(max(1, size) for size in shape)
    values = da.zeros(shape, chunks=chunks) if lazy else np.zeros(shape)
    coords: dict[str, object] = {
        dim: np.arange(size) for dim, size in zip(batch_dims, batch_shape, strict=True)
    }
    coords["sample"] = np.arange(3)
    coords["score"] = (dims, np.empty(shape, dtype=float))
    coords["batch_meta"] = (batch_dims, np.empty(batch_shape, dtype=float))
    return AnalysisObject.from_data(
        xr.Dataset(
            {
                "value": (dims, values),
                "batch_static": (batch_dims, np.empty(batch_shape, dtype=float)),
                "constant": 2.0,
            },
            coords=coords,
        ),
        sequence_dim="sample",
        batch_dims=batch_dims,
        core_dims=(),
    )


@pytest.mark.parametrize("batch_shape", [(0,), (2, 0)])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("include_empty", [False, True])
def test_preserved_batch_empty_axes_keep_resolved_topology(
    batch_shape: tuple[int, ...],
    lazy: bool,
    include_empty: bool,
) -> None:
    """ID: GROUP_EMPTY_054_preserved_batch_zero_axis_topology."""
    source = _empty_sequence_source(batch_shape, lazy=lazy)
    grouped = source.group.groupby_bins(
        "score",
        [0.0, 1.0, 2.0],
        labels=["low", "high"],
        opts=GroupByOptions(preserve_batch=True),
    )
    options = GroupMaterializeOptions(include_empty_groups=include_empty)
    padded = grouped.padded(opts=options).as_dataset(copy="none")
    expected_groups = 2 if include_empty else 0
    assert padded.sizes == {
        **dict(zip(source.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["batch_dims"], batch_shape, strict=True)),
        "group_key": expected_groups,
        "sample": 3,
    }
    expected_labels = ["low", "high"] if include_empty else []
    np.testing.assert_array_equal(padded.coords["group_key"], expected_labels)
    assert padded.attrs["tal"]["core"]["validity"] == {
        "sequence_size_coord": "group_size",
        "layout": "left_packed",
    }
    assert padded.coords["group_size"].shape == (*batch_shape, expected_groups)
    assert isinstance(padded["value"].data, da.Array) is lazy
    batch_dims = tuple(padded.attrs["tal"]["core"]["roles"]["batch_dims"][:-1])
    assert padded["batch_static"].dims == (*batch_dims, "group_key", "sample")
    assert padded.coords["batch_meta"].dims == batch_dims
    assert padded["constant"].dims == (*batch_dims, "group_key", "sample")
    stacked = grouped.stacked().as_dataset(copy="none")
    assert stacked.sizes["group_member"] == 0
    assert stacked["batch_static"].dims == (*batch_dims, "group_member")
    assert stacked["constant"].dims == (*batch_dims, "group_member")
    reduced = grouped.mean(opts=options).as_dataset(copy="none")
    assert reduced.sizes["group_key"] == expected_groups
    assert isinstance(reduced["value"].data, da.Array) is lazy
    assert reduced["batch_static"].dims == (*batch_dims, "group_key")
    assert reduced["constant"].dims == (*batch_dims, "group_key")
    reduced.compute()


class _NoEvaluateTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size: int) -> None:
        super().__init__(("trial",), {"trial": size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("positional ndarray weights must not evaluate indexes")

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("positional ndarray weights must not evaluate indexes")

    def equals(self, other: object, **kwargs: object) -> bool:
        return isinstance(other, _NoEvaluateTransform) and self.dim_size == other.dim_size


def _native_weight_source(kind: str) -> tuple[AnalysisObject, xr.Coordinates]:
    lane_index: xr.Index
    if kind == "range":
        lane_index = xr.indexes.RangeIndex.arange(3, dim="trial")
    else:
        lane_index = xr.indexes.CoordinateTransformIndex(_NoEvaluateTransform(3))
    lane = xr.Coordinates.from_xindex(lane_index)
    axis_index = xr.indexes.RangeIndex.arange(2, dim="axis")
    coords = lane.merge(xr.Coordinates.from_xindex(axis_index)).coords
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "axis"), np.arange(1.0, 7.0).reshape(3, 2))},
            coords=coords,
        ),
        batch_dims=("trial",),
        core_dims=("axis",),
    )
    return source, lane


@pytest.mark.parametrize("kind", ["range", "transform"])
@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_positional_ndarray_weights_preserve_native_indexes(
    kind: str,
    mapping: bool,
    grouped: bool,
) -> None:
    """ID: REDUCE_WEIGHT_055_positional_ndarray_native_indexes."""
    source, lane = _native_weight_source(kind)
    raw = np.asarray([1.0, 2.0, 1.0])
    weights = {"trial": raw} if mapping else raw
    if grouped:
        key = xr.DataArray(["a", "a", "b"], dims="trial", coords=lane)
        actual = source.group.groupby(key).mean(weights=weights).as_dataset(copy="none")
        np.testing.assert_allclose(actual["value"], [[7.0 / 3.0, 10.0 / 3.0], [5.0, 6.0]])
    else:
        actual = source.mean(dim="trial", weights=weights).as_dataset(copy="none")
        np.testing.assert_allclose(actual["value"], [3.0, 4.0])
    assert isinstance(actual.xindexes["axis"], xr.indexes.RangeIndex)


def _allocation_fixture(trials: int, width: int):
    source = AnalysisObject.from_data(
        xr.Dataset(
            {
                "value": (
                    ("trial", "lane"),
                    da.zeros((trials, width), chunks=(max(1, trials), width)),
                )
            },
            coords={"trial": np.arange(trials), "lane": np.arange(width)},
        ),
        batch_dims=("trial", "lane"),
        core_dims=(),
    )
    key = xr.DataArray(
        np.full(trials, np.nan),
        dims="trial",
        coords={"trial": np.arange(trials)},
    )
    grouped = source.group.groupby(
        key,
        opts=GroupByOptions(
            foundation_opts=GroupingFoundationOptions(na_key_policy="drop")
        ),
    )
    return grouped, {"trial": np.ones(trials), "lane": np.ones(width)}


def _grouped_weight_peak(trials: int, width: int) -> int:
    grouped, weights = _allocation_fixture(trials, width)
    gc.collect()
    tracemalloc.start()
    try:
        result = grouped.mean(dim=("trial", "lane"), weights=weights)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.as_dataset(copy="none").sizes["group_key"] == 0
    return peak


def test_weight_alignment_preflight_does_not_build_cartesian_factors() -> None:
    """ID: REDUCE_ALLOC_056_weight_alignment_factor_scaling."""
    _grouped_weight_peak(8, 32)
    small = _grouped_weight_peak(32, 2048)
    large = _grouped_weight_peak(2048, 2048)
    assert large - small < 4 * 1024**2, (small, large)


def _disjoint_weight_source(calls: list[str]) -> AnalysisObject:
    @delayed
    def key_values() -> np.ndarray:
        calls.append("key")
        return np.asarray(["a", "b"], dtype=object)

    return AnalysisObject.from_data(
        xr.Dataset(
            {
                "by_trial": ("trial", [1.0, 3.0]),
                "by_axis": ("axis", [10.0, 20.0]),
                "key": (
                    "trial",
                    da.from_delayed(key_values(), shape=(2,), dtype=object),
                ),
            },
            coords={"trial": [0, 1], "axis": ["x", "y"]},
        ),
        batch_dims=("trial",),
        core_dims=("axis",),
    )


def test_ndarray_weight_dimension_is_resolved_for_the_whole_request() -> None:
    """ID: REDUCE_WEIGHT_063_request_wide_ndarray_dimension."""
    calls: list[str] = []
    source = _disjoint_weight_source(calls)
    dims = ("trial", "axis")
    ndarray = np.asarray([1.0, 3.0])

    with pytest.raises(ValueError, match="exactly one active reduced payload dimension"):
        source.mean(dim=dims, weights=ndarray)
    grouped = source.group.groupby("key")
    with pytest.raises(ValueError, match="exactly one active reduced payload dimension"):
        grouped.mean(dim=dims, weights=ndarray)
    assert calls == []

    mapping = {"trial": ndarray, "axis": ndarray}
    direct = source.mean(dim=dims, weights=mapping).as_dataset(copy="none")
    assert direct["by_trial"].item() == 2.5
    assert direct["by_axis"].item() == 17.5

    actual = grouped.mean(dim=dims, weights=mapping).as_dataset(copy="none")
    np.testing.assert_array_equal(actual.coords["group_key"], ["a", "b"])
    np.testing.assert_allclose(actual["by_trial"], [1.0, 3.0])
    assert actual["by_axis"].dims == ()
    assert actual["by_axis"].item() == 17.5
    assert calls == ["key"]


def _inactive_weight_source(calls: list[str]) -> AnalysisObject:
    @delayed
    def key_values() -> np.ndarray:
        calls.append("key")
        return np.asarray(["a", "b"], dtype=object)

    return AnalysisObject.from_data(
        xr.Dataset(
            {
                "value": ("axis", [1.0, 2.0]),
                "key": (
                    "trial",
                    da.from_delayed(key_values(), shape=(2,), dtype=object),
                ),
            },
            coords={"trial": [0, 1], "axis": ["x", "y"]},
        ),
        batch_dims=("trial",),
        core_dims=("axis",),
    )


def test_weights_require_an_active_reduced_payload_dimension() -> None:
    """ID: REDUCE_WEIGHT_064_no_active_payload_dimension."""
    calls: list[str] = []
    source = _inactive_weight_source(calls)
    cases = (
        np.ones(2),
        xr.DataArray(np.ones(2), dims="trial"),
        {"trial": np.ones(2)},
    )

    for weights in cases:
        with pytest.raises(ValueError, match="at least one reduced payload dimension"):
            source.mean(dim="trial", weights=weights)
        with pytest.raises(ValueError, match="at least one reduced payload dimension"):
            source.mean(dim=(), weights=weights)
        with pytest.raises(ValueError, match="at least one reduced payload dimension"):
            source.group.groupby("key").mean(weights=weights)
        assert calls == []

    sequence = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", [1.0, 2.0])},
            coords={"sample": [0, 1], "key": ("sample", ["a", "b"])},
        ),
        sequence_dim="sample",
        core_dims=(),
    )
    with pytest.raises(ValueError, match="at least one reduced payload dimension"):
        sequence.group.groupby("key").mean(dim=(), weights=np.ones(2))
