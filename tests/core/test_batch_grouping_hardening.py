from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import (
    AnalysisObject,
    BatchGroupReduceOptions,
    GroupByOptions,
    GroupMaterializeOptions,
)
from tal.spatial import Rotation


def _indexed_batch_source(coords: xr.Coordinates) -> AnalysisObject:
    ds = xr.Dataset(
        {"value": ("trial", [1.0, 2.0, 3.0])},
        coords=coords,
    )
    return AnalysisObject.from_data(
        ds,
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )


def _renamed_index_key(labels: tuple[str, ...]) -> xr.DataArray:
    return xr.DataArray(
        ["a", "b", "a"],
        dims=("trial",),
        coords={"label": ("trial", list(labels))},
    ).set_xindex("label")


class _OffsetCoordinateTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size: int, *, offset: float, dim: str = "trial") -> None:
        self.dim = dim
        self.offset = offset
        super().__init__((dim,), {dim: size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        return {self.dim: dim_positions[self.dim] + self.offset}

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        return {self.dim: coord_labels[self.dim] - self.offset}

    def equals(self, other: object, **kwargs: object) -> bool:
        return (
            isinstance(other, _OffsetCoordinateTransform)
            and self.dim_size == other.dim_size
            and self.dim == other.dim
            and self.offset == other.offset
        )


def _transform_key(*, offset: float) -> xr.DataArray:
    index = xr.indexes.CoordinateTransformIndex(
        _OffsetCoordinateTransform(3, offset=offset)
    )
    return xr.DataArray(
        ["a", "b", "a"],
        dims=("trial",),
        coords=xr.Coordinates.from_xindex(index),
    )


def test_batch_grouping_exact_alignment_observes_renamed_index_groups() -> None:
    """ID: BATCH_GROUP_017_renamed_index_groups_align_exactly."""
    source = _indexed_batch_source(
        xr.Dataset(
            coords={"label": ("trial", ["x", "y", "z"])}
        ).set_xindex("label").coords
    )

    actual = source.group.groupby(
        _renamed_index_key(("x", "y", "z"))
    ).sum()
    assert actual.as_dataset(copy="none").coords["group_key"].values.tolist() == [
        "a",
        "b",
    ]
    with pytest.raises(ValueError, match="index must exactly match"):
        source.group.groupby(_renamed_index_key(("z", "y", "x")))
    with pytest.raises(ValueError, match="cannot mix indexed and unindexed"):
        source.group.groupby(xr.DataArray(["a", "b", "a"], dims=("trial",)))


def test_batch_grouping_exact_alignment_observes_all_lane_index_groups() -> None:
    """ID: BATCH_GROUP_018_all_primary_lane_index_groups_are_compared."""
    source_ds = xr.Dataset(
        {
            "value": ("trial", [1.0, 2.0, 3.0]),
            "key": ("trial", ["a", "b", "a"]),
        },
        coords={
            "trial": ["t0", "t1", "t2"],
            "label": ("trial", ["x", "y", "z"]),
        },
    ).set_xindex("label")
    source = AnalysisObject.from_data(
        source_ds,
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )

    assert source.group.groupby(source_ds["key"]).sum().as_dataset(
        copy="none"
    ).sizes["group_key"] == 2
    missing_auxiliary = source_ds["key"].drop_indexes("label").drop_vars("label")
    with pytest.raises(ValueError, match="relevant coordinate topology differs"):
        source.group.groupby(missing_auxiliary)


def test_batch_grouping_exact_alignment_supports_coordinate_transform_index() -> None:
    """ID: BATCH_GROUP_019_coordinate_transform_indexes_align_exactly."""
    source = _indexed_batch_source(_transform_key(offset=0.5).coords)

    assert source.group.groupby(_transform_key(offset=0.5)).mean().as_dataset(
        copy="none"
    ).sizes["group_key"] == 2
    with pytest.raises(ValueError, match="index must exactly match"):
        source.group.groupby(_transform_key(offset=10.5))


def _sequence_bin_source() -> AnalysisObject:
    return AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "sample"), [[1.0, 2.0], [3.0, 4.0]])},
            coords={
                "trial": ["a", "b"],
                "sample": [0, 1],
                "score": (("trial", "sample"), [[0.2, 1.2], [0.4, 1.4]]),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )


def test_tuple_bin_labels_remain_scalar_in_sequence_and_batch_outputs() -> None:
    """ID: BATCH_GROUP_020_tuple_bin_labels_remain_scalar_objects."""
    labels = (("low", 0), ("high", 1), ("empty", 2))
    sequence = _sequence_bin_source().group.groupby_bins(
        "score",
        bins=(0.0, 1.0, 2.0, 3.0),
        labels=labels,
    )
    padded = sequence.padded().as_dataset(copy="none")
    stacked = sequence.stacked().as_dataset(copy="none")
    assert padded.coords["group_key"].values.tolist() == list(labels)
    stacked_labels = stacked.coords["group_key"].values.tolist()
    assert all(isinstance(label, tuple) for label in stacked_labels)
    assert list(dict.fromkeys(stacked_labels)) == list(labels[:2])

    batch_source = _indexed_batch_source(
        xr.Coordinates(
            {
                "trial": ["t0", "t1", "t2"],
                "score": ("trial", [0.2, 1.2, 0.4]),
            }
        )
    )
    batch = batch_source.group.groupby_bins(
        "score",
        bins=(0.0, 1.0, 2.0, 3.0),
        labels=labels,
    ).mean(opts=BatchGroupReduceOptions(include_empty_groups=True))
    assert batch.as_dataset(copy="none").coords["group_key"].values.tolist() == list(
        labels
    )


def test_lazy_tuple_bin_key_realizes_without_realizing_payload() -> None:
    """ID: BATCH_GROUP_021_lazy_tuple_bin_labels_preserve_payload_laziness."""
    calls: list[str] = []

    @delayed
    def record(name: str, values: np.ndarray) -> np.ndarray:
        calls.append(name)
        return values

    key = da.from_delayed(
        record("key", np.asarray([0.2, 1.2, 0.4], dtype=float)),
        shape=(3,),
        dtype=float,
    )
    payload = da.from_delayed(
        record("payload", np.asarray([1.0, 2.0, 3.0], dtype=float)),
        shape=(3,),
        dtype=float,
    )
    source = AnalysisObject.from_data(
        xr.Dataset({"value": ("trial", payload), "score": ("trial", key)}),
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    grouped = source.group.groupby_bins(
        "score",
        bins=(0.0, 1.0, 2.0),
        labels=(("low", 0), ("high", 1)),
    )

    assert calls == []
    out = grouped.mean()
    assert calls == ["key"]
    assert isinstance(out.as_dataset(copy="none")["value"].data, da.Array)
    assert out.as_dataset(copy="none").coords["group_key"].values.tolist() == [
        ("low", 0),
        ("high", 1),
    ]
    assert "payload" not in calls


def _lazy_batch_source(calls: list[str]) -> AnalysisObject:
    @delayed
    def key_values() -> np.ndarray:
        calls.append("key")
        return np.asarray(["a", "b", "a"], dtype=object)

    return AnalysisObject.from_data(
        xr.Dataset(
            {
                "value": (("trial", "axis"), np.arange(6.0).reshape(3, 2)),
                "key": (
                    "trial",
                    da.from_delayed(key_values(), shape=(3,), dtype=object),
                ),
            },
            coords={"axis": ["x", "y"]},
        ),
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )


def test_batch_reducer_static_preflight_does_not_realize_lazy_key() -> None:
    """ID: BATCH_GROUP_022_static_reducer_failures_precede_key_realization."""
    calls: list[str] = []
    grouped = _lazy_batch_source(calls).group.groupby("key")

    with pytest.raises(ValueError, match=r"missing=\('missing',\)"):
        grouped.mean(dim=("trial", "missing"))
    with pytest.raises(ValueError, match="must include primary batch dimension"):
        grouped.mean(dim="axis")
    with pytest.raises(ValueError, match="cannot reduce its group dimension"):
        grouped.mean(dim=("trial", "group_key"))
    with pytest.raises(ValueError, match="weights are supported only"):
        grouped.median(weights=np.ones(3))
    assert calls == []


def test_batch_reducer_component_preflight_does_not_realize_lazy_key() -> None:
    """ID: BATCH_GROUP_023_component_dim_failure_precedes_key_realization."""
    calls: list[str] = []

    @delayed
    def key_values() -> np.ndarray:
        calls.append("key")
        return np.asarray(["a", "b"], dtype=object)

    values = np.zeros((2, 4), dtype=float)
    values[:, 3] = 1.0
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"rotation": (("trial", "quat"), values)},
            coords={
                "trial": [0, 1],
                "quat": ["x", "y", "z", "w"],
                "key": (
                    "trial",
                    da.from_delayed(key_values(), shape=(2,), dtype=object),
                ),
            },
        ),
        batch_dims=("trial",),
        core_dims=("quat",),
        validate=True,
    )
    grouped = Rotation(source.as_dataset(copy="none")).group.groupby("key")

    with pytest.raises(ValueError, match="required component dims"):
        grouped.mean(dim=("trial", "quat"))
    with pytest.raises(ValueError, match="required component dims"):
        grouped.sum(dim=("trial", "quat"))
    assert calls == []


def test_sequence_reducer_options_preflight_uses_shared_name_owner() -> None:
    """ID: BATCH_GROUP_024_sequence_reducer_options_share_name_preflight."""
    grouped = _sequence_bin_source().group.groupby("score")
    with pytest.raises(ValueError, match="collides with source namespace"):
        grouped.mean(
            opts=GroupMaterializeOptions(
                group_dim="trial",
                include_empty_groups=False,
            )
        )


def _native_index_batch_source() -> AnalysisObject:
    lane_index = xr.indexes.CoordinateTransformIndex(
        _OffsetCoordinateTransform(2, offset=0.5, dim="lane")
    )
    axis_index = xr.indexes.RangeIndex.arange(2, dim="axis")
    coords = xr.Coordinates.from_xindex(lane_index).merge(
        xr.Coordinates.from_xindex(axis_index)
    )
    ds = xr.Dataset(
        {"value": (("trial", "lane", "axis"), np.arange(12.0).reshape(3, 2, 2))},
        coords=coords,
    ).assign_coords(
        trial=["t0", "t1", "t2"],
        outcome=("trial", ["a", "b", "a"]),
    )
    return AnalysisObject.from_data(
        ds,
        batch_dims=("trial", "lane"),
        core_dims=("axis",),
        validate=True,
    )


def test_batch_partitions_preserve_payload_only_native_indexes() -> None:
    """ID: BATCH_GROUP_025_row_partitions_preserve_native_indexes."""
    source = _native_index_batch_source()
    source_ds = source.as_dataset(copy="none")
    actual = source.group.groupby("outcome").mean(dim="trial").as_dataset(
        copy="none"
    )
    expected = xr.concat(
        (
            source_ds["value"].isel(trial=[0, 2]).mean("trial"),
            source_ds["value"].isel(trial=[1]).mean("trial"),
        ),
        dim=xr.DataArray(["a", "b"], dims="group_key", name="group_key"),
    )

    xr.testing.assert_identical(actual["value"], expected)
    assert type(actual.xindexes["lane"]) is type(source_ds.xindexes["lane"])
    assert actual.xindexes["lane"].equals(source_ds.xindexes["lane"])
    assert type(actual.xindexes["axis"]) is type(source_ds.xindexes["axis"])
    assert actual.xindexes["axis"].equals(source_ds.xindexes["axis"])


def test_weight_validity_mask_preserves_native_reduced_index() -> None:
    """ID: BATCH_GROUP_026_sparse_weight_zone_preserves_native_indexes."""
    source = _native_index_batch_source()
    source_ds = source.as_dataset(copy="none")
    weights = {
        "trial": xr.DataArray(
            [1.0, 1.0, 2.0],
            dims="trial",
            coords={"trial": ["t0", "t1", "t2"]},
        ),
        "axis": xr.DataArray(
            [1.0, 3.0],
            dims="axis",
            coords=xr.Coordinates.from_xindex(source_ds.xindexes["axis"].copy()),
        ),
    }
    actual = source.group.groupby("outcome").mean(
        dim=("trial", "axis"),
        weights=weights,
    ).as_dataset(copy="none")

    assert tuple(actual["value"].dims) == ("group_key", "lane")
    assert type(actual.xindexes["lane"]) is type(source_ds.xindexes["lane"])
    assert actual.xindexes["lane"].equals(source_ds.xindexes["lane"])


def test_native_index_batch_partitions_keep_dask_payload_lazy() -> None:
    """ID: BATCH_GROUP_031_native_index_partitions_preserve_dask_laziness."""
    calls: list[str] = []

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return np.arange(12.0).reshape(3, 2, 2)

    eager = _native_index_batch_source().as_dataset(copy="none")
    lazy = eager.assign(
        value=(
            ("trial", "lane", "axis"),
            da.from_delayed(payload(), shape=(3, 2, 2), dtype=float),
        )
    )
    source = AnalysisObject.from_data(
        lazy,
        batch_dims=("trial", "lane"),
        core_dims=("axis",),
        validate=True,
    )

    actual = source.group.groupby("outcome").mean(dim="trial")
    actual_ds = actual.as_dataset(copy="none")
    assert calls == []
    assert isinstance(actual_ds["value"].data, da.Array)
    assert type(actual_ds.xindexes["lane"]) is type(eager.xindexes["lane"])
    actual_ds.compute()
    assert calls == ["payload"]


def test_batch_topology_options_precede_key_and_bin_resolution() -> None:
    """ID: BATCH_GROUP_027_batch_topology_options_precede_key_work."""
    calls: list[str] = []

    @delayed
    def bin_edges() -> np.ndarray:
        calls.append("bins")
        return np.asarray([0.0, 1.0, 2.0])

    source = _indexed_batch_source(
        xr.Coordinates(
            {
                "trial": ["t0", "t1", "t2"],
                "score": ("trial", [0.2, 1.2, 0.4]),
            }
        )
    )
    bins = xr.DataArray(
        da.from_delayed(bin_edges(), shape=(3,), dtype=float),
        dims="edge",
    )

    invalid = (
        (GroupByOptions(preserve_batch=True), "preserve_batch=True"),
        (GroupByOptions(member_dim="member"), "member_dim applies only"),
        (
            GroupByOptions(sequence_index_coord="source_index"),
            "sequence_index_coord applies only",
        ),
    )
    for opts, match in invalid:
        with pytest.raises(ValueError, match=match):
            source.group.groupby("missing", opts=opts)
    with pytest.raises(ValueError, match="preserve_batch=True"):
        source.group.groupby_bins(
            "score",
            bins,
            opts=GroupByOptions(preserve_batch=True),
        )
    assert calls == []


@pytest.mark.parametrize(
    "weights, match",
    [
        ("bad", "weights must be"),
        (np.ones((3, 1)), "must be 1-D"),
        ({}, "must include every reduced dim"),
        (xr.DataArray([1.0, 1.0], dims="axis"), "weight dims must be a subset"),
    ],
)
def test_batch_weight_structure_preflight_precedes_lazy_key(
    weights: object,
    match: str,
) -> None:
    """ID: BATCH_GROUP_028_weight_structure_precedes_key_realization."""
    calls: list[str] = []
    grouped = _lazy_batch_source(calls).group.groupby("key")

    with pytest.raises((TypeError, ValueError), match=match):
        grouped.mean(weights=weights)  # type: ignore[arg-type]
    assert calls == []


def test_batch_eligibility_preflight_precedes_lazy_key() -> None:
    """ID: BATCH_GROUP_029_eligibility_precedes_key_realization."""
    calls: list[str] = []

    @delayed
    def key_values() -> np.ndarray:
        calls.append("key")
        return np.asarray(["a", "b"], dtype=object)

    source = AnalysisObject.from_data(
        xr.Dataset(
            {
                "text": ("trial", ["x", "y"]),
                "key": (
                    "trial",
                    da.from_delayed(key_values(), shape=(2,), dtype=object),
                ),
            }
        ),
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )

    with pytest.raises(ValueError, match="requires at least one numeric"):
        source.group.groupby("key").mean()
    assert calls == []


def test_batch_weight_value_validation_remains_after_key_realization() -> None:
    """ID: BATCH_GROUP_030_weight_values_remain_post_partition_planning."""
    calls: list[str] = []
    grouped = _lazy_batch_source(calls).group.groupby("key")

    with pytest.raises(ValueError, match="negative weights"):
        grouped.mean(weights=np.asarray([1.0, -1.0, 1.0]))
    assert calls == ["key"]


def _batch_rotation() -> Rotation:
    values = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"rotation": (("trial", "quat"), values)},
            coords={
                "trial": ["t0", "t1", "t2", "t3", "t4"],
                "quat": ["x", "y", "z", "w"],
                "key": ("trial", ["a", "a", "b", "b", "b"]),
                "score": ("trial", [0.2, 0.4, 1.2, 1.4, 1.6]),
            },
        ),
        batch_dims=("trial",),
        core_dims=("quat",),
        validate=True,
    )
    return Rotation(source)


def _primary_independent_batch_rotation(
    *,
    size: int = 3,
    payload: object | None = None,
) -> Rotation:
    values = (
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        if payload is None
        else payload
    )
    labels = np.asarray(["a", "b", "a"], dtype=object)[:size]
    scores = np.asarray([0.2, 1.2, 0.4], dtype=float)[:size]
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"rotation": (("lane", "quat"), values)},
            coords={
                "trial": np.arange(size),
                "lane": ["left", "right"],
                "quat": ["x", "y", "z", "w"],
                "key": ("trial", labels),
                "score": ("trial", scores),
            },
        ),
        batch_dims=("trial", "lane"),
        core_dims=("quat",),
        validate=True,
    )
    return Rotation(source)


def test_batch_typed_reducers_delegate_group_partitions_to_bound_owner() -> None:
    """ID: BATCH_GROUP_032_typed_partitions_delegate_to_bound_reducer_owner."""
    source = _batch_rotation()
    actual = source.group.groupby("key").mean()

    assert isinstance(actual, Rotation)
    for label, rows in (("a", [0, 1]), ("b", [2, 3, 4])):
        expected = source.isel(trial=rows).mean(dim="trial")
        xr.testing.assert_identical(
            actual.as_dataset(copy="none")["rotation"]
            .sel(group_key=label)
            .drop_vars("group_key"),
            expected.as_dataset(copy="none")["rotation"],
        )
    assert type(source.group.groupby("key").sum()) is AnalysisObject


def test_batch_rotation_preserves_primary_independent_typed_payload() -> None:
    """ID: BATCH_GROUP_041_primary_independent_rotation_mean."""
    source = _primary_independent_batch_rotation()
    before = source.as_dataset(copy="deep")

    actual = source.group.groupby("key").mean().as_dataset(copy="none")

    np.testing.assert_array_equal(actual.coords["group_key"], ["a", "b"])
    xr.testing.assert_identical(actual["rotation"], before["rotation"])
    assert actual.attrs["tal"]["core"]["roles"] == {
        "batch_dims": ["group_key", "lane"],
        "core_dims": ["quat"],
    }
    np.testing.assert_array_equal(actual.coords["lane"], ["left", "right"])
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


def test_batch_rotation_grouping_uses_declared_core_role_over_size_candidates() -> None:
    """ID: BATCH_GROUP_044_rotation_declared_core_role_is_authoritative."""
    values = np.zeros((4, 4), dtype=float)
    values[:, 3] = 1.0
    source = Rotation(
        AnalysisObject.from_data(
            xr.Dataset(
                {"rotation": (("lane", "quat"), values)},
                coords={
                    "trial": [0, 1, 2],
                    "lane": ["north", "south", "east", "west"],
                    "quat": ["x", "y", "z", "w"],
                    "key": ("trial", ["a", "b", "a"]),
                },
            ),
            batch_dims=("trial", "lane"),
            core_dims=("quat",),
            validate=True,
        )
    )

    actual = source.group.groupby("key").mean().as_dataset(copy="none")

    np.testing.assert_array_equal(actual.coords["group_key"], ["a", "b"])
    xr.testing.assert_identical(actual["rotation"], source.as_dataset(copy="none")["rotation"])
    assert actual.attrs["tal"]["core"]["roles"] == {
        "batch_dims": ["group_key", "lane"],
        "core_dims": ["quat"],
    }


def test_batch_rotation_primary_independent_empty_groups_and_rows() -> None:
    """ID: BATCH_GROUP_042_primary_independent_rotation_empty_groups."""
    source = _primary_independent_batch_rotation(size=2)
    bins = source.group.groupby_bins(
        "score",
        bins=[0.0, 1.0, 2.0, 3.0],
        labels=["first", "second", "empty"],
    ).mean(opts=BatchGroupReduceOptions(include_empty_groups=True))
    empty = _primary_independent_batch_rotation(size=0).group.groupby("key").mean()

    assert isinstance(bins, Rotation)
    np.testing.assert_array_equal(
        bins.as_dataset(copy="none").coords["group_key"],
        ["first", "second", "empty"],
    )
    assert bins.as_dataset(copy="none")["rotation"].dims == ("lane", "quat")
    assert isinstance(empty, Rotation)
    assert empty.as_dataset(copy="none").sizes["group_key"] == 0
    assert empty.as_dataset(copy="none")["rotation"].dims == ("lane", "quat")


def test_batch_rotation_primary_independent_payload_stays_lazy() -> None:
    """ID: BATCH_GROUP_043_primary_independent_rotation_is_lazy."""
    calls: list[str] = []

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    lazy = da.from_delayed(payload(), shape=(2, 4), dtype=float)
    actual = _primary_independent_batch_rotation(payload=lazy).group.groupby("key").mean()
    data = actual.as_dataset(copy="none")["rotation"]

    assert isinstance(data.data, da.Array)
    assert calls == []
    data.compute()
    assert calls == ["payload"]


def test_batch_typed_empty_groups_and_zero_row_mean_are_missing() -> None:
    """ID: BATCH_GROUP_033_typed_empty_groups_use_reducer_empty_semantics."""
    source = _batch_rotation().isel(trial=[0, 1])
    actual = source.group.groupby_bins(
        "score",
        bins=[0.0, 1.0, 2.0],
        labels=["present", "empty"],
    ).mean(opts=BatchGroupReduceOptions(include_empty_groups=True))

    assert isinstance(actual, Rotation)
    empty = actual.as_dataset(copy="none")["rotation"].sel(group_key="empty")
    assert bool(empty.isnull().all())
    direct_empty = source.isel(trial=[]).mean(dim="trial")
    assert isinstance(direct_empty, Rotation)
    assert bool(direct_empty.as_dataset(copy="none")["rotation"].isnull().all())


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray(["2020-01-01", "2020-01-02", "2020-01-01"], dtype="datetime64[ns]"),
        np.asarray([1, 2, 1], dtype="timedelta64[ns]"),
    ],
)
def test_numpy_temporal_group_labels_remain_temporal_scalars(
    labels: np.ndarray,
) -> None:
    """ID: BATCH_GROUP_034_numpy_temporal_labels_do_not_become_integers."""
    ds = xr.Dataset(
        {"value": ("trial", [1.0, 2.0, 3.0])},
        coords={"trial": [0, 1, 2], "when": ("trial", labels)},
    )
    batch = AnalysisObject.from_data(
        ds,
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    ).group.groupby("when").mean().as_dataset(copy="none")
    sequence = AnalysisObject.from_data(
        ds.rename(trial="sample"),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    ).group.groupby("when").mean().as_dataset(copy="none")

    expected = labels[:2]
    np.testing.assert_array_equal(batch.coords["group_key"], expected)
    np.testing.assert_array_equal(sequence.coords["group_key"], expected)
    np.testing.assert_allclose(batch["value"], [2.0, 2.0])
    np.testing.assert_allclose(sequence["value"], [2.0, 2.0])


def test_numpy_temporal_bin_domains_do_not_duplicate_observed_groups() -> None:
    """ID: BATCH_GROUP_035_temporal_bin_domains_match_observed_labels."""
    labels = np.asarray(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("trial", [1.0, 3.0])},
            coords={"trial": [0, 1], "score": ("trial", [0.2, 0.4])},
        ),
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    actual = source.group.groupby_bins(
        "score",
        bins=[0.0, 1.0, 2.0],
        labels=labels,
    ).mean(opts=BatchGroupReduceOptions(include_empty_groups=True)).as_dataset(
        copy="none"
    )

    np.testing.assert_array_equal(actual.coords["group_key"], labels)
    np.testing.assert_allclose(actual["value"].isel(group_key=0), 2.0)
    assert bool(actual["value"].isel(group_key=1).isnull())


@pytest.mark.parametrize(
    "weights",
    [
        xr.DataArray([1.0, 1.0], dims="trial"),
        {"trial": xr.DataArray([1.0, 1.0], dims="trial")},
    ],
)
def test_dataarray_weight_lengths_fail_before_lazy_key(
    weights: object,
) -> None:
    """ID: BATCH_GROUP_036_dataarray_weight_length_precedes_key_realization."""
    calls: list[str] = []
    grouped = _lazy_batch_source(calls).group.groupby("key")

    with pytest.raises(ValueError, match="DataArray weights length"):
        grouped.mean(weights=weights)  # type: ignore[arg-type]
    assert calls == []


@pytest.mark.parametrize("group_dim", ["group_member", "sequence_index"])
def test_batch_group_name_is_independent_of_sequence_only_names(
    group_dim: str,
) -> None:
    """ID: BATCH_GROUP_037_batch_group_name_ignores_sequence_only_defaults."""
    source = _indexed_batch_source(
        xr.Dataset(
            coords={
                "trial": ["t0", "t1", "t2"],
                "key": ("trial", ["a", "b", "a"]),
            }
        ).coords
    )
    actual = source.group.groupby(
        "key",
        opts=GroupByOptions(group_dim=group_dim),
    ).mean().as_dataset(copy="none")
    assert group_dim in actual.dims

    with pytest.raises(ValueError, match="must be distinct"):
        _sequence_bin_source().group.groupby(
            "score",
            opts=GroupByOptions(group_dim=group_dim),
        )


def test_ndarray_weights_retain_primary_independent_dimension_meaning() -> None:
    """ID: BATCH_GROUP_038_ndarray_weights_keep_per_variable_dimension_meaning."""
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("axis", [1.0, 2.0])},
            coords={
                "trial": [0, 1, 2],
                "axis": ["x", "y"],
                "key": ("trial", ["a", "b", "a"]),
            },
        ),
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    actual = source.group.groupby("key").sum(
        dim=("trial", "axis"),
        weights=np.asarray([1.0, 2.0]),
    ).as_dataset(copy="none")

    np.testing.assert_array_equal(actual.coords["group_key"], ["a", "b"])
    assert actual["value"].dims == ()
    assert actual["value"].item() == 5.0
    assert actual.attrs["tal"]["core"]["roles"] == {
        "batch_dims": ["group_key"],
        "core_dims": [],
    }


def test_weight_indexes_align_before_unselectable_index_partitioning() -> None:
    """ID: BATCH_GROUP_039_weight_indexes_align_before_partition_selection."""
    source_ds = xr.Dataset(
        {"value": ("trial", [1.0, 2.0, 3.0])},
        coords=_transform_key(offset=0.5).coords,
    ).assign_coords(key=("trial", ["a", "b", "a"]))
    source = AnalysisObject.from_data(
        source_ds,
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )
    weights = xr.DataArray(
        [1.0, 1.0, 1.0],
        dims="trial",
        coords=_transform_key(offset=10.5).coords,
    )

    with pytest.raises(ValueError, match="not label-aligned"):
        source.group.groupby("key").mean(weights=weights)


def test_typed_partition_reduction_keeps_dask_payload_lazy() -> None:
    """ID: BATCH_GROUP_040_typed_row_partitions_preserve_dask_laziness."""
    calls: list[str] = []
    eager = _batch_rotation().as_dataset(copy="none")

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return np.asarray(eager["rotation"].data)

    lazy_ds = eager.assign(
        rotation=(
            ("trial", "quat"),
            da.from_delayed(payload(), shape=(5, 4), dtype=float),
        )
    )
    source = Rotation(
        AnalysisObject.from_data(
            lazy_ds,
            batch_dims=("trial",),
            core_dims=("quat",),
            validate=True,
        )
    )
    actual = source.group.groupby("key").mean()

    assert isinstance(actual.as_dataset(copy="none")["rotation"].data, da.Array)
    assert calls == []
    actual.as_dataset(copy="none")["rotation"].compute()
    assert calls == ["payload"]
