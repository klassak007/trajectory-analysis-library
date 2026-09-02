from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.dataset_ownership import (
    coerce_dataset_copy_mode,
    dataset_to_dataarray_view,
    dataset_view,
    deep_public_dataset,
    isolate_external_dataset,
    metadata_isolated_dataset,
    raw_dataset_reference,
)


def _source_dataset() -> xr.Dataset:
    ds = xr.Dataset(
        {"value": ("sample", np.arange(4.0))},
        coords={
            "sample": np.arange(4, dtype=np.int64),
            "aux": ("sample", np.arange(4.0) + 10.0),
        },
        attrs={"nested": {"items": ["dataset"]}},
    )
    ds.encoding = {"nested": {"items": ["dataset-encoding"]}}
    for name in ("value", "sample", "aux"):
        ds[name].attrs = {"nested": {"items": [f"{name}-attrs"]}}
        ds[name].encoding = {"nested": {"items": [f"{name}-encoding"]}}
    return ds


def _renamed_pandas_index_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"value": ("sample", np.arange(4.0))},
        coords={"label": ("sample", np.arange(4, dtype=np.int64))},
    ).set_xindex("label")


def _nd_point_index_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"value": (("y", "x"), np.arange(4.0).reshape(2, 2))},
        coords={
            "xx": (("y", "x"), np.array([[1.0, 2.0], [3.0, 0.0]])),
            "yy": (("y", "x"), np.array([[11.0, 21.0], [29.0, 9.0]])),
        },
    ).set_xindex(("xx", "yy"), xr.indexes.NDPointIndex)


_INDEXED_DATASET_BUILDERS: tuple[Callable[[], xr.Dataset], ...] = (
    _renamed_pandas_index_dataset,
    _nd_point_index_dataset,
)


def _assert_index_groups_detached(
    source: xr.Dataset,
    target: xr.Dataset | xr.DataArray,
) -> None:
    for source_index, coords in source.xindexes.group_by_index():
        names = tuple(coords)
        target_index = target.xindexes[names[0]]
        assert type(target_index) is type(source_index)
        assert target_index.equals(source_index)
        assert target_index is not source_index
        assert all(target.xindexes[name] is target_index for name in names)
        assert all(
            not np.shares_memory(target.coords[name].data, source.coords[name].data)
            for name in names
        )


def _coordinate_snapshots(ds: xr.Dataset) -> dict[str, np.ndarray]:
    return {
        name: np.array(ds.coords[name].data, copy=True)
        for _, coords in ds.xindexes.group_by_index()
        for name in coords
    }


def _mutate_index_coordinates(ds: xr.Dataset | xr.DataArray) -> None:
    for _, coords in ds.xindexes.group_by_index():
        for name in coords:
            ds.coords[name].data.flat[0] += 100


def _mutate_nested_metadata(ds: xr.Dataset) -> None:
    ds.attrs["nested"]["items"].append("changed")
    ds.encoding["nested"]["items"].append("changed")
    for name in ("value", "sample", "aux"):
        ds[name].attrs["nested"]["items"].append("changed")
        ds[name].encoding["nested"]["items"].append("changed")


def _assert_source_metadata_unchanged(ds: xr.Dataset) -> None:
    assert ds.attrs["nested"]["items"] == ["dataset"]
    assert ds.encoding["nested"]["items"] == ["dataset-encoding"]
    for name in ("value", "sample", "aux"):
        assert ds[name].attrs["nested"]["items"] == [f"{name}-attrs"]
        assert ds[name].encoding["nested"]["items"] == [f"{name}-encoding"]


def test_dataset_ownership_001_copy_mode_coercion_is_centralized() -> None:
    """ID: DATASET_OWNERSHIP_001_copy_mode_coercion_is_centralized."""
    for mode in ("deep", "shallow", "none"):
        assert coerce_dataset_copy_mode(mode, owner="ownership.test") == mode
    for invalid in (None, False, 1, "invalid"):
        with pytest.raises(ValueError, match=r"^ownership\.test: copy must be"):
            coerce_dataset_copy_mode(invalid, owner="ownership.test")


def test_dataset_ownership_002_deep_view_isolates_eager_state() -> None:
    """ID: DATASET_OWNERSHIP_002_deep_view_isolates_eager_state."""
    source = _source_dataset()
    out = dataset_view(source, copy="deep", owner="ownership.test")

    assert out is not source
    assert not np.shares_memory(out["value"].data, source["value"].data)
    assert not np.shares_memory(out.coords["sample"].data, source.coords["sample"].data)
    assert not np.shares_memory(out.coords["aux"].data, source.coords["aux"].data)
    assert out.indexes["sample"] is not source.indexes["sample"]

    out["value"].data[0] = 99.0
    out.coords["sample"].data[0] = 99
    out.coords["aux"].data[0] = 99.0
    _mutate_nested_metadata(out)

    assert float(source["value"].data[0]) == 0.0
    assert int(source.coords["sample"].data[0]) == 0
    assert float(source.coords["aux"].data[0]) == 10.0
    _assert_source_metadata_unchanged(source)


def test_dataset_ownership_003_shallow_view_shares_buffers_only() -> None:
    """ID: DATASET_OWNERSHIP_003_shallow_view_shares_buffers_only."""
    source = _source_dataset()
    out = metadata_isolated_dataset(source, owner="ownership.test")

    assert out is not source
    assert np.shares_memory(out["value"].data, source["value"].data)
    assert np.shares_memory(out.coords["sample"].data, source.coords["sample"].data)
    assert np.shares_memory(out.coords["aux"].data, source.coords["aux"].data)

    _mutate_nested_metadata(out)
    _assert_source_metadata_unchanged(source)
    assert "value" in source
    assert "value" not in out.drop_vars("value")

    out["value"].data[0] = 77.0
    assert float(source["value"].data[0]) == 77.0


def test_dataset_ownership_004_copy_views_do_not_steal_close_callback() -> None:
    """ID: DATASET_OWNERSHIP_004_copy_views_do_not_steal_close_callback."""
    source = _source_dataset()
    closed: list[str] = []
    source.set_close(lambda: closed.append("source"))

    deep = deep_public_dataset(source, owner="ownership.test")
    shallow = metadata_isolated_dataset(source, owner="ownership.test")
    deep.close()
    shallow.close()
    assert closed == []

    raw = raw_dataset_reference(source)
    assert raw is source
    assert dataset_view(source, copy="none", owner="ownership.test") is source
    raw.close()
    raw.close()
    assert closed == ["source"]


def test_dataset_ownership_005_external_ingress_is_deep_and_nonowning() -> None:
    """ID: DATASET_OWNERSHIP_005_external_ingress_is_deep_and_nonowning."""
    source = _source_dataset()
    closed: list[str] = []
    source.set_close(lambda: closed.append("source"))

    out = isolate_external_dataset(source)
    out["value"].data[0] = 50.0
    out.coords["sample"].data[0] = 50
    _mutate_nested_metadata(out)
    out.close()

    assert float(source["value"].data[0]) == 0.0
    assert int(source.coords["sample"].data[0]) == 0
    _assert_source_metadata_unchanged(source)
    assert closed == []


class _TaskCounter(Callback):
    def __init__(self) -> None:
        self.keys: list[object] = []
        super().__init__(pretask=self._record)

    def _record(self, key: object, _dsk: object, _state: object) -> None:
        self.keys.append(key)


def test_dataset_ownership_006_dask_views_remain_lazy() -> None:
    """ID: DATASET_OWNERSHIP_006_dask_views_remain_lazy."""
    source = xr.Dataset(
        {"value": ("sample", da.arange(6, chunks=3))},
        coords={
            "sample": np.arange(6, dtype=np.int64),
            "aux": ("sample", da.arange(6, chunks=3) + 10),
        },
    )
    assert isinstance(source["value"].data, da.Array)
    assert isinstance(source.coords["aux"].data, da.Array)

    counter = _TaskCounter()
    with counter:
        deep = dataset_view(source, copy="deep", owner="ownership.test")
        shallow = dataset_view(source, copy="shallow", owner="ownership.test")

    assert counter.keys == []
    assert isinstance(deep["value"].data, da.Array)
    assert isinstance(deep.coords["aux"].data, da.Array)
    assert deep["value"].data.dask is source["value"].data.dask
    assert deep.coords["aux"].data.dask is source.coords["aux"].data.dask
    assert shallow["value"].data is source["value"].data
    assert shallow.coords["aux"].data is source.coords["aux"].data

    closed: list[str] = []
    source.set_close(lambda: closed.append("source"))
    deep.close()
    shallow.close()
    assert closed == []


def test_dataset_ownership_007_dataarray_conversion_uses_same_modes() -> None:
    """ID: DATASET_OWNERSHIP_007_dataarray_conversion_uses_same_modes."""
    source = _source_dataset()
    deep = dataset_to_dataarray_view(
        source, name="renamed", copy="deep", owner="ownership.test"
    )
    shallow = dataset_to_dataarray_view(
        source, name=None, copy="shallow", owner="ownership.test"
    )
    raw = dataset_to_dataarray_view(
        source, name=None, copy="none", owner="ownership.test"
    )

    assert deep.name == "renamed"
    assert not np.shares_memory(deep.data, source["value"].data)
    assert np.shares_memory(shallow.data, source["value"].data)
    assert np.shares_memory(raw.data, source["value"].data)
    deep.attrs["nested"]["items"].append("changed")
    shallow.attrs["nested"]["items"].append("changed")
    deep.coords["sample"].attrs["nested"]["items"].append("changed")
    shallow.coords["sample"].encoding["nested"]["items"].append("changed")
    assert source["value"].attrs["nested"]["items"] == ["value-attrs"]
    assert source.coords["sample"].attrs["nested"]["items"] == ["sample-attrs"]
    assert source.coords["sample"].encoding["nested"]["items"] == [
        "sample-encoding"
    ]


def test_dataset_ownership_008_deep_view_preserves_pandas_index_subclass() -> None:
    """ID: DATASET_OWNERSHIP_008_deep_view_preserves_pandas_index_subclass."""
    labels = pd.CategoricalIndex(["a", "b", "a"], name="sample")
    source = xr.Dataset(
        {"value": ("sample", np.arange(3.0))},
        coords={"sample": labels},
    )

    out = dataset_view(source, copy="deep", owner="ownership.test")

    assert isinstance(out.indexes["sample"], pd.CategoricalIndex)
    assert out.indexes["sample"].equals(source.indexes["sample"])
    assert out.indexes["sample"] is not source.indexes["sample"]
    assert not np.shares_memory(
        out.coords["sample"].data,
        source.coords["sample"].data,
    )


def test_dataset_ownership_009_deep_views_preserve_native_range_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: DATASET_OWNERSHIP_009_deep_views_preserve_native_range_index."""
    index = xr.indexes.RangeIndex.arange(
        0,
        4,
        coord_name="sample",
        dim="sample",
    )
    source = xr.Dataset(
        {"value": ("sample", np.arange(4.0))},
        coords=xr.Coordinates.from_xindex(index),
    )

    def fail_pandas_conversion(_index: xr.indexes.RangeIndex) -> pd.Index:
        raise AssertionError("native RangeIndex must not convert to pandas")

    monkeypatch.setattr(xr.indexes.RangeIndex, "to_pandas_index", fail_pandas_conversion)
    deep = dataset_view(source, copy="deep", owner="ownership.test")
    ingress = isolate_external_dataset(source)

    for out in (deep, ingress):
        assert isinstance(out.xindexes["sample"], xr.indexes.RangeIndex)
        assert out.xindexes["sample"].equals(source.xindexes["sample"])
        assert out.xindexes["sample"] is not source.xindexes["sample"]


class _OffsetCoordinateTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size: int) -> None:
        super().__init__(("sample",), {"sample": size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        return {"sample": dim_positions["sample"] + 0.5}

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        return {"sample": coord_labels["sample"] - 0.5}

    def equals(self, other: object, **kwargs: object) -> bool:
        return (
            isinstance(other, _OffsetCoordinateTransform)
            and self.dim_size == other.dim_size
        )


def test_dataset_ownership_010_deep_views_preserve_coordinate_transform_index(
) -> None:
    """ID: DATASET_OWNERSHIP_010_deep_views_preserve_coordinate_transform_index."""
    index = xr.indexes.CoordinateTransformIndex(_OffsetCoordinateTransform(4))
    source = xr.Dataset(
        {"value": ("sample", np.arange(4.0))},
        coords=xr.Coordinates.from_xindex(index),
    )

    for out in (
        dataset_view(source, copy="deep", owner="ownership.test"),
        isolate_external_dataset(source),
    ):
        assert isinstance(
            out.xindexes["sample"],
            xr.indexes.CoordinateTransformIndex,
        )
        assert out.xindexes["sample"].equals(source.xindexes["sample"])
        assert out.xindexes["sample"] is not source.xindexes["sample"]


@pytest.mark.parametrize(
    "build_source",
    _INDEXED_DATASET_BUILDERS,
    ids=("renamed-pandas", "nd-point"),
)
@pytest.mark.parametrize(
    "construct",
    (AnalysisObject, AnalysisObject.from_data),
    ids=("constructor", "from-data"),
)
def test_dataset_ownership_011_indexed_ingress_is_detached(
    build_source: Callable[[], xr.Dataset],
    construct: Callable[[xr.Dataset], AnalysisObject],
) -> None:
    """ID: DATASET_OWNERSHIP_011_indexed_ingress_is_detached."""
    source = build_source()
    expected = _coordinate_snapshots(source)

    ao = construct(source)
    backing = ao.unsafe_data
    _assert_index_groups_detached(source, backing)
    _mutate_index_coordinates(source)

    for name, values in expected.items():
        np.testing.assert_array_equal(backing.coords[name].data, values)


@pytest.mark.parametrize(
    "build_source",
    _INDEXED_DATASET_BUILDERS,
    ids=("renamed-pandas", "nd-point"),
)
def test_dataset_ownership_012_public_deep_views_detach_index_groups(
    build_source: Callable[[], xr.Dataset],
) -> None:
    """ID: DATASET_OWNERSHIP_012_public_deep_views_detach_index_groups."""
    ao = AnalysisObject(build_source())
    backing = ao.unsafe_data
    expected = _coordinate_snapshots(backing)
    views = (ao.data, ao.as_dataset(), ao.to_dataarray())

    for view in views:
        _assert_index_groups_detached(backing, view)
        _mutate_index_coordinates(view)
        for name, values in expected.items():
            np.testing.assert_array_equal(backing.coords[name].data, values)


def test_dataset_ownership_013_shallow_ndpoint_isolates_coordinate_metadata(
) -> None:
    """ID: DATASET_OWNERSHIP_013_shallow_ndpoint_isolates_coordinate_metadata."""
    source = _nd_point_index_dataset()
    for name in ("xx", "yy"):
        source[name].attrs = {"nested": {"items": [f"{name}-attrs"]}}
        source[name].encoding = {"nested": {"items": [f"{name}-encoding"]}}

    out = metadata_isolated_dataset(source, owner="ownership.test")
    source_index = source.xindexes["xx"]
    out_index = out.xindexes["xx"]

    assert out.variables["xx"] is not source.variables["xx"]
    assert out.variables["yy"] is not source.variables["yy"]
    assert np.shares_memory(out["xx"].data, source["xx"].data)
    assert np.shares_memory(out["yy"].data, source["yy"].data)
    assert type(out_index) is type(source_index)
    assert out_index.equals(source_index)
    assert out_index is not source_index
    assert out.xindexes["yy"] is out_index

    out["xx"].attrs["nested"]["items"].append("changed")
    out["yy"].encoding["nested"]["items"].append("changed")
    assert source["xx"].attrs["nested"]["items"] == ["xx-attrs"]
    assert source["yy"].encoding["nested"]["items"] == ["yy-encoding"]

    out["xx"].data[0, 0] = 99.0
    assert float(source["xx"].data[0, 0]) == 99.0


@pytest.mark.parametrize(
    "labels",
    (
        pd.Index(np.arange(4), name="label"),
        pd.RangeIndex(0, 4, name="label"),
        pd.CategoricalIndex(["a", "b", "a", "c"], name="label"),
        pd.IntervalIndex.from_breaks([0, 1, 2, 3, 4], name="label"),
    ),
    ids=("index", "range", "categorical", "interval"),
)
def test_dataset_ownership_014_shallow_pandas_indexes_follow_public_semantics(
    labels: pd.Index,
) -> None:
    """ID: DATASET_OWNERSHIP_014_shallow_pandas_indexes_follow_public_semantics."""
    pandas_source = xr.Dataset(
        {"value": ("sample", np.arange(4.0))},
        coords={"label": ("sample", labels)},
    ).set_xindex("label")
    pandas_source["label"].attrs = {"nested": {"items": ["source"]}}
    source_index = pandas_source.xindexes["label"]
    pandas_out = metadata_isolated_dataset(pandas_source, owner="ownership.test")
    out_index = pandas_out.xindexes["label"]

    assert type(out_index) is type(source_index)
    assert out_index.equals(source_index)
    assert out_index is not source_index
    assert pandas_out.variables["label"] is not pandas_source.variables["label"]
    assert np.shares_memory(pandas_out["value"].data, pandas_source["value"].data)
    np.testing.assert_array_equal(pandas_out["label"].data, pandas_source["label"].data)

    pandas_out["label"].attrs["nested"]["items"].append("changed")
    assert pandas_source["label"].attrs["nested"]["items"] == ["source"]


@pytest.mark.parametrize(
    "index",
    (
        xr.indexes.RangeIndex.arange(
            0,
            4,
            coord_name="sample",
            dim="sample",
        ),
        xr.indexes.CoordinateTransformIndex(_OffsetCoordinateTransform(4)),
    ),
    ids=("range", "coordinate-transform"),
)
def test_dataset_ownership_015_shallow_transform_indexes_follow_public_semantics(
    index: xr.Index,
) -> None:
    """ID: DATASET_OWNERSHIP_015_shallow_transform_indexes_follow_public_semantics."""
    source = xr.Dataset(
        {"value": ("sample", np.arange(4.0))},
        coords=xr.Coordinates.from_xindex(index),
    )
    source["sample"].attrs = {"nested": {"items": ["source"]}}
    out = metadata_isolated_dataset(source, owner="ownership.test")
    source_index = source.xindexes["sample"]
    out_index = out.xindexes["sample"]

    assert type(out_index) is type(source_index)
    assert out_index.equals(source_index)
    assert out_index is not source_index
    assert out.variables["sample"] is not source.variables["sample"]
    assert np.shares_memory(out["value"].data, source["value"].data)
    np.testing.assert_array_equal(out["sample"].data, source["sample"].data)

    out["sample"].attrs["nested"]["items"].append("changed")
    assert source["sample"].attrs["nested"]["items"] == ["source"]
