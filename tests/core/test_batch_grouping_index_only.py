from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import AnalysisObject, BatchGroupReduceOptions


class _UnrealizedRowTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size: int) -> None:
        super().__init__(("trial",), {"trial": size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Unrelated primary-index coordinates must not be evaluated")

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Positional partition selection must not resolve labels")

    def equals(self, other: object, **kwargs: object) -> bool:
        return isinstance(other, _UnrealizedRowTransform) and self.dim_size == other.dim_size


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("rows", [0, 3])
def test_index_only_batch_lane_survives_empty_and_nonempty_partitions(lazy: bool, rows: int) -> None:
    """ID: BATCH_INDEX_047_primary_independent_native_index_partitions."""
    calls: list[str] = []

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return np.asarray([2.0, 4.0])

    row_index = xr.indexes.CoordinateTransformIndex(_UnrealizedRowTransform(rows))
    axis_index = xr.indexes.RangeIndex.arange(2, dim="axis")
    coords = xr.Coordinates.from_xindex(row_index).merge(xr.Coordinates.from_xindex(axis_index)).coords
    values = da.from_delayed(payload(), shape=(2,), dtype=float) if lazy else [2.0, 4.0]
    source = AnalysisObject.from_data(
        xr.Dataset({"static": ("axis", values)}, coords=coords),
        batch_dims=("trial",), core_dims=("axis",),
    )
    key = xr.DataArray([0.2, 1.2, 0.4][:rows], dims="trial", coords=xr.Coordinates.from_xindex(row_index))
    actual = source.group.groupby_bins(
        key, [0.0, 1.0, 2.0, 3.0], labels=["multi", "single", "empty"]
    ).mean(opts=BatchGroupReduceOptions(include_empty_groups=True)).as_dataset(copy="none")

    assert calls == []
    assert isinstance(actual["static"].data, da.Array) == lazy
    assert actual["static"].dims == ("axis",)
    assert set(actual.coords) == {"axis", "group_key"}
    assert "trial" not in actual.dims
    assert actual.xindexes["axis"].equals(axis_index)
    assert type(actual.xindexes["axis"]) is type(axis_index)
    assert actual.attrs["tal"]["core"]["roles"] == {
        "batch_dims": ["group_key"], "core_dims": ["axis"]
    }
    np.testing.assert_array_equal(actual.coords["group_key"], ["multi", "single", "empty"])
    np.testing.assert_allclose(actual["static"].compute(), [2.0, 4.0])
    assert calls == (["payload"] if lazy else [])


def test_index_only_lane_with_no_groups_does_not_leak_positional_coordinates() -> None:
    """ID: BATCH_INDEX_048_index_only_zero_group_topology."""
    index = xr.indexes.CoordinateTransformIndex(_UnrealizedRowTransform(0))
    coords = xr.Coordinates.from_xindex(index)
    source = AnalysisObject.from_data(
        xr.Dataset({"static": 2.0}, coords=coords), batch_dims=("trial",), core_dims=()
    )
    key = xr.DataArray(np.empty(0), dims="trial", coords=coords)
    actual = source.group.groupby(key).mean().as_dataset(copy="none")
    assert actual.sizes == {"group_key": 0}
    assert actual["static"].dims == ()
    assert actual["static"].item() == 2.0
