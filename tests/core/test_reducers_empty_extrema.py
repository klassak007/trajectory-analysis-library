from __future__ import annotations

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
from tal.core.group_ops import GroupingFoundationOptions


def _source(values: np.ndarray | da.Array, *, sequence: bool = False) -> AnalysisObject:
    ds = xr.Dataset(
        {"value": (("trial", "lane", "axis"), values, {"units": "inert"})},
        coords={"trial": np.arange(values.shape[0]), "lane": ["a", "b"], "axis": ["x", "y"]},
    )
    if sequence:
        return AnalysisObject.from_data(
            ds.rename(trial="sample"), sequence_dim="sample", batch_dims=("lane",), core_dims=("axis",)
        )
    return AnalysisObject.from_data(ds, batch_dims=("trial", "lane"), core_dims=("axis",))


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("dtype", ["int64", "float32", "float64", "complex64"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("multi_dim", [False, True])
def test_empty_extrema_preserve_remaining_topology_and_laziness(
    op: str, skipna: bool, dtype: str, lazy: bool, multi_dim: bool
) -> None:
    """ID: REDUCE_EMPTY_043_shared_extrema_missing_semantics."""
    calls: list[str] = []

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return np.empty((0, 2, 2), dtype=dtype)

    values = da.from_delayed(payload(), shape=(0, 2, 2), dtype=dtype) if lazy else np.empty((0, 2, 2), dtype=dtype)
    source = _source(values)
    dims = ("trial", "lane") if multi_dim else ("trial",)
    actual = getattr(source, op)(dim=dims, skipna=skipna).as_dataset(copy="none")
    assert calls == []
    assert isinstance(actual["value"].data, da.Array) == lazy
    assert actual["value"].dtype == np.dtype("float64" if dtype == "int64" else dtype)
    assert actual["value"].attrs == {"units": "inert"}
    assert actual["value"].dims == (("axis",) if multi_dim else ("lane", "axis"))
    np.testing.assert_array_equal(actual.coords["axis"], ["x", "y"])
    assert "trial" not in actual.dims
    assert actual.attrs["tal"]["core"]["roles"] == {
        "batch_dims": [] if multi_dim else ["lane"], "core_dims": ["axis"]
    }
    assert bool(actual["value"].isnull().all().compute())


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_grouped_extrema_include_missing_empty_bins(op: str, skipna: bool, sequence: bool, lazy: bool) -> None:
    """ID: REDUCE_EMPTY_044_grouped_extrema_include_empty_bins."""
    values = np.arange(8.0).reshape(2, 2, 2)
    source = _source(da.from_array(values) if lazy else values, sequence=sequence)
    row_dim = "sample" if sequence else "trial"
    ds = source.as_dataset().assign_coords(score=(row_dim, [0.2, 0.4]))
    if sequence:
        # Sequence keys must include the primary batch lane.
        ds = ds.assign_coords(score=(("lane", "sample"), [[0.2, 0.4], [0.2, 0.4]]))
    source = AnalysisObject.from_data(ds)
    grouped = source.group.groupby_bins("score", [0.0, 1.0, 2.0], labels=["present", "empty"])
    opts = GroupMaterializeOptions() if sequence else BatchGroupReduceOptions(include_empty_groups=True)
    actual = getattr(grouped, op)(opts=opts, skipna=skipna).as_dataset(copy="none")
    assert isinstance(actual["value"].data, da.Array) == lazy
    np.testing.assert_array_equal(actual.coords["group_key"], ["present", "empty"])
    assert bool(actual["value"].sel(group_key="empty").isnull().all())
    assert bool(actual["value"].sel(group_key="present").notnull().all())


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("empty_source", [False, True])
@pytest.mark.parametrize("topology", ["batch", "sequence", "preserved"])
@pytest.mark.parametrize("lazy", [False, True])
def test_grouped_extrema_with_no_rows_have_empty_group_axis(
    op: str, empty_source: bool, topology: str, lazy: bool
) -> None:
    """ID: REDUCE_EMPTY_045_zero_group_extrema_have_truthful_topology."""
    sequence = topology != "batch"
    row_dim = "sample" if sequence else "trial"
    rows = 0 if empty_source else 2
    values = da.ones(rows) if lazy else np.ones(rows)
    source = AnalysisObject.from_data(
        xr.Dataset({"value": (row_dim, values)}, coords={"key": (row_dim, [None] * rows)}),
        sequence_dim=row_dim if sequence else None,
        batch_dims=() if sequence else (row_dim,), core_dims=(),
    )
    grouped = source.group.groupby(
        "key", opts=GroupByOptions(
            preserve_batch=topology == "preserved",
            foundation_opts=GroupingFoundationOptions(na_key_policy="drop"),
        )
    )
    actual = getattr(grouped, op)(skipna=False).as_dataset(copy="none")
    assert isinstance(actual["value"].data, da.Array) == lazy
    assert actual.sizes["group_key"] == 0
    assert actual["value"].dims == ("group_key",)
    assert row_dim not in actual.dims


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("lazy", [False, True])
def test_empty_extrema_do_not_change_other_dimension_or_weight_rules(op: str, lazy: bool) -> None:
    """ID: REDUCE_EMPTY_046_extrema_keep_noop_nonempty_and_weight_rules."""
    values = np.empty((0, 2, 2), dtype=np.int64)
    source = _source(da.from_array(values) if lazy else values)
    ds = source.as_dataset(copy="none")
    noop = getattr(source, op)(dim=()).as_dataset(copy="none")
    xr.testing.assert_identical(noop["value"], ds["value"])
    remaining_empty = getattr(source, op)(dim="axis").as_dataset(copy="none")
    assert remaining_empty["value"].dims == ("trial", "lane")
    assert remaining_empty["value"].dtype == np.dtype("int64")
    with pytest.raises(ValueError, match="weights are supported only"):
        getattr(source, op)(dim="trial", weights=np.empty(0))
    nonempty = _source(np.arange(8.0).reshape(2, 2, 2))
    actual = getattr(nonempty, op)(dim="trial").as_dataset(copy="none")["value"]
    expected = getattr(nonempty.as_dataset(copy="none")["value"], op)(dim="trial", keep_attrs=True)
    xr.testing.assert_identical(actual, expected)
