from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.param_engine import ParamMap, apply_param_map
from tal.core.param_engine.blocking import prepare_logical_row_blocks
from tal.core.param_engine.physical_partition import prepare_physical_row_partitions


def _logical(*values: xr.DataArray):
    return prepare_logical_row_blocks(
        *values,
        excluded_dims=frozenset(),
        fastest_dim="query",
    )


def test_param_core_physical_partition_001_respects_native_chunk_cuts_and_order() -> None:
    """ID: PARAM_CORE_PHYSICAL_PARTITION_001_native_chunks_and_public_order."""
    da = pytest.importorskip("dask.array")
    left = xr.DataArray(da.zeros((5, 7), chunks=(2, 3)), dims=("trial", "query"))
    right = xr.DataArray(da.zeros((5, 7), chunks=(3, 4)), dims=("trial", "query"))
    logical = _logical(left, right)
    plan = prepare_physical_row_partitions(logical, left, right)

    assert plan.logical is logical
    assert plan.execution.grid_shape == (4, 4)
    assert max(partition.row_count for partition in plan.partitions) <= 65_536
    positions = {
        partition.global_row_position(local)
        for partition in plan.partitions
        for local in range(partition.row_count)
    }
    assert positions == set(range(35))


@pytest.mark.parametrize("size", (0, 1, 65_535, 65_536, 65_537))
def test_param_core_physical_partition_001_preserves_logical_bound(size: int) -> None:
    value = xr.DataArray(np.empty((size, 1)), dims=("trial", "query"))
    logical = _logical(value)
    plan = prepare_physical_row_partitions(logical, value)
    assert plan.has_no_rows is (size == 0)
    assert sum(partition.row_count for partition in plan.partitions) == size
    assert all(partition.row_count <= 65_536 for partition in plan.partitions)


def test_param_lazy_physical_partition_001_planning_executes_no_payload_tasks() -> None:
    """ID: PARAM_LAZY_PHYSICAL_PARTITION_001_metadata_only_partitioning."""
    da = pytest.importorskip("dask.array")
    from dask import delayed
    from dask.callbacks import Callback

    source = delayed(lambda: np.ones((4, 6), dtype=np.float64))()
    value = xr.DataArray(da.from_delayed(source, shape=(4, 6), dtype=float), dims=("trial", "query"))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        logical = _logical(value)
        plan = prepare_physical_row_partitions(logical, value)
    assert tasks == []
    assert len(plan.partitions) == 1


def test_param_lazy_physical_partition_001_map_application_preserves_chunks_and_values() -> None:
    da = pytest.importorskip("dask.array")
    values = xr.DataArray(
        da.from_array(np.arange(40, dtype=float).reshape(4, 10), chunks=(2, 5)),
        dims=("trial", "sample"),
        coords={"trial": ["a", "b", "c", "d"]},
    )
    index = xr.DataArray(
        da.from_array(np.tile(np.arange(7), (4, 1)), chunks=(1, 3)),
        dims=("trial", "query"),
        coords={"trial": ["a", "b", "c", "d"]},
    )
    mapping = ParamMap(index, index, xr.zeros_like(index, dtype=float), xr.ones_like(index, dtype=bool), "query")
    result = apply_param_map(values, param_map=mapping, sequence_dim="sample")
    expected = np.arange(40, dtype=float).reshape(4, 10)[:, :7]

    assert result.chunks is not None
    assert max(result.chunksizes["trial"]) <= 2
    assert max(result.chunksizes["query"]) <= 3
    np.testing.assert_array_equal(result.compute(scheduler="synchronous"), expected)
    assert result.get_index("trial").equals(values.get_index("trial"))


def test_param_lazy_physical_partition_001_public_evaluation_reuses_source_work() -> None:
    da = pytest.importorskip("dask.array")
    from dask import delayed
    from dask.callbacks import Callback

    executed: list[str] = []

    @delayed
    def source() -> np.ndarray:
        executed.append("source")
        return np.arange(40, dtype=float).reshape(4, 10)

    payload = da.from_delayed(source(), shape=(4, 10), dtype=float)
    ds = xr.Dataset(
        {"value": (("trial", "sample"), payload)},
        coords={"trial": list("abcd"), "sample": np.arange(10), "time": ("sample", np.arange(10.0))},
    )
    value = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        param_coord="time",
    )
    query = xr.DataArray(
        da.from_array(np.linspace(0.0, 9.0, 7), chunks=3),
        dims="query",
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = value.param.at(query)

    assert tasks == []
    assert executed == []
    computed = result.as_dataset(copy="none").compute(scheduler="synchronous")
    assert computed.sizes == {"trial": 4, "sample": 7}
    assert executed == ["source"]


@pytest.mark.parametrize("lazy_source,lazy_map", ((True, False), (False, True), (True, True)))
@pytest.mark.parametrize("queries", (0, 4))
def test_param_lazy_physical_partition_001_broadcast_core_and_mixed_storage(
    lazy_source: bool, lazy_map: bool, queries: int,
) -> None:
    from dask.callbacks import Callback

    raw = np.arange(36, dtype=np.float32).reshape(2, 3, 6)
    values = xr.DataArray(
        raw, dims=("feature", "trial", "sample"),
        coords={"feature": ["x", "y"], "trial": [20, 10, 30]},
        attrs={"units": "unchanged"},
    )
    index = xr.DataArray(
        np.tile([0, 1, 4, 8], (3, 1))[:, :queries],
        dims=("trial", "query"), coords={"trial": [20, 10, 30]},
    )
    if lazy_source:
        values = values.chunk({"feature": 1, "trial": 2, "sample": 2})
    if lazy_map:
        index = index.chunk({"trial": 1, "query": 2})
    valid = index < 8
    mapping = ParamMap(index, index + 1, xr.full_like(index, 0.25, dtype=float), valid, "query")
    before = values.copy(deep=True)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = apply_param_map(values, param_map=mapping, sequence_dim="sample")
    assert tasks == []
    assert result.dims == ("feature", "trial", "query")
    assert result.chunks is not None
    expected = np.tile([0.25, 1.25, 4.25, np.nan], (2, 3, 1))[:, :, :queries]
    expected += raw[..., :1]
    np.testing.assert_allclose(result.compute(scheduler="synchronous"), expected)
    for dim in ("feature", "trial"):
        assert result.xindexes[dim].equals(values.xindexes[dim])
    assert result.attrs == values.attrs
    xr.testing.assert_identical(values, before)
