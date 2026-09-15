from __future__ import annotations

import gc
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal.core.param_engine import ParamMapOptions, apply_param_map, build_param_map
from tal.core.param_engine.prepared import prepare_param_evaluation


class _UnevaluatedQueryTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size: int, calls: list[str]) -> None:
        self.calls = calls
        super().__init__(("query",), {"query": size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("forward")
        return {"query": dim_positions["query"] + 0.25}

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("reverse")
        return {"query": coord_labels["query"] - 0.25}

    def equals(self, other: object, **kwargs: object) -> bool:
        _ = kwargs
        return (
            isinstance(other, _UnevaluatedQueryTransform)
            and self.dim_size == other.dim_size
        )


class _ExceptionalEquality:
    def __eq__(self, other: object) -> bool:
        _ = other
        raise OSError("comparison is intentionally unavailable")


def _native_query(size: int, kind: str, calls: list[str]) -> xr.DataArray:
    if kind == "range":
        index = xr.indexes.RangeIndex.arange(size, dim="query")
    else:
        index = xr.indexes.CoordinateTransformIndex(
            _UnevaluatedQueryTransform(size, calls)
        )
    return xr.DataArray(
        np.linspace(0.0, 1.0, size),
        dims="query",
        coords=xr.Coordinates.from_xindex(index),
    )


def _prepare(
    param: xr.DataArray,
    query: xr.DataArray,
    *,
    reuse=(),
    kind: str = "numeric",
):
    valid = param.notnull()
    return prepare_param_evaluation(
        param=param,
        query=query,
        sequence_dim="sample",
        batch_dims=(),
        batch_coords=None,
        valid_mask=valid,
        options=ParamMapOptions(method="linear", duplicate_policy="invalid"),
        param_kind=kind,
        query_dim="query",
        reuse=reuse,
    )


@pytest.mark.parametrize("kind", ("numeric", "datetime64"))
def test_param_core_prepared_eval_001_request_local_equivalence_is_exact(kind: str) -> None:
    """ID: PARAM_CORE_PREPARED_EVAL_001_request_local_equivalence_is_exact."""
    if kind == "datetime64":
        param = xr.DataArray(np.arange(4).astype("timedelta64[D]") + np.datetime64("2025-01-01"), dims="sample")
        query = xr.DataArray(param.data[[0, 2]], dims="query")
    else:
        param = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims="sample")
        query = xr.DataArray([0.25, 2.25], dims="query")
    first = _prepare(param, query, kind=kind)
    equivalent = _prepare(param.copy(deep=True), query.copy(deep=True), reuse=(first,), kind=kind)
    changed = _prepare(param.copy(deep=True), query.roll(query=1), reuse=(first,), kind=kind)
    assert equivalent is first
    assert changed is not first


def test_param_hard_prepared_eval_001_lazy_reuse_requires_proven_identity() -> None:
    """ID: PARAM_HARD_PREPARED_EVAL_001_lazy_reuse_requires_proven_identity."""
    pytest.importorskip("dask.array")
    param = xr.DataArray([0.0, 1.0, 2.0], dims="sample").chunk({"sample": 3})
    query = xr.DataArray([0.25, 1.5], dims="query").chunk({"query": 2})
    valid = param.notnull()
    first = prepare_param_evaluation(
        param=param,
        query=query,
        sequence_dim="sample",
        batch_dims=(),
        batch_coords=None,
        valid_mask=valid,
        options=ParamMapOptions(),
        param_kind="numeric",
        query_dim="query",
    )
    same = prepare_param_evaluation(
        param=param,
        query=query,
        sequence_dim="sample",
        batch_dims=(),
        batch_coords=None,
        valid_mask=valid,
        options=ParamMapOptions(),
        param_kind="numeric",
        query_dim="query",
        reuse=(first,),
    )
    copied = _prepare(param.copy(deep=True), query.copy(deep=True), reuse=(first,))
    assert same is first
    assert copied is not first


def test_param_core_prepared_eval_002_policy_validity_and_indexes_are_proven() -> None:
    """ID: PARAM_CORE_PREPARED_EVAL_002_policy_validity_and_indexes_are_proven."""
    param = xr.DataArray(
        [0.0, 1.0, 2.0],
        dims="sample",
        coords={"sample_label": ("sample", ["a", "b", "c"])},
    ).set_xindex("sample_label")
    query = xr.DataArray([0.25, 1.5], dims="query")
    first = _prepare(param, query)
    changed_valid = param.notnull()
    changed_valid[1] = False
    validity_plan = prepare_param_evaluation(
        param=param,
        query=query,
        sequence_dim="sample",
        batch_dims=(),
        batch_coords=None,
        valid_mask=changed_valid,
        options=ParamMapOptions(method="linear", duplicate_policy="invalid"),
        param_kind="numeric",
        query_dim="query",
        reuse=(first,),
    )
    policy_plan = prepare_param_evaluation(
        param=param,
        query=query,
        sequence_dim="sample",
        batch_dims=(),
        batch_coords=None,
        valid_mask=param.notnull(),
        options=ParamMapOptions(method="nearest", duplicate_policy="invalid"),
        param_kind="numeric",
        query_dim="query",
        reuse=(first,),
    )
    renamed = param.rename(sample_label="other_label")
    index_plan = _prepare(renamed, query, reuse=(first,))
    assert validity_plan is not first
    assert policy_plan is not first
    assert index_plan is not first


def test_param_core_prepared_eval_003_query_topology_participates_in_reuse() -> None:
    """ID: PARAM_CORE_PREPARED_EVAL_003_query_topology_participates_in_reuse."""
    param = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims="sample")
    stacked_query = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("row", "column"),
        coords={"row": ["a", "b"], "column": [10, 20]},
    )
    stacked = _prepare(param, stacked_query)
    flat_query = stacked.grid.values.copy(deep=True)
    flat_fresh = _prepare(param, flat_query)
    flat = _prepare(param, flat_query, reuse=(stacked,))
    restacked = _prepare(param, stacked_query.copy(deep=True), reuse=(flat_fresh,))

    assert stacked.grid.stacked_dims == ("row", "column")
    assert flat_fresh.grid.stacked_dims is None
    assert flat is not stacked
    assert restacked.grid.stacked_dims == ("row", "column")
    assert restacked is not flat_fresh


def test_param_core_prepared_eval_004_query_coordinate_topology_participates_in_reuse() -> None:
    """ID: PARAM_CORE_PREPARED_EVAL_004_query_coordinate_topology_participates_in_reuse."""
    param = xr.DataArray([0.0, 1.0, 2.0], dims="sample")
    query = xr.DataArray(
        [0.25, 1.5],
        dims="query",
        coords={"query_label": ("query", ["a", "b"]), "epoch": 7},
    )
    first = _prepare(param, query)
    renamed = _prepare(param, query.rename(query_label="other_label"), reuse=(first,))
    changed_scalar = _prepare(param, query.assign_coords(epoch=8), reuse=(first,))
    changed_attrs = _prepare(param, query.assign_attrs(units="seconds"), reuse=(first,))
    equivalent = _prepare(param, query.copy(deep=True), reuse=(first,))

    assert renamed is not first
    assert changed_scalar is not first
    assert changed_attrs is not first
    assert equivalent is first


def test_param_hard_prepared_equivalence_002_indeterminate_object_coords_decline_reuse() -> None:
    """ID: PARAM_HARD_PREPARED_EQUIVALENCE_002_indeterminate_object_coords_decline_reuse."""
    param = xr.DataArray([0.0, 1.0, 2.0], dims="sample")
    marker = np.asarray([pd.NA, "known"], dtype=object)
    query = xr.DataArray(
        [0.25, 1.5],
        dims="query",
        coords={"marker": ("query", marker)},
    )
    first = _prepare(param, query)
    identical = _prepare(param, query, reuse=(first,))
    copied = _prepare(param, query.copy(deep=True), reuse=(first,))

    assert identical is first
    assert copied is not first


def test_param_hard_prepared_exceptional_equality_001_declines_reuse() -> None:
    """ID: PARAM_HARD_PREPARED_EXCEPTIONAL_EQUALITY_001."""
    param = xr.DataArray([0.0, 1.0], dims="sample")
    marker = np.asarray([_ExceptionalEquality()], dtype=object)
    query = xr.DataArray([0.5], dims="query", coords={"marker": ("query", marker)})
    first = _prepare(param, query)

    second = _prepare(param, query.copy(deep=True), reuse=(first,))

    assert second is not first
    assert bool(second.param_map.valid.item())


@pytest.mark.parametrize("lazy", (False, True))
def test_param_hard_map_application_alignment_001_rejects_reordered_batch_labels(
    lazy: bool,
) -> None:
    """ID: PARAM_HARD_MAP_APPLICATION_ALIGNMENT_001."""
    param = xr.DataArray(
        [[0.0, 1.0], [0.0, 1.0]],
        dims=("trial", "sample"),
        coords={"trial": ["a", "b"]},
    )
    mapping = build_param_map(
        param=param,
        query=xr.DataArray([0.5], dims="query"),
        sequence_dim="sample",
        query_dim="query",
    )
    values = xr.DataArray(
        [[10.0, 20.0], [30.0, 40.0]],
        dims=("trial", "sample"),
        coords={"trial": ["b", "a"]},
    )
    tasks: list[object] = []
    if lazy:
        pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        values = values.chunk({"trial": 1, "sample": 2})
        with (
            Callback(pretask=lambda key, *_: tasks.append(key)),
            pytest.raises(ValueError, match="apply_param_map:.*not label-aligned"),
        ):
            apply_param_map(values, param_map=mapping, sequence_dim="sample")
    else:
        with pytest.raises(ValueError, match="apply_param_map:.*not label-aligned"):
            apply_param_map(values, param_map=mapping, sequence_dim="sample")
    assert tasks == []


def test_param_core_block_coordinates_001_preserves_complete_coordinate_topology() -> None:
    """ID: PARAM_CORE_BLOCK_COORDINATES_001."""
    param = xr.DataArray(
        [[0.0, 1.0], [0.0, 1.0]],
        dims=("trial", "sample"),
        coords={"trial": ["a", "b"], "batch_note": ("trial", [3, 4]), "source_id": 9},
    )
    query = xr.DataArray(
        [0.25, 0.75],
        dims="query",
        coords={"query": [10, 20], "tag": ("query", ["x", "y"]), "run": 7},
    )
    query.coords["tag"].attrs["kind"] = "label"
    query.coords["tag"].encoding["source"] = "fixture"
    mapping = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )
    values = xr.DataArray(
        [[2.0, 4.0], [20.0, 40.0]],
        dims=("trial", "sample"),
        coords=param.coords,
    )
    actual = apply_param_map(values, param_map=mapping, sequence_dim="sample")

    for result in (mapping.i0, mapping.i1, mapping.alpha, mapping.valid, actual):
        assert result.coords["tag"].variable.identical(query.coords["tag"].variable)
        assert result.coords["batch_note"].variable.identical(param.coords["batch_note"].variable)
        assert result.coords["run"].item() == 7
        assert result.coords["source_id"].item() == 9
        assert result.coords["tag"].encoding == query.coords["tag"].encoding


@pytest.mark.parametrize("kind", ("range", "transform"))
@pytest.mark.parametrize("size", (65_535, 65_536, 65_537))
def test_param_core_block_index_001_native_query_indexes_survive_blocking(
    kind: str,
    size: int,
) -> None:
    """ID: PARAM_CORE_BLOCK_INDEX_001_native_query_indexes_survive_blocking."""
    calls: list[str] = []
    query = _native_query(size, kind, calls)
    param = xr.DataArray([0.0, 1.0], dims="sample")
    mapping = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )
    axis_index = xr.indexes.RangeIndex.arange(2, dim="axis")
    values = xr.DataArray(
        [[2.0, 20.0], [4.0, 40.0]],
        dims=("sample", "axis"),
        coords=xr.Coordinates.from_xindex(axis_index),
    )
    actual = apply_param_map(
        values,
        param_map=mapping,
        sequence_dim="sample",
    )

    for value in (mapping.i0, mapping.i1, mapping.alpha, mapping.valid, actual):
        assert type(value.xindexes["query"]) is type(query.xindexes["query"])
        assert value.xindexes["query"].equals(query.xindexes["query"])
    assert isinstance(actual.xindexes["axis"], xr.indexes.RangeIndex)
    assert actual.xindexes["axis"].equals(values.xindexes["axis"])
    assert calls == []
    assert float(actual.isel(axis=0, query=0)) == pytest.approx(2.0)
    assert float(actual.isel(axis=0, query=-1)) == pytest.approx(4.0)


@pytest.mark.parametrize("size", (65_535, 65_536, 65_537))
@pytest.mark.parametrize("query_size", (1, 2))
def test_param_core_combined_block_001_batch_query_products_are_bounded(
    size: int,
    query_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_CORE_COMBINED_BLOCK_001_batch_query_products_are_bounded."""
    import tal.core.param_engine.map_build as owner

    original = owner.map_block_backend
    logical_rows: list[int] = []

    def tracked(param: np.ndarray, mask: np.ndarray, query: np.ndarray, **kwargs: object):
        outer = np.broadcast_shapes(param.shape[:-1], query.shape[:-1])
        logical_rows.append(int(np.prod((*outer, query.shape[-1]), dtype=np.int64)))
        return original(param, mask, query, **kwargs)

    monkeypatch.setattr(owner, "map_block_backend", tracked)
    param = xr.DataArray(
        np.broadcast_to([0.0, 1.0], (size, 2)),
        dims=("trial", "sample"),
    )
    query = xr.DataArray(np.linspace(0.0, 1.0, query_size), dims="query")
    mapping = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )

    assert mapping.valid.shape == (size, query_size)
    assert logical_rows
    assert max(logical_rows) <= 65_536


def test_param_core_combined_block_002_multiple_batch_dims_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_CORE_COMBINED_BLOCK_002_multiple_batch_dims_are_bounded."""
    import tal.core.param_engine.map_build as owner

    original = owner.map_block_backend
    logical_rows: list[int] = []

    def tracked(param: np.ndarray, mask: np.ndarray, query: np.ndarray, **kwargs: object):
        outer = np.broadcast_shapes(param.shape[:-1], query.shape[:-1])
        logical_rows.append(int(np.prod((*outer, query.shape[-1]), dtype=np.int64)))
        return original(param, mask, query, **kwargs)

    monkeypatch.setattr(owner, "map_block_backend", tracked)
    param = xr.DataArray(
        np.broadcast_to([0.0, 1.0], (257, 256, 2)),
        dims=("run", "trial", "sample"),
    )
    mapping = build_param_map(
        param=param,
        query=xr.DataArray([0.25, 0.75], dims="query"),
        sequence_dim="sample",
        query_dim="query",
    )

    assert mapping.valid.shape == (257, 256, 2)
    assert logical_rows
    assert max(logical_rows) <= 65_536


def test_param_perf_combined_block_002_dask_tasks_are_bounded_and_lazy() -> None:
    """ID: PARAM_PERF_COMBINED_BLOCK_002_dask_tasks_are_bounded_and_lazy."""
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    size = 65_537
    trial_index = xr.indexes.RangeIndex.arange(size, dim="trial")
    coords = xr.Coordinates.from_xindex(trial_index)
    param = xr.DataArray(
        np.broadcast_to([0.0, 1.0], (size, 2)),
        dims=("trial", "sample"),
        coords=coords,
    ).chunk({"trial": size, "sample": 2})
    query = xr.DataArray([0.25, 0.75], dims="query").chunk({"query": 2})
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )
    assert tasks == []
    assert mapping.valid.chunks is not None
    trial_chunks = mapping.valid.chunks[mapping.valid.get_axis_num("trial")]
    query_chunks = mapping.valid.chunks[mapping.valid.get_axis_num("query")]
    assert max(trial_chunks) * max(query_chunks) <= 65_536
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        computed = mapping.valid.compute(scheduler="synchronous")
    assert tasks
    assert bool(computed.all())
    assert isinstance(mapping.valid.xindexes["trial"], xr.indexes.RangeIndex)


@pytest.mark.parametrize("lazy", (False, True))
def test_param_hard_combined_block_001_failure_precedence_spans_both_partitions(
    lazy: bool,
) -> None:
    """ID: PARAM_HARD_COMBINED_BLOCK_001_failure_precedence_spans_both_partitions."""
    size = 65_537
    values = np.broadcast_to([0.0, 1.0], (size, 2)).copy()
    values[0] = [0.0, 0.0]
    values[-1] = [1.0, 0.0]
    param = xr.DataArray(values, dims=("trial", "sample"))
    query = xr.DataArray([0.0, 0.5], dims="query")
    if lazy:
        pytest.importorskip("dask.array")
        param = param.chunk({"trial": size, "sample": 2})
        query = query.chunk({"query": 2})
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            options=ParamMapOptions(duplicate_policy="raise"),
        )
        with pytest.raises(ValueError, match="duplicate parameter bracket"):
            mapping.valid.compute(scheduler="synchronous")
        return
    with pytest.raises(ValueError, match="duplicate parameter bracket"):
        build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            options=ParamMapOptions(duplicate_policy="raise"),
        )


@pytest.mark.parametrize("lazy", (False, True))
def test_param_hard_ordered_map_failure_001_batch_row_precedes_failure_kind(
    lazy: bool,
) -> None:
    """ID: PARAM_HARD_ORDERED_MAP_FAILURE_001."""
    size = 65_537
    param = xr.DataArray(
        [[0.0, 1.0], [1.0, 0.0]],
        dims=("trial", "sample"),
        coords={"trial": ["first", "second"]},
    )
    query_values = np.zeros((2, size), dtype=np.uint64)
    query_values[0, -1] = np.uint64(2**63 + 1)
    query = xr.DataArray(
        query_values,
        dims=("trial", "query"),
        coords={"trial": ["first", "second"]},
    )
    if lazy:
        pytest.importorskip("dask.array")
        param = param.chunk({"trial": 2, "sample": 2})
        query = query.chunk({"trial": 2, "query": size})
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )
        with pytest.raises(ValueError, match=str(2**63 + 1)):
            mapping.valid.compute(scheduler="synchronous")
        return
    with pytest.raises(ValueError, match=str(2**63 + 1)):
        build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )


@pytest.mark.parametrize("lazy", (False, True))
def test_param_hard_ordered_map_failure_001_datetime_span_respects_row_order(
    lazy: bool,
) -> None:
    """Exercise the ordered-failure family with datetime span competition."""
    size = 65_537
    param = xr.DataArray(
        np.asarray(
            [
                ["2020-01-01", "2020-01-01", "2020-01-03"],
                [
                    "1677-09-21T00:12:43.145224193",
                    "2000-01-01",
                    "2262-04-11T23:47:16.854775807",
                ],
            ],
            dtype="datetime64[ns]",
        ),
        dims=("trial", "sample"),
    )
    query_values = np.full((2, size), np.datetime64("2020-01-02", "ns"))
    query_values[0, -1] = np.datetime64("2020-01-01", "ns")
    query = xr.DataArray(query_values, dims=("trial", "query"))
    options = ParamMapOptions(method="linear", duplicate_policy="raise")
    if lazy:
        pytest.importorskip("dask.array")
        param = param.chunk({"trial": 1, "sample": 3})
        query = query.chunk({"trial": 1, "query": size})
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            options=options,
            param_kind="datetime64",
        )
        with pytest.raises(ValueError, match="duplicate parameter bracket"):
            mapping.valid.compute(scheduler="synchronous")
        return
    with pytest.raises(ValueError, match="duplicate parameter bracket"):
        build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            options=options,
            param_kind="datetime64",
        )


def _map_peak(param: xr.DataArray, query: xr.DataArray) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert mapping.valid.shape == (param.sizes["trial"], query.sizes["query"])
    return peak


def test_param_perf_combined_block_001_boundary_allocation_has_no_cartesian_jump() -> None:
    """ID: PARAM_PERF_COMBINED_BLOCK_001_boundary_allocation_has_no_cartesian_jump."""
    small_param = xr.DataArray(
        np.broadcast_to([0.0, 1.0], (65_536, 2)),
        dims=("trial", "sample"),
    )
    large_param = xr.DataArray(
        np.broadcast_to([0.0, 1.0], (65_537, 2)),
        dims=("trial", "sample"),
    )
    query = xr.DataArray([0.5], dims="query")
    build_param_map(
        param=xr.DataArray([[0.0, 1.0]], dims=("trial", "sample")),
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )

    small = _map_peak(small_param, query)
    large = _map_peak(large_param, query)
    assert large - small < 8 * 1024**2, (small, large)


def _apply_excess_peak(mapping, *, query_size: int) -> int:
    values = xr.DataArray([2.0, 4.0], dims="sample")
    gc.collect()
    tracemalloc.start()
    try:
        result = apply_param_map(values, param_map=mapping, sequence_dim="sample")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.shape == (query_size,)
    return peak - result.nbytes


def test_param_perf_eager_block_lifetime_001_retains_only_one_working_block() -> None:
    """ID: PARAM_PERF_EAGER_BLOCK_LIFETIME_001."""
    sizes = (65_536, 16 * 65_536)
    mappings = tuple(
        build_param_map(
            param=xr.DataArray([0.0, 1.0], dims="sample"),
            query=xr.DataArray(np.linspace(0.0, 1.0, size), dims="query"),
            sequence_dim="sample",
            query_dim="query",
        )
        for size in sizes
    )
    apply_param_map(
        xr.DataArray([2.0, 4.0], dims="sample"),
        param_map=mappings[0],
        sequence_dim="sample",
    )

    excess = tuple(
        _apply_excess_peak(mapping, query_size=size)
        for mapping, size in zip(mappings, sizes, strict=True)
    )
    assert excess[1] - excess[0] < 4 * 1024**2, excess


@pytest.mark.parametrize("size", (0, 1, 65_535, 65_536, 65_537))
def test_param_core_block_apply_001_boundary_sizes_preserve_values(size: int) -> None:
    """ID: PARAM_CORE_BLOCK_APPLY_001_boundary_sizes_preserve_values."""
    param = xr.DataArray([0.0, 1.0], dims="sample")
    query = xr.DataArray(
        np.linspace(0.0, 1.0, size),
        dims="query",
        coords={"query_label": ("query", np.arange(size))},
    ).set_xindex("query_label")
    mapping = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )
    values = xr.DataArray([2.0, 4.0], dims="sample")
    actual = apply_param_map(values, param_map=mapping, sequence_dim="sample")
    expected = 2.0 + 2.0 * query
    xr.testing.assert_allclose(actual, expected)
    assert mapping.i0.xindexes["query_label"].equals(query.xindexes["query_label"])
    assert actual.xindexes["query_label"].equals(query.xindexes["query_label"])


@pytest.mark.parametrize("size", (0, 65_537))
def test_param_perf_block_apply_001_dask_partitions_before_gathering(size: int) -> None:
    """ID: PARAM_PERF_BLOCK_APPLY_001_dask_partitions_before_gathering."""
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    param = xr.DataArray([0.0, 1.0], dims="sample").chunk({"sample": 2})
    query = xr.DataArray(np.linspace(0.0, 1.0, size), dims="query").chunk({"query": max(size, 1)})
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )
        actual = apply_param_map(
            xr.DataArray([2.0, 4.0], dims="sample").chunk({"sample": 2}),
            param_map=mapping,
            sequence_dim="sample",
        )
    assert tasks == []
    assert mapping.i0.chunks is not None
    assert actual.chunks is not None
    query_axis = actual.get_axis_num("query")
    assert max(actual.chunks[query_axis]) <= 65_536
    np.testing.assert_allclose(actual.compute(), 2.0 + 2.0 * query.compute())
