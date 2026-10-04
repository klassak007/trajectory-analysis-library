"""Public behavior of the 134C lazy batched spatial path route."""

from __future__ import annotations

import time
from unittest.mock import patch

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from benchmarks.bench_batched_fused_path_reference import prepare_reference_fixture
from benchmarks.bench_capstone_workflow import CapstoneConfig
from tal.frames import FrameGraph
from tal.spatial import Pose, solve_pose_path_transform
from tests.core.test_spatial_batched_path_execution import (
    _prepare_batched_between,
    _renamed_position_caller,
)
from tests.core.test_spatial_path_execution import _edge_pose, _registered_path


@pytest.mark.parametrize("edges", (1, 2, 3))
@pytest.mark.parametrize("reverse", (False, True))
def test_spatial_lazy_batched_path_134c_pose_parity(edges: int, reverse: bool) -> None:
    lazy_graph, lazy_providers = _registered_path(edges, lazy=True)
    eager_graph, _ = _registered_path(edges)
    query = xr.DataArray(
        [[0.2, 0.8], [0.3, 0.7]], dims=("trial", "when"),
        coords={"trial": ["a", "b"], "when": [10, 20]},
    ).chunk({"trial": 1, "when": 1})
    source, destination = ("f0", f"f{edges}") if reverse else (f"f{edges}", "f0")
    plan = _prepare_batched_between(lazy_graph, source, destination, query)
    assert plan.kind == "batched-dask"
    before = tuple(provider.as_dataset(copy="none") for provider in lazy_providers)
    with patch(
        "tal.spatial.ops.path_execution._numba_available",
        side_effect=AssertionError("client Numba check"),
    ):
        actual = solve_pose_path_transform(source, destination, graph=lazy_graph, query=query)
    lazy_ds = actual.as_dataset(copy="none")
    assert all(var.chunks is not None for var in lazy_ds.data_vars.values())
    actual_ds = lazy_ds.compute(scheduler="synchronous")
    expected = solve_pose_path_transform(
        source, destination, graph=eager_graph, query=query.compute(),
    ).as_dataset(copy="none")
    xr.testing.assert_allclose(actual_ds, expected)
    for provider, snapshot in zip(lazy_providers, before, strict=True):
        xr.testing.assert_identical(provider.as_dataset(copy="none"), snapshot)


@pytest.mark.parametrize("batch_dims", (("trial",), ("run", "trial"), ("case", "run", "trial")))
def test_spatial_lazy_batched_path_134c_query_only_batch_layouts(batch_dims) -> None:
    lazy_graph, _ = _registered_path(2, lazy=True)
    eager_graph, _ = _registered_path(2)
    dims = (*batch_dims, "when")
    shape = (2,) * len(dims)
    query = xr.DataArray(
        np.linspace(0.2, 0.8, int(np.prod(shape))).reshape(shape),
        dims=dims,
        coords={dim: [f"{dim}-0", f"{dim}-1"] for dim in batch_dims},
    )
    observed = solve_pose_path_transform(
        "f2", "f0", graph=lazy_graph, query=query.chunk({"when": 1}),
    ).as_dataset(copy="none")
    reference = solve_pose_path_transform(
        "f2", "f0", graph=eager_graph, query=query,
    ).as_dataset(copy="none")
    xr.testing.assert_allclose(observed.compute(scheduler="synchronous"), reference)


def test_spatial_lazy_batched_path_134c_distinct_domains_and_mixed_directions() -> None:
    graphs = (FrameGraph(), FrameGraph())
    for lazy, graph in zip((True, False), graphs, strict=True):
        for edge, child in enumerate(("left", "right")):
            source = _edge_pose(edge).as_dataset(copy="none")
            if edge:
                source = source.assign_coords(time=("sample", np.linspace(0.0, 2.0, 5)))
            if lazy:
                source = source.chunk({"sample": 2 if edge == 0 else 3})
            Pose(source, parent="root", child=child, graph=graph).register()
    query = xr.DataArray([[0.25, 0.75], [0.5, 1.0]], dims=("trial", "when"))
    observed = solve_pose_path_transform(
        "left", "right", graph=graphs[0], query=query.chunk({"when": 1}),
    ).as_dataset(copy="none")
    reference = solve_pose_path_transform(
        "left", "right", graph=graphs[1], query=query,
    ).as_dataset(copy="none")
    xr.testing.assert_allclose(observed.compute(scheduler="synchronous"), reference)


def test_spatial_lazy_batched_path_134c_native_sequence_chunks_are_selected() -> None:
    graph = FrameGraph()
    source = _edge_pose(0).as_dataset(copy="deep")
    calls: list[tuple[str, int]] = []

    @delayed
    def load(values, name, row):
        calls.append((name, row))
        return values

    for name in ("position", "rotation"):
        values = source[name].data
        pieces = [
            da.from_delayed(load(values[row:row + 1], name, row), shape=(1, values.shape[-1]), dtype=float)
            for row in range(5)
        ]
        source[name] = source[name].copy(data=da.concatenate(pieces))
    provider = Pose(source, parent="f0", child="f1", graph=graph)
    provider.register()
    query = xr.DataArray([[0.1, 0.2]], dims=("trial", "when"))
    result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    assert calls == []
    actual = result.as_dataset(copy="none").compute(scheduler="synchronous")
    assert sorted(calls) == [(name, row) for name in ("position", "rotation") for row in (0, 1)]
    eager_graph = FrameGraph()
    eager = Pose(_edge_pose(0), parent="f0", child="f1", graph=eager_graph)
    eager.register()
    expected = solve_pose_path_transform("f1", "f0", graph=eager_graph, query=query)
    xr.testing.assert_allclose(actual, expected.as_dataset(copy="none"))


def test_spatial_lazy_batched_path_134c_independent_requests_share_only_source_work() -> None:
    """ID: SPATIAL_LAZY_BATCHED_GRAPH_001_independent_requests_share_source_work."""
    graph = FrameGraph()
    source = _edge_pose(0).as_dataset(copy="deep")
    calls: list[str] = []

    @delayed
    def load():
        calls.append("source")
        return np.concatenate((source["position"].data, source["rotation"].data), axis=-1)

    payload = da.from_delayed(load(), shape=(5, 7), dtype=float)
    lazy = source.copy(deep=True)
    lazy["position"] = lazy["position"].copy(data=payload[:, :3].rechunk((2, 3)))
    lazy["rotation"] = lazy["rotation"].copy(data=payload[:, 3:].rechunk((3, 4)))
    Pose(lazy, parent="f0", child="f1", graph=graph).register()
    queries = [xr.DataArray([[x, x + 0.1]], dims=("trial", "when")) for x in (0.2, 0.6)]
    results = [solve_pose_path_transform("f1", "f0", graph=graph, query=q) for q in queries]
    assert calls == []
    actual = dask.compute(*(result.as_dataset(copy="none") for result in results), scheduler="synchronous")
    assert calls == ["source"]
    eager_graph, _ = _registered_path(1)
    for observed, query in zip(actual, queries, strict=True):
        expected = solve_pose_path_transform("f1", "f0", graph=eager_graph, query=query)
        xr.testing.assert_allclose(observed, expected.as_dataset(copy="none"))


def test_spatial_lazy_batched_path_134c_slices_native_batch_chunk() -> None:
    config = CapstoneConfig(trials=3, ship_samples=5, drone_samples=9, trial_chunk=3, sample_chunk=2)
    lazy = prepare_reference_fixture(config, lazy=True)
    eager = prepare_reference_fixture(config, lazy=False)
    query = xr.DataArray(
        [[2.0, 4.0], [3.0, 5.0], [6.0, 8.0]],
        dims=("trial", "when"), coords={"trial": [0, 1, 2]},
    )
    observed = solve_pose_path_transform(
        "ship", "world", graph=lazy.provider.graph, query=query.chunk({"trial": 1, "when": 1}),
    ).as_dataset(copy="none")
    reference = solve_pose_path_transform(
        "ship", "world", graph=eager.provider.graph, query=query,
    ).as_dataset(copy="none")
    xr.testing.assert_allclose(observed.compute(scheduler="synchronous"), reference)


@pytest.mark.parametrize("lazy_part", ("position", "rotation", "caller"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_spatial_lazy_batched_path_134c_mixed_storage_position_parity(
    lazy_part, dtype,
) -> None:
    graph = FrameGraph()
    eager_source = _edge_pose(0).as_dataset(copy="none").astype(dtype)
    source = eager_source.copy()
    if lazy_part in {"position", "rotation"}:
        source[lazy_part] = source[lazy_part].chunk({"sample": 2})
    provider = Pose(source, parent="f0", child="f1", graph=graph)
    provider.register()
    caller = _renamed_position_caller(graph, lazy=lazy_part == "caller")
    lazy = caller.to_frame("f0", graph=graph).as_dataset(copy="none")
    assert lazy["point"].chunks is not None
    actual = lazy.compute(scheduler="synchronous")

    eager_graph = FrameGraph()
    eager_provider = Pose(eager_source, parent="f0", child="f1", graph=eager_graph)
    eager_provider.register()
    expected = _renamed_position_caller(eager_graph, lazy=False).to_frame("f0", graph=eager_graph)
    xr.testing.assert_allclose(actual, expected.as_dataset(copy="none"))


@pytest.mark.parametrize("sizes", ((0, 0), (3, 5)))
def test_spatial_lazy_batched_path_134c_ragged_and_all_missing_rows(sizes) -> None:
    lazy_graph, _ = _registered_path(1, lazy=True)
    eager_graph, _ = _registered_path(1)
    observed_caller = _renamed_position_caller(lazy_graph, lazy=True)
    reference_caller = _renamed_position_caller(eager_graph, lazy=False)

    def with_sizes(caller, graph):
        source = caller.as_dataset(copy="none").assign_coords(
            group_size=("trial", np.asarray(sizes, dtype=np.int64)),
        )
        return type(caller)(source, parent="f1", child="point", graph=graph)

    observed = with_sizes(observed_caller, lazy_graph).to_frame(
        "f0", graph=lazy_graph,
    ).as_dataset(copy="none")
    reference = with_sizes(reference_caller, eager_graph).to_frame(
        "f0", graph=eager_graph,
    ).as_dataset(copy="none")
    xr.testing.assert_allclose(observed.compute(scheduler="synchronous"), reference)


def test_spatial_lazy_batched_path_134c_failure_gate_orders_edge_then_public_row() -> None:
    graph = FrameGraph()
    sources = [_edge_pose(edge).as_dataset(copy="deep") for edge in range(2)]
    sources[0]["rotation"].data[3] = 0.0
    sources[1]["rotation"].data[1] = 0.0
    for edge, source in enumerate(sources):
        provider = Pose(
            source.chunk({"sample": 2}),
            parent=f"f{edge}", child=f"f{edge + 1}", graph=graph,
        )
        provider.register()
    query = xr.DataArray([[0.75, 0.25]], dims=("trial", "when")).chunk({"when": 1})
    result = solve_pose_path_transform("f2", "f0", graph=graph, query=query)
    dataset = result.as_dataset(copy="none")
    for component in ("position", "rotation"):
        with pytest.raises(
            ValueError,
            match=r"^spatial\.path_solve\.pose: edge 0, query 1: quaternion norm must be finite and > 0$",
        ) as caught:
            dataset[component].compute(scheduler="threads", num_workers=2)
        assert isinstance(caught.value.__cause__, ValueError)
        assert str(caught.value.__cause__).startswith("edge 0, query 1:")


def test_spatial_lazy_batched_path_134c_failure_order_ignores_completion_order() -> None:
    graph = FrameGraph()
    finished: list[str] = []

    @delayed(pure=False)
    def load(values: np.ndarray, label: str, delay: float) -> np.ndarray:
        time.sleep(delay)
        finished.append(label)
        return values

    for edge in range(2):
        source = _edge_pose(edge).as_dataset(copy="deep")
        invalid = 3 if edge == 0 else 1
        source["rotation"].data[invalid] = 0.0
        values = np.asarray(source["rotation"].data)
        chunks = tuple(
            da.from_delayed(
                load(values[start:end], f"{edge}:{start}", 0.08 if edge == 1 and start == 0 else 0.0),
                shape=(end - start, 4), dtype=values.dtype,
            )
            for start, end in ((0, 2), (2, 4), (4, 5))
        )
        source["rotation"] = source["rotation"].copy(data=da.concatenate(chunks))
        Pose(source, parent=f"f{edge}", child=f"f{edge + 1}", graph=graph).register()
    query = xr.DataArray([[0.75, 0.25]], dims=("trial", "when")).chunk({"when": 1})
    output = solve_pose_path_transform("f2", "f0", graph=graph, query=query)
    with pytest.raises(ValueError, match=r"edge 0, query 1:"):
        output.as_dataset(copy="none")["position"].compute(scheduler="threads", num_workers=4)
    assert finished.index("0:2") < finished.index("1:0")


def test_spatial_lazy_batched_path_134c_zero_rows_launch_no_worker() -> None:
    graph, _ = _registered_path(1, lazy=True)
    query = xr.DataArray(np.empty((0, 2)), dims=("trial", "when"))
    with patch(
        "tal.spatial.ops.batched_path_dask._numerical_partition",
        side_effect=AssertionError("numerical task"),
    ):
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
        dataset = result.as_dataset(copy="none").compute(scheduler="synchronous")
    assert dataset.sizes["trial"] == 0
    assert dataset["position"].shape == (0, 2, 3)
