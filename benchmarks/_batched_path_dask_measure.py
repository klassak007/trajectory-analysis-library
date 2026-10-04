"""Separate graph, execution, and allocation measurements for 134C."""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc

import numpy as np
import xarray as xr

from benchmarks._batched_path_process_protocol import dataset_chunks
from tal.core.schema_read import read_param_coord_name, read_roles
from tal.core.schema_validate import validate_schema

from .bench_capstone_workflow import (
    CapstoneConfig,
    capstone_fixture,
    direct_xarray_route,
    materialize,
    public_tal_route,
    task_count,
    validate_outputs,
)


def _growth_result(source: xr.Dataset, *, edges: int, query_chunk: int) -> xr.Dataset:
    from tal.frames import FrameGraph
    from tal.spatial import Pose, solve_pose_path_transform
    from tal.utils.frame_schema import set_frames

    graph = FrameGraph()
    for edge in range(edges):
        parent, child = f"f{edge}", f"f{edge + 1}"
        tagged = set_frames(source, parent=parent, child=child, validate=False)
        Pose(tagged, parent=parent, child=child, graph=graph).register()
    query = xr.DataArray(
        [[2.0, 4.0, 6.0, 8.0]], dims=("trial", "when"),
        coords={"trial": source.coords["trial"]},
    )
    if query_chunk < 4:
        query = query.chunk({"when": query_chunk})
    return solve_pose_path_transform(
        f"f{edges}", "f0", graph=graph, query=query,
    ).as_dataset(copy="none")


def _dataset_tasks(value: xr.Dataset) -> int:
    keys: set[object] = set()
    for variable in value.data_vars.values():
        graph = getattr(variable.data, "dask", None)
        if graph is not None:
            keys.update(graph.keys())
    return len(keys)


def _growth_reference(source: xr.Dataset, *, edges: int) -> xr.Dataset:
    """Use xarray interpolation and the fixture's identity rotations."""
    query = xr.DataArray(
        [[2.0, 4.0, 6.0, 8.0]], dims=("trial", "query"),
        coords={"trial": source.coords["trial"], "query": np.arange(4)},
    )
    position = source["position"].swap_dims({"sample": "time"}).drop_vars("sample")
    position = (edges * position.interp(time=query)).rename(time="query_value")
    shape = (source.sizes["trial"], query.sizes["query"], source.sizes["quat"])
    rotation = np.broadcast_to((0.0, 0.0, 0.0, 1.0), shape).copy()
    return xr.Dataset(
        {
            "position": position,
            "rotation": (("trial", "query", "quat"), rotation),
        },
        coords={"quat": source.coords["quat"]},
    )


def _validate_growth_result(observed: xr.Dataset, reference: xr.Dataset, *, edges: int) -> None:
    xr.testing.assert_allclose(observed, reference)
    assert tuple(observed.xindexes) == ("trial", "query", "axis", "quat")
    assert read_roles(observed) == (True, "query", ("trial",), ("axis", "quat"))
    assert read_param_coord_name(observed) == "query_value"
    assert observed.attrs["tal"]["ext"]["frames"] == {"parent": "f0", "child": f"f{edges}"}
    validate_schema(observed)


def _graph_growth_report() -> dict:
    from .bench_batched_fused_path_reference import prepare_reference_fixture

    config = CapstoneConfig(trials=1, ship_samples=5, drone_samples=5, trial_chunk=1, sample_chunk=5)
    lazy = prepare_reference_fixture(config, lazy=True).provider.as_dataset(copy="none")
    eager = lazy.compute(scheduler="synchronous")
    cases = ((1, 4), (1, 1), (4, 4))
    counts = {}
    for edges, chunk in cases:
        observed = _growth_result(lazy, edges=edges, query_chunk=chunk)
        reference = _growth_reference(eager, edges=edges)
        _validate_growth_result(observed.compute(scheduler="synchronous"), reference, edges=edges)
        counts[f"edges_{edges}_query_chunk_{chunk}"] = _dataset_tasks(observed)
    one = counts["edges_1_query_chunk_4"]
    return {
        "unioned_tasks": counts,
        "partition_growth_pass": one > 0 and counts["edges_1_query_chunk_1"] <= 5 * one,
        "edge_growth_pass": one > 0 and counts["edges_4_query_chunk_4"] <= 5 * one,
    }


def _timed(operation, expected) -> float:
    gc.collect()
    started = time.perf_counter()
    value = operation()
    elapsed = time.perf_counter() - started
    validate_outputs(value, expected)
    return elapsed


def _alternating(public, direct, expected, *, warmups: int, repeats: int):
    operations = {"public": public, "direct": direct}
    for _ in range(warmups):
        for operation in operations.values():
            validate_outputs(operation(), expected)
    samples = {name: [] for name in operations}
    for repeat in range(repeats):
        order = tuple(operations) if repeat % 2 == 0 else tuple(reversed(operations))
        for name in order:
            samples[name].append(_timed(operations[name], expected))
    return {
        name: {"seconds": rows, "median_seconds": statistics.median(rows)}
        for name, rows in samples.items()
    }


def _single_worker_peak(operation, expected) -> int:
    gc.collect()
    tracemalloc.start()
    value = operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    validate_outputs(value, expected)
    return peak


def dask_134c_report(config: CapstoneConfig, *, warmups: int, repeats: int) -> dict:
    """Measure the frozen public and direct Dask routes with one protocol."""
    lazy = capstone_fixture(config, lazy=True)
    eager = capstone_fixture(config, lazy=False)
    expected = direct_xarray_route(eager)
    build = _alternating(
        lambda: public_tal_route(lazy),
        lambda: direct_xarray_route(lazy),
        expected,
        warmups=warmups,
        repeats=repeats,
    )
    public = public_tal_route(lazy)
    direct = direct_xarray_route(lazy)
    execution = _alternating(
        lambda: materialize(public),
        lambda: materialize(direct),
        expected,
        warmups=warmups,
        repeats=repeats,
    )
    counts = {"public": task_count(public), "direct": task_count(direct)}
    chunks = {
        "public": dataset_chunks(public.relative_distance),
        "direct": dataset_chunks(direct.relative_distance),
    }
    ratios = {
        "graph_build": build["public"]["median_seconds"] / build["direct"]["median_seconds"],
        "materialization": execution["public"]["median_seconds"] / execution["direct"]["median_seconds"],
        "tasks": counts["public"] / counts["direct"],
    }
    growth = _graph_growth_report()
    gates = {
        "graph_build": ratios["graph_build"] <= 4.0,
        "materialization": ratios["materialization"] <= 1.75,
        "tasks": ratios["tasks"] <= 1.75,
        "partition_growth": growth["partition_growth_pass"],
        "edge_growth": growth["edge_growth_pass"],
    }
    return {
        "graph_build": build,
        "prebuilt_materialization": execution,
        "unioned_tasks": counts,
        "chunks": chunks,
        "single_worker_peak_bytes": _single_worker_peak(lambda: materialize(public), expected),
        "graph_growth": growth,
        "gate_ratios": ratios,
        "gate_pass": gates,
        "all_gates_pass": all(gates.values()),
    }


__all__ = ["dask_134c_report"]
