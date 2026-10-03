"""Compare object-centered path queries with direct typed composition.

This is a manual, non-CI benchmark. Fixtures are allocated before timing and
reported timings and Dask graph sizes are evidence, not pass/fail thresholds.
"""

from __future__ import annotations

import argparse
import gc
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from itertools import product
from statistics import median
from time import perf_counter

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from tal.core import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation, solve_pose_path_transform


@dataclass(frozen=True)
class _PathCase:
    graph: FrameGraph
    source: str
    destination: str
    providers: tuple[Pose, ...]
    query: np.ndarray
    identity: Pose


def _static_components(
    position: np.ndarray | da.Array,
    rotation: np.ndarray | da.Array,
) -> tuple[AnalysisObject, AnalysisObject]:
    position_da = xr.DataArray(
        position,
        dims="axis",
        coords={"axis": ["x", "y", "z"]},
        name="position",
    )
    rotation_da = xr.DataArray(
        rotation,
        dims="quat",
        coords={"quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    return (
        AnalysisObject.from_data(position_da, core_dims=("axis",)),
        AnalysisObject.from_data(rotation_da, core_dims=("quat",)),
    )


def _dynamic_components(
    position: np.ndarray | da.Array,
    rotation: np.ndarray | da.Array,
    parameter: np.ndarray,
) -> tuple[AnalysisObject, AnalysisObject]:
    coords = {"sample": np.arange(parameter.size), "time": ("sample", parameter)}
    position_da = xr.DataArray(
        position,
        dims=("sample", "axis"),
        coords={**coords, "axis": ["x", "y", "z"]},
        name="position",
    )
    rotation_da = xr.DataArray(
        rotation,
        dims=("sample", "quat"),
        coords={**coords, "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    kwargs = {"sequence_dim": "sample", "param_coord": "time"}
    return (
        AnalysisObject.from_data(position_da, core_dims=("axis",), **kwargs),
        AnalysisObject.from_data(rotation_da, core_dims=("quat",), **kwargs),
    )


def _pose(
    position: np.ndarray | da.Array,
    rotation: np.ndarray | da.Array,
    *,
    parameter: np.ndarray | None,
    parent: str | None,
    child: str | None,
    graph: FrameGraph | None,
) -> Pose:
    components = (
        _static_components(position, rotation)
        if parameter is None
        else _dynamic_components(position, rotation, parameter)
    )
    position_ao, rotation_ao = components
    return Pose.from_components(
        Rotation(rotation_ao),
        Position(position_ao),
        parent=parent,
        child=child,
        graph=graph,
    )


def _identity(query: np.ndarray, *, lazy: bool) -> Pose:
    position = np.zeros((query.size, 3), dtype=np.float64)
    rotation = np.zeros((query.size, 4), dtype=np.float64)
    rotation[:, 3] = 1.0
    if lazy:
        position = da.from_array(position, chunks=(min(query.size, 4096), 3))
        rotation = da.from_array(rotation, chunks=(min(query.size, 4096), 4))
    return _pose(
        position,
        rotation,
        parameter=query,
        parent=None,
        child=None,
        graph=None,
    )


def _path_case(
    *,
    edges: int,
    query_size: int,
    dynamic: bool,
    lazy: bool,
) -> _PathCase:
    graph = FrameGraph()
    query = np.linspace(0.0, 1.0, query_size, dtype=np.float64)
    native = np.linspace(0.0, 1.0, 129, dtype=np.float64) if dynamic else None
    providers: list[Pose] = []
    for index in range(edges):
        count = 1 if native is None else native.size
        position = np.zeros((count, 3), dtype=np.float64)
        position[:, 0] = 1.0 if native is None else native
        rotation = np.zeros((count, 4), dtype=np.float64)
        rotation[:, 3] = 1.0
        if lazy:
            position = da.from_array(position, chunks=(min(count, 64), 3))
            rotation = da.from_array(rotation, chunks=(min(count, 64), 4))
        parent, child = f"frame_{index}", f"frame_{index + 1}"
        provider = _pose(
            position[0] if native is None else position,
            rotation[0] if native is None else rotation,
            parameter=native,
            parent=parent,
            child=child,
            graph=graph,
        )
        provider.register()
        providers.append(provider)
    return _PathCase(
        graph,
        f"frame_{edges}",
        "frame_0",
        tuple(providers),
        query,
        _identity(query, lazy=lazy),
    )


def _compose(values: tuple[Pose, ...], *, initial: Pose | None = None) -> Pose:
    result = initial
    for value in values:
        result = value if result is None else result.compose(value, validate=False)
    if result is None:
        raise RuntimeError("benchmark path must contain at least one provider")
    return result


def _direct(case: _PathCase, *, dynamic: bool) -> Pose:
    if dynamic:
        values = tuple(provider.param.at(case.query, validate=False) for provider in case.providers)
        return _compose(values)
    return _compose(case.providers, initial=case.identity)


def _tal(case: _PathCase) -> Pose:
    return solve_pose_path_transform(
        case.source,
        case.destination,
        graph=case.graph,
        query=case.query,
    )


def _graph_tasks(value: Pose) -> int:
    keys: set[object] = set()
    for variable in value.as_dataset(copy="none").data_vars.values():
        graph = variable.data.__dask_graph__() if hasattr(variable.data, "__dask_graph__") else None
        if graph is not None:
            keys.update(graph.keys())
    return len(keys)


def _time(operation: Callable[[], Pose]) -> tuple[float, Pose]:
    gc.collect()
    started = perf_counter()
    result = operation()
    return perf_counter() - started, result


def _measure(
    case: _PathCase,
    *,
    dynamic: bool,
    warmups: int,
    repeats: int,
) -> tuple[float, float, int, int]:
    operations = (("direct", lambda: _direct(case, dynamic=dynamic)), ("tal", lambda: _tal(case)))
    for _ in range(warmups):
        for _, operation in operations:
            operation()
    samples = {name: [] for name, _ in operations}
    results: dict[str, Pose] = {}
    for index in range(repeats):
        ordered = operations if index % 2 == 0 else tuple(reversed(operations))
        for name, operation in ordered:
            elapsed, results[name] = _time(operation)
            samples[name].append(elapsed)
    return (
        median(samples["direct"]),
        median(samples["tal"]),
        _graph_tasks(results["direct"]),
        _graph_tasks(results["tal"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(1_000, 100_000, 1_000_000))
    parser.add_argument("--edges", type=int, nargs="+", default=(1, 4, 8))
    parser.add_argument(
        "--modes",
        choices=("static", "dynamic"),
        nargs="+",
        default=("static", "dynamic"),
    )
    parser.add_argument(
        "--backends",
        choices=("eager", "dask"),
        nargs="+",
        default=("eager", "dask"),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def _run_case(
    *,
    dynamic: bool,
    lazy: bool,
    edges: int,
    size: int,
    warmups: int,
    repeats: int,
) -> None:
    case = _path_case(edges=edges, query_size=size, dynamic=dynamic, lazy=lazy)
    direct, tal, direct_tasks, tal_tasks = _measure(
        case,
        dynamic=dynamic,
        warmups=warmups,
        repeats=repeats,
    )
    print(
        f"mode={'dynamic' if dynamic else 'static'}; "
        f"backend={'dask' if lazy else 'eager'}; edges={edges}; query={size:,}: "
        f"direct={direct:.6f}s; tal={tal:.6f}s; ratio={tal / direct:.2f}x; "
        f"tasks={direct_tasks}/{tal_tasks}"
    )


def main() -> int:
    args = _parse_args()
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")
    print(
        f"numpy: {np.__version__}; pandas: {pd.__version__}; "
        f"xarray: {xr.__version__}; dask: {dask.__version__}; tal: {version('tal')}"
    )
    print(
        f"warmups: {args.warmups}; repeats: {args.repeats}; "
        "timings exclude fixtures and computation"
    )
    dynamic_modes = tuple(mode == "dynamic" for mode in args.modes)
    lazy_backends = tuple(backend == "dask" for backend in args.backends)
    cases = product(dynamic_modes, lazy_backends, args.edges, args.sizes)
    for dynamic, lazy, edges, size in cases:
        _run_case(
            dynamic=dynamic,
            lazy=lazy,
            edges=edges,
            size=size,
            warmups=args.warmups,
            repeats=args.repeats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
