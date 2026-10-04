from __future__ import annotations

from dataclasses import replace
from functools import partial
from unittest.mock import patch

import numpy as np
import xarray as xr

from benchmarks._spatial_path_fixtures import (
    NATIVE_SIZE,
    FrozenPathFixture,
    PreparedFixture,
)
from tal.core import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation, solve_pose_path_transform
from tal.spatial.kernels.streaming_pose_path import stream_pose_path_blocks
from tal.spatial.ops.frame_owner_common import clear_framing
from tal.utils.numba_support import _numba_available


def bounded_scipy_stream(prepared: PreparedFixture) -> tuple[np.ndarray, np.ndarray]:
    """Run the maintained bounded eager SciPy interpolation/fold kernel."""
    fixture = prepared.fixture
    return stream_pose_path_blocks(
        tuple(fixture.translation),
        tuple(fixture.quaternion),
        fixture.directions,
        i0=prepared.i0,
        i1=prepared.i1,
        alpha=prepared.alpha,
        valid=prepared.valid,
    )


def _analysis_components(fixture: FrozenPathFixture, edge: int) -> tuple[AnalysisObject, AnalysisObject]:
    coords = {"sample": np.arange(NATIVE_SIZE), "time": ("sample", fixture.parameter)}
    position = xr.DataArray(
        fixture.translation[edge],
        dims=("sample", "axis"),
        coords={**coords, "axis": ["x", "y", "z"]},
        name="position",
    )
    rotation = xr.DataArray(
        fixture.quaternion[edge],
        dims=("sample", "quat"),
        coords={**coords, "quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    kwargs = {"sequence_dim": "sample", "param_coord": "time"}
    return (
        AnalysisObject.from_data(position, core_dims=("axis",), **kwargs),
        AnalysisObject.from_data(rotation, core_dims=("quat",), **kwargs),
    )


def _public_case(fixture: FrozenPathFixture) -> tuple[FrameGraph, tuple[Pose, ...]]:
    graph = FrameGraph()
    providers = []
    for edge, (parent, child) in enumerate(fixture.relations):
        position, rotation = _analysis_components(fixture, edge)
        provider = Pose.from_components(
            Rotation(rotation),
            Position(position),
            parent=parent,
            child=child,
            graph=graph,
        )
        provider.register()
        providers.append(provider)
    return graph, tuple(providers)


def _public_route(case: tuple[FrozenPathFixture, FrameGraph, tuple[Pose, ...]]) -> Pose:
    fixture, graph, _ = case
    return solve_pose_path_transform(
        fixture.source,
        fixture.destination,
        graph=graph,
        query=fixture.query,
    )


def _generic_public_route(case: tuple[FrozenPathFixture, FrameGraph, tuple[Pose, ...]]) -> Pose:
    """Benchmark the accepted generic executor without a production selector."""
    import tal.spatial.ops.path_solve_ops as solver

    prepare = solver.prepare_pose_path_execution

    def select_generic(path, query):
        return replace(prepare(path, query), kind="generic")

    with patch.object(solver, "prepare_pose_path_execution", select_generic):
        return _public_route(case)


def _direct_typed(case: tuple[FrozenPathFixture, FrameGraph, tuple[Pose, ...]]) -> Pose:
    fixture, _, providers = case
    result = None
    for provider, direction in zip(providers, fixture.directions, strict=True):
        evaluated = provider.param.at(fixture.query, validate=False)
        oriented = evaluated if direction > 0 else evaluated.inverse(validate=False)
        value = clear_framing(oriented, owner="benchmarks.spatial_paths.direct_typed")
        result = value if result is None else result.compose(value, validate=False)
    if result is None:
        raise RuntimeError("benchmark path requires at least one edge")
    return result


def typed_route_operation(name: str, fixture: FrozenPathFixture):
    """Prepare one public or direct typed benchmark route."""
    if name not in {"public", "direct-typed", "promoted-generic", "dask-auto-leaf"}:
        raise ValueError(f"unknown benchmark route {name!r}")
    graph, providers = _public_case(fixture)
    if name == "dask-auto-leaf":
        providers = tuple(
            Pose(provider.as_dataset(copy="none").chunk({"sample": NATIVE_SIZE}), graph=graph)
            for provider in providers
        )
        for provider in providers:
            provider.register(on_conflict="replace")
    case = (fixture, graph, providers)
    if name == "direct-typed":
        route = _direct_typed
    elif name == "promoted-generic":
        route = _generic_public_route
    else:
        route = _public_route
    return partial(route, case)


def route_effective_backend(name: str) -> str:
    """Report the backend selected by one benchmark route."""
    if name in {"public", "direct-typed", "promoted-generic", "dask-auto-leaf"}:
        return "numba" if _numba_available() else "scipy"
    if name in {"compiled-leaf", "compiled-fold", "fused-reference"}:
        return "numba"
    if name in {"scipy-direct-path", "scipy-streaming", "scipy-vectorized", "scipy-stacked"}:
        return "scipy"
    return "automatic"


def materialize_route_result(result: object, *, synchronous: bool) -> object:
    """Materialize one complete route result exactly once."""
    if isinstance(result, tuple):
        return tuple(np.asarray(value) for value in result)
    if isinstance(result, Pose):
        dataset = result.as_dataset(copy="none")
    elif isinstance(result, xr.Dataset):
        dataset = result
    else:
        return result
    if not any(value.chunks is not None for value in dataset.data_vars.values()):
        return dataset
    if not synchronous:
        return dataset.compute()
    import dask

    with dask.config.set(scheduler="synchronous", num_workers=1):
        return dataset.compute()


def dask_graph_evidence(fixture: FrozenPathFixture) -> tuple[int, int, str]:
    """Return task shape and the automatic leaf backend without execution."""
    result = typed_route_operation("dask-auto-leaf", fixture)()
    dataset = result.as_dataset(copy="none")
    keys: set[object] = set()
    largest = 0
    for value in dataset.data_vars.values():
        graph = value.data.__dask_graph__()
        keys.update(graph.keys())
        largest = max(largest, max(value.chunks[value.get_axis_num("query")]))
    return len(keys), largest, route_effective_backend("dask-auto-leaf")


__all__ = [
    "bounded_scipy_stream",
    "dask_graph_evidence",
    "materialize_route_result",
    "route_effective_backend",
    "typed_route_operation",
]
