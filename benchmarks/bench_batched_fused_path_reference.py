"""Independent schema-free reference for batched spatial path planning."""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import xarray as xr

from benchmarks._batched_path_dask_measure import dask_134c_report
from benchmarks._batched_path_direct_executor import execute_direct_packed
from benchmarks._batched_path_execution_routes import (
    ProductionBatchedRoute,
    prepare_production_route,
    production_dispatch,
    production_execute,
    production_finalize,
    production_pack,
)
from benchmarks._batched_path_process_protocol import (
    benchmark_environment,
    dataset_chunks,
)
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.param_engine.blocking import select_logical_block
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.core.schema_validate import validate_schema
from tal.frames import FrameGraph, FramePath, find_path
from tal.spatial import Pose, Position
from tal.spatial.association import SpatialAssociationPlan
from tal.spatial.kernels.fixed_size_backends import (
    SPATIAL_FIXED_BACKEND_SCIPY,
    pose_compose_translation_block_backend,
    pose_inverse_translation_block_backend,
    quat_compose_block_backend,
    quat_inverse_block_backend,
)
from tal.spatial.kernels.pose_apply_kernels import pose_apply_position_kernel
from tal.spatial.kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_SCIPY,
    slerp_quat_backend,
)
from tal.spatial.ops.batched_path_plan import PreparedBatchedPathExecution
from tal.spatial.ops.path_execution import prepare_pose_path_execution
from tal.spatial.ops.path_query_plan import prepare_path_query
from tal.spatial.ops.pose_component_ops import resolve_pose_component_specs
from tal.spatial.temporal.options import PoseTemporalOptions

from ._composite_measurement import _assert_result_parity
from .bench_capstone_workflow import (
    DEFAULT_CONFIG,
    MAIN_LAYOUT,
    PROVIDER_LAYOUT,
    CapstoneConfig,
    CapstoneFixture,
    CapstoneOutputs,
    _interpolate_ship,
    capstone_fixture,
    direct_xarray_route,
    identity_rotation,
    public_tal_route,
    task_count,
    validate_outputs,
)


@dataclass(frozen=True)
class BatchedReferenceResult:
    """Schema-free arrays produced by the independent reference."""

    translation: np.ndarray
    quaternion: np.ndarray
    position: np.ndarray | None


@dataclass(frozen=True)
class BatchedReferenceFixture:
    """One frozen capstone request and its metadata-only execution plan."""

    source: CapstoneFixture
    path: FramePath
    provider: Pose
    candidate: PreparedBatchedPathExecution
    caller: Position
    expected_position: np.ndarray | None
    expected_result: xr.Dataset | None
    expected_outputs: CapstoneOutputs | None


@dataclass(frozen=True)
class _MappedBlock:
    i0: np.ndarray
    i1: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


def _reference_provider(source: xr.Dataset, graph: FrameGraph) -> Pose:
    rotation = identity_rotation(graph)
    position = Position(
        PROVIDER_LAYOUT.wrap(source, data_vars="position"),
        parent="world",
        child="ship",
        graph=graph,
    )
    provider = Pose.from_components(
        rotation,
        position,
        parent="world",
        child="ship",
        graph=graph,
    )
    provider.register()
    return provider


def _expected_position_result(
    fixture: CapstoneFixture,
    graph: FrameGraph,
    values: np.ndarray,
) -> xr.Dataset:
    source = fixture.drone[["position"]].copy(deep=True)
    configured = Position(
        MAIN_LAYOUT.wrap(source, data_vars="position"),
        parent="ship",
        child="drone",
        graph=graph,
    ).as_dataset(copy="deep")
    result = xr.Dataset(
        coords={
            name: fixture.drone.coords[name].variable.copy(deep=True)
            for name in ("time", "group_size", "outcome", "approach_time")
        },
        attrs=configured.attrs,
    )
    result = result.assign_coords(
        {
            name: fixture.drone.coords[name].variable.copy(deep=True)
            for name in ("trial", "sample", "axis")
        }
    )
    result.encoding = configured.encoding
    result["position"] = configured["position"].copy(data=values)
    return result


def prepare_reference_fixture(
    config: CapstoneConfig = DEFAULT_CONFIG,
    *,
    lazy: bool = False,
) -> BatchedReferenceFixture:
    """Prepare the frozen eager capstone request without executing path payloads."""
    fixture = capstone_fixture(config, lazy=lazy)
    graph = FrameGraph()
    provider = _reference_provider(fixture.ship_provider, graph)
    caller = Position(
        MAIN_LAYOUT.wrap(fixture.drone, data_vars="position"),
        parent="world",
        child="drone",
        graph=graph,
    )
    path = find_path(graph.get_or_create_frame("world"), graph.get_frame("ship"))
    candidate = _prepare_candidate(path, provider, caller)
    expected_position = None
    expected_result = None
    expected_outputs = None
    if not lazy:
        expected_position = np.asarray(fixture.drone["position"] - _interpolate_ship(fixture))
        expected_result = _expected_position_result(fixture, graph, expected_position)
    if not lazy and config.trials:
        expected_outputs = direct_xarray_route(fixture)
    return BatchedReferenceFixture(
        fixture,
        path,
        provider,
        candidate,
        caller,
        expected_position,
        expected_result,
        expected_outputs,
    )


def _prepare_candidate(
    path: FramePath,
    provider: Pose,
    caller: Position,
) -> PreparedBatchedPathExecution:
    query = prepare_path_query(
        (provider,),
        query=None,
        caller=caller,
        temporal=PoseTemporalOptions(),
        owner="benchmarks.batched_path_reference",
        result_context=SpatialAssociationPlan(caller.graph),
        result_prototype=caller,
        result_validate=True,
    )
    execution = prepare_pose_path_execution(path, query)
    if execution.batched is None or not execution.batched.eligible:
        raise RuntimeError("batched reference fixture did not produce an eligible candidate.")
    return execution.batched


def _block_template(plan: PreparedBatchedPathExecution, index: int) -> xr.DataArray:
    partition = plan.physical_rows.partitions[index]
    return xr.DataArray(
        np.empty(partition.shape, dtype=np.int8),
        dims=plan.logical_rows.dims,
    )


def _mapped_column(
    value: xr.DataArray,
    plan: PreparedBatchedPathExecution,
    index: int,
) -> np.ndarray:
    partition = plan.physical_rows.partitions[index]
    selected = select_logical_block(value, partition.block)
    template = _block_template(plan, index)
    return np.asarray(xr.broadcast(selected, template)[0].transpose(*plan.logical_rows.dims).data)


def _provider_components(item, batch_dims: tuple[str, ...]) -> tuple[xr.DataArray, xr.DataArray, str]:
    ds = analysis_object_dataset(item.projection.value)
    _, sequence_dim, _, core_dims = read_roles(ds)
    specs = resolve_pose_component_specs(
        ds,
        owner="benchmarks.batched_path_reference",
        core_dims=core_dims,
    )
    (_, position_var), (_, rotation_var) = specs
    return ds[position_var], ds[rotation_var], sequence_dim


def _provider_block(
    value: xr.DataArray,
    *,
    sequence_dim: str,
    core_dim: str,
    plan: PreparedBatchedPathExecution,
    index: int,
) -> np.ndarray:
    partition = plan.physical_rows.partitions[index]
    selected = select_logical_block(value, partition.block)
    batch_dims = plan.finalization.batch_dims
    batch_shape = tuple(partition.shape[plan.logical_rows.dims.index(dim)] for dim in batch_dims)
    template = xr.DataArray(np.empty(batch_shape, dtype=np.int8), dims=batch_dims)
    broadcast = xr.broadcast(selected, template)[0]
    return np.asarray(broadcast.transpose(*batch_dims, sequence_dim, core_dim).data)


def _mapped_block(evaluation, plan: PreparedBatchedPathExecution, index: int) -> _MappedBlock:
    mapping = evaluation.param_map
    return _MappedBlock(
        _mapped_column(mapping.i0, plan, index).astype(np.int64, copy=False),
        _mapped_column(mapping.i1, plan, index).astype(np.int64, copy=False),
        _mapped_column(mapping.alpha, plan, index).astype(np.float64, copy=False),
        _mapped_column(mapping.valid, plan, index).astype(bool, copy=False),
    )


def _gather_component(values: np.ndarray, mapping: _MappedBlock) -> tuple[np.ndarray, np.ndarray]:
    outer = int(np.prod(values.shape[:-2], dtype=np.int64))
    query = int(mapping.alpha.shape[-1])
    width = int(values.shape[-1])
    rows = values.reshape(outer, values.shape[-2], width)
    left_index = np.broadcast_to(mapping.i0.reshape(outer, query, 1), (outer, query, width))
    right_index = np.broadcast_to(mapping.i1.reshape(outer, query, 1), (outer, query, width))
    return (
        np.take_along_axis(rows, left_index, axis=1),
        np.take_along_axis(rows, right_index, axis=1),
    )


def _interpolate_edge(item, plan: PreparedBatchedPathExecution, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position, rotation, sequence_dim = _provider_components(item, plan.finalization.batch_dims)
    position_dim, rotation_dim = position.dims[-1], rotation.dims[-1]
    position_values = _provider_block(
        position, sequence_dim=sequence_dim, core_dim=position_dim, plan=plan, index=index,
    )
    rotation_values = _provider_block(
        rotation, sequence_dim=sequence_dim, core_dim=rotation_dim, plan=plan, index=index,
    )
    position_map = _mapped_block(item.evaluations[0], plan, index)
    rotation_map = _mapped_block(item.evaluations[-1], plan, index)
    left_t, right_t = _gather_component(position_values, position_map)
    left_q, right_q = _gather_component(rotation_values, rotation_map)
    outer, query = left_t.shape[:2]
    position_alpha = position_map.alpha.reshape(outer, query)
    rotation_alpha = rotation_map.alpha.reshape(outer, query)
    rotation_valid = rotation_map.valid.reshape(outer, query)
    edge_t = (1.0 - position_alpha[..., None]) * left_t + position_alpha[..., None] * right_t
    edge_q = slerp_quat_backend(
        left_q,
        right_q,
        rotation_alpha,
        rotation_valid,
        backend=ROTATION_INTERP_BACKEND_SCIPY,
    )
    valid = position_map.valid & rotation_map.valid
    shape = (*position_map.alpha.shape, 3)
    edge_t = np.where(valid[..., None], edge_t.reshape(shape), 0.0)
    edge_q = edge_q.reshape((*rotation_map.alpha.shape, 4))
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))
    edge_q = np.where(valid[..., None], edge_q, identity)
    return edge_t, edge_q, valid


def _orient_edge(
    translation: np.ndarray,
    quaternion: np.ndarray,
    *,
    invert: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not invert:
        return translation, quaternion
    inverse_t = pose_inverse_translation_block_backend(
        translation,
        quaternion,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    inverse_q = quat_inverse_block_backend(
        quaternion,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    return inverse_t, inverse_q

def _compose(
    left_t: np.ndarray,
    left_q: np.ndarray,
    right_t: np.ndarray,
    right_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    translation = pose_compose_translation_block_backend(
        left_t,
        right_t,
        right_q,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    quaternion = quat_compose_block_backend(
        left_q,
        right_q,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    return translation, quaternion


def _target_slices(plan: PreparedBatchedPathExecution, index: int) -> tuple[object, ...]:
    block = plan.physical_rows.partitions[index].block
    selected = dict(block.selections)
    return tuple(selected[dim] for dim in plan.logical_rows.dims)


def _caller_block(plan: PreparedBatchedPathExecution, caller: Position, index: int) -> tuple[np.ndarray, np.ndarray]:
    ds = analysis_object_dataset(caller)
    name = next(iter(ds.data_vars))
    _, sequence_dim, batch_dims, core_dims = read_roles(ds)
    partition = plan.physical_rows.partitions[index]
    selected = select_logical_block(ds[name], partition.block)
    query_slice = partition.block.selection_for_dim(plan.finalization.query_dim)
    selected = selected.isel({sequence_dim: query_slice})
    values = np.asarray(selected.transpose(*batch_dims, sequence_dim, core_dims[0]).data)
    valid = plan.query.topology.caller.valid_mask
    valid = select_logical_block(valid, partition.block).isel({sequence_dim: query_slice})
    return values, np.asarray(valid.transpose(*batch_dims, sequence_dim).data, dtype=bool)


def _fold_edge_partition(
    plan: PreparedBatchedPathExecution,
    edge: int,
    item: object,
    index: int,
    out_t: np.ndarray,
    out_q: np.ndarray,
) -> None:
    edge_t, edge_q, valid = _interpolate_edge(item, plan, index)
    edge_t, edge_q = _orient_edge(edge_t, edge_q, invert=plan.path.steps[edge].invert)
    target = _target_slices(plan, index)
    active = valid
    if edge:
        active = active & np.isfinite(out_q[target]).all(axis=-1)
        identity = np.asarray((0.0, 0.0, 0.0, 1.0))
        left_t = np.where(active[..., None], out_t[target], 0.0)
        left_q = np.where(active[..., None], out_q[target], identity)
        edge_t, edge_q = _compose(left_t, left_q, edge_t, edge_q)
    out_t[target] = np.where(active[..., None], edge_t, np.nan)
    out_q[target] = np.where(active[..., None], edge_q, np.nan)


def execute_batched_reference(
    plan: PreparedBatchedPathExecution,
    *,
    caller: Position | None = None,
) -> BatchedReferenceResult:
    """Evaluate one eager candidate without production fused orchestration."""
    if plan.storage != "eager" or not plan.eligible:
        raise ValueError("batched reference requires an eligible eager plan.")
    shape = plan.logical_rows.sizes
    out_t = np.full((*shape, 3), np.nan, dtype=np.float64)
    out_q = np.full((*shape, 4), np.nan, dtype=np.float64)
    for edge, item in enumerate(plan.query.items):
        for index in range(len(plan.physical_rows.partitions)):
            _fold_edge_partition(plan, edge, item, index, out_t, out_q)
    position = None
    if caller is not None:
        position = np.full((*shape, 3), np.nan, dtype=np.float64)
        for index in range(len(plan.physical_rows.partitions)):
            values, valid = _caller_block(plan, caller, index)
            target = _target_slices(plan, index)
            active = valid & np.isfinite(out_q[target]).all(axis=-1)
            translation = np.where(active[..., None], out_t[target], 0.0)
            identity = np.asarray((0.0, 0.0, 0.0, 1.0))
            quaternion = np.where(active[..., None], out_q[target], identity)
            applied = pose_apply_position_kernel(values, translation, quaternion)
            position[target] = np.where(active[..., None], applied, np.nan)
    return BatchedReferenceResult(out_t, out_q, position)

def _public_position(prepared: BatchedReferenceFixture) -> Position:
    return prepared.caller.to_frame("ship", graph=prepared.caller.graph)


def _direct_position(prepared: BatchedReferenceFixture) -> np.ndarray:
    return np.asarray(prepared.source.drone["position"] - _interpolate_ship(prepared.source))


def _validate_position_topology(value: Position, prepared: BatchedReferenceFixture) -> None:
    if prepared.expected_result is None:
        raise RuntimeError("eager benchmark fixture is missing its typed expectation.")
    ds = validate_schema(analysis_object_dataset(value))
    expected = validate_schema(prepared.expected_result)
    _, sequence_dim, batch_dims, core_dims = read_roles(ds)
    assert (sequence_dim, batch_dims, core_dims) == ("sample", ("trial",), ("axis",))
    assert read_param_coord_name(ds) == "time"
    assert read_sequence_size_coord_name(ds) == "group_size"
    _assert_result_parity(ds, expected)
    assert value.graph is prepared.caller.graph


def _validate_reference(result: BatchedReferenceResult, prepared: BatchedReferenceFixture) -> None:
    if prepared.expected_position is None:
        raise RuntimeError("eager benchmark fixture is missing its numerical expectation.")
    expected_translation = prepared.expected_position - np.asarray(
        prepared.source.drone["position"]
    )
    expected_quaternion = np.zeros((*expected_translation.shape[:-1], 4), dtype=np.float64)
    expected_quaternion[..., 3] = 1.0
    np.testing.assert_allclose(result.translation, expected_translation, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.quaternion, expected_quaternion, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.position, prepared.expected_position, rtol=1e-12, atol=1e-12)


def _validate_route(name: str, result: object, prepared: BatchedReferenceFixture) -> None:
    if name == "planning":
        assert isinstance(result, PreparedBatchedPathExecution) and result.eligible
        return
    if name == "production_packing":
        assert result.providers and result.maps
        return
    if name in {"production_executor", "direct_packed_executor"}:
        first, second = result
        np.testing.assert_allclose(first, prepared.expected_position, rtol=1e-12, atol=1e-12)
        assert second is None
        return
    if name == "independent_scipy_reference":
        assert isinstance(result, BatchedReferenceResult)
        _validate_reference(result, prepared)
        return
    if name in {"production_finalization", "production_dispatch", "public_transform"}:
        assert isinstance(result, Position)
        _validate_position_topology(result, prepared)
        return
    if name in {"public_eager", "direct_xarray"}:
        if prepared.expected_outputs is None:
            raise RuntimeError("eager benchmark fixture is missing its capstone expectation.")
        assert isinstance(result, CapstoneOutputs)
        validate_outputs(result, prepared.expected_outputs)
        return
    np.testing.assert_allclose(result, prepared.expected_position, rtol=1e-12, atol=1e-12)


def _measure(operation, validator) -> tuple[float, int]:
    gc.collect()
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    validator(result)
    del result
    gc.collect()
    tracemalloc.start()
    result = operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    validator(result)
    del result
    return elapsed, peak


def _sample_routes(
    routes: dict[str, Callable[[], object]],
    prepared: BatchedReferenceFixture,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, dict[str, object]]:
    for _ in range(warmups):
        for name, operation in routes.items():
            _validate_route(name, operation(), prepared)
    samples: dict[str, list[tuple[float, int]]] = {name: [] for name in routes}
    for repetition in range(repeats):
        names = tuple(routes) if repetition % 2 == 0 else tuple(reversed(routes))
        for name in names:
            validator = lambda result, name=name: _validate_route(name, result, prepared)
            samples[name].append(_measure(routes[name], validator))
    return {
        name: {
            "seconds": [sample[0] for sample in rows],
            "peak_bytes": [sample[1] for sample in rows],
            "median_seconds": statistics.median(sample[0] for sample in rows),
            "median_peak_bytes": statistics.median(sample[1] for sample in rows),
        }
        for name, rows in samples.items()
    }


def _benchmark_routes(
    prepared: BatchedReferenceFixture,
    production: ProductionBatchedRoute,
) -> dict[str, Callable[[], object]]:
    return {
        "planning": lambda: _prepare_candidate(prepared.path, prepared.provider, prepared.caller),
        "production_packing": lambda: production_pack(prepared.candidate),
        "production_executor": lambda: production_execute(production),
        "production_finalization": lambda: production_finalize(production),
        "production_dispatch": lambda: production_dispatch(production),
        "direct_packed_executor": lambda: execute_direct_packed(
            production.plan, production.packed, backend=production.backend,
        ),
        "independent_scipy_reference": lambda: execute_batched_reference(
            prepared.candidate, caller=prepared.caller,
        ),
        "public_transform": lambda: _public_position(prepared),
        "direct_transform": lambda: _direct_position(prepared),
        "public_eager": lambda: public_tal_route(prepared.source),
        "direct_xarray": lambda: direct_xarray_route(prepared.source),
    }


def benchmark_report(
    config: CapstoneConfig = DEFAULT_CONFIG,
    *,
    warmups: int = 2,
    repeats: int = 7,
) -> dict[str, object]:
    """Profile planning and the independent reference under a fixed protocol."""
    prepared = prepare_reference_fixture(config)
    if prepared.expected_position is None:
        raise RuntimeError("eager benchmark fixture is missing its independent reference.")
    production = prepare_production_route(prepared.candidate)
    samples = _sample_routes(
        _benchmark_routes(prepared, production), prepared, warmups=warmups, repeats=repeats,
    )
    public = public_tal_route(prepared.source)
    direct = direct_xarray_route(prepared.source)
    lazy_source = capstone_fixture(config, lazy=True)
    lazy_public = public_tal_route(lazy_source)
    lazy_direct = direct_xarray_route(lazy_source)
    lazy_plan = prepare_reference_fixture(config, lazy=True).candidate
    return {
        "fixture": config.__dict__,
        "environment": benchmark_environment(backend=production.backend),
        "routes": samples,
        "physical_partitions": len(prepared.candidate.physical_rows.partitions),
        "lazy_physical_partitions": len(lazy_plan.physical_rows.partitions),
        "dask_tasks": {
            "public_dask_batched": task_count(lazy_public),
            "direct_xarray": task_count(lazy_direct),
        },
        "dask_chunks": {
            "public_dask_batched": dataset_chunks(lazy_public.relative_distance),
            "direct_xarray": dataset_chunks(lazy_direct.relative_distance),
        },
        "dask_134c": dask_134c_report(config, warmups=warmups, repeats=repeats),
        "public_result_shape": public.relative_distance[next(iter(public.relative_distance.data_vars))].shape,
        "direct_result_shape": direct.relative_distance[next(iter(direct.relative_distance.data_vars))].shape,
    }


def main() -> None:
    from benchmarks._batched_path_process_protocol import run_benchmark_cli

    run_benchmark_cli(benchmark_report)


if __name__ == "__main__":
    main()
