from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation

from benchmarks._batched_path_direct_executor import execute_direct_packed
from benchmarks._batched_path_process_protocol import (
    measure_batched_rss,
    measure_cold_public,
)
from benchmarks._spatial_path_execution_routes import _public_case
from benchmarks.bench_batched_fused_path_reference import (
    _public_position,
    _validate_position_topology,
    benchmark_report,
    execute_batched_reference,
    prepare_reference_fixture,
)
from benchmarks.bench_capstone_workflow import CapstoneConfig
from benchmarks.bench_spatial_fused_temporal_paths import (
    frozen_fixture,
    scipy_direct_path,
)
from tal.core.param_ops.types import ParamEvalOptions
from tal.core.schema import set_validity
from tal.frames import find_path
from tal.spatial import PathSolveOptions, Pose, solve_pose_path_transform
from tal.spatial.ops.batched_path_inputs import pack_batched_path_inputs
from tal.spatial.ops.path_execution import prepare_pose_path_execution
from tal.spatial.ops.path_query_plan import prepare_path_query
from tal.spatial.ops.pose_provider_ops import resolve_bound_pose
from tal.spatial.temporal.options import PoseTemporalOptions, RotationTemporalOptions


def test_spatial_bench_batched_path_reference_001_matches_capstone_reference() -> None:
    """ID: SPATIAL_BENCH_BATCHED_PATH_REFERENCE_001_independent_numeric_parity."""
    config = CapstoneConfig(trials=3, ship_samples=9, drone_samples=17, trial_chunk=2, sample_chunk=8)
    prepared = prepare_reference_fixture(config)
    result = execute_batched_reference(prepared.candidate, caller=prepared.caller)
    assert prepared.expected_position is not None
    np.testing.assert_allclose(result.position, prepared.expected_position, rtol=1e-12, atol=1e-12)
    assert result.translation.shape == (3, 17, 3)
    assert result.quaternion.shape == (3, 17, 4)


def test_spatial_bench_batched_path_reference_001_reduced_protocol_reports_separate_evidence() -> None:
    config = CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
    report = benchmark_report(config, warmups=0, repeats=1)
    assert set(report["routes"]) == {
        "planning",
        "production_packing",
        "production_executor",
        "production_finalization",
        "production_dispatch",
        "direct_packed_executor",
        "independent_scipy_reference",
        "public_transform",
        "direct_transform",
        "public_eager",
        "direct_xarray",
    }
    reference = report["routes"]["production_executor"]
    assert len(reference["seconds"]) == 1
    assert len(reference["peak_bytes"]) == 1
    assert reference["median_seconds"] > 0.0
    assert reference["median_peak_bytes"] > 0
    assert report["public_result_shape"] == report["direct_result_shape"] == (2, 9)
    environment = report["environment"]
    assert environment["selected_backend"] in {"numba", "scipy"}
    if environment["selected_backend"] == "numba":
        assert int(environment["numba_threads"]) > 0
        assert environment["numba_threading_layer"] != "not-selected"
    else:
        assert environment["numba_threads"] == "not-selected"
        assert environment["numba_threading_layer"] == "not-selected"


def test_spatial_bench_batched_path_execution_001_isolated_rss_protocol() -> None:
    """ID: SPATIAL_BENCH_BATCHED_PATH_EXECUTION_001_isolated_rss_protocol."""
    pytest.importorskip("psutil")
    config = CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
    result = measure_batched_rss(config)
    assert result.config == config
    assert result.backend in {"numba", "scipy"}
    assert result.sample_count >= 20
    assert result.peak_bytes >= result.baseline_bytes


def test_spatial_bench_batched_path_execution_001_cold_protocol_reports_boundaries() -> None:
    config = CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
    result = measure_cold_public(config)

    assert result.config == config
    assert result.backend in {"numba", "scipy"}
    assert result.process_seconds > 0.0
    assert result.import_seconds > 0.0
    assert result.setup_seconds > 0.0
    assert result.first_call_seconds > 0.0
    assert result.validation_seconds > 0.0
    assert result.process_seconds >= (
        result.import_seconds
        + result.setup_seconds
        + result.first_call_seconds
        + result.validation_seconds
    )


def test_spatial_bench_batched_path_reference_001_validator_rejects_metadata_drift() -> None:
    prepared = prepare_reference_fixture(
        CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
    )
    result = _public_position(prepared)
    result.as_dataset(copy="none")["position"].encoding["probe"] = "changed"

    with pytest.raises(AssertionError, match="metadata changed"):
        _validate_position_topology(result, prepared)


def test_spatial_bench_batched_path_reference_001_rejects_lazy_execution() -> None:
    pytest.importorskip("dask.array")
    prepared = prepare_reference_fixture(
        CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4),
        lazy=True,
    )
    with pytest.raises(ValueError, match="eligible eager plan"):
        execute_batched_reference(prepared.candidate, caller=prepared.caller)


def test_spatial_bench_batched_path_reference_001_empty_batch_is_truthful() -> None:
    prepared = prepare_reference_fixture(
        CapstoneConfig(trials=0, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4),
    )
    result = execute_batched_reference(prepared.candidate, caller=prepared.caller)

    assert prepared.candidate.physical_rows.has_no_rows
    assert prepared.candidate.physical_rows.partitions == ()
    assert result.position is not None
    assert result.position.shape == (0, 9, 3)


def test_spatial_bench_batched_path_reference_001_ragged_and_all_missing_lanes() -> None:
    config = CapstoneConfig(trials=3, ship_samples=9, drone_samples=17, trial_chunk=2, sample_chunk=8)
    prepared = prepare_reference_fixture(config)
    ds = prepared.provider.as_dataset(copy="deep")
    ds = ds.assign_coords(group_size=("trial", [0, 7, 9]))
    ds = set_validity(ds, sequence_size_coord="group_size", validate=True)
    provider = Pose(ds)
    query = prepare_path_query(
        (provider,),
        query=None,
        caller=prepared.caller,
        temporal=PoseTemporalOptions(),
        owner="benchmarks.batched_path_reference",
    )
    execution = prepare_pose_path_execution(prepared.path, query)
    assert execution.batched is not None and execution.batched.eligible

    result = execute_batched_reference(execution.batched, caller=prepared.caller)
    assert result.position is not None
    assert np.isnan(result.position[0]).all()
    assert np.isfinite(result.position[2]).all()
    assert np.isfinite(result.position[1, :13]).all()
    assert np.isnan(result.position[1, 13:]).all()


@pytest.mark.parametrize("case", ("h0", "h1"))
@pytest.mark.parametrize("edges", (1, 4, 8))
def test_spatial_bench_batched_path_reference_001_path_directions_and_depth(
    case: str,
    edges: int,
) -> None:
    fixture = frozen_fixture(case, query_size=9, edges=edges)
    graph, providers = _public_case(fixture)
    sources = tuple(provider.as_dataset(copy="deep") for provider in providers)
    path = find_path(graph.get_frame(fixture.source), graph.get_frame(fixture.destination))
    values = tuple(
        resolve_bound_pose(step.child, step.parent, owner="benchmarks.batched_path_reference")
        for step in path.steps
    )
    query = xr.DataArray(np.tile(fixture.query, (2, 1)), dims=("trial", "when"))
    prepared = prepare_path_query(
        values,
        query=query,
        caller=None,
        temporal=PoseTemporalOptions(),
        owner="benchmarks.batched_path_reference",
    )
    execution = prepare_pose_path_execution(path, prepared)
    assert execution.batched is not None
    result = execute_batched_reference(execution.batched)
    direct_t, direct_q = execute_direct_packed(
        execution.batched,
        pack_batched_path_inputs(execution.batched),
        backend="scipy",
    )
    public = solve_pose_path_transform(
        fixture.source,
        fixture.destination,
        graph=graph,
        query=query,
    ).as_dataset(copy="none")
    expected_t, expected_q = scipy_direct_path(fixture)

    np.testing.assert_allclose(result.translation, np.tile(expected_t, (2, 1, 1)), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(public["position"], result.translation, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(direct_t, result.translation, rtol=1e-12, atol=1e-12)
    assert direct_q is not None
    expected_matrix = SciRotation.from_quat(np.tile(expected_q, (2, 1, 1)).reshape(-1, 4)).as_matrix()
    actual_matrix = SciRotation.from_quat(result.quaternion.reshape(-1, 4)).as_matrix()
    public_matrix = SciRotation.from_quat(np.asarray(public["rotation"]).reshape(-1, 4)).as_matrix()
    direct_matrix = SciRotation.from_quat(direct_q.reshape(-1, 4)).as_matrix()
    np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(public_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(direct_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    for provider, source in zip(providers, sources, strict=True):
        xr.testing.assert_identical(provider.as_dataset(copy="none"), source)


@pytest.mark.parametrize("batch_shape", ((2, 3), (2, 2, 3)))
def test_spatial_bench_batched_path_reference_001_multiple_batch_dimensions(
    batch_shape: tuple[int, ...],
) -> None:
    fixture = frozen_fixture("h1", query_size=4, edges=1)
    graph, _ = _public_case(fixture)
    path = find_path(graph.get_frame(fixture.source), graph.get_frame(fixture.destination))
    values = tuple(
        resolve_bound_pose(step.child, step.parent, owner="benchmarks.batched_path_reference")
        for step in path.steps
    )
    batch_dims = tuple(f"batch_{index}" for index in range(len(batch_shape)))
    query = xr.DataArray(
        np.broadcast_to(fixture.query, (*batch_shape, fixture.query.size)),
        dims=(*batch_dims, "when"),
    )
    prepared = prepare_path_query(
        values,
        query=query,
        caller=None,
        temporal=PoseTemporalOptions(),
        owner="benchmarks.batched_path_reference",
    )
    execution = prepare_pose_path_execution(path, prepared)
    assert execution.batched is not None and execution.batched.eligible

    result = execute_batched_reference(execution.batched)
    expected_t, expected_q = scipy_direct_path(fixture)
    np.testing.assert_allclose(
        result.translation,
        np.broadcast_to(expected_t, (*batch_shape, *expected_t.shape)),
        rtol=1e-12,
        atol=1e-12,
    )
    expected_matrix = SciRotation.from_quat(
        np.broadcast_to(expected_q, (*batch_shape, *expected_q.shape)).reshape(-1, 4)
    ).as_matrix()
    actual_matrix = SciRotation.from_quat(result.quaternion.reshape(-1, 4)).as_matrix()
    np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=1e-12, atol=1e-12)


def test_spatial_bench_batched_path_reference_001_uses_component_maps() -> None:
    fixture = frozen_fixture("h0", query_size=1, edges=1)
    parameter = fixture.parameter.copy()
    parameter[1] = parameter[0]
    translation = fixture.translation.copy()
    translation[0, 0, 0] = 2.0
    translation[0, 1, 0] = 7.0
    quaternion = fixture.quaternion.copy()
    quaternion[0, 1] = SciRotation.from_euler("z", 90.0, degrees=True).as_quat()
    fixture = replace(
        fixture,
        parameter=parameter,
        translation=translation,
        quaternion=quaternion,
    )
    graph, _ = _public_case(fixture)
    path = find_path(graph.get_frame(fixture.source), graph.get_frame(fixture.destination))
    values = tuple(
        resolve_bound_pose(step.child, step.parent, owner="benchmarks.batched_path_reference")
        for step in path.steps
    )
    query = xr.DataArray(np.zeros((2, 1)), dims=("trial", "when"))
    temporal = PoseTemporalOptions(
        position_opts=ParamEvalOptions(method="linear", duplicate_policy="left"),
        rotation_opts=RotationTemporalOptions(method="slerp", duplicate_policy="right"),
    )
    prepared = prepare_path_query(
        values,
        query=query,
        caller=None,
        temporal=temporal,
        owner="benchmarks.batched_path_reference",
    )
    execution = prepare_pose_path_execution(path, prepared)
    assert execution.batched is not None and execution.batched.eligible
    assert len(execution.batched.query.items[0].evaluations) == 2

    result = execute_batched_reference(execution.batched)
    public = solve_pose_path_transform(
        fixture.source,
        fixture.destination,
        graph=graph,
        query=query,
        opts=PathSolveOptions(temporal=temporal),
    ).as_dataset(copy="none")
    np.testing.assert_allclose(result.translation[..., 0], 2.0)
    np.testing.assert_allclose(result.translation, public["position"])
    expected = np.broadcast_to(
        SciRotation.from_euler("z", 90.0, degrees=True).as_matrix(),
        (2, 3, 3),
    )
    actual = SciRotation.from_quat(result.quaternion.reshape(-1, 4)).as_matrix()
    public_matrix = SciRotation.from_quat(np.asarray(public["rotation"]).reshape(-1, 4)).as_matrix()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, public_matrix, rtol=1e-12, atol=1e-12)
