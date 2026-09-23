from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from benchmarks.bench_batched_fused_path_reference import prepare_reference_fixture
from benchmarks.bench_capstone_workflow import CapstoneConfig
from tal.core import AnalysisObject
from tal.core.param_ops.types import ParamEvalOptions
from tal.frames import FrameGraph, find_path
from tal.spatial import Pose, Position, solve_pose_path_transform
from tal.spatial.ops.batched_path_plan import BatchedPathReason
from tal.spatial.ops.path_execution import prepare_pose_path_execution
from tal.spatial.ops.path_query_plan import prepare_path_query
from tal.spatial.temporal.options import PoseTemporalOptions, RotationTemporalOptions
from tests.core.test_spatial_path_execution import _registered_path
from tests.core.test_spatial_path_solve import _batched_dynamic_pose

_QUERY_UNSET = object()


def _candidate(
    provider: Pose,
    *,
    temporal: PoseTemporalOptions | None = None,
    representation: str | None = None,
    query: object = _QUERY_UNSET,
    caller: object | None = None,
    path=None,
):
    if path is None:
        graph = FrameGraph()
        with graph:
            parent = graph.get_or_create_frame("f0")
            child = graph.get_or_create_frame("f1", parent=parent)
        path = find_path(child, parent)
    if query is _QUERY_UNSET and caller is None:
        query = xr.DataArray([[0.25, 0.75]], dims=("trial", "when"))
    elif query is _QUERY_UNSET:
        query = None
    prepared = prepare_path_query(
        (provider,),
        query=query,
        caller=caller,
        temporal=temporal or PoseTemporalOptions(),
        owner="spatial.path_solve.pose",
        source_representations=(representation,),
    )
    result = prepare_pose_path_execution(path, prepared)
    assert result.batched is not None
    return result.batched


def _provider_metadata_case(provider: Pose, case: str) -> Pose:
    ds = provider.as_dataset(copy="deep")
    if case == "dataset-attr":
        ds.attrs["source"] = "fixture"
    elif case == "dataset-encoding":
        ds.encoding["source"] = "fixture"
    elif case == "variable-attr":
        ds["position"].attrs["units"] = "m"
    elif case == "variable-encoding":
        ds["position"].encoding["dtype"] = "float64"
    elif case == "coordinate-attr":
        ds.coords["time"].attrs["units"] = "s"
    elif case == "coordinate-encoding":
        ds.coords["time"].encoding["dtype"] = "float64"
    elif case == "aux-variable":
        ds["quality"] = ("sample", np.ones(ds.sizes["sample"]))
    elif case == "aux-coordinate":
        ds = ds.assign_coords(phase=("sample", np.arange(ds.sizes["sample"])))
    elif case == "extension":
        schema = deepcopy(ds.attrs["tal"])
        schema["ext"]["calibration"] = {"version": 1}
        ds.attrs["tal"] = schema
    else:
        schema = deepcopy(ds.attrs["tal"])
        schema["ext"]["spatial"]["roles"] = {"future_role": "opaque"}
        ds.attrs["tal"] = schema
    return Pose(ds)


def _caller_metadata_case(caller: Position, case: str) -> Position:
    ds = caller.as_dataset(copy="deep")
    if case == "dataset-attr":
        ds.attrs["source"] = "fixture"
    elif case == "dataset-encoding":
        ds.encoding["source"] = "fixture"
    elif case == "variable-attr":
        ds["position"].attrs["units"] = "m"
    elif case == "coordinate-encoding":
        ds.coords["time"].encoding["dtype"] = "float64"
    elif case == "aux-coordinate":
        ds = ds.assign_coords(quality=("sample", np.arange(ds.sizes["sample"])))
    else:
        schema = deepcopy(ds.attrs["tal"])
        schema["ext"]["calibration"] = {"version": 1}
        ds.attrs["tal"] = schema
    return Position(ds, graph=caller.graph)


def test_spatial_core_batched_path_plan_001_direct_query_is_eligible_but_generic() -> None:
    """ID: SPATIAL_CORE_BATCHED_PATH_PLAN_001_direct_query_candidate."""
    graph, providers = _registered_path(1)
    path = find_path(graph.get_frame("f1"), graph.get_frame("f0"))
    query = xr.DataArray(
        np.asarray([[0.25, 0.75], [0.25, 0.75]]),
        dims=("trial", "when"),
        coords={"trial": ["a", "b"]},
    )
    prepared = prepare_path_query(
        providers,
        query=query,
        caller=None,
        temporal=PoseTemporalOptions(),
        owner="spatial.path_solve.pose",
    )
    with patch("tal.spatial.ops.path_execution._numba_available", side_effect=AssertionError):
        execution = prepare_pose_path_execution(path, prepared)

    assert execution.kind == "generic"
    assert execution.batched is not None
    assert execution.batched.eligible
    assert execution.batched.storage == "eager"
    assert execution.batched.finalization.output == "pose"
    assert execution.batched.finalization.parent_frame == "f0"
    assert execution.batched.finalization.child_frame == "f1"
    assert execution.batched.finalization.expressed_in == "f0"
    assert execution.batched.logical_rows.dims == ("trial", execution.query.topology.query_dim)


def test_spatial_lazy_batched_path_plan_001_capstone_uses_native_chunks_without_work() -> None:
    """ID: SPATIAL_LAZY_BATCHED_PATH_PLAN_001_capstone_partition_is_metadata_only."""
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    config = CapstoneConfig(trials=16, ship_samples=17, drone_samples=33, trial_chunk=4, sample_chunk=8)
    tasks: list[object] = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        patch("tal.core.param_engine.backend_selection._numba_available", side_effect=AssertionError),
        patch("tal.spatial.ops.path_execution._numba_available", side_effect=AssertionError),
    ):
        prepared = prepare_reference_fixture(config, lazy=True)
    candidate = prepared.candidate

    assert tasks == []
    assert candidate.storage == "dask"
    assert candidate.mixed_storage
    assert candidate.payload_storage == ("dask", "eager")
    assert candidate.reason is BatchedPathReason.ELIGIBLE
    assert candidate.finalization.output == "position"
    assert candidate.finalization.parent_frame == "ship"
    assert candidate.finalization.child_frame == "drone"
    assert candidate.finalization.expressed_in == "ship"
    assert candidate.logical_rows.dims == ("trial", "query")
    assert all(partition.row_count <= 65_536 for partition in candidate.physical_rows.partitions)
    assert candidate.query.topology is not None
    assert candidate.query.topology.query.chunks is None
    assert candidate.query.topology.caller is not None
    assert candidate.query.topology.caller.ds["position"].chunks[1] == (8, 8, 8, 8, 1)
    assert candidate.physical_rows.execution.chunks[-1] == (
        slice(0, 8),
        slice(8, 16),
        slice(16, 24),
        slice(24, 32),
        slice(32, 33),
    )


def test_spatial_hard_batched_path_eligibility_001_static_and_datetime_fallbacks() -> None:
    dynamic = _registered_path(1)[1][0]
    static = dynamic.isel(sample=0)
    static_candidate = _candidate(static)
    assert static_candidate.reason is BatchedPathReason.NON_DYNAMIC

    ds = dynamic.as_dataset(copy="deep")
    ds = ds.assign_coords(
        time=(
            "sample",
            np.asarray("2026-01-01", dtype="datetime64[D]")
            + np.arange(ds.sizes["sample"]),
        )
    )
    datetime_provider = Pose(ds)
    datetime_query = xr.DataArray(
        [[np.datetime64("2026-01-01"), np.datetime64("2026-01-02")]],
        dims=("trial", "when"),
    )
    datetime_candidate = _candidate(datetime_provider, query=datetime_query)
    assert datetime_candidate.reason is BatchedPathReason.PARAMETER_KIND


@pytest.mark.parametrize("lazy", (False, True))
def test_spatial_hard_batched_path_eligibility_001_public_route_remains_generic(lazy: bool) -> None:
    """ID: SPATIAL_HARD_BATCHED_PATH_ELIGIBILITY_001_no_public_route_change."""
    graph, providers = _registered_path(1, lazy=lazy)
    before = providers[0].as_dataset(copy="deep")
    query = xr.DataArray([[0.25, 0.75]], dims=("trial", "when"), coords={"trial": [0]})
    result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    expected = solve_pose_path_transform("f1", "f0", graph=graph, query=query.values[0])
    actual_ds = result.as_dataset(copy="none")
    expected_ds = expected.as_dataset(copy="none")

    np.testing.assert_allclose(actual_ds["position"].isel(trial=0), expected_ds["position"])
    assert actual_ds.sizes["trial"] == 1
    assert result.graph is graph
    xr.testing.assert_identical(providers[0].as_dataset(copy="none"), before)


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("matrix", BatchedPathReason.REPRESENTATION),
        ("metadata", BatchedPathReason.PROVIDER_METADATA),
        ("extension", BatchedPathReason.PROVIDER_METADATA),
        ("dtype", BatchedPathReason.PAYLOAD_LAYOUT),
        ("policy", BatchedPathReason.TEMPORAL_POLICY),
    ),
)
def test_spatial_hard_batched_path_eligibility_001_exclusions_are_structured(
    case: str,
    reason: BatchedPathReason,
) -> None:
    provider = _registered_path(1)[1][0]
    temporal = PoseTemporalOptions()
    representation = None
    if case == "matrix":
        representation = "matrix"
    elif case == "metadata":
        ds = provider.as_dataset(copy="deep")
        ds["position"].attrs["units"] = "m"
        provider = Pose(ds)
    elif case == "extension":
        ds = provider.as_dataset(copy="deep")
        schema = deepcopy(ds.attrs["tal"])
        schema["ext"]["calibration"] = {"version": 1}
        ds.attrs["tal"] = schema
        provider = Pose(ds)
    elif case == "dtype":
        ds = provider.as_dataset(copy="deep")
        ds["position"] = ds["position"].astype(np.float16)
        ds["rotation"] = ds["rotation"].astype(np.float16)
        provider = Pose(ds)
    else:
        temporal = PoseTemporalOptions(
            position_opts=ParamEvalOptions(method="nearest"),
            rotation_opts=RotationTemporalOptions(method="nearest"),
        )
    candidate = _candidate(
        provider,
        temporal=temporal,
        representation=representation,
    )
    assert not candidate.eligible
    assert candidate.reason is reason


def test_spatial_hard_batched_path_eligibility_001_mixed_payloads_select_dask() -> None:
    provider = _registered_path(1, lazy=True)[1][0]
    ds = provider.as_dataset(copy="deep")
    ds["rotation"] = ds["rotation"].compute()
    provider = Pose(ds)
    candidate = _candidate(provider)

    assert candidate.eligible
    assert candidate.storage == "dask"
    assert candidate.mixed_storage
    assert candidate.payload_storage == ("dask", "eager")


@pytest.mark.parametrize(
    "case",
    (
        "dataset-attr",
        "dataset-encoding",
        "variable-attr",
        "variable-encoding",
        "coordinate-attr",
        "coordinate-encoding",
        "aux-variable",
        "aux-coordinate",
        "extension",
        "spatial-role",
    ),
)
def test_spatial_hard_batched_path_eligibility_001_provider_metadata_is_generic(
    case: str,
) -> None:
    provider = _provider_metadata_case(_registered_path(1)[1][0], case)
    candidate = _candidate(provider)

    assert not candidate.eligible
    assert candidate.reason is BatchedPathReason.PROVIDER_METADATA


@pytest.mark.parametrize(
    "case",
    (
        "dataset-attr",
        "dataset-encoding",
        "variable-attr",
        "coordinate-encoding",
        "aux-coordinate",
        "extension",
    ),
)
def test_spatial_hard_batched_path_eligibility_001_caller_metadata_remains_eligible(
    case: str,
) -> None:
    prepared = prepare_reference_fixture(
        CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
    )
    caller = _caller_metadata_case(prepared.caller, case)
    candidate = _candidate(
        prepared.provider,
        caller=caller,
        path=prepared.path,
    )

    assert candidate.eligible
    assert candidate.reason is BatchedPathReason.ELIGIBLE
    assert candidate.finalization.output == "position"


@pytest.mark.parametrize("case", ("attrs", "encoding", "aux-coordinate"))
def test_spatial_hard_batched_path_eligibility_001_query_metadata_remains_eligible(
    case: str,
) -> None:
    provider = _registered_path(1)[1][0]
    query = xr.DataArray(
        [[0.25, 0.75], [0.25, 0.75]],
        dims=("trial", "when"),
        coords={"trial": ["a", "b"]},
    )
    if case == "attrs":
        query.attrs["source"] = "fixture"
    elif case == "encoding":
        query.encoding["dtype"] = "float64"
    else:
        query = query.assign_coords(label=("trial", ["left", "right"]))

    candidate = _candidate(provider, query=query)

    assert candidate.eligible
    assert candidate.reason is BatchedPathReason.ELIGIBLE


def test_spatial_core_batched_path_plan_001_native_index_and_batch_projection() -> None:
    trial = xr.Coordinates.from_xindex(
        xr.indexes.RangeIndex.arange(2, coord_name="trial", dim="trial")
    )
    provider_ds = _batched_dynamic_pose().as_dataset(copy="none")
    provider_ds = provider_ds.drop_indexes("trial").drop_vars("trial").assign_coords(trial)
    provider = Pose(provider_ds)
    query = xr.DataArray(
        np.full((3, 2, 2), 0.5),
        dims=("run", "trial", "when"),
        coords=trial.assign(run=("run", [0, 1, 2])),
    )
    candidate = _candidate(provider, query=query)

    assert candidate.eligible
    assert candidate.logical_rows.dims == ("run", "trial", candidate.finalization.query_dim)
    assert isinstance(query.xindexes["trial"], xr.indexes.RangeIndex)

    singleton = Pose(provider.as_dataset(copy="none").isel(trial=slice(0, 1)))
    broadcast_query = xr.DataArray(
        np.full((3, 2), 0.5),
        dims=("run", "when"),
    )
    projected = _candidate(singleton, query=broadcast_query)
    assert projected.eligible
    assert "trial" not in projected.logical_rows.dims


def test_spatial_hard_batched_path_eligibility_001_preflight_and_output_boundary() -> None:
    provider = _batched_dynamic_pose()
    mismatch = xr.DataArray(
        [[0.25], [0.75]],
        dims=("trial", "when"),
        coords={"trial": ["b", "a"]},
    )
    with pytest.raises(ValueError, match="provider batch index must exactly match"):
        _candidate(provider, query=mismatch)

    with pytest.raises(ValueError, match="provider-only batch dims.*trial"):
        _candidate(provider, query=xr.DataArray([[0.5]], dims=("run", "when")))

    prepared = prepare_reference_fixture(
        CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
    )
    caller = AnalysisObject(prepared.caller.as_dataset(copy="deep"))
    unsupported = _candidate(
        prepared.provider,
        caller=caller,
        path=prepared.path,
    )
    assert unsupported.reason is BatchedPathReason.UNSUPPORTED_OUTPUT


def test_spatial_core_batched_path_plan_001_provider_callback_runs_once() -> None:
    graph, providers = _registered_path(1)
    calls: list[tuple[str, str]] = []

    def resolve(parent, child):
        calls.append((parent.id, child.id))
        return providers[0]

    query = xr.DataArray([[0.25, 0.75]], dims=("trial", "when"))
    result = solve_pose_path_transform(
        "f1",
        "f0",
        graph=graph,
        query=query,
        edge_pose_fn=resolve,
    )

    assert calls == [("f1", "f0")]
    assert result.as_dataset(copy="none").sizes["trial"] == 1


def test_spatial_core_batched_path_plan_001_providers_keep_distinct_domains() -> None:
    graph, providers = _registered_path(2)
    shorter_ds = providers[1].as_dataset(copy="none").isel(sample=[0, 2, 4])
    shorter = Pose(shorter_ds, graph=graph)
    path = find_path(graph.get_frame("f2"), graph.get_frame("f0"))
    query = xr.DataArray(
        [[0.25, 0.75], [0.25, 0.75]],
        dims=("trial", "when"),
    )
    prepared = prepare_path_query(
        (providers[0], shorter),
        query=query,
        caller=None,
        temporal=PoseTemporalOptions(),
        owner="spatial.path_solve.pose",
    )
    execution = prepare_pose_path_execution(path, prepared)

    assert execution.batched is not None and execution.batched.eligible
    sample_sizes = tuple(item.context.ds.sizes["sample"] for item in prepared.items)
    assert sample_sizes == (5, 3)
    assert prepared.items[0].evaluations[0] is not prepared.items[1].evaluations[0]
