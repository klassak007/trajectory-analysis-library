from __future__ import annotations

import gc
import tracemalloc
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from benchmarks.bench_batched_fused_path_reference import (
    _public_position,
    execute_batched_reference,
    prepare_reference_fixture,
)
from benchmarks.bench_capstone_workflow import CapstoneConfig
from tal.core import AnalysisObject
from tal.core.component_ops import read_components
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.frames import FrameGraph, find_path
from tal.spatial import Pose, Position, bind_pose, solve_pose_path_transform
from tal.spatial.metadata import get_pose_rep
from tal.spatial.ops.batched_path_execution import _caller_block
from tal.spatial.ops.batched_path_inputs import pack_batched_path_inputs
from tal.spatial.ops.path_execution import prepare_pose_path_execution
from tal.spatial.ops.path_query_plan import prepare_path_query
from tal.spatial.ops.pose_provider_ops import resolve_bound_pose
from tal.spatial.temporal.options import PoseTemporalOptions
from tal.utils.frame_schema import get_frames
from tests.core.test_spatial_path_execution import _edge_pose, _registered_path
from tests.core.test_spatial_path_solve import (
    _batched_dynamic_pose,
    _dynamic_pose,
    _pose_from_translation_and_quat,
    _static_pose,
)


class _FailingCopy:
    def __deepcopy__(self, memo: object) -> object:
        _ = memo
        raise UnicodeDecodeError("utf-8", b"x", 0, 1, "copy failed")


def _prepare_batched_between(
    graph: FrameGraph,
    source: str,
    destination: str,
    query: xr.DataArray,
):
    path = find_path(graph.get_frame(source), graph.get_frame(destination))
    providers = tuple(
        resolve_bound_pose(step.child, step.parent, owner="test.batched_path")
        for step in path.steps
    )
    prepared = prepare_path_query(
        providers,
        query=query,
        caller=None,
        temporal=PoseTemporalOptions(),
        owner="test.batched_path",
    )
    execution = prepare_pose_path_execution(path, prepared)
    assert execution.batched is not None and execution.batched.eligible
    return execution


def _batched_candidate(graph, edges: int, query: xr.DataArray):
    return _prepare_batched_between(graph, f"f{edges}", "f0", query).batched


def _directional_path(
    case: str,
    edges: int,
) -> tuple[FrameGraph, tuple[Pose, ...], str, str]:
    if case != "mixed":
        graph, providers = _registered_path(edges)
        endpoints = (f"f{edges}", "f0") if case == "forward" else ("f0", f"f{edges}")
        return graph, providers, *endpoints
    graph = FrameGraph()
    providers: list[Pose] = []
    edge = 0
    for branch in ("a", "b"):
        parent = "root"
        for level in range(1, edges // 2 + 1):
            child = f"{branch}{level}"
            provider = Pose(_edge_pose(edge), parent=parent, child=child, graph=graph)
            provider.register()
            providers.append(provider)
            parent = child
            edge += 1
    return graph, tuple(providers), f"a{edges // 2}", f"b{edges // 2}"


def _empty_position_caller(
    graph: FrameGraph,
    *,
    trials: int,
    samples: int,
) -> Position:
    ds = xr.Dataset(
        {"position": (("trial", "sample", "axis"), np.empty((trials, samples, 3)))},
        coords={
            "trial": np.arange(trials),
            "sample": np.arange(samples),
            "axis": ["x", "y", "z"],
            "time": ("sample", np.arange(samples, dtype=np.float64)),
            "group_size": ("trial", np.full(trials, samples, dtype=np.int64)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="time",
        sequence_size_coord="group_size",
    )
    return Position(ao, parent="f1", child="point", graph=graph)


def _renamed_position_caller(graph: FrameGraph, *, lazy: bool) -> Position:
    values = np.zeros((2, 5, 3), dtype=np.float64)
    values[..., 1] = 1.0
    ds = xr.Dataset(
        {"point": (("trial", "sample", "cart"), values)},
        coords={
            "trial": ["a", "b"],
            "sample": np.arange(5),
            "cart": ["x", "y", "z"],
            "time": ("sample", np.linspace(0.0, 1.0, 5)),
            "group_size": ("trial", [5, 5]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("cart",),
        param_coord="time",
        sequence_size_coord="group_size",
    )
    if lazy:
        lazy_ds = ao.as_dataset(copy="shallow")
        lazy_ds["point"] = lazy_ds["point"].chunk({"trial": 1, "sample": 2})
        ao = AnalysisObject._from_unvalidated(lazy_ds)
    return Position(ao, parent="f1", child="point", graph=graph)


def _large_position_caller(graph: FrameGraph, *, samples: int) -> Position:
    time = np.linspace(0.0, 1.0, samples)
    values = np.column_stack((time, np.zeros((samples, 2))))[None, ...]
    data = xr.Dataset(
        {"position": (("trial", "sample", "axis"), values)},
        coords={
            "trial": ["a"],
            "time": ("sample", time),
            "axis": ["x", "y", "z"],
            "group_size": ("trial", [samples]),
        },
    )
    ao = AnalysisObject.from_data(
        data,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="time",
        sequence_size_coord="group_size",
    )
    return Position(ao, parent="f1", child="point", graph=graph)


def _position_candidate(caller: Position):
    graph = caller.graph
    assert graph is not None
    path = find_path(graph.get_frame("f1"), graph.get_frame("f0"))
    providers = tuple(
        resolve_bound_pose(step.child, step.parent, owner="test.batched_path")
        for step in path.steps
    )
    prepared = prepare_path_query(
        providers,
        query=None,
        caller=caller,
        temporal=PoseTemporalOptions(),
        owner="test.batched_path",
        result_prototype=caller,
    )
    execution = prepare_pose_path_execution(path, prepared)
    assert execution.batched is not None and execution.batched.eligible
    return execution.batched


@pytest.mark.parametrize("compiled", (False, True))
def test_spatial_core_batched_path_execution_001_position_matches_frozen_topology(
    compiled: bool,
) -> None:
    """ID: SPATIAL_CORE_BATCHED_PATH_EXECUTION_001_eager_position_parity."""
    if compiled:
        pytest.importorskip("numba")
    fixture = prepare_reference_fixture(
        CapstoneConfig(trials=3, ship_samples=9, drone_samples=17, trial_chunk=2, sample_chunk=8)
    )
    before = fixture.caller.as_dataset(copy="deep")
    with patch("tal.spatial.ops.path_execution._numba_available", return_value=compiled):
        result = _public_position(fixture)

    assert fixture.expected_result is not None
    xr.testing.assert_identical(result.as_dataset(copy="none"), fixture.expected_result)
    xr.testing.assert_identical(fixture.caller.as_dataset(copy="none"), before)
    assert result.graph is fixture.caller.graph


@pytest.mark.parametrize("compiled", (False, True), ids=("scipy", "numba"))
@pytest.mark.parametrize(
    ("case", "edges", "directions"),
    (
        pytest.param("forward", 1, (False,), id="one-forward"),
        pytest.param("inverse", 4, (True,) * 4, id="four-inverse"),
        pytest.param("mixed", 8, (False,) * 4 + (True,) * 4, id="eight-mixed"),
    ),
)
def test_spatial_core_batched_path_execution_001_direct_pose_matches_reference(
    compiled: bool,
    case: str,
    edges: int,
    directions: tuple[bool, ...],
) -> None:
    if compiled:
        pytest.importorskip("numba")
    graph, providers, source, destination = _directional_path(case, edges)
    before = tuple(provider.as_dataset(copy="deep") for provider in providers)
    query = xr.DataArray(
        [[0.2, 0.8], [0.3, 0.7]],
        dims=("trial", "when"),
        coords={"trial": ["a", "b"], "when": [10, 20]},
    )
    with patch("tal.spatial.ops.path_execution._numba_available", return_value=compiled):
        execution = _prepare_batched_between(graph, source, destination, query)
        result = solve_pose_path_transform(source, destination, graph=graph, query=query)
    assert execution.kind == ("batched-numba" if compiled else "batched-scipy")
    assert tuple(step.invert for step in execution.path.steps) == directions
    candidate = execution.batched
    assert candidate is not None
    expected = execute_batched_reference(candidate)
    dataset = result.as_dataset(copy="none")

    np.testing.assert_allclose(dataset["position"], expected.translation, rtol=1e-12, atol=1e-12)
    actual_matrix = SciRotation.from_quat(np.asarray(dataset["rotation"]).reshape(-1, 4)).as_matrix()
    expected_matrix = SciRotation.from_quat(expected.quaternion.reshape(-1, 4)).as_matrix()
    np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=1e-12, atol=1e-12)
    assert dict(dataset.sizes) == {"trial": 2, "query": 2, "axis": 3, "quat": 4}
    assert set(dataset.coords) == {"trial", "query", "axis", "quat", "query_value"}
    assert read_roles(dataset) == (True, "query", ("trial",), ("axis", "quat"))
    assert read_param_coord_name(dataset) == "query_value"
    assert read_sequence_size_coord_name(dataset) is None
    assert set(read_components(result)) == {"position", "rotation"}
    assert get_pose_rep(dataset, owner="test.batched_path") == "components"
    assert set(dataset.xindexes) == {"trial", "query", "axis", "quat"}
    assert get_frames(dataset) == (destination, source)
    assert result.graph is graph
    for provider, source in zip(providers, before, strict=True):
        xr.testing.assert_identical(provider.as_dataset(copy="none"), source)


def test_spatial_core_batched_path_execution_001_provider_batch_subset_broadcasts() -> None:
    graph = FrameGraph()
    provider = Pose(
        _batched_dynamic_pose().as_dataset(copy="none"),
        parent="f0",
        child="f1",
        graph=graph,
    )
    provider.register()
    query = xr.DataArray(
        np.full((3, 2, 2), 0.5),
        dims=("run", "trial", "when"),
        coords={"run": [0, 1, 2], "trial": ["a", "b"]},
    )
    candidate = _batched_candidate(graph, 1, query)
    expected = execute_batched_reference(candidate)
    result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    dataset = result.as_dataset(copy="none")

    np.testing.assert_allclose(dataset["position"], expected.translation, rtol=1e-12, atol=1e-12)
    assert dataset["position"].dims == ("run", "trial", "query", "axis")
    assert dataset.sizes["run"] == 3
    assert dataset.sizes["trial"] == 2


def test_spatial_core_batched_path_execution_001_renamed_position_generic_fused_parity() -> None:
    graph, _ = _registered_path(1)
    eager = _renamed_position_caller(graph, lazy=False)
    lazy = _renamed_position_caller(graph, lazy=True)
    eager_before = eager.as_dataset(copy="deep")
    lazy_before = lazy.as_dataset(copy="deep")

    fused = eager.to_frame("f0", graph=graph)
    generic = lazy.to_frame("f0", graph=graph)
    actual = generic.as_dataset(copy="none").compute(scheduler="synchronous")
    expected = fused.as_dataset(copy="none")

    xr.testing.assert_identical(actual.drop_vars("point"), expected.drop_vars("point"))
    np.testing.assert_allclose(actual["point"], expected["point"], rtol=1e-12, atol=1e-12)
    assert actual["point"].attrs == expected["point"].attrs
    assert actual["point"].encoding == expected["point"].encoding
    assert set(actual.data_vars) == {"point"}
    assert actual["point"].dims == ("trial", "sample", "cart")
    xr.testing.assert_identical(eager.as_dataset(copy="none"), eager_before)
    xr.testing.assert_identical(lazy.as_dataset(copy="none"), lazy_before)


def test_spatial_core_batched_path_execution_001_position_metadata_matches_generic() -> None:
    graph, _ = _registered_path(1)
    eager = _renamed_position_caller(graph, lazy=False)
    source = eager.as_dataset(copy="shallow")
    source.attrs["ordinary"] = {"nested": [1, 2]}
    source["point"].attrs["units"] = "m"
    source["point"].encoding["dtype"] = "float64"
    source.attrs["tal"].setdefault("ext", {})["opaque_probe"] = {"value": [3]}
    eager = Position(
        AnalysisObject._from_unvalidated(source),
        parent="f1",
        child="point",
        graph=graph,
    )
    before = eager.as_dataset(copy="deep")
    lazy_source = eager.as_dataset(copy="shallow")
    lazy_source["point"] = lazy_source["point"].chunk({"trial": 1, "sample": 2})
    lazy = Position(
        AnalysisObject._from_unvalidated(lazy_source),
        parent="f1",
        child="point",
        graph=graph,
    )

    with patch("tal.spatial.ops.path_execution._numba_available", return_value=False):
        fused = eager.to_frame("f0", graph=graph).as_dataset(copy="none")
    generic = lazy.to_frame("f0", graph=graph).as_dataset(copy="none")
    computed = generic.compute(scheduler="synchronous")

    xr.testing.assert_identical(fused.drop_vars("point"), computed.drop_vars("point"))
    np.testing.assert_allclose(fused["point"], computed["point"], rtol=1e-12, atol=1e-12)
    assert fused["point"].attrs == computed["point"].attrs == {"units": "m"}
    assert fused["point"].encoding == generic["point"].encoding == {"dtype": "float64"}
    xr.testing.assert_identical(eager.as_dataset(copy="none"), before)


@pytest.mark.parametrize("topology", ("static", "exact"))
def test_spatial_core_batched_path_execution_001_preserves_generic_provider_topology(
    topology: str,
) -> None:
    graph = FrameGraph()
    if topology == "static":
        source = _static_pose(2.0)
    else:
        translation = np.tile([2.0, 0.0, 0.0], (5, 1))
        quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))
        source = _pose_from_translation_and_quat(translation, quaternion)
    if topology == "static":
        provider = Pose(source, parent="f0", child="f1", graph=graph)
        provider.register()
    else:
        bind_pose(graph, "f0", "f1", source)
    caller = _renamed_position_caller(graph, lazy=False)
    before = caller.as_dataset(copy="deep")

    result = caller.to_frame("f0", graph=graph)
    dataset = result.as_dataset(copy="none")

    expected = np.asarray(before["point"]).copy()
    expected[..., 0] += 2.0
    np.testing.assert_allclose(dataset["point"], expected, rtol=0.0, atol=0.0)
    assert dataset["point"].dims == ("trial", "sample", "cart")
    assert set(dataset.data_vars) == {"point"}
    xr.testing.assert_identical(caller.as_dataset(copy="none"), before)


def test_spatial_hard_batched_path_execution_001_output_namespace_precedes_backend() -> None:
    graph, _ = _registered_path(1)
    query = xr.DataArray(
        [[0.5]],
        dims=("trial", "when"),
        coords={"position": (("trial", "when"), [[9.0]])},
    )
    with (
        patch("tal.spatial.ops.path_execution._numba_available", side_effect=AssertionError),
        patch("tal.spatial.ops.batched_path_execution.pack_batched_path_inputs", side_effect=AssertionError),
        pytest.raises(
            ValueError,
            match=r"^spatial\.path_solve\.pose: query name 'position' collides",
        ),
    ):
        solve_pose_path_transform("f1", "f0", graph=graph, query=query)


@pytest.mark.parametrize("lazy", (False, True), ids=("eager-fused", "lazy-generic"))
def test_spatial_core_batched_path_execution_001_restores_query_coordinates(
    lazy: bool,
) -> None:
    from dask.callbacks import Callback

    graph, _ = _registered_path(1)
    query = xr.DataArray(
        [[0.2, 0.8], [0.3, 0.7]],
        dims=("trial", "when"),
        coords={
            "trial": ["a", "b"],
            "when": [10, 20],
            "label": (("trial", "when"), [["x", "y"], ["z", "w"]]),
            "query_value": (("trial", "when"), [[99.0, 99.0], [99.0, 99.0]]),
            "batch_label": ("trial", [1, 2]),
            "request": "caller-only",
        },
    )
    query = query.chunk({"trial": 1, "when": 2}) if lazy else query
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    dataset = result.as_dataset(copy="none")

    if not lazy:
        assert tasks == []
    assert dataset.coords["label"].dims == ("trial", "query")
    np.testing.assert_array_equal(dataset.coords["label"], query.coords["label"])
    xr.testing.assert_identical(dataset.coords["batch_label"], query.coords["batch_label"])
    assert dataset.coords["request"].item() == "caller-only"
    np.testing.assert_allclose(dataset.coords["query_value"], 99.0)
    param_name = read_param_coord_name(dataset)
    assert param_name is not None and param_name != "query_value"
    np.testing.assert_allclose(dataset.coords[param_name], query)


@pytest.mark.parametrize(("trials", "samples"), ((2, 0), (0, 2)))
def test_spatial_core_batched_path_execution_001_empty_position_topology(
    trials: int,
    samples: int,
) -> None:
    graph, providers = _registered_path(1)
    caller = _empty_position_caller(graph, trials=trials, samples=samples)
    before_caller = caller.as_dataset(copy="deep")
    before_provider = providers[0].as_dataset(copy="deep")

    result = caller.to_frame("f0", graph=graph)
    dataset = result.as_dataset(copy="none")

    assert dataset["position"].dims == ("trial", "sample", "axis")
    assert dict(dataset.sizes) == {"sample": samples, "trial": trials, "axis": 3}
    for name in ("trial", "sample", "axis", "time", "group_size"):
        xr.testing.assert_identical(dataset.coords[name], before_caller.coords[name])
    xr.testing.assert_identical(caller.as_dataset(copy="none"), before_caller)
    xr.testing.assert_identical(providers[0].as_dataset(copy="none"), before_provider)


@pytest.mark.parametrize("compiled", (False, True), ids=("scipy", "numba"))
def test_spatial_core_batched_path_execution_001_zero_sample_provider_skips_unreachable_rows(
    compiled: bool,
) -> None:
    if compiled:
        pytest.importorskip("numba")
    graph = FrameGraph()
    provider = Pose(
        _edge_pose(0).as_dataset(copy="deep").isel(sample=slice(0, 0)),
        parent="f0",
        child="f1",
        graph=graph,
    )
    provider.register()
    caller = _empty_position_caller(graph, trials=2, samples=3)
    caller_ds = caller.as_dataset(copy="deep").assign_coords(
        group_size=("trial", np.zeros(2, dtype=np.int64)),
    )
    caller_ds["position"].data[...] = 1.0
    caller = Position(caller_ds, parent="f1", child="point", graph=graph)
    before_provider = provider.as_dataset(copy="deep")
    before_caller = caller.as_dataset(copy="deep")

    with (
        patch("tal.spatial.ops.path_execution._numba_available", return_value=compiled),
        patch("tal.spatial.ops.path_execution.execute_path_query", side_effect=AssertionError("generic")),
    ):
        result = caller.to_frame("f0", graph=graph)
    dataset = result.as_dataset(copy="none")

    assert np.isnan(dataset["position"]).all()
    np.testing.assert_array_equal(dataset.coords["group_size"], np.zeros(2, dtype=np.int64))
    assert result.graph is graph
    xr.testing.assert_identical(provider.as_dataset(copy="none"), before_provider)
    xr.testing.assert_identical(caller.as_dataset(copy="none"), before_caller)


def test_spatial_ownership_batched_path_execution_001_native_provider_storage_is_bounded() -> None:
    samples = 2_049
    batches = 512
    graph = FrameGraph()
    time = np.linspace(0.0, 1.0, samples)
    translation = np.zeros((samples, 3), dtype=np.float64)
    translation[:, 0] = time
    source = _dynamic_pose(
        translation,
        np.zeros(samples).tolist(),
        param=time.tolist(),
    )
    provider = Pose(source, parent="f0", child="f1", graph=graph)
    provider.register()
    before = provider.as_dataset(copy="deep")
    query = xr.DataArray(
        np.full((batches, 1), 0.5),
        dims=("trial", "when"),
    )

    tracemalloc.start()
    with patch("tal.spatial.ops.path_execution._numba_available", return_value=False):
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    np.testing.assert_allclose(result.as_dataset(copy="none")["position"][..., 0], 0.5)
    assert peak < 32 * 1024 * 1024
    xr.testing.assert_identical(provider.as_dataset(copy="none"), before)


def test_spatial_ownership_batched_path_execution_001_caller_blocks_retain_bounded_storage() -> None:
    samples = 131_073
    graph, _ = _registered_path(1)
    caller = _large_position_caller(graph, samples=samples)
    plan = _position_candidate(caller)
    packed = pack_batched_path_inputs(plan)

    assert len(plan.physical_rows.partitions) == 3
    gc.collect()
    tracemalloc.start()
    values, valid = _caller_block(packed, plan, 0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert values.shape == (1, 65_536, 3)
    assert valid.shape == (1, 65_536)
    assert peak < 2_250_000
    assert values.nbytes + valid.nbytes < peak


def test_spatial_lazy_batched_path_execution_001_dask_keeps_generic_route_lazy() -> None:
    """ID: SPATIAL_LAZY_BATCHED_PATH_EXECUTION_001_generic_fallback_is_lazy."""
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    config = CapstoneConfig(trials=3, ship_samples=9, drone_samples=17, trial_chunk=2, sample_chunk=8)
    lazy = prepare_reference_fixture(config, lazy=True)
    eager = prepare_reference_fixture(config)
    tasks: list[object] = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        patch("tal.spatial.ops.path_execution._numba_available", side_effect=AssertionError),
    ):
        result = _public_position(lazy)
    assert tasks == []
    assert result.as_dataset(copy="none")["position"].chunks is not None
    xr.testing.assert_allclose(
        result.as_dataset(copy="none").compute(scheduler="synchronous"),
        _public_position(eager).as_dataset(copy="none"),
    )


@pytest.mark.parametrize("compiled", (False, True))
def test_spatial_lazy_batched_path_execution_001_materializes_each_map_once(
    compiled: bool,
) -> None:
    if compiled:
        pytest.importorskip("numba")
    pytest.importorskip("dask.array")
    import dask.array as da
    from dask import delayed

    calls: list[str] = []

    @delayed
    def query_values() -> np.ndarray:
        calls.append("query")
        return np.asarray([[0.2, 0.8], [0.3, 0.7]])

    query = xr.DataArray(
        da.from_delayed(query_values(), shape=(2, 2), dtype=float),
        dims=("trial", "when"),
    )
    graph, _ = _registered_path(1)
    with patch("tal.spatial.ops.path_execution._numba_available", return_value=compiled):
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)

    assert calls == ["query", "query"]
    assert result.as_dataset(copy="none")["position"].chunks is None


@pytest.mark.parametrize("compiled", (False, True))
def test_spatial_hard_batched_path_execution_001_failure_is_owned_without_retry(
    compiled: bool,
) -> None:
    """ID: SPATIAL_HARD_BATCHED_PATH_EXECUTION_001_owned_no_retry_failure."""
    if compiled:
        pytest.importorskip("numba")
    graph = FrameGraph()
    source = _edge_pose(0).as_dataset(copy="deep")
    source["rotation"].data[2] = 0.0
    provider = Pose(source, parent="f0", child="f1", graph=graph)
    provider.register()
    query = xr.DataArray([[0.5]], dims=("trial", "when"))
    with (
        patch("tal.spatial.ops.path_execution._numba_available", return_value=compiled),
        patch("tal.spatial.ops.path_execution.execute_path_query", side_effect=AssertionError("retry")),
        pytest.raises(ValueError, match=r"^spatial\.path_solve\.pose: edge 0, query 0:"),
    ):
        solve_pose_path_transform("f1", "f0", graph=graph, query=query)


@pytest.mark.parametrize("compiled", (False, True))
def test_spatial_hard_batched_path_execution_001_failure_precedence_crosses_blocks(
    compiled: bool,
) -> None:
    if compiled:
        pytest.importorskip("numba")
    graph = FrameGraph()
    sources = [_edge_pose(edge).as_dataset(copy="deep") for edge in range(2)]
    sources[1]["rotation"].data[3] = 0.0
    sources[0]["rotation"].data[1] = 0.0
    for edge, source in enumerate(sources):
        provider = Pose(source, parent=f"f{edge}", child=f"f{edge + 1}", graph=graph)
        provider.register()
    values = np.full((1, 65_537), 0.5)
    values[0, 0] = 0.25
    values[0, -1] = 0.75
    query = xr.DataArray(values, dims=("trial", "when"))

    with (
        patch("tal.spatial.ops.path_execution._numba_available", return_value=compiled),
        pytest.raises(ValueError, match=r"edge 0, query 65536:"),
    ):
        solve_pose_path_transform("f2", "f0", graph=graph, query=query)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("compiled", (False, True))
def test_spatial_core_batched_path_execution_001_half_turn_and_cancellation_parity(
    dtype,
    compiled: bool,
) -> None:
    if compiled:
        pytest.importorskip("numba")
    graph = FrameGraph()
    source = _edge_pose(0).as_dataset(copy="deep")
    source["position"] = source["position"].astype(dtype)
    source["rotation"] = source["rotation"].astype(dtype)
    source["position"].data[0, 0] = dtype(1.0e16)
    source["position"].data[1, 0] = dtype(-1.0e16 + 2.0)
    identity = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    half_turn = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    source["rotation"].data[:] = half_turn
    source["rotation"].data[0] = identity
    provider = Pose(source, parent="f0", child="f1", graph=graph)
    provider.register()
    query = xr.DataArray([[0.125]], dims=("trial", "when"))
    with patch("tal.spatial.ops.path_execution._numba_available", return_value=compiled):
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    dataset = result.as_dataset(copy="none")

    expected_quaternion = Slerp(
        [0.0, 0.25],
        SciRotation.from_quat(np.stack((identity, half_turn))),
    )([0.125]).as_quat()[0]
    left = np.float64(source["position"].data[0, 0])
    right = np.float64(source["position"].data[1, 0])
    np.testing.assert_allclose(dataset["position"][..., 0], 0.5 * left + 0.5 * right, atol=0.0)
    actual_matrix = SciRotation.from_quat(np.asarray(dataset["rotation"]).reshape(-1, 4)).as_matrix()[0]
    expected_matrix = SciRotation.from_quat(expected_quaternion).as_matrix()
    np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=1e-6 if dtype is np.float32 else 1e-12)


def test_spatial_ownership_batched_path_execution_001_commit_callback_is_unchanged() -> None:
    fixture = prepare_reference_fixture(
        CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
    )
    fixture.caller.as_dataset(copy="none")["position"].attrs["copy_probe"] = _FailingCopy()

    with pytest.raises(UnicodeDecodeError, match="copy failed") as caught:
        _public_position(fixture)
    assert caught.value.__cause__ is None


def test_spatial_ownership_batched_path_execution_001_eager_result_does_not_steal_source() -> None:
    """ID: SPATIAL_OWNERSHIP_BATCHED_PATH_EXECUTION_001_sources_remain_owned."""
    fixture = prepare_reference_fixture(
        CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
    )
    closed: list[str] = []
    fixture.caller.as_dataset(copy="none").set_close(lambda: closed.append("caller"))
    fixture.provider.as_dataset(copy="none").set_close(lambda: closed.append("provider"))
    result = _public_position(fixture)

    result.close()
    assert closed == []
    fixture.caller.close()
    fixture.provider.close()
    assert closed == ["caller", "provider"]
