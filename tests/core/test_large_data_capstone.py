from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed
from dask.callbacks import Callback

from benchmarks.bench_capstone_workflow import (
    MAIN_LAYOUT,
    CapstoneConfig,
    CapstoneFixture,
    CapstoneOutputs,
    benchmark_report,
    capstone_fixture,
    direct_xarray_route,
    identity_rotation,
    materialize,
    public_tal_route,
    register_provider,
    task_count,
    validate_outputs,
)
from tal.core import AnalysisObject
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.io import AOZarrReadOptions

SMALL = CapstoneConfig(
    trials=4,
    ship_samples=9,
    drone_samples=33,
    trial_chunk=2,
    sample_chunk=8,
)


def _bomb_array(shape: tuple[int, ...]) -> da.Array:
    @delayed
    def fail() -> np.ndarray:
        raise AssertionError("unselected capstone payload executed")

    return da.from_delayed(fail(), shape=shape, dtype=np.float64)


def _with_unused_bombs(fixture: CapstoneFixture) -> CapstoneFixture:
    ship = fixture.ship.assign(
        unused=(("trial", "sample", "axis"), _bomb_array(fixture.ship["unused"].shape))
    )
    drone = fixture.drone.assign(
        unused=(("trial", "sample", "axis"), _bomb_array(fixture.drone["unused"].shape))
    )
    return replace(fixture, ship=ship, drone=drone)


def _recorded_array(
    values: np.ndarray,
    *,
    name: str,
    calls: dict[str, int],
) -> da.Array:
    @delayed
    def record() -> np.ndarray:
        calls[name] += 1
        return values

    return da.from_delayed(record(), shape=values.shape, dtype=values.dtype)


def _with_shared_recorders(
    fixture: CapstoneFixture,
    calls: dict[str, int],
) -> CapstoneFixture:
    ship_values = np.asarray(fixture.ship_provider["position"].data)
    drone_values = np.asarray(fixture.drone_provider["position"].data)
    ship_data = _recorded_array(ship_values, name="ship", calls=calls)
    drone_data = _recorded_array(drone_values, name="drone", calls=calls)
    ship = fixture.ship.assign(position=(("trial", "sample", "axis"), ship_data))
    drone = fixture.drone.assign(position=(("trial", "sample", "axis"), drone_data))
    ship_provider = fixture.ship_provider.assign(position=(("trial", "sample", "axis"), ship_data))
    drone_provider = fixture.drone_provider.assign(position=(("trial", "sample", "axis"), drone_data))
    return CapstoneFixture(ship, drone, ship_provider, drone_provider)


def _assert_output_topology(outputs: object) -> None:
    relative = outputs.relative_distance
    grouped = outputs.grouped_minimum
    window = outputs.approach_window
    assert read_roles(relative) == (True, "sample", ("trial",), ())
    assert read_param_coord_name(relative) == "time"
    assert read_sequence_size_coord_name(relative) == "group_size"
    assert tuple(relative.coords["outcome"].data) == ("intercept", "miss", "intercept", "miss")
    assert read_roles(grouped) == (True, None, ("group_key",), ())
    event_dim = next(dim for dim in window.dims if dim.startswith("event"))
    assert read_roles(window) == (True, "tau", ("trial", event_dim), ())
    assert read_param_coord_name(window) == "tau"
    assert read_sequence_size_coord_name(window) == "tau_len"
    assert type(relative.xindexes["trial"]) is type(window.xindexes["trial"])
    assert relative.xindexes["trial"].equals(window.xindexes["trial"])


def test_core_lazy_large_workflow_001_stays_lazy_and_matches_independent_reference() -> None:
    """ID: CORE_LAZY_LARGE_WORKFLOW_001_public_capstone_is_lazy_and_equivalent."""
    eager = capstone_fixture(SMALL, lazy=False)
    expected = materialize(direct_xarray_route(eager))
    eager_public = materialize(public_tal_route(eager))
    validate_outputs(eager_public, expected)
    calls = {"ship": 0, "drone": 0}
    lazy = _with_unused_bombs(_with_shared_recorders(eager, calls))
    source_snapshots = tuple(ds.copy(deep=True) for ds in lazy.__dict__.values())
    tasks: list[object] = []
    with Callback(pretask=lambda key, _dsk, _state: tasks.append(key)):
        ship = MAIN_LAYOUT.wrap(lazy.ship, data_vars="position")
        drone = MAIN_LAYOUT.wrap(lazy.drone, data_vars="position")
        aligned_difference = drone - ship.param.interp_like(drone, on="time")
        planned = public_tal_route(lazy)
    assert tasks == []
    assert calls == {"ship": 0, "drone": 0}
    with Callback(pretask=lambda key, _dsk, _state: tasks.append(key)):
        computed = dask.compute(
            aligned_difference.as_dataset(copy="none"),
            planned.relative_distance,
            planned.grouped_minimum,
            planned.approach_window,
            scheduler="synchronous",
        )
    actual = CapstoneOutputs(*computed[1:])
    direct_ship = eager.ship["position"].swap_dims({"sample": "time"}).drop_vars("sample")
    direct_difference = eager.drone["position"] - direct_ship.interp(time=eager.drone.coords["time"])
    np.testing.assert_allclose(computed[0]["position"], direct_difference)
    assert read_roles(computed[0]) == (True, "sample", ("trial",), ("axis",))
    np.testing.assert_array_equal(
        computed[0].coords["group_size"],
        np.full(SMALL.trials, SMALL.drone_samples),
    )
    assert bool(computed[0].coords["valid"].all())
    validate_outputs(actual, eager_public)
    validate_outputs(actual, expected)
    assert calls == {"ship": 1, "drone": 1}
    assert tasks
    _assert_output_topology(actual)
    for source, snapshot in zip(lazy.__dict__.values(), source_snapshots, strict=True):
        xr.testing.assert_identical(source, snapshot)


def test_spatial_lazy_bound_path_001_registered_dynamic_path_preserves_association() -> None:
    """ID: SPATIAL_LAZY_BOUND_PATH_001_registered_path_is_lazy_and_graph_owned."""
    from benchmarks.bench_capstone_workflow import MAIN_LAYOUT
    from tal.frames import FrameGraph
    from tal.spatial import Position

    fixture = capstone_fixture(SMALL, lazy=True)
    graph = FrameGraph()
    rotation = identity_rotation(graph)
    register_provider(fixture.ship_provider, child="ship", graph=graph, rotation=rotation)
    register_provider(fixture.drone_provider, child="drone", graph=graph, rotation=rotation)
    source = Position(
        MAIN_LAYOUT.wrap(fixture.drone, data_vars="position"),
        parent="world",
        child="drone",
        graph=graph,
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, _dsk, _state: tasks.append(key)):
        result = source.to_frame("ship", graph=graph)
    assert tasks == []
    assert result.graph is graph
    assert result.frames.ids() == ("ship", "drone")
    assert graph.get_frame("ship") is not None
    assert graph.get_frame("drone") is not None


@pytest.mark.parametrize(
    ("parent", "target", "extra_edge"),
    (("world", "ship", False), ("ship", "world", False), ("deck", "world", True)),
)
def test_spatial_lazy_bound_path_001_ragged_queries_skip_invalid_pose_placeholders(
    parent: str,
    target: str,
    extra_edge: bool,
) -> None:
    """Ragged-query coverage for SPATIAL_LAZY_BOUND_PATH_001."""
    from benchmarks.bench_capstone_workflow import MAIN_LAYOUT
    from tal.frames import FrameGraph
    from tal.spatial import Position

    fixture = capstone_fixture(SMALL, lazy=True)
    source = fixture.drone.drop_vars("unused").copy(deep=False)
    sizes = np.asarray([33, 21, 33, 21], dtype=np.int64)
    time = np.broadcast_to(source.coords["time"].data, (4, 33)).copy()
    values = np.asarray(source["position"].compute().data)
    for row, size in enumerate(sizes):
        time[row, size:] = np.nan
        values[row, size:, :] = np.nan
    source = source.assign(
        position=(("trial", "sample", "axis"), da.from_array(values, chunks=(2, 8, 3)))
    ).assign_coords(time=(("trial", "sample"), time), group_size=("trial", sizes))
    graph = FrameGraph()
    register_provider(
        fixture.ship_provider,
        child="ship",
        graph=graph,
        rotation=identity_rotation(graph),
    )
    if extra_edge:
        register_provider(
            fixture.ship_provider,
            parent="ship",
            child="deck",
            graph=graph,
            rotation=identity_rotation(graph),
        )
    position = Position(
        MAIN_LAYOUT.wrap(source, data_vars="position"),
        parent=parent,
        child="drone",
        graph=graph,
    )
    result = position.to_frame(target, graph=graph).as_dataset(copy="none").compute()
    np.testing.assert_array_equal(result.coords["group_size"], sizes)
    assert np.isnan(result["position"].isel(trial=1, sample=slice(21, None))).all()


def test_core_perf_lazy_graph_growth_001_is_bounded_and_approximately_linear() -> None:
    """ID: CORE_PERF_LAZY_GRAPH_GROWTH_001_partition_growth_is_behaviorally_bounded."""
    one = CapstoneConfig(8, 17, 65, trial_chunk=8, sample_chunk=128)
    four = replace(one, trial_chunk=2)
    one_tasks = task_count(public_tal_route(capstone_fixture(one, lazy=True)))
    four_tasks = task_count(public_tal_route(capstone_fixture(four, lazy=True)))
    assert one_tasks > 0
    assert one_tasks < four_tasks <= 5 * one_tasks
    eager_reference = materialize(direct_xarray_route(capstone_fixture(four, lazy=False)))
    validate_outputs(public_tal_route(capstone_fixture(four, lazy=True)), eager_reference)


def test_io_ownership_lazy_roundtrip_001_persists_selects_and_closes_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_OWNERSHIP_LAZY_ROUNDTRIP_001_capstone_zarr_resource_closes_once."""
    output = public_tal_route(capstone_fixture(SMALL, lazy=True)).relative_distance
    source = AnalysisObject(output)
    store = tmp_path / "capstone.zarr"
    source.io.to_zarr(str(store))
    backend = xr.open_zarr(str(store), chunks={})
    opened = backend.copy(deep=False)
    closed: list[bool] = []

    def close_backend() -> None:
        closed.append(True)
        backend.close()

    opened.set_close(close_backend)
    monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: opened)
    loaded = AnalysisObject.from_zarr(
        str(store),
        opts=AOZarrReadOptions(chunks={}),
    )
    selected = loaded.select_vars("datavar")
    xr.testing.assert_allclose(
        selected.as_dataset(copy="none")["datavar"].compute(),
        output["datavar"].compute(),
    )
    selected.close()
    loaded.close()
    selected.close()
    loaded.close()
    assert closed == [True]


def test_core_hard_eager_boundary_001_capstone_planning_avoids_materialization() -> None:
    """ID: CORE_HARD_EAGER_BOUNDARY_001_capstone_planning_has_no_payload_boundary."""
    fixture = _with_unused_bombs(capstone_fixture(SMALL, lazy=True))
    tasks: list[object] = []
    with Callback(pretask=lambda key, _dsk, _state: tasks.append(key)):
        result = public_tal_route(fixture)
    assert tasks == []
    assert task_count(result) > 0
    assert all(getattr(ds[next(iter(ds.data_vars))].data, "chunks", None) for ds in result.__dict__.values())


def test_capstone_benchmark_protocol_small_fixture() -> None:
    report = benchmark_report(SMALL, warmups=0, repeats=1)
    assert report["fixture"]["dtype"] == "float64"
    assert report["environment"]["scheduler"] == "synchronous"
    assert set(report["dask_tasks"]) == {"direct_xarray", "public_tal"}
    assert all(value > 0 for value in report["dask_tasks"].values())
    for section in ("eager", "dask_graph_build", "dask_materialize"):
        for route in ("direct_xarray", "public_tal"):
            assert len(report[section][route]["seconds"]) == 1
            assert report[section][route]["peak_bytes"][0] > 0


def test_doc_capstone_workflow_001_uses_accepted_public_boundaries() -> None:
    """ID: DOC_CAPSTONE_WORKFLOW_001_tutorial_uses_current_public_workflow."""
    notebook = json.loads(
        Path("examples/tutorial/14_capstone_autonomous_landing.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "trajectory_layout = AnalysisLayoutSpec(" in code
    assert "Pose.from_components(" in code
    assert ").register()" in code
    assert "as_dataset(copy='shallow')" in code
    for retired in ("frame_retag", "get_or_create_frame", "bind_pose", "copy='none'", "._data"):
        assert retired not in code
