"""Manual eager/Dask capstone benchmark; no CI wall-clock threshold."""

from __future__ import annotations

import json
import platform
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import dask
import dask.array as da
import numpy as np
import scipy
import xarray as xr

from benchmarks._spatial_path_benchmark_protocol import (
    MeasuredRoute,
    measure_allocation,
    measure_routes,
)
from tal.core import AnalysisLayoutSpec
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation


@dataclass(frozen=True)
class CapstoneConfig:
    """Frozen dimensions and chunking for the representative workflow."""

    trials: int = 64
    ship_samples: int = 129
    drone_samples: int = 1_025
    trial_chunk: int = 8
    sample_chunk: int = 256


@dataclass(frozen=True)
class CapstoneFixture:
    """External inputs for one benchmark route."""

    ship: xr.Dataset
    drone: xr.Dataset
    ship_provider: xr.Dataset
    drone_provider: xr.Dataset


@dataclass(frozen=True)
class CapstoneOutputs:
    """Numerical outputs shared by TAL and direct-xarray routes."""

    relative_distance: xr.Dataset
    grouped_minimum: xr.Dataset
    approach_window: xr.Dataset


@dataclass(frozen=True)
class PreparedCapstone:
    """Public caller and graph prepared without evaluating any payload."""

    position: Position
    graph: FrameGraph
    anchors: xr.DataArray


DEFAULT_CONFIG = CapstoneConfig()
MAIN_LAYOUT = AnalysisLayoutSpec(
    sequence_dim="sample",
    batch_dims=("trial",),
    core_dims=("axis",),
    param_coord="time",
    sequence_size_coord="group_size",
)
PROVIDER_LAYOUT = AnalysisLayoutSpec(
    sequence_dim="sample",
    batch_dims=("trial",),
    core_dims=("axis",),
    param_coord="time",
)
STATIC_ROTATION_LAYOUT = AnalysisLayoutSpec(core_dims=("quat",))
TAU = np.arange(-2.0, 1.0 + 0.05, 0.1, dtype=np.float64)


def _positions(config: CapstoneConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ship_time = np.linspace(0.0, 10.0, config.ship_samples, dtype=np.float64)
    drone_time = np.linspace(0.0, 10.0, config.drone_samples, dtype=np.float64)
    trial_offset = np.arange(config.trials, dtype=np.float64)[:, None] * 0.01
    ship = np.zeros((config.trials, config.ship_samples, 3), dtype=np.float64)
    drone = np.zeros((config.trials, config.drone_samples, 3), dtype=np.float64)
    ship[..., 0] = 0.4 * ship_time + trial_offset
    drone[..., 0] = 24.0 - 2.2 * drone_time + trial_offset
    drone[..., 1] = 4.0 * (np.arange(config.trials) % 2)[:, None]
    drone[..., 2] = 6.0 - 0.45 * drone_time
    return ship_time, drone_time, ship, drone


def _array(values: np.ndarray, *, config: CapstoneConfig, lazy: bool) -> object:
    if not lazy:
        return values
    chunks = (config.trial_chunk, min(config.sample_chunk, values.shape[1]), values.shape[2])
    return da.from_array(values, chunks=chunks)


def _dataset(
    values: np.ndarray,
    time_values: np.ndarray,
    *,
    config: CapstoneConfig,
    lazy: bool,
    metadata: bool,
) -> xr.Dataset:
    coords: dict[str, object] = {
        "trial": np.arange(config.trials),
        "sample": np.arange(time_values.size),
        "axis": ["x", "y", "z"],
        "time": ("sample", time_values),
    }
    if metadata:
        coords["group_size"] = ("trial", np.full(config.trials, time_values.size, dtype=np.int64))
        coords["outcome"] = ("trial", np.where(np.arange(config.trials) % 2, "miss", "intercept"))
        coords["approach_time"] = ("trial", 3.8 + 0.3 * (np.arange(config.trials) % 2))
    payload = _array(values, config=config, lazy=lazy)
    return xr.Dataset({"position": (("trial", "sample", "axis"), payload)}, coords=coords)


def capstone_fixture(config: CapstoneConfig = DEFAULT_CONFIG, *, lazy: bool) -> CapstoneFixture:
    """Build the fixed float64 source outside measured regions."""
    ship_time, drone_time, ship_values, drone_values = _positions(config)
    ship = _dataset(ship_values, ship_time, config=config, lazy=lazy, metadata=True)
    drone = _dataset(drone_values, drone_time, config=config, lazy=lazy, metadata=True)
    ship["unused"] = ship["position"] + 100.0
    drone["unused"] = drone["position"] + 100.0
    ship_provider = _dataset(ship_values, ship_time, config=config, lazy=lazy, metadata=False)
    drone_provider = _dataset(drone_values, drone_time, config=config, lazy=lazy, metadata=False)
    return CapstoneFixture(ship, drone, ship_provider, drone_provider)


def identity_rotation(graph: FrameGraph) -> Rotation:
    data = xr.DataArray(
        [0.0, 0.0, 0.0, 1.0],
        dims="quat",
        coords={"quat": ["x", "y", "z", "w"]},
        name="rotation",
    )
    return Rotation(STATIC_ROTATION_LAYOUT.wrap(data), graph=graph)


def register_provider(
    source: xr.Dataset,
    *,
    parent: str = "world",
    child: str,
    graph: FrameGraph,
    rotation: Rotation,
) -> None:
    ao = PROVIDER_LAYOUT.wrap(source, data_vars="position")
    position = Position(ao, parent=parent, child=child, graph=graph)
    Pose.from_components(
        rotation,
        position,
        parent=parent,
        child=child,
        graph=graph,
    ).register()


def _dataset_view(ao: object) -> xr.Dataset:
    return ao.as_dataset(copy="shallow")  # type: ignore[attr-defined]


def _primary(ds: xr.Dataset) -> xr.DataArray:
    return ds[next(iter(ds.data_vars))]


def prepare_public_tal(fixture: CapstoneFixture) -> PreparedCapstone:
    """Wrap inputs and register providers through the public TAL surface."""
    _ = MAIN_LAYOUT.wrap(fixture.ship, data_vars="position")
    drone = MAIN_LAYOUT.wrap(fixture.drone, data_vars="position")
    graph = FrameGraph()
    rotation = identity_rotation(graph)
    register_provider(fixture.ship_provider, child="ship", graph=graph, rotation=rotation)
    register_provider(fixture.drone_provider, child="drone", graph=graph, rotation=rotation)
    drone_position = Position(drone, parent="world", child="drone", graph=graph)
    return PreparedCapstone(drone_position, graph, fixture.drone.coords["approach_time"])


def analyze_public_tal(prepared: PreparedCapstone) -> CapstoneOutputs:
    """Run all three analyses against an already prepared public graph."""
    distance = prepared.position.to_frame("ship", graph=prepared.graph).norm()
    grouped = distance.min(dim="sample").group.groupby("outcome").mean(dim="trial")
    window = distance.events.around(
        prepared.anchors,
        pre=2.0,
        post=1.0,
        dt=0.1,
        layout="segments",
    )
    return CapstoneOutputs(_dataset_view(distance), _dataset_view(grouped), _dataset_view(window))


def public_tal_route(fixture: CapstoneFixture) -> CapstoneOutputs:
    """Execute the complete public route, including preparation."""
    return analyze_public_tal(prepare_public_tal(fixture))


def _interpolate_ship(fixture: CapstoneFixture) -> xr.DataArray:
    source = fixture.ship_provider["position"].swap_dims({"sample": "time"}).drop_vars("sample")
    query = fixture.drone_provider.coords["time"]
    return source.interp(time=query)


def _direct_window(
    distance: xr.DataArray,
    time_values: xr.DataArray,
    anchors: xr.DataArray,
) -> xr.DataArray:
    query = anchors + xr.DataArray(TAU, dims="tau", coords={"tau": TAU})
    source = distance.assign_coords(time=time_values).swap_dims({"sample": "time"}).drop_vars("sample")
    return source.interp(time=query).expand_dims(event=[0]).transpose("trial", "event", "tau")


def direct_xarray_route(fixture: CapstoneFixture) -> CapstoneOutputs:
    """Execute an equivalent direct-xarray numerical pipeline."""
    ship = _interpolate_ship(fixture)
    drone = fixture.drone["position"]
    relative = np.sqrt(((drone - ship) ** 2).sum("axis"))
    grouped = relative.min("sample").groupby(fixture.drone.coords["outcome"]).mean("trial")
    grouped = grouped.rename({"outcome": "group_key"})
    window = _direct_window(
        relative,
        fixture.drone.coords["time"],
        fixture.drone.coords["approach_time"],
    )
    return CapstoneOutputs(
        relative.to_dataset(name="datavar"),
        grouped.to_dataset(name="datavar"),
        window.to_dataset(name="datavar"),
    )


def materialize(outputs: CapstoneOutputs) -> CapstoneOutputs:
    """Compute all outputs together so shared source work remains shared."""
    arrays = dask.compute(
        outputs.relative_distance,
        outputs.grouped_minimum,
        outputs.approach_window,
        scheduler="synchronous",
    )
    return CapstoneOutputs(*arrays)


def validate_outputs(actual: CapstoneOutputs, expected: CapstoneOutputs) -> None:
    """Validate every retained numerical result against the independent route."""
    if any(getattr(_primary(value).data, "chunks", None) is not None for value in actual.__dict__.values()):
        actual = materialize(actual)
    pairs = zip(actual.__dict__.values(), expected.__dict__.values(), strict=True)
    for observed_ds, reference_ds in pairs:
        observed = _primary(observed_ds)
        reference = _primary(reference_ds)
        assert observed.shape == reference.shape
        np.testing.assert_allclose(observed.data, reference.data, rtol=1.0e-10, atol=1.0e-10)


def task_count(outputs: CapstoneOutputs) -> int:
    """Return unioned Dask task count without depending on private task names."""
    keys: set[object] = set()
    for ds in outputs.__dict__.values():
        keys.update(_dataset_task_keys(ds))
    return len(keys)


def _dataset_task_keys(ds: xr.Dataset) -> set[object]:
    keys: set[object] = set()
    for variable in ds.variables.values():
        graph = getattr(variable.data, "dask", None)
        if graph is not None:
            keys.update(graph.keys())
    return keys


def _sample_routes(
    routes: dict[str, Callable[[], object]],
    expected: CapstoneOutputs,
    *,
    warmups: int = 2,
    repeats: int = 7,
    validate: Callable[[object], None] | None = None,
) -> dict[str, object]:
    validator = validate or (lambda actual: validate_outputs(actual, expected))
    measured = tuple(
        MeasuredRoute(name, operation, lambda value: value, validator)
        for name, operation in routes.items()
    )
    timings = measure_routes(measured, warmups=warmups, repeats=repeats)
    peaks: dict[str, list[int]] = {name: [] for name in routes}
    for repetition in range(repeats):
        order = measured if repetition % 2 == 0 else tuple(reversed(measured))
        for route in order:
            peaks[route.name].append(measure_allocation(route))
    return {row.route: _summarize(row.samples, peaks[row.route]) for row in timings}


def _summarize(seconds: tuple[float, ...], peaks: list[int]) -> dict[str, object]:
    return {
        "seconds": list(seconds),
        "peak_bytes": peaks,
        "median_seconds": statistics.median(seconds),
        "median_peak_bytes": statistics.median(peaks),
    }


def _prepared_report(fixture, expected, *, warmups, repeats):
    prepared = prepare_public_tal(fixture)
    preparation = _sample_routes(
        {"public_tal": lambda: prepare_public_tal(fixture)},
        expected,
        warmups=warmups,
        repeats=repeats,
        validate=lambda value: validate_outputs(analyze_public_tal(value), expected),
    )
    analysis = _sample_routes(
        {"public_tal": lambda: analyze_public_tal(prepared)},
        expected,
        warmups=warmups,
        repeats=repeats,
    )
    return {"preparation": preparation, "prepared_analysis": analysis}


def benchmark_report(
    config: CapstoneConfig = DEFAULT_CONFIG,
    *,
    warmups: int = 2,
    repeats: int = 7,
) -> dict[str, object]:
    """Run the reproducible eager/Dask benchmark protocol."""
    eager = capstone_fixture(config, lazy=False)
    lazy = capstone_fixture(config, lazy=True)
    expected = materialize(direct_xarray_route(eager))
    eager_routes = {
        "direct_xarray": lambda: direct_xarray_route(eager),
        "public_tal": lambda: public_tal_route(eager),
    }
    graph_routes = {
        "direct_xarray": lambda: direct_xarray_route(lazy),
        "public_tal": lambda: public_tal_route(lazy),
    }
    lazy_graphs = {name: operation() for name, operation in graph_routes.items()}
    materialize_routes = {name: (lambda value=value: materialize(value)) for name, value in lazy_graphs.items()}
    report = _report(config, eager_routes, graph_routes, materialize_routes, lazy_graphs, expected, warmups, repeats)
    report["public_stages"] = {
        name: _prepared_report(fixture, expected, warmups=warmups, repeats=repeats)
        for name, fixture in (("eager", eager), ("dask_graph_build", lazy))
    }
    return report


def _report(
    config: CapstoneConfig,
    eager_routes: dict[str, Callable[[], CapstoneOutputs]],
    graph_routes: dict[str, Callable[[], CapstoneOutputs]],
    materialize_routes: dict[str, Callable[[], CapstoneOutputs]],
    lazy_graphs: dict[str, CapstoneOutputs],
    expected: CapstoneOutputs,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    return {
        "fixture": {**config.__dict__, "dtype": "float64"},
        "measurement": {
            "timing_allocation_separate": True,
            "complete_routes_include_preparation": True,
            "warmups": warmups,
            "repeats": repeats,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xarray": xr.__version__,
            "scipy": scipy.__version__,
            "dask": dask.__version__,
            "scheduler": "synchronous",
        },
        "eager": _sample_routes(eager_routes, expected, warmups=warmups, repeats=repeats),
        "dask_graph_build": _sample_routes(graph_routes, expected, warmups=warmups, repeats=repeats),
        "dask_materialize": _sample_routes(materialize_routes, expected, warmups=warmups, repeats=repeats),
        "dask_tasks": {name: task_count(value) for name, value in lazy_graphs.items()},
        "dask_chunks": {
            name: _primary(value.relative_distance).chunks
            for name, value in lazy_graphs.items()
        },
    }


def main() -> None:
    """Print the maintained benchmark report as JSON."""
    print(json.dumps(benchmark_report(), indent=2))


if __name__ == "__main__":
    main()
