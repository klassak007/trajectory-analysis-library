from __future__ import annotations

import gc
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import traceback
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np

from benchmarks._spatial_path_fixtures import (
    BenchmarkCaseConfig,
    ColdTimingResult,
    RssResult,
    TimingResult,
)
from tal.utils.numba_support import require_numba

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKER_POLL_SECONDS = 0.01
_RSS_INTERVAL_SECONDS = 0.001
_REPORTED_PACKAGES = ("numpy", "pandas", "xarray", "scipy", "dask", "numba", "llvmlite", "tal")


@dataclass(frozen=True)
class MeasuredRoute:
    """One timed operation/materialization with validation outside timing."""

    name: str
    operation: Callable[[], object]
    materialize: Callable[[object], object]
    validate: Callable[[object], None]
    effective_backend: str = "not-applicable"


@dataclass(frozen=True)
class RssRoutePlan:
    """One route's confirmed configuration and sampled operation."""

    config: BenchmarkCaseConfig
    operation: Callable[[], object]
    materialize: Callable[[object], object]
    effective_backend: str


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def runtime_environment_lines() -> tuple[str, str]:
    """Report runtime and distribution versions without requiring optional packages."""
    runtime = f"python={sys.version.split()[0]}; platform={platform.platform()}"
    packages = "; ".join(f"{name}={_distribution_version(name)}" for name in _REPORTED_PACKAGES)
    return runtime, packages


def numba_environment_line(*, required: bool) -> str:
    """Report Numba runtime state only when a selected route requires it."""
    if not required:
        return "numba_runtime=not-selected"
    numba = require_numba("benchmarks.spatial_paths.environment")
    return f"numba_threads={numba.get_num_threads()}; threading_layer={numba.threading_layer()}"


def _execute_route(route: MeasuredRoute) -> float:
    gc.collect()
    result: object | None = None
    try:
        started = perf_counter()
        result = route.operation()
        result = route.materialize(result)
        elapsed = perf_counter() - started
        route.validate(result)
        return elapsed
    finally:
        result = None


def measure_routes(
    routes: Sequence[MeasuredRoute],
    *,
    warmups: int,
    repeats: int,
) -> tuple[TimingResult, ...]:
    """Measure alternating routes while validating and releasing every output."""
    for _ in range(warmups):
        for route in routes:
            _execute_route(route)
    samples = {route.name: [] for route in routes}
    for repetition in range(repeats):
        ordered = routes if repetition % 2 == 0 else tuple(reversed(routes))
        for route in ordered:
            samples[route.name].append(_execute_route(route))
    return tuple(
        TimingResult(route.name, route.effective_backend, tuple(samples[route.name]))
        for route in routes
    )


def measure_route(route: MeasuredRoute, *, warmups: int, repeats: int) -> TimingResult:
    """Measure one route through the shared validated timing protocol."""
    return measure_routes((route,), warmups=warmups, repeats=repeats)[0]


def measure_allocation(route: MeasuredRoute) -> int:
    """Measure allocation separately, releasing results even after failures."""
    gc.collect()
    result: object | None = None
    try:
        tracemalloc.start()
        try:
            result = route.materialize(route.operation())
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        route.validate(result)
        return peak
    finally:
        result = None


def print_timing_result(result: TimingResult, *, case: str, size: int, edges: int) -> None:
    """Print one validated timing result."""
    samples = ",".join(f"{sample:.6f}" for sample in result.samples)
    print(
        f"case={case}; query={size:,}; edges={edges}; route={result.route}; "
        f"effective_backend={result.effective_backend}; "
        f"samples=[{samples}]; median={result.median:.6f}s; "
        f"min={min(result.samples):.6f}s; max={max(result.samples):.6f}s"
    )


def print_stopgap_estimate(
    route: str,
    *,
    case: str,
    size: int,
    measured_size: int,
    measured: float,
) -> None:
    """Print one explicitly labelled linear stopgap estimate."""
    estimate = measured * size / measured_size
    print(
        f"estimate-only route={route}; case={case}; query={size:,}; "
        f"basis_query={measured_size:,}; linear_estimate={estimate:.3f}s"
    )


def _cold_command(route: str, config: BenchmarkCaseConfig) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "benchmarks.bench_spatial_fused_temporal_paths",
        "--cold-child",
        route,
        "--cases",
        config.case,
        "--sizes",
        str(config.query_size),
        "--edges",
        str(config.edges),
    )


def _cold_marker(route: str, config: BenchmarkCaseConfig) -> str:
    return (
        f"cold-child-complete; route={route}; case={config.case}; "
        f"query={config.query_size}; edges={config.edges}"
    )


def _cold_backend(output: str, *, marker: str) -> str:
    for line in output.splitlines():
        if marker in line:
            fields = dict(part.split("=", 1) for part in line.split("; ") if "=" in part)
            return fields.get("effective_backend", "missing")
    raise RuntimeError("benchmarks.spatial_paths.cold: worker configuration was not confirmed")


def measure_cold_route(route: str, *, config: BenchmarkCaseConfig) -> ColdTimingResult:
    """Measure one fresh module invocation without relying on ``PYTHONPATH``."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    started = perf_counter()
    completed = subprocess.run(
        _cold_command(route, config),
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPOSITORY_ROOT,
        env=environment,
    )
    elapsed = perf_counter() - started
    marker = _cold_marker(route, config)
    backend = _cold_backend(completed.stdout, marker=marker)
    if backend == "missing":
        raise RuntimeError("benchmarks.spatial_paths.cold: worker backend was not confirmed")
    return ColdTimingResult(route, config, backend, elapsed)


def _safe_send(connection: Any, message: tuple[str, object]) -> None:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _rss_worker(connection: Any, config: BenchmarkCaseConfig, route: str) -> None:
    result: object | None = None
    try:
        from benchmarks.bench_spatial_fused_temporal_paths import (
            _prepare_rss_route_plan,
            frozen_fixture,
        )

        actual = config
        backend = "not-applicable"
        if route != "allocation-probe":
            warm = frozen_fixture(config.case, query_size=8, edges=config.edges)
            warm_plan = _prepare_rss_route_plan(route, warm)
            warm_result = warm_plan.materialize(warm_plan.operation())
            del warm_result
            warm_plan = None
            warm = None
            fixture = frozen_fixture(config.case, query_size=config.query_size, edges=config.edges)
            plan = _prepare_rss_route_plan(route, fixture)
            actual = plan.config
            operation = plan.operation
            materialize = plan.materialize
            backend = plan.effective_backend
            plan = None
            fixture = None
        gc.collect()
        connection.send(("ready", (actual, backend)))
        if connection.recv() != "go":
            raise RuntimeError("RSS worker received an invalid start command")
        result = np.ones(config.query_size, dtype=np.float64) if route == "allocation-probe" else materialize(operation())
        connection.send(("done", (actual, backend)))
        if connection.recv() != "release":
            raise RuntimeError("RSS worker received an invalid release command")
    except Exception as exc:  # noqa: BLE001 - child must report arbitrary route failures
        detail = (type(exc).__name__, str(exc), traceback.format_exc())
        _safe_send(connection, ("error", detail))
    finally:
        del result
        connection.close()


def _receive_pipe_message(connection: Any) -> tuple[str, object]:
    try:
        return connection.recv()
    except EOFError as exc:
        raise RuntimeError("benchmarks.spatial_paths.rss: worker pipe closed unexpectedly") from exc


def _receive_after_worker_exit(connection: Any, process: mp.Process) -> tuple[str, object]:
    process.join()
    if connection.poll():
        return _receive_pipe_message(connection)
    raise RuntimeError(f"benchmarks.spatial_paths.rss: worker exited with status {process.exitcode}")


def _receive_message(connection: Any, process: mp.Process) -> tuple[str, object]:
    while True:
        if connection.poll(_WORKER_POLL_SECONDS):
            return _receive_pipe_message(connection)
        if not process.is_alive():
            return _receive_after_worker_exit(connection, process)


def _unwrap_message(message: tuple[str, object], *, expected: str) -> object:
    kind, payload = message
    if kind == "error":
        error_type, error_message, worker_traceback = payload
        raise RuntimeError(
            f"benchmarks.spatial_paths.rss: worker {error_type}: {error_message}\n{worker_traceback}"
        )
    if kind != expected:
        raise RuntimeError(f"benchmarks.spatial_paths.rss: expected {expected!r}, received {kind!r}")
    return payload


def _sample_rss(observed: Any, process: mp.Process) -> int | None:
    try:
        return observed.memory_info().rss
    except Exception as exc:
        if process.is_alive():
            raise RuntimeError("benchmarks.spatial_paths.rss: RSS sampling failed") from exc
        return None


def _sample_until_done(connection: Any, process: mp.Process, observed: Any, samples: list[int]) -> object:
    while True:
        if connection.poll():
            return _unwrap_message(_receive_pipe_message(connection), expected="done")
        if not process.is_alive():
            return _unwrap_message(_receive_message(connection, process), expected="done")
        sample = _sample_rss(observed, process)
        if sample is not None:
            samples.append(sample)
        sleep(_RSS_INTERVAL_SECONDS)


def measure_rss(route: str, *, config: BenchmarkCaseConfig) -> RssResult:
    """Measure one route in a fresh spawned process using the frozen protocol."""
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - manual environment boundary
        raise RuntimeError("Spatial path benchmark memory mode requires the development dependency psutil") from exc
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_rss_worker, args=(child, config, route))
    started = False
    try:
        process.start()
        started = True
        child.close()
        actual, backend = _unwrap_message(_receive_message(parent, process), expected="ready")
        baseline_samples = [psutil.Process(process.pid).memory_info().rss for _ in range(20)]
        observed = psutil.Process(process.pid)
        parent.send("go")
        samples = list(baseline_samples)
        completed, completed_backend = _sample_until_done(parent, process, observed, samples)
        if actual != config or completed != config or backend != completed_backend:
            raise RuntimeError("benchmarks.spatial_paths.rss: worker configuration mismatch")
        try:
            samples.append(observed.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        parent.send("release")
        process.join(timeout=10.0)
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError(f"benchmarks.spatial_paths.rss: worker exited with status {process.exitcode}")
        return RssResult(
            route,
            config,
            backend,
            int(np.median(baseline_samples)),
            max(samples),
            len(samples),
        )
    finally:
        parent.close()
        child.close()
        if started and process.is_alive():
            process.terminate()
            process.join()


__all__ = [
    "MeasuredRoute",
    "RssRoutePlan",
    "measure_allocation",
    "measure_cold_route",
    "measure_route",
    "measure_routes",
    "measure_rss",
    "numba_environment_line",
    "print_stopgap_estimate",
    "print_timing_result",
    "runtime_environment_lines",
]
