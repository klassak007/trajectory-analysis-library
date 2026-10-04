from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import dask
import numpy as np
import scipy
import xarray as xr

from benchmarks.bench_capstone_workflow import CapstoneConfig
from tal.utils.numba_support import _numba_available, require_numba

_ROOT = Path(__file__).resolve().parents[1]
_POLL_SECONDS = 0.01
_RSS_INTERVAL_SECONDS = 0.001
_COLD_MARKER = "BATCHED_PATH_COLD="
_RSS_MARKER = "BATCHED_PATH_RSS="


@dataclass(frozen=True)
class BatchedColdResult:
    process_seconds: float
    import_seconds: float
    setup_seconds: float
    first_call_seconds: float
    validation_seconds: float
    backend: str
    config: CapstoneConfig


@dataclass(frozen=True)
class BatchedRssResult:
    baseline_bytes: int
    peak_bytes: int
    sample_count: int
    backend: str
    config: CapstoneConfig

    @property
    def incremental_bytes(self) -> int:
        return self.peak_bytes - self.baseline_bytes


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _numba_runtime_environment(backend: str) -> dict[str, str]:
    if backend != "numba":
        return {
            "numba_threads": "not-selected",
            "numba_threading_layer": "not-selected",
        }
    numba = require_numba("benchmarks.batched_path.environment")
    return {
        "numba_threads": str(numba.get_num_threads()),
        "numba_threading_layer": str(numba.threading_layer()),
    }


def benchmark_environment(*, backend: str) -> dict[str, str]:
    """Describe the frozen benchmark's execution environment."""
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": _distribution_version("pandas"),
        "scipy": scipy.__version__,
        "xarray": xr.__version__,
        "dask": dask.__version__,
        "numba": _distribution_version("numba"),
        "llvmlite": _distribution_version("llvmlite"),
        "scheduler": "synchronous",
        "selected_backend": backend,
    }
    return {**environment, **_numba_runtime_environment(backend)}


def dataset_chunks(value: xr.Dataset) -> tuple[tuple[int, ...], ...] | None:
    """Report chunks for the first measured Dataset variable."""
    chunks = value[next(iter(value.data_vars))].chunks
    return None if chunks is None else tuple(tuple(int(size) for size in lane) for lane in chunks)


def cold_child(config: CapstoneConfig) -> dict[str, object]:
    """Measure setup, first call, and validation inside the cold child."""
    setup_started = perf_counter()
    from benchmarks.bench_batched_fused_path_reference import (
        _public_position,
        _validate_position_topology,
        prepare_reference_fixture,
    )

    prepared = prepare_reference_fixture(config)
    setup_seconds = perf_counter() - setup_started
    started = perf_counter()
    result = _public_position(prepared)
    first_call_seconds = perf_counter() - started
    validation_started = perf_counter()
    _validate_position_topology(result, prepared)
    validation_seconds = perf_counter() - validation_started
    return {
        "setup_seconds": setup_seconds,
        "first_call_seconds": first_call_seconds,
        "validation_seconds": validation_seconds,
        "backend": "numba" if _numba_available() else "scipy",
        "config": config.__dict__,
    }


def measure_cold_public(config: CapstoneConfig) -> BatchedColdResult:
    """Measure import, setup, and first call in a fresh empty-cache process."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="tal-batched-path-cache-") as cache:
        environment["NUMBA_CACHE_DIR"] = cache
        process_started = perf_counter()
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "benchmarks._batched_path_cold_entry",
                json.dumps(config.__dict__),
            ),
            cwd=_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        process_seconds = perf_counter() - process_started
    line = next(line for line in completed.stdout.splitlines() if line.startswith(_COLD_MARKER))
    payload = json.loads(line.removeprefix(_COLD_MARKER))
    return BatchedColdResult(
        process_seconds,
        float(payload["import_seconds"]),
        float(payload["setup_seconds"]),
        float(payload["first_call_seconds"]),
        float(payload["validation_seconds"]),
        str(payload["backend"]),
        config,
    )


def _safe_send(connection: Any, message: tuple[str, object]) -> None:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _rss_worker(connection: Any, config: CapstoneConfig) -> None:
    result: object | None = None
    try:
        from benchmarks.bench_batched_fused_path_reference import (
            _public_position,
            prepare_reference_fixture,
        )

        warm = prepare_reference_fixture(
            CapstoneConfig(trials=2, ship_samples=5, drone_samples=9, trial_chunk=1, sample_chunk=4)
        )
        warm_result = _public_position(warm)
        del warm_result, warm
        prepared = prepare_reference_fixture(config)
        backend = "numba" if _numba_available() else "scipy"
        gc.collect()
        connection.send(("ready", (config, backend)))
        if connection.recv() != "go":
            raise RuntimeError("batched path RSS worker received an invalid start command")
        result = _public_position(prepared)
        connection.send(("done", (config, backend)))
        if connection.recv() != "release":
            raise RuntimeError("batched path RSS worker received an invalid release command")
    except Exception as exc:  # noqa: BLE001 - child reports arbitrary failures.
        _safe_send(connection, ("error", (type(exc).__name__, str(exc), traceback.format_exc())))
    finally:
        del result
        connection.close()


def _receive(connection: Any, process: mp.Process, expected: str) -> object:
    while process.is_alive() and not connection.poll(_POLL_SECONDS):
        pass
    if not connection.poll():
        process.join()
        raise RuntimeError(f"batched path RSS worker exited with status {process.exitcode}")
    kind, payload = connection.recv()
    if kind == "error":
        name, message, worker_traceback = payload
        raise RuntimeError(f"batched path RSS worker {name}: {message}\n{worker_traceback}")
    if kind != expected:
        raise RuntimeError(f"batched path RSS expected {expected!r}, received {kind!r}")
    return payload


def _sample_until_done(connection: Any, process: mp.Process, observed: Any) -> tuple[list[int], object]:
    samples: list[int] = []
    while True:
        if connection.poll():
            return samples, _receive(connection, process, "done")
        if not process.is_alive():
            return samples, _receive(connection, process, "done")
        samples.append(observed.memory_info().rss)
        sleep(_RSS_INTERVAL_SECONDS)


def measure_batched_rss(config: CapstoneConfig) -> BatchedRssResult:
    """Measure incremental RSS for one retained public result in a fresh process."""
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - manual optional boundary.
        raise RuntimeError("batched path RSS mode requires psutil") from exc
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_rss_worker, args=(child, config))
    started = False
    try:
        process.start()
        started = True
        child.close()
        actual, backend = _receive(parent, process, "ready")
        observed = psutil.Process(process.pid)
        baseline = [observed.memory_info().rss for _ in range(20)]
        parent.send("go")
        samples, completed = _sample_until_done(parent, process, observed)
        if actual != config or completed != (config, backend):
            raise RuntimeError("batched path RSS worker configuration changed")
        parent.send("release")
        process.join(timeout=10.0)
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError(f"batched path RSS worker exited with status {process.exitcode}")
        return BatchedRssResult(
            int(statistics.median(baseline)),
            max((*baseline, *samples)),
            len(baseline) + len(samples),
            backend,
            config,
        )
    finally:
        parent.close()
        child.close()
        if started and process.is_alive():
            process.terminate()
            process.join()


def run_benchmark_cli(report) -> None:
    """Run the maintained benchmark, cold child, or isolated RSS protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rss-public")
    parser.add_argument("--check-134c", action="store_true")
    args = parser.parse_args()
    if args.rss_public:
        config = CapstoneConfig(**json.loads(args.rss_public))
        print(f"{_RSS_MARKER}{json.dumps(asdict(measure_batched_rss(config)))}")
        return
    result = report()
    print(json.dumps(result, indent=2))
    if args.check_134c and not result["dask_134c"]["all_gates_pass"]:
        parser.exit(1, "Contract 134C Dask performance gates failed.\n")


__all__ = [
    "BatchedColdResult",
    "BatchedRssResult",
    "benchmark_environment",
    "cold_child",
    "dataset_chunks",
    "measure_batched_rss",
    "measure_cold_public",
    "run_benchmark_cli",
]
