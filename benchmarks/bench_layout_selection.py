"""Manual retained-payload ingress benchmark; no CI wall-clock threshold."""

from __future__ import annotations

import gc
import json
import statistics
import time
import tracemalloc
from collections.abc import Callable

import dask.array as da
import numpy as np
import xarray as xr

from tal.core import AnalysisLayoutSpec
from tal.core.layout_ingress import _preflight_source

VARIABLE_COUNT = 64
ROW_COUNT = 16_384
CHUNK_ROWS = 1_024
SELECTED = ("v00", "v01")


def fixture(*, lazy: bool) -> xr.Dataset:
    """Build the frozen wide float64 fixture outside measured regions."""
    variables: dict[str, tuple[str, object]] = {}
    for index in range(VARIABLE_COUNT):
        payload = np.full(ROW_COUNT, float(index), dtype=np.float64)
        values = da.from_array(payload, chunks=CHUNK_ROWS) if lazy else payload
        variables[f"v{index:02d}"] = ("sample", values)
    return xr.Dataset(variables, coords={"sample": np.arange(ROW_COUNT)})


def _measure_once(operation: Callable[[], object]) -> tuple[float, int]:
    gc.collect()
    tracemalloc.start()
    began = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - began
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return elapsed, peak


def _samples(routes: dict[str, Callable[[], object]]) -> dict[str, object]:
    for _ in range(2):
        for operation in routes.values():
            result = operation()
            del result
    samples: dict[str, list[tuple[float, int]]] = {name: [] for name in routes}
    for repetition in range(7):
        order = tuple(routes) if repetition % 2 == 0 else tuple(reversed(routes))
        for name in order:
            samples[name].append(_measure_once(routes[name]))
    return {
        name: {
            "seconds": [sample[0] for sample in rows],
            "peak_bytes": [sample[1] for sample in rows],
            "median_seconds": statistics.median(sample[0] for sample in rows),
            "median_peak_bytes": statistics.median(sample[1] for sample in rows),
        }
        for name, rows in samples.items()
    }


def _task_count(ds: xr.Dataset) -> int:
    keys: set[object] = set()
    for variable in ds.data_vars.values():
        graph = getattr(variable.data, "dask", None)
        if graph is not None:
            keys.update(graph.keys())
    return len(keys)


def main() -> None:
    """Report eager timing/allocation and lazy graph sizes as JSON."""
    eager = fixture(lazy=False)
    lazy = fixture(lazy=True)
    layout = AnalysisLayoutSpec(sequence_dim="sample")
    schema_bearing = layout.wrap(eager).as_dataset(copy="none")
    eager_routes = {
        "selected_external": lambda: layout.wrap(eager, data_vars=SELECTED),
        "complete_then_select": lambda: layout.wrap(eager).select_vars(SELECTED),
        "source_preflight_only": lambda: _preflight_source(schema_bearing, owner="benchmark.layout_selection"),
    }
    lazy_selected = layout.wrap(lazy, data_vars=SELECTED).as_dataset(copy="none")
    lazy_complete = layout.wrap(lazy).as_dataset(copy="none")
    report = {
        "fixture": {"variables": VARIABLE_COUNT, "rows": ROW_COUNT, "dtype": "float64", "chunks": CHUNK_ROWS},
        "eager": _samples(eager_routes),
        "dask_tasks": {
            "source": _task_count(lazy),
            "complete": _task_count(lazy_complete),
            "selected": _task_count(lazy_selected),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
