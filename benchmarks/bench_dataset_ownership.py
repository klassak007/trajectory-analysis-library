"""Compare xarray and TAL Dataset ownership and typed-promotion costs.

This is a manual, non-CI benchmark. Inputs are allocated before timing and the
reported timings and traced peak allocations are evidence, not thresholds.
"""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from tal.core import AnalysisObject
from tal.linalg import Array


@dataclass(frozen=True)
class _BenchmarkCase:
    source: xr.Dataset
    ao: AnalysisObject


@dataclass(frozen=True)
class _Measurement:
    seconds: float
    peak_bytes: int


def _case(size: int) -> _BenchmarkCase:
    source = xr.Dataset(
        {"value": ("sample", np.linspace(0.0, 1.0, size, dtype=np.float64))},
        coords={"sample": np.arange(size, dtype=np.int64)},
    )
    ao = AnalysisObject.from_data(
        source,
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    return _BenchmarkCase(source=source, ao=ao)


def _sample(operation: Callable[[], object]) -> _Measurement:
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    result = operation()
    seconds = perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if result is None:
        raise RuntimeError("benchmark operation returned no result")
    return _Measurement(seconds=seconds, peak_bytes=peak_bytes)


def _operations(case: _BenchmarkCase) -> tuple[tuple[str, Callable[[], object]], ...]:
    return (
        ("xarray-shallow", lambda: case.source.copy(deep=False)),
        ("tal-shallow", lambda: case.ao.as_dataset(copy="shallow")),
        ("tal-deep", lambda: case.ao.as_dataset(copy="deep")),
        ("array-promotion", lambda: Array(case.ao)),
    )


def _measure(
    case: _BenchmarkCase,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, _Measurement]:
    operations = _operations(case)
    for _ in range(warmups):
        for _, operation in operations:
            operation()
    samples: dict[str, list[_Measurement]] = {name: [] for name, _ in operations}
    for index in range(repeats):
        order = operations if index % 2 == 0 else tuple(reversed(operations))
        for name, operation in order:
            samples[name].append(_sample(operation))
    return {
        name: _Measurement(
            seconds=median(sample.seconds for sample in values),
            peak_bytes=int(median(sample.peak_bytes for sample in values)),
        )
        for name, values in samples.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(1_000, 100_000, 1_000_000))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"numpy: {np.__version__}; pandas: {pd.__version__}; xarray: {xr.__version__}; tal: {version('tal')}")
    print(f"warmups: {args.warmups}; repeats: {args.repeats}; memory: tracemalloc peak")
    for size in args.sizes:
        measured = _measure(_case(size), warmups=args.warmups, repeats=args.repeats)
        reference = measured["xarray-shallow"]
        print(f"samples={size:,}")
        for name, result in measured.items():
            time_ratio = result.seconds / reference.seconds
            memory_ratio = result.peak_bytes / reference.peak_bytes
            print(
                f"  {name}: {result.seconds:.6f}s; "
                f"peak={result.peak_bytes:,} bytes; "
                f"xarray-time={time_ratio:.2f}x; "
                f"xarray-peak={memory_ratio:.2f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
