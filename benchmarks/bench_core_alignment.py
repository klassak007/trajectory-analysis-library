"""Compare exact TAL arithmetic with equivalent direct xarray arithmetic.

This is a manual, non-CI benchmark. Inputs are allocated before timing and the
reported ratios are review evidence, not pass/fail thresholds.
"""

from __future__ import annotations

import argparse
import gc
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from tal.core import AnalysisObject


@dataclass(frozen=True)
class _BenchmarkCase:
    left: AnalysisObject
    right: AnalysisObject
    direct_left: xr.DataArray
    direct_right: xr.DataArray


def _case(size: int) -> _BenchmarkCase:
    left_values = np.linspace(0.0, 1.0, size, dtype=np.float64)
    right_values = np.linspace(1.0, 2.0, size, dtype=np.float64)
    coords = {"sample": np.arange(size, dtype=np.int64)}
    direct_left = xr.DataArray(left_values, dims="sample", coords=coords, name="value")
    direct_right = xr.DataArray(right_values, dims="sample", coords=coords, name="value")
    left = AnalysisObject.from_data(
        direct_left.to_dataset(),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    right = AnalysisObject.from_data(
        direct_right.to_dataset(),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    return _BenchmarkCase(left, right, direct_left, direct_right)


def _time(operation: Callable[[], object]) -> float:
    gc.collect()
    started = perf_counter()
    result = operation()
    elapsed = perf_counter() - started
    if result is None:
        raise RuntimeError("benchmark operation returned no result")
    return elapsed


def _measure(case: _BenchmarkCase, *, warmups: int, repeats: int) -> tuple[float, float]:
    direct = lambda: case.direct_left + case.direct_right
    tal = lambda: case.left + case.right
    for _ in range(warmups):
        direct()
        tal()
    direct_samples: list[float] = []
    tal_samples: list[float] = []
    for index in range(repeats):
        order = ((direct, direct_samples), (tal, tal_samples))
        if index % 2:
            order = tuple(reversed(order))
        for operation, samples in order:
            samples.append(_time(operation))
    return median(direct_samples), median(tal_samples)


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
    print(f"warmups: {args.warmups}; repeats: {args.repeats}")
    for size in args.sizes:
        direct, tal = _measure(_case(size), warmups=args.warmups, repeats=args.repeats)
        print(
            f"samples={size:,}: direct xarray={direct:.6f}s; "
            f"TAL exact={tal:.6f}s; ratio={tal / direct:.3f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
