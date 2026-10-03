from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from _numba_bench import break_even_calls, cold_subprocess, time_once, warm_median
from tal.spatial.kernels.kinematics_smoothing_backends import (
    KINEMATICS_SMOOTHING_BACKEND_NUMBA,
    KINEMATICS_SMOOTHING_BACKEND_NUMPY,
    gaussian_smoothing_block_backend,
    moving_average_smoothing_block_backend,
)


@dataclass(frozen=True)
class StencilCase:
    name: str
    method: str
    title: str
    values: np.ndarray
    param: np.ndarray
    valid: np.ndarray
    window: int
    sigma: float


@dataclass(frozen=True)
class StencilResult:
    method: str
    eligible: bool
    baseline: float
    warm: float


def _shape_case(shape_name: str) -> tuple[str, int, int, int]:
    if shape_name == "many-short":
        return "many-short smoothing rows", 4096, 32, 3
    if shape_name == "fewer-long":
        return "fewer-long smoothing rows", 128, 512, 3
    if shape_name == "high-core":
        return "high-core smoothing rows", 512, 64, 32
    raise ValueError(f"unknown smoothing shape case {shape_name!r}")


def _case_names() -> tuple[str, ...]:
    return (
        "moving-many-short-small",
        "moving-fewer-long-medium",
        "moving-high-core-large",
        "gaussian-many-short-small",
        "gaussian-fewer-long-medium",
        "gaussian-high-core-large",
    )


def _case_parts(name: str) -> tuple[str, str, int]:
    parts = name.split("-")
    method = parts[0]
    shape_name = "-".join(parts[1:3])
    window_name = parts[3]
    windows = {"small": 3, "medium": 7, "large": 15}
    return method, shape_name, windows[window_name]


def _case_data(name: str) -> StencilCase:
    method, shape_name, window = _case_parts(name)
    title, rows, seq, core = _shape_case(shape_name)
    base = np.linspace(0.0, 10.0, seq, dtype=np.float64)
    offsets = np.arange(rows, dtype=np.float64)[:, None] * 1.0e-6
    param = np.broadcast_to(base, (rows, seq)).copy() + offsets
    valid = np.ones((rows, seq), dtype=bool)
    comps = np.arange(1, core + 1, dtype=np.float64)
    values = np.sin(param[..., None] / comps) + 0.1 * comps
    return StencilCase(name, method, f"{method} {title} window={window}", values, param, valid, window, 1.0)


def _run_backend(case: StencilCase, *, backend: str) -> None:
    if case.method == "moving":
        moving_average_smoothing_block_backend(case.values, case.param, case.valid, window=case.window, backend=backend)
        return
    gaussian_smoothing_block_backend(
        case.values,
        case.param,
        case.valid,
        window=case.window,
        sigma=case.sigma,
        backend=backend,
    )


def _baseline(case: StencilCase) -> None:
    _run_backend(case, backend=KINEMATICS_SMOOTHING_BACKEND_NUMPY)


def _numba(case: StencilCase) -> None:
    _run_backend(case, backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA)


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-spatial-stencil-numba-cache-")


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_numba, _case_data(case_name)):.9f}")


def _report_case(case_name: str) -> StencilResult:
    case = _case_data(case_name)
    _numba(case)
    baseline = warm_median(_baseline, case, repeats=5)
    cold = _cold_subprocess(case_name)
    warm = warm_median(_numba, case, repeats=7)
    print(f"\n{case.title}")
    print(f"  shape: rows={case.values.shape[0]} seq={case.values.shape[1]} core={case.values.shape[2]}")
    print(f"  stencil baseline numpy warm median: {baseline:.6f}s")
    print(f"  stencil numba first-call fresh process: {cold:.6f}s")
    print(f"  stencil numba warm median:              {warm:.6f}s")
    print(f"  stencil warm speedup:                   {baseline / warm:.3f}x")
    print(f"  stencil break-even warm calls:          {break_even_calls(baseline, cold, warm):.2f}")
    return StencilResult(case.method, "many-short" in case.name, baseline, warm)


def _print_spatial_f2c_summary(method: str, results: list[StencilResult]) -> None:
    method_results = [result for result in results if result.method == method]
    eligible = [result for result in method_results if result.eligible]
    fast_enough = any(result.warm <= 0.8 * result.baseline for result in eligible)
    not_too_slow = all(result.warm <= 1.1 * result.baseline for result in eligible)
    target = "kinematics_moving_average" if method == "moving" else "kinematics_gaussian"
    print(f"\nSpatial F2C decision input: {target}")
    print("  public routing status: baseline smoothing kernels")
    print(f"  eligible many-short >=20% warm win:     {fast_enough}")
    print(f"  eligible stencil <=10% slowdowns:       {not_too_slow}")
    print(f"  next Spatial F2C-B action:              evaluate public {method} smoothing default migration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    results = [_report_case(name) for name in _case_names()]
    _print_spatial_f2c_summary("moving", results)
    _print_spatial_f2c_summary("gaussian", results)


if __name__ == "__main__":
    main()
