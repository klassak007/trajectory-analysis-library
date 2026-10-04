from __future__ import annotations

import argparse
from dataclasses import dataclass
import warnings

import numpy as np

from _numba_bench import break_even_calls, cold_subprocess, time_once, warm_median
from tal.spatial.kernels.kinematics_temporal_backends import (
    KINEMATICS_TEMPORAL_BACKEND_NUMBA,
    KINEMATICS_TEMPORAL_BACKEND_NUMPY,
    cumulative_simpson_block_backend,
    cumulative_trapezoid_block_backend,
)
from tal.spatial.kernels.kinematics_temporal_numba_backends import _compiled_simpson_block

_LABEL_SENTINELS = (
    "trapezoid numba first-call fresh process",
    "trapezoid numba warm median",
    "simpson numba first-call fresh process",
    "simpson numba warm median",
)


@dataclass(frozen=True)
class TemporalCase:
    name: str
    title: str
    kind: str
    values: np.ndarray
    param: np.ndarray
    valid: np.ndarray
    initial_value: float


@dataclass(frozen=True)
class SimpsonResult:
    name: str
    eligible: bool
    baseline: float
    warm: float


@dataclass(frozen=True)
class TrapezoidResult:
    name: str
    eligible: bool
    baseline: float
    warm: float


def _trapezoid_case_names() -> tuple[str, ...]:
    return ("many-short", "fewer-long", "high-core")


def _simpson_case_names() -> tuple[str, ...]:
    return ("simpson-many-short", "simpson-fewer-long", "simpson-high-core")


def _case_names() -> tuple[str, ...]:
    return _trapezoid_case_names() + _simpson_case_names()


def _case_shape(name: str) -> tuple[str, str, int, int, int]:
    if name == "many-short":
        return "many-short cumulative trapezoid rows", "trapezoid", 4096, 16, 3
    if name == "fewer-long":
        return "fewer-long cumulative trapezoid rows", "trapezoid", 128, 1024, 3
    if name == "high-core":
        return "high-core cumulative trapezoid rows", "trapezoid", 512, 32, 32
    if name == "simpson-many-short":
        return "many-short cumulative Simpson rows", "simpson", 4096, 17, 3
    if name == "simpson-fewer-long":
        return "fewer-long cumulative Simpson rows", "simpson", 128, 1025, 3
    if name == "simpson-high-core":
        return "high-core cumulative Simpson rows", "simpson", 512, 33, 32
    raise ValueError(f"unknown benchmark case {name!r}")


def _case_data(name: str) -> TemporalCase:
    title, kind, rows, seq, core = _case_shape(name)
    base = np.linspace(0.0, 10.0, seq, dtype=np.float64)
    offsets = np.arange(rows, dtype=np.float64)[:, None] * 1.0e-6
    param = np.broadcast_to(base, (rows, seq)).copy() + offsets
    valid = np.ones((rows, seq), dtype=bool)
    comps = np.arange(1, core + 1, dtype=np.float64)
    values = np.sin(param[..., None] / comps) + 0.1 * comps
    return TemporalCase(name, title, kind, values, param, valid, 0.25)


def _baseline(case: TemporalCase) -> np.ndarray:
    if case.kind == "simpson":
        return cumulative_simpson_block_backend(
            case.values,
            case.param,
            case.valid,
            initial_value=case.initial_value,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMPY,
        )
    return cumulative_trapezoid_block_backend(
        case.values,
        case.param,
        case.valid,
        initial_value=case.initial_value,
        backend=KINEMATICS_TEMPORAL_BACKEND_NUMPY,
    )


def _numba(case: TemporalCase) -> np.ndarray:
    if case.kind == "simpson":
        return cumulative_simpson_block_backend(
            case.values,
            case.param,
            case.valid,
            initial_value=case.initial_value,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
        )
    return cumulative_trapezoid_block_backend(
        case.values,
        case.param,
        case.valid,
        initial_value=case.initial_value,
        backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
    )


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-spatial-kinematics-numba-cache-")


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_numba, _case_data(case_name)):.9f}")


def _assert_parity(case: TemporalCase) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = _baseline(case)
        actual = _numba(case)
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-10, atol=1e-10)
    if case.kind == "simpson":
        assert _compiled_simpson_block().nopython_signatures


def _report_case(case_name: str) -> SimpsonResult | TrapezoidResult:
    case = _case_data(case_name)
    _assert_parity(case)
    baseline = warm_median(_baseline, case, repeats=5)
    cold = _cold_subprocess(case_name)
    warm = warm_median(_numba, case, repeats=7)
    speedup = baseline / warm
    print(f"\n{case.title}")
    print(f"  shape: rows={case.values.shape[0]} seq={case.values.shape[1]} core={case.values.shape[2]}")
    print(f"  {case.kind} baseline numpy warm median: {baseline:.6f}s")
    print(f"  {case.kind} numba first-call fresh process: {cold:.6f}s")
    print(f"  {case.kind} numba warm median:              {warm:.6f}s")
    print(f"  {case.kind} warm speedup:                   {speedup:.3f}x")
    print(f"  {case.kind} break-even warm calls:          {break_even_calls(baseline, cold, warm):.2f}")
    if case.kind != "simpson":
        return TrapezoidResult(case.name, case.name == "many-short", baseline, warm)
    return SimpsonResult(case.name, case.name != "simpson-high-core", baseline, warm)


def _print_trapezoid_decision(results: list[TrapezoidResult]) -> None:
    eligible = [result for result in results if result.eligible]
    fast_enough = any(result.warm <= 0.8 * result.baseline for result in eligible)
    not_too_slow = all(result.warm <= 1.1 * result.baseline for result in eligible)
    print("\nSpatial F2C decision input: kinematics_trapezoid")
    print("  public routing status: baseline NumPy/SciPy temporal integral")
    print(f"  eligible many-short >=20% warm win:     {fast_enough}")
    print(f"  eligible trapezoid <=10% slowdowns:     {not_too_slow}")
    print("  next Spatial F2C-B action:              evaluate public trapezoid default migration")


def _print_simpson_gate(results: list[SimpsonResult]) -> None:
    eligible = [result for result in results if result.eligible]
    fast_enough = all(result.warm <= 0.8 * result.baseline for result in eligible)
    not_too_slow = all(result.warm <= 1.1 * result.baseline for result in eligible)
    retained = fast_enough and not_too_slow
    print("\nsimpson retention gate")
    print("  nopython/parity:                          pass")
    print(f"  many-short/fewer-long >=20% warm win:     {fast_enough}")
    print(f"  eligible simpson <=10% slowdowns:         {not_too_slow}")
    print(f"  retain numba backend:                     {retained}")
    print("\nSpatial F2C decision input: kinematics_simpson")
    print("  public routing status: baseline SciPy cumulative Simpson")
    print(f"  eligible simpson >=20% warm win:          {fast_enough}")
    print(f"  eligible simpson <=10% slowdowns:         {not_too_slow}")
    print("  next Spatial F2C-B action:                evaluate public Simpson default migration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    trapezoid_results = []
    simpson_results = []
    for name in _case_names():
        result = _report_case(name)
        if isinstance(result, SimpsonResult):
            simpson_results.append(result)
        else:
            trapezoid_results.append(result)
    _print_trapezoid_decision(trapezoid_results)
    _print_simpson_gate(simpson_results)


if __name__ == "__main__":
    main()
