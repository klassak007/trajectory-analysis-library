from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from _numba_bench import cold_subprocess, time_once, warm_median

from tal.spatial.kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_NUMBA,
    ROTATION_INTERP_BACKEND_SCIPY,
    slerp_quat_backend,
)


@dataclass(frozen=True)
class SlerpCase:
    name: str
    title: str
    q0: np.ndarray
    q1: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class SlerpResult:
    name: str
    baseline: float
    warm: float


def _z_quats(angles: np.ndarray) -> np.ndarray:
    out = np.empty(angles.shape + (4,), dtype=np.float64)
    half = 0.5 * angles
    out[..., 0] = 0.0
    out[..., 1] = 0.0
    out[..., 2] = np.sin(half)
    out[..., 3] = np.cos(half)
    return out


def _case_data(name: str) -> SlerpCase:
    if name == "many-short":
        rows, queries = 1024, 8
        title = "many-short SLERP rows"
    elif name == "fewer-long":
        rows, queries = 32, 256
        title = "fewer-long SLERP rows"
    else:
        raise ValueError(f"unknown benchmark case {name!r}")
    base = np.linspace(0.0, np.pi, rows, dtype=np.float64)[:, None]
    offsets = np.linspace(0.05, 0.95, queries, dtype=np.float64)[None, :]
    q0 = _z_quats(base + 0.05 * offsets)
    q1 = _z_quats(base + 1.25 + 0.1 * offsets)
    alpha = np.broadcast_to(np.linspace(0.0, 1.0, queries, dtype=np.float64), (rows, queries)).copy()
    valid = np.ones((rows, queries), dtype=bool)
    return SlerpCase(name, title, q0, q1, alpha, valid)


def _case_names() -> tuple[str, ...]:
    return ("many-short", "fewer-long")


def _scipy_slerp(case: SlerpCase) -> None:
    slerp_quat_backend(case.q0, case.q1, case.alpha, case.valid, backend=ROTATION_INTERP_BACKEND_SCIPY)


def _numba_slerp(case: SlerpCase) -> None:
    slerp_quat_backend(case.q0, case.q1, case.alpha, case.valid, backend=ROTATION_INTERP_BACKEND_NUMBA)


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-spatial-slerp-numba-cache-")


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_numba_slerp, _case_data(case_name)):.9f}")


def _report_case(case_name: str) -> SlerpResult:
    case = _case_data(case_name)
    _numba_slerp(case)
    baseline = warm_median(_scipy_slerp, case, repeats=3)
    warm = warm_median(_numba_slerp, case, repeats=7)
    print(f"\n{case.title}")
    print(f"  shape: rows={case.q0.shape[0]} query={case.q0.shape[1]}")
    print(f"  slerp scipy warm median:              {baseline:.6f}s")
    print(f"  slerp numba first-call fresh process: {_cold_subprocess(case_name):.6f}s")
    print(f"  slerp numba warm median:              {warm:.6f}s")
    return SlerpResult(case.name, baseline, warm)


def _print_backend_summary(results: list[SlerpResult]) -> None:
    high_row_win = any(result.name == "many-short" and result.warm <= 0.8 * result.baseline for result in results)
    no_slowdown = all(result.warm <= 1.1 * result.baseline for result in results)
    print("\nSpatial SLERP backend summary")
    print("  eligible public routing: automatic Numba/SciPy selection")
    print(f"  eligible many-short >=20% warm win:     {high_row_win}")
    print(f"  eligible cases <=10% slowdowns:         {no_slowdown}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    results = [_report_case(name) for name in _case_names()]
    _print_backend_summary(results)


if __name__ == "__main__":
    main()
