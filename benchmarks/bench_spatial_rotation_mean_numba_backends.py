from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from _numba_bench import break_even_calls, cold_subprocess, time_once, warm_median
from tal.spatial.kernels.rotation_mean_backends import (
    ROTATION_MEAN_BACKEND_NUMBA,
    ROTATION_MEAN_BACKEND_NUMPY,
    quat_mean_block_backend,
)
from tal.spatial.kernels.rotation_mean_numba_backends import _compiled_quat_mean_block


@dataclass(frozen=True)
class MeanCase:
    name: str
    title: str
    rows: int
    reduce: int
    values: np.ndarray
    weights: np.ndarray


def _z_quat(degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(degrees)
    quat = np.zeros(degrees.shape + (4,), dtype=np.float64)
    quat[..., 2] = np.sin(0.5 * radians)
    quat[..., 3] = np.cos(0.5 * radians)
    return quat


def _case_names() -> tuple[str, ...]:
    return ("many-small", "weighted-many-small", "fewer-long", "high-reduction")


def _case_shape(name: str) -> tuple[str, int, int, bool]:
    if name == "many-small":
        return "many-row small-reduction rotation mean", 8192, 16, False
    if name == "weighted-many-small":
        return "weighted many-row small-reduction rotation mean", 8192, 16, True
    if name == "fewer-long":
        return "fewer-row long-reduction rotation mean", 128, 256, True
    if name == "high-reduction":
        return "high-reduction rotation mean", 64, 1024, True
    raise ValueError(f"unknown benchmark case {name!r}")


def _case_data(name: str) -> MeanCase:
    title, rows, reduce, weighted = _case_shape(name)
    row_idx = np.arange(rows, dtype=np.float64)[:, None]
    reduce_idx = np.arange(reduce, dtype=np.float64)[None, :]
    values = _z_quat(0.01 * row_idx + 0.1 * reduce_idx)
    if weighted:
        weights = 1.0 + 0.1 * np.sin(0.03 * row_idx + 0.2 * reduce_idx)
    else:
        weights = np.asarray(1.0, dtype=np.float64)
    return MeanCase(name, title, rows, reduce, values, weights)


def _baseline(case: MeanCase) -> np.ndarray:
    return quat_mean_block_backend(case.values, case.weights, backend=ROTATION_MEAN_BACKEND_NUMPY)


def _numba(case: MeanCase) -> np.ndarray:
    return quat_mean_block_backend(case.values, case.weights, backend=ROTATION_MEAN_BACKEND_NUMBA)


def _assert_quat_equivalent(actual: np.ndarray, expected: np.ndarray) -> None:
    dots = np.sum(actual * expected, axis=-1)
    aligned = np.where(dots[..., None] < 0.0, -actual, actual)
    np.testing.assert_allclose(aligned, expected, equal_nan=True, rtol=1e-10, atol=1e-10)


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-spatial-rotmean-numba-cache-")


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_numba, _case_data(case_name)):.9f}")


def _case_gate(baseline: float, warm: float, *, case: MeanCase) -> tuple[bool, bool]:
    eligible = case.rows >= 4096 and case.reduce <= 32
    fast_enough = eligible and warm <= baseline * 0.8
    not_too_slow = (not eligible) or warm <= baseline * 1.1
    return fast_enough, not_too_slow


def _report_case(case_name: str) -> tuple[bool, bool]:
    case = _case_data(case_name)
    expected = _baseline(case)
    actual = _numba(case)
    _assert_quat_equivalent(actual, expected)
    assert _compiled_quat_mean_block().nopython_signatures
    baseline = warm_median(_baseline, case, repeats=5)
    cold = _cold_subprocess(case_name)
    warm = warm_median(_numba, case, repeats=7)
    fast_enough, not_too_slow = _case_gate(baseline, warm, case=case)
    print(f"\n{case.title}")
    print(f"  shape: rows={case.rows} reduce={case.reduce}")
    print(f"  rotation-mean numpy warm median:          {baseline:.6f}s")
    print(f"  rotation-mean numba first-call fresh process:{cold:.6f}s")
    print(f"  rotation-mean numba warm median:          {warm:.6f}s")
    print(f"  rotation-mean warm speedup:               {baseline / warm:.3f}x")
    print(f"  rotation-mean break-even warm calls:      {break_even_calls(baseline, cold, warm):.2f}")
    print(f"  eligible many-row gate:                   {case.rows >= 4096 and case.reduce <= 32}")
    return fast_enough, not_too_slow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    gate_hits = []
    slowdown_ok = []
    for name in _case_names():
        fast_enough, not_too_slow = _report_case(name)
        gate_hits.append(fast_enough)
        slowdown_ok.append(not_too_slow)
    print("\nrotation mean retention gate")
    print(f"  nopython/parity:                          pass")
    print(f"  many-row >=20% warm win:                  {any(gate_hits)}")
    print(f"  eligible many-row <=10% slowdowns:        {all(slowdown_ok)}")
    print(f"  retain numba backend:                     {any(gate_hits) and all(slowdown_ok)}")
    print("\nSpatial F2C decision input: rotation_mean")
    print("  public routing status: baseline reducer kernel")
    print(f"  eligible many-row >=20% warm win:         {any(gate_hits)}")
    print(f"  eligible rotation-mean <=10% slowdowns:   {all(slowdown_ok)}")
    print("  next Spatial F2C-B action:                evaluate public rotation mean default migration")


if __name__ == "__main__":
    main()
