from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from _numba_bench import break_even_calls, cold_subprocess, time_once, warm_median
from tal.linalg.ops.solve_backends import (
    LSTSQ_BACKEND_NUMBA,
    LSTSQ_BACKEND_NUMPY_BLOCK,
    _select_lstsq_backend,
    lstsq_block_backend,
    _lstsq_numpy_row,
)


@dataclass(frozen=True)
class LstsqCase:
    name: str
    title: str
    a: np.ndarray
    b: np.ndarray
    rhs_is_vector: bool
    shape_class: str


@dataclass(frozen=True)
class LstsqResult:
    shape_class: str
    selected: str
    rows: int
    baseline: float
    numba_warm: float | None


def _matrix_rows(seed: int, rows: int, equations: int, solutions: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(rows, equations, solutions))
    a += 2.0 * np.broadcast_to(np.eye(equations, solutions), (rows, equations, solutions))
    return a


def _small_case(rows: int, equations: int, solutions: int, *, matrix_rhs: bool) -> LstsqCase:
    rhs_cols = 3 if matrix_rhs else 1
    suffix = "matrix" if matrix_rhs else "vector"
    seed = rows + equations * 101 + solutions * 1009 + rhs_cols * 10007
    a = _matrix_rows(seed, rows, equations, solutions)
    if matrix_rhs:
        b = np.random.default_rng(seed + 1).normal(size=(rows, equations, rhs_cols))
    else:
        b = np.random.default_rng(seed + 1).normal(size=(rows, equations))
    name = f"small-{equations}x{solutions}-{suffix}-{rows}"
    title = f"small-core {equations}x{solutions} {suffix} RHS rows={rows}"
    return LstsqCase(name, title, a, b, not matrix_rhs, "small")


def _case_data(name: str) -> LstsqCase:
    if name == "fewer-large-vector":
        a = _matrix_rows(31, 128, 128, 32)
        b = np.random.default_rng(32).normal(size=(128, 128))
        return LstsqCase(name, "fewer-large vector RHS", a, b, True, "large")
    if name == "fewer-large-matrix":
        a = _matrix_rows(41, 128, 128, 32)
        b = np.random.default_rng(42).normal(size=(128, 128, 4))
        return LstsqCase(name, "fewer-large matrix RHS", a, b, False, "large")
    parts = name.split("-")
    if len(parts) == 4 and parts[0] == "small":
        equations, solutions = (int(value) for value in parts[1].split("x"))
        return _small_case(int(parts[3]), equations, solutions, matrix_rhs=parts[2] == "matrix")
    raise ValueError(f"unknown benchmark case {name!r}")


def _case_names() -> list[str]:
    names = []
    for equations, solutions in ((4, 2), (8, 3), (16, 4)):
        for rows in (32, 128, 512, 4096, 16384):
            for rhs in ("vector", "matrix"):
                names.append(f"small-{equations}x{solutions}-{rhs}-{rows}")
    return names + ["fewer-large-vector", "fewer-large-matrix"]


def _baseline_lstsq(case: LstsqCase) -> None:
    for row in range(case.a.shape[0]):
        _lstsq_numpy_row(
            case.a[row],
            case.b[row],
            rcond=None,
        )


def _numpy_block_lstsq(case: LstsqCase) -> None:
    lstsq_block_backend(
        case.a,
        case.b,
        rcond=None,
        rhs_is_vector=case.rhs_is_vector,
        backend=LSTSQ_BACKEND_NUMPY_BLOCK,
    )


def _numba_lstsq(case: LstsqCase) -> None:
    lstsq_block_backend(
        case.a,
        case.b,
        rcond=None,
        rhs_is_vector=case.rhs_is_vector,
        backend=LSTSQ_BACKEND_NUMBA,
    )


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-linalg-numba-cache-")


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_numba_lstsq, _case_data(case_name)):.9f}")


def _report_case(case_name: str) -> LstsqResult:
    case = _case_data(case_name)
    selected = _select_lstsq_backend(case.a, case.b, rhs_is_vector=case.rhs_is_vector)
    baseline = warm_median(_baseline_lstsq, case, repeats=5)
    numpy_block = warm_median(_numpy_block_lstsq, case, repeats=5)
    print(f"\n{case.title}")
    print(f"  selector: {selected}")
    print(f"  shape: rows={case.a.shape[0]} a_core={case.a.shape[1:]} b_core={case.b.shape[1:]}")
    print(f"  lstsq private row baseline warm median: {baseline:.6f}s")
    print(f"  lstsq numpy_block fallback warm median: {numpy_block:.6f}s")
    if selected != LSTSQ_BACKEND_NUMBA:
        return LstsqResult(case.shape_class, selected, case.a.shape[0], baseline, None)
    _numba_lstsq(case)
    cold = _cold_subprocess(case_name)
    warm = warm_median(_numba_lstsq, case, repeats=5)
    ratio = warm / baseline
    break_even = break_even_calls(baseline, cold, warm)
    print(f"  lstsq numba first-call fresh process: {cold:.6f}s")
    print(f"  lstsq numba warm median:              {warm:.6f}s ({ratio:.3f}x baseline)")
    print(f"  estimated warm calls to amortize JIT: {break_even:.1f}")
    return LstsqResult(case.shape_class, selected, case.a.shape[0], baseline, warm)


def _print_gate(results: list[LstsqResult]) -> None:
    selected = [result for result in results if result.selected == LSTSQ_BACKEND_NUMBA]
    small_win = any(
        result.shape_class == "small"
        and result.rows >= 4096
        and result.numba_warm is not None
        and result.baseline >= result.numba_warm / 0.8
        for result in selected
    )
    no_selected_slow = all(
        result.numba_warm is not None and result.numba_warm <= 1.1 * result.baseline
        for result in selected
    )
    large_fallback = all(
        result.selected == LSTSQ_BACKEND_NUMPY_BLOCK
        for result in results
        if result.shape_class == "large"
    )
    status = "PASS" if small_win and no_selected_slow and large_fallback else "FAIL"
    print(f"\nbenchmark gate: {status}")
    print("  requires >=20% warm speedup in at least one selected small-core high-row case")
    print("  requires no selector-chosen numba case more than 10% slower than baseline")
    print("  requires fewer-large cases to remain numpy_block-selected")
    print("\nlinalg lstsq F2C decision input")
    print(f"  selected small-core high-row >=20% warm win: {small_win}")
    print(f"  selector-chosen numba <=10% slowdowns:      {no_selected_slow}")
    print(f"  fewer-large cases numpy_block-selected:     {large_fallback}")
    print(f"  lstsq promotion gate:                       {status}")
    print("\nlinalg lstsq F2C-B3 normal-path migration")
    print("  selected normal backend:                    shape-aware numba")
    print("  no-numba fallback backend:                  numpy_block")
    print("  fewer-large selected backend:               numpy_block")
    print("  public normal path:                         blockwise vectorize=False")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    _print_gate([_report_case(name) for name in _case_names()])


if __name__ == "__main__":
    main()
