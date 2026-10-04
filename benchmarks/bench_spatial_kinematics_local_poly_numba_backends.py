from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from _numba_bench import time_once, warm_median
from tal.spatial.kernels.kinematics_local_poly_backends import (
    KINEMATICS_LOCAL_POLY_BACKEND_NUMPY,
    local_poly_derivative_block_backend,
    local_poly_smoothing_block_backend,
)

_NO_RETENTION_DECISION = (
    "local-poly Numba prototype passed nopython/parity but failed the retention gate: "
    "no eligible many-row smoothing or derivative case reached a 20% warm win, and an "
    "eligible smoothing case exceeded the 10% slowdown limit."
)


@dataclass(frozen=True)
class LocalPolyCase:
    name: str
    kind: str
    title: str
    values: np.ndarray
    param: np.ndarray
    valid: np.ndarray
    window: int
    poly_order: int
    eligible: bool


def _case_names() -> tuple[str, ...]:
    return (
        "smooth-many-row-small-window",
        "smooth-many-row-medium-window",
        "smooth-fewer-long",
        "smooth-high-core",
        "derivative-many-row-small-window",
        "derivative-many-row-medium-window",
        "derivative-fewer-long",
        "derivative-high-core",
    )


def _case_parts(name: str) -> tuple[str, str, int, int, int, int, int, bool]:
    if name.endswith("many-row-small-window"):
        return name.split("-")[0], "many-row small-window", 4096, 32, 3, 5, 2, True
    if name.endswith("many-row-medium-window"):
        return name.split("-")[0], "many-row medium-window", 4096, 64, 3, 9, 3, True
    if name.endswith("fewer-long"):
        return name.split("-")[0], "fewer-long evidence", 128, 512, 3, 9, 3, False
    if name.endswith("high-core"):
        return name.split("-")[0], "high-core evidence", 512, 64, 32, 7, 2, False
    raise ValueError(f"unknown local-poly case {name!r}")


def _case_data(name: str) -> LocalPolyCase:
    kind, title, rows, seq, core, window, poly_order, eligible = _case_parts(name)
    base = np.linspace(0.0, 10.0, seq, dtype=np.float64)
    offsets = np.arange(rows, dtype=np.float64)[:, None] * 1.0e-6
    param = np.broadcast_to(base, (rows, seq)).copy() + offsets
    valid = np.ones((rows, seq), dtype=bool)
    comps = np.arange(1, core + 1, dtype=np.float64)
    values = np.sin(param[..., None] / comps) + 0.1 * np.cos(param[..., None] * comps)
    return LocalPolyCase(name, kind, f"{kind} local-poly {title}", values, param, valid, window, poly_order, eligible)


def _baseline(case: LocalPolyCase) -> np.ndarray:
    kwargs = {"window": case.window, "poly_order": case.poly_order, "backend": KINEMATICS_LOCAL_POLY_BACKEND_NUMPY}
    if case.kind == "smooth":
        return local_poly_smoothing_block_backend(case.values, case.param, case.valid, **kwargs)
    return local_poly_derivative_block_backend(case.values, case.param, case.valid, **kwargs)


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_baseline, _case_data(case_name)):.9f}")


def _report_case(case_name: str) -> None:
    case = _case_data(case_name)
    baseline = warm_median(_baseline, case, repeats=3)
    print(f"\n{case.title}")
    print(f"  shape: rows={case.values.shape[0]} seq={case.values.shape[1]} core={case.values.shape[2]}")
    print(f"  window/poly_order: {case.window}/{case.poly_order}")
    print(f"  local-poly baseline numpy warm median: {baseline:.6f}s")
    print("  local-poly numba backend:                not retained")
    print(f"  eligible gate case:                       {case.eligible}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    for name in _case_names():
        _report_case(name)
    print("\nlocal-poly retention gate")
    print("  nopython/parity:                          pass during prototype")
    print("  smoothing many-row >=20% warm win:        False")
    print("  derivative many-row >=20% warm win:       False")
    print("  eligible local-poly <=10% slowdowns:      False")
    print("  retain numba backend:                     False")
    print(f"  decision:                                 {_NO_RETENTION_DECISION}")


if __name__ == "__main__":
    main()
