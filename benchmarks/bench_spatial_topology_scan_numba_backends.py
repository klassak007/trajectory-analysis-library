from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from _numba_bench import break_even_calls, cold_subprocess, time_once, warm_median
from tal.spatial.kernels.topology_scan_backends import (
    SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
    SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY,
    chain_pose_compose_block_backend,
)


@dataclass(frozen=True)
class TopologyCase:
    name: str
    title: str
    translation: np.ndarray
    quat: np.ndarray
    valid: np.ndarray
    direction: np.ndarray
    topology_len: int


@dataclass(frozen=True)
class TopologyResult:
    eligible: bool
    baseline: float
    warm: float


def _case_shape(name: str) -> tuple[str, tuple[int, ...], int]:
    if name == "many-short":
        return "many-short topology chains", (4096,), 8
    if name == "fewer-long":
        return "fewer-long topology chains", (128,), 512
    if name == "time-as-outer-by-chain":
        return "time-as-outer-by-chain topology chains", (128, 8), 32
    raise ValueError(f"unknown benchmark case {name!r}")


def _case_names() -> tuple[str, ...]:
    return ("many-short", "fewer-long", "time-as-outer-by-chain")


def _z_quat(degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(degrees)
    quat = np.zeros(degrees.shape + (4,), dtype=np.float64)
    quat[..., 2] = np.sin(0.5 * radians)
    quat[..., 3] = np.cos(0.5 * radians)
    return quat


def _case_data(name: str) -> TopologyCase:
    title, outer_shape, topology_len = _case_shape(name)
    outer = np.indices(outer_shape, dtype=np.float64).sum(axis=0)
    chain = np.arange(topology_len, dtype=np.float64)
    degrees = outer[..., None] * 0.01 + chain * 0.5
    translation = np.zeros(outer_shape + (topology_len, 3), dtype=np.float64)
    translation[..., 0] = 0.01 + 0.001 * chain
    translation[..., 1] = np.sin(0.1 * chain)
    translation[..., 2] = np.cos(0.05 * chain)
    quat = _z_quat(degrees)
    valid = np.ones(outer_shape + (topology_len,), dtype=bool)
    direction = np.ones(outer_shape + (topology_len,), dtype=np.int64)
    direction[..., 1::4] = -1
    return TopologyCase(name, title, translation, quat, valid, direction, topology_len)


def _baseline_topology(case: TopologyCase) -> None:
    chain_pose_compose_block_backend(
        case.translation,
        case.quat,
        case.valid,
        case.direction,
        backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY,
    )


def _numba_topology(case: TopologyCase) -> None:
    chain_pose_compose_block_backend(
        case.translation,
        case.quat,
        case.valid,
        case.direction,
        backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
    )


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-spatial-topology-numba-cache-")


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_numba_topology, _case_data(case_name)):.9f}")


def _report_case(case_name: str) -> TopologyResult:
    case = _case_data(case_name)
    _numba_topology(case)
    baseline = warm_median(_baseline_topology, case, repeats=5)
    cold = _cold_subprocess(case_name)
    warm = warm_median(_numba_topology, case, repeats=7)
    outer_rows = int(np.prod(case.valid.shape[:-1], dtype=np.int64))
    print(f"\n{case.title}")
    print(f"  shape: outer_rows={outer_rows} topology_len={case.topology_len}")
    print(f"  topology numpy warm median:             {baseline:.6f}s")
    print(f"  topology numba first-call fresh process:{cold:.6f}s")
    print(f"  topology numba warm median:             {warm:.6f}s")
    print(f"  topology warm speedup:                  {baseline / warm:.3f}x")
    print(f"  topology break-even warm calls:         {break_even_calls(baseline, cold, warm):.2f}")
    return TopologyResult(case.name == "many-short", baseline, warm)


def _print_spatial_f2c_summary(results: list[TopologyResult]) -> None:
    eligible = [result for result in results if result.eligible]
    fast_enough = any(result.warm <= 0.8 * result.baseline for result in eligible)
    not_too_slow = all(result.warm <= 1.1 * result.baseline for result in eligible)
    print("\nSpatial F2C decision input: topology_chain_pose")
    print("  public routing status: baseline path solving")
    print(f"  eligible many-short >=20% warm win:     {fast_enough}")
    print(f"  eligible topology <=10% slowdowns:      {not_too_slow}")
    print("  next Spatial F2C-B action:              keep sidecar-only until public topology scan route exists")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    results = [_report_case(name) for name in _case_names()]
    _print_spatial_f2c_summary(results)


if __name__ == "__main__":
    main()
