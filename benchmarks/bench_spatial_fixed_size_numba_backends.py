from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from _numba_bench import break_even_calls, cold_subprocess, time_once, warm_median
from tal.spatial.kernels.fixed_size_backends import (
    SPATIAL_FIXED_BACKEND_NUMBA,
    SPATIAL_FIXED_BACKEND_SCIPY,
    matrix_to_quat_block_backend,
    pose_compose_translation_block_backend,
    pose_components_to_matrix_block_backend,
    pose_inverse_translation_block_backend,
    quat_compose_block_backend,
    quat_inverse_block_backend,
    quat_to_matrix_block_backend,
    rotate_vec3_block_backend,
)


@dataclass(frozen=True)
class FixedCase:
    name: str
    title: str
    kind: str
    left: np.ndarray
    right: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class FixedResult:
    eligible: bool
    baseline: float
    warm: float


def _z_quat(degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(degrees)
    quat = np.zeros(degrees.shape + (4,), dtype=np.float64)
    quat[..., 2] = np.sin(0.5 * radians)
    quat[..., 3] = np.cos(0.5 * radians)
    return quat


def _case_names() -> tuple[str, ...]:
    return ("rotation-many-row", "rotation-fewer-row", "pose-many-row", "pose-fewer-row")


def _case_shape(name: str) -> tuple[str, str, int]:
    if name == "rotation-many-row":
        return "many-row fixed-size rotation primitives", "rotation", 16384
    if name == "rotation-fewer-row":
        return "fewer-row fixed-size rotation primitives", "rotation", 512
    if name == "pose-many-row":
        return "many-row fixed-size pose primitives", "pose", 16384
    if name == "pose-fewer-row":
        return "fewer-row fixed-size pose primitives", "pose", 512
    raise ValueError(f"unknown benchmark case {name!r}")


def _case_data(name: str) -> FixedCase:
    title, kind, rows = _case_shape(name)
    idx = np.arange(rows, dtype=np.float64)
    left = _z_quat(0.01 * idx).reshape(rows, 4)
    right = 2.0 * _z_quat(0.02 * idx + 5.0).reshape(rows, 4)
    values = np.zeros((rows, 3), dtype=np.float64)
    values[:, 0] = np.sin(idx * 0.01)
    values[:, 1] = np.cos(idx * 0.01)
    values[:, 2] = idx * 1.0e-4
    return FixedCase(name, title, kind, left, right, values)


def _baseline_rotation(case: FixedCase) -> None:
    composed = quat_compose_block_backend(case.left, case.right, backend=SPATIAL_FIXED_BACKEND_SCIPY)
    inverse = quat_inverse_block_backend(case.left, backend=SPATIAL_FIXED_BACKEND_SCIPY)
    matrices = quat_to_matrix_block_backend(composed, backend=SPATIAL_FIXED_BACKEND_SCIPY)
    matrix_to_quat_block_backend(matrices, backend=SPATIAL_FIXED_BACKEND_SCIPY)
    rotate_vec3_block_backend(case.values, inverse, backend=SPATIAL_FIXED_BACKEND_SCIPY)


def _numba_rotation(case: FixedCase) -> None:
    composed = quat_compose_block_backend(case.left, case.right, backend=SPATIAL_FIXED_BACKEND_NUMBA)
    inverse = quat_inverse_block_backend(case.left, backend=SPATIAL_FIXED_BACKEND_NUMBA)
    matrices = quat_to_matrix_block_backend(composed, backend=SPATIAL_FIXED_BACKEND_NUMBA)
    matrix_to_quat_block_backend(matrices, backend=SPATIAL_FIXED_BACKEND_NUMBA)
    rotate_vec3_block_backend(case.values, inverse, backend=SPATIAL_FIXED_BACKEND_NUMBA)


def _baseline_pose(case: FixedCase) -> None:
    pose_compose_translation_block_backend(case.values, case.values + 0.1, case.right, backend=SPATIAL_FIXED_BACKEND_SCIPY)
    pose_inverse_translation_block_backend(case.values, case.right, backend=SPATIAL_FIXED_BACKEND_SCIPY)
    pose_components_to_matrix_block_backend(case.values, case.right, backend=SPATIAL_FIXED_BACKEND_SCIPY)


def _numba_pose(case: FixedCase) -> None:
    pose_compose_translation_block_backend(case.values, case.values + 0.1, case.right, backend=SPATIAL_FIXED_BACKEND_NUMBA)
    pose_inverse_translation_block_backend(case.values, case.right, backend=SPATIAL_FIXED_BACKEND_NUMBA)
    pose_components_to_matrix_block_backend(case.values, case.right, backend=SPATIAL_FIXED_BACKEND_NUMBA)


def _baseline(case: FixedCase) -> None:
    if case.kind == "rotation":
        _baseline_rotation(case)
        return
    _baseline_pose(case)


def _numba(case: FixedCase) -> None:
    if case.kind == "rotation":
        _numba_rotation(case)
        return
    _numba_pose(case)


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-spatial-fixed-numba-cache-")


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_numba, _case_data(case_name)):.9f}")


def _report_case(case_name: str) -> FixedResult:
    case = _case_data(case_name)
    _numba(case)
    baseline = warm_median(_baseline, case, repeats=5)
    cold = _cold_subprocess(case_name)
    warm = warm_median(_numba, case, repeats=7)
    print(f"\n{case.title}")
    print(f"  shape: rows={case.left.shape[0]} kind={case.kind}")
    print(f"  fixed-size scipy warm median:          {baseline:.6f}s")
    print(f"  fixed-size numba first-call fresh process:{cold:.6f}s")
    print(f"  fixed-size numba warm median:          {warm:.6f}s")
    print(f"  fixed-size warm speedup:               {baseline / warm:.3f}x")
    print(f"  fixed-size break-even warm calls:      {break_even_calls(baseline, cold, warm):.2f}")
    return FixedResult("many-row" in case.name, baseline, warm)


def _print_spatial_f2c_summary(results: list[FixedResult]) -> None:
    eligible = [result for result in results if result.eligible]
    fast_enough = any(result.warm <= 0.8 * result.baseline for result in eligible)
    not_too_slow = all(result.warm <= 1.1 * result.baseline for result in eligible)
    print("\nSpatial F2C decision input: fixed_size_spatial_math")
    print("  subkernels: quat compose, quat inverse, quat-to-matrix, matrix-to-quat, rotate-vec3, pose component kernels")
    print("  public routing status: baseline spatial kernel modules")
    print(f"  eligible many-row >=20% warm win:       {fast_enough}")
    print(f"  eligible fixed-size <=10% slowdowns:    {not_too_slow}")
    print("  next Spatial F2C-B action:              keep sidecar-only unless public fixed-size backend owner is introduced")


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
