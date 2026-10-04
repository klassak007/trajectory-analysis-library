from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from _numba_bench import break_even_calls, cold_subprocess, time_once, warm_median
from tal.spatial.kernels.higher_order_interp_backends import (
    POSE_HIGHER_ORDER_BACKEND_NUMBA,
    POSE_HIGHER_ORDER_BACKEND_NUMPY,
    ROTATION_HIGHER_ORDER_BACKEND_NUMBA,
    ROTATION_HIGHER_ORDER_BACKEND_NUMPY,
    PoseInterpWindow,
    QuatInterpWindow,
    pose_cubic_squad_block_backend,
    squad_quat_block_backend,
)
from tal.spatial.kernels.higher_order_interp_numba_backends import _compiled_pose_block, _compiled_squad_block


@dataclass(frozen=True)
class HigherOrderCase:
    name: str
    kind: str
    title: str
    rows: int
    query: int
    quat_window: QuatInterpWindow
    pose_window: PoseInterpWindow
    alpha: np.ndarray
    valid: np.ndarray
    eligible: bool


def _z_quat(degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(degrees)
    quat = np.zeros(degrees.shape + (4,), dtype=np.float64)
    quat[..., 2] = np.sin(0.5 * radians)
    quat[..., 3] = np.cos(0.5 * radians)
    return quat


def _case_names() -> tuple[str, ...]:
    return ("quat-many-query", "quat-fewer-long", "pose-many-query", "pose-fewer-long")


def _case_shape(name: str) -> tuple[str, str, int, int, bool]:
    if name == "quat-many-query":
        return "quat", "quaternion SQUAD many-query eligible", 4096, 32, True
    if name == "quat-fewer-long":
        return "quat", "quaternion SQUAD fewer-long evidence", 128, 256, False
    if name == "pose-many-query":
        return "pose", "pose Catmull-Rom+SQUAD many-query eligible", 4096, 32, True
    if name == "pose-fewer-long":
        return "pose", "pose Catmull-Rom+SQUAD fewer-long evidence", 128, 256, False
    raise ValueError(f"unknown higher-order interpolation case {name!r}")


def _case_data(name: str) -> HigherOrderCase:
    kind, title, rows, query, eligible = _case_shape(name)
    row_idx = np.arange(rows, dtype=np.float64)[:, None]
    query_idx = np.arange(query, dtype=np.float64)[None, :]
    base = 0.01 * row_idx + 0.25 * query_idx
    q_prev = _z_quat(base - 1.5)
    q0 = _z_quat(base)
    q1 = _z_quat(base + 1.5)
    q_next = _z_quat(base + 3.0)
    alpha = np.broadcast_to(np.linspace(0.0, 1.0, query, dtype=np.float64), (rows, query)).copy()
    valid = np.ones((rows, query), dtype=bool)
    t_prev = np.stack((base - 1.0, 0.5 * np.sin(base), 0.1 * np.cos(base)), axis=-1)
    t0 = np.stack((base, 0.5 * np.sin(base + 0.1), 0.1 * np.cos(base + 0.1)), axis=-1)
    t1 = np.stack((base + 1.0, 0.5 * np.sin(base + 0.2), 0.1 * np.cos(base + 0.2)), axis=-1)
    t_next = np.stack((base + 2.0, 0.5 * np.sin(base + 0.3), 0.1 * np.cos(base + 0.3)), axis=-1)
    return HigherOrderCase(
        name,
        kind,
        title,
        rows,
        query,
        QuatInterpWindow(q_prev, q0, q1, q_next),
        PoseInterpWindow(t_prev, t0, t1, t_next, q_prev, q0, q1, q_next),
        alpha,
        valid,
        eligible,
    )


def _baseline(case: HigherOrderCase):
    if case.kind == "quat":
        return squad_quat_block_backend(
            case.quat_window,
            case.alpha,
            case.valid,
            backend=ROTATION_HIGHER_ORDER_BACKEND_NUMPY,
        )
    return pose_cubic_squad_block_backend(
        case.pose_window,
        case.alpha,
        case.valid,
        backend=POSE_HIGHER_ORDER_BACKEND_NUMPY,
    )


def _numba(case: HigherOrderCase):
    if case.kind == "quat":
        return squad_quat_block_backend(
            case.quat_window,
            case.alpha,
            case.valid,
            backend=ROTATION_HIGHER_ORDER_BACKEND_NUMBA,
        )
    return pose_cubic_squad_block_backend(
        case.pose_window,
        case.alpha,
        case.valid,
        backend=POSE_HIGHER_ORDER_BACKEND_NUMBA,
    )


def _assert_quat_equivalent(actual: np.ndarray, expected: np.ndarray) -> None:
    dots = np.sum(actual * expected, axis=-1)
    aligned = np.where(dots[..., None] < 0.0, -actual, actual)
    np.testing.assert_allclose(aligned, expected, equal_nan=True, rtol=1e-10, atol=1e-10)


def _assert_parity(actual, expected, *, kind: str) -> None:
    if kind == "quat":
        _assert_quat_equivalent(actual, expected)
        return
    actual_t, actual_q = actual
    expected_t, expected_q = expected
    np.testing.assert_allclose(actual_t, expected_t, equal_nan=True, rtol=1e-10, atol=1e-10)
    _assert_quat_equivalent(actual_q, expected_q)


def _cold_subprocess(case_name: str) -> float:
    return cold_subprocess(__file__, ("--cold-case", case_name), cache_prefix="tal-spatial-hiinterp-numba-cache-")


def _run_cold(case_name: str) -> None:
    print(f"{time_once(_numba, _case_data(case_name)):.9f}")


def _report_case(case_name: str) -> tuple[bool, bool]:
    case = _case_data(case_name)
    expected = _baseline(case)
    actual = _numba(case)
    _assert_parity(actual, expected, kind=case.kind)
    if case.kind == "quat":
        assert _compiled_squad_block().nopython_signatures
    else:
        assert _compiled_pose_block().nopython_signatures
    baseline = warm_median(_baseline, case, repeats=3)
    cold = _cold_subprocess(case_name)
    warm = warm_median(_numba, case, repeats=5)
    fast_enough = case.eligible and warm <= baseline * 0.8
    not_too_slow = (not case.eligible) or warm <= baseline * 1.1
    print(f"\n{case.title}")
    print(f"  shape: rows={case.rows} query={case.query}")
    print(f"  higher-order numpy warm median:           {baseline:.6f}s")
    print(f"  higher-order numba first-call fresh process:{cold:.6f}s")
    print(f"  higher-order numba warm median:           {warm:.6f}s")
    print(f"  higher-order warm speedup:                {baseline / warm:.3f}x")
    print(f"  higher-order break-even warm calls:       {break_even_calls(baseline, cold, warm):.2f}")
    print(f"  eligible many-query gate:                 {case.eligible}")
    return fast_enough, not_too_slow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-case")
    args = parser.parse_args()
    if args.cold_case:
        _run_cold(args.cold_case)
        return
    quat_hits: list[bool] = []
    quat_slowdown_ok: list[bool] = []
    pose_hits: list[bool] = []
    pose_slowdown_ok: list[bool] = []
    for name in _case_names():
        fast_enough, not_too_slow = _report_case(name)
        if name.startswith("quat-"):
            quat_hits.append(fast_enough)
            quat_slowdown_ok.append(not_too_slow)
        else:
            pose_hits.append(fast_enough)
            pose_slowdown_ok.append(not_too_slow)
    quat_retained = any(quat_hits) and all(quat_slowdown_ok)
    pose_retained = quat_retained and any(pose_hits) and all(pose_slowdown_ok)
    print("\nhigher-order quaternion retention gate")
    print("  nopython/parity:                          pass")
    print(f"  eligible many-query >=20% warm win:       {any(quat_hits)}")
    print(f"  eligible many-query <=10% slowdowns:      {all(quat_slowdown_ok)}")
    print(f"  retain numba backend:                     {quat_retained}")
    print("\nSpatial F2C decision input: higher_order_quaternion_squad")
    print("  public routing status: no public higher-order interpolation route")
    print(f"  eligible many-query >=20% warm win:       {any(quat_hits)}")
    print(f"  eligible quaternion <=10% slowdowns:      {all(quat_slowdown_ok)}")
    print("  next Spatial F2C-B action:                keep sidecar-only until public route exists")
    print("\nhigher-order pose retention gate")
    print("  nopython/parity:                          pass")
    print(f"  eligible many-query >=20% warm win:       {any(pose_hits)}")
    print(f"  eligible many-query <=10% slowdowns:      {all(pose_slowdown_ok)}")
    print(f"  quaternion dependency retained:           {quat_retained}")
    print(f"  retain numba backend:                     {pose_retained}")
    print("\nSpatial F2C decision input: higher_order_pose_cubic_squad")
    print("  public routing status: no public higher-order interpolation route")
    print(f"  eligible many-query >=20% warm win:       {any(pose_hits)}")
    print(f"  eligible pose <=10% slowdowns:            {all(pose_slowdown_ok)}")
    print("  next Spatial F2C-B action:                keep sidecar-only until public route exists")


if __name__ == "__main__":
    main()
