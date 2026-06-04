from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tal.spatial.kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_NUMBA,
    ROTATION_INTERP_BACKEND_SCIPY,
    slerp_quat_backend,
)
from tal.spatial.kernels.kinematics_smoothing_backends import (
    KINEMATICS_SMOOTHING_BACKEND_NUMBA,
    KINEMATICS_SMOOTHING_BACKEND_NUMPY,
    gaussian_smoothing_block_backend,
    moving_average_smoothing_block_backend,
)
from tal.spatial.kernels.kinematics_temporal_backends import (
    KINEMATICS_TEMPORAL_BACKEND_NUMBA,
    KINEMATICS_TEMPORAL_BACKEND_NUMPY,
    cumulative_trapezoid_block_backend,
)
from tal.spatial.kernels.topology_scan_backends import (
    SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
    SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY,
    chain_pose_compose_block_backend,
)
from tal.utils.numba_scan import ScanAxisSpec, ScanInputSpec, prepare_scan_rows


def _require_numba() -> None:
    pytest.importorskip("numba")


def _z_quat(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.asarray([0.0, 0.0, np.sin(0.5 * radians), np.cos(0.5 * radians)], dtype=np.float64)


def _assert_quat_equivalent(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-7) -> None:
    assert actual.shape == expected.shape
    actual_rows = actual.reshape((-1, 4))
    expected_rows = expected.reshape((-1, 4))
    for idx in range(expected_rows.shape[0]):
        if np.isnan(expected_rows[idx]).all():
            assert np.isnan(actual_rows[idx]).all()
            continue
        if np.allclose(actual_rows[idx], expected_rows[idx], atol=atol, rtol=0.0):
            continue
        if np.allclose(actual_rows[idx], -expected_rows[idx], atol=atol, rtol=0.0):
            continue
        raise AssertionError(f"quaternion mismatch at row {idx}: {actual_rows[idx]!r} vs {expected_rows[idx]!r}")


def _direct_slerp_pair(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected = slerp_quat_backend(q0, q1, alpha, valid, backend=ROTATION_INTERP_BACKEND_SCIPY)
    actual = slerp_quat_backend(q0, q1, alpha, valid, backend=ROTATION_INTERP_BACKEND_NUMBA)
    return actual, expected


def _assert_smoothing_status_translations(kernel, **kwargs: object) -> None:
    values = np.ones((1, 4, 2), dtype=np.float64)
    param = np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    owner = r"spatial\.kinematics\.temporal\.smoothing_backend"
    with pytest.raises(ValueError, match=owner + r": valid mask must be left-packed"):
        kernel(values, param, np.asarray([[True, False, True, False]], dtype=bool), **kwargs)
    with pytest.raises(ValueError, match=owner + r": temporal operation requires at least 1 valid samples"):
        kernel(values, param, np.asarray([[False, False, False, False]], dtype=bool), **kwargs)
    with pytest.raises(ValueError, match=owner + r": param domain must be finite"):
        kernel(
            values,
            np.asarray([[0.0, 1.0, 0.5, 3.0]], dtype=np.float64),
            np.asarray([[True, True, True, False]], dtype=bool),
            **kwargs,
        )


def _topology_chain_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    translation = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.0, 1.0], [999.0, 999.0, 999.0], [-5.0, -5.0, -5.0]],
            [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 0.0], [0.25, 0.0, 0.5]],
        ],
        dtype=np.float64,
    )
    quat = np.asarray(
        [
            [2.0 * _z_quat(0.0), 1.5 * _z_quat(30.0), _z_quat(-20.0), np.zeros(4), np.zeros(4)],
            [_z_quat(10.0), 0.5 * _z_quat(45.0), _z_quat(90.0), _z_quat(-30.0), _z_quat(15.0)],
        ],
        dtype=np.float64,
    )
    valid = np.asarray([[True, True, True, False, False], [True, True, True, True, True]], dtype=bool)
    direction = np.asarray([[1, -1, 1, 99, 99], [1, 1, -1, 1, -1]], dtype=np.int64)
    return translation, quat, valid, direction


def test_spatial_numba_001_slerp_backend_parity() -> None:
    """ID: SPATIAL_NUMBA_001_slerp_backend_parity."""
    _require_numba()
    q0 = np.asarray(
        [
            [_z_quat(0.0), _z_quat(10.0), _z_quat(20.0)],
            [_z_quat(90.0), _z_quat(120.0), _z_quat(150.0)],
        ]
    )
    q1 = np.asarray(
        [
            [_z_quat(60.0), _z_quat(80.0), _z_quat(100.0)],
            [_z_quat(180.0), _z_quat(210.0), _z_quat(240.0)],
        ]
    )
    alpha = np.asarray([[0.0, 0.5, 1.0], [0.25, np.nan, 0.75]])
    valid = np.asarray([[True, True, True], [True, True, False]])
    actual, expected = _direct_slerp_pair(q0, q1, alpha, valid)
    _assert_quat_equivalent(actual, expected)


def test_spatial_numba_002_slerp_shortest_arc_parity() -> None:
    """ID: SPATIAL_NUMBA_002_slerp_shortest_arc_parity."""
    _require_numba()
    q0 = np.asarray([[_z_quat(0.0), _z_quat(15.0)]])
    q1 = -np.asarray([[_z_quat(170.0), _z_quat(175.0)]])
    alpha = np.asarray([[0.5, 0.75]])
    valid = np.asarray([[True, True]])
    actual, expected = _direct_slerp_pair(q0, q1, alpha, valid)
    _assert_quat_equivalent(actual, expected)
    assert np.all(np.sum(actual * q0, axis=-1) > 0.0)


def test_spatial_numba_003_slerp_invalid_inputs_fail_closed() -> None:
    """ID: SPATIAL_NUMBA_003_slerp_invalid_inputs_fail_closed."""
    _require_numba()
    with pytest.raises(ValueError, match=r"spatial\.rotation\.interp_backend: quaternion norm must be finite and > 0"):
        slerp_quat_backend(
            np.asarray([[[0.0, 0.0, 0.0, 0.0]]]),
            np.asarray([[_z_quat(90.0)]]),
            np.asarray([[0.5]]),
            np.asarray([[True]]),
            backend=ROTATION_INTERP_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.rotation\.interp_backend: quaternion norm must be finite and > 0"):
        slerp_quat_backend(
            np.asarray([[[1e308, 0.0, 0.0, 0.0]]]),
            np.asarray([[_z_quat(90.0)]]),
            np.asarray([[0.5]]),
            np.asarray([[True]]),
            backend=ROTATION_INTERP_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.rotation\.interp_backend: finite alpha values must be within"):
        slerp_quat_backend(
            np.asarray([[_z_quat(0.0)]]),
            np.asarray([[_z_quat(90.0)]]),
            np.asarray([[1.5]]),
            np.asarray([[True]]),
            backend=ROTATION_INTERP_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.rotation\.interp_backend: q0 and q1 must share shape"):
        slerp_quat_backend(
            np.zeros((1, 3, 4)),
            np.zeros((2, 3, 4)),
            np.zeros((1, 3)),
            np.ones((1, 3), dtype=bool),
            backend=ROTATION_INTERP_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.rotation\.interp_backend: alpha and valid must share shape"):
        slerp_quat_backend(
            np.zeros((1, 3, 4)),
            np.zeros((1, 3, 4)),
            np.zeros((1, 3)),
            np.ones((2, 3), dtype=bool),
            backend=ROTATION_INTERP_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.rotation\.interp_backend: alpha/valid must match q0/q1"):
        slerp_quat_backend(
            np.zeros((1, 3, 4)),
            np.zeros((1, 3, 4)),
            np.zeros((3,)),
            np.ones((3,), dtype=bool),
            backend=ROTATION_INTERP_BACKEND_NUMBA,
        )
    out = slerp_quat_backend(
        np.asarray([[[0.0, 0.0, 0.0, 0.0]]]),
        np.asarray([[_z_quat(90.0)]]),
        np.asarray([[0.5]]),
        np.asarray([[False]]),
        backend=ROTATION_INTERP_BACKEND_NUMBA,
    )
    assert np.isnan(out).all()


def test_spatial_numba_004_slerp_cold_warm_benchmark_recorded() -> None:
    """ID: SPATIAL_NUMBA_004_slerp_cold_warm_benchmark_recorded."""
    text = Path("benchmarks/bench_spatial_slerp_numba_backends.py").read_text(encoding="utf-8")
    assert "from _numba_bench import cold_subprocess, time_once, warm_median" in text
    assert 'return ("many-short", "fewer-long")' in text
    assert "--cold-case" in text
    assert "slerp numba first-call fresh process" in text
    assert "slerp numba warm median" in text


def test_spatial_numba_011_moving_average_backend_parity() -> None:
    """ID: SPATIAL_NUMBA_011_moving_average_backend_parity."""
    _require_numba()
    param = np.asarray(
        [
            [[0.0, 0.5, 1.5, 3.0, 99.0], [0.0, 1.0, 2.0, 3.0, 4.0]],
            [[0.0, 0.25, 1.0, 2.5, 4.5], [0.0, 2.0, 4.0, 6.0, 8.0]],
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [
            [[True, True, True, True, False], [True, True, True, True, True]],
            [[True, True, True, True, True], [True, True, True, False, False]],
        ],
        dtype=bool,
    )
    values = np.stack((param + 1.0, 2.0 * param, param * param), axis=-1)
    values[0, 0, 4, :] = 1000.0
    expected = moving_average_smoothing_block_backend(values, param, valid, window=3)
    actual = moving_average_smoothing_block_backend(
        values,
        param,
        valid,
        window=3,
        backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
    assert actual.shape == values.shape

    for bad_window in (0, 4):
        with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.smoothing_backend: window must"):
            moving_average_smoothing_block_backend(
                values,
                param,
                valid,
                window=bad_window,
                backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA,
            )
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.smoothing_backend: param/valid shapes"):
        moving_average_smoothing_block_backend(values, param, valid[0], window=3, backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA)
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.smoothing_backend: values non-core shape"):
        moving_average_smoothing_block_backend(values[0], param, valid, window=3, backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA)
    _assert_smoothing_status_translations(
        moving_average_smoothing_block_backend,
        window=3,
        backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA,
    )


def test_spatial_numba_012_gaussian_smoothing_backend_parity() -> None:
    """ID: SPATIAL_NUMBA_012_gaussian_smoothing_backend_parity."""
    _require_numba()
    param = np.asarray(
        [
            [[0.0, 0.1, 10.0, 10.1, 10.2], [0.0, 1.0, 3.0, 6.0, 10.0]],
            [[0.0, 0.3, 0.9, 2.0, 4.5], [0.0, 2.0, 4.0, 6.0, 8.0]],
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [
            [[True, True, True, True, True], [True, True, True, True, False]],
            [[True, True, True, True, True], [True, True, True, False, False]],
        ],
        dtype=bool,
    )
    values = np.zeros(param.shape + (3,), dtype=np.float64)
    values[..., 0] = param
    values[..., 1] = np.sin(param)
    values[..., 2] = np.cos(param)
    expected = gaussian_smoothing_block_backend(values, param, valid, window=5, sigma=1.0)
    actual = gaussian_smoothing_block_backend(
        values,
        param,
        valid,
        window=5,
        sigma=1.0,
        backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
    assert actual.shape == values.shape

    for bad_sigma in (0.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.smoothing_backend: sigma must"):
            gaussian_smoothing_block_backend(
                values,
                param,
                valid,
                window=5,
                sigma=bad_sigma,
                backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA,
            )
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.smoothing_backend: window must"):
        gaussian_smoothing_block_backend(values, param, valid, window=4, sigma=1.0, backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA)
    _assert_smoothing_status_translations(
        gaussian_smoothing_block_backend,
        window=3,
        sigma=1.0,
        backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA,
    )


def test_spatial_numba_013_trapezoid_backend_parity() -> None:
    """ID: SPATIAL_NUMBA_013_trapezoid_backend_parity."""
    _require_numba()
    param = np.asarray(
        [
            [[0.0, 0.5, 1.5, 3.0, 99.0], [0.0, 1.0, 2.0, 3.0, 4.0]],
            [[0.0, 0.25, 1.0, 2.5, 4.5], [0.0, 2.0, 4.0, 6.0, 8.0]],
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [
            [[True, True, True, True, False], [True, True, True, True, True]],
            [[True, True, True, True, True], [True, True, True, False, False]],
        ],
        dtype=bool,
    )
    values = np.empty(param.shape + (3,), dtype=np.float64)
    values[..., 0] = param + 1.0
    values[..., 1] = 2.0 * param
    values[..., 2] = param * param
    values[0, 0, 4, :] = 1000.0
    expected = cumulative_trapezoid_block_backend(
        values,
        param,
        valid,
        initial_value=2.5,
        backend=KINEMATICS_TEMPORAL_BACKEND_NUMPY,
    )
    actual = cumulative_trapezoid_block_backend(
        values,
        param,
        valid,
        initial_value=2.5,
        backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)
    assert actual.shape == values.shape


def test_spatial_numba_014_temporal_left_packed_validation_parity() -> None:
    """ID: SPATIAL_NUMBA_014_temporal_left_packed_validation_parity."""
    _require_numba()
    values = np.ones((1, 4, 2), dtype=np.float64)
    param = np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.integral_backend: valid mask must be left-packed"):
        cumulative_trapezoid_block_backend(
            values,
            param,
            np.asarray([[True, False, True, False]], dtype=bool),
            initial_value=0.0,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.integral_backend: temporal operation requires"):
        cumulative_trapezoid_block_backend(
            values,
            param,
            np.asarray([[False, False, False, False]], dtype=bool),
            initial_value=0.0,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.integral_backend: param domain must be finite"):
        cumulative_trapezoid_block_backend(
            values,
            np.asarray([[0.0, 1.0, 0.5, 3.0]], dtype=np.float64),
            np.asarray([[True, True, True, False]], dtype=bool),
            initial_value=0.0,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.integral_backend: param/valid shapes must match"):
        cumulative_trapezoid_block_backend(
            values,
            param,
            np.ones((4,), dtype=bool),
            initial_value=0.0,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.integral_backend: values non-core shape must match"):
        cumulative_trapezoid_block_backend(
            np.ones((2, 4, 2), dtype=np.float64),
            param,
            np.ones((1, 4), dtype=bool),
            initial_value=0.0,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=r"spatial\.kinematics\.temporal\.integral_backend: values must include"):
        cumulative_trapezoid_block_backend(
            np.ones((4,), dtype=np.float64),
            param,
            np.ones((1, 4), dtype=bool),
            initial_value=0.0,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
        )


def test_spatial_numba_015_trapezoid_scan_cold_warm_benchmark_recorded() -> None:
    """ID: SPATIAL_NUMBA_015_trapezoid_scan_cold_warm_benchmark_recorded."""
    text = Path("benchmarks/bench_spatial_kinematics_scan_numba_backends.py").read_text(encoding="utf-8")
    assert "from _numba_bench import break_even_calls, cold_subprocess, time_once, warm_median" in text
    assert 'return ("many-short", "fewer-long", "high-core")' in text
    assert "--cold-case" in text
    assert "trapezoid numba first-call fresh process" in text
    assert "trapezoid numba warm median" in text


def test_spatial_numba_020_local_poly_backend_decision_is_explicit() -> None:
    """ID: SPATIAL_NUMBA_020_local_poly_backend_decision_is_explicit."""
    import tal.spatial.kernels.kinematics_smoothing_backends as smoothing_backends

    assert not hasattr(smoothing_backends, "local_poly_smoothing_block_backend")
    assert not hasattr(smoothing_backends, "local_poly_derivative_block_backend")
    numba_text = Path("tal/spatial/kernels/kinematics_smoothing_numba_backends.py").read_text(encoding="utf-8")
    ops_text = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    temporal_text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    assert "local_poly" not in numba_text
    assert "local_poly_smooth_kernel" in ops_text
    assert "local_poly_first_derivative_kernel" in temporal_text


def test_spatial_topo_numba_001_chain_pose_compose_backend_parity() -> None:
    """ID: SPATIAL_TOPO_NUMBA_001_chain_pose_compose_backend_parity."""
    _require_numba()
    translation, quat, valid, direction = _topology_chain_inputs()
    expected_t, expected_q = chain_pose_compose_block_backend(
        translation,
        quat,
        valid,
        direction,
        backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY,
    )
    actual_t, actual_q = chain_pose_compose_block_backend(
        translation,
        quat,
        valid,
        direction,
        backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
    )
    np.testing.assert_allclose(actual_t, expected_t, equal_nan=True, rtol=1e-12, atol=1e-12)
    _assert_quat_equivalent(actual_q, expected_q, atol=1e-12)
    assert np.isnan(actual_t[0, 3:]).all()
    assert np.isnan(actual_q[0, 3:]).all()


def test_spatial_topo_numba_002_chain_pose_invalid_inputs_fail_closed() -> None:
    """ID: SPATIAL_TOPO_NUMBA_002_chain_pose_invalid_inputs_fail_closed."""
    _require_numba()
    owner = r"spatial\.topology_scan\.pose_backend"
    translation, quat, valid, direction = _topology_chain_inputs()
    with pytest.raises(ValueError, match=owner + r": translation, quat, valid, and direction shapes must match exactly"):
        chain_pose_compose_block_backend(
            np.zeros((1, 3, 3)),
            np.zeros((2, 3, 4)),
            np.ones((2, 3), dtype=bool),
            np.ones((2, 3), dtype=np.int64),
            backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=owner + r": translation must have trailing vector dim length 3"):
        chain_pose_compose_block_backend(
            np.zeros((1, 3, 2)),
            np.zeros((1, 3, 4)),
            np.ones((1, 3), dtype=bool),
            np.ones((1, 3), dtype=np.int64),
            backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=owner + r": quat must have trailing quaternion dim length 4"):
        chain_pose_compose_block_backend(
            np.zeros((1, 3, 3)),
            np.zeros((1, 3, 3)),
            np.ones((1, 3), dtype=bool),
            np.ones((1, 3), dtype=np.int64),
            backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
        )
    bad_quat = quat.copy()
    bad_quat[0, 1, :] = 0.0
    with pytest.raises(ValueError, match=owner + r": quaternion norm must be finite and > 0"):
        chain_pose_compose_block_backend(translation, bad_quat, valid, direction, backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA)
    bad_direction = direction.copy()
    bad_direction[1, 2] = 0
    with pytest.raises(ValueError, match=owner + r": direction values must be 1 or -1"):
        chain_pose_compose_block_backend(
            translation,
            quat,
            valid,
            bad_direction,
            backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
        )
    with pytest.raises(ValueError, match=owner + r": topology scan requires at least 1 valid edge"):
        chain_pose_compose_block_backend(
            translation[:1],
            quat[:1],
            np.zeros((1, valid.shape[1]), dtype=bool),
            direction[:1],
            backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
        )
    non_left_packed = valid.copy()
    non_left_packed[0] = np.asarray([True, False, True, False, False])
    with pytest.raises(ValueError, match=owner + r": valid mask must be left-packed"):
        chain_pose_compose_block_backend(
            translation,
            quat,
            non_left_packed,
            direction,
            backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
        )


def test_spatial_topo_numba_003_topology_scan_batch_isolation_preserved() -> None:
    """ID: SPATIAL_TOPO_NUMBA_003_topology_scan_batch_isolation_preserved."""
    _require_numba()
    translation, quat, valid, direction = _topology_chain_inputs()
    all_t, all_q = chain_pose_compose_block_backend(
        translation,
        quat,
        valid,
        direction,
        backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
    )
    row0_t, row0_q = chain_pose_compose_block_backend(
        translation[:1],
        quat[:1],
        valid[:1],
        direction[:1],
        backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
    )
    row1_t, row1_q = chain_pose_compose_block_backend(
        translation[1:],
        quat[1:],
        valid[1:],
        direction[1:],
        backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
    )
    np.testing.assert_allclose(all_t[:1], row0_t, equal_nan=True, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(all_t[1:], row1_t, equal_nan=True, rtol=1e-12, atol=1e-12)
    _assert_quat_equivalent(all_q[:1], row0_q, atol=1e-12)
    _assert_quat_equivalent(all_q[1:], row1_q, atol=1e-12)


def test_spatial_topo_numba_004_nested_time_chain_scan_decision_is_explicit() -> None:
    """ID: SPATIAL_TOPO_NUMBA_004_nested_time_chain_scan_decision_is_explicit."""
    prepared = prepare_scan_rows(
        (np.zeros((2, 3, 4, 3)), np.ones((2, 3, 4), dtype=bool)),
        (ScanInputSpec("translation", 2, 1, np.float64), ScanInputSpec("valid", 2, 0, bool)),
        ordered_axes=(ScanAxisSpec("time", "scan"), ScanAxisSpec("chain", "topology")),
        output_core_shapes=((3,),),
        owner="spatial.topology_scan.metadata",
    )
    assert prepared.outer_shape == (2,)
    assert prepared.ordered_shape == (3, 4)
    assert prepared.output_shapes == ((2, 3, 4, 3),)
    backend_text = Path("tal/spatial/kernels/_topology_scan_common.py").read_text(encoding="utf-8")
    assert 'ScanAxisSpec("chain", "topology")' in backend_text
    assert 'ScanAxisSpec("time", "scan")' not in backend_text


def test_numba_opt_006_spatial_backends_skip_cleanly_without_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: NUMBA_OPT_006_spatial_backends_skip_cleanly_without_numba."""
    import tal.utils.numba_support as numba_support

    def _raise_import_error():
        raise ImportError("missing numba")

    monkeypatch.setattr(numba_support, "_import_numba", _raise_import_error)
    out = slerp_quat_backend(
        np.asarray([[_z_quat(0.0)]]),
        np.asarray([[_z_quat(90.0)]]),
        np.asarray([[0.5]]),
        np.asarray([[True]]),
        backend=ROTATION_INTERP_BACKEND_SCIPY,
    )
    assert out.shape == (1, 1, 4)
    integrated = cumulative_trapezoid_block_backend(
        np.ones((1, 2, 1), dtype=np.float64),
        np.asarray([[0.0, 1.0]], dtype=np.float64),
        np.asarray([[True, True]], dtype=bool),
        initial_value=0.0,
        backend=KINEMATICS_TEMPORAL_BACKEND_NUMPY,
    )
    assert integrated.shape == (1, 2, 1)
    smoothed = moving_average_smoothing_block_backend(
        np.ones((1, 3, 1), dtype=np.float64),
        np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
        np.asarray([[True, True, True]], dtype=bool),
        window=3,
        backend=KINEMATICS_SMOOTHING_BACKEND_NUMPY,
    )
    assert smoothed.shape == (1, 3, 1)
    gaussian = gaussian_smoothing_block_backend(
        np.ones((1, 3, 1), dtype=np.float64),
        np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
        np.asarray([[True, True, True]], dtype=bool),
        window=3,
        sigma=1.0,
        backend=KINEMATICS_SMOOTHING_BACKEND_NUMPY,
    )
    assert gaussian.shape == (1, 3, 1)
    topo_t, topo_q = chain_pose_compose_block_backend(
        np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64),
        np.asarray([[2.0 * _z_quat(0.0)]], dtype=np.float64),
        np.asarray([[True]], dtype=bool),
        np.asarray([[1]], dtype=np.int64),
        backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMPY,
    )
    assert topo_t.shape == (1, 1, 3)
    assert topo_q.shape == (1, 1, 4)


def test_numba_opt_007_spatial_backends_fail_closed_when_numba_requested_without_numba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: NUMBA_OPT_007_spatial_backends_fail_closed_when_numba_requested_without_numba."""
    import tal.utils.numba_support as numba_support

    def _raise_import_error():
        raise ImportError("missing numba")

    monkeypatch.setattr(numba_support, "_import_numba", _raise_import_error)
    with pytest.raises(ImportError, match=r"spatial\.rotation\.interp_backend: numba is required for backend='numba'"):
        slerp_quat_backend(
            np.asarray([[_z_quat(0.0)]]),
            np.asarray([[_z_quat(90.0)]]),
            np.asarray([[0.5]]),
            np.asarray([[True]]),
            backend=ROTATION_INTERP_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"spatial\.kinematics\.temporal\.integral_backend: numba is required"):
        cumulative_trapezoid_block_backend(
            np.ones((1, 2, 1), dtype=np.float64),
            np.asarray([[0.0, 1.0]], dtype=np.float64),
            np.asarray([[True, True]], dtype=bool),
            initial_value=0.0,
            backend=KINEMATICS_TEMPORAL_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"spatial\.kinematics\.temporal\.smoothing_backend: numba is required"):
        moving_average_smoothing_block_backend(
            np.ones((1, 3, 1), dtype=np.float64),
            np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
            np.asarray([[True, True, True]], dtype=bool),
            window=3,
            backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"spatial\.kinematics\.temporal\.smoothing_backend: numba is required"):
        gaussian_smoothing_block_backend(
            np.ones((1, 3, 1), dtype=np.float64),
            np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
            np.asarray([[True, True, True]], dtype=bool),
            window=3,
            sigma=1.0,
            backend=KINEMATICS_SMOOTHING_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"spatial\.topology_scan\.pose_backend: numba is required"):
        chain_pose_compose_block_backend(
            np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64),
            np.asarray([[_z_quat(0.0)]], dtype=np.float64),
            np.asarray([[True]], dtype=bool),
            np.asarray([[1]], dtype=np.int64),
            backend=SPATIAL_TOPOLOGY_SCAN_BACKEND_NUMBA,
        )
