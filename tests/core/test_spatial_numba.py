from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tal.spatial.kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_NUMBA,
    ROTATION_INTERP_BACKEND_SCIPY,
    slerp_quat_backend,
)
from tal.spatial.kernels.kinematics_temporal_backends import (
    KINEMATICS_TEMPORAL_BACKEND_NUMBA,
    KINEMATICS_TEMPORAL_BACKEND_NUMPY,
    cumulative_trapezoid_block_backend,
)


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
