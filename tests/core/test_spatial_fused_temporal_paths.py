from __future__ import annotations

import gc
import importlib.util
import multiprocessing as mp
import weakref
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

import benchmarks._spatial_path_benchmark_protocol as benchmark_protocol
from benchmarks._spatial_path_benchmark_protocol import MeasuredRoute
from benchmarks._spatial_path_execution_routes import (
    materialize_route_result,
    route_effective_backend,
)
from benchmarks.bench_spatial_fused_temporal_paths import (
    BenchmarkCaseConfig,
    frozen_fixture,
    measure_cold_route,
    measure_route,
    measure_rss,
    prepare_fixture,
    validate_prepared_fixture,
)
from benchmarks.bench_spatial_fused_temporal_paths import main as benchmark_main
from tal.spatial.kernels.higher_order_interp_backends import (
    ROTATION_HIGHER_ORDER_BACKEND_NUMBA,
    ROTATION_HIGHER_ORDER_BACKEND_NUMPY,
    QuatInterpWindow,
    squad_quat_block_backend,
)
from tal.spatial.kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_NUMBA,
    ROTATION_INTERP_BACKEND_SCIPY,
    slerp_quat_backend,
)


def _axis_angle_quat(angle: float, dtype: np.dtype) -> np.ndarray:
    axis = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    quaternion = np.empty(4, dtype=np.float64)
    quaternion[:3] = axis * np.sin(0.5 * angle)
    quaternion[3] = np.cos(0.5 * angle)
    return quaternion.astype(dtype)


def _slerp_pair(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray, backend: str) -> np.ndarray:
    count = alpha.size
    left = np.broadcast_to(q0, (1, count, 4)).copy()
    right = np.broadcast_to(q1, (1, count, 4)).copy()
    valid = np.ones((1, count), dtype=bool)
    return slerp_quat_backend(left, right, alpha[None, :], valid, backend=backend)[0]


@pytest.mark.parametrize("dtype", (np.dtype("float32"), np.dtype("float64")))
def test_spatial_core_fused_path_001_reconciled_numba_slerp_matches_scipy(dtype: np.dtype) -> None:
    """ID: SPATIAL_CORE_FUSED_PATH_001_reconciled_numba_slerp_matches_scipy."""
    pytest.importorskip("numba")
    threshold_angle = 2.0 * np.arccos(0.9995)
    angles = (0.04, threshold_angle - 1.0e-6, threshold_angle + 1.0e-6, 0.08)
    alpha = np.asarray((0.0, 0.25, 0.75, 1.0), dtype=dtype)
    identity = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=dtype)
    tolerance = 1.0e-6 if dtype == np.dtype("float32") else 1.0e-12
    for angle in angles:
        target = _axis_angle_quat(angle, dtype)
        actual = _slerp_pair(identity, target, alpha, ROTATION_INTERP_BACKEND_NUMBA)
        expected = _slerp_pair(identity, target, alpha, ROTATION_INTERP_BACKEND_SCIPY)
        np.testing.assert_allclose(
            SciRotation.from_quat(actual).as_matrix(),
            SciRotation.from_quat(expected).as_matrix(),
            rtol=tolerance,
            atol=tolerance,
        )


def test_spatial_hard_fused_path_001_slerp_threshold_endpoints_and_shortest_arc() -> None:
    """ID: SPATIAL_HARD_FUSED_PATH_001_slerp_threshold_endpoints_and_shortest_arc."""
    pytest.importorskip("numba")
    q0 = _axis_angle_quat(0.31, np.dtype("float64"))
    q1 = -_axis_angle_quat(0.310001, np.dtype("float64"))
    alpha = np.asarray((0.0, 0.25, 0.75, 1.0))
    actual = _slerp_pair(q0, q1, alpha, ROTATION_INTERP_BACKEND_NUMBA)
    expected = Slerp(np.asarray((0.0, 1.0)), SciRotation.from_quat(np.stack((q0, q1))))(alpha).as_quat()
    np.testing.assert_allclose(
        SciRotation.from_quat(actual).as_matrix(),
        SciRotation.from_quat(expected).as_matrix(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    invalid = np.zeros((1, 2, 4), dtype=np.float64)
    valid = np.asarray([[False, False]])
    masked = slerp_quat_backend(invalid, invalid, np.zeros((1, 2)), valid, backend=ROTATION_INTERP_BACKEND_NUMBA)
    assert np.isnan(masked).all()
    with pytest.raises(ValueError, match="finite alpha values must be within"):
        _slerp_pair(q0, q1, np.asarray((1.1,)), ROTATION_INTERP_BACKEND_NUMBA)
    with pytest.raises(ValueError, match="quaternion norm must be finite and > 0"):
        _slerp_pair(np.zeros(4), q1, np.asarray((0.5,)), ROTATION_INTERP_BACKEND_NUMBA)


@pytest.mark.parametrize("dtype", (np.dtype("float32"), np.dtype("float64")))
@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_spatial_core_slerp_half_turn_001_matches_scipy_convention(
    dtype: np.dtype,
    sign: float,
) -> None:
    """ID: SPATIAL_CORE_SLERP_HALF_TURN_001_matches_scipy_convention."""
    pytest.importorskip("numba")
    start = SciRotation.from_rotvec(np.asarray((0.21, -0.08, 0.14))).as_quat().astype(dtype)
    axis = np.asarray((sign, -2.0 * sign, 0.5 * sign))
    axis /= np.linalg.norm(axis)
    target = (SciRotation.from_quat(start) * SciRotation.from_rotvec(np.pi * axis)).as_quat().astype(dtype)
    alpha = np.asarray((0.0, 0.25, 0.75, 1.0), dtype=dtype)
    expected = _slerp_pair(start, target, alpha, ROTATION_INTERP_BACKEND_SCIPY)
    actual = _slerp_pair(start, target, alpha, ROTATION_INTERP_BACKEND_NUMBA)
    tolerance = 2.0e-6 if dtype == np.dtype("float32") else 1.0e-12
    np.testing.assert_allclose(
        SciRotation.from_quat(actual).as_matrix(),
        SciRotation.from_quat(expected).as_matrix(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype", (np.dtype("float32"), np.dtype("float64")))
@pytest.mark.parametrize("backend", (ROTATION_INTERP_BACKEND_SCIPY, ROTATION_INTERP_BACKEND_NUMBA))
@pytest.mark.parametrize(
    ("left_values", "right_values"),
    (
        ((1.0, 1.0, 1.0, 3.0), (-1.0, 1.0, -3.0, 1.0)),
        ((1.0, 2.0, 1.0, 1.0), (-2.0, 1.0, -1.0, 1.0)),
    ),
)
def test_spatial_core_slerp_principal_arc_001_uses_independent_scipy_reference(
    dtype: np.dtype,
    backend: str,
    left_values: tuple[float, ...],
    right_values: tuple[float, ...],
) -> None:
    """ID: SPATIAL_CORE_SLERP_PRINCIPAL_ARC_001_uses_independent_scipy_reference."""
    if backend == ROTATION_INTERP_BACKEND_NUMBA:
        pytest.importorskip("numba")
    left = np.asarray(left_values, dtype=dtype)
    right = np.asarray(right_values, dtype=dtype)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    fractions = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=dtype)
    expected = Slerp(
        np.asarray((0.0, 1.0)),
        SciRotation.from_quat(np.stack((left, right))),
    )(fractions).as_matrix()
    tolerance = 2.0e-6 if dtype == np.dtype("float32") else 1.0e-12
    for sign in (-1.0, 1.0):
        signed = sign * right
        actual = _slerp_pair(left, signed, fractions, backend)
        np.testing.assert_allclose(
            SciRotation.from_quat(actual).as_matrix(),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )


def test_spatial_core_fused_path_reconciled_slerp_preserves_squad_parity() -> None:
    pytest.importorskip("numba")
    angles = (-0.04, 0.0, 0.04, 0.08)
    rows = tuple(_axis_angle_quat(angle, np.dtype("float64"))[None, None, :] for angle in angles)
    alpha = np.asarray([[0.25, 0.75]])
    expanded = QuatInterpWindow(*(np.repeat(row, 2, axis=1) for row in rows))
    valid = np.ones_like(alpha, dtype=bool)
    expected = squad_quat_block_backend(
        expanded,
        alpha,
        valid,
        backend=ROTATION_HIGHER_ORDER_BACKEND_NUMPY,
    )
    actual = squad_quat_block_backend(
        expanded,
        alpha,
        valid,
        backend=ROTATION_HIGHER_ORDER_BACKEND_NUMBA,
    )
    np.testing.assert_allclose(
        SciRotation.from_quat(actual.reshape(-1, 4)).as_matrix(),
        SciRotation.from_quat(expected.reshape(-1, 4)).as_matrix(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def _expected_h1_values(parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translations: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    for edge in range(8):
        raw_axis = np.asarray((1 + edge, 2 + edge % 3, 3 + (2 * edge) % 5), dtype=np.float64)
        axis = raw_axis / np.linalg.norm(raw_axis)
        translation = np.column_stack(
            (
                0.25 * (edge + 1) + (0.5 + edge / 32.0) * parameter,
                np.sin(0.3 * (edge + 1) + (0.75 + edge / 16.0) * parameter),
                np.cos(0.2 * (edge + 1) - (0.5 + edge / 32.0) * parameter),
            )
        )
        half_angle = 0.5 * (0.125 * (edge + 1) + (9.0 + edge / 4.0) * parameter)
        quaternion = np.column_stack((np.sin(half_angle)[:, None] * axis, np.cos(half_angle)))
        translations.append(translation)
        quaternions.append(quaternion)
    return np.stack(translations), np.stack(quaternions)


def test_spatial_bench_fused_path_001_frozen_h0_h1_fixtures_and_directions() -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_001_frozen_h0_h1_fixtures_and_directions."""
    h0 = frozen_fixture("h0", query_size=1_000_000)
    h1 = frozen_fixture("h1", query_size=1_000_000)
    parameter = np.arange(129, dtype=np.float64) / 128.0
    query = np.arange(1_000_000, dtype=np.float64) / 999_999.0
    h0_translation = np.zeros((8, 129, 3), dtype=np.float64)
    h0_translation[:, :, 0] = parameter
    h0_quaternion = np.zeros((8, 129, 4), dtype=np.float64)
    h0_quaternion[:, :, 3] = 1.0
    h1_translation, h1_quaternion = _expected_h1_values(parameter)
    h0_relations = tuple((f"f{edge}", f"f{edge + 1}") for edge in range(8))
    h1_relations = tuple(
        (f"f{edge + 1}", f"f{edge}") if edge < 4 else (f"f{edge}", f"f{edge + 1}")
        for edge in range(8)
    )
    assert h0.name == "h0"
    assert h1.name == "h1"
    assert h0.directions == (1,) * 8
    assert h1.directions == (1, 1, 1, 1, -1, -1, -1, -1)
    np.testing.assert_array_equal(h0.parameter, parameter)
    np.testing.assert_array_equal(h1.parameter, parameter)
    np.testing.assert_array_equal(h0.query, query)
    np.testing.assert_array_equal(h1.query, query)
    np.testing.assert_array_equal(h0.translation, h0_translation)
    np.testing.assert_array_equal(h0.quaternion, h0_quaternion)
    np.testing.assert_allclose(h1.translation, h1_translation, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(h1.quaternion, h1_quaternion, rtol=0.0, atol=1.0e-15)
    assert h0.relations == h0_relations
    assert h1.relations == h1_relations
    assert (h0.source, h0.destination) == ("f8", "f0")
    assert (h1.source, h1.destination) == ("f0", "f8")


def test_spatial_bench_fused_path_003_compiled_comparators_are_numba_guarded() -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_003_compiled_comparators_are_numba_guarded."""
    pytest.importorskip("numba")
    validate_prepared_fixture(prepare_fixture(frozen_fixture("h1", query_size=17)))


def test_spatial_bench_fused_path_002_isolated_rss_protocol_reports_native_peak() -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_002_isolated_rss_protocol_reports_native_peak."""
    pytest.importorskip("psutil", exc_type=ImportError)
    config = BenchmarkCaseConfig("h1", 4_000_000, 4)
    result = measure_rss("allocation-probe", config=config)
    assert result.config == config
    assert result.baseline > 0
    assert result.peak >= result.baseline
    assert result.increment > 0
    assert result.sample_count >= 21


def test_spatial_bench_fused_path_004_measured_samples_validate_and_release_outputs() -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_004_measured_samples_validate_and_release_outputs."""
    references: list[weakref.ReferenceType[np.ndarray]] = []
    validations = 0

    def operation() -> np.ndarray:
        gc.collect()
        assert all(reference() is None for reference in references)
        result = np.asarray((1.0, 2.0, 3.0))
        references.append(weakref.ref(result))
        return result

    def validate(result: object) -> None:
        nonlocal validations
        np.testing.assert_array_equal(result, np.asarray((1.0, 2.0, 3.0)))
        validations += 1

    route = MeasuredRoute("probe", operation, lambda result: result, validate)
    timing = measure_route(route, warmups=2, repeats=3)
    gc.collect()
    assert len(timing.samples) == 3
    assert validations == 5
    assert all(reference() is None for reference in references)

    invalid = MeasuredRoute(
        "invalid",
        lambda: np.asarray((1.0,)),
        lambda result: result,
        lambda result: np.testing.assert_array_equal(result, np.asarray((2.0,))),
    )
    with pytest.raises(AssertionError):
        measure_route(invalid, warmups=0, repeats=1)


@pytest.mark.parametrize("edges", (1, 4, 8))
def test_spatial_bench_fused_path_005_cold_subprocess_preserves_case_configuration_without_numba(
    edges: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_005_cold_subprocess_preserves_case_configuration_without_numba."""
    monkeypatch.setenv("PYTHONPATH", "/benchmark-must-not-use-pythonpath")
    config = BenchmarkCaseConfig("h1", 17, edges)
    cold = measure_cold_route("scipy-vectorized", config=config)
    assert cold.config == config


def test_spatial_bench_fused_cold_cache_001_reference_prepares_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_BENCH_FUSED_COLD_CACHE_001_reference_prepares_dependencies."""
    pytest.importorskip("numba")
    cache = tmp_path / "empty-numba-cache"
    cache.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache))
    cold = measure_cold_route(
        "fused-reference",
        config=BenchmarkCaseConfig("h1", 17, 1),
    )
    assert cold.effective_backend == "numba"


@pytest.mark.parametrize("route", ("scipy-vectorized", "public", "direct-typed", "dask-auto-leaf"))
@pytest.mark.parametrize("edges", (1, 4, 8))
def test_spatial_bench_fused_path_007_rss_subprocess_preserves_case_configuration(
    edges: int,
    route: str,
) -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_007_rss_subprocess_preserves_case_configuration."""
    pytest.importorskip("psutil", exc_type=ImportError)
    config = BenchmarkCaseConfig("h1", 17, edges)
    rss = measure_rss(route, config=config)
    assert rss.config == config
    if route == "dask-auto-leaf":
        expected = "numba" if importlib.util.find_spec("numba") is not None else "scipy"
        assert rss.effective_backend == expected


def test_spatial_bench_fused_path_008_scipy_main_does_not_require_numba(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_008_scipy_main_does_not_require_numba."""
    installed_version = benchmark_protocol.version

    def version_or_missing(name: str) -> str:
        if name in {"numba", "llvmlite"}:
            raise benchmark_protocol.PackageNotFoundError(name)
        return installed_version(name)

    def fail_numba_import(_owner: str) -> object:
        raise AssertionError("SciPy-only reporting attempted to import Numba")

    monkeypatch.setattr(benchmark_protocol, "version", version_or_missing)
    monkeypatch.setattr(benchmark_protocol, "require_numba", fail_numba_import)
    result = benchmark_main(
        (
            "--cases",
            "h0",
            "--sizes",
            "17",
            "--edges",
            "1",
            "--routes",
            "scipy-vectorized",
            "--warmups",
            "0",
            "--repeats",
            "1",
        )
    )
    output = capsys.readouterr().out
    assert result == 0
    assert "numba=not-installed" in output
    assert "llvmlite=not-installed" in output
    assert "numba_runtime=not-selected" in output


def test_spatial_bench_fused_path_006_rss_worker_failure_is_bounded_and_owned() -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_006_rss_worker_failure_is_bounded_and_owned."""
    pytest.importorskip("psutil", exc_type=ImportError)
    before = {process.pid for process in mp.active_children()}
    config = BenchmarkCaseConfig("h1", 17, 1)
    with pytest.raises(RuntimeError, match=r"benchmarks\.spatial_paths\.rss: worker ValueError: unknown benchmark route"):
        measure_rss("unknown", config=config)
    assert {process.pid for process in mp.active_children()} <= before


def test_spatial_bench_fused_path_009_materializes_shared_dask_graph_once() -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_009_materializes_shared_dask_graph_once."""
    da = pytest.importorskip("dask.array")
    import xarray as xr
    from dask import delayed
    from dask.callbacks import Callback

    calls: list[str] = []

    @delayed
    def shared_values() -> np.ndarray:
        calls.append("load")
        return np.arange(12, dtype=np.float64).reshape(3, 4)

    shared = da.from_delayed(shared_values(), shape=(3, 4), dtype=np.float64)
    dataset = xr.Dataset(
        {
            "position": (("query", "axis"), shared[:, :3]),
            "rotation": (("query", "quat"), shared),
        }
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        eager = materialize_route_result(dataset, synchronous=True)
    task_count = len(tasks)
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        np.testing.assert_array_equal(eager["rotation"], np.arange(12).reshape(3, 4))
    assert calls == ["load"]
    assert task_count > 0
    assert len(tasks) == task_count
    assert all(value.chunks is None for value in eager.data_vars.values())


def test_spatial_bench_fused_path_010_reports_automatic_leaf_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_010_reports_automatic_leaf_backend."""
    import benchmarks._spatial_path_execution_routes as routes

    monkeypatch.setattr(routes, "_numba_available", lambda: True)
    assert route_effective_backend("dask-auto-leaf") == "numba"
    monkeypatch.setattr(routes, "_numba_available", lambda: False)
    assert route_effective_backend("dask-auto-leaf") == "scipy"


def test_spatial_bench_fused_path_010_warm_and_cold_reports_confirmed_backend(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Automatic Dask timing and cold workers report their selected leaf."""
    expected = "numba" if importlib.util.find_spec("numba") is not None else "scipy"
    assert benchmark_main(
        (
            "--cases",
            "h1",
            "--sizes",
            "17",
            "--edges",
            "1",
            "--routes",
            "dask-auto-leaf",
            "--warmups",
            "0",
            "--repeats",
            "1",
        )
    ) == 0
    assert f"route=dask-auto-leaf; effective_backend={expected}" in capsys.readouterr().out
    cold = measure_cold_route(
        "dask-auto-leaf",
        config=BenchmarkCaseConfig("h1", 17, 1),
    )
    assert cold.effective_backend == expected


@pytest.mark.parametrize("case", ("h0", "h1"))
def test_spatial_bench_fused_path_011_single_edge_named_component_order(case, capsys) -> None:
    """ID: SPATIAL_BENCH_FUSED_PATH_011_single_edge_named_component_order."""
    assert benchmark_main((
        "--cases", case, "--sizes", "17", "--edges", "1",
        "--routes", "public", "direct-typed", "dask-auto-leaf",
        "--warmups", "0", "--repeats", "1",
    )) == 0
    assert f"parity=ok; case={case}; query=17; comparators=3" in capsys.readouterr().out
