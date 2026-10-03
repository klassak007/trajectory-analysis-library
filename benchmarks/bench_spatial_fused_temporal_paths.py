"""Benchmark fused temporal path execution candidates.

This is a manual, non-CI benchmark.  It freezes the H0/H1 fixtures, reports
timing and process RSS evidence, and keeps benchmark-only numerical execution
separate from TAL's public routing.  Wall-clock and memory results never decide
pytest success.  ``scipy-direct-path`` is an opt-in, complete NumPy/SciPy path
comparator; it is excluded from the default routes because it is expensive at
one million queries.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import lru_cache, partial
from time import perf_counter

import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from benchmarks._spatial_path_benchmark_protocol import (
    MeasuredRoute,
    RssRoutePlan,
    measure_cold_route,
    measure_route,
    measure_routes,
    measure_rss,
    numba_environment_line,
    print_timing_result,
    runtime_environment_lines,
)
from benchmarks._spatial_path_execution_routes import (
    bounded_scipy_stream,
    dask_graph_evidence,
    materialize_route_result,
    route_effective_backend,
    typed_route_operation,
)
from benchmarks._spatial_path_fixtures import (
    BenchmarkCaseConfig,
    ColdTimingResult,
    FrozenPathFixture,
    PreparedFixture,
    RssResult,
    TimingResult,
    frozen_fixture,
)
from tal.core.param_engine import ParamMapOptions, build_param_map
from tal.spatial import Pose
from tal.spatial.kernels import fixed_size_primitives as _fixed
from tal.spatial.kernels import higher_order_interp_primitives as _interp
from tal.spatial.kernels.rotation_interp_numba_backends import (
    _compile_slerp_dependencies,
    slerp_quat_numba,
)
from tal.utils.numba_support import njit_kernel, require_numba

_DEFAULT_QUERY_SIZE = 1_000_000
_REFERENCE_JITTED = False


def _core_map(fixture: FrozenPathFixture):
    parameter = xr.DataArray(fixture.parameter, dims="sample")
    query = xr.DataArray(fixture.query, dims="query")
    return build_param_map(
        param=parameter,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear", duplicate_policy="invalid"),
    )


def prepare_fixture(fixture: FrozenPathFixture) -> PreparedFixture:
    """Obtain benchmark decisions from the core parameter owner."""
    mapping = _core_map(fixture)
    return PreparedFixture(
        fixture,
        np.asarray(mapping.i0.data),
        np.asarray(mapping.i1.data),
        np.asarray(mapping.alpha.data),
        np.asarray(mapping.valid.data),
    )


def _jit_reference_helpers(numba) -> None:
    global _REFERENCE_JITTED, _compose_pose, _inverse_pose, _orient_pose, _slerp_unit
    if _REFERENCE_JITTED:
        return
    _, _, _interp._shared_slerp_quat = _compile_slerp_dependencies()
    slerp_unit = getattr(_interp.slerp_unit, "py_func", _interp.slerp_unit)
    _slerp_unit = njit_kernel(numba, slerp_unit)
    _compose_pose = njit_kernel(numba, _fixed.compose_pose)
    _inverse_pose = njit_kernel(numba, _fixed.inverse_pose)
    _orient_pose = njit_kernel(numba, _orient_pose_impl)
    _REFERENCE_JITTED = True


def _orient_pose_impl(translation, quaternion, direction):
    if direction < 0:
        return _inverse_pose(translation, quaternion)
    return translation, quaternion


def _fused_pose_impl(translation, quaternion, i0, i1, alpha, valid, directions):
    query_size = alpha.size
    out_t = np.empty((query_size, 3), dtype=translation.dtype)
    out_q = np.empty((query_size, 4), dtype=quaternion.dtype)
    for query_index in range(query_size):
        result_t = (0.0, 0.0, 0.0)
        result_q = (0.0, 0.0, 0.0, 1.0)
        if not valid[query_index]:
            out_t[query_index].fill(np.nan)
            out_q[query_index].fill(np.nan)
            continue
        for edge in range(translation.shape[0]):
            left = i0[query_index]
            right = i1[query_index]
            fraction = alpha[query_index]
            edge_t = (
                translation[edge, left, 0]
                + fraction * (translation[edge, right, 0] - translation[edge, left, 0]),
                translation[edge, left, 1]
                + fraction * (translation[edge, right, 1] - translation[edge, left, 1]),
                translation[edge, left, 2]
                + fraction * (translation[edge, right, 2] - translation[edge, left, 2]),
            )
            left_q = (
                quaternion[edge, left, 0],
                quaternion[edge, left, 1],
                quaternion[edge, left, 2],
                quaternion[edge, left, 3],
            )
            right_q = (
                quaternion[edge, right, 0],
                quaternion[edge, right, 1],
                quaternion[edge, right, 2],
                quaternion[edge, right, 3],
            )
            edge_q = _slerp_unit(left_q, right_q, fraction)
            edge_t, edge_q = _orient_pose(edge_t, edge_q, directions[edge])
            result_t, result_q = _compose_pose(result_t, result_q, edge_t, edge_q)
        for axis in range(3):
            out_t[query_index, axis] = result_t[axis]
        for axis in range(4):
            out_q[query_index, axis] = result_q[axis]
    return out_t, out_q


@lru_cache(maxsize=1)
def _compiled_fused_reference():
    numba = require_numba("benchmarks.spatial_paths.fused_reference")
    _jit_reference_helpers(numba)
    return njit_kernel(numba, _fused_pose_impl)


def direct_fused_reference(prepared: PreparedFixture) -> tuple[np.ndarray, np.ndarray]:
    fixture = prepared.fixture
    directions = np.asarray(fixture.directions, dtype=np.int8)
    return _compiled_fused_reference()(
        fixture.translation,
        fixture.quaternion,
        prepared.i0,
        prepared.i1,
        prepared.alpha,
        prepared.valid,
        directions,
    )


def _gather_rows(values: np.ndarray, prepared: PreparedFixture) -> tuple[np.ndarray, np.ndarray]:
    edge = np.arange(values.shape[0])[:, None]
    return values[edge, prepared.i0[None, :]], values[edge, prepared.i1[None, :]]


def compiled_leaf(prepared: PreparedFixture) -> tuple[np.ndarray, np.ndarray]:
    fixture = prepared.fixture
    t0, t1 = _gather_rows(fixture.translation, prepared)
    q0, q1 = _gather_rows(fixture.quaternion, prepared)
    alpha = np.broadcast_to(prepared.alpha, q0.shape[:-1])
    valid = np.broadcast_to(prepared.valid, q0.shape[:-1])
    translation = t0 + alpha[..., None] * (t1 - t0)
    quaternion = slerp_quat_numba(q0, q1, alpha, valid, owner="benchmarks.spatial_paths")
    return translation, quaternion


def _fold_interpolated_impl(translation, quaternion, valid, directions):
    query_size = valid.size
    out_t = np.empty((query_size, 3), dtype=translation.dtype)
    out_q = np.empty((query_size, 4), dtype=quaternion.dtype)
    for query_index in range(query_size):
        result_t = (0.0, 0.0, 0.0)
        result_q = (0.0, 0.0, 0.0, 1.0)
        if not valid[query_index]:
            out_t[query_index].fill(np.nan)
            out_q[query_index].fill(np.nan)
            continue
        for edge in range(translation.shape[0]):
            edge_t = (
                translation[edge, query_index, 0],
                translation[edge, query_index, 1],
                translation[edge, query_index, 2],
            )
            edge_q = (
                quaternion[edge, query_index, 0],
                quaternion[edge, query_index, 1],
                quaternion[edge, query_index, 2],
                quaternion[edge, query_index, 3],
            )
            edge_t, edge_q = _orient_pose(edge_t, edge_q, directions[edge])
            result_t, result_q = _compose_pose(result_t, result_q, edge_t, edge_q)
        for axis in range(3):
            out_t[query_index, axis] = result_t[axis]
        for axis in range(4):
            out_q[query_index, axis] = result_q[axis]
    return out_t, out_q


@lru_cache(maxsize=1)
def _compiled_fold():
    numba = require_numba("benchmarks.spatial_paths.compiled_fold")
    _jit_reference_helpers(numba)
    return njit_kernel(numba, _fold_interpolated_impl)


def compiled_interpolate_fold(prepared: PreparedFixture) -> tuple[np.ndarray, np.ndarray]:
    translation, quaternion = compiled_leaf(prepared)
    directions = np.asarray(prepared.fixture.directions, dtype=np.int8)
    return _compiled_fold()(translation, quaternion, prepared.valid, directions)


def _interpolated_translation(prepared: PreparedFixture) -> np.ndarray:
    t0, t1 = _gather_rows(prepared.fixture.translation, prepared)
    alpha = prepared.alpha[None, :, None]
    return t0 + alpha * (t1 - t0)


def vectorized_scipy(prepared: PreparedFixture) -> tuple[np.ndarray, np.ndarray]:
    fixture = prepared.fixture
    translation = _interpolated_translation(prepared)
    quaternion = np.empty((fixture.quaternion.shape[0], fixture.query.size, 4))
    for edge in range(fixture.quaternion.shape[0]):
        rotations = SciRotation.from_quat(fixture.quaternion[edge])
        quaternion[edge] = Slerp(fixture.parameter, rotations)(fixture.query).as_quat()
    return translation, quaternion


def synthetic_stacked_scipy(prepared: PreparedFixture) -> tuple[np.ndarray, np.ndarray]:
    fixture = prepared.fixture
    edge_count = fixture.quaternion.shape[0]
    offsets = np.arange(edge_count, dtype=np.float64) * 2.0
    native = np.concatenate(tuple(fixture.parameter + offset for offset in offsets))
    query = np.concatenate(tuple(fixture.query + offset for offset in offsets))
    rotations = SciRotation.from_quat(fixture.quaternion.reshape(-1, 4))
    quaternion = Slerp(native, rotations)(query).as_quat().reshape(edge_count, -1, 4)
    return _interpolated_translation(prepared), quaternion


def _scipy_edge_values(fixture: FrozenPathFixture, edge: int) -> tuple[np.ndarray, SciRotation]:
    translation = np.column_stack(
        tuple(np.interp(fixture.query, fixture.parameter, fixture.translation[edge, :, axis]) for axis in range(3))
    )
    rotations = SciRotation.from_quat(fixture.quaternion[edge])
    return translation, Slerp(fixture.parameter, rotations)(fixture.query)


def scipy_direct_path(fixture: FrozenPathFixture) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate and compose a complete path using only NumPy and SciPy."""
    result_t = np.zeros((fixture.query.size, 3), dtype=np.float64)
    identity = np.zeros((fixture.query.size, 4), dtype=np.float64)
    identity[:, 3] = 1.0
    result_r = SciRotation.from_quat(identity)
    for edge, direction in enumerate(fixture.directions):
        edge_t, edge_r = _scipy_edge_values(fixture, edge)
        if direction < 0:
            edge_r = edge_r.inv()
            edge_t = -edge_r.apply(edge_t)
        result_t = edge_r.apply(result_t) + edge_t
        result_r = edge_r * result_r
    return result_t, result_r.as_quat()


def _scipy_direct_prepared(prepared: PreparedFixture) -> tuple[np.ndarray, np.ndarray]:
    return scipy_direct_path(prepared.fixture)


_RAW_ROUTES = {
    "scipy-direct-path": _scipy_direct_prepared,
    "scipy-streaming": bounded_scipy_stream,
    "compiled-leaf": compiled_leaf,
    "compiled-fold": compiled_interpolate_fold,
    "fused-reference": direct_fused_reference,
    "scipy-vectorized": vectorized_scipy,
    "scipy-stacked": synthetic_stacked_scipy,
}


def _route_operation(name: str, prepared: PreparedFixture):
    route = _RAW_ROUTES.get(name)
    if route is not None:
        return partial(route, prepared)
    return typed_route_operation(name, prepared.fixture)


def _mapped_route_result(name: str, fixture: FrozenPathFixture):
    return _RAW_ROUTES[name](prepare_fixture(fixture))


def _prepare_rss_route_plan(name: str, fixture: FrozenPathFixture) -> RssRoutePlan:
    """Prepare only topology that belongs outside the sampled route."""
    config = BenchmarkCaseConfig(fixture.name, fixture.query.size, fixture.translation.shape[0])
    materialize = partial(materialize_route_result, synchronous=name == "dask-auto-leaf")
    backend = route_effective_backend(name)
    if name == "scipy-direct-path":
        operation = partial(scipy_direct_path, fixture)
    elif name in _RAW_ROUTES:
        operation = partial(_mapped_route_result, name, fixture)
    else:
        operation = typed_route_operation(name, fixture)
    return RssRoutePlan(config, operation, materialize, backend)


def _assert_quaternion_parity(actual: np.ndarray, expected: np.ndarray) -> None:
    actual_matrix = SciRotation.from_quat(actual.reshape(-1, 4)).as_matrix()
    expected_matrix = SciRotation.from_quat(expected.reshape(-1, 4)).as_matrix()
    np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=1.0e-12, atol=1.0e-12)


def _pose_arrays(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(dataset["position"].transpose(..., "axis").data),
        np.asarray(dataset["rotation"].transpose(..., "quat").data),
    )


def validate_prepared_fixture(prepared: PreparedFixture) -> None:
    """Validate every maintained comparator outside timed regions."""
    names = (
        "public",
        "promoted-generic",
        "direct-typed",
        "compiled-leaf",
        "compiled-fold",
        "scipy-vectorized",
        "scipy-stacked",
        "fused-reference",
    )
    for route in _measured_routes(prepared, names):
        result: object | None = None
        try:
            result = route.operation()
            result = route.materialize(result)
            route.validate(result)
        finally:
            result = None


def _route_values(result: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(result, Pose):
        return _pose_arrays(result.as_dataset(copy="none"))
    if isinstance(result, xr.Dataset):
        return _pose_arrays(result)
    if isinstance(result, tuple) and len(result) == 2:
        return np.asarray(result[0]), np.asarray(result[1])
    raise TypeError(f"benchmark route returned unsupported result {type(result).__name__}")


def _validate_route_result(result: object, *, expected: tuple[np.ndarray, np.ndarray]) -> None:
    actual_t, actual_q = _route_values(result)
    expected_t, expected_q = expected
    if actual_t.shape != expected_t.shape or actual_q.shape != expected_q.shape:
        raise AssertionError("benchmark route result shape differs from its independent reference")
    np.testing.assert_allclose(actual_t, expected_t, rtol=1.0e-12, atol=1.0e-12)
    _assert_quaternion_parity(actual_q, expected_q)


def _route_expectations(
    prepared: PreparedFixture,
    routes: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    expected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if set(routes) & {"compiled-leaf", "scipy-stacked"}:
        edge_reference = vectorized_scipy(prepared)
        expected.update({name: edge_reference for name in ("compiled-leaf", "scipy-stacked")})
    if "scipy-vectorized" in routes:
        expected["scipy-vectorized"] = synthetic_stacked_scipy(prepared)
    path_routes = {
        "compiled-fold",
        "dask-auto-leaf",
        "fused-reference",
        "promoted-generic",
        "public",
        "direct-typed",
        "scipy-streaming",
    }
    if set(routes) & path_routes:
        path_reference = scipy_direct_path(prepared.fixture)
        expected.update({name: path_reference for name in path_routes})
    if "scipy-direct-path" in routes:
        expected["scipy-direct-path"] = bounded_scipy_stream(prepared)
    return expected


def _measured_routes(prepared: PreparedFixture, routes: Sequence[str]) -> tuple[MeasuredRoute, ...]:
    operations = {name: _route_operation(name, prepared) for name in routes}
    expected = _route_expectations(prepared, routes)
    return tuple(
        MeasuredRoute(
            name,
            operation,
            partial(materialize_route_result, synchronous=name == "dask-auto-leaf"),
            lambda result, reference=expected[name]: _validate_route_result(result, expected=reference),
            route_effective_backend(name),
        )
        for name, operation in operations.items()
    )


def _selected_routes(args: argparse.Namespace) -> tuple[str, ...]:
    selected = args.cold_child or args.rss_route or args.cold_route
    return (selected,) if selected is not None else tuple(args.routes)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=("h0", "h1"), default=("h0", "h1"))
    parser.add_argument("--sizes", nargs="+", type=int, default=(1_000, 100_000))
    parser.add_argument("--edges", type=int, default=8)
    parser.add_argument(
        "--routes",
        nargs="+",
        help="benchmark routes; scipy-direct-path is an opt-in complete NumPy/SciPy comparator",
        default=(
            "public",
            "direct-typed",
            "promoted-generic",
            "compiled-leaf",
            "compiled-fold",
            "scipy-vectorized",
            "scipy-stacked",
            "scipy-streaming",
            "dask-auto-leaf",
            "fused-reference",
        ),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rss-route")
    parser.add_argument("--rss-size", type=int, default=_DEFAULT_QUERY_SIZE)
    parser.add_argument("--cold-route")
    parser.add_argument("--cold-child", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _run_timing_case(
    args: argparse.Namespace,
    *,
    case_name: str,
    size: int,
) -> None:
    fixture = frozen_fixture(case_name, query_size=size, edges=args.edges)
    started = perf_counter()
    prepared = prepare_fixture(fixture)
    map_elapsed = perf_counter() - started
    map_bytes = sum(value.nbytes for value in (prepared.i0, prepared.i1, prepared.alpha, prepared.valid))
    print(
        f"stage=core-map; case={case_name}; query={size:,}; "
        f"elapsed={map_elapsed:.6f}s; maps=1; bytes={map_bytes}"
    )
    if "dask-auto-leaf" in args.routes:
        tasks, largest, backend = dask_graph_evidence(fixture)
        print(
            f"dask-evidence; scheduler=synchronous; workers=1; "
            f"effective_backend={backend}; tasks={tasks}; max_query_chunk={largest}"
        )
    routes = _measured_routes(prepared, args.routes)
    results = measure_routes(routes, warmups=args.warmups, repeats=args.repeats)
    print(
        f"parity=ok; case={case_name}; query={size:,}; comparators={len(routes)}; "
        f"validated_runs={len(routes) * (args.warmups + args.repeats)}"
    )
    for result in results:
        print_timing_result(result, case=case_name, size=size, edges=args.edges)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for line in runtime_environment_lines():
        print(line)
    selected_numba = any(route_effective_backend(name) == "numba" for name in _selected_routes(args))
    print(numba_environment_line(required=selected_numba))
    print(f"warmups={args.warmups}; repeats={args.repeats}; timing excludes fixture construction")
    if args.cold_child:
        fixture = frozen_fixture(args.cases[0], query_size=args.sizes[0], edges=args.edges)
        plan = _prepare_rss_route_plan(args.cold_child, fixture)
        plan.materialize(plan.operation())
        print(
            f"cold-child-complete; route={args.cold_child}; case={plan.config.case}; "
            f"query={plan.config.query_size}; edges={plan.config.edges}; "
            f"effective_backend={plan.effective_backend}"
        )
        return 0
    if args.rss_route:
        config = BenchmarkCaseConfig(args.cases[0], args.rss_size, args.edges)
        result = measure_rss(args.rss_route, config=config)
        print(
            f"rss route={result.route}; case={result.config.case}; query={result.config.query_size:,}; "
            f"edges={result.config.edges}; effective_backend={result.effective_backend}; "
            f"baseline={result.baseline}; peak={result.peak}; "
            f"increment={result.increment}; samples={result.sample_count}; interval_ms=1"
        )
        return 0
    if args.cold_route:
        config = BenchmarkCaseConfig(args.cases[0], args.sizes[0], args.edges)
        result = measure_cold_route(args.cold_route, config=config)
        print(
            f"cold route={result.route}; case={result.config.case}; query={result.config.query_size:,}; "
            f"edges={result.config.edges}; effective_backend={result.effective_backend}; "
            f"elapsed={result.elapsed:.6f}s"
        )
        return 0
    for case_name in args.cases:
        for size in sorted(args.sizes):
            _run_timing_case(args, case_name=case_name, size=size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkCaseConfig",
    "ColdTimingResult",
    "FrozenPathFixture",
    "PreparedFixture",
    "RssResult",
    "TimingResult",
    "direct_fused_reference",
    "frozen_fixture",
    "measure_cold_route",
    "measure_route",
    "measure_routes",
    "measure_rss",
    "prepare_fixture",
    "validate_prepared_fixture",
]
