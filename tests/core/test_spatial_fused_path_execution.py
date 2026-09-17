from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from benchmarks._spatial_path_execution_routes import (
    _generic_public_route,
    _public_case,
    bounded_scipy_stream,
)
from benchmarks.bench_spatial_fused_temporal_paths import (
    frozen_fixture,
    prepare_fixture,
    scipy_direct_path,
)
from tal.frames import FrameGraph
from tal.spatial import Pose, solve_pose_path_transform


def _represented_matrices(value: Pose) -> np.ndarray:
    quaternion = value.as_dataset(copy="none")["rotation"].data
    return SciRotation.from_quat(np.asarray(quaternion)).as_matrix()


@pytest.mark.parametrize(
    ("left", "right", "midpoint"),
    (
        (1.0e6, np.nextafter(-1.0e6, 0.0), 5.820766091346741e-11),
        (1.0e16, -1.0e16 + 2.0, 1.0),
        (1.0e308, -1.0e308, 0.0),
    ),
)
@pytest.mark.parametrize("query_size", (3, 100_000))
def test_spatial_core_fused_translation_uses_weighted_endpoint_parity(
    left: float,
    right: float,
    midpoint: float,
    query_size: int,
) -> None:
    """ID: SPATIAL_HARD_FUSED_EXECUTION_002_weighted_translation_parity."""
    pytest.importorskip("numba")
    query = np.full(query_size, 1.0 / 256.0)
    query[0], query[-1] = 0.0, 1.0 / 128.0
    fixture = frozen_fixture("h0", query_size=query_size, edges=1)
    translation = fixture.translation.copy()
    translation[0, 0, 0], translation[0, 1, 0] = left, right
    fixture = replace(fixture, query=query, translation=translation)
    graph, providers = _public_case(fixture)
    source = providers[0].as_dataset(copy="deep")

    actual = solve_pose_path_transform(fixture.source, fixture.destination, graph=graph, query=query)
    generic = _generic_public_route((fixture, graph, providers))
    streamed_t, _ = bounded_scipy_stream(prepare_fixture(fixture))
    actual_t = actual.as_dataset(copy="none")["position"].data
    generic_t = generic.as_dataset(copy="none")["position"].data
    np.testing.assert_array_equal(actual_t, generic_t)
    np.testing.assert_array_equal(actual_t, streamed_t)
    assert actual_t[0, 0] == left
    assert actual_t[1, 0] == midpoint
    assert actual_t[-1, 0] == right
    assert np.isfinite(actual_t).all()
    xr.testing.assert_identical(providers[0].as_dataset(copy="none"), source)


@pytest.mark.parametrize("case", ("h0", "h1"))
@pytest.mark.parametrize("edges", (1, 4, 8))
def test_spatial_core_fused_path_execution_matches_independent_reference(case: str, edges: int) -> None:
    """ID: SPATIAL_CORE_FUSED_EXECUTION_001_public_numeric_and_topology_parity."""
    pytest.importorskip("numba")
    fixture = frozen_fixture(case, query_size=17, edges=edges)
    graph, providers = _public_case(fixture)
    sources = tuple(provider.as_dataset(copy="deep") for provider in providers)
    result = solve_pose_path_transform(fixture.source, fixture.destination, graph=graph, query=fixture.query)
    generic = _generic_public_route((fixture, graph, providers))
    expected_t, expected_q = scipy_direct_path(fixture)

    actual_ds = result.as_dataset(copy="none")
    generic_ds = generic.as_dataset(copy="none")
    xr.testing.assert_identical(
        actual_ds.drop_vars(("position", "rotation")),
        generic_ds.drop_vars(("position", "rotation")),
    )
    assert tuple(actual_ds.data_vars) == tuple(generic_ds.data_vars)
    assert result.graph is graph
    np.testing.assert_allclose(actual_ds["position"].data, expected_t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_represented_matrices(result), SciRotation.from_quat(expected_q).as_matrix(), rtol=1e-12, atol=1e-12)
    for provider, source in zip(providers, sources, strict=True):
        xr.testing.assert_identical(provider.as_dataset(copy="none"), source)


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_spatial_core_fused_path_principal_arc_uses_scipy(sign: float) -> None:
    """ID: SPATIAL_CORE_FUSED_EXECUTION_002_scipy_principal_arc."""
    pytest.importorskip("numba")
    from tests.core.test_spatial_path_execution import _edge_pose

    left = np.asarray((1.0, 1.0, 1.0, 3.0)) / np.sqrt(12.0)
    right = sign * np.asarray((-1.0, 1.0, -3.0, 1.0)) / np.sqrt(12.0)
    source = _edge_pose(0).as_dataset(copy="deep")
    quaternion = np.repeat(right[None, :], source.sizes["sample"], axis=0)
    quaternion[0] = left
    source["rotation"] = source["rotation"].copy(data=quaternion)
    graph = FrameGraph()
    Pose(source, parent="f0", child="f1", graph=graph).register()
    query = np.asarray((0.0, 0.05, 0.125, 0.2, 0.25))
    result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    expected = Slerp(
        np.asarray((0.0, 0.25)),
        SciRotation.from_quat(np.stack((left, right))),
    )(query).as_matrix()
    np.testing.assert_allclose(_represented_matrices(result), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("reverse", (False, True))
def test_spatial_hard_fused_path_failure_precedence_is_provider_major(reverse: bool) -> None:
    """ID: SPATIAL_HARD_FUSED_EXECUTION_001_provider_major_failure_order."""
    pytest.importorskip("numba")
    from tests.core.test_spatial_path_execution import _edge_pose

    graph = FrameGraph()
    invalid_samples = (1, 3) if reverse else (3, 1)
    for edge, invalid_sample in enumerate(invalid_samples):
        source = _edge_pose(edge).as_dataset(copy="deep")
        quaternion = source["rotation"].data.copy()
        quaternion[invalid_sample] = 0.0
        source["rotation"] = source["rotation"].copy(data=quaternion)
        Pose(source, parent=f"f{edge}", child=f"f{edge + 1}", graph=graph).register()

    with pytest.raises(ValueError, match=r"^spatial\.path_solve\.pose: .*quaternion norm") as failure:
        solve_pose_path_transform("f2", "f0", graph=graph, query=np.asarray((0.25, 0.75)))
    assert isinstance(failure.value.__cause__, ValueError)
    expected_query = 1 if reverse else 0
    assert f"edge 0, query {expected_query}" in str(failure.value.__cause__)


def test_spatial_core_fused_path_empty_query_keeps_generic_topology() -> None:
    """ID: SPATIAL_CORE_FUSED_EXECUTION_003_empty_fallback_topology."""
    pytest.importorskip("numba")
    fixture = frozen_fixture("h1", query_size=0, edges=1)
    graph, providers = _public_case(fixture)
    actual = solve_pose_path_transform(fixture.source, fixture.destination, graph=graph, query=fixture.query)
    generic = _generic_public_route((fixture, graph, providers))
    xr.testing.assert_identical(actual.as_dataset(copy="none"), generic.as_dataset(copy="none"))


def test_spatial_core_fused_path_explicit_resolver_runs_once() -> None:
    """ID: SPATIAL_CORE_FUSED_EXECUTION_004_explicit_resolver_once."""
    pytest.importorskip("numba")
    fixture = frozen_fixture("h1", query_size=9, edges=1)
    graph, providers = _public_case(fixture)
    calls: list[tuple[str, str]] = []

    def resolver(child, parent):
        calls.append((child.id, parent.id))
        return providers[0]

    result = solve_pose_path_transform(
        fixture.source,
        fixture.destination,
        graph=graph,
        query=fixture.query,
        edge_pose_fn=resolver,
    )
    expected_t, expected_q = scipy_direct_path(fixture)
    assert calls == [(fixture.relations[0][1], fixture.relations[0][0])]
    assert result.graph is graph
    np.testing.assert_allclose(result.as_dataset(copy="none")["position"].data, expected_t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_represented_matrices(result), SciRotation.from_quat(expected_q).as_matrix(), rtol=1e-12, atol=1e-12)


def test_spatial_core_fused_path_parallel_size_matches_reference() -> None:
    """ID: SPATIAL_CORE_FUSED_EXECUTION_005_large_query_parallel_parity."""
    pytest.importorskip("numba")
    fixture = frozen_fixture("h1", query_size=100_000, edges=4)
    graph, _ = _public_case(fixture)
    result = solve_pose_path_transform(fixture.source, fixture.destination, graph=graph, query=fixture.query)
    expected_t, expected_q = scipy_direct_path(fixture)
    np.testing.assert_allclose(result.as_dataset(copy="none")["position"].data, expected_t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_represented_matrices(result), SciRotation.from_quat(expected_q).as_matrix(), rtol=1e-12, atol=1e-12)


def test_spatial_perf_fused_path_lazy_provider_never_enters_eager_route() -> None:
    """ID: SPATIAL_PERF_FUSED_EXECUTION_001_lazy_provider_stays_lazy."""
    from dask.callbacks import Callback

    from tests.core.test_spatial_path_execution import _registered_path

    graph, providers = _registered_path(1, lazy=True)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=np.asarray((0.25, 0.75)))
    assert tasks == []
    assert all(value.chunks is not None for value in result.as_dataset(copy="none").data_vars.values())
    assert result.graph is graph
    assert providers[0].as_dataset(copy="none")["rotation"].chunks is not None
