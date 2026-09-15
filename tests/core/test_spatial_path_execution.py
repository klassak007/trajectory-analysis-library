from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation

from tal.core import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import (
    Pose,
    Position,
    Rotation,
    solve_pose_path_transform,
    solve_rotation_path_transform,
)
from tal.spatial.kernels.fixed_size_backends import (
    SPATIAL_FIXED_BACKEND_NUMBA,
    SPATIAL_FIXED_BACKEND_SCIPY,
    quat_compose_block_backend,
)
from tal.spatial.kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_NUMBA,
    ROTATION_INTERP_BACKEND_SCIPY,
    slerp_quat_backend,
)

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


class _UnevaluatedPathQueryTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size: int, calls: list[str]) -> None:
        self.calls = calls
        super().__init__(("trial",), {"trial": size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("forward")
        return {"trial": dim_positions["trial"] + 0.5}

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("reverse")
        return {"trial": coord_labels["trial"] - 0.5}

    def equals(self, other: object, **kwargs: object) -> bool:
        _ = kwargs
        return (
            isinstance(other, _UnevaluatedPathQueryTransform)
            and self.dim_size == other.dim_size
        )


class _ExceptionalPathCoordinate:
    def __eq__(self, other: object) -> bool:
        _ = other
        raise OSError("comparison is intentionally unavailable")


def _path_query_index(size: int, kind: str, calls: list[str]) -> xr.Index:
    if kind == "range":
        return xr.indexes.RangeIndex.arange(size, dim="trial")
    return xr.indexes.CoordinateTransformIndex(
        _UnevaluatedPathQueryTransform(size, calls)
    )


def _edge_pose(edge: int, *, lazy: bool = False) -> Pose:
    sample = np.linspace(0.0, 1.0, 5)
    translation = np.column_stack((sample * (edge + 1), np.sin(sample + edge), np.cos(sample - edge)))
    angles = np.deg2rad(sample * (20.0 + edge))
    rotations = SciRotation.from_rotvec(np.column_stack((np.zeros((5, 2)), angles))).as_quat()
    if lazy:
        da = pytest.importorskip("dask.array")
        translation = da.from_array(translation, chunks=(5, 3))
        rotations = da.from_array(rotations, chunks=(5, 4))
    coords = {"sample": np.arange(5), "time": ("sample", sample)}
    position = AnalysisObject.from_data(
        xr.DataArray(translation, dims=("sample", "axis"), coords={**coords, "axis": list(_XYZ)}, name="position"),
        sequence_dim="sample",
        core_dims=("axis",),
        param_coord="time",
    )
    rotation = AnalysisObject.from_data(
        xr.DataArray(rotations, dims=("sample", "quat"), coords={**coords, "quat": list(_QUAT)}, name="rotation"),
        sequence_dim="sample",
        core_dims=("quat",),
        param_coord="time",
    )
    return Pose.from_components(Rotation(rotation), Position(position))


def _registered_path(edges: int, *, lazy: bool = False) -> tuple[FrameGraph, tuple[Pose, ...]]:
    graph = FrameGraph()
    values: list[Pose] = []
    for edge in range(edges):
        value = Pose(
            _edge_pose(edge, lazy=lazy),
            parent=f"f{edge}",
            child=f"f{edge + 1}",
            graph=graph,
        )
        value.register()
        values.append(value)
    return graph, tuple(values)


@pytest.mark.parametrize("kind", ("range", "transform"))
@pytest.mark.parametrize("result_type", ("pose", "rotation"))
def test_spatial_core_path_block_index_001_direct_queries_preserve_native_batch_indexes(
    kind: str,
    result_type: str,
) -> None:
    """ID: SPATIAL_CORE_PATH_BLOCK_INDEX_001_direct_queries_preserve_native_batch_indexes."""
    size = 65_537
    calls: list[str] = []
    index = _path_query_index(size, kind, calls)
    query = xr.DataArray(
        np.full((size, 1), 0.5),
        dims=("trial", "when"),
        coords=xr.Coordinates.from_xindex(index),
    )
    graph, providers = _registered_path(1)
    source = providers[0].as_dataset(copy="deep")
    calls.clear()

    if result_type == "pose":
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    else:
        result = solve_rotation_path_transform("f1", "f0", graph=graph, query=query)
    dataset = result.as_dataset(copy="none")

    assert type(dataset.xindexes["trial"]) is type(query.xindexes["trial"])
    assert dataset.xindexes["trial"].equals(query.xindexes["trial"])
    assert dataset.sizes["trial"] == size
    assert dataset.sizes["query"] == 1
    assert calls == []
    xr.testing.assert_identical(providers[0].as_dataset(copy="none"), source)


def test_spatial_perf_path_block_index_001_lazy_direct_query_planning_executes_no_tasks() -> None:
    """ID: SPATIAL_PERF_PATH_BLOCK_INDEX_001_lazy_direct_query_planning_executes_no_tasks."""
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    size = 65_537
    index = xr.indexes.RangeIndex.arange(size, dim="trial")
    query = xr.DataArray(
        np.full((size, 1), 0.5),
        dims=("trial", "when"),
        coords=xr.Coordinates.from_xindex(index),
    )
    graph, _ = _registered_path(1, lazy=True)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=query)

    dataset = result.as_dataset(copy="none")
    assert tasks == []
    assert isinstance(dataset.xindexes["trial"], xr.indexes.RangeIndex)
    for value in dataset.data_vars.values():
        assert value.chunks is not None
        trial_chunks = value.chunks[value.get_axis_num("trial")]
        query_chunks = value.chunks[value.get_axis_num("query")]
        assert max(trial_chunks) * max(query_chunks) <= 65_536


def test_spatial_hard_prepared_equivalence_001_object_query_metadata_never_escapes() -> None:
    """ID: SPATIAL_HARD_PREPARED_EQUIVALENCE_001_object_query_metadata_never_escapes."""
    graph, _ = _registered_path(2)
    query = xr.DataArray(
        [0.25, 0.75],
        dims="when",
        coords={"marker": ("when", np.asarray([pd.NA, "known"], dtype=object))},
    )

    result = solve_pose_path_transform("f2", "f0", graph=graph, query=query)

    assert result.as_dataset(copy="none").sizes["query"] == 2
    assert result.graph is graph

    exceptional_graph = FrameGraph()
    for edge in range(2):
        dataset = _edge_pose(edge).as_dataset(copy="none").assign_coords(
            marker=(
                "sample",
                np.asarray([_ExceptionalPathCoordinate() for _ in range(5)], dtype=object),
            )
        )
        Pose(
            dataset,
            parent=f"f{edge}",
            child=f"f{edge + 1}",
            graph=exceptional_graph,
        ).register()

    exceptional = solve_pose_path_transform(
        "f2",
        "f0",
        graph=exceptional_graph,
        query=np.asarray([0.25, 0.75]),
    )

    assert exceptional.as_dataset(copy="none").sizes["query"] == 2
    assert exceptional.graph is exceptional_graph


@pytest.mark.parametrize("edges", (1, 4, 8))
def test_spatial_core_streaming_path_001_matches_generic_pose_execution(
    monkeypatch: pytest.MonkeyPatch,
    edges: int,
) -> None:
    """ID: SPATIAL_CORE_STREAMING_PATH_001_matches_generic_pose_execution."""
    import tal.spatial.ops.path_execution as execution

    graph, _ = _registered_path(edges)
    query = np.linspace(0.0, 1.0, 257)
    monkeypatch.setattr(execution, "_numba_available", lambda: False)
    streamed = solve_pose_path_transform(f"f{edges}", "f0", graph=graph, query=query)
    monkeypatch.setattr(execution, "_numba_available", lambda: True)
    generic = solve_pose_path_transform(f"f{edges}", "f0", graph=graph, query=query)
    xr.testing.assert_allclose(streamed.as_dataset(copy="none"), generic.as_dataset(copy="none"))
    assert streamed.graph is graph


def test_spatial_core_streaming_path_002_auxiliary_metadata_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_CORE_STREAMING_PATH_002_auxiliary_metadata_is_preserved."""
    import tal.spatial.ops.path_execution as execution

    graph = FrameGraph()
    for edge in range(2):
        dataset = _edge_pose(edge).as_dataset(copy="none")
        dataset = dataset.assign_coords({f"calibration_{edge}": float(edge)})
        provider = Pose(
            dataset,
            parent=f"f{edge}",
            child=f"f{edge + 1}",
            graph=graph,
        )
        provider.register()
    query = np.linspace(0.0, 1.0, 17)
    monkeypatch.setattr(execution, "_numba_available", lambda: False)
    fallback = solve_pose_path_transform("f2", "f0", graph=graph, query=query)
    monkeypatch.setattr(execution, "_numba_available", lambda: True)
    generic = solve_pose_path_transform("f2", "f0", graph=graph, query=query)
    xr.testing.assert_allclose(fallback.as_dataset(copy="none"), generic.as_dataset(copy="none"))
    assert fallback.as_dataset(copy="none").coords["calibration_0"] == 0.0
    assert fallback.as_dataset(copy="none").coords["calibration_1"] == 1.0


def test_spatial_core_streaming_path_003_unsupported_dtype_and_metadata_use_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_CORE_STREAMING_PATH_003_unsupported_dtype_and_metadata_use_generic."""
    import tal.spatial.ops.path_execution as execution

    graph = FrameGraph()
    dataset = _edge_pose(0).as_dataset(copy="deep")
    dataset["position"] = dataset["position"].astype(np.complex128) * (1.0 + 0.25j)
    dataset.attrs["source_note"] = "preserve"
    dataset["position"].attrs["component_note"] = "preserve"
    dataset["position"].encoding["source"] = "synthetic"
    provider = Pose(dataset, parent="f0", child="f1", graph=graph)
    provider.register()
    query = np.linspace(0.0, 1.0, 17)
    monkeypatch.setattr(execution, "_numba_available", lambda: True)
    expected = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    monkeypatch.setattr(execution, "_numba_available", lambda: False)
    monkeypatch.setattr(
        execution,
        "stream_pose_path_blocks",
        lambda *args, **kwargs: pytest.fail("unsupported provider entered streaming"),
    )
    actual = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    xr.testing.assert_identical(actual.as_dataset(copy="none"), expected.as_dataset(copy="none"))
    assert actual.as_dataset(copy="none")["position"].dtype == np.dtype(np.complex128)


def test_spatial_hard_streaming_path_002_failure_is_owned_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_HARD_STREAMING_PATH_002_failure_is_owned_without_retry."""
    import tal.spatial.ops.path_execution as execution

    graph = FrameGraph()
    dataset = _edge_pose(0).as_dataset(copy="deep")
    dataset["rotation"] = xr.zeros_like(dataset["rotation"])
    Pose(dataset, parent="f0", child="f1", graph=graph).register()
    calls = 0
    original = execution.stream_pose_path_blocks

    def tracked(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_numba_available", lambda: False)
    monkeypatch.setattr(execution, "stream_pose_path_blocks", tracked)
    with pytest.raises(ValueError, match=r"^spatial\.path_solve\.pose:") as error:
        solve_pose_path_transform("f1", "f0", graph=graph, query=np.asarray([0.25]))
    assert isinstance(error.value.__cause__, ValueError)
    assert calls == 1


def test_spatial_hard_streaming_path_001_inverse_direction_and_block_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_HARD_STREAMING_PATH_001_inverse_direction_and_block_boundary."""
    import tal.spatial.ops.path_execution as execution

    graph, _ = _registered_path(1)
    query = np.linspace(0.0, 1.0, 65_537)
    monkeypatch.setattr(execution, "_numba_available", lambda: False)
    forward = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    reverse = solve_pose_path_transform("f0", "f1", graph=graph, query=query)
    identity = forward.compose(reverse, validate=True).as_matrix(validate=True)
    matrix = identity.as_dataset(copy="none")["pose_matrix"]
    expected = np.broadcast_to(np.eye(4), matrix.shape)
    np.testing.assert_allclose(matrix, expected, atol=1.0e-11)


def test_spatial_perf_streaming_path_001_lazy_payload_stays_generic_and_unexecuted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_PERF_STREAMING_PATH_001_lazy_payload_stays_generic_and_unexecuted."""
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    import tal.spatial.ops.path_execution as execution

    graph, _ = _registered_path(1, lazy=True)
    monkeypatch.setattr(execution, "_numba_available", lambda: False)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = solve_pose_path_transform("f1", "f0", graph=graph, query=np.linspace(0.0, 1.0, 65_537))
    assert tasks == []
    dataset = result.as_dataset(copy="none")
    assert all(value.chunks is not None for value in dataset.data_vars.values())
    for value in dataset.data_vars.values():
        query_axis = value.get_axis_num("query")
        assert max(value.chunks[query_axis]) <= 65_536


def test_spatial_core_leaf_selection_001_implicit_no_numba_is_exact_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_CORE_LEAF_SELECTION_001_implicit_no_numba_is_exact_scipy."""
    import tal.spatial.kernels.fixed_size_backends as fixed
    import tal.spatial.kernels.rotation_interp_backends as interpolation

    monkeypatch.setattr(fixed, "_numba_available", lambda: False)
    monkeypatch.setattr(interpolation, "_numba_available", lambda: False)
    left_angles = np.deg2rad([0.0, 15.0])
    right_angles = np.deg2rad([45.0, 90.0])
    left = SciRotation.from_rotvec(np.column_stack((np.zeros((2, 2)), left_angles))).as_quat()
    right = SciRotation.from_rotvec(np.column_stack((np.zeros((2, 2)), right_angles))).as_quat()
    alpha = np.asarray([0.25, 0.75])
    valid = np.asarray([True, True])
    expected_slerp = slerp_quat_backend(
        left,
        right,
        alpha,
        valid,
        backend=ROTATION_INTERP_BACKEND_SCIPY,
    )
    expected_compose = quat_compose_block_backend(left, right, backend=SPATIAL_FIXED_BACKEND_SCIPY)
    np.testing.assert_allclose(slerp_quat_backend(left, right, alpha, valid), expected_slerp)
    np.testing.assert_allclose(quat_compose_block_backend(left, right), expected_compose)


def test_spatial_core_leaf_selection_002_implicit_numba_matches_explicit_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_CORE_LEAF_SELECTION_002_implicit_numba_matches_explicit_backend."""
    pytest.importorskip("numba")
    import tal.spatial.kernels.fixed_size_backends as fixed
    import tal.spatial.kernels.rotation_interp_backends as interpolation

    monkeypatch.setattr(fixed, "_numba_available", lambda: True)
    monkeypatch.setattr(interpolation, "_numba_available", lambda: True)
    left_angles = np.deg2rad([0.0, 15.0])
    right_angles = np.deg2rad([45.0, 90.0])
    left = SciRotation.from_rotvec(np.column_stack((np.zeros((2, 2)), left_angles))).as_quat()
    right = SciRotation.from_rotvec(np.column_stack((np.zeros((2, 2)), right_angles))).as_quat()
    alpha = np.asarray([0.25, 0.75])
    valid = np.asarray([True, True])
    expected_slerp = slerp_quat_backend(
        left,
        right,
        alpha,
        valid,
        backend=ROTATION_INTERP_BACKEND_NUMBA,
    )
    expected_compose = quat_compose_block_backend(left, right, backend=SPATIAL_FIXED_BACKEND_NUMBA)
    np.testing.assert_allclose(slerp_quat_backend(left, right, alpha, valid), expected_slerp)
    np.testing.assert_allclose(quat_compose_block_backend(left, right), expected_compose)


def test_spatial_perf_streaming_path_002_dask_task_shape_matches_typed_pipeline() -> None:
    """ID: SPATIAL_PERF_STREAMING_PATH_002_dask_task_shape_matches_typed_pipeline."""
    pytest.importorskip("dask.array")
    graph, providers = _registered_path(1, lazy=True)
    query = np.linspace(0.0, 1.0, 65_537)
    public = solve_pose_path_transform("f1", "f0", graph=graph, query=query)
    direct = providers[0].param.at(query, validate=False)
    public_ds = public.as_dataset(copy="none")
    direct_ds = direct.as_dataset(copy="none")
    for name in public_ds.data_vars:
        public_graph = public_ds[name].data.__dask_graph__()
        direct_graph = direct_ds[name].data.__dask_graph__()
        assert len(public_graph) <= len(direct_graph) + 2
        assert max(public_ds[name].chunks[public_ds[name].get_axis_num("query")]) <= 65_536


@pytest.mark.parametrize("route", ("pose-generic", "pose-streaming", "rotation"))
def test_spatial_hard_path_error_001_single_owner_and_original_cause(route, monkeypatch) -> None:
    """ID: SPATIAL_HARD_PATH_ERROR_001_single_owner_and_original_cause."""
    import tal.spatial.ops.path_execution as execution

    graph = FrameGraph()
    dataset = _edge_pose(0).as_dataset(copy="deep")
    dataset["rotation"] = xr.zeros_like(dataset["rotation"])
    Pose(dataset, parent="f0", child="f1", graph=graph).register()
    monkeypatch.setattr(execution, "_numba_available", lambda: route != "pose-streaming")
    calls = []
    original = execution.stream_pose_path_blocks

    def tracked(*args, **kwargs):
        calls.append("stream")
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "stream_pose_path_blocks", tracked)
    solver = solve_rotation_path_transform if route == "rotation" else solve_pose_path_transform
    owner = "spatial.path_solve.rotation" if route == "rotation" else "spatial.path_solve.pose"
    with pytest.raises(ValueError, match="quaternion norm") as failure:
        solver("f1", "f0", graph=graph, query=np.asarray([0.25]))
    assert str(failure.value).count(f"{owner}:") == 1
    cause = failure.value.__cause__
    assert isinstance(cause, ValueError)
    assert str(cause).startswith("spatial.rotation.interp_backend:")
    assert cause.__cause__ is None
    assert calls == (["stream"] if route == "pose-streaming" else [])
