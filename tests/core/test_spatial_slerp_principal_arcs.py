from __future__ import annotations

import gc
import tracemalloc
from typing import Any

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from tal import AnalysisObject
from tal.spatial import Pose, Position, Rotation
from tal.spatial.kernels import rotation_interp_backends as interpolation
from tal.spatial.kernels import rotation_interp_reference as reference


def test_spatial_slerp_equal_endpoints_match_scipy_normalization() -> None:
    endpoints = np.asarray([[0.0, 0.0, 0.0, 2.0], [1.0, -2.0, 3.0, 4.0], [3.0, 0.0, 0.0, 1.0]])
    alpha = np.asarray([0.25, 0.75, np.nan])
    valid = np.asarray([True, True, False])
    actual = interpolation.slerp_quat_backend(endpoints, endpoints, alpha, valid, backend="scipy")
    expected = reference.scipy_slerp_rows(endpoints[:2], endpoints[:2], alpha[:2], owner="test")
    np.testing.assert_allclose(actual[:2], expected, rtol=0.0, atol=1e-15)
    assert np.isnan(actual[2]).all()


def _endpoint_pairs(dtype: np.dtype) -> np.ndarray:
    reported = np.asarray([
        [[1, 1, 1, 3], [-1, 1, -3, 1]],
        [[1, 2, 1, 1], [-2, 1, -1, 1]],
        [[2, 4, 3, 5], [-4, 2, -5, 3]],
    ], dtype=np.float64)
    reported /= np.linalg.norm(reported, axis=-1, keepdims=True)
    rng = np.random.default_rng(314159)
    left = SciRotation.random(64, random_state=rng)
    axes = rng.normal(size=(64, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    right = left * SciRotation.from_quat(np.column_stack((axes, np.zeros(64))))
    half_turns = np.stack((left.as_quat(), right.as_quat()), axis=1)
    ordinary = np.stack((left.as_quat(), (left * SciRotation.from_rotvec(axes * 0.4)).as_quat()), axis=1)
    return np.concatenate((reported, half_turns, ordinary)).astype(dtype)


def _expected_matrices(pairs: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    return np.stack([Slerp([0.0, 1.0], SciRotation.from_quat(pair))(fractions).as_matrix() for pair in pairs])


def _rotation(pairs: np.ndarray, *, lazy: bool = False) -> Rotation:
    array = xr.DataArray(
        pairs,
        dims=("trial", "sample", "quat"),
        coords={"trial": np.arange(len(pairs)), "sample": [0, 1], "quat": ["x", "y", "z", "w"], "time": ("sample", [0.0, 1.0])},
        name="rotation",
    )
    if lazy:
        pytest.importorskip("dask.array")
        array = array.chunk({"trial": len(pairs), "sample": 2, "quat": 4})
    return Rotation.from_data(array, sequence_dim="sample", batch_dims=("trial",), core_dims=("quat",), param_coord="time")


def _pose(pairs: np.ndarray, *, lazy: bool = False) -> Pose:
    rotation = _rotation(pairs, lazy=lazy)
    array = xr.DataArray(
        np.zeros((len(pairs), 2, 3), dtype=np.float64),
        dims=("trial", "sample", "axis"),
        coords={
            "trial": np.arange(len(pairs)),
            "sample": [0, 1],
            "axis": ["x", "y", "z"],
            "time": ("sample", [0.0, 1.0]),
        },
        name="position",
    )
    if lazy:
        array = array.chunk({"trial": len(pairs), "sample": 2, "axis": 3})
    position = Position(
        AnalysisObject.from_data(
            array,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("axis",),
            param_coord="time",
        )
    )
    return Pose.from_components(rotation, position)


class _UnevaluatedBatchTransform(xr.indexes.CoordinateTransform):
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
            isinstance(other, _UnevaluatedBatchTransform)
            and self.dim_size == other.dim_size
        )


def _native_batch_index(
    size: int,
    kind: str,
    calls: list[str],
) -> xr.Index:
    if kind == "range":
        return xr.indexes.RangeIndex.arange(size, dim="trial")
    return xr.indexes.CoordinateTransformIndex(
        _UnevaluatedBatchTransform(size, calls)
    )


def _native_batched_spatial(
    size: int,
    *,
    kind: str,
    spatial_type: str,
    lazy: bool,
    calls: list[str],
) -> Rotation | Pose:
    index = _native_batch_index(size, kind, calls)
    index_coords = xr.Coordinates.from_xindex(index)
    pairs = np.broadcast_to(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        (size, 2, 4),
    )
    rotation_data = xr.DataArray(
        pairs,
        dims=("trial", "sample", "quat"),
        coords=index_coords.assign(
            sample=[0, 1],
            quat=["x", "y", "z", "w"],
            time=("sample", [0.0, 1.0]),
        ),
        name="rotation",
    )
    if lazy:
        rotation_data = rotation_data.chunk({"trial": size, "sample": 2, "quat": 4})
    rotation = Rotation.from_data(
        rotation_data,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("quat",),
        param_coord="time",
    )
    if spatial_type == "rotation":
        return rotation
    position_data = xr.DataArray(
        np.zeros((size, 2, 3), dtype=np.float64),
        dims=("trial", "sample", "axis"),
        coords=index_coords.assign(
            sample=[0, 1],
            axis=["x", "y", "z"],
            time=("sample", [0.0, 1.0]),
        ),
        name="position",
    )
    if lazy:
        position_data = position_data.chunk({"trial": size, "sample": 2, "axis": 3})
    position = Position(
        AnalysisObject.from_data(
            position_data,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("axis",),
            param_coord="time",
        )
    )
    return Pose.from_components(rotation, position)


@pytest.mark.parametrize("dtype", (np.dtype("float32"), np.dtype("float64")))
@pytest.mark.parametrize("backend", ("scipy", "numba"))
@pytest.mark.parametrize("left_sign,right_sign", ((1, 1), (1, -1), (-1, 1), (-1, -1)))
def test_spatial_core_principal_arc_002_authoritative_preparation(dtype, backend, left_sign, right_sign) -> None:
    """ID: SPATIAL_CORE_PRINCIPAL_ARC_002_authoritative_preparation."""
    if backend == "numba":
        pytest.importorskip("numba")
    pairs = _endpoint_pairs(dtype) * np.asarray([left_sign, right_sign], dtype=dtype)[None, :, None]
    fractions = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=dtype)
    left = np.broadcast_to(pairs[:, :1], (len(pairs), 5, 4))
    right = np.broadcast_to(pairs[:, 1:], left.shape)
    before = pairs.copy()
    actual = interpolation.slerp_quat_backend(left, right, np.broadcast_to(fractions, left.shape[:-1]), np.ones(left.shape[:-1], bool), backend=backend)
    matrices = SciRotation.from_quat(actual.reshape(-1, 4)).as_matrix().reshape(len(pairs), 5, 3, 3)
    tolerance = 1e-6 if dtype.itemsize == 4 else 1e-12
    np.testing.assert_allclose(matrices, _expected_matrices(pairs, fractions), rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(pairs, before)


@pytest.mark.parametrize("dtype", (np.dtype("float32"), np.dtype("float64")))
@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("numba_available", (False, True))
def test_spatial_core_principal_arc_003_public_batched_parity(dtype, lazy, numba_available, monkeypatch) -> None:
    """ID: SPATIAL_CORE_PRINCIPAL_ARC_003_public_batched_parity."""
    from dask.callbacks import Callback

    if numba_available:
        pytest.importorskip("numba")
    pairs = _endpoint_pairs(dtype)
    source = _rotation(pairs, lazy=lazy)
    before = source.as_dataset(copy="deep")
    monkeypatch.setattr(interpolation, "_numba_available", lambda: numba_available)
    fractions = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=dtype)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.param.at(fractions)
    assert not tasks
    dataset = result.as_dataset(copy="none")
    assert (dataset["rotation"].chunks is not None) == lazy
    actual = dataset.compute()["rotation"].data
    matrices = SciRotation.from_quat(actual.reshape(-1, 4)).as_matrix().reshape(len(pairs), 5, 3, 3)
    tolerance = 1e-6 if dtype.itemsize == 4 else 1e-12
    np.testing.assert_allclose(matrices, _expected_matrices(pairs, fractions), rtol=tolerance, atol=tolerance)
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("scale", (np.float32(1.0e20), np.float32(1.0e-30)))
@pytest.mark.parametrize("fraction", (0.0, 0.5, 1.0))
@pytest.mark.parametrize("backend", ("scipy", "numba"))
def test_spatial_hard_slerp_work_dtype_001_extreme_float32_norms(
    scale: np.float32,
    fraction: float,
    backend: str,
) -> None:
    """ID: SPATIAL_HARD_SLERP_WORK_DTYPE_001."""
    if backend == "numba":
        pytest.importorskip("numba")
    normalized = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    endpoints = (normalized * float(scale)).astype(np.float32)
    left = np.stack((endpoints[0], np.zeros(4, dtype=np.float32)))
    right = np.stack((endpoints[1], np.zeros(4, dtype=np.float32)))
    actual = interpolation.slerp_quat_backend(
        left,
        right,
        np.asarray([fraction, 0.5], dtype=np.float32),
        np.asarray([True, False]),
        backend=backend,
    )
    expected = Slerp(
        [0.0, 1.0],
        SciRotation.from_quat(normalized),
    )([fraction]).as_matrix()[0]

    np.testing.assert_allclose(
        SciRotation.from_quat(actual[0]).as_matrix(),
        expected,
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    assert bool(np.isnan(actual[1]).all())


def _observe_reference(monkeypatch) -> list[int]:
    calls = []
    original = reference.scipy_slerp_rows

    def tracked(left, right, alpha, **kwargs):
        calls.append(len(alpha))
        return original(left, right, alpha, **kwargs)

    monkeypatch.setattr(reference, "scipy_slerp_rows", tracked)
    monkeypatch.setattr(interpolation, "scipy_slerp_rows", tracked)
    return calls


@pytest.mark.parametrize("size", (0, 1, 65_535, 65_536, 65_537))
@pytest.mark.parametrize("backend", ("scipy", "numba"))
def test_spatial_perf_principal_arc_001_noncontiguous_bounded_rows(size, backend, monkeypatch) -> None:
    """ID: SPATIAL_PERF_PRINCIPAL_ARC_001_noncontiguous_bounded_rows."""
    if backend == "numba":
        pytest.importorskip("numba")
    pair = _endpoint_pairs(np.dtype("float64"))[2]
    storage = np.empty((size, 8))
    storage[:, ::2] = pair[0]
    storage[:, 1::2] = pair[1]
    left, right = storage[:, ::2], storage[:, 1::2]
    before = storage.copy()
    calls = _observe_reference(monkeypatch)
    actual = interpolation.slerp_quat_backend(left, right, np.full(size, 0.5), np.ones(size, bool), backend=backend)
    assert all(count <= 65_536 for count in calls)
    assert sum(calls) == size
    assert actual.shape == (size, 4)
    if size:
        expected = Slerp([0.0, 1.0], SciRotation.from_quat(pair))([0.5]).as_matrix()[0]
        np.testing.assert_allclose(SciRotation.from_quat(actual).as_matrix(), np.broadcast_to(expected, (size, 3, 3)), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(storage, before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("numba_available", (False, True))
def test_spatial_perf_principal_arc_002_public_batch_query_bound(lazy, numba_available, monkeypatch) -> None:
    """ID: SPATIAL_PERF_PRINCIPAL_ARC_002_public_batch_query_bound."""
    from dask.callbacks import Callback

    if numba_available:
        pytest.importorskip("numba")
    pairs = np.broadcast_to([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]], (2, 2, 4)).copy()
    source = _rotation(pairs, lazy=lazy)
    monkeypatch.setattr(interpolation, "_numba_available", lambda: numba_available)
    calls = _observe_reference(monkeypatch)
    tasks = []
    fractions = np.linspace(0.0, 1.0, 40_000)
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.param.at(fractions)
    assert not tasks
    actual = result.as_dataset(copy="none").compute(scheduler="synchronous")["rotation"].data
    assert max(calls) <= 65_536
    assert sum(calls) >= 79_998
    expected = _expected_matrices(pairs, fractions)
    matrices = SciRotation.from_quat(actual.reshape(-1, 4)).as_matrix().reshape(2, 40_000, 3, 3)
    np.testing.assert_allclose(matrices, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ("range", "transform"))
@pytest.mark.parametrize("spatial_type", ("rotation", "pose"))
@pytest.mark.parametrize("lazy", (False, True))
def test_spatial_core_block_index_001_typed_temporal_batch_indexes_survive(
    kind: str,
    spatial_type: str,
    lazy: bool,
) -> None:
    """ID: SPATIAL_CORE_BLOCK_INDEX_001_typed_temporal_batch_indexes_survive."""
    from dask.callbacks import Callback

    size = 65_537
    calls: list[str] = []
    source = _native_batched_spatial(
        size,
        kind=kind,
        spatial_type=spatial_type,
        lazy=lazy,
        calls=calls,
    )
    query_index = _native_batch_index(size, kind, calls)
    query = xr.DataArray(
        np.full((size, 1), 0.5),
        dims=("trial", "when"),
        coords=xr.Coordinates.from_xindex(query_index),
    )
    tasks: list[object] = []
    calls.clear()
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.param.at(query)
    assert tasks == []
    dataset = result.as_dataset(copy="none")
    assert type(dataset.xindexes["trial"]) is type(query.xindexes["trial"])
    assert dataset.xindexes["trial"].equals(query.xindexes["trial"])
    assert calls == []
    if lazy:
        dataset = dataset.compute(scheduler="synchronous")
    assert dataset.sizes["trial"] == size
    assert dataset.sizes["sample"] == 1


@pytest.mark.parametrize("first_error", ("alpha", "quaternion"))
def test_spatial_hard_principal_arc_001_masking_and_failure_order(first_error) -> None:
    """ID: SPATIAL_HARD_PRINCIPAL_ARC_001_masking_and_failure_order."""
    pytest.importorskip("numba")
    pairs = _endpoint_pairs(np.dtype("float64"))[:3]
    left, right = pairs[:, 0].copy(), pairs[:, 1].copy()
    left[0] = 0.0
    valid = np.array([False, True, True])
    alpha = np.full(3, 0.5)
    actual = interpolation.slerp_quat_backend(left, right, alpha, valid, backend="numba")
    assert np.isnan(actual[0]).all()
    assert np.isfinite(actual[1:]).all()
    if first_error == "alpha":
        alpha[1], left[2] = 1.1, 0.0
        message = "finite alpha values"
    else:
        left[1], alpha[2] = 0.0, 1.1
        message = "quaternion norm"
    with pytest.raises(ValueError, match=message):
        interpolation.slerp_quat_backend(left, right, alpha, valid, backend="numba")


@pytest.mark.parametrize("backend", ("scipy", "auto_scipy", "numba"))
@pytest.mark.parametrize("first_error", ("alpha", "quaternion"))
@pytest.mark.parametrize("cross_block", (False, True))
def test_spatial_hard_slerp_reference_001_scipy_uses_public_row_failure_order(
    backend: str,
    first_error: str,
    cross_block: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_HARD_SLERP_REFERENCE_001_scipy_uses_public_row_failure_order."""
    if backend == "numba":
        pytest.importorskip("numba")
    if backend == "auto_scipy":
        monkeypatch.setattr(interpolation, "_numba_available", lambda: False)
        backend = "auto"
    size = 65_537 if cross_block else 3
    first, second = ((65_535, 65_536) if cross_block else (1, 2))
    left = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (size, 4)).copy()
    right = left.copy()
    alpha = np.full(size, 0.5)
    valid = np.ones(size, dtype=bool)
    if first_error == "alpha":
        alpha[first] = 1.1
        left[second] = 0.0
        message = "finite alpha values"
    else:
        left[first] = 0.0
        alpha[second] = 1.1
        message = "quaternion norm"
    with pytest.raises(ValueError, match=message):
        interpolation.slerp_quat_backend(left, right, alpha, valid, backend=backend)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("kind", ("rotation", "pose"))
def test_spatial_perf_slerp_gather_001_eager_brackets_are_block_bounded(
    lazy: bool,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_PERF_SLERP_GATHER_001_eager_brackets_are_block_bounded."""
    import tal.spatial.ops.rotation_temporal_ops as temporal

    pairs = np.broadcast_to(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        (2, 2, 4),
    ).copy()
    source = _rotation(pairs, lazy=lazy) if kind == "rotation" else _pose(pairs, lazy=lazy)
    original = temporal.gather_sequence_block
    logical_rows: list[int] = []

    def tracked(values, indexer, **kwargs):
        gathered = original(values, indexer, **kwargs)
        logical_rows.append(int(gathered.size // 4))
        return gathered

    monkeypatch.setattr(temporal, "gather_sequence_block", tracked)
    monkeypatch.setattr(interpolation, "_numba_available", lambda: False)
    result = source.param.at(np.linspace(0.0, 1.0, 40_000))
    if lazy:
        result.as_dataset(copy="none").compute(scheduler="synchronous")
    assert logical_rows
    assert max(logical_rows) <= 65_536


def _public_slerp_peak(source: Rotation | Pose, query: np.ndarray) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        result = source.param.at(query)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.as_dataset(copy="none").sizes["sample"] == len(query)
    return peak


@pytest.mark.parametrize(
    "kind,limit_mib",
    (("rotation", 24), ("pose", 68)),
)
def test_spatial_perf_slerp_gather_002_eager_peak_retains_one_output_and_one_block(
    kind: str,
    limit_mib: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: SPATIAL_PERF_SLERP_GATHER_002_eager_peak_retains_one_output_and_one_block."""
    pairs = _endpoint_pairs(np.dtype("float64"))[:1]
    source = _rotation(pairs) if kind == "rotation" else _pose(pairs)
    monkeypatch.setattr(interpolation, "_numba_available", lambda: False)
    source.param.at(np.asarray([0.5]))
    small_query = np.linspace(0.0, 1.0, 65_536)
    large_query = np.linspace(0.0, 1.0, 262_144)
    small = _public_slerp_peak(source, small_query)
    large = _public_slerp_peak(source, large_query)
    assert large - small < limit_mib * 1024**2, (small, large)
