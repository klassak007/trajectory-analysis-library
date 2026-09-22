from __future__ import annotations

from copy import deepcopy

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed
from dask.callbacks import Callback

from benchmarks.bench_composite_finalization import (
    DEFAULT_CONFIG,
    CompositeBenchmarkConfig,
    composite_fixture,
)
from tal.core.schema_read import read_sequence_size_coord_name
from tal.frames import FrameGraph
from tal.spatial import Acceleration, Pose, Velocity
from tal.spatial.ops import pose_temporal_ops

_SMALL = CompositeBenchmarkConfig(
    trials=4,
    source_samples=17,
    query_samples=33,
    trial_chunk=1,
    source_chunk=8,
    query_chunk=16,
)


class _CopyProbe:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __deepcopy__(self, memo: dict[int, object]) -> _CopyProbe:
        _ = memo
        self.calls.append("copy")
        return _CopyProbe(self.calls)


def _attach_extension(values: tuple[object, ...], probe: object) -> None:
    for value in values:
        dataset = value.as_dataset(copy="none")
        tal = dict(dataset.attrs["tal"])
        ext = dict(tal.get("ext", {}))
        ext["benchmark_probe"] = {"value": probe}
        tal["ext"] = ext
        dataset.attrs = {**dataset.attrs, "tal": tal}


class _RaisingCopyProbe:
    def __init__(self, failure: Exception) -> None:
        self.failure = failure

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        _ = memo
        raise self.failure


@pytest.mark.parametrize(
    "operation",
    ("temporal", "matrix_temporal", "pose", "velocity", "acceleration"),
)
def test_spatial_composite_commit_copies_extensions_once(operation: str) -> None:
    """ID: SPATIAL_OWNERSHIP_COMPOSITE_COMMIT_001_one_extension_copy."""
    fixture = composite_fixture(_SMALL, lazy=False)
    calls: list[str] = []
    probe = _CopyProbe(calls)
    if operation == "temporal":
        sources = (fixture.pose,)
        _attach_extension(sources, probe)
        result = fixture.pose.param.at([0.25, 0.75], validate=False)
    elif operation == "matrix_temporal":
        sources = (fixture.pose.as_matrix(validate=False),)
        _attach_extension(sources, probe)
        result = sources[0].param.at([0.25, 0.75], validate=False)
    elif operation == "pose":
        sources = (fixture.rotation, fixture.position)
        _attach_extension(sources, probe)
        result = Pose.from_components(*sources, validate=False)
    elif operation == "velocity":
        sources = (fixture.linear_velocity, fixture.angular_velocity)
        _attach_extension(sources, probe)
        result = Velocity.from_linear_angular(*sources, validate=False)
    else:
        sources = (fixture.linear_acceleration, fixture.angular_acceleration)
        _attach_extension(sources, probe)
        result = Acceleration.from_linear_angular(*sources, validate=False)

    assert calls == ["copy"]
    copied = result.as_dataset(copy="none").attrs["tal"]["ext"]["benchmark_probe"]
    assert copied["value"] is not probe


class _CountingPose(Pose):
    wrappers = 0

    @classmethod
    def _from_composite_committed(
        cls,
        ds: xr.Dataset,
        *,
        validate: bool,
    ) -> _CountingPose:
        cls.wrappers += 1
        return super()._from_composite_committed(ds, validate=validate)


@pytest.mark.parametrize("validate", (False, True))
def test_spatial_composite_commit_constructs_one_pose_wrapper(validate: bool) -> None:
    """ID: SPATIAL_PERF_COMPOSITE_COMMIT_001_one_public_result_wrapper."""
    fixture = composite_fixture(_SMALL, lazy=False)
    _CountingPose.wrappers = 0
    pose = _CountingPose.from_components(
        fixture.rotation,
        fixture.position,
        validate=validate,
    )
    assert _CountingPose.wrappers == 1

    _CountingPose.wrappers = 0
    result = pose.param.at([0.25, 0.75], validate=validate)
    assert isinstance(result, _CountingPose)
    assert _CountingPose.wrappers == 1


def test_spatial_composite_commit_preserves_dask_laziness_and_parity() -> None:
    """ID: SPATIAL_LAZY_COMPOSITE_COMMIT_001_no_payload_work_during_commit."""
    eager = composite_fixture(_SMALL, lazy=False)
    lazy = composite_fixture(_SMALL, lazy=True)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        temporal = lazy.pose.param.at([0.25, 0.75], validate=False)
        pose = Pose.from_components(lazy.rotation, lazy.position, validate=False)
        velocity = Velocity.from_linear_angular(
            lazy.linear_velocity,
            lazy.angular_velocity,
            validate=False,
        )
    assert tasks == []

    expected = (
        eager.pose.param.at([0.25, 0.75], validate=False),
        Pose.from_components(eager.rotation, eager.position, validate=False),
        Velocity.from_linear_angular(
            eager.linear_velocity,
            eager.angular_velocity,
            validate=False,
        ),
    )
    for actual, reference in zip((temporal, pose, velocity), expected, strict=True):
        xr.testing.assert_identical(
            actual.as_dataset(copy="none").compute(scheduler="synchronous"),
            reference.as_dataset(copy="none"),
        )


@pytest.mark.parametrize("representation", ("components", "matrix"))
def test_pose_temporal_empty_lazy_placeholders_stay_lazy(
    representation: str,
) -> None:
    """ID: SPATIAL_LAZY_COMPOSITE_COMMIT_004_empty_placeholders."""
    fixture = composite_fixture(_SMALL, lazy=True)
    pose = (
        fixture.pose
        if representation == "components"
        else fixture.pose.as_matrix(validate=False)
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = pose.param.at(xr.DataArray(np.empty(0), dims="when"), validate=False)

    assert tasks == []
    assert all(
        variable.chunks is not None
        for variable in result.as_dataset(copy="none").data_vars.values()
    )


def _all_missing_lazy_pose(*, zero_samples: bool, representation: str) -> Pose:
    source = composite_fixture(_SMALL, lazy=True).pose
    dataset = source.as_dataset(copy="none")
    if zero_samples:
        dataset = dataset.isel(sample=slice(0, 0))
    size_name = read_sequence_size_coord_name(dataset)
    assert size_name is not None
    dataset = dataset.assign_coords({size_name: xr.zeros_like(dataset.coords[size_name])})
    pose = Pose(dataset)
    return pose if representation == "components" else pose.as_matrix(validate=True)


@pytest.mark.parametrize("zero_samples", (False, True), ids=("declared-zero", "zero-sample"))
@pytest.mark.parametrize("representation", ("components", "matrix"))
@pytest.mark.parametrize("operation", ("at", "resample_to"))
@pytest.mark.parametrize("query_chunks", (1, 2), ids=("split-query", "single-query-block"))
@pytest.mark.parametrize("validate", (False, True))
def test_pose_temporal_all_missing_lazy_query_stays_structurally_missing(
    zero_samples: bool,
    representation: str,
    operation: str,
    query_chunks: int,
    validate: bool,
) -> None:
    """ID: SPATIAL_LAZY_COMPOSITE_COMMIT_007_all_missing_query_reuse."""
    source = _all_missing_lazy_pose(
        zero_samples=zero_samples,
        representation=representation,
    )
    before = source.as_dataset(copy="deep")
    label_chunks = 2 if query_chunks == 1 else 1
    query = xr.DataArray(
        da.from_array(np.asarray([0.0, 1.0]), chunks=query_chunks),
        dims="query",
        coords={
            "label": (
                "query",
                da.from_array(np.asarray(["left", "right"]), chunks=label_chunks),
            ),
        },
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        first = getattr(source.param, operation)(query, validate=validate)
        second = first.param.at([0.5], validate=validate)

    assert tasks == []
    assert first.as_dataset(copy="none").coords["label"].chunks is not None
    for result in (first, second):
        computed = result.as_dataset(copy="none").compute(scheduler="synchronous")
        size_name = read_sequence_size_coord_name(computed)
        assert size_name is not None
        np.testing.assert_array_equal(
            computed.coords[size_name], np.zeros(computed.coords[size_name].shape),
        )
        assert not bool(computed.coords["valid"].any())
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("operation", ("at", "resample_to"))
def test_pose_temporal_ragged_lazy_query_is_deferred_and_shared(
    operation: str,
) -> None:
    """ID: SPATIAL_LAZY_COMPOSITE_COMMIT_005_ragged_query_deferred."""
    fixture = composite_fixture(_SMALL, lazy=True)
    calls: list[str] = []

    @delayed
    def query_values() -> np.ndarray:
        calls.append("query")
        return np.asarray([0.25, 0.75], dtype=np.float64)

    query = xr.DataArray(
        da.from_delayed(query_values(), shape=(2,), dtype=np.float64),
        dims="query",
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = getattr(fixture.pose.param, operation)(query, validate=False)

    assert calls == []
    assert tasks == []
    result.as_dataset(copy="none").compute(scheduler="synchronous")
    assert calls == ["query"]


def test_pose_temporal_lazy_query_failure_remains_deferred() -> None:
    """ID: SPATIAL_LAZY_COMPOSITE_COMMIT_006_query_failure_deferred."""
    fixture = composite_fixture(_SMALL, lazy=True)
    tasks: list[object] = []

    @delayed
    def unavailable_query() -> np.ndarray:
        raise RuntimeError("deferred query failure")

    query = xr.DataArray(
        da.from_delayed(unavailable_query(), shape=(2,), dtype=np.float64),
        dims="query",
    )
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = fixture.pose.param.at(query, validate=False)
    assert tasks == []
    with pytest.raises(RuntimeError, match="deferred query failure"):
        result.as_dataset(copy="none").compute(scheduler="synchronous")


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("operation", ("at", "resample_to"))
def test_pose_temporal_empty_rotation_preserves_variable_metadata(
    lazy: bool,
    operation: str,
) -> None:
    """ID: SPATIAL_CORE_COMPOSITE_COMMIT_002_empty_rotation_metadata."""
    pose = composite_fixture(_SMALL, lazy=lazy).pose
    rotation = pose.as_dataset(copy="none")["rotation"]
    rotation.attrs = {"units": "quaternion", "nested": {"source": "fixture"}}
    rotation.encoding = {"benchmark_codec": "identity"}
    query_data = np.empty(0, dtype=np.float64)
    query = xr.DataArray(
        da.from_array(query_data, chunks=(1,)) if lazy else query_data,
        dims="query",
    )

    result = getattr(pose.param, operation)(query, validate=False)
    actual = result.as_dataset(copy="none")["rotation"]
    assert actual.attrs == rotation.attrs
    assert actual.encoding == rotation.encoding
    computed = result.as_dataset(copy="none").compute(scheduler="synchronous")
    assert computed["rotation"].attrs == rotation.attrs
    assert computed["rotation"].encoding == rotation.encoding


@pytest.mark.parametrize("validate", (False, True))
def test_matrix_pose_temporal_regenerates_ragged_validity(validate: bool) -> None:
    """ID: SPATIAL_CORE_COMPOSITE_COMMIT_001_matrix_pose_validity_parity."""
    eager = composite_fixture(DEFAULT_CONFIG, lazy=False).pose.as_matrix(
        validate=False
    )
    lazy = composite_fixture(DEFAULT_CONFIG, lazy=True).pose.as_matrix(
        validate=False
    )
    query = np.arange(DEFAULT_CONFIG.query_samples, dtype=np.float64) / 1024.0
    actual = lazy.param.at(query, validate=validate)

    expected = eager.param.at(query, validate=validate).as_dataset(copy="none")
    computed = actual.as_dataset(copy="none").compute(scheduler="synchronous")
    xr.testing.assert_identical(computed, expected)
    np.testing.assert_array_equal(
        computed["group_size"].data,
        np.where(np.arange(DEFAULT_CONFIG.trials) % 2 == 0, 1_025, 769),
    )


def test_spatial_composite_commit_resources_close_once_in_participant_order() -> None:
    """ID: SPATIAL_OWNERSHIP_COMPOSITE_COMMIT_002_ordered_resource_lifetime."""
    fixture = composite_fixture(_SMALL, lazy=False)
    calls: list[str] = []
    rotation_ds = fixture.rotation.as_dataset(copy="none")
    position_ds = fixture.position.as_dataset(copy="none")
    rotation_ds.set_close(lambda: calls.append("rotation"))
    position_ds.set_close(lambda: calls.append("position"))

    result = Pose.from_components(fixture.rotation, fixture.position, validate=False)
    result.close()
    result.close()
    fixture.rotation.close()
    fixture.position.close()
    assert calls == ["rotation", "position"]


def test_spatial_composite_commit_deduplicates_aliased_resource() -> None:
    """ID: SPATIAL_OWNERSHIP_COMPOSITE_COMMIT_003_aliased_resource_once."""
    fixture = composite_fixture(_SMALL, lazy=False)
    calls: list[str] = []

    def close() -> None:
        calls.append("shared")

    fixture.rotation.as_dataset(copy="none").set_close(close)
    fixture.position.as_dataset(copy="none").set_close(close)
    result = Pose.from_components(fixture.rotation, fixture.position, validate=False)
    result.close()
    fixture.rotation.close()
    fixture.position.close()
    assert calls == ["shared"]


def _shared_lazy_components(calls: list[str]):
    fixture = composite_fixture(_SMALL, lazy=False)

    @delayed
    def source() -> np.ndarray:
        calls.append("source")
        position = fixture.position.as_dataset(copy="none")["position"].data
        rotation = fixture.rotation.as_dataset(copy="none")["rotation"].data
        return np.concatenate((position, rotation), axis=-1)

    shared = da.from_delayed(
        source(),
        shape=(_SMALL.trials, _SMALL.source_samples, 7),
        dtype=np.float64,
    )
    position_ds = fixture.position.as_dataset(copy="deep")
    rotation_ds = fixture.rotation.as_dataset(copy="deep")
    position_ds["position"] = position_ds["position"].copy(data=shared[..., :3])
    rotation_ds["rotation"] = rotation_ds["rotation"].copy(data=shared[..., 3:])
    return type(fixture.position)(position_ds), type(fixture.rotation)(rotation_ds)


def test_spatial_composite_commit_shared_source_executes_once() -> None:
    """ID: SPATIAL_LAZY_COMPOSITE_COMMIT_002_shared_source_work_once."""
    calls: list[str] = []
    position, rotation = _shared_lazy_components(calls)
    result = Pose.from_components(rotation, position, validate=False)
    assert calls == []
    result.as_dataset(copy="none").compute(scheduler="synchronous")
    assert calls == ["source"]


def test_pose_temporal_shared_source_stays_lazy_and_executes_once() -> None:
    """ID: SPATIAL_LAZY_COMPOSITE_COMMIT_003_temporal_shared_source_once."""
    calls: list[str] = []
    position, rotation = _shared_lazy_components(calls)
    pose = Pose.from_components(rotation, position, validate=False)
    query = xr.DataArray(
        da.from_array(np.asarray([0.25, 0.75]), chunks=1),
        dims="query",
    )
    result = pose.param.at(query, validate=False)
    assert calls == []
    result.as_dataset(copy="none").compute(scheduler="synchronous")
    assert calls == ["source"]


def test_spatial_composite_failure_precedes_copy_payload_and_resource() -> None:
    """ID: SPATIAL_HARD_COMPOSITE_COMMIT_001_failure_precedence."""
    fixture = composite_fixture(_SMALL, lazy=True)
    copy_calls: list[str] = []
    close_calls: list[str] = []
    probe = _CopyProbe(copy_calls)
    fixture.rotation.as_dataset(copy="none").set_close(
        lambda: close_calls.append("rotation")
    )
    fixture.position.as_dataset(copy="none").set_close(
        lambda: close_calls.append("position")
    )
    left = fixture.rotation.with_graph(FrameGraph())
    right = fixture.position.with_graph(FrameGraph())
    _attach_extension((left, right), probe)
    tasks: list[object] = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(ValueError, match="different FrameGraph"),
    ):
        Pose.from_components(left, right, validate=False)
    assert copy_calls == []
    assert tasks == []
    assert close_calls == []


def test_pose_temporal_output_verification_failure_is_atomic(monkeypatch) -> None:
    """ID: SPATIAL_HARD_COMPOSITE_COMMIT_002_post_assembly_atomicity."""
    fixture = composite_fixture(_SMALL, lazy=True)
    source = fixture.pose.as_dataset(copy="none")
    copy_calls: list[str] = []
    close_calls: list[str] = []
    _attach_extension((fixture.pose,), _CopyProbe(copy_calls))
    source.set_close(lambda: close_calls.append("pose"))
    source_close = source._close
    tasks: list[object] = []

    def reject_output(*_args, **_kwargs) -> None:
        raise ValueError("assembled output does not match its plan")

    monkeypatch.setattr(
        pose_temporal_ops,
        "verify_query_output_plan",
        reject_output,
    )
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(
            ValueError,
            match="spatial.pose.param.at: assembled output does not match its plan",
        ),
    ):
        fixture.pose.param.at([0.25, 0.75], validate=False)

    assert copy_calls == []
    assert tasks == []
    assert close_calls == []
    assert source._close is source_close


@pytest.mark.parametrize(
    "failure",
    (
        ValueError("copy value failure"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
    ),
)
def test_pose_temporal_commit_callback_failure_is_unchanged(
    failure: Exception,
) -> None:
    """ID: SPATIAL_HARD_COMPOSITE_COMMIT_003_callback_identity."""
    pose = composite_fixture(_SMALL, lazy=False).pose
    _attach_extension((pose,), _RaisingCopyProbe(failure))

    with pytest.raises(type(failure)) as captured:
        pose.param.at([0.25, 0.75], validate=False)

    assert captured.value is failure
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("operation", ("at", "resample_to"))
def test_matrix_pose_auxiliary_rejection_precedes_copy_and_payload(
    operation: str,
) -> None:
    """ID: SPATIAL_HARD_COMPOSITE_COMMIT_004_matrix_aux_preflight."""
    pose = composite_fixture(_SMALL, lazy=True).pose.as_matrix(validate=False)
    dataset = pose.as_dataset(copy="none")
    dataset["temperature"] = dataset["pose_matrix"].isel(row=0, col=0)
    copy_calls: list[str] = []
    _attach_extension((pose,), _CopyProbe(copy_calls))
    tasks: list[object] = []

    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(ValueError, match=f"spatial.pose.param.{operation}"),
    ):
        getattr(pose.param, operation)([0.25, 0.75], validate=True)

    assert copy_calls == []
    assert tasks == []


def test_spatial_composite_deferred_failure_can_close_after_compute() -> None:
    """ID: SPATIAL_OWNERSHIP_COMPOSITE_COMMIT_004_deferred_failure_cleanup."""
    fixture = composite_fixture(_SMALL, lazy=False)
    calls: list[str] = []

    @delayed
    def fail() -> np.ndarray:
        raise RuntimeError("deferred spatial failure")

    position_ds = fixture.position.as_dataset(copy="deep")
    position = position_ds["position"]
    position_ds["position"] = position.copy(data=da.from_delayed(
        fail(), shape=position.shape, dtype=np.float64,
    ))
    source = type(fixture.position)(position_ds)
    source.as_dataset(copy="none").set_close(lambda: calls.append("position"))
    result = Pose.from_components(fixture.rotation, source, validate=False)

    with pytest.raises(RuntimeError, match="deferred spatial failure"):
        result.as_dataset(copy="none").compute(scheduler="synchronous")
    result.close()
    result.close()
    assert calls == ["position"]


def test_spatial_composite_metadata_isolation_preserves_sources() -> None:
    """ID: SPATIAL_OWNERSHIP_COMPOSITE_COMMIT_005_metadata_isolation."""
    fixture = composite_fixture(_SMALL, lazy=False)
    shared = {"nested": {"value": 1}}
    fixture.rotation.as_dataset(copy="none").attrs["ordinary"] = shared
    fixture.position.as_dataset(copy="none").attrs["ordinary"] = shared
    source_before = tuple(
        deepcopy(value.as_dataset(copy="none").attrs)
        for value in (fixture.rotation, fixture.position)
    )
    result = Pose.from_components(fixture.rotation, fixture.position, validate=False)
    result.as_dataset(copy="none").attrs["ordinary"]["nested"]["value"] = 2
    for value, expected in zip(
        (fixture.rotation, fixture.position), source_before, strict=True
    ):
        assert value.as_dataset(copy="none").attrs == expected
