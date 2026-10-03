"""Maintained composite-operation benchmark; no CI timing threshold."""

from __future__ import annotations

import json
import platform
import statistics
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import dask
import dask.array as da
import numpy as np
import scipy
import xarray as xr

from benchmarks._composite_measurement import (
    _assert_result_parity,
    _identical_results,
    _materialize,
    _measure_routes,
    _task_count,
    _timing_routes,
)
from tal.core import AnalysisLayoutSpec
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Pose,
    Position,
    Rotation,
    Velocity,
)


@dataclass(frozen=True)
class CompositeBenchmarkConfig:
    """Frozen fixture and sampling configuration."""

    trials: int = 16
    source_samples: int = 129
    query_samples: int = 1_025
    trial_chunk: int = 4
    source_chunk: int = 64
    query_chunk: int = 256


@dataclass(frozen=True)
class CompositeFixture:
    """Prepared public inputs for the three measured operations."""

    pose: Pose
    position: Position
    rotation: Rotation
    linear_velocity: LinearVelocity
    angular_velocity: AngularVelocity
    linear_acceleration: LinearAcceleration
    angular_acceleration: AngularAcceleration
    query: xr.DataArray


DEFAULT_CONFIG = CompositeBenchmarkConfig()
_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")
_RESERVED_VALID_ATTRS = {
    "tal_reserved_owner": "param_ops",
    "tal_reserved_name": "valid",
    "tal_reserved_token": "tal:param_ops:reserved:v1",
}
_RESERVED_SIZE_ATTRS = {
    "tal_reserved_owner": "param_ops",
    "tal_reserved_name": "group_size",
    "tal_reserved_token": "tal:param_ops:reserved:v1",
}


def _payload(values: np.ndarray, *, chunks: tuple[int, ...] | None) -> object:
    return values if chunks is None else da.from_array(values, chunks=chunks)


def _trajectory_layout(core_dim: str) -> AnalysisLayoutSpec:
    return AnalysisLayoutSpec(
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(core_dim,),
        param_coord="time",
        sequence_size_coord="group_size",
    )


def _source_values(config: CompositeBenchmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    u = np.arange(config.source_samples, dtype=np.float64) / 128.0
    trial = np.arange(config.trials, dtype=np.float64)[:, None]
    translation = np.empty((config.trials, config.source_samples, 3), dtype=np.float64)
    translation[..., 0] = 0.01 * trial + u
    translation[..., 1] = np.sin(u)
    translation[..., 2] = np.cos(u)
    theta = 0.002 * trial + 0.25 * u
    quaternion = np.zeros((config.trials, config.source_samples, 4), dtype=np.float64)
    quaternion[..., 2] = np.sin(theta / 2.0)
    quaternion[..., 3] = np.cos(theta / 2.0)
    return translation, quaternion


def _source_valid_lengths(config: CompositeBenchmarkConfig) -> np.ndarray:
    return np.where(
        np.arange(config.trials) % 2 == 0,
        config.source_samples,
        min(config.source_samples, 97),
    ).astype(np.int64)


def _projected_query_valid_lengths(config: CompositeBenchmarkConfig) -> np.ndarray:
    """Project ``u[j] = j/128`` validity onto ``q[k] = k/1024``."""
    source_lengths = _source_valid_lengths(config)
    projected = np.where(source_lengths == 0, 0, 8 * (source_lengths - 1) + 1)
    return np.minimum(projected, config.query_samples).astype(np.int64)


def _trajectory_dataset(
    values: np.ndarray,
    *,
    var_name: str,
    core_dim: str,
    labels: tuple[str, ...],
    config: CompositeBenchmarkConfig,
    lazy: bool,
) -> xr.Dataset:
    chunks = (config.trial_chunk, config.source_chunk, len(labels)) if lazy else None
    return xr.Dataset(
        {var_name: (("trial", "sample", core_dim), _payload(values, chunks=chunks))},
        coords={
            "trial": np.arange(config.trials),
            "sample": np.arange(config.source_samples),
            core_dim: list(labels),
            "time": ("sample", np.arange(config.source_samples, dtype=np.float64) / 128.0),
            "group_size": ("trial", _source_valid_lengths(config)),
        },
    )


def _typed_source(
    values: np.ndarray,
    *,
    cls: type,
    var_name: str,
    core_dim: str,
    labels: tuple[str, ...],
    config: CompositeBenchmarkConfig,
    lazy: bool,
):
    ds = _trajectory_dataset(
        values,
        var_name=var_name,
        core_dim=core_dim,
        labels=labels,
        config=config,
        lazy=lazy,
    )
    return cls(_trajectory_layout(core_dim).wrap(ds, data_vars=var_name))


def _kinematic_values(
    config: CompositeBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.arange(config.query_samples, dtype=np.float64) / 1024.0
    trial = np.arange(config.trials, dtype=np.float64)[:, None]
    linear_v = np.broadcast_to(np.stack((q, 2.0 * q, -q), axis=-1), (config.trials, q.size, 3)).copy()
    angular_v = np.broadcast_to(
        np.stack((0.1 + q, 0.2 - 0.5 * q, 0.3 * q), axis=-1),
        linear_v.shape,
    ).copy()
    linear_a = np.broadcast_to(np.stack((0.5 * q, -q, 1.0 + q), axis=-1), linear_v.shape).copy()
    angular_a = np.broadcast_to(np.stack((0.2 * q, 0.3 - q, 0.4 + q), axis=-1), linear_v.shape).copy()
    linear_v[..., 0] += 0.01 * trial
    linear_a[..., 0] += 0.01 * trial
    return linear_v, angular_v, linear_a, angular_a


def _kinematic_source(
    values: np.ndarray,
    *,
    cls: type,
    var_name: str,
    core_dim: str,
    config: CompositeBenchmarkConfig,
    lazy: bool,
):
    chunks = (config.trial_chunk, config.query_chunk, 3) if lazy else None
    ds = xr.Dataset(
        {var_name: (("trial", "sample", core_dim), _payload(values, chunks=chunks))},
        coords={
            "trial": np.arange(config.trials),
            "sample": np.arange(config.query_samples),
            core_dim: list(_XYZ),
            "time": ("sample", np.arange(config.query_samples, dtype=np.float64) / 1024.0),
            "group_size": ("trial", _projected_query_valid_lengths(config)),
        },
    )
    return cls(_trajectory_layout(core_dim).wrap(ds, data_vars=var_name))


def composite_fixture(
    config: CompositeBenchmarkConfig = DEFAULT_CONFIG,
    *,
    lazy: bool,
) -> CompositeFixture:
    """Construct the frozen public fixture outside measured regions."""
    translation, quaternion = _source_values(config)
    position = _typed_source(
        translation, cls=Position, var_name="position", core_dim="axis",
        labels=_XYZ, config=config, lazy=lazy,
    )
    rotation = _typed_source(
        quaternion, cls=Rotation, var_name="rotation", core_dim="quat",
        labels=_QUAT, config=config, lazy=lazy,
    )
    linear_v, angular_v, linear_a, angular_a = _kinematic_values(config)
    query_data = np.arange(config.query_samples, dtype=np.float64) / 1024.0
    query = xr.DataArray(
        _payload(query_data, chunks=(config.query_chunk,) if lazy else None),
        dims="query",
    )
    return CompositeFixture(
        Pose.from_components(rotation, position), position, rotation,
        _kinematic_source(linear_v, cls=LinearVelocity, var_name="linear_velocity", core_dim="linear_axis", config=config, lazy=lazy),
        _kinematic_source(angular_v, cls=AngularVelocity, var_name="angular_velocity", core_dim="angular_axis", config=config, lazy=lazy),
        _kinematic_source(linear_a, cls=LinearAcceleration, var_name="linear_acceleration", core_dim="linear_axis", config=config, lazy=lazy),
        _kinematic_source(angular_a, cls=AngularAcceleration, var_name="angular_acceleration", core_dim="angular_axis", config=config, lazy=lazy),
        query,
    )


def operation_routes(fixture: CompositeFixture) -> dict[str, Callable[[], object]]:
    """Return the three existing public composite operations."""
    return {
        "pose_temporal": lambda: fixture.pose.param.at(fixture.query, validate=False),
        "pose_components": lambda: Pose.from_components(fixture.rotation, fixture.position, validate=False),
        "paired_kinematics": lambda: (
            Velocity.from_linear_angular(fixture.linear_velocity, fixture.angular_velocity, validate=False),
            Acceleration.from_linear_angular(
                fixture.linear_acceleration, fixture.angular_acceleration, validate=False
            ),
        ),
    }


def _expected_schema(
    *,
    core_dims: tuple[str, str],
    components: tuple[tuple[str, str, str, tuple[str, ...]], ...],
    kind: str | None = None,
) -> dict[str, object]:
    spatial: dict[str, object] = {"representation": {"rep": "components"}}
    if kind is not None:
        spatial["roles"] = {"kinematics_kind": kind}
    registry = {
        name: {"core_dim": dim, "labels": list(labels), "var": var}
        for name, dim, var, labels in components
    }
    return {
        "tal": {
            "ext": {
                "spatial": spatial,
                "components": {"version": 1, "registry": registry},
            },
            "version": 1,
            "core": {
                "roles": {
                    "batch_dims": ["trial"],
                    "core_dims": list(core_dims),
                    "sequence_dim": "sample",
                },
                "param_coord": {"name": "time"},
                "validity": {
                    "sequence_size_coord": "group_size",
                    "layout": "left_packed",
                },
            },
        }
    }


def _expected_pose_temporal(config: CompositeBenchmarkConfig) -> xr.Dataset:
    source_position, _ = _source_values(config)
    query = np.arange(config.query_samples, dtype=np.float64) / 1024.0
    scaled = query * 128.0
    left = np.minimum(np.floor(scaled).astype(np.int64), config.source_samples - 1)
    right = np.minimum(left + 1, config.source_samples - 1)
    alpha = scaled - left
    position = (
        (1.0 - alpha)[None, :, None] * source_position[:, left, :]
        + alpha[None, :, None] * source_position[:, right, :]
    )
    trial = np.arange(config.trials, dtype=np.float64)[:, None]
    theta = 0.002 * trial + 0.25 * query[None, :]
    rotation = np.zeros((config.trials, config.query_samples, 4))
    rotation[..., 2] = np.sin(theta / 2.0)
    rotation[..., 3] = np.cos(theta / 2.0)
    sizes = _projected_query_valid_lengths(config)
    valid = np.arange(config.query_samples)[None, :] < sizes[:, None]
    position = np.where(valid[..., None], position, np.nan)
    rotation = np.where(valid[..., None], rotation, np.nan)
    result = xr.Dataset(
        coords={
            "valid": (("trial", "sample"), valid, _RESERVED_VALID_ATTRS),
            "time": ("sample", query),
            "group_size": ("trial", sizes, _RESERVED_SIZE_ATTRS),
            "trial": np.arange(config.trials),
            "sample": np.arange(config.query_samples),
            "axis": list(_XYZ),
            "quat": list(_QUAT),
        },
        attrs=_expected_schema(
            core_dims=("axis", "quat"),
            components=(
                ("position", "axis", "position", _XYZ),
                ("rotation", "quat", "rotation", _QUAT),
            ),
        ),
    )
    result["position"] = (
        ("trial", "axis", "sample"),
        position.transpose(0, 2, 1),
    )
    result["rotation"] = (("trial", "sample", "quat"), rotation)
    return result


def _expected_pose_components(config: CompositeBenchmarkConfig) -> xr.Dataset:
    position, rotation = _source_values(config)
    return xr.Dataset(
        {
            "position": (("trial", "sample", "axis"), position),
            "rotation": (("trial", "sample", "quat"), rotation),
        },
        coords={
            "time": ("sample", np.arange(config.source_samples) / 128.0),
            "group_size": ("trial", _source_valid_lengths(config)),
            "trial": np.arange(config.trials),
            "sample": np.arange(config.source_samples),
            "axis": list(_XYZ),
            "quat": list(_QUAT),
        },
        attrs=_expected_schema(
            core_dims=("axis", "quat"),
            components=(
                ("position", "axis", "position", _XYZ),
                ("rotation", "quat", "rotation", _QUAT),
            ),
        ),
    )


def _expected_kinematic_pair(
    config: CompositeBenchmarkConfig,
    *,
    kind: str,
    linear: np.ndarray,
    angular: np.ndarray,
) -> xr.Dataset:
    linear_var = f"linear_{kind}"
    angular_var = f"angular_{kind}"
    return xr.Dataset(
        {
            linear_var: (("trial", "sample", "linear_axis"), linear),
            angular_var: (("trial", "sample", "angular_axis"), angular),
        },
        coords={
            "time": ("sample", np.arange(config.query_samples) / 1024.0),
            "group_size": ("trial", _projected_query_valid_lengths(config)),
            "trial": np.arange(config.trials),
            "sample": np.arange(config.query_samples),
            "linear_axis": list(_XYZ),
            "angular_axis": list(_XYZ),
        },
        attrs=_expected_schema(
            core_dims=("linear_axis", "angular_axis"),
            components=(
                ("linear", "linear_axis", linear_var, _XYZ),
                ("angular", "angular_axis", angular_var, _XYZ),
            ),
            kind=kind,
        ),
    )


def expected_results(
    config: CompositeBenchmarkConfig,
) -> dict[str, tuple[xr.Dataset, ...]]:
    """Return frozen expectations without invoking a measured public route."""
    linear_v, angular_v, linear_a, angular_a = _kinematic_values(config)
    return {
        "pose_temporal": (_expected_pose_temporal(config),),
        "pose_components": (_expected_pose_components(config),),
        "paired_kinematics": (
            _expected_kinematic_pair(
                config,
                kind="velocity",
                linear=linear_v,
                angular=angular_v,
            ),
            _expected_kinematic_pair(
                config,
                kind="acceleration",
                linear=linear_a,
                angular=angular_a,
            ),
        ),
    }


def _lazy_expected_results(
    config: CompositeBenchmarkConfig,
) -> dict[str, tuple[xr.Dataset, ...]]:
    """Return expectations after fail-conservative lazy validity finalization."""
    expected = expected_results(config)
    pose = expected["pose_temporal"][0].drop_vars("group_size")
    attrs = deepcopy(pose.attrs)
    attrs["tal"]["core"].pop("validity", None)
    pose.attrs = attrs
    return {**expected, "pose_temporal": (pose,)}


def _dask_report(
    fixture: CompositeFixture,
    expected: dict[str, tuple[xr.Dataset, ...]],
    *,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    graph_routes = operation_routes(fixture)
    graphs = {name: route() for name, route in graph_routes.items()}
    materialize = {name: (lambda value=value: _materialize(value)) for name, value in graphs.items()}
    graph_samples = _measure_routes(
        graph_routes,
        expected,
        warmups=warmups,
        repeats=repeats,
    )
    timing = _timing_routes(
        materialize,
        expected,
        warmups=warmups,
        repeats=repeats,
    )
    materialize_samples = {
        name: {"seconds": rows, "median_seconds": statistics.median(rows)}
        for name, rows in timing.items()
    }
    return {
        "graph_build": graph_samples,
        "materialize": materialize_samples,
        "tasks": {name: _task_count(value) for name, value in graphs.items()},
        "expected_topology_parity": {
            name: _identical_results(_materialize(value), expected[name])
            for name, value in graphs.items()
        },
    }


def _partition_scaling(config: CompositeBenchmarkConfig) -> dict[str, object]:
    four_chunk = max(1, (config.trials + 3) // 4)
    one = composite_fixture(
        CompositeBenchmarkConfig(
            **{**config.__dict__, "trial_chunk": config.trials}
        ),
        lazy=True,
    )
    four = composite_fixture(
        CompositeBenchmarkConfig(**{**config.__dict__, "trial_chunk": four_chunk}),
        lazy=True,
    )
    report: dict[str, object] = {}
    for name in operation_routes(one):
        first = operation_routes(one)[name]()
        fourth = operation_routes(four)[name]()
        first_value = _materialize(first)
        fourth_value = _materialize(fourth)
        for left, right in zip(first_value, fourth_value, strict=True):
            _assert_result_parity(left, right)
        report[name] = {"one_partition": _task_count(first), "four_partitions": _task_count(fourth)}
    return report


def benchmark_report(
    config: CompositeBenchmarkConfig = DEFAULT_CONFIG,
    *,
    warmups: int = 2,
    repeats: int = 7,
) -> dict[str, object]:
    """Run the frozen eager/Dask composite-finalization protocol."""
    eager = composite_fixture(config, lazy=False)
    lazy = composite_fixture(config, lazy=True)
    eager_routes = operation_routes(eager)
    expected = expected_results(config)
    lazy_expected = _lazy_expected_results(config)
    return {
        "fixture": {
            **config.__dict__,
            "dtype": "float64",
            "source_group_size": _source_valid_lengths(config).tolist(),
            "kinematic_group_size": _projected_query_valid_lengths(config).tolist(),
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "xarray": xr.__version__, "scipy": scipy.__version__,
            "dask": dask.__version__, "scheduler": "synchronous",
        },
        "eager": _measure_routes(eager_routes, expected, warmups=warmups, repeats=repeats),
        "dask": _dask_report(
            lazy,
            lazy_expected,
            warmups=warmups,
            repeats=repeats,
        ),
        "partition_scaling": _partition_scaling(config),
    }


def main() -> None:
    """Print the maintained benchmark report as JSON."""
    print(json.dumps(benchmark_report(), indent=2))


if __name__ == "__main__":
    main()
