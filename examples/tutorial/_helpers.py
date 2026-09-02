"""Notebook tutorial helper utilities.

This file contains boilerplate NumPy/xarray data generation used to keep the
release notebooks focused on TAL APIs and semantics.

In real workflows, you would typically load data from CSV, ROS bag/MCAP, or a
database using TAL I/O surfaces (`tal.io`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tal import AnalysisObject
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    PathSolveOptions,
    Pose,
    Position,
    Rotation,
    Velocity,
    propagate_inertial_status,
    set_edge_motion_class,
    set_frame_inertial_status,
)
from tal.utils.frame_ops import frame_retag


def _readonly_dataset(obj: Any) -> xr.Dataset:
    """Return a read-only tutorial view of an AO-like object."""
    if isinstance(obj, AnalysisObject):
        return obj.as_dataset(copy="shallow")
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=obj.name or "value")
    if hasattr(obj, "as_dataset"):
        data = obj.as_dataset()
        if isinstance(data, xr.Dataset):
            return data
    raise TypeError(f"Expected an AnalysisObject-like object or xarray data; got {type(obj)!r}.")


def summarize_ao(obj: Any, *, snapshot: xr.Dataset | None = None) -> pd.DataFrame:
    """Return a compact, tutorial-friendly summary for AO or typed TAL objects.

    The release notebooks use this instead of importing schema internals. It
    keeps the visible tutorial vocabulary close to user-facing concepts:
    dimensions, variables, roles, param coordinate, validity coordinate, and
    frame metadata.
    """
    ds = snapshot if snapshot is not None else _readonly_dataset(obj)
    tal_schema = ds.attrs.get("tal", {})
    core = tal_schema.get("core", {}) if isinstance(tal_schema, dict) else {}
    roles = core.get("roles", {}) if isinstance(core, dict) else {}
    validity = core.get("validity", {}) if isinstance(core, dict) else {}
    frames = tal_schema.get("ext", {}).get("frames", {}) if isinstance(tal_schema, dict) else {}
    spatial = tal_schema.get("ext", {}).get("spatial", {}) if isinstance(tal_schema, dict) else {}

    rows = [
        ("type", type(obj).__name__),
        ("dims", dict(ds.sizes)),
        ("data_vars", list(ds.data_vars)),
        ("coords", list(ds.coords)),
        ("batch_dims", tuple(roles.get("batch_dims", ()))),
        ("sequence_dim", roles.get("sequence_dim")),
        ("core_dims", tuple(roles.get("core_dims", ()))),
        ("param_coord", core.get("param_coord", {}).get("name") if isinstance(core.get("param_coord"), dict) else None),
        ("sequence_size_coord", validity.get("sequence_size_coord")),
        ("frames", frames or None),
        ("spatial", spatial or None),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])


def show_ao(label: str, obj: Any, *, data: bool = True) -> Any:
    """Print a label, display a compact AO summary, and optionally show data."""
    from IPython.display import HTML, Markdown, display

    snapshot = _readonly_dataset(obj)
    display(Markdown(f"#### {label}"))
    display(summarize_ao(obj, snapshot=snapshot))
    if data:
        display(snapshot)
    display(HTML("<div style='height: 0.75rem'></div>"))
    return obj


def compact_dataset(obj: Any, *, max_rows: int = 12) -> pd.DataFrame:
    """Return a compact variable/coordinate inventory for xarray-like data."""
    ds = _readonly_dataset(obj)
    rows: list[dict[str, object]] = []
    for name, da in ds.data_vars.items():
        rows.append(
            {
                "kind": "data_var",
                "name": name,
                "dims": tuple(da.dims),
                "shape": tuple(int(da.sizes[dim]) for dim in da.dims),
                "dtype": str(da.dtype),
            }
        )
    for name, coord in ds.coords.items():
        rows.append(
            {
                "kind": "coord",
                "name": name,
                "dims": tuple(coord.dims),
                "shape": tuple(int(coord.sizes[dim]) for dim in coord.dims),
                "dtype": str(coord.dtype),
            }
        )
    return pd.DataFrame(rows).head(max_rows)


def display_result(label: str, obj: Any, *, data: bool = False, max_rows: int = 12) -> Any:
    """Display one labeled result with compact output by default."""
    from IPython.display import HTML, Markdown, display

    display(Markdown(f"#### {label}"))
    if isinstance(obj, pd.DataFrame):
        display(obj.head(max_rows))
    elif isinstance(obj, (xr.Dataset, xr.DataArray)) or hasattr(obj, "as_dataset"):
        snapshot = _readonly_dataset(obj)
        display(summarize_ao(obj, snapshot=snapshot))
        if data:
            display(snapshot)
        else:
            display(compact_dataset(snapshot, max_rows=max_rows))
    else:
        display(obj)
    display(HTML("<div style='height: 0.75rem'></div>"))
    return obj


def explain_expected_failure(fn: Any, label: str) -> None:
    """Run a callable and print a compact expected-failure message."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - tutorial helper intentionally summarizes failures.
        print(f"{label}: expected {type(exc).__name__}: {exc}")
    else:
        raise AssertionError(f"{label}: expected an exception, but the operation succeeded.")


def _scalar_group_size(size: int) -> xr.DataArray:
    return xr.DataArray(np.array(size, dtype=np.int64), dims=())


def _rotation_from_quat_samples(quat_values: np.ndarray) -> Rotation:
    rot_ds = xr.Dataset(
        data_vars={"rotation": (("sample", "quat"), quat_values)},
        coords={"sample": np.arange(quat_values.shape[0]), "quat": ["x", "y", "z", "w"]},
    )
    return Rotation(
        AnalysisObject.from_data(
            rot_ds,
            sequence_dim="sample",
            core_dims=("quat",),
            validate=True,
        )
    )


def _position_from_xyz_samples(xyz_values: np.ndarray) -> Position:
    pos_ds = xr.Dataset(
        data_vars={"position": (("sample", "axis"), xyz_values)},
        coords={"sample": np.arange(xyz_values.shape[0]), "axis": ["x", "y", "z"]},
    )
    return Position(
        AnalysisObject.from_data(
            pos_ds,
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        )
    )


def _tile_vector(vec: np.ndarray, *, n_trial: int, n_sample: int) -> np.ndarray:
    return np.broadcast_to(vec.reshape(1, 1, -1), (n_trial, n_sample, vec.size)).copy()


def _tile_quat(quat: np.ndarray, *, n_trial: int, n_sample: int) -> np.ndarray:
    return np.broadcast_to(quat.reshape(1, 1, 4), (n_trial, n_sample, 4)).copy()


def _quat_z(theta_rad: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(theta_rad / 2.0), np.cos(theta_rad / 2.0)], dtype=float)


def _position_from_values_2d(name: str, values: np.ndarray, trial: np.ndarray, sample: np.ndarray, time_s: np.ndarray) -> Position:
    ds = xr.Dataset(
        data_vars={name: (("trial", "sample", "axis"), values)},
        coords={
            "trial": trial,
            "sample": sample,
            "axis": ["x", "y", "z"],
            "time_s": ("sample", time_s),
        },
    )
    return Position(
        AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("axis",),
            param_coord="time_s",
            validate=True,
        )
    )


def _rotation_from_values_2d(name: str, values: np.ndarray, trial: np.ndarray, sample: np.ndarray, time_s: np.ndarray) -> Rotation:
    ds = xr.Dataset(
        data_vars={name: (("trial", "sample", "quat"), values)},
        coords={
            "trial": trial,
            "sample": sample,
            "quat": ["x", "y", "z", "w"],
            "time_s": ("sample", time_s),
        },
    )
    return Rotation(
        AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("quat",),
            param_coord="time_s",
            validate=True,
        )
    )


def make_imu_gps_alignment_aos() -> tuple[AnalysisObject, AnalysisObject]:
    """Return a high-rate IMU AO and low-rate GPS AO on mismatched time grids."""
    imu_time = np.linspace(0.0, 2.0, 201, dtype=float)  # 100 Hz over 2 seconds
    gps_time = np.linspace(0.0, 2.0, 21, dtype=float)  # 10 Hz over 2 seconds

    imu_speed = 5.0 + 0.6 * np.sin(2.0 * np.pi * imu_time) + 0.1 * np.cos(6.0 * np.pi * imu_time)
    gps_speed = 5.0 + 0.55 * np.sin(2.0 * np.pi * gps_time + 0.1)

    imu_ds = xr.Dataset(
        data_vars={"imu_speed_mps": ("sample", imu_speed)},
        coords={
            "sample": np.arange(imu_time.size, dtype=np.int64),
            "time_s": ("sample", imu_time),
            "group_size": _scalar_group_size(imu_time.size),
        },
    )
    gps_ds = xr.Dataset(
        data_vars={"gps_speed_mps": ("sample", gps_speed)},
        coords={
            "sample": np.arange(gps_time.size, dtype=np.int64),
            "time_s": ("sample", gps_time),
            "group_size": _scalar_group_size(gps_time.size),
        },
    )

    imu_ao = AnalysisObject.from_data(
        imu_ds,
        sequence_dim="sample",
        core_dims=(),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )
    gps_ao = AnalysisObject.from_data(
        gps_ds,
        sequence_dim="sample",
        core_dims=(),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )
    return imu_ao, gps_ao


def make_pose_demo_objects() -> tuple[Position, Rotation, Pose]:
    """Return a compact Position/Rotation/Pose trio for notebook walkthroughs."""
    time_s = np.array([0.0, 0.5, 1.0], dtype=float)
    sample = np.arange(time_s.size, dtype=np.int64)

    pos_ds = xr.Dataset(
        data_vars={
            "position": (
                ("sample", "axis"),
                np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.1, 0.0]], dtype=float),
            )
        },
        coords={
            "sample": sample,
            "axis": ["x", "y", "z"],
            "time_s": ("sample", time_s),
            "group_size": _scalar_group_size(sample.size),
        },
    )
    rot_ds = xr.Dataset(
        data_vars={
            "rotation": (
                ("sample", "quat"),
                np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)],
                        [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)],
                    ],
                    dtype=float,
                ),
            )
        },
        coords={
            "sample": sample,
            "quat": ["x", "y", "z", "w"],
            "time_s": ("sample", time_s),
            "group_size": _scalar_group_size(sample.size),
        },
    )

    pos = Position(
        AnalysisObject.from_data(
            pos_ds,
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
            sequence_size_coord="group_size",
            validate=True,
        )
    )
    rot = Rotation(
        AnalysisObject.from_data(
            rot_ds,
            sequence_dim="sample",
            core_dims=("quat",),
            param_coord="time_s",
            sequence_size_coord="group_size",
            validate=True,
        )
    )
    pose = Pose.from_components(rot, pos, validate=True)
    return pos, rot, pose


def make_kinematics_demo_objects() -> tuple[np.ndarray, Position, LinearVelocity, AngularVelocity, Velocity]:
    """Return position + velocity-family objects for temporal calculus demos."""
    t = np.linspace(0.0, 1.0, 11, dtype=float)
    sample = np.arange(t.size, dtype=np.int64)

    position_values = np.stack([0.5 * t**2, 0.2 * t, 0.0 * t], axis=-1)
    pos_ds = xr.Dataset(
        data_vars={"position": (("sample", "axis"), position_values)},
        coords={
            "sample": sample,
            "axis": ["x", "y", "z"],
            "time_s": ("sample", t),
            "group_size": _scalar_group_size(t.size),
        },
    )
    position = Position(
        AnalysisObject.from_data(
            pos_ds,
            sequence_dim="sample",
            core_dims=("axis",),
            param_coord="time_s",
            sequence_size_coord="group_size",
            validate=True,
        )
    )
    linear_velocity = position.differentiate(on="time_s", validate=True)

    angular_values = np.stack([0.0 * t, 0.0 * t, 0.5 * t], axis=-1)
    av_ds = xr.Dataset(
        data_vars={"angular_velocity": (("sample", "angular_axis"), angular_values)},
        coords={
            "sample": sample,
            "angular_axis": ["x", "y", "z"],
            "time_s": ("sample", t),
            "group_size": _scalar_group_size(t.size),
        },
    )
    angular_velocity = AngularVelocity(
        AnalysisObject.from_data(
            av_ds,
            sequence_dim="sample",
            core_dims=("angular_axis",),
            param_coord="time_s",
            sequence_size_coord="group_size",
            validate=True,
        )
    )
    velocity = Velocity.from_linear_angular(linear_velocity, angular_velocity, validate=True)
    return t, position, linear_velocity, angular_velocity, velocity


def build_topology_edge_payloads() -> tuple[Pose, Pose]:
    """Return two frame-tagged edge poses for notebook 10 path solve examples."""
    q_identity = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
    q_z_45 = np.array([[0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)]], dtype=float)

    edge_sb = frame_retag(
        Pose.from_components(
            _rotation_from_quat_samples(q_z_45),
            _position_from_xyz_samples(np.array([[0.5, 0.0, 0.0]], dtype=float)),
            validate=True,
        ),
        parent="body",
        child="sensor",
        validate=True,
    )
    edge_bw = frame_retag(
        Pose.from_components(
            _rotation_from_quat_samples(q_identity),
            _position_from_xyz_samples(np.array([[0.0, 1.0, 0.0]], dtype=float)),
            validate=True,
        ),
        parent="world",
        child="body",
        validate=True,
    )
    return edge_sb, edge_bw


def build_framegraph_spatial_payloads(*, graph: Any, world: Any, base: Any, sensor: Any, tool: Any, map_frame: Any) -> dict[str, Any]:
    """Build shared typed payloads + resolvers for frame-aware spatial walkthroughs."""
    trial = np.array(["trial_0", "trial_1"], dtype=object)
    sample = np.arange(5, dtype=np.int64)
    time_s = np.linspace(0.0, 1.0, sample.size, dtype=float)
    n_trial = trial.size
    n_sample = sample.size

    position = _position_from_values_2d(
        "position",
        np.stack(
            [
                np.stack([0.2 * sample + 0.0, 0.05 * sample, 0.0 * sample], axis=-1),
                np.stack([0.2 * sample + 0.3, -0.04 * sample, 0.1 * np.ones_like(sample)], axis=-1),
            ],
            axis=0,
        ).astype(float),
        trial,
        sample,
        time_s,
    )
    rotation = _rotation_from_values_2d(
        "rotation",
        _tile_quat(_quat_z(np.pi / 12.0), n_trial=n_trial, n_sample=n_sample),
        trial,
        sample,
        time_s,
    )
    pose = Pose.from_components(rotation, position, validate=True)

    linear_velocity = LinearVelocity(
        AnalysisObject.from_data(
            xr.Dataset(
                data_vars={
                    "linear_velocity": (
                        ("trial", "sample", "linear_axis"),
                        np.stack(
                            [
                                np.stack([0.4 + 0.1 * sample, 0.0 * sample, 0.0 * sample], axis=-1),
                                np.stack([0.3 + 0.08 * sample, 0.02 * sample, 0.0 * sample], axis=-1),
                            ],
                            axis=0,
                        ).astype(float),
                    )
                },
                coords={"trial": trial, "sample": sample, "linear_axis": ["x", "y", "z"], "time_s": ("sample", time_s)},
            ),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("linear_axis",),
            param_coord="time_s",
            validate=True,
        )
    )
    angular_velocity = AngularVelocity(
        AnalysisObject.from_data(
            xr.Dataset(
                data_vars={
                    "angular_velocity": (
                        ("trial", "sample", "angular_axis"),
                        np.stack(
                            [
                                np.stack([0.0 * sample, 0.0 * sample, 0.2 + 0.02 * sample], axis=-1),
                                np.stack([0.0 * sample, 0.0 * sample, 0.15 + 0.03 * sample], axis=-1),
                            ],
                            axis=0,
                        ).astype(float),
                    )
                },
                coords={"trial": trial, "sample": sample, "angular_axis": ["x", "y", "z"], "time_s": ("sample", time_s)},
            ),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("angular_axis",),
            param_coord="time_s",
            validate=True,
        )
    )
    velocity = Velocity.from_linear_angular(linear_velocity, angular_velocity, validate=True)

    linear_acceleration = linear_velocity.differentiate(on="time_s", validate=True)
    angular_acceleration = angular_velocity.differentiate(on="time_s", validate=True)
    acceleration = Acceleration.from_linear_angular(linear_acceleration, angular_acceleration, validate=True)

    position = frame_retag(position, parent="tool", child="tip", validate=True)
    rotation = frame_retag(rotation, parent="tool", child="tip", validate=True)
    pose = frame_retag(pose, parent="tool", child="tip", validate=True)
    linear_velocity = frame_retag(linear_velocity, parent="tool", child="tip", validate=True)
    angular_velocity = frame_retag(angular_velocity, parent="tool", child="tip", validate=True)
    velocity = frame_retag(velocity, parent="tool", child="tip", validate=True)
    linear_acceleration = frame_retag(linear_acceleration, parent="tool", child="tip", validate=True)
    angular_acceleration = frame_retag(angular_acceleration, parent="tool", child="tip", validate=True)
    acceleration = frame_retag(acceleration, parent="tool", child="tip", validate=True)

    edge_rot_tool_sensor = frame_retag(
        _rotation_from_values_2d("rotation", _tile_quat(_quat_z(np.pi / 18.0), n_trial=n_trial, n_sample=n_sample), trial, sample, time_s),
        parent="sensor",
        child="tool",
        validate=True,
    )
    edge_rot_sensor_base = frame_retag(
        _rotation_from_values_2d("rotation", _tile_quat(_quat_z(-np.pi / 10.0), n_trial=n_trial, n_sample=n_sample), trial, sample, time_s),
        parent="base",
        child="sensor",
        validate=True,
    )
    edge_rot_base_world = frame_retag(
        _rotation_from_values_2d("rotation", _tile_quat(_quat_z(np.pi / 14.0), n_trial=n_trial, n_sample=n_sample), trial, sample, time_s),
        parent="world",
        child="base",
        validate=True,
    )
    edge_rot_map_world = frame_retag(
        _rotation_from_values_2d("rotation", _tile_quat(_quat_z(0.0), n_trial=n_trial, n_sample=n_sample), trial, sample, time_s),
        parent="world",
        child="map",
        validate=True,
    )

    edge_pose_tool_sensor = frame_retag(
        Pose.from_components(
            edge_rot_tool_sensor,
            _position_from_values_2d(
                "position",
                _tile_vector(np.array([0.15, 0.0, 0.0], dtype=float), n_trial=n_trial, n_sample=n_sample),
                trial,
                sample,
                time_s,
            ),
            validate=True,
        ),
        parent="sensor",
        child="tool",
        validate=True,
    )
    edge_pose_sensor_base = frame_retag(
        Pose.from_components(
            edge_rot_sensor_base,
            _position_from_values_2d(
                "position",
                _tile_vector(np.array([0.2, 0.08, 0.0], dtype=float), n_trial=n_trial, n_sample=n_sample),
                trial,
                sample,
                time_s,
            ),
            validate=True,
        ),
        parent="base",
        child="sensor",
        validate=True,
    )
    edge_pose_base_world = frame_retag(
        Pose.from_components(
            edge_rot_base_world,
            _position_from_values_2d(
                "position",
                _tile_vector(np.array([1.0, 0.0, 0.0], dtype=float), n_trial=n_trial, n_sample=n_sample),
                trial,
                sample,
                time_s,
            ),
            validate=True,
        ),
        parent="world",
        child="base",
        validate=True,
    )
    edge_pose_map_world = frame_retag(
        Pose.from_components(
            edge_rot_map_world,
            _position_from_values_2d(
                "position",
                _tile_vector(np.array([2.0, 1.0, 0.0], dtype=float), n_trial=n_trial, n_sample=n_sample),
                trial,
                sample,
                time_s,
            ),
            validate=True,
        ),
        parent="world",
        child="map",
        validate=True,
    )

    edge_rotation_map = {
        ("tool", "sensor"): edge_rot_tool_sensor,
        ("sensor", "base"): edge_rot_sensor_base,
        ("base", "world"): edge_rot_base_world,
        ("map", "world"): edge_rot_map_world,
    }
    edge_pose_map = {
        ("tool", "sensor"): edge_pose_tool_sensor,
        ("sensor", "base"): edge_pose_sensor_base,
        ("base", "world"): edge_pose_base_world,
        ("map", "world"): edge_pose_map_world,
    }

    for child, parent in [(base, world), (sensor, base), (tool, sensor), (map_frame, world)]:
        set_edge_motion_class(child, "static", parent=parent)
    set_frame_inertial_status(world, "inertial")
    for frame in (base, sensor, tool, map_frame):
        propagate_inertial_status(frame)

    path_opts = PathSolveOptions(graph=graph)

    def edge_rotation_fn(child: Any, parent: Any) -> Rotation:
        return edge_rotation_map[(child.id, parent.id)]

    def edge_pose_fn(child: Any, parent: Any) -> Pose:
        return edge_pose_map[(child.id, parent.id)]

    return {
        "position": position,
        "rotation": rotation,
        "pose": pose,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
        "velocity": velocity,
        "linear_acceleration": linear_acceleration,
        "angular_acceleration": angular_acceleration,
        "acceleration": acceleration,
        "path_opts": path_opts,
        "edge_rotation_fn": edge_rotation_fn,
        "edge_pose_fn": edge_pose_fn,
        "summary": {
            "trials": trial.tolist(),
            "samples": sample.size,
            "time_range_s": (float(time_s[0]), float(time_s[-1])),
        },
    }
