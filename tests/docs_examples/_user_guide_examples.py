from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr

from tal import ufuncs
from tal.astro import AstroIERSOptions, AstroOptions, AstroTimeOptions, TopocentricDirection
from tal.astro.sun import SunDirectionOptions, direction_to_sun
from tal.core import AnalysisObject, ParamEvalOptions, ParamSyncOptions, synchronize
from tal.core.event_ops import (
    AroundOptions,
    AtBoundariesOptions,
    Condition,
    ConditionEvalOptions,
    EventExtractOptions,
    IntervalExtractOptions,
    WhenOptions,
)
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.frames import FrameGraph, find_path, fold_path, render_snapshot_ascii, snapshot_from_seeds
from tal.geo import (
    ENUOptions,
    GeodeticInterpolationOptions,
    GeodesicOptions,
    GeodeticPosition,
    LocalOrigin,
    ProjectedPosition,
    transform_crs,
)
from tal.linalg import Matrix, Vector, Vector3, add, dot, inv, matmul, norm, pinv, solve, sub
from tal.spatial import Pose, Position, Rotation


def _scalar_signal_ao() -> AnalysisObject:
    sample = np.arange(6)
    time_s = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7])
    return AnalysisObject.from_data(
        xr.Dataset(
            {"signal": (("trial", "sample"), [[0.0, 1.0, 0.0, 2.0, 1.0, 0.0]])},
            coords={
                "trial": ["t0"],
                "sample": sample,
                "time_s": (("trial", "sample"), time_s[None, :]),
                "group_size": ("trial", np.array([sample.size], dtype=np.int64)),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )


def _event_ao() -> AnalysisObject:
    sample = np.arange(8)
    time_s = np.linspace(0.0, 0.7, sample.size)
    speed = np.array([[0.0, 1.0, 2.0, 5.5, 6.2, 4.0, 1.0, 0.0]])
    return AnalysisObject.from_data(
        xr.Dataset(
            {"speed_mps": (("trial", "sample"), speed)},
            coords={
                "trial": ["flight_0"],
                "sample": sample,
                "time_s": (("trial", "sample"), time_s[None, :]),
                "group_size": ("trial", np.array([sample.size], dtype=np.int64)),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )


def _make_position_rotation() -> tuple[Position, Rotation]:
    sample = np.arange(3)
    time_s = np.array([0.0, 0.5, 1.0])
    pos = Position(AnalysisObject.from_data(
        xr.Dataset(
            {"position": (("sample", "axis"), [[1.0, 0.0, 0.0], [1.5, 0.5, 0.0], [2.0, 1.0, 0.0]])},
            coords={"sample": sample, "axis": ["x", "y", "z"], "time_s": ("sample", time_s)},
        ),
        sequence_dim="sample",
        core_dims=("axis",),
        param_coord="time_s",
        validate=True,
    ))
    rot = Rotation(AnalysisObject.from_data(
        xr.Dataset(
            {
                "rotation": (
                    ("sample", "quat"),
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.3826834, 0.9238795],
                        [0.0, 0.0, 0.7071068, 0.7071068],
                    ],
                )
            },
            coords={"sample": sample, "quat": ["x", "y", "z", "w"], "time_s": ("sample", time_s)},
        ),
        sequence_dim="sample",
        core_dims=("quat",),
        param_coord="time_s",
        validate=True,
    ))
    return pos, rot


def example_guide_overview_basic_workflow() -> None:
    time = np.linspace(0.0, 4.0, 5)
    values = np.zeros((2, 5, 3))
    values[:, :, 0] = time
    values[:, :, 1] = time**2
    values[:, :, 2] = 1.0

    ds = xr.Dataset(
        {"position": (("run", "sample", "axis"), values)},
        coords={
            "run": ["run_a", "run_b"],
            "sample": np.arange(time.size),
            "time": ("sample", time),
            "axis": ["x", "y", "z"],
        },
    )

    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("run",),
        core_dims=("axis",),
        param_coord="time",
        validate=True,
    )
    resampled = ao.param.at([0.5, 1.5, 2.5], on="time")
    mean_position = ao.mean(dim="sample")
    assert resampled.unsafe_data.sizes["sample"] == 3
    assert mean_position.unsafe_data["position"].dims == ("run", "axis")


def example_guide_core_concepts_roles() -> None:
    sample = np.arange(3)
    time_s = np.array([0.0, 0.1, 0.2])
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"speed": (("trial", "sample"), [[0.0, 1.0, 2.0], [0.5, 1.5, 0.0]])},
            coords={
                "trial": ["a", "b"],
                "sample": sample,
                "time_s": (("trial", "sample"), np.broadcast_to(time_s, (2, 3))),
                "group_size": ("trial", np.array([3, 2], dtype=np.int64)),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )
    declared, sequence_dim, batch_dims, core_dims = read_roles(ao.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ()
    assert read_param_coord_name(ao.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(ao.unsafe_data) == "group_size"


def example_guide_creating_sequence_ao() -> None:
    sample = np.arange(5)
    time_s = np.linspace(0.0, 0.4, sample.size)
    values = np.stack([
        np.column_stack([time_s, time_s**2, np.zeros_like(time_s)]),
        np.column_stack([time_s + 1.0, time_s**2, np.ones_like(time_s)]),
    ])
    core_only = AnalysisObject.from_data(
        xr.Dataset({"value": ("axis", np.array([1.0, 2.0, 3.0]))}, coords={"axis": ["x", "y", "z"]}),
        core_dims=("axis",),
        validate=True,
    )
    batch_core = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("trial", "axis"), [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])},
            coords={"trial": ["a", "b"], "axis": ["x", "y", "z"]},
        ),
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    full = AnalysisObject.from_data(
        xr.Dataset(
            {"position": (("trial", "sample", "axis"), values)},
            coords={
                "trial": ["a", "b"],
                "sample": sample,
                "axis": ["x", "y", "z"],
                "time_s": (("trial", "sample"), np.broadcast_to(time_s, (2, sample.size))),
                "group_size": ("trial", np.array([5, 4], dtype=np.int64)),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )
    updated = AnalysisObject(
        xr.Dataset(
            {"position": (("trial", "sample", "axis"), values)},
            coords={
                "trial": ["a", "b"],
                "sample": sample,
                "axis": ["x", "y", "z"],
                "time_s": (("trial", "sample"), np.broadcast_to(time_s, (2, sample.size))),
                "group_size": ("trial", np.array([5, 4], dtype=np.int64)),
            },
        )
    )
    updated = updated.set_roles(sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",), validate=True)
    updated = updated.set_param_coord(name="time_s", validate=True)
    updated = updated.set_validity(sequence_size_coord="group_size", validate=True)
    assert core_only.unsafe_data["value"].dims == ("axis",)
    assert batch_core.unsafe_data["value"].dims == ("trial", "axis")
    assert full.unsafe_data["position"].dims == ("trial", "sample", "axis")
    assert read_param_coord_name(updated.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(updated.unsafe_data) == "group_size"
    assert full.to_dataarray(name="position").name == "position"


def example_guide_indexing_param_query() -> None:
    ao = _scalar_signal_ao()
    head = ao.isel(sample=slice(0, 3))
    trial_t0 = ao.sel(trial="t0")
    positive = ao.where(ao.unsafe_data["signal"] > 0.0)
    index = ao.param.index([0.18, 0.52], on="time_s")
    nearest = ao.param.sel([0.18, 0.52], on="time_s")
    interp = ao.param.at([0.18, 0.52], on="time_s", opts=ParamEvalOptions(method="linear"))
    assert head.unsafe_data.sizes["sample"] == 3
    assert trial_t0.unsafe_data.sizes["sample"] == 6
    assert positive.unsafe_data["signal"].isnull().any()
    assert index.sizes["query"] == 2
    assert nearest.unsafe_data.sizes["sample"] == 2
    assert interp.unsafe_data.sizes["sample"] == 2


def example_guide_time_synchronize() -> None:
    imu_t = np.linspace(0.0, 1.0, 11)
    gps_t = np.linspace(0.0, 1.0, 3)
    imu = _time_source("accel", np.sin(2.0 * np.pi * imu_t), imu_t)
    gps = _time_source("speed", 0.5 + 0.1 * np.cos(2.0 * np.pi * gps_t), gps_t)
    imu_at = imu.param.at([0.05, 0.25, 0.75], on="time_s", opts=ParamEvalOptions(method="linear", query_dim="query"))
    imu_rs = imu.param.resample_to(np.linspace(0.0, 1.0, 6), on="time_s")
    gps_on_imu = gps.param.interp_like(imu, on="time_s", batch_join="inner")
    synced_imu, synced_gps = synchronize(
        [imu, gps],
        on="time_s",
        opts=ParamSyncOptions(join="domain", how="interp", batch_join="inner", query_dim="query"),
    )
    assert imu_at.unsafe_data.sizes["sample"] == 3
    assert imu_rs.unsafe_data.sizes["sample"] == 6
    assert gps_on_imu.unsafe_data.sizes["sample"] == imu.unsafe_data.sizes["sample"]
    np.testing.assert_allclose(synced_imu.unsafe_data.coords["time_s"], synced_gps.unsafe_data.coords["time_s"])


def _time_source(var_name: str, values: np.ndarray, time_s: np.ndarray) -> AnalysisObject:
    return AnalysisObject.from_data(
        xr.Dataset(
            {var_name: (("trial", "sample"), values[None, :])},
            coords={
                "trial": ["run_0"],
                "sample": np.arange(time_s.size),
                "time_s": (("trial", "sample"), time_s[None, :]),
                "group_size": ("trial", np.array([time_s.size], dtype=np.int64)),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )


def example_guide_events_windows() -> None:
    ao = _event_ao()
    fast = Condition.compare(Condition.var("speed_mps"), "gt", 5.0)
    eval_opts = ConditionEvalOptions(coord_name="time_s")
    mask = ao.events.mask(fast, opts=eval_opts)
    events = ao.events.events(fast, opts=EventExtractOptions(eval=eval_opts))
    intervals = ao.events.intervals(fast, opts=IntervalExtractOptions(eval=eval_opts))
    boundaries = ao.events.at_boundaries(fast, opts=AtBoundariesOptions(eval=eval_opts, edges="enter"))
    masked = ao.events.when(fast, opts=WhenOptions(eval=eval_opts, layout="mask"))
    stream = ao.events.when(fast, opts=WhenOptions(eval=eval_opts, layout="stream"))
    around = ao.events.around(fast, opts=AroundOptions(eval=eval_opts, edge="enter", pre=0.1, post=0.2, dt=0.1))
    around_stacked = ao.events.around(
        fast,
        opts=AroundOptions(layout="stacked", eval=eval_opts, edge="enter", pre=0.1, post=0.2, dt=0.1),
    )
    assert mask.dtype == bool
    assert "event" in events.dims
    assert "segment" in intervals.dims
    assert "event_edge_code" in boundaries.unsafe_data.coords
    assert masked.unsafe_data["speed_mps"].dims == ao.unsafe_data["speed_mps"].dims
    assert stream.unsafe_data.sizes["stream_sample"] >= 1
    assert around.unsafe_data.sizes["tau"] >= 1
    assert "window_event_index" in around_stacked.unsafe_data.coords


def example_guide_linalg_basic() -> None:
    A, v = _matrix_vector_pair()
    x_component = AnalysisObject.from_data(
        xr.Dataset({"x": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    Av = A @ v
    Av2 = matmul(A, v)
    sum_v = add(v, v)
    diff_v = sub(v, v)
    energy = dot(v, v)
    energy2 = v.dot(v)
    mag = norm(v)
    mag2 = v.norm()
    rhs = _solve_rhs()
    x = solve(A, rhs)
    x2 = A.solve(rhs)
    Ainv = inv(A)
    Ainv2 = A.inv()
    Apinv = pinv(A)
    Apinv2 = A.pinv()
    vec3 = Vector3.from_xyz(x_component, 0.0, 1.0, axis="axis", output_var="vec3")
    np.testing.assert_allclose(Av.unsafe_data["datavar"], Av2.unsafe_data["datavar"])
    np.testing.assert_allclose(energy.unsafe_data["datavar"], energy2.unsafe_data["datavar"])
    np.testing.assert_allclose(mag.unsafe_data["datavar"], mag2.unsafe_data["datavar"])
    assert energy.unsafe_data["datavar"].dims == ("sample",)
    assert mag.unsafe_data["datavar"].dims == ("sample",)
    assert x.unsafe_data["datavar"].dims == ("sample", "col")
    assert x2.unsafe_data["datavar"].dims == ("sample", "col")
    assert sum_v.unsafe_data["datavar"].dims == ("sample", "col")
    assert diff_v.unsafe_data["datavar"].dims == ("sample", "col")
    assert Ainv.unsafe_data["datavar"].shape[-2:] == (2, 2)
    assert Ainv2.unsafe_data["datavar"].shape[-2:] == (2, 2)
    assert Apinv.unsafe_data["datavar"].shape[-2:] == (2, 2)
    assert Apinv2.unsafe_data["datavar"].shape[-2:] == (2, 2)
    assert tuple(vec3.unsafe_data.coords["axis"].to_numpy().tolist()) == ("x", "y", "z")


def _matrix_vector_pair() -> tuple[Matrix, Vector]:
    A = Matrix(AnalysisObject.from_data(
        xr.Dataset(
            {"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 3.0]], [[1.0, 1.0], [0.0, 2.0]]])},
            coords={"sample": [0, 1], "row": ["r0", "r1"], "col": ["c0", "c1"]},
        ),
        sequence_dim="sample",
        core_dims=("row", "col"),
        validate=True,
    ))
    v = Vector(AnalysisObject.from_data(
        xr.Dataset(
            {"v": (("sample", "col"), [[4.0, 5.0], [2.0, 8.0]])},
            coords={"sample": [0, 1], "col": ["c0", "c1"]},
        ),
        sequence_dim="sample",
        core_dims=("col",),
        validate=True,
    ))
    return A, v


def _solve_rhs() -> Vector:
    return Vector(AnalysisObject.from_data(
        xr.Dataset(
            {"rhs": (("sample", "row"), [[4.0, 5.0], [2.0, 8.0]])},
            coords={"sample": [0, 1], "row": ["r0", "r1"]},
        ),
        sequence_dim="sample",
        core_dims=("row",),
        validate=True,
    ))


def example_guide_numpy_ufuncs() -> None:
    x = _numeric_ao([[0.0, 1.0], [2.0, 3.0]])
    y = _numeric_ao([[1.0, 2.0], [3.0, 4.0]])
    bias = AnalysisObject.from_data(
        xr.Dataset({"value": ("axis", [10.0, 20.0])}, coords={"axis": ["a", "b"]}),
        core_dims=("axis",),
        validate=True,
    )
    sin_x = ufuncs.sin(x)
    exp_x = ufuncs.exp(x)
    sum_xy = x + y
    aligned_xy = x.a(on="sequence", sequence_join="inner") + y
    broadcast_bias = x + bias.b()
    condition = ufuncs.greater(x, 1.0)
    np.testing.assert_allclose(sin_x.unsafe_data["value"], np.sin(x.unsafe_data["value"]))
    np.testing.assert_allclose(exp_x.unsafe_data["value"], np.exp(x.unsafe_data["value"]))
    np.testing.assert_allclose(sum_xy.unsafe_data["value"], x.unsafe_data["value"] + y.unsafe_data["value"])
    np.testing.assert_allclose(aligned_xy.unsafe_data["value"], x.unsafe_data["value"] + y.unsafe_data["value"])
    np.testing.assert_allclose(
        broadcast_bias.unsafe_data["value"],
        x.unsafe_data["value"] + np.asarray([10.0, 20.0]),
    )
    assert isinstance(condition, Condition)


def _numeric_ao(values: list[list[float]]) -> AnalysisObject:
    return AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("sample", "axis"), values)},
            coords={"sample": [0, 1], "axis": ["a", "b"]},
        ),
        sequence_dim="sample",
        core_dims=("axis",),
        validate=True,
    )


def example_guide_spatial_pose() -> None:
    pos, rot = _make_position_rotation()
    pose = Pose.from_components(rot, pos)
    out_pos, out_rot = pose.decompose()
    identity_like = pose.compose(pose.inverse())
    rotated = rot.apply(pos)
    transformed = pose.apply(pos)
    rot_m = rot.as_matrix()
    rot_q = rot_m.as_quat()
    pose_m = pose.as_matrix()
    rot_at = rot.param.at([0.25], on="time_s")
    pose_rs = pose.param.resample_to(np.linspace(0.0, 1.0, 5), on="time_s")
    assert isinstance(out_pos, Position)
    assert isinstance(out_rot, Rotation)
    assert isinstance(identity_like, Pose)
    assert isinstance(rotated, Position)
    assert isinstance(transformed, Position)
    assert isinstance(rot_q, Rotation)
    assert pose_m.unsafe_data["pose_matrix"].shape[-2:] == (4, 4)
    assert rot_at.unsafe_data.sizes["sample"] == 1
    assert pose_rs.unsafe_data.sizes["sample"] == 5


def example_guide_geo_lla() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"position": (("sample", "lla"), np.array([[45.0, -75.0, 100.0]]))},
            coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ),
        sequence_dim="sample",
        core_dims=("lla",),
        validate=True,
    )
    lla = GeodeticPosition.from_lla(ao)
    geo_block = lla.unsafe_data.attrs["tal"]["ext"]["geo"]
    assert geo_block["kind"] == "geodetic_position"


def example_guide_geo_conversion() -> None:
    from tal.geo import from_ecef as geo_from_ecef

    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"position": (("sample", "lla"), np.array([[45.0, -75.0, 100.0]]))},
            coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ),
        sequence_dim="sample",
        core_dims=("lla",),
        validate=True,
    )
    with (
        patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected),
        patch(
            "tal.geo.conversion.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat + 1.0, lon + 2.0, alt + 3.0),
        ),
        patch(
            "tal.geo.conversion.transform_ecef_to_lla",
            lambda x, y, z, crs, ecef_crs, owner: (x - 1.0, y - 2.0, z - 3.0),
        ),
    ):
        lla = GeodeticPosition.from_lla(ao)
        ecef = lla.to_ecef()
        roundtrip = GeodeticPosition.from_ecef(ecef)
        via_module = geo_from_ecef(ecef)
    assert list(ecef.unsafe_data["axis"].values) == ["x", "y", "z"]
    np.testing.assert_allclose(roundtrip.unsafe_data["position"], lla.unsafe_data["position"])
    np.testing.assert_allclose(via_module.unsafe_data["position"], lla.unsafe_data["position"])


def example_guide_geo_enu() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"position": (("sample", "lla"), np.array([[45.0, -75.0, 100.0]]))},
            coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ),
        sequence_dim="sample",
        core_dims=("lla",),
        validate=True,
    )
    with (
        patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected),
        patch(
            "tal.geo.conversion.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
        ),
        patch(
            "tal.geo.local.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
        ),
    ):
        lla = GeodeticPosition.from_lla(ao)
        origin = LocalOrigin(45.0, -75.0, 100.0)
        enu = lla.to_enu(opts=ENUOptions(origin=origin, output_frame="site_enu"))
        ecef_again = enu.geo.to_ecef()
    assert enu.unsafe_data.attrs["tal"]["ext"]["geo"]["cartesian_system"] == "enu"
    assert ecef_again.unsafe_data.attrs["tal"]["ext"]["geo"]["cartesian_system"] == "ecef"


def example_guide_geo_distance() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"position": (("sample", "lla"), np.array([[45.0, -75.0, 100.0]]))},
            coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ),
        sequence_dim="sample",
        core_dims=("lla",),
        validate=True,
    )

    def inverse(lat1, lon1, lat2, lon2, crs, owner):
        shape = np.broadcast_shapes(np.shape(lat1), np.shape(lon1), np.shape(lat2), np.shape(lon2))
        return np.full(shape, 90.0), np.full(shape, -90.0), np.zeros(shape)

    with (
        patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected),
        patch("tal.geo.distance.geod_inverse", inverse),
    ):
        lla = GeodeticPosition.from_lla(ao)
        distance = lla.distance_to(lla)
        bearing = lla.initial_bearing_to(lla, opts=GeodesicOptions())
    assert "distance_m" in distance.unsafe_data.data_vars
    assert "initial_bearing_deg" in bearing.unsafe_data.data_vars


def example_guide_geo_interpolation() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"position": (("sample", "lla"), np.array([[45.0, -75.0, 100.0]]))},
            coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ),
        sequence_dim="sample",
        core_dims=("lla",),
        validate=True,
    )

    def interpolate(lat1, lon1, lat2, lon2, alpha, crs, owner):
        return lat1, lon1

    with patch("tal.geo.interpolation.geod_interpolate", interpolate):
        lla = GeodeticPosition.from_lla(ao)
        interpolated = lla.param.at([0.0], on="sample", opts=GeodeticInterpolationOptions())
        nearest = lla.param.resample_to([0.0], on="sample", opts=GeodeticInterpolationOptions(method="nearest"))
        matched = lla.param.interp_like(nearest, on="sample", opts=GeodeticInterpolationOptions(method="nearest"))
    assert isinstance(interpolated, GeodeticPosition)
    assert isinstance(nearest, GeodeticPosition)
    assert isinstance(matched, GeodeticPosition)


def example_guide_geo_crs() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"position": (("sample", "lla"), np.array([[34.0, -118.0, 20.0]]))},
            coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
        ),
        sequence_dim="sample",
        core_dims=("lla",),
        validate=True,
    )

    def kind(value, owner, field="crs"):
        if str(value).endswith("4978"):
            return "geocentric"
        if str(value).endswith("32611"):
            return "projected"
        return "geographic"

    def xyz(x, y, z, src_crs, dst_crs, owner):
        if dst_crs == "EPSG:32611":
            return x + 1000.0, y + 2000.0, z
        if dst_crs == "EPSG:4979":
            return x - 1000.0, y - 2000.0, z
        return x + 1.0, y + 2.0, z + 3.0

    with (
        patch(
            "tal.geo.crs_transform.normalize_crs_with_class",
            lambda value, owner, field="crs": SimpleNamespace(text=str(value), kind=kind(value, owner, field)),
        ),
        patch("tal.geo.crs_transform.crs_has_height_axis", lambda value, owner, field="crs": False),
        patch("tal.geo.crs_transform.base_geodetic_crs", lambda value, owner, field="crs": "EPSG:4326"),
        patch("tal.geo.crs_transform.transform_crs_xyz", xyz),
        patch("tal.geo.options.normalize_supported_crs", lambda value, expected, owner: expected),
        patch("tal.geo.metadata.normalize_crs_for_class", lambda value, expected, owner, field="crs": str(value)),
        patch("tal.geo.metadata.base_geodetic_crs", lambda value, owner, field="crs": "EPSG:4326"),
    ):
        lla = GeodeticPosition.from_lla(ao)
        projected = lla.to_crs("EPSG:32611")
        ecef = transform_crs(lla, dst="EPSG:4978")
        roundtrip = projected.to_crs("EPSG:4979")
    assert isinstance(projected, ProjectedPosition)
    assert isinstance(ecef, Position)
    assert isinstance(roundtrip, GeodeticPosition)


def example_guide_astro_foundation() -> None:
    opts = AstroOptions(time=AstroTimeOptions(scale="utc", source="utc_time"))
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"direction": (("sample", "enu"), np.array([[1.0, 0.0, 0.0]]))},
            coords={"sample": [0], "enu": ["east", "north", "up"]},
        ),
        sequence_dim="sample",
        core_dims=("enu",),
        validate=True,
    )
    direction = TopocentricDirection(ao)
    altitude = direction.unsafe_data["altitude_deg"]
    azimuth = direction.unsafe_data["azimuth_deg"]
    assert opts.backend == "astropy"
    assert opts.time is not None
    assert opts.time.source == "utc_time"
    assert altitude.dims == ("sample",)
    assert azimuth.dims == ("sample",)


def example_guide_astro_sun_direction() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"lla": (("sample", "lla_axis"), np.array([[35.0, -106.0, 1600.0]], dtype=float))},
            coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
        ),
        sequence_dim="sample",
        core_dims=("lla_axis",),
        validate=True,
    )
    opts = SunDirectionOptions(iers=AstroIERSOptions(auto_download=False, degraded_accuracy="ignore"))
    sun = direction_to_sun(GeodeticPosition.from_lla(ao), time="2024-06-01T12:00:00", opts=opts)
    sun_xyz = sun.to_vector3()
    assert sun.unsafe_data["direction"].dims == ("sample", "enu")
    assert sun_xyz.unsafe_data["direction"].dims == ("sample", "axis")
    assert sun.unsafe_data.attrs["tal"]["ext"]["astro"]["backend"] == "astropy"


def example_guide_frames_basic() -> None:
    with FrameGraph() as graph:
        world = graph.get_or_create_frame("world")
        ship = graph.get_or_create_frame("ship", parent=world)
        drone = graph.get_or_create_frame("drone", parent=ship)
    path = find_path(drone, world)
    steps = fold_path(
        path,
        edge_value_fn=lambda child, parent: [(child.id, parent.id)],
        compose=lambda acc, value: acc + value,
        inverse=lambda value: [(value[0][1], value[0][0])],
        identity=lambda: [],
    )
    snapshot = snapshot_from_seeds(("drone",), graph=graph)
    ascii_tree = render_snapshot_ascii(snapshot)
    ao = AnalysisObject.from_data(
        xr.Dataset({"value": ("sample", np.array([1.0, 2.0]))}, coords={"sample": [0, 1]}),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    )
    retagged = ao.frames.retag(parent="ship", child="drone")
    parent, child = retagged.frames.ids()
    bound_parent, bound_child = retagged.frames.bind(graph=graph, create_missing=True)
    assert [node.id for node in path.nodes] == ["drone", "ship", "world"]
    assert steps == [("drone", "ship"), ("ship", "world")]
    assert "drone" in ascii_tree
    assert (parent, child) == ("ship", "drone")
    assert bound_parent is ship
    assert bound_child.id == "drone"
    renamed = retagged.frames.rename_frame("drone", "drone_0", graph=graph)
    assert renamed.frames.ids() == ("ship", "drone_0")


def example_guide_viewing_schema() -> None:
    sample = np.arange(3)
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {"value": ("sample", np.array([1.0, 2.0, 3.0]))},
            coords={"sample": sample, "time_s": ("sample", np.array([0.0, 0.1, 0.2]))},
        ),
        sequence_dim="sample",
        core_dims=(),
        param_coord="time_s",
        validate=True,
    )
    out = ao.param.at([0.05, 0.15], on="time_s")
    safe_snapshot = ao.data
    backing_store = ao.unsafe_data
    roles = read_roles(backing_store)
    param_name = read_param_coord_name(backing_store)
    before_schema = ao.unsafe_data.attrs["tal"]
    after_schema = out.unsafe_data.attrs["tal"]
    assert safe_snapshot is not backing_store
    assert roles[1] == "sample"
    assert param_name == "time_s"
    assert before_schema["core"]["param_coord"]["name"] == "time_s"
    assert after_schema["core"]["param_coord"]["name"] == "time_s"


USER_GUIDE_EXECUTABLE_EXAMPLES: dict[str, Callable[[], None]] = {
    "UG-OVERVIEW-BASIC-WORKFLOW": example_guide_overview_basic_workflow,
    "UG-CORE-CONCEPTS-ROLES": example_guide_core_concepts_roles,
    "UG-CREATING-SEQUENCE-AO": example_guide_creating_sequence_ao,
    "UG-INDEXING-PARAM-QUERY": example_guide_indexing_param_query,
    "UG-TIME-SYNCHRONIZE": example_guide_time_synchronize,
    "UG-EVENTS-WINDOWS": example_guide_events_windows,
    "UG-LINALG-BASIC": example_guide_linalg_basic,
    "UG-NUMPY-UFUNCS": example_guide_numpy_ufuncs,
    "UG-SPATIAL-POSE": example_guide_spatial_pose,
    "UG-GEO-LLA": example_guide_geo_lla,
    "UG-GEO-CONVERSION": example_guide_geo_conversion,
    "UG-GEO-ENU": example_guide_geo_enu,
    "UG-GEO-DISTANCE": example_guide_geo_distance,
    "UG-GEO-INTERPOLATION": example_guide_geo_interpolation,
    "UG-GEO-CRS": example_guide_geo_crs,
    "UG-ASTRO-OPTIONS": example_guide_astro_foundation,
    "UG-ASTRO-DIRECTION": example_guide_astro_sun_direction,
    "UG-FRAMES-BASIC": example_guide_frames_basic,
    "UG-VIEWING-SCHEMA": example_guide_viewing_schema,
}
