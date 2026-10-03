from __future__ import annotations
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch
import numpy as np
import xarray as xr
from tal_extensions.astro import (
    AstroBackend,
    AstroIERSOptions,
    AstroOptions,
    AstroTimeOptions,
    TopocentricDirection,
)
from tal_extensions.astro.sun import (
    SpiceSunOptions,
    SunDirectionOptions,
    direction_to_sun,
)
from tal.core import (
    AnalysisObject,
)
from tal_extensions.geo import (
    ENUOptions,
    GeodesicOptions,
    GeodeticInterpolationOptions,
    GeodeticOptions,
    GeodeticPosition,
    LocalOrigin,
    ProjectedPosition,
    transform_crs,
)
from tal.linalg import (
    Vector3,
)
from tal.spatial import (
    Position,
)


def example_geo_geodetic_options() -> None:
    opts = GeodeticOptions()
    assert opts.datum == "WGS84"
    assert opts.crs == "EPSG:4979"
    assert opts.ecef_crs == "EPSG:4978"


def example_geo_enu_options() -> None:
    origin = LocalOrigin(45.0, -75.0, 100.0)
    opts = ENUOptions(origin=origin, output_frame="site_enu")
    assert opts.origin == origin
    assert opts.output_frame == "site_enu"


def example_geo_geodesic_options() -> None:
    opts = GeodesicOptions(method="local_enu", local_origin=LocalOrigin(45.0, -75.0))
    assert opts.method == "local_enu"
    assert opts.local_origin is not None


def example_geo_interpolation_options() -> None:
    opts = GeodeticInterpolationOptions(method="ecef_linear", query_dim="target")
    assert opts.method == "ecef_linear"
    assert opts.query_dim == "target"


def example_geo_geodetic_from_lla() -> None:
    ds = xr.Dataset(
        {
            "position": (
                ("sample", "lla"),
                np.asarray([[45.0, -75.0, 100.0]], dtype=float),
            )
        },
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("lla",), validate=True
    )
    opts = GeodeticOptions(longitude_wrap="[0, 360)")
    with patch(
        "tal_extensions.geo.options.normalize_supported_crs",
        lambda value, expected, owner: expected,
    ):
        lla = GeodeticPosition.from_lla(ao, opts=opts)
    geo = lla.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]
    assert geo["kind"] == "geodetic_position"
    assert geo["longitude_wrap"] == "[0, 360)"
    assert list(lla.as_dataset(copy="none")["lla"].values) == ["lat", "lon", "alt"]


def example_geo_geodetic_conversion() -> None:
    from tal_extensions.geo import from_ecef as geo_from_ecef

    ds = xr.Dataset(
        {
            "position": (
                ("sample", "lla"),
                np.asarray([[45.0, -75.0, 100.0]], dtype=float),
            )
        },
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("lla",), validate=True
    )
    opts = GeodeticOptions(longitude_wrap="[0, 360)")
    with (
        patch(
            "tal_extensions.geo.options.normalize_supported_crs",
            lambda value, expected, owner: expected,
        ),
        patch(
            "tal_extensions.geo.conversion.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (
                lat + 1.0,
                lon + 2.0,
                alt + 3.0,
            ),
        ),
        patch(
            "tal_extensions.geo.conversion.transform_ecef_to_lla",
            lambda x, y, z, crs, ecef_crs, owner: (x - 1.0, y - 2.0, z - 3.0),
        ),
    ):
        lla = GeodeticPosition.from_lla(ao)
        ecef = lla.to_ecef(opts=opts)
        roundtrip = GeodeticPosition.from_ecef(ecef, opts=opts)
        via_module = geo_from_ecef(ecef, opts=opts)
    assert list(ecef.as_dataset(copy="none")["axis"].values) == ["x", "y", "z"]
    assert list(roundtrip.as_dataset(copy="none")["lla"].values) == [
        "lat",
        "lon",
        "alt",
    ]
    assert (
        roundtrip.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]["longitude_wrap"]
        == "[0, 360)"
    )
    np.testing.assert_allclose(
        roundtrip.as_dataset(copy="none")["position"],
        lla.as_dataset(copy="none")["position"],
    )
    np.testing.assert_allclose(
        via_module.as_dataset(copy="none")["position"],
        lla.as_dataset(copy="none")["position"],
    )


def example_geo_enu_conversion() -> None:
    from tal_extensions.geo import register_position_accessor

    register_position_accessor()
    ds = xr.Dataset(
        {"position": (("sample", "lla"), np.asarray([[0.0, 1.0, 0.0]], dtype=float))},
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("lla",), validate=True
    )
    origin = LocalOrigin(0.0, 0.0, 0.0)
    opts = ENUOptions(origin=origin, output_frame="site_enu")
    with (
        patch(
            "tal_extensions.geo.options.normalize_supported_crs",
            lambda value, expected, owner: expected,
        ),
        patch(
            "tal_extensions.geo.conversion.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
        ),
        patch(
            "tal_extensions.geo.conversion.transform_ecef_to_lla",
            lambda x, y, z, crs, ecef_crs, owner: (x, y, z),
        ),
        patch(
            "tal_extensions.geo.local.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
        ),
    ):
        lla = GeodeticPosition.from_lla(ao)
        ecef = lla.to_ecef()
        enu = lla.to_enu(opts=opts)
        enu_from_ecef = ecef.geo.to_enu(origin=origin)
        ecef_roundtrip = enu.geo.to_ecef()
        lla_roundtrip = ecef.geo.to_lla()
    assert list(enu.as_dataset(copy="none")["axis"].values) == ["x", "y", "z"]
    assert (
        enu.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]["cartesian_system"]
        == "enu"
    )
    assert (
        enu.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]["origin"]["storage"]
        == "inline"
    )
    np.testing.assert_allclose(
        enu.as_dataset(copy="none")["position"],
        enu_from_ecef.as_dataset(copy="none")["position"],
    )
    np.testing.assert_allclose(
        ecef_roundtrip.as_dataset(copy="none")["position"],
        ecef.as_dataset(copy="none")["position"],
    )
    np.testing.assert_allclose(
        lla_roundtrip.as_dataset(copy="none")["position"],
        lla.as_dataset(copy="none")["position"],
    )


def example_geo_distance_bearing() -> None:
    ds = xr.Dataset(
        {
            "position": (
                ("sample", "lla"),
                np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            )
        },
        coords={"sample": [0, 1], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("lla",), validate=True
    )

    def inverse(lat1, lon1, lat2, lon2, crs, owner):
        shape = np.broadcast_shapes(
            np.shape(lat1), np.shape(lon1), np.shape(lat2), np.shape(lon2)
        )
        return np.full(shape, 90.0), np.full(shape, -90.0), np.abs(lon2 - lon1) * 1000.0

    with (
        patch(
            "tal_extensions.geo.options.normalize_supported_crs",
            lambda value, expected, owner: expected,
        ),
        patch("tal_extensions.geo.distance.geod_inverse", inverse),
    ):
        lla = GeodeticPosition.from_lla(ao)
        distance = lla.distance_to(lla)
        bearing = lla.initial_bearing_to(lla)
    assert (
        tuple(
            distance.as_dataset(copy="none").attrs["tal"]["core"]["roles"]["core_dims"]
        )
        == ()
    )
    assert "distance_m" in distance.as_dataset(copy="none").data_vars
    assert "initial_bearing_deg" in bearing.as_dataset(copy="none").data_vars


def example_geo_interpolation() -> None:
    ds = xr.Dataset(
        {
            "position": (
                ("sample", "lla"),
                np.asarray([[0.0, 170.0, 0.0], [0.0, 190.0, 10.0]], dtype=float),
            )
        },
        coords={
            "sample": [0, 1],
            "lla": ["lat", "lon", "alt"],
            "time_s": ("sample", [0.0, 10.0]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=("lla",),
        param_coord="time_s",
        validate=True,
    )

    def interpolate(lat1, lon1, lat2, lon2, alpha, crs, owner):
        return lat1 + (lat2 - lat1) * alpha, lon1 + (lon2 - lon1) * alpha

    with patch("tal_extensions.geo.interpolation.geod_interpolate", interpolate):
        lla = GeodeticPosition.from_lla(ao)
        out = lla.param.at([5.0], on="time_s")
        nearest = lla.param.resample_to(
            [6.0], on="time_s", opts=GeodeticInterpolationOptions(method="nearest")
        )
        like = lla.param.interp_like(
            nearest, on="time_s", opts=GeodeticInterpolationOptions(method="nearest")
        )
    assert isinstance(out, GeodeticPosition)
    assert isinstance(nearest, GeodeticPosition)
    assert isinstance(like, GeodeticPosition)
    assert (
        out.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]["kind"]
        == "geodetic_position"
    )


def example_geo_crs_transform() -> None:
    ds = xr.Dataset(
        {
            "position": (
                ("sample", "lla"),
                np.asarray([[34.0, -118.0, 20.0]], dtype=float),
            )
        },
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("lla",), validate=True
    )
    projected_ds = xr.Dataset(
        {
            "position": (
                ("sample", "projected"),
                np.asarray([[500000.0, 4100000.0]], dtype=float),
            )
        },
        coords={"sample": [0], "projected": ["easting", "northing"]},
    )
    projected_ao = AnalysisObject.from_data(
        projected_ds,
        sequence_dim="sample",
        core_dims=("projected",),
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
            "tal_extensions.geo.crs_transform.normalize_crs_with_class",
            lambda value, owner, field="crs": SimpleNamespace(
                text=str(value), kind=kind(value, owner, field)
            ),
        ),
        patch(
            "tal_extensions.geo.crs_transform.crs_has_height_axis",
            lambda value, owner, field="crs": False,
        ),
        patch(
            "tal_extensions.geo.crs_transform.base_geodetic_crs",
            lambda value, owner, field="crs": "EPSG:4326",
        ),
        patch("tal_extensions.geo.crs_transform.transform_crs_xyz", xyz),
        patch(
            "tal_extensions.geo.crs_transform.transform_crs_xy",
            lambda x, y, src_crs, dst_crs, owner: (x + 10.0, y + 20.0),
        ),
        patch(
            "tal_extensions.geo.options.normalize_supported_crs",
            lambda value, expected, owner: expected,
        ),
        patch(
            "tal_extensions.geo.metadata.normalize_crs_for_class",
            lambda value, expected, owner, field="crs": str(value),
        ),
        patch(
            "tal_extensions.geo.metadata.base_geodetic_crs",
            lambda value, owner, field="crs": "EPSG:4326",
        ),
    ):
        lla = GeodeticPosition.from_lla(ao)
        projected = lla.to_crs("EPSG:32611")
        direct = transform_crs(lla, dst="EPSG:4978")
        projected_input = ProjectedPosition.from_projected(
            projected_ao, crs="EPSG:32611"
        )
        roundtrip = projected.to_crs("EPSG:4979")
    assert isinstance(projected, ProjectedPosition)
    assert isinstance(projected_input, ProjectedPosition)
    assert isinstance(direct, Position)
    assert isinstance(roundtrip, GeodeticPosition)
    assert list(projected.as_dataset(copy="none")["projected"].values) == [
        "easting",
        "northing",
        "height",
    ]
    assert (
        projected.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]["kind"]
        == "projected_position"
    )


def example_astro_options() -> None:
    backend: AstroBackend = "astropy"
    opts = AstroOptions(
        backend=backend,
        time=AstroTimeOptions(scale="tt", source="utc"),
        iers=AstroIERSOptions(auto_download=False),
    )
    assert get_args(AstroBackend) == ("astropy", "spice")
    assert opts.backend == "astropy"
    assert opts.time is not None
    assert opts.time.scale == "tt"
    assert opts.time.source == "utc"
    assert opts.iers is not None
    assert opts.iers.degraded_accuracy == "error"


def example_astro_topocentric_direction() -> None:
    ds = xr.Dataset(
        {"direction": (("sample", "enu"), np.asarray([[1.0, 0.0, 0.0]], dtype=float))},
        coords={"sample": [0], "enu": ["east", "north", "up"]},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("enu",), validate=True
    )
    direction = TopocentricDirection(ao)
    astro = direction.as_dataset(copy="none").attrs["tal"]["ext"]["astro"]
    assert astro["kind"] == "topocentric_direction"
    assert set(direction.as_dataset(copy="none").data_vars) == {
        "direction",
        "altitude_deg",
        "azimuth_deg",
    }


def example_astro_direction_to_vector3() -> None:
    ds = xr.Dataset(
        {"direction": (("sample", "enu"), np.asarray([[2.0, 3.0, 4.0]], dtype=float))},
        coords={"sample": [0], "enu": ["east", "north", "up"]},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("enu",), validate=True
    )
    vector = TopocentricDirection(ao).to_vector3(axis="axis", output_var="sun")
    assert isinstance(vector, Vector3)
    assert tuple(vector.as_dataset(copy="none").coords["axis"].to_numpy().tolist()) == (
        "x",
        "y",
        "z",
    )
    np.testing.assert_allclose(
        vector.as_dataset(copy="none")["sun"].to_numpy(), np.asarray([[2.0, 3.0, 4.0]])
    )


def example_astro_sun_options() -> None:
    opts = SunDirectionOptions(
        iers=AstroIERSOptions(auto_download=False, degraded_accuracy="ignore")
    )
    assert opts.backend == "astropy"
    assert opts.iers is not None
    assert opts.iers.auto_download is False
    assert SpiceSunOptions() == SpiceSunOptions()


def example_astro_sun_direction() -> None:
    ds = xr.Dataset(
        {
            "lla": (
                ("sample", "lla_axis"),
                np.array([[35.0, -106.0, 1600.0]], dtype=float),
            )
        },
        coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True
    )
    opts = SunDirectionOptions(
        iers=AstroIERSOptions(auto_download=False, degraded_accuracy="ignore")
    )
    sun = direction_to_sun(
        GeodeticPosition.from_lla(ao), time="2024-06-01T12:00:00", opts=opts
    )
    assert set(sun.as_dataset(copy="none").data_vars) == {
        "direction",
        "altitude_deg",
        "azimuth_deg",
    }
    np.testing.assert_allclose(
        np.linalg.norm(sun.as_dataset(copy="none")["direction"].values, axis=-1),
        1.0,
        atol=1e-12,
    )


def example_geo_registration() -> None:
    from tal.spatial import Position
    from tal_extensions.geo import register_position_accessor

    register_position_accessor()
    register_position_accessor()
    assert isinstance(Position.geo, property)


EXECUTABLE_EXAMPLES = {
    "GEO-REGISTRATION": example_geo_registration,
    "GEO-GEODETIC-OPTIONS": example_geo_geodetic_options,
    "GEO-ENU-OPTIONS": example_geo_enu_options,
    "GEO-GEODESIC-OPTIONS": example_geo_geodesic_options,
    "GEO-INTERPOLATION-OPTIONS": example_geo_interpolation_options,
    "GEO-GEODETIC-FROM-LLA": example_geo_geodetic_from_lla,
    "GEO-GEODETIC-CONVERSION": example_geo_geodetic_conversion,
    "GEO-ENU-CONVERSION": example_geo_enu_conversion,
    "GEO-DISTANCE-BEARING": example_geo_distance_bearing,
    "GEO-INTERPOLATION": example_geo_interpolation,
    "GEO-CRS-TRANSFORM": example_geo_crs_transform,
    "ASTRO-OPTIONS": example_astro_options,
    "ASTRO-TOPOCENTRIC-DIRECTION": example_astro_topocentric_direction,
    "ASTRO-DIRECTION-TO-VECTOR3": example_astro_direction_to_vector3,
    "ASTRO-SUN-OPTIONS": example_astro_sun_options,
    "ASTRO-SUN-DIRECTION": example_astro_sun_direction,
}
