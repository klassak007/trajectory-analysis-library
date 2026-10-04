from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import xarray as xr
from tal_extensions.astro import (
    AstroIERSOptions,
    AstroOptions,
    AstroTimeOptions,
    TopocentricDirection,
)
from tal_extensions.astro.sun import SunDirectionOptions, direction_to_sun
from tal.core import (
    AnalysisObject,
)
from tal_extensions.geo import (
    ENUOptions,
    GeodesicOptions,
    GeodeticInterpolationOptions,
    GeodeticPosition,
    LocalOrigin,
    ProjectedPosition,
    transform_crs,
)
from tal.spatial import Position


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
    geo_block = lla.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]
    assert geo_block["kind"] == "geodetic_position"


def example_guide_geo_conversion() -> None:
    from tal_extensions.geo import from_ecef as geo_from_ecef

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
        ecef = lla.to_ecef()
        roundtrip = GeodeticPosition.from_ecef(ecef)
        via_module = geo_from_ecef(ecef)
    assert list(ecef.as_dataset(copy="none")["axis"].values) == ["x", "y", "z"]
    np.testing.assert_allclose(
        roundtrip.as_dataset(copy="none")["position"],
        lla.as_dataset(copy="none")["position"],
    )
    np.testing.assert_allclose(
        via_module.as_dataset(copy="none")["position"],
        lla.as_dataset(copy="none")["position"],
    )


def example_guide_geo_enu() -> None:
    from tal_extensions.geo import register_position_accessor

    register_position_accessor()
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
        patch(
            "tal_extensions.geo.options.normalize_supported_crs",
            lambda value, expected, owner: expected,
        ),
        patch(
            "tal_extensions.geo.conversion.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
        ),
        patch(
            "tal_extensions.geo.local.transform_lla_to_ecef",
            lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
        ),
    ):
        lla = GeodeticPosition.from_lla(ao)
        origin = LocalOrigin(45.0, -75.0, 100.0)
        enu = lla.to_enu(opts=ENUOptions(origin=origin, output_frame="site_enu"))
        ecef_again = enu.geo.to_ecef()
    assert (
        enu.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]["cartesian_system"]
        == "enu"
    )
    assert (
        ecef_again.as_dataset(copy="none").attrs["tal"]["ext"]["geo"][
            "cartesian_system"
        ]
        == "ecef"
    )


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
        shape = np.broadcast_shapes(
            np.shape(lat1), np.shape(lon1), np.shape(lat2), np.shape(lon2)
        )
        return np.full(shape, 90.0), np.full(shape, -90.0), np.zeros(shape)

    with (
        patch(
            "tal_extensions.geo.options.normalize_supported_crs",
            lambda value, expected, owner: expected,
        ),
        patch("tal_extensions.geo.distance.geod_inverse", inverse),
    ):
        lla = GeodeticPosition.from_lla(ao)
        distance = lla.distance_to(lla)
        bearing = lla.initial_bearing_to(lla, opts=GeodesicOptions())
    assert "distance_m" in distance.as_dataset(copy="none").data_vars
    assert "initial_bearing_deg" in bearing.as_dataset(copy="none").data_vars


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

    with patch("tal_extensions.geo.interpolation.geod_interpolate", interpolate):
        lla = GeodeticPosition.from_lla(ao)
        interpolated = lla.param.at(
            [0.0], on="sample", opts=GeodeticInterpolationOptions()
        )
        nearest = lla.param.resample_to(
            [0.0], on="sample", opts=GeodeticInterpolationOptions(method="nearest")
        )
        matched = lla.param.interp_like(
            nearest, on="sample", opts=GeodeticInterpolationOptions(method="nearest")
        )
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
    altitude = direction.as_dataset(copy="none")["altitude_deg"]
    azimuth = direction.as_dataset(copy="none")["azimuth_deg"]
    assert opts.backend == "astropy"
    assert opts.time is not None
    assert opts.time.source == "utc_time"
    assert altitude.dims == ("sample",)
    assert azimuth.dims == ("sample",)


def example_guide_astro_sun_direction() -> None:
    ao = AnalysisObject.from_data(
        xr.Dataset(
            {
                "lla": (
                    ("sample", "lla_axis"),
                    np.array([[35.0, -106.0, 1600.0]], dtype=float),
                )
            },
            coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
        ),
        sequence_dim="sample",
        core_dims=("lla_axis",),
        validate=True,
    )
    opts = SunDirectionOptions(
        iers=AstroIERSOptions(auto_download=False, degraded_accuracy="ignore")
    )
    sun = direction_to_sun(
        GeodeticPosition.from_lla(ao), time="2024-06-01T12:00:00", opts=opts
    )
    sun_xyz = sun.to_vector3()
    assert sun.as_dataset(copy="none")["direction"].dims == ("sample", "enu")
    assert sun_xyz.as_dataset(copy="none")["direction"].dims == ("sample", "axis")
    assert (
        sun.as_dataset(copy="none").attrs["tal"]["ext"]["astro"]["backend"] == "astropy"
    )


USER_GUIDE_EXECUTABLE_EXAMPLES = {
    "UG-GEO-LLA": example_guide_geo_lla,
    "UG-GEO-CONVERSION": example_guide_geo_conversion,
    "UG-GEO-ENU": example_guide_geo_enu,
    "UG-GEO-DISTANCE": example_guide_geo_distance,
    "UG-GEO-INTERPOLATION": example_guide_geo_interpolation,
    "UG-GEO-CRS": example_guide_geo_crs,
    "UG-ASTRO-OPTIONS": example_guide_astro_foundation,
    "UG-ASTRO-DIRECTION": example_guide_astro_sun_direction,
}
