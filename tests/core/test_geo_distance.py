from __future__ import annotations

import dask.array as da
from dask.base import is_dask_collection
import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.geo import GeodesicOptions, GeodeticPosition, LocalOrigin
from tal.linalg import Array


def _lla(values: np.ndarray | None = None, *, times: list[float] | None = None) -> GeodeticPosition:
    if values is None:
        values = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 10.0]], dtype="float64")
    if times is None:
        times = [float(i) for i in range(values.shape[0])]
    ds = xr.Dataset(
        {"position": (("sample", "lla"), values)},
        coords={
            "sample": np.arange(values.shape[0]),
            "lla": ["lat", "lon", "alt"],
            "time_s": ("sample", times),
            "group_size": np.asarray(values.shape[0], dtype=np.int64),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=("lla",),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )
    return GeodeticPosition.from_lla(ao)


def _stub_geod(monkeypatch: pytest.MonkeyPatch) -> None:
    import tal.geo.distance as distance
    import tal.geo.options as options

    monkeypatch.setattr(options, "normalize_supported_crs", lambda value, expected, owner: expected)

    def inverse(lat1, lon1, lat2, lon2, crs, owner):
        shape = np.broadcast_shapes(np.shape(lat1), np.shape(lon1), np.shape(lat2), np.shape(lon2))
        az12 = np.full(shape, 90.0, dtype="float64")
        az21 = np.full(shape, -90.0, dtype="float64")
        dist = np.hypot(np.asarray(lat2) - np.asarray(lat1), np.asarray(lon2) - np.asarray(lon1)) * 1000.0
        return az12, az21, dist

    monkeypatch.setattr(distance, "geod_inverse", inverse)


def _stub_conversions(monkeypatch: pytest.MonkeyPatch) -> None:
    import tal.geo.conversion as conversion
    import tal.geo.local as local
    import tal.geo.options as options

    monkeypatch.setattr(options, "normalize_supported_crs", lambda value, expected, owner: expected)
    monkeypatch.setattr(conversion, "transform_lla_to_ecef", lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt))
    monkeypatch.setattr(conversion, "transform_ecef_to_lla", lambda x, y, z, crs, ecef_crs, owner: (x, y, z))
    monkeypatch.setattr(local, "transform_lla_to_ecef", lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt))


def test_geo_core_g3_001_geodesic_distance_known_wgs84_fixture() -> None:
    """ID: GEO_CORE_G3_001_geodesic_distance_known_wgs84_fixture."""
    pytest.importorskip("pyproj")
    left = _lla(np.asarray([[0.0, 0.0, 0.0]], dtype="float64"))
    right = _lla(np.asarray([[0.0, 1.0, 0.0]], dtype="float64"))

    out = left.distance_to(right)

    np.testing.assert_allclose(out.unsafe_data["distance_m"], [111319.49079327357], rtol=0.0, atol=1e-6)


def test_geo_core_g3_002_geodesic_initial_bearing_known_fixture() -> None:
    """ID: GEO_CORE_G3_002_geodesic_initial_bearing_known_fixture."""
    pytest.importorskip("pyproj")
    left = _lla(np.asarray([[0.0, 0.0, 0.0]], dtype="float64"))
    right = _lla(np.asarray([[0.0, 1.0, 0.0]], dtype="float64"))

    out = left.initial_bearing_to(right)

    np.testing.assert_allclose(out.unsafe_data["initial_bearing_deg"], [90.0], atol=1e-10)


def test_geo_core_g3_003_distance_aligns_batch_and_sequence_by_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_003_distance_aligns_batch_and_sequence_by_topology."""
    _stub_geod(monkeypatch)
    left = _lla()
    right = _lla(np.asarray([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]], dtype="float64"))

    out = left.distance_to(right)

    assert isinstance(out, Array)
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ()
    assert core_dims == ()
    assert read_param_coord_name(out.unsafe_data) == "time_s"
    np.testing.assert_allclose(out.unsafe_data["distance_m"], [1000.0, 2000.0])
    assert "ext" not in out.unsafe_data.attrs["tal"] or "geo" not in out.unsafe_data.attrs["tal"]["ext"]


def test_geo_core_g3_004_distance_preserves_validity_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_004_distance_preserves_validity_semantics."""
    _stub_geod(monkeypatch)

    out = _lla().distance_to(_lla())

    assert read_sequence_size_coord_name(out.unsafe_data) == "group_size"


def test_geo_core_g3_005_local_enu_distance_requires_origin() -> None:
    """ID: GEO_CORE_G3_005_local_enu_distance_requires_origin."""
    with pytest.raises(ValueError, match="local_origin"):
        _lla().distance_to(_lla(), opts=GeodesicOptions(method="local_enu"))


def test_geo_distance_local_enu_approximation_and_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local ENU distance and bearing use explicit tangent-plane semantics."""
    _stub_conversions(monkeypatch)
    left = _lla(np.asarray([[0.0, 0.0, 0.0]], dtype="float64"))
    right = _lla(np.asarray([[0.0, 3.0, 4.0]], dtype="float64"))
    opts = GeodesicOptions(method="local_enu", local_origin=LocalOrigin(0.0, 0.0, 0.0))

    distance = left.distance_to(right, opts=opts)
    bearing = left.initial_bearing_to(right, opts=opts)

    np.testing.assert_allclose(distance.unsafe_data["distance_m"], [5.0])
    np.testing.assert_allclose(bearing.unsafe_data["initial_bearing_deg"], [np.degrees(np.arctan2(3.0, 4.0))])


def test_geo_distance_include_altitude_endpoint_adjustment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Altitude inclusion is endpoint hypotenuse adjustment, not path integration."""
    _stub_geod(monkeypatch)
    left = _lla(np.asarray([[0.0, 0.0, 0.0]], dtype="float64"))
    right = _lla(np.asarray([[0.0, 1.0, 10.0]], dtype="float64"))

    out = left.distance_to(right, opts=GeodesicOptions(include_altitude=True))

    np.testing.assert_allclose(out.unsafe_data["distance_m"], [np.hypot(1000.0, 10.0)])


def test_geo_distance_preserves_dask_payload_laziness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Payload math remains lazy across the geodesic apply_ufunc boundary."""
    _stub_geod(monkeypatch)
    source = _lla()
    ds = source.unsafe_data.copy(deep=True)
    ds["position"] = ds["position"].copy(data=da.from_array(ds["position"].data, chunks=(1, 3)))

    out = GeodeticPosition(ds).distance_to(GeodeticPosition(ds), validate=False)

    assert is_dask_collection(out.unsafe_data["distance_m"].data)


def test_geo_hard_g3_001_distance_rejects_incompatible_geo_metadata() -> None:
    """ID: GEO_HARD_G3_001_distance_rejects_incompatible_geo_metadata."""
    good = _lla()
    bad = GeodeticPosition._from_unvalidated(good.unsafe_data.copy(deep=True))
    bad.unsafe_data.attrs["tal"]["ext"]["geo"] = {
        "kind": "cartesian_geo_position",
        "cartesian_system": "enu",
        "geodetic_crs": "EPSG:4979",
        "datum": "WGS84",
        "angular_unit": "degree",
        "height_reference": "ellipsoidal",
        "longitude_wrap": "[-180, 180)",
        "origin": {
            "storage": "inline",
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
        },
    }

    with pytest.raises(ValueError, match="kind='geodetic_position'"):
        bad.distance_to(good, validate=False)


def test_geo_hard_g3_002_distance_rejects_unsupported_method() -> None:
    """ID: GEO_HARD_G3_002_distance_rejects_unsupported_method."""
    with pytest.raises(ValueError, match="opts.method"):
        _lla().distance_to(_lla(), opts=GeodesicOptions(method="bad"))  # type: ignore[arg-type]


def test_geo_distance_rejects_malformed_local_origin_for_unused_method() -> None:
    """Geodesic options fail closed for unsupported local origins."""
    opts = GeodesicOptions(method="geodesic", local_origin=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="local_origin"):
        _lla().distance_to(_lla(), opts=opts)


def test_geo_distance_accepts_valid_geodetic_local_origin_when_unused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supported AO-like origins are valid even when the method does not consume them."""
    _stub_geod(monkeypatch)
    opts = GeodesicOptions(method="geodesic", local_origin=_lla())

    out = _lla().distance_to(_lla(), opts=opts)

    assert isinstance(out, Array)
