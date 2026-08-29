from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.geo import GeodeticOptions, GeodeticPosition, LocalOrigin
from tal.spatial import Position


def _lla_dataset(values: np.ndarray | None = None) -> xr.Dataset:
    if values is None:
        values = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    ds = xr.Dataset(
        {"position": (("sample", "lla"), values)},
        coords={"sample": np.arange(values.shape[0]), "lla": ["lat", "lon", "alt"]},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True).unsafe_data


def _position_dataset(values: np.ndarray | None = None) -> xr.Dataset:
    if values is None:
        values = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    ds = xr.Dataset(
        {"position": (("sample", "axis"), values)},
        coords={"sample": np.arange(values.shape[0]), "axis": ["x", "y", "z"]},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True).unsafe_data


def _install_geo_backend_stub(monkeypatch) -> None:
    import tal.geo.conversion as conversion
    import tal.geo.local as local
    import tal.geo.options as options

    monkeypatch.setattr(options, "normalize_supported_crs", lambda value, expected, owner: expected)
    monkeypatch.setattr(
        conversion,
        "transform_lla_to_ecef",
        lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
    )
    monkeypatch.setattr(
        conversion,
        "transform_ecef_to_lla",
        lambda x, y, z, crs, ecef_crs, owner: (x, y, z),
    )
    monkeypatch.setattr(
        local,
        "transform_lla_to_ecef",
        lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt),
    )


def test_geo_core_g2_001_position_geo_accessor_to_lla_roundtrips_ecef(monkeypatch) -> None:
    """ID: GEO_CORE_G2_001_position_geo_accessor_to_lla_roundtrips_ecef."""
    _install_geo_backend_stub(monkeypatch)
    lla = GeodeticPosition.from_lla(_lla_dataset())
    ecef = lla.to_ecef()

    roundtrip = ecef.geo.to_lla()

    assert isinstance(roundtrip, GeodeticPosition)
    np.testing.assert_allclose(roundtrip.unsafe_data["position"], lla.unsafe_data["position"])


def test_geo_core_g2_010_position_geo_property_is_installed() -> None:
    """ID: GEO_CORE_G2_010_position_geo_property_is_installed."""
    assert isinstance(Position.geo, property)
    assert isinstance(Position(_position_dataset()).geo._position, Position)


def test_geo_core_g2_011_position_geo_accessor_delegates(monkeypatch) -> None:
    """ID: GEO_CORE_G2_011_position_geo_accessor_delegates."""
    calls: list[tuple[str, object]] = []

    def fake_to_lla(position, *, opts=None, validate=True):
        calls.append(("to_lla", position))
        return "lla"

    def fake_to_enu(position, origin=None, *, opts=None, validate=True):
        calls.append(("to_enu", origin))
        return "enu"

    def fake_to_ecef(position, origin=None, *, opts=None, validate=True):
        calls.append(("to_ecef", origin))
        return "ecef"

    import tal.geo.local as local

    monkeypatch.setattr(local, "position_to_lla", fake_to_lla)
    monkeypatch.setattr(local, "ecef_to_enu", fake_to_enu)
    monkeypatch.setattr(local, "enu_to_ecef", fake_to_ecef)

    position = Position(_position_dataset())
    assert position.geo.to_lla(opts=GeodeticOptions(ecef_frame=None)) == "lla"
    assert position.geo.to_enu(LocalOrigin(0.0, 0.0, 0.0)) == "enu"
    assert position.geo.to_ecef(LocalOrigin(0.0, 0.0, 0.0)) == "ecef"
    assert calls == [
        ("to_lla", position),
        ("to_enu", LocalOrigin(0.0, 0.0, 0.0)),
        ("to_ecef", LocalOrigin(0.0, 0.0, 0.0)),
    ]


def test_geodetic_position_has_no_geo_accessor_in_g2() -> None:
    geo = GeodeticPosition.from_lla(_lla_dataset())
    assert not hasattr(geo, "geo")
