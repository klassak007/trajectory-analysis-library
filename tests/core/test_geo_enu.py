from __future__ import annotations

import dask.array as da
from dask.base import is_dask_collection
import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema import merge_schema
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.geo import ENUOptions, GeodeticOptions, GeodeticPosition, LocalOrigin
from tal.spatial import Position
from tal.utils.frame_schema import get_frames, set_frames


def _lla_dataset(values: np.ndarray | None = None, *, sample_dim: str = "sample") -> xr.Dataset:
    if values is None:
        values = np.asarray([[0.0, 1.0, 0.0]], dtype=float)
    ds = xr.Dataset(
        {"position": ((sample_dim, "lla"), values)},
        coords={
            sample_dim: np.arange(values.shape[0]),
            "lla": ["lat", "lon", "alt"],
            "time_s": (sample_dim, np.arange(values.shape[0], dtype=float)),
            "group_size": np.asarray(values.shape[0], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim=sample_dim,
        core_dims=("lla",),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    ).unsafe_data.copy(deep=True)


def _position_dataset(values: np.ndarray | None = None) -> xr.Dataset:
    if values is None:
        values = np.asarray([[0.0, 1.0, 0.0]], dtype=float)
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


def _install_pyproj_failure_guard(monkeypatch) -> None:
    import tal.geo.options as options

    def fail_if_reached(value, expected, owner):
        raise ImportError("pyproj boundary should not run before origin preflight")

    monkeypatch.setattr(options, "normalize_supported_crs", fail_if_reached)


def _install_g4_crs_option_stub(monkeypatch) -> None:
    import tal.geo.options as options

    def normalize_for_class(value, expected, owner, field="crs"):
        text = str(value)
        actual = "projected" if text.endswith("32611") else "geographic"
        if field == "ecef_crs":
            actual = "geocentric"
        if actual != expected:
            raise ValueError(f"{owner}: {field} must be {expected}; got {actual}.")
        return text

    monkeypatch.setattr(options, "normalize_crs_for_class", normalize_for_class)


def test_geo_core_g2_002_geodetic_to_enu_known_fixture(monkeypatch) -> None:
    """ID: GEO_CORE_G2_002_geodetic_to_enu_known_fixture."""
    _install_geo_backend_stub(monkeypatch)
    lla = GeodeticPosition.from_lla(_lla_dataset())

    enu = lla.to_enu(LocalOrigin(0.0, 0.0, 0.0))

    assert isinstance(enu, Position)
    assert list(enu.unsafe_data["axis"].values) == ["x", "y", "z"]
    np.testing.assert_allclose(enu.unsafe_data["position"], [[1.0, 0.0, 0.0]], atol=1e-12)
    geo = enu.unsafe_data.attrs["tal"]["ext"]["geo"]
    assert geo["kind"] == "cartesian_geo_position"
    assert geo["cartesian_system"] == "enu"
    assert geo["origin"] == {"storage": "inline", "lat": 0.0, "lon": 0.0, "alt": 0.0}


def test_geo_core_g2_003_ecef_to_enu_known_fixture(monkeypatch) -> None:
    """ID: GEO_CORE_G2_003_ecef_to_enu_known_fixture."""
    _install_geo_backend_stub(monkeypatch)
    position = Position(_position_dataset())
    opts = ENUOptions(origin=LocalOrigin(0.0, 0.0, 0.0), ecef_frame=None)

    enu = position.geo.to_enu(opts=opts)

    np.testing.assert_allclose(enu.unsafe_data["position"], [[1.0, 0.0, 0.0]], atol=1e-12)


def test_geo_core_g4_local_origin_opts_accepts_geographic_crs(monkeypatch) -> None:
    """LocalOrigin opts should use G4 CRS-aware geodetic option validation."""
    _install_geo_backend_stub(monkeypatch)
    _install_g4_crs_option_stub(monkeypatch)
    origin = LocalOrigin(0.0, 0.0, 0.0, opts=GeodeticOptions(crs="EPSG:4326"))

    enu = GeodeticPosition.from_lla(_lla_dataset()).to_enu(origin)

    geo = enu.unsafe_data.attrs["tal"]["ext"]["geo"]
    assert geo["geodetic_crs"] == "EPSG:4326"
    assert geo["origin"] == {"storage": "inline", "lat": 0.0, "lon": 0.0, "alt": 0.0}


def test_geo_hard_g4_local_origin_opts_rejects_projected_crs(monkeypatch) -> None:
    """LocalOrigin opts must reject projected CRS values for geodetic origins."""
    _install_geo_backend_stub(monkeypatch)
    _install_g4_crs_option_stub(monkeypatch)
    origin = LocalOrigin(0.0, 0.0, 0.0, opts=GeodeticOptions(crs="EPSG:32611"))

    with pytest.raises(ValueError, match="geographic"):
        GeodeticPosition.from_lla(_lla_dataset()).to_enu(origin)


def test_geo_core_g2_004_enu_to_ecef_roundtrip_within_tolerance(monkeypatch) -> None:
    """ID: GEO_CORE_G2_004_enu_to_ecef_roundtrip_within_tolerance."""
    _install_geo_backend_stub(monkeypatch)
    lla = GeodeticPosition.from_lla(_lla_dataset())
    origin = LocalOrigin(0.0, 0.0, 0.0)

    ecef = lla.to_ecef()
    roundtrip = lla.to_enu(origin).geo.to_ecef()

    np.testing.assert_allclose(roundtrip.unsafe_data["position"], ecef.unsafe_data["position"], atol=1e-12)
    assert roundtrip.unsafe_data.attrs["tal"]["ext"]["geo"]["cartesian_system"] == "ecef"


def test_geo_core_g2_005_enu_conversion_preserves_topology_and_output_frame(monkeypatch) -> None:
    """ID: GEO_CORE_G2_005_enu_conversion_preserves_topology."""
    _install_geo_backend_stub(monkeypatch)
    ds = set_frames(_lla_dataset(), parent="earth_ecef", child="receiver", validate=False)
    lla = GeodeticPosition.from_lla(ds)
    opts = ENUOptions(origin=LocalOrigin(0.0, 0.0, 0.0), output_frame="site_enu")

    enu = lla.to_enu(opts=opts)

    declared, sequence_dim, batch_dims, core_dims = read_roles(enu.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ()
    assert core_dims == ("axis",)
    assert read_param_coord_name(enu.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(enu.unsafe_data) == "group_size"
    assert get_frames(enu.unsafe_data) == ("site_enu", "receiver")
    assert enu.unsafe_data.attrs["tal"]["ext"]["spatial"]["relation"]["expressed_in"] == "site_enu"


def test_geo_core_g2_006_enu_origin_broadcasts_by_semantic_topology(monkeypatch) -> None:
    """ID: GEO_CORE_G2_006_enu_origin_broadcasts_by_semantic_topology."""
    _install_geo_backend_stub(monkeypatch)
    values = np.asarray([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    source = GeodeticPosition.from_lla(
        _lla_dataset(values).assign_coords(tal_geo_origin_lat=("sample", [9.0, 9.0]))
    )
    origin = GeodeticPosition.from_lla(_lla_dataset(np.zeros((2, 3), dtype=float)))

    enu = source.to_enu(origin)

    origin_meta = enu.unsafe_data.attrs["tal"]["ext"]["geo"]["origin"]
    assert origin_meta == {
        "storage": "coordinates",
        "lat_coord": "tal_geo_origin_lat_2",
        "lon_coord": "tal_geo_origin_lon_2",
        "alt_coord": "tal_geo_origin_alt_2",
    }
    assert set(origin_meta.values()).issubset(set(enu.unsafe_data.coords) | {"coordinates"})


def test_geo_core_g2_007_enu_output_is_cartesian_position(monkeypatch) -> None:
    """ID: GEO_CORE_G2_007_enu_output_is_cartesian_position."""
    _install_geo_backend_stub(monkeypatch)
    enu = GeodeticPosition.from_lla(_lla_dataset()).to_enu(LocalOrigin(0.0, 0.0, 0.0))
    assert isinstance(enu, Position)
    assert list(enu.unsafe_data["axis"].values) == ["x", "y", "z"]


def test_geo_core_g2_008_enu_conversion_preserves_dask_laziness(monkeypatch) -> None:
    """ID: GEO_CORE_G2_008_enu_conversion_preserves_dask_laziness."""
    _install_geo_backend_stub(monkeypatch)
    ds = _lla_dataset()
    ds["position"] = ds["position"].copy(data=da.from_array(ds["position"].data, chunks=(1, 3)))

    enu = GeodeticPosition.from_lla(ds).to_enu(LocalOrigin(0.0, 0.0, 0.0), validate=False)

    assert is_dask_collection(enu.unsafe_data["position"].data)


def test_geo_core_g2_009_geo_provenance_metadata_records_enu_origin(monkeypatch) -> None:
    """ID: GEO_CORE_G2_009_geo_provenance_metadata_records_enu_origin."""
    _install_geo_backend_stub(monkeypatch)
    enu = GeodeticPosition.from_lla(_lla_dataset()).to_enu(LocalOrigin(0.0, 0.0, 0.0))
    geo = enu.unsafe_data.attrs["tal"]["ext"]["geo"]
    assert geo["kind"] == "cartesian_geo_position"
    assert geo["cartesian_system"] == "enu"
    assert geo["origin"]["storage"] == "inline"


def test_geo_hard_g2_001_to_enu_rejects_missing_origin(monkeypatch) -> None:
    """ID: GEO_HARD_G2_001_to_enu_rejects_missing_origin."""
    _install_geo_backend_stub(monkeypatch)
    with pytest.raises(ValueError, match="origin is required"):
        GeodeticPosition.from_lla(_lla_dataset()).to_enu()


def test_geo_hard_g2_002_to_enu_rejects_incompatible_origin_topology(monkeypatch) -> None:
    """ID: GEO_HARD_G2_002_to_enu_rejects_incompatible_origin_topology."""
    _install_geo_backend_stub(monkeypatch)
    source = GeodeticPosition.from_lla(_lla_dataset())
    origin = GeodeticPosition.from_lla(_lla_dataset(sample_dim="other"))
    with pytest.raises(ValueError, match="not in source semantic dims"):
        source.to_enu(origin)


def test_geo_hard_g2_002b_to_enu_rejects_origin_arg_type_before_pyproj(monkeypatch) -> None:
    _install_pyproj_failure_guard(monkeypatch)
    source = GeodeticPosition.from_lla(_lla_dataset())
    with pytest.raises(TypeError, match="expected AnalysisObject"):
        source.to_enu(object())


def test_geo_hard_g2_002c_to_enu_rejects_opts_origin_type_before_pyproj(monkeypatch) -> None:
    _install_pyproj_failure_guard(monkeypatch)
    source = GeodeticPosition.from_lla(_lla_dataset())
    with pytest.raises(TypeError, match="expected AnalysisObject"):
        source.to_enu(opts=ENUOptions(origin=object()))


def test_geo_hard_g2_002d_to_enu_rejects_origin_topology_before_pyproj(monkeypatch) -> None:
    _install_pyproj_failure_guard(monkeypatch)
    source = GeodeticPosition.from_lla(_lla_dataset())
    origin = GeodeticPosition.from_lla(_lla_dataset(sample_dim="other"))
    with pytest.raises(ValueError, match="not in source semantic dims"):
        source.to_enu(origin)


def test_geo_hard_g2_003_to_lla_rejects_non_geo_cartesian_without_explicit_opts(monkeypatch) -> None:
    """ID: GEO_HARD_G2_003_to_lla_rejects_non_geo_cartesian_without_explicit_opts."""
    _install_geo_backend_stub(monkeypatch)
    with pytest.raises(ValueError, match="ECEF geo provenance is required"):
        Position(_position_dataset()).geo.to_lla()


def test_geo_hard_g2_003b_to_lla_rejects_enu_even_with_explicit_opts(monkeypatch) -> None:
    _install_geo_backend_stub(monkeypatch)
    enu = GeodeticPosition.from_lla(_lla_dataset()).to_enu(LocalOrigin(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="cartesian_system='ecef'"):
        enu.geo.to_lla(opts=GeodeticOptions(ecef_frame=None))


def test_geo_hard_g2_003c_to_lla_rejects_enu_without_opts(monkeypatch) -> None:
    _install_geo_backend_stub(monkeypatch)
    enu = GeodeticPosition.from_lla(_lla_dataset()).to_enu(LocalOrigin(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="cartesian_system='ecef'"):
        enu.geo.to_lla()


def test_geo_hard_g2_004_strict_frame_conflict_fails_closed(monkeypatch) -> None:
    """ID: GEO_HARD_G2_004_strict_frame_conflict_fails_closed."""
    _install_geo_backend_stub(monkeypatch)
    ds = set_frames(_position_dataset(), parent="mars", child="receiver", validate=False)
    opts = ENUOptions(origin=LocalOrigin(0.0, 0.0, 0.0), ecef_frame="earth_ecef", strict_frame=True)
    with pytest.raises(ValueError, match="not compatible"):
        Position(ds).geo.to_enu(opts=opts)


def test_geo_hard_g2_005_accessor_does_not_enable_hidden_vector_ops(monkeypatch) -> None:
    """ID: GEO_HARD_G2_005_accessor_does_not_enable_hidden_vector_ops."""
    _install_geo_backend_stub(monkeypatch)
    geo = GeodeticPosition.from_lla(_lla_dataset())
    assert not hasattr(geo, "geo")
    assert not hasattr(geo, "norm")


def test_enu_output_without_output_frame_clears_frame_metadata(monkeypatch) -> None:
    _install_geo_backend_stub(monkeypatch)
    ds = set_frames(_lla_dataset(), parent="earth_ecef", child="receiver", validate=False)
    enu = GeodeticPosition.from_lla(ds).to_enu(LocalOrigin(0.0, 0.0, 0.0))
    assert get_frames(enu.unsafe_data) == (None, None)
    relation = enu.unsafe_data.attrs["tal"]["ext"]["spatial"].get("relation", {})
    assert relation.get("expressed_in") is None


def test_enu_to_ecef_rejects_non_enu_input(monkeypatch) -> None:
    _install_geo_backend_stub(monkeypatch)
    ecef = GeodeticPosition.from_lla(_lla_dataset()).to_ecef()
    with pytest.raises(ValueError, match="canonical ENU"):
        ecef.geo.to_ecef(origin=LocalOrigin(0.0, 0.0, 0.0))


def test_enu_to_ecef_rejects_malformed_origin_provenance(monkeypatch) -> None:
    _install_geo_backend_stub(monkeypatch)
    enu = GeodeticPosition.from_lla(_lla_dataset()).to_enu(LocalOrigin(0.0, 0.0, 0.0))
    geo = dict(enu.unsafe_data.attrs["tal"]["ext"]["geo"])
    geo["origin"] = {**geo["origin"], "extra": "bad"}
    corrupted = merge_schema(enu.unsafe_data, {"ext": {"geo": None}}, validate=False)
    corrupted = merge_schema(corrupted, {"ext": {"geo": geo}}, validate=False)
    with pytest.raises(ValueError, match="unknown key"):
        Position(corrupted).geo.to_ecef()
