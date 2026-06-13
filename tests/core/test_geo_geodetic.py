from __future__ import annotations

import importlib
from importlib.machinery import ModuleSpec

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema import merge_schema
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.geo import GeodeticOptions, GeodeticPosition, LocalOrigin, from_ecef
from tal.spatial import Position
from tal.utils.frame_schema import get_frames, set_frames


def _lla_dataset(values: np.ndarray | None = None, *, labels: tuple[str, str, str] = ("lat", "lon", "alt")) -> xr.Dataset:
    if values is None:
        values = np.asarray([[0.0, 0.0, 0.0], [45.0, -75.0, 100.0]], dtype="float64")
    ds = xr.Dataset(
        {"position": (("sample", "lla_axis"), values)},
        coords={
            "sample": [0, 1],
            "lla_axis": list(labels),
            "time_s": ("sample", [0.0, 1.0]),
            "group_size": np.asarray(2, dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=("lla_axis",),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    ).unsafe_data.copy(deep=True)


def _ecef_dataset(values: np.ndarray | None = None) -> xr.Dataset:
    if values is None:
        values = np.asarray([[6378137.0, 0.0, 0.0]], dtype="float64")
    ds = xr.Dataset(
        {"position": (("sample", "axis"), values)},
        coords={"sample": np.arange(values.shape[0]), "axis": ["x", "y", "z"]},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True).unsafe_data.copy(deep=True)


def _as_dataarray_with_schema(ds: xr.Dataset) -> xr.DataArray:
    arr = ds["position"].copy(deep=True)
    arr.attrs["tal"] = ds.attrs["tal"]
    return arr


def _require_pyproj() -> None:
    pytest.importorskip("pyproj")


def _install_geo_backend_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    import tal.geo.conversion as conversion
    import tal.geo.local as local
    import tal.geo.options as options

    monkeypatch.setattr(options, "normalize_supported_crs", lambda value, expected, owner: expected)
    monkeypatch.setattr(
        conversion,
        "transform_lla_to_ecef",
        lambda lat, lon, alt, crs, ecef_crs, owner: (lat + 1.0, lon + 2.0, alt + 3.0),
    )
    monkeypatch.setattr(
        conversion,
        "transform_ecef_to_lla",
        lambda x, y, z, crs, ecef_crs, owner: (x - 1.0, y - 2.0, z - 3.0),
    )
    monkeypatch.setattr(
        local,
        "transform_lla_to_ecef",
        lambda lat, lon, alt, crs, ecef_crs, owner: (lat + 1.0, lon + 2.0, alt + 3.0),
    )


def test_geo_core_g1_001_geodetic_position_constructor_accepts_ao_dataset_dataarray() -> None:
    """ID: GEO_CORE_G1_001_geodetic_position_constructor_accepts_ao_dataset_dataarray."""
    ds = _lla_dataset()
    ao = AnalysisObject._from_validated(ds)
    arr = _as_dataarray_with_schema(ds)

    assert isinstance(GeodeticPosition(ao), GeodeticPosition)
    assert isinstance(GeodeticPosition(ds), GeodeticPosition)
    assert isinstance(GeodeticPosition(arr), GeodeticPosition)


def test_geo_core_g1_002_geodetic_position_requires_lla_core_labels() -> None:
    """ID: GEO_CORE_G1_002_geodetic_position_requires_lla_core_labels."""
    with pytest.raises(ValueError, match="labels must equal"):
        GeodeticPosition(_lla_dataset(labels=("lat", "lon", "height")))


def test_geo_core_g1_003_geodetic_metadata_normalizes_defaults() -> None:
    """ID: GEO_CORE_G1_003_geodetic_metadata_normalizes_defaults."""
    geo = GeodeticPosition(_lla_dataset()).unsafe_data.attrs["tal"]["ext"]["geo"]
    assert geo == {
        "kind": "geodetic_position",
        "crs": "EPSG:4979",
        "datum": "WGS84",
        "angular_unit": "degree",
        "height_reference": "ellipsoidal",
        "longitude_wrap": "[-180, 180)",
    }


def test_geo_core_g1_004_lla_to_ecef_known_wgs84_fixture() -> None:
    """ID: GEO_CORE_G1_004_lla_to_ecef_known_wgs84_fixture."""
    _require_pyproj()
    lla = GeodeticPosition.from_lla(_lla_dataset(values=np.asarray([[0.0, 0.0, 0.0]])))
    ecef = lla.to_ecef()
    np.testing.assert_allclose(ecef.unsafe_data["position"], [[6378137.0, 0.0, 0.0]], atol=1e-6)
    assert list(ecef.unsafe_data["axis"].values) == ["x", "y", "z"]


def test_geo_core_g1_005_ecef_to_lla_known_wgs84_fixture() -> None:
    """ID: GEO_CORE_G1_005_ecef_to_lla_known_wgs84_fixture."""
    _require_pyproj()
    lla = GeodeticPosition.from_ecef(_ecef_dataset())
    np.testing.assert_allclose(lla.unsafe_data["position"], [[0.0, 0.0, 0.0]], atol=1e-8)
    assert list(lla.unsafe_data["lla"].values) == ["lat", "lon", "alt"]


def test_geo_core_g1_006_lla_ecef_roundtrip_within_tolerance() -> None:
    """ID: GEO_CORE_G1_006_lla_ecef_roundtrip_within_tolerance."""
    _require_pyproj()
    lla = GeodeticPosition.from_lla(_lla_dataset())
    roundtrip = GeodeticPosition.from_ecef(lla.to_ecef())
    np.testing.assert_allclose(roundtrip.unsafe_data["position"], lla.unsafe_data["position"], atol=1e-6)


def test_geo_core_g1_007_conversion_preserves_roles_param_and_validity() -> None:
    """ID: GEO_CORE_G1_007_conversion_preserves_roles_param_and_validity."""
    _require_pyproj()
    ecef = GeodeticPosition(_lla_dataset()).to_ecef()
    declared, sequence_dim, batch_dims, core_dims = read_roles(ecef.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ()
    assert core_dims == ("axis",)
    assert read_param_coord_name(ecef.unsafe_data) == "time_s"
    assert read_sequence_size_coord_name(ecef.unsafe_data) == "group_size"


def test_geo_core_g1_008_conversion_preserves_frame_metadata() -> None:
    """ID: GEO_CORE_G1_008_conversion_preserves_frame_metadata."""
    _require_pyproj()
    ds = set_frames(_lla_dataset(), parent="earth_ecef", child="sensor", validate=False)
    ecef = GeodeticPosition(ds).to_ecef()
    assert get_frames(ecef.unsafe_data) == ("earth_ecef", "sensor")
    assert ecef.unsafe_data.attrs["tal"]["ext"]["spatial"]["relation"]["expressed_in"] == "earth_ecef"


def test_geo_core_g1_009_conversion_preserves_dask_laziness() -> None:
    """ID: GEO_CORE_G1_009_conversion_preserves_dask_laziness."""
    _require_pyproj()
    ds = _lla_dataset()
    ds["position"] = ds["position"].copy(data=da.from_array(ds["position"].data, chunks=(1, 3)))
    ecef = GeodeticPosition(ds).to_ecef(validate=False)
    assert da.is_dask_collection(ecef.unsafe_data["position"].data)


def test_geo_core_g1_010_from_ecef_accepts_cartesian_position() -> None:
    """ID: GEO_CORE_G1_010_from_ecef_accepts_cartesian_position."""
    _require_pyproj()
    lla = from_ecef(Position(_ecef_dataset()))
    assert isinstance(lla, GeodeticPosition)


def test_geo_conversion_orchestration_roundtrips_with_backend_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise conversion finalization without requiring the optional backend."""
    import tal.geo.conversion as conversion

    monkeypatch.setattr(
        conversion,
        "coerce_geodetic_options",
        lambda opts, owner, validate_crs: opts or GeodeticOptions(),
    )
    monkeypatch.setattr(
        conversion,
        "transform_lla_to_ecef",
        lambda lat, lon, alt, crs, ecef_crs, owner: (lat + 1.0, lon + 2.0, alt + 3.0),
    )
    monkeypatch.setattr(
        conversion,
        "transform_ecef_to_lla",
        lambda x, y, z, crs, ecef_crs, owner: (x - 1.0, y - 2.0, z - 3.0),
    )
    values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    lla = GeodeticPosition(_lla_dataset(values=values))
    ecef = lla.to_ecef()
    roundtrip = GeodeticPosition.from_ecef(ecef)
    np.testing.assert_allclose(roundtrip.unsafe_data["position"], lla.unsafe_data["position"])


def test_geo_from_ecef_opts_none_preserves_inferred_custom_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """ECEF provenance should preserve a non-default frame parent on roundtrip."""
    _install_geo_backend_stub(monkeypatch)
    values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    ds = set_frames(_lla_dataset(values=values), parent="sensor_ecef", child="receiver", validate=False)

    lla = GeodeticPosition(ds)
    ecef = lla.to_ecef(opts=GeodeticOptions(ecef_frame="custom_ecef"))
    roundtrip = GeodeticPosition.from_ecef(ecef)

    assert get_frames(ecef.unsafe_data) == ("custom_ecef", "receiver")
    assert get_frames(roundtrip.unsafe_data) == ("custom_ecef", "receiver")
    assert roundtrip.unsafe_data.attrs["tal"]["ext"]["spatial"]["relation"]["expressed_in"] == "custom_ecef"
    np.testing.assert_allclose(roundtrip.unsafe_data["position"], lla.unsafe_data["position"])


def test_geo_from_ecef_rejects_superseded_ecef_position_metadata_with_explicit_opts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G2 rejects G1-only ECEF provenance even when explicit opts are supplied."""
    _install_geo_backend_stub(monkeypatch)
    ds = merge_schema(
        _ecef_dataset(),
        {
            "ext": {
                "geo": {
                    "kind": "ecef_position",
                    "crs": "EPSG:4978",
                    "geodetic_crs": "EPSG:4979",
                    "datum": "WGS84",
                    "angular_unit": "degree",
                    "height_reference": "ellipsoidal",
                    "longitude_wrap": "[-180, 180)",
                }
            }
        },
        validate=False,
    )
    with pytest.raises(ValueError, match="superseded"):
        GeodeticPosition.from_ecef(Position(ds), opts=GeodeticOptions(ecef_frame=None))


def test_geo_from_ecef_rejects_enu_metadata_with_explicit_opts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct from_ecef must not treat ENU provenance as ECEF."""
    _install_geo_backend_stub(monkeypatch)
    enu = GeodeticPosition.from_lla(_lla_dataset()).to_enu(LocalOrigin(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="cartesian_system='ecef'"):
        GeodeticPosition.from_ecef(enu, opts=GeodeticOptions(ecef_frame=None))


def test_geo_from_ecef_raw_position_allows_explicit_opts_without_geo_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit opts may disambiguate raw ECEF data when no geo block is present."""
    _install_geo_backend_stub(monkeypatch)
    lla = GeodeticPosition.from_ecef(
        Position(_ecef_dataset()),
        opts=GeodeticOptions(ecef_frame=None),
    )
    np.testing.assert_allclose(lla.unsafe_data["position"], [[6378136.0, -2.0, -3.0]])


def test_geo_from_ecef_raw_position_default_options_without_geo_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct from_ecef keeps the existing no-provenance default behavior."""
    _install_geo_backend_stub(monkeypatch)
    lla = GeodeticPosition.from_ecef(Position(_ecef_dataset()))
    np.testing.assert_allclose(lla.unsafe_data["position"], [[6378136.0, -2.0, -3.0]])


def test_geo_core_g1_011_geo_optional_dependency_group_declared() -> None:
    """ID: GEO_CORE_G1_011_geo_optional_dependency_group_declared."""
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]
    assert optional["geo"] == ["pyproj>=3.7"]
    assert "tal[docs,frames,geo,viz]" in optional["test"]


def test_geo_hard_g1_001_constructor_rejects_cartesian_position_payload() -> None:
    """ID: GEO_HARD_G1_001_constructor_rejects_cartesian_position_payload."""
    with pytest.raises(ValueError, match="labels must equal"):
        GeodeticPosition(_ecef_dataset())


def test_geo_hard_g1_002_constructor_rejects_missing_altitude() -> None:
    """ID: GEO_HARD_G1_002_constructor_rejects_missing_altitude."""
    ds = xr.Dataset(
        {"position": (("sample", "lla_axis"), [[0.0, 0.0]])},
        coords={"sample": [0], "lla_axis": ["lat", "lon"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True)
    with pytest.raises(ValueError, match="length 3"):
        GeodeticPosition(ao)


def test_geo_hard_g1_003_constructor_rejects_malformed_geo_metadata() -> None:
    """ID: GEO_HARD_G1_003_constructor_rejects_malformed_geo_metadata."""
    ds = merge_schema(_lla_dataset(), {"ext": {"geo": {"kind": "geodetic_position", "bogus": True}}}, validate=False)
    with pytest.raises(ValueError, match="unknown key"):
        GeodeticPosition.from_lla(ds, opts=GeodeticOptions())


def test_geo_hard_g1_003b_constructor_rejects_unsupported_metadata_crs() -> None:
    ds = merge_schema(_lla_dataset(), {"ext": {"geo": {"kind": "geodetic_position", "crs": "EPSG:4326"}}}, validate=False)
    with pytest.raises(ValueError, match="unsupported CRS"):
        GeodeticPosition(ds)


def test_geo_hard_g1_004_conversion_rejects_unsupported_datum() -> None:
    """ID: GEO_HARD_G1_004_conversion_rejects_unsupported_datum."""
    bad = GeodeticOptions(datum="NAD83")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported datum"):
        GeodeticPosition.from_lla(_lla_dataset(), opts=bad)


def test_geo_hard_g1_005_conversion_rejects_incompatible_strict_frame() -> None:
    """ID: GEO_HARD_G1_005_conversion_rejects_incompatible_strict_frame."""
    _require_pyproj()
    ds = set_frames(_lla_dataset(), parent="mars", child="sensor", validate=False)
    opts = GeodeticOptions(ecef_frame="earth_ecef", strict_frame=True)
    with pytest.raises(ValueError, match="not compatible"):
        GeodeticPosition(ds).to_ecef(opts=opts)


def test_geo_hard_g1_006_no_generic_vector_ops_on_geodetic_position() -> None:
    """ID: GEO_HARD_G1_006_no_generic_vector_ops_on_geodetic_position."""
    geo = GeodeticPosition(_lla_dataset())
    assert not hasattr(geo, "norm")
    assert not hasattr(geo, "to_frame")
    assert hasattr(geo, "to_ecef")


def test_geo_hard_g1_007_missing_pyproj_conversion_raises_guided_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G1_007_missing_pyproj_conversion_raises_guided_importerror."""
    import tal.geo.backends as backends

    def missing_import(name: str) -> ModuleSpec:
        if name == "pyproj":
            raise ImportError("forced missing pyproj")
        return importlib.import_module(name)

    monkeypatch.setattr(backends, "import_module", missing_import)
    with pytest.raises(ImportError, match="tal\\[geo\\]"):
        GeodeticPosition(_lla_dataset()).to_ecef()
