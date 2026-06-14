from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema import merge_schema
from tal.core.schema_read import read_param_coord_name, read_roles
from tal.geo import GeodeticOptions, GeodeticPosition, ProjectedPosition, transform_crs
from tal.geo.backends import NormalizedCRS
from tal.geo.options import coerce_geodetic_options
from tal.spatial import Position


def _lla_dataset(values: np.ndarray | None = None) -> xr.Dataset:
    if values is None:
        values = np.asarray([[34.0, -118.0, 20.0]], dtype=float)
    ds = xr.Dataset(
        {"position": (("sample", "lla"), values)},
        coords={"sample": np.arange(values.shape[0]), "lla": ["lat", "lon", "alt"]},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), validate=True).unsafe_data


def _lla_dataset_with_topology() -> xr.Dataset:
    ds = xr.Dataset(
        {"position": (("sample", "lla"), np.asarray([[34.0, -118.0, 20.0], [35.0, -117.0, 21.0]]))},
        coords={
            "sample": [0, 1],
            "lla": ["lat", "lon", "alt"],
            "time_s": ("sample", [0.0, 1.0]),
            "group_size": np.asarray(2, dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=("lla",),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    ).unsafe_data


def _projected_dataset(*, height: bool = False) -> xr.Dataset:
    labels = ["easting", "northing", "height"] if height else ["easting", "northing"]
    values = np.asarray([[500000.0, 4100000.0, 20.0]] if height else [[500000.0, 4100000.0]], dtype=float)
    ds = xr.Dataset(
        {"position": (("sample", "projected"), values)},
        coords={"sample": [0], "projected": labels},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("projected",), validate=True).unsafe_data


def _install_crs_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    import tal.geo.crs_transform as crs_transform
    import tal.geo.metadata as metadata
    import tal.geo.options as options

    def kind(value, owner, field="crs"):
        text = str(value)
        if text.endswith("4978"):
            return "geocentric"
        if text.endswith("32611"):
            return "projected"
        return "geographic"

    def normalize_for_class(value, expected, owner, field="crs"):
        actual = kind(value, owner, field)
        if actual != expected:
            raise ValueError(f"{owner}: {field} must be {expected}; got {actual}.")
        return str(value)

    def xyz(x, y, z, src_crs, dst_crs, owner):
        if dst_crs == "EPSG:32611":
            return x + 1000.0, y + 2000.0, z + 3000.0
        if dst_crs in {"EPSG:4979", "EPSG:4326"}:
            return x - 1000.0, y - 2000.0, z
        return x + 1.0, y + 2.0, z + 3.0

    monkeypatch.setattr(options, "normalize_supported_crs", lambda value, expected, owner: expected)
    monkeypatch.setattr(options, "normalize_crs_for_class", normalize_for_class)
    monkeypatch.setattr(metadata, "normalize_crs_for_class", normalize_for_class)
    monkeypatch.setattr(metadata, "base_geodetic_crs", lambda value, owner, field="crs": "EPSG:4326")
    monkeypatch.setattr(
        crs_transform,
        "normalize_crs_with_class",
        lambda value, owner, field="crs": NormalizedCRS(text=str(value), kind=kind(value, owner, field)),
    )
    monkeypatch.setattr(crs_transform, "base_geodetic_crs", lambda value, owner, field="crs": "EPSG:4326")
    monkeypatch.setattr(crs_transform, "crs_has_height_axis", lambda value, owner, field="crs": False)
    monkeypatch.setattr(crs_transform, "transform_crs_xyz", xyz)
    monkeypatch.setattr(crs_transform, "transform_crs_xy", lambda x, y, src_crs, dst_crs, owner: (x + 10.0, y + 20.0))


def test_geo_core_g4_001_crs_metadata_roundtrips(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G4_001_crs_metadata_roundtrips."""
    _install_crs_stub(monkeypatch)
    projected = ProjectedPosition.from_projected(_projected_dataset(), crs="EPSG:32611")
    geo = projected.unsafe_data.attrs["tal"]["ext"]["geo"]
    assert geo == {
        "kind": "projected_position",
        "crs": "EPSG:32611",
        "geodetic_crs": "EPSG:4326",
        "datum": "WGS84",
        "height_reference": "ellipsoidal",
    }


def test_geo_core_g4_projected_metadata_aliases_restamp_canonical_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing projected metadata should normalize accepted aliases to canonical text."""
    _install_crs_stub(monkeypatch)
    import tal.geo.metadata as metadata

    def normalize_for_class(value, expected, owner, field="crs"):
        aliases = {"UTM11_ALIAS": ("projected", "EPSG:32611"), "WGS84_ALIAS": ("geographic", "EPSG:4326")}
        actual, canonical = aliases.get(str(value), (expected, str(value)))
        if actual != expected:
            raise ValueError(f"{owner}: {field} must be {expected}; got {actual}.")
        return canonical

    monkeypatch.setattr(metadata, "normalize_crs_for_class", normalize_for_class)
    ds = merge_schema(
        _projected_dataset(),
        {
            "ext": {
                "geo": {
                    "kind": "projected_position",
                    "crs": "UTM11_ALIAS",
                    "geodetic_crs": "WGS84_ALIAS",
                    "datum": "WGS84",
                    "height_reference": "ellipsoidal",
                }
            }
        },
        validate=False,
    )

    projected = ProjectedPosition(ds)

    geo = projected.unsafe_data.attrs["tal"]["ext"]["geo"]
    assert geo["crs"] == "EPSG:32611"
    assert geo["geodetic_crs"] == "EPSG:4326"


def test_geo_core_g4_002_epsg_4326_to_4979_transform_preserves_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G4_002_epsg_4326_to_4979_transform_preserves_type."""
    _install_crs_stub(monkeypatch)
    lla = GeodeticPosition.from_lla(_lla_dataset(), opts=GeodeticOptions(crs="EPSG:4326"))
    out = lla.to_crs("EPSG:4979")
    assert isinstance(out, GeodeticPosition)
    assert out.unsafe_data.attrs["tal"]["ext"]["geo"]["crs"] == "EPSG:4979"
    _, _, _, core_dims = read_roles(out.unsafe_data)
    assert list(out.unsafe_data[core_dims[0]].values) == ["lat", "lon", "alt"]


def test_geo_core_g4_003_ecef_crs_transform_returns_cartesian_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G4_003_ecef_crs_transform_returns_cartesian_position."""
    _install_crs_stub(monkeypatch)
    out = GeodeticPosition.from_lla(_lla_dataset()).to_crs("EPSG:4978")
    assert isinstance(out, Position)
    assert list(out.unsafe_data["axis"].values) == ["x", "y", "z"]
    assert out.unsafe_data.attrs["tal"]["ext"]["geo"]["cartesian_system"] == "ecef"


def test_geo_core_g4_004_crs_transform_preserves_topology_and_validity(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G4_004_crs_transform_preserves_topology_and_validity."""
    _install_crs_stub(monkeypatch)
    out = GeodeticPosition.from_lla(_lla_dataset_with_topology()).to_crs("EPSG:32611")
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.unsafe_data)
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ()
    assert core_dims == ("projected",)
    assert read_param_coord_name(out.unsafe_data) == "time_s"


def test_geo_core_g4_005_projected_destination_returns_projected_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G4_005_projected_destination_returns_projected_position."""
    _install_crs_stub(monkeypatch)
    out = transform_crs(GeodeticPosition.from_lla(_lla_dataset()), dst="EPSG:32611")
    assert isinstance(out, ProjectedPosition)
    assert list(out.unsafe_data["projected"].values) == ["easting", "northing", "height"]


def test_geo_core_g4_006_crs_axis_order_policy_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G4_006_crs_axis_order_policy_is_explicit."""
    _install_crs_stub(monkeypatch)
    import tal.geo.crs_transform as crs_transform

    seen = {}

    def capture(x, y, z, src_crs, dst_crs, owner):
        seen["x"] = np.asarray(x).copy()
        seen["y"] = np.asarray(y).copy()
        return x, y, z

    monkeypatch.setattr(crs_transform, "transform_crs_xyz", capture)
    _ = GeodeticPosition.from_lla(_lla_dataset()).to_crs("EPSG:4978")
    np.testing.assert_allclose(seen["x"], [-118.0])
    np.testing.assert_allclose(seen["y"], [34.0])


def test_geo_core_g4_007_projected_position_from_projected_validates_crs_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: GEO_CORE_G4_007_projected_position_from_projected_validates_crs_class."""
    _install_crs_stub(monkeypatch)
    with pytest.raises(ValueError, match="projected"):
        ProjectedPosition.from_projected(_projected_dataset(), crs="EPSG:4326")


def test_geo_core_g4_008_projected_height_semantics_are_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G4_008_projected_height_semantics_are_stable."""
    _install_crs_stub(monkeypatch)
    lla = GeodeticPosition.from_lla(_lla_dataset(values=np.asarray([[34.0, -118.0, 20.0]], dtype=float)))
    projected = lla.to_crs("EPSG:32611")
    np.testing.assert_allclose(projected.unsafe_data["position"].sel(projected="height"), [20.0])
    projected2d = ProjectedPosition.from_projected(_projected_dataset(), crs="EPSG:32611")
    with pytest.raises(ValueError, match="2D projected"):
        projected2d.to_crs("EPSG:4979")


def test_geo_hard_g4_002_crs_metadata_conflict_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G4_002_crs_metadata_conflict_fails_closed."""
    _install_crs_stub(monkeypatch)
    ds = ProjectedPosition.from_projected(_projected_dataset(), crs="EPSG:32611").unsafe_data.copy(deep=True)
    ds.attrs["tal"]["ext"]["geo"]["geodetic_crs"] = "EPSG:4979"
    with pytest.raises(ValueError, match="conflicts"):
        ProjectedPosition(ds)


def test_geo_hard_g4_001_crs_transform_rejects_ambiguous_source_crs(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G4_001_crs_transform_rejects_ambiguous_source_crs."""
    _install_crs_stub(monkeypatch)
    ds = xr.Dataset(
        {"position": (("sample", "axis"), np.asarray([[1.0, 2.0, 3.0]], dtype=float))},
        coords={"sample": [0], "axis": ["x", "y", "z"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("axis",), validate=True)
    with pytest.raises(ValueError, match="geo metadata"):
        transform_crs(Position(ao), dst="EPSG:4979")


def test_geo_hard_g4_transform_crs_invalid_value_skips_backend_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported value should fail before CRS backend normalization."""
    import tal.geo.crs_transform as crs_transform

    calls = []

    def normalize(value, owner, field="crs"):
        calls.append((value, owner, field))
        return NormalizedCRS(text=str(value), kind="geographic")

    monkeypatch.setattr(crs_transform, "normalize_crs_with_class", normalize)

    with pytest.raises(TypeError, match="value must be GeodeticPosition"):
        transform_crs(object(), dst="EPSG:4979")

    assert calls == []


def test_geo_hard_g4_003_projected_position_rejects_vector_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G4_003_projected_position_rejects_vector_semantics."""
    _install_crs_stub(monkeypatch)
    projected = ProjectedPosition.from_projected(_projected_dataset(height=True), crs="EPSG:32611")
    assert not isinstance(projected, Position)
    assert not hasattr(projected, "as_delta")


def test_geo_hard_g4_005_no_hidden_crs_conversion_in_generic_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G4_005_no_hidden_crs_conversion_in_generic_ops."""
    _install_crs_stub(monkeypatch)
    lla = GeodeticPosition.from_lla(_lla_dataset())
    projected = lla.to_crs("EPSG:32611")
    assert isinstance(projected, ProjectedPosition)
    assert not isinstance(projected, Position)


def test_geo_hard_g4_007_per_field_crs_class_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G4_007_per_field_crs_class_mismatch_fails_closed."""
    _install_crs_stub(monkeypatch)
    with pytest.raises(ValueError, match="geographic"):
        GeodeticPosition.from_lla(_lla_dataset(), opts=GeodeticOptions(crs="EPSG:32611"))
    with pytest.raises(ValueError, match="geocentric"):
        coerce_geodetic_options(GeodeticOptions(ecef_crs="EPSG:4326"), owner="test", validate_crs=True)
