from __future__ import annotations

import dask.array as da
from dask.base import is_dask_collection
import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.geo import GeodeticInterpolationOptions, GeodeticPosition, LocalOrigin


def _lla(values: np.ndarray | None = None, *, times: list[float] | None = None) -> GeodeticPosition:
    if values is None:
        values = np.asarray([[0.0, 170.0, 0.0], [0.0, 190.0, 10.0]], dtype="float64")
    if times is None:
        times = [float(i * 10) for i in range(values.shape[0])]
    ds = xr.Dataset(
        {"position": (("sample", "lla"), values)},
        coords={"sample": np.arange(values.shape[0]), "lla": ["lat", "lon", "alt"], "time_s": ("sample", times)},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla",), param_coord="time_s", validate=True)
    return GeodeticPosition.from_lla(ao)


def _stub_geod(monkeypatch: pytest.MonkeyPatch) -> None:
    import tal.geo.interpolation as interpolation

    def interpolate(lat1, lon1, lat2, lon2, alpha, crs, owner):
        return lat1 + (lat2 - lat1) * alpha, lon1 + (lon2 - lon1) * alpha

    monkeypatch.setattr(interpolation, "geod_interpolate", interpolate)


def _stub_conversions(monkeypatch: pytest.MonkeyPatch) -> None:
    import tal.geo.conversion as conversion
    import tal.geo.local as local
    import tal.geo.options as options

    monkeypatch.setattr(options, "normalize_supported_crs", lambda value, expected, owner: expected)
    monkeypatch.setattr(conversion, "transform_lla_to_ecef", lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt))
    monkeypatch.setattr(conversion, "transform_ecef_to_lla", lambda x, y, z, crs, ecef_crs, owner: (x, y, z))
    monkeypatch.setattr(local, "transform_lla_to_ecef", lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt))


def _position_values(value: GeodeticPosition) -> np.ndarray:
    arr = value.unsafe_data["position"].transpose("sample", "lla")
    return np.asarray(arr)


def test_geo_core_g3_006_geodetic_param_at_uses_geodetic_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_006_geodetic_param_at_uses_geodetic_default."""
    _stub_geod(monkeypatch)

    out = _lla().param.at([5.0], on="time_s")

    assert isinstance(out, GeodeticPosition)
    np.testing.assert_allclose(_position_values(out), [[0.0, -180.0, 5.0]])


def test_geo_core_g3_007_geodetic_resample_to_preserves_lla_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_007_geodetic_resample_to_preserves_lla_type."""
    _stub_geod(monkeypatch)

    out = _lla().param.resample_to([0.0, 5.0, 10.0], on="time_s")

    assert isinstance(out, GeodeticPosition)
    assert out.unsafe_data.sizes["sample"] == 3


def test_geo_core_g3_008_geodetic_interp_like_uses_geodetic_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_008_geodetic_interp_like_uses_geodetic_semantics."""
    import tal.geo.interpolation as interpolation

    def interpolate(lat1, lon1, lat2, lon2, alpha, crs, owner):
        shape = np.broadcast_shapes(np.shape(alpha), np.shape(lat1), np.shape(lon1))
        return np.full(shape, 12.0, dtype="float64"), np.full(shape, 34.0, dtype="float64")

    monkeypatch.setattr(interpolation, "geod_interpolate", interpolate)
    target = _lla(np.asarray([[0.0, 0.0, 0.0]], dtype="float64"), times=[5.0])

    out = _lla().param.interp_like(target, on="time_s")

    assert isinstance(out, GeodeticPosition)
    np.testing.assert_allclose(_position_values(out), [[12.0, 34.0, 5.0]])


def test_geo_core_g3_009_interpolation_handles_antimeridian_shortest_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_009_interpolation_handles_antimeridian_shortest_path."""
    _stub_geod(monkeypatch)

    out = _lla().param.at([5.0], on="time_s", opts=GeodeticInterpolationOptions(longitude_wrap="[-180, 180)"))

    np.testing.assert_allclose(_position_values(out), [[0.0, -180.0, 5.0]])
    assert out.unsafe_data.attrs["tal"]["ext"]["geo"]["longitude_wrap"] == "[-180, 180)"


def test_geo_core_g3_010_interpolation_preserves_dask_laziness(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_010_interpolation_preserves_dask_laziness."""
    _stub_geod(monkeypatch)
    source = _lla()
    ds = source.unsafe_data.copy(deep=True)
    ds["position"] = ds["position"].copy(data=da.from_array(ds["position"].data, chunks=(1, 3)))

    out = GeodeticPosition(ds).param.at([5.0], on="time_s", validate=False)

    assert is_dask_collection(out.unsafe_data["position"].data)


def test_geo_core_g3_011_ecef_linear_interpolation_roundtrips_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_011_ecef_linear_interpolation_roundtrips_type."""
    _stub_conversions(monkeypatch)

    out = _lla().param.at([5.0], on="time_s", opts=GeodeticInterpolationOptions(method="ecef_linear"))

    assert isinstance(out, GeodeticPosition)
    np.testing.assert_allclose(_position_values(out), [[0.0, -180.0, 5.0]])


def test_geo_interpolation_local_enu_linear_uses_explicit_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local ENU interpolation requires and uses explicit origin semantics."""
    _stub_conversions(monkeypatch)
    opts = GeodeticInterpolationOptions(method="local_enu_linear", local_origin=LocalOrigin(0.0, 0.0, 0.0))

    out = _lla().param.at([5.0], on="time_s", opts=opts)

    assert isinstance(out, GeodeticPosition)
    np.testing.assert_allclose(_position_values(out), [[0.0, -180.0, 5.0]])


def test_geo_interpolation_nearest_follows_core_duplicate_behavior() -> None:
    """Nearest does not add geo-local duplicate rejection."""
    source = _lla(
        np.asarray([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]], dtype="float64"),
        times=[0.0, 0.0, 10.0],
    )

    out = source.param.at([0.0], on="time_s", opts=GeodeticInterpolationOptions(method="nearest"))

    np.testing.assert_allclose(_position_values(out), [[0.0, 1.0, 0.0]])


def test_geo_hard_g3_003_interpolation_rejects_duplicate_param_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G3_003_interpolation_rejects_duplicate_param_labels."""
    _stub_geod(monkeypatch)
    source = _lla(
        np.asarray([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]], dtype="float64"),
        times=[0.0, 0.0, 10.0],
    )

    opts = GeodeticInterpolationOptions(duplicate_policy="raise")
    with pytest.raises(ValueError, match="duplicate"):
        source.param.at([0.0], on="time_s", opts=opts)


def test_geo_hard_g3_004_interpolation_rejects_missing_local_origin() -> None:
    """Local ENU interpolation rejects missing origin."""
    with pytest.raises(ValueError, match="local_origin"):
        _lla().param.at([5.0], on="time_s", opts=GeodeticInterpolationOptions(method="local_enu_linear"))


def test_geo_hard_g3_004_interpolation_rejects_ambiguous_pole_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G3_004_interpolation_rejects_ambiguous_pole_case."""
    _stub_geod(monkeypatch)
    source = _lla(np.asarray([[90.0, 0.0, 0.0], [80.0, 20.0, 0.0]], dtype="float64"))

    with pytest.raises(ValueError, match="ambiguous at a pole"):
        source.param.at([5.0], on="time_s")


def test_geo_interpolation_rejects_malformed_local_origin_for_unused_method() -> None:
    """Geodetic interpolation options fail closed for unsupported local origins."""
    opts = GeodeticInterpolationOptions(method="geodesic_linear", local_origin=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="local_origin"):
        _lla().param.at([5.0], on="time_s", opts=opts)


def test_geo_interpolation_accepts_valid_geodetic_local_origin_when_unused() -> None:
    """Supported AO-like origins are valid even when the method does not consume them."""
    opts = GeodeticInterpolationOptions(method="geodesic_linear", local_origin=_lla())

    out = _lla().param.at([0.0], on="time_s", opts=GeodeticInterpolationOptions(method="nearest", local_origin=opts.local_origin))

    assert isinstance(out, GeodeticPosition)


def test_geo_hard_g3_005_no_hidden_interpolation_in_generic_ops() -> None:
    """ID: GEO_HARD_G3_005_no_hidden_interpolation_in_generic_ops."""
    source = _lla()
    generic = AnalysisObject._from_unvalidated(source.unsafe_data)

    out = generic.param.at([5.0], on="time_s")

    assert not isinstance(out, GeodeticPosition)
