from __future__ import annotations

import warnings

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.base import is_dask_collection
from dask.callbacks import Callback

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
    from tal.geo import interpolation

    def interpolate(lat1, lon1, lat2, lon2, alpha, crs, owner):
        return lat1 + (lat2 - lat1) * alpha, lon1 + (lon2 - lon1) * alpha

    monkeypatch.setattr(interpolation, "geod_interpolate", interpolate)


def _stub_conversions(monkeypatch: pytest.MonkeyPatch) -> None:
    from tal.geo import conversion, local, options

    monkeypatch.setattr(options, "normalize_supported_crs", lambda value, expected, owner: expected)
    monkeypatch.setattr(conversion, "transform_lla_to_ecef", lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt))
    monkeypatch.setattr(conversion, "transform_ecef_to_lla", lambda x, y, z, crs, ecef_crs, owner: (x, y, z))
    monkeypatch.setattr(local, "transform_lla_to_ecef", lambda lat, lon, alt, crs, ecef_crs, owner: (lat, lon, alt))


def _position_values(value: GeodeticPosition) -> np.ndarray:
    arr = value.as_dataset(copy="none")["position"].transpose("sample", "lla")
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
    assert out.as_dataset(copy="none").sizes["sample"] == 3


def test_geo_core_g3_008_geodetic_interp_like_uses_geodetic_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_008_geodetic_interp_like_uses_geodetic_semantics."""
    from tal.geo import interpolation

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
    assert out.as_dataset(copy="none").attrs["tal"]["ext"]["geo"]["longitude_wrap"] == "[-180, 180)"


def test_geo_core_g3_010_interpolation_preserves_dask_laziness(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_010_interpolation_preserves_dask_laziness."""
    _stub_geod(monkeypatch)
    source = _lla()
    ds = source.as_dataset(copy="none").copy(deep=True)
    ds["position"] = ds["position"].copy(data=da.from_array(ds["position"].data, chunks=(1, 3)))

    out = GeodeticPosition(ds).param.at([5.0], on="time_s", validate=False)

    assert is_dask_collection(out.as_dataset(copy="none")["position"].data)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("query", ([0.5], []))
def test_geodesic_empty_source_uses_all_invalid_output(
    monkeypatch: pytest.MonkeyPatch, lazy: bool, query: list[float],
) -> None:
    """ID: GEO_CORE_QUERY_OUTPUT_001_zero_sample_geodesic_avoids_gather."""
    _stub_geod(monkeypatch)
    source = _lla(np.empty((0, 3), dtype=float), times=[])
    if lazy:
        ds = source.as_dataset(copy="none").chunk({"sample": 1})
        source = GeodeticPosition(ds)
    before = source.as_dataset(copy="none").copy(deep=True)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        actual = source.param.at(query, on="time_s", opts=GeodeticInterpolationOptions(method="geodesic_linear"))
    assert tasks == []
    dataset = actual.as_dataset(copy="none")
    assert isinstance(actual, GeodeticPosition)
    assert dataset.sizes["sample"] == len(query)
    np.testing.assert_array_equal(dataset["position"].compute(scheduler="synchronous"), np.full((len(query), 3), np.nan))
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


@pytest.mark.parametrize("name", ("time_s", "lla"))
def test_geodesic_query_axis_cannot_replace_output_metadata(name: str) -> None:
    """ID: GEO_HARD_QUERY_OUTPUT_001_geodesic_uses_core_name_preflight."""
    source = _lla()
    query = xr.DataArray([0.25, 0.75], dims=name)
    with pytest.raises(ValueError, match=r"^geo\.GeodeticPosition\.param\.at: query"):
        source.param.at(query, on="time_s")


def test_geo_core_g3_011_ecef_linear_interpolation_roundtrips_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_CORE_G3_011_ecef_linear_interpolation_roundtrips_type."""
    _stub_conversions(monkeypatch)

    out = _lla().param.at([5.0], on="time_s", opts=GeodeticInterpolationOptions(method="ecef_linear"))

    assert isinstance(out, GeodeticPosition)
    np.testing.assert_allclose(_position_values(out), [[0.0, -180.0, 5.0]])


@pytest.mark.parametrize("shape", ((2, 0), (0, 2), (2, 0, 3)))
@pytest.mark.parametrize("lazy", (False, True))
def test_geo_core_empty_query_topology_001_preserves_labeled_cartesian_shape(
    shape: tuple[int, ...],
    lazy: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: GEO_CORE_EMPTY_QUERY_TOPOLOGY_001_preserves_labeled_cartesian_shape."""
    _stub_geod(monkeypatch)
    dims = tuple(f"query_axis_{index}" for index in range(len(shape)))
    coords = {dim: np.arange(size, dtype=np.int64) for dim, size in zip(dims, shape, strict=True)}
    query = xr.DataArray(np.empty(shape), dims=dims, coords=coords)
    if lazy:
        query = query.chunk({dim: max(size, 1) for dim, size in zip(dims, shape, strict=True)})
    source = _lla()
    before = source.as_dataset(copy="deep")
    tasks: list[object] = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        result = source.param.at(query, on="time_s")

    actual = result.as_dataset(copy="none")
    assert tasks == []
    assert tuple(actual.sizes[dim] for dim in dims) == shape
    assert actual["position"].dims == ("lla", *dims)
    for dim in dims:
        assert actual.xindexes[dim].equals(query.xindexes[dim])
    actual.compute(scheduler="synchronous")
    xr.testing.assert_identical(source.as_dataset(copy="none"), before)


def test_geo_core_query_topology_002_preserves_public_name_and_native_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: GEO_CORE_QUERY_TOPOLOGY_002_preserves_public_name_and_native_indexes."""
    _stub_geod(monkeypatch)
    query = xr.DataArray(
        [[0.0, 5.0], [5.0, 10.0]],
        dims=("query", "col"),
        coords=xr.Coordinates.from_xindex(
            xr.indexes.RangeIndex.arange(2, dim="query")
        ),
    ).assign_coords(
        xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(2, dim="col"))
    )

    actual = _lla().param.at(query, on="time_s").as_dataset(copy="none")

    assert actual["position"].dims == ("query", "col", "lla")
    np.testing.assert_array_equal(actual.coords["time_s"], query)
    for dim in query.dims:
        assert type(actual.xindexes[dim]) is type(query.xindexes[dim])
        assert actual.xindexes[dim].equals(query.xindexes[dim])


@pytest.mark.parametrize("shape", ((1,), (0,), (1, 2)))
@pytest.mark.parametrize("validate", (False, True))
def test_geo_query_coordinate_cannot_replace_position_payload(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], validate: bool,
) -> None:
    """ID: GEO_HARD_QUERY_NAMESPACE_001_payload_name_is_preflighted."""
    from tal.geo import interpolation

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("geodesic kernel must not run")

    monkeypatch.setattr(interpolation, "geod_interpolate", unexpected)
    source = _lla()
    dims = ("when",) if len(shape) == 1 else ("row", "col")
    query = xr.DataArray(
        np.full(shape, 5.0), dims=dims,
        coords={"position": (dims, np.full(shape, 99.0))},
    )
    with pytest.raises(ValueError, match="^geo.GeodeticPosition.param.at: query name 'position' collides"):
        source.param.at(query, on="time_s", validate=validate)


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
    generic = AnalysisObject._from_unvalidated(source.as_dataset(copy="none"))

    out = generic.param.at([5.0], on="time_s")

    assert not isinstance(out, GeodeticPosition)
