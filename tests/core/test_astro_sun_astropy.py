from __future__ import annotations

import datetime as dt
from dataclasses import fields
from importlib.util import find_spec
from typing import get_type_hints

import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr

from tal.astro import AstroIERSOptions, AstroTimeOptions
from tal.astro.direction import TopocentricDirection
from tal.astro.metadata import read_astro_block
from tal.astro.sun import SpiceSunOptions, SunDirectionOptions, direction_to_sun
from tal.core import AnalysisObject
from tal.core.schema_read import read_param_coord_name, read_roles
from tal.geo import GeodeticPosition


requires_astropy = pytest.mark.skipif(
    find_spec("astropy") is None,
    reason="Astropy backend tests require the optional tal[astro] extra.",
)


def _iers() -> AstroIERSOptions:
    return AstroIERSOptions(auto_download=False, degraded_accuracy="ignore")


def _scalar_lla() -> AnalysisObject:
    ds = xr.Dataset(
        {"lla": ("lla_axis", np.array([35.0, -106.0, 1600.0], dtype=float))},
        coords={"lla_axis": ["lat", "lon", "alt"]},
    )
    return AnalysisObject.from_data(ds, core_dims=("lla_axis",), validate=True)


def _sequence_lla(*, datetime_param: bool = False, numeric_param: bool = False) -> AnalysisObject:
    times = np.asarray(["2024-06-01T12:00:00", "2024-06-01T12:00:10"], dtype="datetime64[ns]")
    param_values = np.array([0.0, 10.0], dtype=float) if numeric_param else times
    ds = xr.Dataset(
        {"lla": (("sample", "lla_axis"), np.array([[35.0, -106.0, 1600.0], [36.0, -105.0, 1500.0]]))},
        coords={
            "sample": [0, 1],
            "lla_axis": ["lat", "lon", "alt"],
            "time": ("sample", param_values),
        },
    )
    param_coord = "time" if datetime_param or numeric_param else None
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), param_coord=param_coord)


def _batched_lla() -> AnalysisObject:
    values = np.array(
        [
            [[35.0, -106.0, 1600.0], [35.1, -106.1, 1601.0]],
            [[36.0, -105.0, 1500.0], [36.1, -105.1, 1501.0]],
        ],
        dtype=float,
    )
    ds = xr.Dataset(
        {"lla": (("trial", "sample", "lla_axis"), values)},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1],
            "lla_axis": ["lat", "lon", "alt"],
            "utc": ("sample", np.asarray(["2024-06-01T12:00:00", "2024-06-01T12:00:10"], dtype="datetime64[ns]")),
        },
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("lla_axis",))


def _sun(location: object, *, time: object | None = "2024-06-01T12:00:00", source: str | None = None) -> TopocentricDirection:
    opts = SunDirectionOptions(time=AstroTimeOptions(source=source), iers=_iers())
    return direction_to_sun(location, time=time if source is None else None, opts=opts)


@requires_astropy
def test_astro_core_a2_001_direction_to_sun_scalar_location_scalar_time_astropy() -> None:
    """ID: ASTRO_CORE_A2_001_direction_to_sun_scalar_location_scalar_time_astropy."""
    out = _sun(GeodeticPosition.from_lla(_scalar_lla()))
    assert isinstance(out, TopocentricDirection)
    assert out.as_dataset(copy="none")["direction"].dims == ("enu",)


@requires_astropy
def test_astro_core_a2_002_direction_to_sun_sequence_time_coord_astropy() -> None:
    """ID: ASTRO_CORE_A2_002_direction_to_sun_sequence_time_coord_astropy."""
    out = _sun(GeodeticPosition.from_lla(_sequence_lla()), source="time")
    assert out.as_dataset(copy="none")["direction"].dims == ("sample", "enu")
    assert out.as_dataset(copy="none")["altitude_deg"].dims == ("sample",)


@requires_astropy
def test_astro_core_a2_003_direction_to_sun_preserves_batch_topology() -> None:
    """ID: ASTRO_CORE_A2_003_direction_to_sun_preserves_batch_topology."""
    out = _sun(GeodeticPosition.from_lla(_batched_lla()), source="utc")
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ("enu",)
    assert out.as_dataset(copy="none")["direction"].dims == ("sample", "trial", "enu")


@requires_astropy
def test_astro_core_a2_004_direction_to_sun_outputs_unit_enu_vector() -> None:
    """ID: ASTRO_CORE_A2_004_direction_to_sun_outputs_unit_enu_vector."""
    out = _sun(GeodeticPosition.from_lla(_sequence_lla()), source="time")
    norm = np.linalg.norm(out.as_dataset(copy="none")["direction"].to_numpy(), axis=-1)
    np.testing.assert_allclose(norm, 1.0, atol=1e-12)


@requires_astropy
def test_astro_core_a2_005_altitude_azimuth_match_direction_vector() -> None:
    """ID: ASTRO_CORE_A2_005_altitude_azimuth_match_direction_vector."""
    out = _sun(GeodeticPosition.from_lla(_sequence_lla()), source="time")
    alt = np.radians(out.as_dataset(copy="none")["altitude_deg"].to_numpy())
    az = np.radians(out.as_dataset(copy="none")["azimuth_deg"].to_numpy())
    expected = np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], axis=-1)
    np.testing.assert_allclose(out.as_dataset(copy="none")["direction"].to_numpy(), expected, atol=1e-12)


def test_astro_core_a2_006_iers_options_are_applied_and_restored() -> None:
    """ID: ASTRO_CORE_A2_006_iers_options_are_applied_and_restored."""
    pytest.importorskip("astropy")
    from astropy.utils import iers

    old_auto_download = iers.conf.auto_download
    old_degraded_accuracy = iers.conf.iers_degraded_accuracy
    _sun(GeodeticPosition.from_lla(_scalar_lla()))
    assert iers.conf.auto_download == old_auto_download
    assert iers.conf.iers_degraded_accuracy == old_degraded_accuracy


@requires_astropy
def test_astro_core_a2_007_output_metadata_records_astropy_backend() -> None:
    """ID: ASTRO_CORE_A2_007_output_metadata_records_astropy_backend."""
    out = _sun(GeodeticPosition.from_lla(_scalar_lla()))
    block = read_astro_block(out.as_dataset(copy="none"), owner="test")
    assert block["kind"] == "topocentric_direction"
    assert block["target"] == "sun"
    assert block["backend"] == "astropy"
    assert block["direction_var"] == "direction"
    assert set(out.as_dataset(copy="none").data_vars) == {"direction", "altitude_deg", "azimuth_deg"}


def test_astro_core_a2_008_sun_direction_options_shape_includes_spice_field() -> None:
    """ID: ASTRO_CORE_A2_008_sun_direction_options_shape_includes_spice_field."""
    assert [field.name for field in fields(SunDirectionOptions)] == ["backend", "time", "iers", "spice"]
    assert get_type_hints(SunDirectionOptions)["spice"] == SpiceSunOptions | None


@requires_astropy
def test_astro_core_a2_009_datetime64_param_time_source_astropy() -> None:
    """ID: ASTRO_CORE_A2_009_datetime64_param_time_source_astropy."""
    out = _sun(GeodeticPosition.from_lla(_sequence_lla(datetime_param=True)), source="time")
    assert read_param_coord_name(out.as_dataset(copy="none")) == "time"
    assert np.issubdtype(out.as_dataset(copy="none").coords["time"].dtype, np.datetime64)


def test_astro_hard_a2_001_spice_backend_request_before_a3_fails_closed() -> None:
    """ID: ASTRO_HARD_A2_001_spice_backend_request_before_a3_fails_closed."""
    with pytest.raises(ValueError, match="backend='spice'.*reserved"):
        direction_to_sun(_scalar_lla(), time="2024-06-01T12:00:00", opts=SunDirectionOptions(backend="spice"))
    with pytest.raises(ValueError, match="spice options.*reserved"):
        direction_to_sun(_scalar_lla(), time="2024-06-01T12:00:00", opts=SunDirectionOptions(spice=SpiceSunOptions()))


def test_astro_hard_a2_002_missing_astropy_raises_guided_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: ASTRO_HARD_A2_002_missing_astropy_raises_guided_importerror."""

    def fail_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("tal.astro.backends.astropy.import_module", fail_import)
    with pytest.raises(ImportError, match=r"tal\[astro\]"):
        _sun(GeodeticPosition.from_lla(_scalar_lla()))


def test_astro_hard_a2_003_invalid_location_topology_fails_closed() -> None:
    """ID: ASTRO_HARD_A2_003_invalid_location_topology_fails_closed."""
    ds = xr.Dataset(
        {"lla": (("sample", "extra", "lla_axis"), np.ones((1, 2, 3), dtype=float))},
        coords={"sample": [0], "extra": ["x", "y"], "lla_axis": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True)
    with pytest.raises(ValueError, match="semantic|topology|observer"):
        _sun(ao)


def test_astro_hard_a2_004_invalid_time_topology_fails_closed() -> None:
    """ID: ASTRO_HARD_A2_004_invalid_time_topology_fails_closed."""
    time = xr.DataArray(
        np.asarray([["2024-06-01T12:00:00"]], dtype="datetime64[ns]"),
        dims=("other", "sample"),
        coords={"other": ["x"], "sample": [0]},
    )
    with pytest.raises(ValueError, match="time dims.*incompatible|independent expansion"):
        direction_to_sun(GeodeticPosition.from_lla(_sequence_lla()), time=time, opts=SunDirectionOptions(iers=_iers()))


def test_astro_hard_a2_005_dask_inputs_do_not_compute_silently() -> None:
    """ID: ASTRO_HARD_A2_005_dask_inputs_do_not_compute_silently."""

    def fail_compute() -> np.ndarray:
        raise AssertionError("dask time unexpectedly computed")

    raw = da.from_delayed(delayed(fail_compute)(), shape=(1,), dtype="datetime64[ns]")
    time = xr.DataArray(raw, dims=("astro_time",))
    with pytest.raises(ValueError, match="Dask-backed time"):
        direction_to_sun(GeodeticPosition.from_lla(_scalar_lla()), time=time, opts=SunDirectionOptions(iers=_iers()))


def test_astro_hard_a2_006_no_hidden_iers_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: ASTRO_HARD_A2_006_no_hidden_iers_download."""
    pytest.importorskip("astropy")
    from astropy.utils import iers

    monkeypatch.setattr(iers.conf, "auto_download", False)
    out = direction_to_sun(
        GeodeticPosition.from_lla(_scalar_lla()),
        time=dt.datetime(2024, 6, 1, 12, 0, 0),
        opts=SunDirectionOptions(iers=AstroIERSOptions(auto_download=False, degraded_accuracy="ignore")),
    )
    assert isinstance(out, TopocentricDirection)
    assert iers.conf.auto_download is False


def test_astro_hard_a2_007_numeric_param_time_source_fails_closed() -> None:
    """ID: ASTRO_HARD_A2_007_numeric_param_time_source_fails_closed."""
    with pytest.raises(ValueError, match="absolute datetime64"):
        _sun(GeodeticPosition.from_lla(_sequence_lla(numeric_param=True)), source="time")


def test_astro_hard_a2_008_observer_dask_components_fail_before_compute() -> None:
    """ID: ASTRO_HARD_A2_008_observer_dask_components_fail_before_compute."""

    def fail_compute() -> np.ndarray:
        raise AssertionError("dask observer unexpectedly computed")

    raw = da.from_delayed(delayed(fail_compute)(), shape=(1, 3), dtype=float)
    ds = xr.Dataset(
        {"lla": (("sample", "lla_axis"), raw)},
        coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("lla_axis",), validate=True)
    with pytest.raises(ValueError, match="Dask-backed observer"):
        _sun(ao)


def test_astro_hard_a2_object_dtype_time_fails_before_generic_conversion() -> None:
    time = np.asarray([dt.datetime(2024, 6, 1, 12, 0, 0)], dtype=object)
    with pytest.raises(ValueError, match="object dtype"):
        direction_to_sun(GeodeticPosition.from_lla(_scalar_lla()), time=time, opts=SunDirectionOptions(iers=_iers()))
