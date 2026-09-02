from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from tal.astro import AstroOptions, AstroTimeOptions, TopocentricDirection
from tal.astro.backends.astropy import require_astropy
from tal.astro.finalize import finalize_topocentric_direction
from tal.astro.metadata import normalize_topocentric_metadata, read_astro_block
from tal.astro.options import coerce_astro_options
from tal.astro.orchestration import resolve_direction_runtime_context
from tal.core import AnalysisObject
from tal.core.schema import merge_schema
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.geo import GeodeticPosition


def _direction_ao(labels: tuple[str, str, str] = ("east", "north", "up")) -> AnalysisObject:
    ds = xr.Dataset(
        {"direction": (("sample", "enu"), np.array([[1.0, 0.0, 0.0]]))},
        coords={"sample": [0], "enu": list(labels)},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("enu",), validate=True)


def _direction_ao_with_validity() -> AnalysisObject:
    values = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]],
            [[0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
        ],
        dtype=float,
    )
    ds = xr.Dataset(
        {"direction": (("sample", "trial", "enu"), values)},
        coords={
            "sample": [0, 1, 2],
            "trial": ["a", "b"],
            "enu": ["east", "north", "up"],
            "time_s": ("sample", np.array([0.0, 1.0, 2.0], dtype=float)),
            "group_size": ("trial", np.array([2, 3], dtype=np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("enu",),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )


def _lla_ao() -> AnalysisObject:
    ds = xr.Dataset(
        {"lla": (("sample", "lla_axis"), np.array([[45.0, -75.0, 100.0]]))},
        coords={
            "sample": [0],
            "lla_axis": ["lat", "lon", "alt"],
            "time_s": ("sample", np.array([0.0], dtype=float)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=("lla_axis",),
        param_coord="time_s",
        validate=True,
    )


def _lla_ao_with_validity() -> AnalysisObject:
    values = np.array(
        [
            [[45.0, -75.0, 100.0], [46.0, -76.0, 110.0], [47.0, -77.0, 120.0]],
            [[35.0, -65.0, 200.0], [36.0, -66.0, 210.0], [37.0, -67.0, 220.0]],
        ],
        dtype=float,
    )
    ds = xr.Dataset(
        {"lla": (("trial", "sample", "lla_axis"), values)},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "lla_axis": ["lat", "lon", "alt"],
            "time_s": ("sample", np.array([0.0, 1.0, 2.0], dtype=float)),
            "group_size": ("trial", np.array([2, 3], dtype=np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("lla_axis",),
        param_coord="time_s",
        sequence_size_coord="group_size",
        validate=True,
    )


def _direction_dataset(dims: tuple[str, ...] = ("trial", "sample", "enu")) -> xr.Dataset:
    sizes = {"trial": 2, "sample": 3, "astro_time": 3, "enu": 3}
    shape = tuple(sizes[dim] for dim in dims)
    values = np.zeros(shape, dtype=float)
    values[..., 2] = 1.0
    coords = {
        "trial": ["a", "b"],
        "sample": [0, 1, 2],
        "astro_time": [10, 20, 30],
        "enu": ["east", "north", "up"],
    }
    return xr.Dataset(
        {"direction": (dims, values)},
        coords={dim: coords[dim] for dim in dims},
    )


def test_astro_core_a1_001_topocentric_direction_constructor_accepts_ao_dataset_dataarray() -> None:
    """ID: ASTRO_CORE_A1_001_topocentric_direction_constructor_accepts_ao_dataset_dataarray."""
    ao = _direction_ao()
    from_ao = TopocentricDirection(ao)
    from_ds = TopocentricDirection(ao.as_dataset(copy="none"))
    da_in = ao.as_dataset(copy="none")["direction"].copy()
    da_in.attrs["tal"] = deepcopy(ao.as_dataset(copy="none").attrs["tal"])
    from_da = TopocentricDirection(da_in)
    assert isinstance(from_ao, TopocentricDirection)
    assert isinstance(from_ds, TopocentricDirection)
    assert isinstance(from_da, TopocentricDirection)


def test_astro_core_a1_011_topocentric_direction_to_vector3_maps_enu_to_xyz() -> None:
    """ID: ASTRO_CORE_A1_011_topocentric_direction_to_vector3_maps_enu_to_xyz."""
    direction = TopocentricDirection(_direction_ao())
    vector = direction.to_vector3(axis="axis", output_var="sun")
    assert tuple(vector.as_dataset(copy="none").coords["axis"].to_numpy().tolist()) == ("x", "y", "z")
    np.testing.assert_allclose(vector.as_dataset(copy="none")["sun"].to_numpy(), direction.as_dataset(copy="none")["direction"].to_numpy())


def test_astro_core_a1_012_to_vector3_preserves_topology_param_and_validity() -> None:
    """ID: ASTRO_CORE_A1_012_to_vector3_preserves_topology_param_and_validity."""
    direction = TopocentricDirection(_direction_ao_with_validity())
    vector = direction.to_vector3()
    declared, sequence_dim, batch_dims, core_dims = read_roles(vector.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ("axis",)
    assert vector.as_dataset(copy="none")["direction"].dims == ("sample", "trial", "axis")
    assert read_param_coord_name(vector.as_dataset(copy="none")) == "time_s"
    assert read_sequence_size_coord_name(vector.as_dataset(copy="none")) == "group_size"
    np.testing.assert_array_equal(vector.as_dataset(copy="none").coords["group_size"].to_numpy(), np.array([2, 3]))


def test_astro_core_a1_013_topocentric_direction_preserves_non_unit_magnitude() -> None:
    """ID: ASTRO_CORE_A1_013_topocentric_direction_preserves_non_unit_magnitude."""
    values = np.array([[2.0, 0.0, 0.0]], dtype=float)
    direction = TopocentricDirection(_direction_ao())
    replaced = direction.as_dataset(copy="none").copy()
    replaced["direction"] = replaced["direction"].copy(data=values)
    out = TopocentricDirection(replaced)
    np.testing.assert_allclose(out.as_dataset(copy="none")["direction"].to_numpy(), values)


def test_astro_core_a1_014_to_vector3_clears_astro_metadata_preserves_siblings() -> None:
    """ID: ASTRO_CORE_A1_014_to_vector3_clears_astro_metadata_preserves_siblings."""
    direction = TopocentricDirection(_direction_ao())
    with_custom = merge_schema(
        direction.as_dataset(copy="none"),
        {"ext": {"custom": {"owner": "test", "version": 1}}},
        validate=True,
    )
    vector = TopocentricDirection(with_custom).to_vector3()
    ext = vector.as_dataset(copy="none").attrs["tal"]["ext"]
    assert ext["custom"] == {"owner": "test", "version": 1}
    assert "astro" not in ext


def test_astro_core_a1_002_topocentric_direction_requires_enu_core_labels() -> None:
    """ID: ASTRO_CORE_A1_002_topocentric_direction_requires_enu_core_labels."""
    with pytest.raises(ValueError, match="east.*north.*up"):
        TopocentricDirection(_direction_ao(labels=("x", "y", "z")))


def test_astro_core_a1_003_astro_metadata_normalizes_defaults() -> None:
    """ID: ASTRO_CORE_A1_003_astro_metadata_normalizes_defaults."""
    direction = TopocentricDirection(_direction_ao())
    block = read_astro_block(direction.as_dataset(copy="none"), owner="test")
    assert block == {
        "kind": "topocentric_direction",
        "target": "sun",
        "backend": "astropy",
        "observer_frame": "enu",
        "time_scale": "utc",
        "direction_var": "direction",
        "altitude_var": "altitude_deg",
        "azimuth_var": "azimuth_deg",
    }


def test_astro_core_a1_004_time_options_normalize_deterministically() -> None:
    """ID: ASTRO_CORE_A1_004_time_options_normalize_deterministically."""
    opts = coerce_astro_options(AstroOptions(time=AstroTimeOptions(scale="tt", source=" utc ")), owner="test")
    assert opts.time is not None
    assert opts.time.scale == "tt"
    assert opts.time.source == "utc"


def test_astro_core_a1_005_location_coercion_accepts_geodetic_position() -> None:
    """ID: ASTRO_CORE_A1_005_location_coercion_accepts_geodetic_position."""
    lla = GeodeticPosition.from_lla(_lla_ao())
    ctx = resolve_direction_runtime_context(location=lla, opts=AstroTimeOptions(source="time_s"))
    assert ctx.observer.location is lla
    assert ctx.output_sequence_dim == "sample"


def test_astro_core_a1_006_time_context_accepts_param_coord_source() -> None:
    """ID: ASTRO_CORE_A1_006_time_context_accepts_param_coord_source."""
    ctx = resolve_direction_runtime_context(location=_lla_ao(), opts=AstroTimeOptions(source="time_s"))
    assert ctx.time.coord.name == "time_s"
    assert np.issubdtype(np.dtype(ctx.time.coord.dtype), np.number)
    assert ctx.time.sequence_dim == "sample"


def test_astro_core_a1_007_iers_options_are_explicit() -> None:
    """ID: ASTRO_CORE_A1_007_iers_options_are_explicit."""
    opts = coerce_astro_options(AstroOptions(), owner="test")
    assert opts.iers is None
    assert coerce_astro_options(None, owner="test").backend == "astropy"


def test_astro_core_a1_008_observer_context_populates_param_validity() -> None:
    """ID: ASTRO_CORE_A1_008_observer_context_populates_param_validity."""
    ctx = resolve_direction_runtime_context(location=_lla_ao_with_validity(), opts=AstroTimeOptions(source="time_s"))
    assert ctx.observer.sequence_size_coord == "group_size"
    assert ctx.observer.valid_mask is not None
    assert ctx.observer.valid_mask.dims == ("trial", "sample")
    np.testing.assert_array_equal(
        ctx.observer.valid_mask.to_numpy(),
        np.array([[True, True, False], [True, True, True]]),
    )


def test_astro_core_a1_009_finalize_preserves_observer_validity_when_sequence_preserved() -> None:
    """ID: ASTRO_CORE_A1_009_finalize_preserves_observer_validity_when_sequence_preserved."""
    ctx = resolve_direction_runtime_context(location=_lla_ao_with_validity(), opts=AstroTimeOptions(source="time_s"))
    out = finalize_topocentric_direction(
        ctx,
        _direction_dataset(),
        core_dim="enu",
        validate=True,
        owner="test",
    )
    assert read_sequence_size_coord_name(out.as_dataset(copy="none")) == "group_size"
    assert "group_size" in out.as_dataset(copy="none").coords
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["group_size"].to_numpy(), np.array([2, 3]))


def test_astro_core_a1_010_finalize_clears_validity_when_sequence_changes() -> None:
    """ID: ASTRO_CORE_A1_010_finalize_clears_validity_when_sequence_changes."""
    ctx = resolve_direction_runtime_context(location=_lla_ao_with_validity(), opts=AstroTimeOptions(source="time_s"))
    time_ctx = replace(
        ctx.time,
        coord=xr.DataArray(np.array([10.0, 20.0, 30.0]), dims=("astro_time",), name="astro_time"),
        sequence_dim="astro_time",
    )
    changed = replace(ctx, time=time_ctx, output_sequence_dim="astro_time")
    out = finalize_topocentric_direction(
        changed,
        _direction_dataset(dims=("trial", "astro_time", "enu")),
        core_dim="enu",
        validate=True,
        owner="test",
    )
    assert read_sequence_size_coord_name(out.as_dataset(copy="none")) is None
    assert "group_size" not in out.as_dataset(copy="none").coords


def test_astro_hard_a1_001_missing_astropy_backend_raises_guided_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: ASTRO_HARD_A1_001_missing_astropy_backend_raises_guided_importerror."""

    def fail_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("tal.astro.backends.astropy.import_module", fail_import)
    with pytest.raises(ImportError, match=r"tal\[astro\]"):
        require_astropy("test")


def test_astro_hard_a1_002_invalid_backend_option_fails_closed() -> None:
    """ID: ASTRO_HARD_A1_002_invalid_backend_option_fails_closed."""
    with pytest.raises(ValueError, match="backend"):
        coerce_astro_options(AstroOptions(backend="auto"), owner="test")  # type: ignore[arg-type]


def test_astro_hard_a1_003_invalid_time_source_fails_closed() -> None:
    """ID: ASTRO_HARD_A1_003_invalid_time_source_fails_closed."""
    with pytest.raises(ValueError, match="either explicit time or time.source"):
        resolve_direction_runtime_context(
            location=_lla_ao(),
            time=datetime(2026, 6, 14, 12, 0, 0),
            opts=AstroTimeOptions(source="time_s"),
        )


def test_astro_hard_a1_004_direction_rejects_position_semantics() -> None:
    """ID: ASTRO_HARD_A1_004_direction_rejects_position_semantics."""
    with pytest.raises(ValueError, match="TopocentricDirection"):
        TopocentricDirection(_direction_ao(labels=("x", "y", "z")))


def test_astro_hard_a1_005_dask_inputs_do_not_compute_silently() -> None:
    """ID: ASTRO_HARD_A1_005_dask_inputs_do_not_compute_silently."""
    values = da.from_array(np.array([[0.0, 1.0, 0.0]]), chunks=(1, 3))
    ds = xr.Dataset(
        {"direction": (("sample", "enu"), values)},
        coords={"sample": [0], "enu": ["east", "north", "up"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("enu",), validate=True)
    out = TopocentricDirection(ao)
    assert getattr(out.as_dataset(copy="none")["altitude_deg"].data, "chunks", None) is not None


def test_astro_hard_a1_006_direction_rejects_undeclared_direction_dims() -> None:
    """ID: ASTRO_HARD_A1_006_direction_rejects_undeclared_direction_dims."""
    ds = xr.Dataset(
        {
            "direction": (("sample", "extra", "enu"), np.ones((1, 2, 3), dtype=float)),
            "altitude_deg": ("sample", np.array([0.0])),
            "azimuth_deg": ("sample", np.array([90.0])),
        },
        coords={"sample": [0], "extra": ["x", "y"], "enu": ["east", "north", "up"]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("enu",), validate=True)
    with pytest.raises(ValueError, match="direction.*non-semantic dims.*extra"):
        TopocentricDirection(ao)


def test_astro_hard_a1_007_param_runtime_errors_propagate_for_param_time_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ASTRO_HARD_A1_007_param_runtime_errors_propagate_for_param_time_source."""
    import tal.astro.orchestration as orchestration

    def fail_runtime(*args: object, **kwargs: object) -> object:
        raise ValueError("core param runtime failure")

    monkeypatch.setattr(orchestration, "resolve_param_runtime_context", fail_runtime)
    with pytest.raises(ValueError, match="core param runtime failure"):
        resolve_direction_runtime_context(location=_lla_ao(), opts=AstroTimeOptions(source="time_s"))


def test_astro_hard_a1_008_raw_dask_time_fails_without_compute() -> None:
    """ID: ASTRO_HARD_A1_008_raw_dask_time_fails_without_compute."""
    from dask import delayed

    def fail_compute() -> np.ndarray:
        raise AssertionError("raw dask time unexpectedly computed")

    raw_time = da.from_delayed(delayed(fail_compute)(), shape=(1,), dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="raw lazy time arrays are not supported in astro A1"):
        resolve_direction_runtime_context(location=_lla_ao(), time=raw_time)


def test_unknown_astro_key_fails_closed_and_sibling_ext_is_preserved() -> None:
    ao = _direction_ao()
    ds = ao.as_dataset(copy="none").copy()
    attrs = deepcopy(ds.attrs)
    attrs["tal"]["ext"] = {"custom": {"ok": True}, "astro": {"kind": "topocentric_direction", "extra": "bad"}}
    ds.attrs = attrs
    with pytest.raises(ValueError, match="unknown key"):
        normalize_topocentric_metadata(ds, validate=True, owner="test")
    attrs["tal"]["ext"] = {"custom": {"ok": True}}
    ds.attrs = attrs
    out = normalize_topocentric_metadata(ds, validate=True, owner="test")
    assert out.attrs["tal"]["ext"]["custom"] == {"ok": True}
