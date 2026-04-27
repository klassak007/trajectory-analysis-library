from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
import xarray as xr

from tal import AnalysisObject
from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec
from tal.core.group_ops import GroupingFoundationOptions
from tal.viz import AOVizOptions
from tal.viz import line as viz_line


class _FakeHvAccessor:
    def __init__(self, data: xr.DataArray | xr.Dataset) -> None:
        self._data = data

    def line(self, **kwargs: Any) -> dict[str, Any]:
        return {"method": "line", "kwargs": kwargs, "data": self._data}

    def scatter(self, **kwargs: Any) -> dict[str, Any]:
        return {"method": "scatter", "kwargs": kwargs, "data": self._data}

    def explorer(self, **kwargs: Any) -> dict[str, Any]:
        return {"method": "explorer", "kwargs": kwargs, "data": self._data}


def _install_fake_hv_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def _fake_import(name: str) -> Any:
        if name == "holoviews":
            return object()
        if name == "hvplot.xarray":
            return object()
        return original(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    def _hvplot(self: xr.DataArray) -> _FakeHvAccessor:
        return _FakeHvAccessor(self)

    def _ds_hvplot(self: xr.Dataset) -> _FakeHvAccessor:
        return _FakeHvAccessor(self)

    monkeypatch.setattr(xr.DataArray, "hvplot", property(_hvplot), raising=False)
    monkeypatch.setattr(xr.Dataset, "hvplot", property(_ds_hvplot), raising=False)


def _build_base_ao(*, multivar: bool = False, with_param_coord: bool = True, with_sequence_coord: bool = True) -> AnalysisObject:
    signal = np.asarray(
        [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
        dtype=float,
    )
    ds = xr.Dataset(
        data_vars={"signal": (("trial", "time"), signal)},
        coords={
            "trial": ("trial", np.asarray([0, 1], dtype=np.int64)),
            "size": ("trial", np.asarray([2, 3], dtype=np.int64)),
        },
    )
    if with_sequence_coord:
        ds = ds.assign_coords(time=("time", np.asarray([0.0, 1.0, 2.0], dtype=float)))
    if with_param_coord:
        ds = ds.assign_coords(t=("time", np.asarray([0.1, 1.1, 2.1], dtype=float)))
    if multivar:
        ds["other"] = (("trial", "time"), signal + 100.0)
    return AnalysisObject.from_data(
        ds,
        sequence_dim="time",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="t" if with_param_coord else None,
        sequence_size_coord="size",
        validate=True,
    )


def _build_component_ao() -> AnalysisObject:
    values = np.arange(18, dtype=float).reshape(2, 3, 3)
    ds = xr.Dataset(
        data_vars={"signal": (("trial", "time", "xyz"), values)},
        coords={
            "trial": ("trial", np.asarray([0, 1], dtype=np.int64)),
            "time": ("time", np.asarray([0.0, 1.0, 2.0], dtype=float)),
            "xyz": ("xyz", np.asarray(["x", "y", "z"], dtype=object)),
            "size": ("trial", np.asarray([3, 3], dtype=np.int64)),
        },
    )
    base = AnalysisObject.from_data(
        ds,
        sequence_dim="time",
        batch_dims=("trial",),
        core_dims=("xyz",),
        sequence_size_coord="size",
        validate=True,
    )
    return base.components.define(
        opts=ComponentRegistryOptions(
            registry={"xyz": ComponentSpec(core_dim="xyz", labels=("x", "y", "z"))},
            replace=True,
        ),
        validate=True,
    )


def test_viz_hard_001_missing_holoviews_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_HARD_001_missing_holoviews_fails_closed."""
    ao = _build_base_ao()
    original = importlib.import_module

    def _fake_import(name: str) -> Any:
        if name == "holoviews":
            raise ImportError("missing holoviews")
        return original(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(ImportError, match="ao.viz.line: holoviews and hvplot are required"):
        ao.viz.line()


def test_viz_hard_002_missing_hvplot_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_HARD_002_missing_hvplot_fails_closed."""
    ao = _build_base_ao()
    original = importlib.import_module

    def _fake_import(name: str) -> Any:
        if name == "holoviews":
            return object()
        if name == "hvplot.xarray":
            raise ImportError("missing hvplot")
        return original(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(ImportError, match="ao.viz.line: holoviews and hvplot are required"):
        ao.viz.line()


def test_viz_core_001_x_precedence_param_then_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_001_x_precedence_param_then_explicit."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao(with_param_coord=True)
    default_plot = ao.viz.line()
    assert default_plot["kwargs"]["x"] == "t"
    explicit_plot = ao.viz.line(opts=AOVizOptions(x="time"))
    assert explicit_plot["kwargs"]["x"] == "time"


def test_viz_core_002_x_fallback_to_synthetic_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_002_x_fallback_to_synthetic_index."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao(with_param_coord=False, with_sequence_coord=False)
    plot = ao.viz.line()
    x_name = plot["kwargs"]["x"]
    assert x_name.startswith("time_index")
    assert x_name in plot["data"].coords


def test_viz_core_003_line_requires_single_numeric_var_without_explicit_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_003_line_requires_single_numeric_var_without_explicit_var."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao(multivar=True)
    with pytest.raises(ValueError, match="requires exactly one data variable"):
        ao.viz.line()


def test_viz_core_004_explorer_uses_dataset_for_multivar_without_explicit_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_004_explorer_uses_dataset_for_multivar_without_explicit_var."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao(multivar=True)
    plot = ao.viz.explorer()
    assert plot["method"] == "explorer"
    assert isinstance(plot["data"], xr.Dataset)
    assert tuple(sorted(str(name) for name in plot["data"].data_vars)) == ("other", "signal")


def test_viz_core_005_validity_respect_vs_ignore(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_005_validity_respect_vs_ignore."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao()
    masked = ao.viz.line()["data"]
    unmasked = ao.viz.line(opts=AOVizOptions(validity="ignore"))["data"]
    assert np.isnan(float(masked.sel(trial=0, time=2.0).item()))
    assert float(unmasked.sel(trial=0, time=2.0).item()) == 3.0


def test_viz_core_006_explicit_by_overlay_cap_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_006_explicit_by_overlay_cap_fail_closed."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao()
    with pytest.raises(ValueError, match="use opts.groupby"):
        ao.viz.line(opts=AOVizOptions(by=("trial",), max_overlay_items=1))


def test_viz_core_007_group_key_defaults_channel_and_attaches_coord(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_007_group_key_defaults_channel_and_attaches_coord."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao()
    plot = ao.viz.line(opts=AOVizOptions(group_key="trial"))
    by_channel = plot["kwargs"]["by"]
    assert isinstance(by_channel, str)
    assert by_channel.startswith("__tal_group_key__")
    assert by_channel in plot["data"].coords


def test_viz_core_008_group_key_uses_foundation_na_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_008_group_key_uses_foundation_na_policy."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao()
    group_key = xr.DataArray(
        np.asarray([[1.0, 2.0, np.nan], [1.0, 2.0, 3.0]], dtype=float),
        dims=("trial", "time"),
        coords={
            "trial": ao.unsafe_data.coords["trial"],
            "time": ao.unsafe_data.coords["time"],
        },
    )
    with pytest.raises(ValueError, match="na_key_policy='error'"):
        ao.viz.line(opts=AOVizOptions(group_key=group_key))
    grouped = ao.viz.line(
        opts=AOVizOptions(
            group_key=group_key,
            group_foundation_opts=GroupingFoundationOptions(na_key_policy="group"),
        )
    )
    assert grouped["method"] == "line"


def test_viz_core_009_component_unknown_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_009_component_unknown_fails_closed."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_component_ao()
    with pytest.raises(ValueError, match="unknown component"):
        ao.viz.component("missing")


def test_viz_core_010_component_default_by_and_groupby(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_010_component_default_by_and_groupby."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_component_ao()
    by_plot = ao.viz.component("xyz", kind="line")
    assert by_plot["kwargs"]["by"] == "xyz"
    explorer_plot = ao.viz.component("xyz", kind="explorer")
    assert explorer_plot["kwargs"]["by"] == ["xyz"]
    group_plot = ao.viz.component("xyz", kind="line", opts=AOVizOptions(max_overlay_items=2))
    assert group_plot["kwargs"]["groupby"] == "xyz"


def test_viz_core_011_component_explicit_channel_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_011_component_explicit_channel_overrides_defaults."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_component_ao()
    plot = ao.viz.component("xyz", kind="line", opts=AOVizOptions(by=("trial",)))
    assert plot["kwargs"]["by"] == "trial"


def test_viz_core_012_functional_surface_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_CORE_012_functional_surface_parity."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao()
    functional = viz_line(ao)
    accessor = ao.viz.line()
    assert functional["method"] == accessor["method"]
    assert functional["kwargs"]["x"] == accessor["kwargs"]["x"]


def test_viz_hard_003_explorer_missing_on_hvplot_accessor_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: VIZ_HARD_003_explorer_missing_on_hvplot_accessor_fails_closed."""
    _install_fake_hv_modules(monkeypatch)
    ao = _build_base_ao(multivar=True)

    class _NoExplorerAccessor:
        def __init__(self, data: xr.DataArray | xr.Dataset) -> None:
            self._data = data

        def line(self, **kwargs: Any) -> dict[str, Any]:
            return {"method": "line", "kwargs": kwargs, "data": self._data}

        def scatter(self, **kwargs: Any) -> dict[str, Any]:
            return {"method": "scatter", "kwargs": kwargs, "data": self._data}

    def _hvplot(self: xr.DataArray) -> _NoExplorerAccessor:
        return _NoExplorerAccessor(self)

    def _ds_hvplot(self: xr.Dataset) -> _NoExplorerAccessor:
        return _NoExplorerAccessor(self)

    monkeypatch.setattr(xr.DataArray, "hvplot", property(_hvplot), raising=False)
    monkeypatch.setattr(xr.Dataset, "hvplot", property(_ds_hvplot), raising=False)
    with pytest.raises(ValueError, match="does not provide explorer\\(\\)"):
        ao.viz.explorer()
