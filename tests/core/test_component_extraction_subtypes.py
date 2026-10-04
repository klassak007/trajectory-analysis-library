"""Contract 046: generic extraction returns isolated, resource-coupled base AOs."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import (
    AnalysisLayoutSpec,
    AnalysisObject,
    ComponentExtractOptions,
    ComponentPatchOptions,
    extract_components,
    patch_components,
    read_components,
)
from tal.frames import FrameGraph
from tal.spatial import (
    Acceleration,
    AngularAcceleration,
    AngularVelocity,
    LinearAcceleration,
    LinearVelocity,
    Pose,
    Position,
    Rotation,
    Velocity,
)
from tests.core.test_viz_holoviews import _install_fake_hv_modules


class ExternalComposite(AnalysisObject):
    pass


def _composite(kind, *, lazy, empty):
    n = 0 if empty else 3
    ds = xr.Dataset({"p": (("sample", "axis"), np.arange(n * 3).reshape(n, 3).astype(float)),
                     "q": (("sample", "quat"), np.tile([0., 0., 0., 1.], (n, 1)))},
                    coords={"sample": np.arange(n), "time": ("sample", np.arange(n).astype(float)), "axis": list("xyz"), "quat": list("xyzw")},
                    attrs={"notes": {"source": ["fixture"]}})
    if lazy:
        ds = ds.chunk({"sample": 1})
    p = Position(AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",), param_coord="time").wrap(ds[["p"]]))
    q = Rotation(AnalysisLayoutSpec(sequence_dim="sample", core_dims=("quat",), param_coord="time").wrap(ds[["q"]]))
    if kind in ("Pose", "AO", "external"):
        pose = Pose.from_components(q, p).with_graph(FrameGraph())
        if kind == "Pose":
            return pose
        cls = ExternalComposite if kind == "external" else AnalysisObject
        return cls(pose.as_dataset())
    linear, angular, composite = (LinearVelocity, AngularVelocity, Velocity) if kind == "Velocity" else (LinearAcceleration, AngularAcceleration, Acceleration)
    return composite.from_linear_angular(linear(p), angular(p.rename({"axis": "spin", "p": "angular"})))


@pytest.mark.parametrize("kind", ["AO", "external", "Pose", "Velocity", "Acceleration"])
@pytest.mark.parametrize("lazy,empty", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("validate", [False, True])
def test_extraction_returns_base_and_preserves_ownership(kind, lazy, empty, validate):
    """ID: COMPONENT_BASE_RESULT_001; output type, registry, labels, and lifetime."""
    source = _composite(kind, lazy=lazy, empty=empty)
    source.as_dataset(copy="none").attrs["notes"] = {"source": ["fixture"]}
    before = source.as_dataset()
    closed = []
    source.as_dataset(copy="none").set_close(lambda: closed.append(True))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        parts = extract_components(source, validate=validate)
    assert not tasks
    for name, part in parts.items():
        assert type(part) is AnalysisObject
        spec = read_components(source)[name]
        actual = part.as_dataset(copy="shallow")
        expected = before[spec.var].sel({spec.core_dim: list(spec.labels)})
        xr.testing.assert_equal(actual[spec.var], expected)
        assert name in read_components(part)
        actual.attrs["notes"]["source"].append("changed")
        assert source.as_dataset(copy="none").attrs["notes"]["source"] == ["fixture"]
    roundtrip = patch_components(source, parts, opts=ComponentPatchOptions(on_overlap="replace"), validate=validate)
    xr.testing.assert_equal(roundtrip.as_dataset(), before)
    for part in parts.values():
        part.close()
    source.close()
    assert closed == [True]
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("kind", ["Pose", "Velocity", "Acceleration"])
@pytest.mark.parametrize("plot_kind", ["line", "scatter", "explorer"])
def test_registered_composite_plotting(kind, plot_kind, monkeypatch):
    """ID: COMPONENT_BASE_RESULT_002; every plotting root accepts generic components."""
    pytest.importorskip("hvplot")
    hv = pytest.importorskip("holoviews")
    hv.extension("bokeh")
    source = _composite(kind, lazy=False, empty=False)
    name = next(iter(read_components(source)))
    if plot_kind == "explorer":
        # hvPlot 0.12.2 cannot explore a string-labeled DataArray independently
        # of TAL. Check the public adapter payload with the existing backend double.
        _install_fake_hv_modules(monkeypatch)
        plot = source.viz.component(name, kind=plot_kind)
        spec = read_components(source)[name]
        np.testing.assert_array_equal(plot["data"][spec.var], source.as_dataset()[spec.var])
    else:
        assert source.viz.component(name, kind=plot_kind) is not None


def test_component_output_rename_preserves_registry_and_typed_decomposition():
    """ID: COMPONENT_BASE_RESULT_003; typed decomposition remains domain-owned."""
    source = _composite("Pose", lazy=False, empty=False)
    part = source.components.extract(opts=ComponentExtractOptions(names=("position",), output_var="selected"))["position"]
    assert set(part.as_dataset().data_vars) == {"selected"}
    assert read_components(part)["position"].var == "selected"
    position, rotation = source.decompose()
    assert type(position) is Position and type(rotation) is Rotation
    assert position.graph is source.graph and rotation.graph is source.graph
