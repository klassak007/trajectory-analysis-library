from __future__ import annotations

import ast
import html
import tracemalloc
from html.parser import HTMLParser

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal import AnalysisObject
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.schema import set_roles
from tal.frames import FrameGraph
from tal.linalg import Vector3
from tal.spatial import (
    AngularVelocity,
    LinearVelocity,
    Pose,
    Position,
    Rotation,
    Velocity,
)


def _signal(*, lazy=False, empty=False):
    values = np.arange(8.0).reshape(2, 4)[:, :0 if empty else 4]
    data = da.from_array(values, chunks=(1, 2)) if lazy else values
    times = da.from_array(np.arange(values.shape[1]), chunks=2) if lazy else np.arange(values.shape[1])
    return AnalysisObject.from_data(
        xr.Dataset(
            {"signal": (("trial", "sample"), data)},
            coords={"trial": ["a", "b"], "time": ("sample", times),
                    "length": ("trial", [values.shape[1], max(0, values.shape[1] - 1)])},
            attrs={"experiment": "walking"},
        ),
        sequence_dim="sample", batch_dims=("trial",), core_dims=(),
        param_coord="time", sequence_size_coord="length",
    )


def _render(value, mode):
    if mode == "text":
        return repr(value)
    if mode == "text-style":
        with xr.set_options(display_style="text"):
            return value._repr_html_()
    return value._repr_html_()


class _HTML(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.tags = []
        self.words = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.words.append(data)


@pytest.mark.parametrize("mode", ["text", "html", "text-style"])
@pytest.mark.parametrize("lazy,empty", [(False, False), (True, False), (True, True)])
def test_ao_display_001_semantics_topology_and_laziness(mode, lazy, empty, capsys):
    """ID: AO_DISPLAY_001_semantics_topology_and_laziness."""
    source = _signal(lazy=lazy, empty=empty)
    ds = analysis_object_dataset(source)
    before = ds.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        output = _render(source, mode)
        print(source)
    assert tasks == []
    for text in (output, capsys.readouterr().out):
        for name in ("tal.AnalysisObject", "Dimensions", "TAL:", "Sequence", "Batch", "Core",
                     "Parameter", "Validity", "length", "left_packed", "Coordinates", "signal", "walking"):
            assert name in text
        assert "xarray.Dataset" not in text
    assert "dask.array" in output if lazy else "float64" in output
    xr.testing.assert_identical(ds, before)


@pytest.mark.parametrize("ds", [xr.Dataset(), xr.Dataset({"scalar": 4}), xr.Dataset({"a": 1, "b": 2})])
def test_ao_display_002_bootstrap_and_empty_roles(ds):
    """ID: AO_DISPLAY_002_bootstrap_and_declared_empty_roles."""
    bootstrap = AnalysisObject(ds)
    assert "Roles: undeclared" in repr(bootstrap)
    declared = AnalysisObject(set_roles(ds, sequence_dim=None, batch_dims=(), core_dims=()))
    output = repr(declared)
    assert "Sequence: none" in output
    assert "Batch: ()" in output and "Core: ()" in output
    assert "Parameter:" not in output and "Validity:" not in output


def _position():
    return Position(AnalysisObject.from_data(
        xr.Dataset({"point": ("axis", [1.0, 2.0, 3.0])}, coords={"axis": list("xyz")}),
        sequence_dim=None, core_dims=("axis",),
    ), parent="world", child="body")


def _rotation():
    return Rotation(AnalysisObject.from_data(
        xr.Dataset({"q": ("quat", [0.0, 0.0, 0.0, 1.0])}, coords={"quat": list("xyzw")}),
        sequence_dim=None, core_dims=("quat",),
    ), parent="world", child="body")


def _velocity():
    source = Position(AnalysisObject.from_data(
        _position().as_dataset().expand_dims(sample=[0]), sequence_dim="sample", core_dims=("axis",),
    ))
    return Velocity.from_linear_angular(
        LinearVelocity(source), AngularVelocity(source.rename({"axis": "angular_axis", "point": "spin"})),
    )


@pytest.mark.parametrize("construct", [_position, _rotation, lambda: Pose.from_components(_rotation(), _position()),
                                      lambda: Vector3(_position()), _velocity])
def test_ao_display_003_plain_and_typed_declarations(construct):
    """ID: AO_DISPLAY_003_concrete_type_and_domain_rows."""
    typed = construct()
    plain = AnalysisObject(typed.as_dataset())
    text = repr(typed)
    assert f"<tal.{type(typed).__name__}>" in text
    assert "Frames:" not in repr(plain) and "Spatial:" not in repr(plain)
    assert "Extensions:" in repr(plain)
    if not isinstance(typed, Vector3):
        assert "Frames:" in text and "world" in text and "body" in text
        assert "Spatial:" in text
    for value in (typed, plain):
        assert "TAL schema" in value._repr_html_()
        assert "frames" in value._repr_html_()


def test_ao_display_003_external_subclass_uses_actual_module():
    class MySignal(AnalysisObject):
        pass

    source = MySignal(xr.Dataset({"value": 1}))
    assert f"<{MySignal.__module__}.{MySignal.__qualname__}>" in repr(source)
    assert "tal.MySignal" not in source._repr_html_()


def test_ao_display_004_stored_only_fault_isolation():
    """ID: AO_DISPLAY_004_stored_only_fault_isolation."""
    source = _position()
    schema = analysis_object_dataset(source).attrs["tal"]
    schema["core"]["roles"]["sequence_dim"] = "missing_sequence"
    schema["core"]["roles"]["batch_dims"] = 7
    schema["core"]["param_coord"] = {"name": "missing_time"}
    schema["core"]["validity"] = "broken"
    schema["ext"]["components"] = {"registry": {"missing_component": {"var": "does_not_exist"}}}
    schema["ext"]["spatial"].pop("representation")
    schema["ext"]["frames"] = "broken"
    output = repr(source)
    for expected in ("missing_sequence", "Batch: unavailable", "Core: (axis)", "missing_time",
                     "Validity: unavailable", "missing_component", "Frames: unavailable", "point"):
        assert expected in output
    assert "Spatial:" not in output
    assert schema["core"]["roles"]["sequence_dim"] == "missing_sequence"
    assert schema["ext"]["frames"] == "broken"


class _Opaque:
    def __repr__(self):
        raise AssertionError("custom repr must not be invoked")

    def __str__(self):
        raise AssertionError("custom str must not be invoked")


@pytest.mark.parametrize("payload,marker", [
    (_Opaque(), "value omitted"), ([0] * 1000, "items omitted"), ("z" * 30000, "text omitted"),
    (10**100000, "text omitted"), ([[[[[[[[[[0]]]]]]]]]], "depth omitted"),
], ids=["opaque", "many-items", "large-text", "large-integer", "deep-container"])
def test_ao_display_005_bounded_schema(payload, marker):
    """ID: AO_DISPLAY_005_bounded_and_opaque_metadata."""
    source = AnalysisObject(xr.Dataset({"value": 1}))
    analysis_object_dataset(source).attrs["tal"]["ext"] = {"custom": payload}
    output = source._repr_html_()
    assert marker in output
    assert len(output) < 120000


def test_ao_display_005_cycles_and_shared_references():
    source = AnalysisObject(xr.Dataset({"value": 1}))
    shared = ["shared", 3]
    cycle = [None]
    cycle[0] = cycle
    analysis_object_dataset(source).attrs["tal"]["ext"] = {"first": shared, "second": shared, "cycle": cycle}
    output = html.unescape(source._repr_html_())
    assert output.count("['shared', 3]") == 2
    assert "recursive container: omitted" in output


def test_ao_display_005_exact_builtin_detail_and_allocation():
    from tal.utils.display_values import DETAIL_BUDGET, format_stored

    value = {"tuple": (1,), "set": {1, 2}, "text": "<tag>", "bytes": b"abc", "complex": 1 + 2j}
    assert ast.literal_eval(format_stored(value, budget=DETAIL_BUDGET)) == value
    assert int(format_stored(10**5000, budget=DETAIL_BUDGET), 16) == 10**5000
    huge = {"text": "x" * 10_000_000, "items": [0] * 1_000_000}
    tracemalloc.start()
    try:
        output = format_stored(huge, budget=DETAIL_BUDGET)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(output) < 17000
    assert peak < 1_000_000


class _Transform(xr.indexes.CoordinateTransform):
    def __init__(self, calls):
        super().__init__(("label", "other_label"), {"orphan": 3})
        self.calls = calls

    def forward(self, *args, **kwargs):
        self.calls.append("forward")
        raise AssertionError("display must not transform coordinates")

    def reverse(self, *args, **kwargs):
        self.calls.append("reverse")
        raise AssertionError("display must not transform coordinates")

    def equals(self, other, **kwargs):
        return self is other


@pytest.mark.parametrize("mode", ["text", "html", "text-style", "fallback"])
def test_ao_display_006_transform_safe_projection(mode, monkeypatch):
    """ID: AO_DISPLAY_006_transform_safe_native_projection."""
    calls = []
    coords = xr.Coordinates.from_xindex(xr.indexes.CoordinateTransformIndex(_Transform(calls)))
    source = AnalysisObject(xr.Dataset({"value": ("sample", da.ones(2, chunks=1))}, coords=coords))
    calls = analysis_object_dataset(source).xindexes["label"].transform.calls
    if mode == "fallback":
        from xarray.core import formatting_html

        monkeypatch.setattr(formatting_html, "_obj_repr", None)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        output = _render(source, mode)
    assert tasks == [] and calls == []
    for expected in ("label", "other_label", "orphan", "3", "CoordinateTransformIndex",
                     "values omitted: transform-backed", "value"):
        assert expected in output
    assert set(analysis_object_dataset(source).xindexes) == {"label", "other_label"}
    assert dict(analysis_object_dataset(source).sizes) == {"sample": 2, "orphan": 3}


@pytest.mark.parametrize("kind", ["pandas", "range", "point"])
def test_ao_display_007_native_index_preview(kind):
    """ID: AO_DISPLAY_007_native_indexes_and_options."""
    ds = xr.Dataset({"value": ("sample", [1., 2., 3.])}, coords={"sample": [0, 1, 2]})
    if kind == "range":
        ds = xr.Dataset({"value": ("sample", [1., 2., 3.])},
                        coords=xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(3, dim="sample")))
    if kind == "point":
        ds = ds.assign_coords(x=("sample", [0., 1., 2.]))
        ds = ds.set_xindex("x", xr.indexes.NDPointIndex)
    source = AnalysisObject(ds)
    with xr.set_options(display_default_indexes=True, display_expand_indexes=True, display_max_rows=2):
        text = repr(source)
        assert "values omitted: transform-backed" in text if kind == "range" else "Indexes:" in text
        for index in ds.xindexes.values():
            assert type(index).__name__ in text
        assert ("transform-backed" if kind == "range" else "Indexes") in source._repr_html_()


def test_ao_display_008_escaping_controls_options_and_fallback(monkeypatch):
    """ID: AO_DISPLAY_008_html_escaping_controls_and_fallback."""
    source = AnalysisObject(xr.Dataset({"<script>alert(1)</script>": 1}, attrs={"<tag>": "<script>"}))
    analysis_object_dataset(source).attrs["tal"]["ext"] = {"<script>": "<img onerror='bad'>"}
    with xr.set_options(display_expand_data_vars=False, display_expand_attrs=False):
        first = _HTML(source._repr_html_())
        second = _HTML(source._repr_html_())
    assert not any(tag in ("script", "img") for tag, _ in first.tags)
    ids = lambda parsed: {attrs["id"] for tag, attrs in parsed.tags if tag == "input" and "id" in attrs}
    assert ids(first).isdisjoint(ids(second))
    labels = {attrs["for"] for tag, attrs in first.tags if tag == "label"}
    assert labels <= ids(first)
    assert any("TAL schema" in text for text in first.words)
    assert not any("checked" in attrs for tag, attrs in first.tags
                   if tag == "input" and attrs.get("class") == "xr-section-summary-in")
    from xarray.core import formatting_html

    def fail(*args):
        raise RuntimeError("incompatible upstream formatter")

    monkeypatch.setattr(formatting_html, "_obj_repr", fail)
    assert source._repr_html_() == f"<pre>{html.escape(repr(source))}</pre>"


def test_ao_display_009_ownership_intent_and_resource_lifetime(monkeypatch):
    """ID: AO_DISPLAY_009_readonly_alias_and_resource_lifetime."""
    source = _position().with_graph(FrameGraph())
    ds = analysis_object_dataset(source)
    closed = []
    ds.set_close(lambda: closed.append(True))
    alias = source.b()
    before = ds.copy(deep=True)
    callback = ds._close
    graph = source.graph
    intent = alias._broadcast_intent

    def forbidden(*args, **kwargs):
        raise AssertionError("display must not expose or validate the Dataset")

    monkeypatch.setattr(AnalysisObject, "as_dataset", forbidden)
    for value in (source, alias):
        repr(value)
        value._repr_html_()
    assert closed == [] and ds._close is callback
    assert source.graph is graph and alias._broadcast_intent is intent
    xr.testing.assert_identical(ds, before)
    alias.close()
    source.close()
    assert closed == [True]


@pytest.mark.parametrize("path", [(), ("core",), ("core", "roles"), ("ext",)])
def test_ao_display_004_malformed_envelopes_keep_data_visible(path):
    source = _signal()
    parent = analysis_object_dataset(source).attrs
    keys = ("tal", *path)
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = _Opaque()
    for mode in ("text", "html", "text-style"):
        output = _render(source, mode)
        assert "unavailable" in output and "signal" in output and "Coordinates" in output


def test_ao_display_007_long_names_follow_width_and_row_settings():
    source = _signal()
    schema = analysis_object_dataset(source).attrs["tal"]
    schema["core"]["param_coord"]["name"] = "a" * 100000
    schema["ext"] = {"components": {"registry": {f"component{i}": {} for i in range(1000)}}}
    with xr.set_options(display_width=45, display_max_rows=2):
        text = repr(source)
    summary = text.split("TAL:\n", 1)[1].split("Coordinates:", 1)[0]
    assert all(len(line) <= 45 for line in summary.splitlines())
    assert "omitted" in text
    assert len(summary) < 2600


def test_ao_display_009_eager_payload_is_not_copied():
    source = AnalysisObject(xr.Dataset({"signal": ("sample", np.zeros(1_000_000))}))
    data = analysis_object_dataset(source)["signal"].data
    repr(source)
    source._repr_html_()
    tracemalloc.start()
    try:
        repr(source)
        source._repr_html_()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000
    assert analysis_object_dataset(source)["signal"].data is data


def test_ao_display_010_notebook_mime_protocol():
    """ID: AO_DISPLAY_010_notebook_protocol."""
    ipython = pytest.importorskip("IPython.core.formatters")
    formatter = ipython.DisplayFormatter()
    source = _signal(lazy=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        bundle, metadata = formatter.format(source)
    assert tasks == []
    assert "tal.AnalysisObject" in bundle["text/plain"]
    assert "TAL schema" in bundle["text/html"]
    assert metadata == {}
