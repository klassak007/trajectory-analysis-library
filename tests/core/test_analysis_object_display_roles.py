from __future__ import annotations

import html
from html.parser import HTMLParser

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal import AnalysisObject
from tal.core.dataset_ownership import analysis_object_dataset


class _CoordinateRows(HTMLParser):
    """Read the rendered coordinate names and role cells, without HTML snapshots."""

    def __init__(self, text):
        super().__init__()
        self.rows = {}
        self.current = None
        self.field = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "").split()
        if tag == "li" and "tal-coordinate-row" in classes:
            self.current = {"xr-var-name": "", "tal-coordinate-role": ""}
        if tag == "div" and self.current is not None:
            self.field = next((name for name in self.current if name in classes), None)

    def handle_endtag(self, tag):
        if tag == "div":
            self.field = None
        if tag == "li" and self.current is not None:
            self.rows[self.current["xr-var-name"]] = self.current["tal-coordinate-role"]
            self.current = None

    def handle_data(self, text):
        if self.current is not None and self.field:
            self.current[self.field] += text


def _source(*, lazy=False, empty=False, parameter="time"):
    count = 0 if empty else 3
    data = np.ones((2, count, 3))
    times = np.arange(count, dtype=float)
    if lazy:
        data = da.from_array(data, chunks=(1, 2, 3))
        times = da.from_array(times, chunks=2)
    return AnalysisObject.from_data(
        xr.Dataset(
            {"speed": (("trial", "sample", "axis"), data)},
            coords={"trial": ["a", "b"], "sample": np.arange(count), "axis": list("xyz"),
                    "time": ("sample", times), "aux": ("sample", np.arange(count))},
        ),
        sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",), param_coord=parameter,
    )


def _text_roles(output):
    section = output.split("Coordinates:\n", 1)[1].split("Data variables:", 1)[0]
    heading, *rows = section.splitlines()
    start = heading.index("Role")
    result = {}
    for line in rows:
        if not line.strip() or "Dimensions without coordinates:" in line:
            continue
        name = line[:start].strip().lstrip("* ")
        result[name] = line[start:].split("(", 1)[0].strip()
    return result


@pytest.mark.parametrize("mode", ["text", "html", "text-style"])
@pytest.mark.parametrize("lazy,empty", [(False, False), (True, False), (True, True)])
def test_ao_display_011_coordinate_role_column(mode, lazy, empty):
    """ID: AO_DISPLAY_011_coordinate_role_column."""
    source = _source(lazy=lazy, empty=empty)
    before = source.as_dataset()
    tasks = []
    with (
        Callback(pretask=lambda *args: tasks.append(args[0])),
        xr.set_options(display_style="text" if mode == "text-style" else "html"),
    ):
        output = repr(source) if mode == "text" else source._repr_html_()
    if mode == "html":
        roles = _CoordinateRows(output).rows
    else:
        roles = _text_roles(html.unescape(output).removeprefix("<pre>").removesuffix("</pre>"))
    assert roles == {"trial": "batch", "sample": "sequence", "axis": "core", "time": "parameter", "aux": ""}
    assert tasks == []
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("indexed", [False, True])
def test_ao_display_012_multiple_roles_and_index_independence(indexed):
    """ID: AO_DISPLAY_012_multiple_roles_and_missing_coordinates."""
    source = _source(parameter="sample")
    if not indexed:
        source = AnalysisObject(source.as_dataset().drop_indexes("sample"))
    assert _text_roles(repr(source))["sample"] == "sequence, parameter"
    assert _CoordinateRows(source._repr_html_()).rows["sample"] == "sequence, parameter"
    row = next(line for line in repr(source).splitlines() if line.lstrip(" *").startswith("sample "))
    assert row.startswith("  *") == indexed
    without_labels = AnalysisObject(_source().as_dataset().drop_vars("sample"))
    assert "sample" not in _CoordinateRows(without_labels._repr_html_()).rows
    assert "Sequence: sample" in repr(without_labels)
    assert "Dimensions without coordinates: sample" in repr(without_labels)


def test_ao_display_013_role_diagnostics_and_display_settings(monkeypatch):
    """ID: AO_DISPLAY_013_role_diagnostics_and_display_settings."""
    from xarray.core import formatting_html

    source = _source()
    schema = analysis_object_dataset(source).attrs["tal"]
    schema["core"]["roles"]["batch_dims"] = object()
    schema["core"]["roles"]["sequence_dim"] = "aux"
    schema["core"]["param_coord"]["name"] = "absent"
    assert _CoordinateRows(source._repr_html_()).rows == {
        "trial": "", "sample": "", "axis": "core", "time": "", "aux": "",
    }
    assert "Batch: unavailable" in repr(source)
    with xr.set_options(display_expand_coords=False):
        assert "Coordinates: (5)" in repr(source)
        assert "Role\n" not in repr(source)
    with xr.set_options(display_max_rows=2, display_width=60):
        output = repr(source)
        assert "Coordinates: (2/5)" in output
        assert "Role\n" in output
    monkeypatch.setattr(formatting_html, "summarize_variable", None)
    assert source._repr_html_() == f"<pre>{html.escape(repr(source))}</pre>"


def test_ao_display_014_role_names_remain_escaped():
    """ID: AO_DISPLAY_014_role_coordinate_escaping."""
    source = _source().rename({"sample": "<sample>"})
    output = source._repr_html_()
    assert _CoordinateRows(output).rows["<sample>"] == "sequence"
    assert "<sample>" not in output
