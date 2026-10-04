"""Public regressions for the four tutorial-discovered boundary defects."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisLayoutSpec, SequenceConcatOptions
from tal.core.event_ops import AtBoundariesOptions
from tal.frames import FrameGraph
from tal.io import CsvIngestOptions, read_csv_logs
from tal.spatial import Pose, Position

DATA = Path(__file__).resolve().parents[2] / "examples/tutorial/data/flight_telemetry"


def _logs(folder):
    return read_csv_logs(str(DATA / folder / "*.csv"), opts=CsvIngestOptions(time_col="time"))


def _ship_graph():
    graph = FrameGraph()
    ship = Pose.from_fields(
        _logs("ship_poses"), position="ship.position.{x,y,z}", rotation="ship.rotation.{x,y,z,w}",
        parent="world", child="ship", graph=graph,
    ).register()
    return graph, ship


def test_indexed_sequence_concat_keeps_both_segments():
    """Contract 013 §8 requires contiguous packing in input order."""
    layout = AnalysisLayoutSpec(sequence_dim="sample", param_coord="time")
    head = layout.wrap(xr.Dataset({"v": ("sample", [1., 2.])}, coords={"sample": [0, 1], "time": ("sample", [0., 1.])}))
    tail = layout.wrap(xr.Dataset({"v": ("sample", [3., 4.])}, coords={"sample": [2, 3], "time": ("sample", [2., 3.])}))
    out = head.combine.concat_sequence([tail], opts=SequenceConcatOptions(overlap="error"))
    np.testing.assert_allclose(out.to_dataarray(copy="shallow"), [1., 2., 3., 4.])
    np.testing.assert_allclose(out.as_dataset(copy="shallow")["time"], [0., 1., 2., 3.])


def test_boundary_timestamp_can_anchor_window():
    """Contracts 009/027 reserve regenerated metadata for the current operation."""
    altitude = _logs("observations").select_vars("altitude")
    condition = altitude < 2.
    boundary = altitude.events.at_boundaries(condition, opts=AtBoundariesOptions(edges="enter", mode="first"))
    anchors = boundary.as_dataset(copy="shallow")["time"]
    actual = altitude.events.around(anchors, pre=0.1, post=0.1, dt=0.05, layout="segments")
    expected = altitude.events.around(condition, edge="enter", pre=0.1, post=0.1, dt=0.05, layout="segments")
    np.testing.assert_allclose(actual.to_dataarray(copy="shallow"), expected.to_dataarray(copy="shallow"))


def test_ragged_expression_ignores_padding():
    """Contracts 088/129 preserve caller query validity on dynamic expressions."""
    graph, _ = _ship_graph()
    position = Position.from_fields(_logs("observations"), "drone.position.{x,y,z}", parent="world", child="drone", graph=graph)
    out = position.express_in("ship", graph=graph)
    assert out.frames.ids() == ("world", "drone")
    source = position.as_dataset(copy="shallow")
    target = out.as_dataset(copy="shallow")
    np.testing.assert_array_equal(target["sequence_size"], source["sequence_size"])
    assert np.isnan(target["position"].isel(trial=-1, sample=slice(101, None))).all()


def test_pose_component_plot_accepts_registered_position():
    """Contract 101 and docs/api/viz.md promise registered-component plotting."""
    pytest.importorskip("hvplot")
    hv = pytest.importorskip("holoviews")
    hv.extension("bokeh")
    _, ship = _ship_graph()
    plot = ship.isel(trial=0).viz.component("position", kind="line")
    assert plot is not None
