from __future__ import annotations

import pytest
import xarray as xr

from tal import AnalysisObject
from tal.core.schema_errors import SchemaError
from tal.frames import Frame, FrameGraph
from tal.spatial import Position
from tal.utils.frame_ops import (
    frame_ids,
    frame_remap_ids,
    frame_retag,
)


def _make_base_ao() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": ("sample", [1.0, 2.0, 3.0])},
        coords={"sample": [0, 1, 2]},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", validate=True)


def _make_tagged_ao(*, parent: str = "world", child: str = "camera") -> AnalysisObject:
    return frame_retag(_make_base_ao(), parent=parent, child=child, validate=True)


def _with_invalid_roles(ao: AnalysisObject) -> AnalysisObject:
    broken_ds = ao.as_dataset(copy="none").copy(deep=True)
    broken_tal = dict(broken_ds.attrs["tal"])
    broken_core = dict(broken_tal.get("core", {}))
    broken_core["roles"] = {"sequence_dim": 123, "batch_dims": [], "core_dims": []}
    broken_tal["core"] = broken_core
    broken_ds.attrs["tal"] = broken_tal
    return AnalysisObject._from_unvalidated(broken_ds)


def test_frame_core_014_ao_frames_ids_and_retag_schema_bridge_parity() -> None:
    """ID: FRAME_CORE_014_ao_frames_ids_and_retag_schema_bridge_parity."""
    base = _make_base_ao()
    assert frame_ids(base) == (None, None)
    assert base.frames.ids() == (None, None)

    via_fn = frame_retag(base, parent="world", child="camera", validate=True)
    via_accessor = base.frames.retag(parent="world", child="camera", validate=True)
    assert frame_ids(via_fn) == ("world", "camera")
    assert frame_ids(via_accessor) == ("world", "camera")

    clear_parent = via_fn.frames.retag(parent=None, validate=True)
    clear_child = via_fn.frames.retag(child=None, validate=True)
    assert frame_ids(clear_parent) == (None, "camera")
    assert frame_ids(clear_child) == ("world", None)


def test_frame_core_015_ao_frames_remap_ids_deterministic_injective_behavior() -> None:
    """ID: FRAME_CORE_015_ao_frames_remap_ids_deterministic_injective_behavior."""
    ao = _make_tagged_ao(parent="world", child="camera")
    mapping = {"world": "map", "camera": "sensor", "unused": "ignored"}
    via_fn = frame_remap_ids(ao, mapping, validate=True)
    via_accessor = ao.frames.remap_ids(mapping, validate=True)
    assert frame_ids(via_fn) == ("map", "sensor")
    assert frame_ids(via_accessor) == ("map", "sensor")

    one_pass = frame_remap_ids(ao, {"world": "camera", "camera": "scope"}, validate=True)
    assert frame_ids(one_pass) == ("camera", "scope")


def test_frame_core_018_ao_frames_functional_accessor_parity() -> None:
    """ID: FRAME_CORE_018_ao_frames_functional_accessor_parity."""
    ao = _make_tagged_ao(parent="world", child="camera")
    assert ao.frames.ids() == frame_ids(ao)

    fn_retag = frame_retag(ao, parent="map", child="sensor", validate=True)
    acc_retag = ao.frames.retag(parent="map", child="sensor", validate=True)
    assert frame_ids(fn_retag) == frame_ids(acc_retag)

    fn_remap = frame_remap_ids(ao, {"world": "earth", "camera": "cam"}, validate=True)
    acc_remap = ao.frames.remap_ids({"world": "earth", "camera": "cam"}, validate=True)
    assert frame_ids(fn_remap) == frame_ids(acc_remap)


def test_frame_hard_021_ao_frames_remap_ids_rejects_invalid_mapping_shape() -> None:
    """ID: FRAME_HARD_021_ao_frames_remap_ids_rejects_invalid_mapping_shape."""
    ao = _make_tagged_ao()
    with pytest.raises(TypeError, match="mapping must be a mapping\\[str, str\\]"):
        frame_remap_ids(ao, [("a", "b")], validate=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping key must be a non-empty string frame id"):
        frame_remap_ids(ao, {1: "x"}, validate=True)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="mapping value must be a non-empty string frame id"):
        frame_remap_ids(ao, {"a": "   "}, validate=True)
    with pytest.raises(ValueError, match="mapping values must be injective"):
        frame_remap_ids(ao, {"a": "x", "b": "x"}, validate=True)


def test_frame_hard_023_ao_frames_accessor_type_boundary_no_raw_runtime_exceptions() -> None:
    """ID: FRAME_HARD_023_ao_frames_accessor_type_boundary_no_raw_runtime_exceptions."""
    ao = _make_tagged_ao()
    with pytest.raises(TypeError, match="expected AnalysisObject, xr.Dataset, or xr.DataArray"):
        frame_ids(object())
    with pytest.raises(TypeError, match="frames.resolve: graph must be FrameGraph or None"):
        ao.frames.resolve("bad")  # type: ignore[arg-type]


def test_frame_hard_026_ao_frames_validate_true_retag_and_remap_fail_closed_on_invalid_schema() -> None:
    """ID: FRAME_HARD_026_ao_frames_validate_true_retag_and_remap_fail_closed_on_invalid_schema."""
    invalid = _with_invalid_roles(_make_tagged_ao(parent="world", child="camera"))

    with pytest.raises(SchemaError, match="tal.core.roles.sequence_dim"):
        frame_retag(invalid, parent="map", child="sensor", validate=True)
    with pytest.raises(SchemaError, match="tal.core.roles.sequence_dim"):
        frame_remap_ids(invalid, {"world": "map", "camera": "sensor"}, validate=True)

    permissive_retag = frame_retag(invalid, parent="map", child="sensor", validate=False)
    permissive_remap = frame_remap_ids(invalid, {"world": "map", "camera": "sensor"}, validate=False)
    assert frame_ids(permissive_retag) == ("map", "sensor")
    assert frame_ids(permissive_remap) == ("map", "sensor")


def test_frame_core_129c_005_frames_resolve_read_only_selection() -> None:
    """ID: FRAME_CORE_129C_005_frames_resolve_read_only_selection."""
    associated = FrameGraph()
    world = associated.get_or_create_frame("world")
    camera = associated.get_or_create_frame("camera", parent=world)
    other = FrameGraph()
    other_world = other.get_or_create_frame("world")
    other_camera = other.get_or_create_frame("camera", parent=other_world)

    data = AnalysisObject.from_data(
        xr.DataArray(
            [1.0, 2.0, 3.0],
            dims="axis",
            coords={"axis": ["x", "y", "z"]},
            name="position",
        ),
        core_dims=("axis",),
    )
    spatial = Position(data, parent="world", child="camera", graph=associated)
    before = tuple((frame.id, frame.parent.id if frame.parent else None) for frame in (world, camera))
    assert spatial.frames.resolve() == (world, camera)
    assert spatial.frames.resolve(other) == (other_world, other_camera)
    after = tuple((frame.id, frame.parent.id if frame.parent else None) for frame in (world, camera))
    assert after == before


def test_frame_hard_129c_006_resolve_absent_and_missing_ids_without_mutation(monkeypatch) -> None:
    """ID: FRAME_HARD_129C_006_resolve_absent_and_missing_ids_without_mutation."""
    graph = FrameGraph()
    assert _make_base_ao().frames.resolve(graph) == (None, None)
    monkeypatch.setattr(
        "tal.utils.frame_ops.get_active_frame_graph",
        lambda: pytest.fail("absent frame IDs must not select a graph"),
    )
    assert _make_base_ao().frames.resolve() == (None, None)
    monkeypatch.undo()

    partial = _make_base_ao().frames.retag(parent="world")
    with pytest.raises(ValueError, match="frames.resolve: parent frame 'world' not found"):
        partial.frames.resolve(graph)
    assert graph.get_frame("world") is None
    world = graph.get_or_create_frame("world")
    assert partial.frames.resolve(graph) == (world, None)

    active = FrameGraph()
    active_world = active.get_or_create_frame("world")
    active_camera = active.get_or_create_frame("camera", parent=active_world)
    with active:
        assert _make_tagged_ao().frames.resolve() == (active_world, active_camera)


def test_frame_hard_129c_007_mutating_accessor_surfaces_are_removed() -> None:
    """ID: FRAME_HARD_129C_007_mutating_accessor_surfaces_are_removed."""
    from tal.utils import frame_ops

    accessor = _make_tagged_ao().frames
    assert not hasattr(accessor, "bind")
    assert not hasattr(accessor, "rename_frame")
    assert not hasattr(frame_ops, "frame_bind")
    assert not hasattr(frame_ops, "frame_rename")


@pytest.mark.parametrize("role", ["parent", "child"])
def test_frame_hard_129c_008_resolve_rejects_foreign_frame_results(role: str) -> None:
    """ID: FRAME_HARD_129C_008_resolve_rejects_foreign_frame_results."""

    class ForeignLookupGraph(FrameGraph):
        def __init__(self, foreign: Frame) -> None:
            super().__init__()
            self.foreign = foreign

        def get_frame(self, name: str) -> Frame | None:
            return self.foreign if name == self.foreign.id else None

    owner_graph = FrameGraph()
    foreign = owner_graph.get_or_create_frame("world")
    selected_graph = ForeignLookupGraph(foreign)
    tagged = _make_base_ao().frames.retag(**{role: "world"})

    with pytest.raises(ValueError, match="frames.resolve: frame belongs to a different FrameGraph"):
        tagged.frames.resolve(selected_graph)

    assert owner_graph.get_frame("world") is foreign
    assert foreign.parent is None
