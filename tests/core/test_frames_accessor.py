from __future__ import annotations

import pytest
import xarray as xr

from tal import AnalysisObject
from tal.core.schema_errors import SchemaError
from tal.frames import FrameGraph
from tal.utils.frame_ops import (
    frame_bind,
    frame_ids,
    frame_remap_ids,
    frame_rename,
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


def test_frame_core_016_ao_frames_bind_resolve_create_missing_and_conflict_policy() -> None:
    """ID: FRAME_CORE_016_ao_frames_bind_resolve_create_missing_and_conflict_policy."""
    ao = _make_tagged_ao(parent="world", child="camera")
    graph = FrameGraph()
    parent, child = frame_bind(ao, graph=graph, create_missing=True, on_conflict="error")
    assert parent is not None and child is not None
    assert parent.id == "world"
    assert child.id == "camera"
    assert child.parent is parent

    graph.get_or_create_frame("other")
    conflict_ao = frame_retag(ao, parent="other", child="camera", validate=True)
    before_parent = graph.get_frame("camera").parent
    with pytest.raises(ValueError, match="frames.bind:"):
        frame_bind(conflict_ao, graph=graph, create_missing=True, on_conflict="error")
    assert graph.get_frame("camera").parent is before_parent

    replace_parent, replace_child = frame_bind(
        conflict_ao,
        graph=graph,
        create_missing=True,
        on_conflict="replace",
    )
    assert replace_parent is not None and replace_child is not None
    assert replace_parent.id == "other"
    assert replace_child.parent is replace_parent


def test_frame_core_017_ao_frames_rename_frame_syncs_graph_and_metadata() -> None:
    """ID: FRAME_CORE_017_ao_frames_rename_frame_syncs_graph_and_metadata."""
    ao = _make_tagged_ao(parent="world", child="camera")
    graph = FrameGraph()
    frame_bind(ao, graph=graph, create_missing=True, on_conflict="error")

    out = frame_rename(ao, "camera", "cam0", graph=graph, on_conflict="error", validate=True)
    assert graph.get_frame("camera") is None
    assert graph.get_frame("cam0") is not None
    assert frame_ids(out) == ("world", "cam0")


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

    graph_a = FrameGraph()
    graph_b = FrameGraph()
    fn_parent, fn_child = frame_bind(ao, graph=graph_a, create_missing=True, on_conflict="error")
    acc_parent, acc_child = ao.frames.bind(graph=graph_b, create_missing=True, on_conflict="error")
    assert (fn_parent.id if fn_parent else None, fn_child.id if fn_child else None) == (
        acc_parent.id if acc_parent else None,
        acc_child.id if acc_child else None,
    )


def test_frame_hard_020_ao_frames_bind_missing_runtime_frame_fail_closed_when_create_missing_false() -> None:
    """ID: FRAME_HARD_020_ao_frames_bind_missing_runtime_frame_fail_closed_when_create_missing_false."""
    ao = _make_tagged_ao(parent="world", child="camera")
    graph = FrameGraph()
    with pytest.raises(ValueError, match="frames.bind: parent frame 'world' not found in graph"):
        frame_bind(ao, graph=graph, create_missing=False, on_conflict="error")
    assert graph.get_frame("world") is None
    assert graph.get_frame("camera") is None


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


def test_frame_hard_022_ao_frames_rename_frame_propagates_graph_conflict_policy_fail_closed() -> None:
    """ID: FRAME_HARD_022_ao_frames_rename_frame_propagates_graph_conflict_policy_fail_closed."""
    ao = _make_tagged_ao(parent="world", child="camera")
    graph = FrameGraph()
    frame_bind(ao, graph=graph, create_missing=True, on_conflict="error")
    graph.get_or_create_frame("cam0")

    before_ids = frame_ids(ao)
    before_graph = tuple(sorted(graph._frames))
    with pytest.raises(ValueError, match="frames.rename_frame:"):
        frame_rename(ao, "camera", "cam0", graph=graph, on_conflict="error", validate=True)
    assert frame_ids(ao) == before_ids
    assert tuple(sorted(graph._frames)) == before_graph
    assert graph.get_frame("camera") is not None


def test_frame_hard_023_ao_frames_accessor_type_boundary_no_raw_runtime_exceptions() -> None:
    """ID: FRAME_HARD_023_ao_frames_accessor_type_boundary_no_raw_runtime_exceptions."""
    ao = _make_tagged_ao()
    with pytest.raises(TypeError, match="expected AnalysisObject, xr.Dataset, or xr.DataArray"):
        frame_ids(object())
    with pytest.raises(TypeError, match="graph must be FrameGraph or None"):
        frame_bind(ao, graph="bad", create_missing=True, on_conflict="error")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="old must be a non-empty string frame id"):
        frame_rename(ao, old=123, new="next", graph=FrameGraph(), on_conflict="error", validate=True)  # type: ignore[arg-type]


def test_frame_hard_024_ao_frames_rename_frame_atomic_no_graph_mutation_on_metadata_failure() -> None:
    """ID: FRAME_HARD_024_ao_frames_rename_frame_atomic_no_graph_mutation_on_metadata_failure."""
    ao = _make_tagged_ao(parent="world", child="camera")
    graph = FrameGraph()
    frame_bind(ao, graph=graph, create_missing=True, on_conflict="error")

    broken_ds = ao.as_dataset(copy="none").copy(deep=True)
    broken_tal = dict(broken_ds.attrs["tal"])
    broken_ext = dict(broken_tal.get("ext", {}))
    broken_frames = dict(broken_ext.get("frames", {}))
    broken_frames["child"] = 123
    broken_ext["frames"] = broken_frames
    broken_tal["ext"] = broken_ext
    broken_ds.attrs["tal"] = broken_tal
    broken_ao = AnalysisObject._from_unvalidated(broken_ds)

    with pytest.raises(SchemaError, match="tal.ext.frames.child"):
        frame_rename(broken_ao, "camera", "cam0", graph=graph, on_conflict="error", validate=True)

    camera = graph.get_frame("camera")
    world = graph.get_frame("world")
    assert camera is not None
    assert world is not None
    assert graph.get_frame("cam0") is None
    assert camera.parent is world


def test_frame_hard_025_ao_frames_bind_rejects_non_frame_registry_entries() -> None:
    """ID: FRAME_HARD_025_ao_frames_bind_rejects_non_frame_registry_entries."""
    ao = _make_tagged_ao(parent="world", child="camera")
    graph = FrameGraph()
    marker = object()
    graph._frames["world"] = marker  # type: ignore[assignment]

    with pytest.raises(
        ValueError,
        match="frames.bind: parent frame 'world' is not a registered Frame object in graph",
    ):
        frame_bind(ao, graph=graph, create_missing=True, on_conflict="error")

    assert graph._frames["world"] is marker
    assert graph.get_frame("camera") is None


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


def test_frame_hard_027_ao_frames_rename_validate_true_schema_failure_prevents_graph_mutation() -> None:
    """ID: FRAME_HARD_027_ao_frames_rename_validate_true_schema_failure_prevents_graph_mutation."""
    base = _make_tagged_ao(parent="world", child="camera")
    invalid = _with_invalid_roles(base)
    graph = FrameGraph()
    frame_bind(base, graph=graph, create_missing=True, on_conflict="error")

    with pytest.raises(SchemaError, match="tal.core.roles.sequence_dim"):
        frame_rename(invalid, "camera", "cam0", graph=graph, on_conflict="error", validate=True)

    camera = graph.get_frame("camera")
    world = graph.get_frame("world")
    assert camera is not None
    assert world is not None
    assert graph.get_frame("cam0") is None
    assert camera.parent is world
