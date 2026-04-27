from __future__ import annotations

import pytest

from tal.frames import FrameGraph
from tal.spatial import (
    get_edge_motion_class,
    get_frame_inertial_status,
    propagate_inertial_status,
    set_edge_motion_class,
    set_frame_inertial_status,
)


def _build_graph() -> tuple[FrameGraph, object, object]:
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
    return graph, world, body


def test_frame_core_c6_001_edge_motion_class_roundtrip_and_validation() -> None:
    """ID: FRAME_CORE_C6_001_edge_motion_class_roundtrip_and_validation."""
    _, world, body = _build_graph()
    assert get_edge_motion_class(body, world) == "unknown"
    set_edge_motion_class(body, "static", world)
    assert get_edge_motion_class(body, world) == "static"
    set_edge_motion_class(body, "galilean", world)
    assert get_edge_motion_class(body, world) == "galilean"
    set_edge_motion_class(body, "dynamic", world)
    assert get_edge_motion_class(body, world) == "dynamic"


def test_frame_core_c6_002_frame_inertial_status_roundtrip_and_validation() -> None:
    """ID: FRAME_CORE_C6_002_frame_inertial_status_roundtrip_and_validation."""
    _, world, body = _build_graph()
    assert get_frame_inertial_status(world) == "unknown"
    set_frame_inertial_status(world, "inertial")
    set_frame_inertial_status(body, "non_inertial")
    assert get_frame_inertial_status(world) == "inertial"
    assert get_frame_inertial_status(body) == "non_inertial"


def test_frame_core_c6_003_inertial_status_propagation_rules_are_deterministic() -> None:
    """ID: FRAME_CORE_C6_003_inertial_status_propagation_rules_are_deterministic."""
    _, world, body = _build_graph()
    set_frame_inertial_status(world, "inertial")
    set_edge_motion_class(body, "static", world)
    assert propagate_inertial_status(body, world) == "inertial"
    set_edge_motion_class(body, "galilean", world)
    assert propagate_inertial_status(body, world) == "inertial"
    set_edge_motion_class(body, "dynamic", world)
    assert propagate_inertial_status(body, world) == "non_inertial"
    set_edge_motion_class(body, "unknown", world)
    assert propagate_inertial_status(body, world) == "unknown"


def test_frame_hard_c6_001_unknown_or_invalid_edge_motion_class_fails_closed_for_required_consumers() -> None:
    """ID: FRAME_HARD_C6_001_unknown_or_invalid_edge_motion_class_fails_closed_for_required_consumers."""
    _, world, body = _build_graph()
    with pytest.raises(ValueError, match="edge motion class"):
        set_edge_motion_class(body, "bad", world)
    set_frame_inertial_status(world, "inertial")
    set_edge_motion_class(body, "unknown", world)
    assert propagate_inertial_status(body, world) == "unknown"


def test_frame_hard_c6_002_scalar_edge_motion_laws_do_not_imply_trajectory_or_static_geometry() -> None:
    """ID: FRAME_HARD_C6_002_scalar_edge_motion_laws_do_not_imply_trajectory_or_static_geometry."""
    _, world, body = _build_graph()
    set_edge_motion_class(body, "galilean", world)
    assert get_edge_motion_class(body, world) == "galilean"
    set_frame_inertial_status(world, "inertial")
    assert propagate_inertial_status(body, world) == "inertial"
    assert get_edge_motion_class(body, world) == "galilean"


def test_frame_hard_c6_003_no_hidden_rigid_relation_inference_from_unrelated_payloads() -> None:
    """ID: FRAME_HARD_C6_003_no_hidden_rigid_relation_inference_from_unrelated_payloads."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=world)
    set_edge_motion_class(body, "static", world)
    assert get_edge_motion_class(body, world) == "static"
    assert get_edge_motion_class(sensor, world) == "unknown"


def test_frame_core_c6_004_non_subtree_remove_orphan_clears_edge_runtime_extension_state() -> None:
    """ID: FRAME_CORE_C6_004_non_subtree_remove_orphan_clears_edge_runtime_extension_state."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        sensor = graph.get_or_create_frame("sensor", parent=body)
        set_edge_motion_class(sensor, "dynamic", body)
        assert get_edge_motion_class(sensor, body) == "dynamic"
        graph.remove_frame(body, subtree=False)
        graph.reparent_frame(sensor, world, on_conflict="replace")
    assert get_edge_motion_class(sensor, world) == "unknown"
