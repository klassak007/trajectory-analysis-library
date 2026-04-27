from __future__ import annotations

import importlib

import pytest

from tal.frames import Frame, FrameGraph, render_snapshot_ascii, snapshot_from_seeds, snapshot_to_networkx


def _build_snapshot_graph() -> tuple[FrameGraph, Frame, Frame, Frame, Frame, Frame]:
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        arm = graph.get_or_create_frame("arm", parent=world)
        camera = graph.get_or_create_frame("camera", parent=arm)
        alpha = graph.get_or_create_frame("alpha")
        zeta = graph.get_or_create_frame("zeta", parent=alpha)
    return graph, world, arm, camera, alpha, zeta


def test_frame_core_010_snapshot_from_seeds_deterministic_ordering() -> None:
    """ID: FRAME_CORE_010_snapshot_from_seeds_deterministic_ordering."""
    graph, _, _, camera, _, _ = _build_snapshot_graph()
    with graph:
        snap_a = snapshot_from_seeds(("camera",), graph=graph, include_unreachable=True)
        snap_b = snapshot_from_seeds(("camera",), graph=graph, include_unreachable=True)
        reachable_only = snapshot_from_seeds(("camera",), graph=graph, include_unreachable=False)

    assert snap_a == snap_b
    assert snap_a.seed_ids == ("camera",)
    assert snap_a.node_ids == tuple(sorted(snap_a.node_ids))
    assert snap_a.root_ids == tuple(sorted(snap_a.root_ids))
    assert any(issue.code == "unreachable_registered_frame" for issue in snap_a.issues)
    assert set(reachable_only.node_ids) == {"arm", "camera", "world"}
    assert not any(issue.code == "unreachable_registered_frame" for issue in reachable_only.issues)


def test_frame_core_011_snapshot_issue_detection_for_stale_topology_non_mutating() -> None:
    """ID: FRAME_CORE_011_snapshot_issue_detection_for_stale_topology_non_mutating."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        robot = graph.get_or_create_frame("robot", parent=world)
        base = graph.get_or_create_frame("base", parent=world)

        ghost = Frame(name="ghost", graph=graph)
        world._children["ghost"] = ghost
        base._children["robot"] = robot
        world._parent = robot
        robot._parent = world

        before = (
            world.parent,
            tuple(sorted(world._children.keys())),
            robot.parent,
            tuple(sorted(robot._children.keys())),
            base.parent,
            tuple(sorted(base._children.keys())),
        )
        snapshot = snapshot_from_seeds(graph=graph)
        after = (
            world.parent,
            tuple(sorted(world._children.keys())),
            robot.parent,
            tuple(sorted(robot._children.keys())),
            base.parent,
            tuple(sorted(base._children.keys())),
        )

    assert before == after
    codes = {issue.code for issue in snapshot.issues}
    assert "unknown_child_ref" in codes
    assert "parent_child_mismatch" in codes
    assert "duplicate_parent_claim" in codes
    assert "cycle_detected" in codes


def test_frame_core_012_render_snapshot_ascii_deterministic_stable() -> None:
    """ID: FRAME_CORE_012_render_snapshot_ascii_deterministic_stable."""
    graph, _, _, camera, _, _ = _build_snapshot_graph()
    with graph:
        snapshot = snapshot_from_seeds(("camera",), graph=graph, include_unreachable=False)
    text_a = render_snapshot_ascii(snapshot)
    text_b = render_snapshot_ascii(snapshot)
    assert text_a == text_b
    assert "FrameSnapshot" in text_a
    assert "Nodes:" in text_a
    assert "Edges:" in text_a
    assert "Issues:" in text_a


def test_frame_core_013_snapshot_to_networkx_nodes_edges_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: FRAME_CORE_013_snapshot_to_networkx_nodes_edges_metadata."""
    graph, _, _, camera, _, _ = _build_snapshot_graph()
    with graph:
        snapshot = snapshot_from_seeds(("camera",), graph=graph, include_unreachable=False)

    class _FakeDiGraph:
        def __init__(self) -> None:
            self.nodes: set[str] = set()
            self.edges: set[tuple[str, str]] = set()
            self.graph: dict[str, object] = {}

        def add_node(self, node_id: str) -> None:
            self.nodes.add(node_id)

        def add_edge(self, src: str, dst: str) -> None:
            self.edges.add((src, dst))

    class _FakeNx:
        DiGraph = _FakeDiGraph

    original_import = importlib.import_module

    def _fake_import(name: str):
        if name == "networkx":
            return _FakeNx
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    nx_graph = snapshot_to_networkx(snapshot)
    assert nx_graph.nodes == set(snapshot.node_ids)
    assert nx_graph.edges == {(parent, child) for child, parent in snapshot.parent_edges}
    assert nx_graph.graph["seed_ids"] == snapshot.seed_ids
    assert isinstance(nx_graph.graph["issues"], tuple)


def test_frame_hard_016_snapshot_seed_validation_fail_closed() -> None:
    """ID: FRAME_HARD_016_snapshot_seed_validation_fail_closed."""
    g1 = FrameGraph()
    g2 = FrameGraph()
    with g1:
        world = g1.get_or_create_frame("world")
    with g2:
        other = g2.get_or_create_frame("other")

    with pytest.raises(TypeError, match="graph must be FrameGraph"):
        snapshot_from_seeds(graph="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seeds must be a tuple"):
        snapshot_from_seeds(["world"], graph=g1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed frame 'missing' not found"):
        snapshot_from_seeds(("missing",), graph=g1)
    with pytest.raises(ValueError, match="different FrameGraph"):
        snapshot_from_seeds((other,), graph=g1)

    malformed = Frame(name="malformed", graph="not_graph")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not bound to a valid FrameGraph"):
        snapshot_from_seeds((malformed,), graph=g1)

    unregistered = Frame(name="unregistered", graph=g1)
    with pytest.raises(ValueError, match="not registered"):
        snapshot_from_seeds((unregistered,), graph=g1)

    with g1:
        snap = snapshot_from_seeds((world,), graph=g1)
    assert "world" in snap.node_ids


def test_frame_hard_017_snapshot_to_networkx_missing_dependency_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: FRAME_HARD_017_snapshot_to_networkx_missing_dependency_fail_closed."""
    graph, _, _, camera, _, _ = _build_snapshot_graph()
    with graph:
        snapshot = snapshot_from_seeds(("camera",), graph=graph, include_unreachable=False)

    original_import = importlib.import_module

    def _fake_import(name: str):
        if name == "networkx":
            raise ImportError("missing networkx")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(ImportError, match="snapshot_to_networkx: networkx is required"):
        snapshot_to_networkx(snapshot)


def test_frame_hard_018_snapshot_string_seed_requires_registered_frame_object() -> None:
    """ID: FRAME_HARD_018_snapshot_string_seed_requires_registered_frame_object."""
    graph = FrameGraph()
    with graph:
        graph.get_or_create_frame("world")
    graph._frames["ghost"] = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="seed frame 'ghost' is not a registered Frame object"):
        snapshot_from_seeds(("ghost",), graph=graph)


def test_frame_hard_019_snapshot_none_seeds_mixed_registry_keys_fail_closed_deterministically() -> None:
    """ID: FRAME_HARD_019_snapshot_none_seeds_mixed_registry_keys_fail_closed_deterministically."""
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
    graph._frames[1] = world  # type: ignore[index]
    with pytest.raises(ValueError, match="snapshot_from_seeds: frame registry contains invalid key"):
        snapshot_from_seeds(graph=graph)
