from __future__ import annotations

import pytest

from tal.frames import Frame, FrameGraph, find_path, fold_path


def test_frame_core_005_find_path_identity_src_eq_dst() -> None:
    """ID: FRAME_CORE_005_find_path_identity_src_eq_dst."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        path = find_path(world, world)
    assert path.nodes == (world,)
    assert path.lca is world
    assert path.lca_index == 0
    assert path.steps == ()


def test_frame_core_006_find_path_lca_nodes_and_steps_deterministic() -> None:
    """ID: FRAME_CORE_006_find_path_lca_nodes_and_steps_deterministic."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot = g.get_or_create_frame("robot", parent=world)
        camera = g.get_or_create_frame("camera", parent=robot)
        base = g.get_or_create_frame("base", parent=world)
        lidar = g.get_or_create_frame("lidar", parent=base)
        path = find_path(camera, lidar)

    assert path.nodes == (camera, robot, world, base, lidar)
    assert path.lca is world
    assert path.lca_index == 2
    assert len(path.steps) == 4
    assert path.steps[0].child is camera and path.steps[0].parent is robot and not path.steps[0].invert
    assert path.steps[1].child is robot and path.steps[1].parent is world and not path.steps[1].invert
    assert path.steps[2].child is base and path.steps[2].parent is world and path.steps[2].invert
    assert path.steps[3].child is lidar and path.steps[3].parent is base and path.steps[3].invert


def test_frame_core_007_fold_path_oriented_composition_deterministic() -> None:
    """ID: FRAME_CORE_007_fold_path_oriented_composition_deterministic."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot = g.get_or_create_frame("robot", parent=world)
        camera = g.get_or_create_frame("camera", parent=robot)
        base = g.get_or_create_frame("base", parent=world)
        lidar = g.get_or_create_frame("lidar", parent=base)
        path = find_path(camera, lidar)

    edge = {
        ("camera", "robot"): "cr",
        ("robot", "world"): "rw",
        ("base", "world"): "bw",
        ("lidar", "base"): "lb",
    }

    out = fold_path(
        path,
        edge_value_fn=lambda child, parent: edge[(child.id, parent.id)],
        compose=lambda acc, value: acc + [value],
        inverse=lambda value: f"inv({value})",
        identity=lambda: [],
    )
    assert out == ["cr", "rw", "inv(bw)", "inv(lb)"]


def test_frame_hard_008_find_path_cross_component_fail_closed() -> None:
    """ID: FRAME_HARD_008_find_path_cross_component_fail_closed."""
    g = FrameGraph()
    with g:
        root_a = g.get_or_create_frame("root_a")
        leaf_a = g.get_or_create_frame("leaf_a", parent=root_a)
        root_b = g.get_or_create_frame("root_b")
        leaf_b = g.get_or_create_frame("leaf_b", parent=root_b)
        with pytest.raises(ValueError, match="no path"):
            find_path(leaf_a, leaf_b)


def test_frame_hard_009_find_path_requires_registered_same_graph_frames() -> None:
    """ID: FRAME_HARD_009_find_path_requires_registered_same_graph_frames."""
    g1 = FrameGraph()
    g2 = FrameGraph()
    with g1:
        world = g1.get_or_create_frame("world")
    with g2:
        robot = g2.get_or_create_frame("robot")
    with pytest.raises(ValueError, match="different FrameGraph"):
        find_path(world, robot)

    g3 = FrameGraph()
    with g3:
        active = g3.get_or_create_frame("active")
        ghost = Frame(name="ghost", graph=g3)
        with pytest.raises(ValueError, match="not registered"):
            find_path(active, ghost)


def test_frame_hard_015_find_path_malformed_frame_graph_binding_fails_closed() -> None:
    """ID: FRAME_HARD_015_find_path_malformed_frame_graph_binding_fails_closed."""
    g = FrameGraph()
    with g:
        valid = g.get_or_create_frame("valid")
        bad_dst = Frame(name="bad_dst", graph="not_graph")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="find_path: dst frame is not bound to a valid FrameGraph"):
            find_path(valid, bad_dst)
        bad_src = Frame(name="bad_src", graph="not_graph")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="find_path: src frame is not bound to a valid FrameGraph"):
            find_path(bad_src, valid)
