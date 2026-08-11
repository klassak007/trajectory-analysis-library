from __future__ import annotations

import asyncio

import pytest

from tal.frames import Frame, FrameGraph, get_active_frame_graph


def test_frame_core_003_framegraph_context_scope_returns_active_graph() -> None:
    """ID: FRAME_CORE_003_framegraph_context_scope_returns_active_graph."""
    default_graph = get_active_frame_graph()
    g1 = FrameGraph()
    g2 = FrameGraph()

    assert get_active_frame_graph() is default_graph
    with g1:
        assert get_active_frame_graph() is g1
        with g2:
            assert get_active_frame_graph() is g2
        assert get_active_frame_graph() is g1
    assert get_active_frame_graph() is default_graph


@pytest.mark.parametrize("shared_graph", (True, False), ids=("shared", "distinct"))
def test_frame_hard_033_framegraph_async_context_tokens_are_task_local(
    shared_graph: bool,
) -> None:
    """ID: FRAME_HARD_033_framegraph_async_context_tokens_are_task_local."""
    default_graph = get_active_frame_graph()

    async def run_interleaved() -> None:
        first_graph = FrameGraph()
        second_graph = first_graph if shared_graph else FrameGraph()
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        first_left = asyncio.Event()

        async def first() -> None:
            previous = get_active_frame_graph()
            try:
                with first_graph:
                    assert get_active_frame_graph() is first_graph
                    first_entered.set()
                    await second_entered.wait()
            finally:
                first_left.set()
            assert get_active_frame_graph() is previous

        async def second() -> None:
            await first_entered.wait()
            previous = get_active_frame_graph()
            with second_graph:
                assert get_active_frame_graph() is second_graph
                second_entered.set()
                await first_left.wait()
            assert get_active_frame_graph() is previous

        await asyncio.gather(first(), second())

    asyncio.run(run_interleaved())
    assert get_active_frame_graph() is default_graph


def test_frame_core_023_framegraph_nested_context_restores_previous_graph() -> None:
    """ID: FRAME_CORE_023_framegraph_nested_context_restores_previous_graph."""
    default_graph = get_active_frame_graph()
    outer = FrameGraph()
    inner = FrameGraph()

    with outer:
        assert get_active_frame_graph() is outer
        with outer:
            assert get_active_frame_graph() is outer
        assert get_active_frame_graph() is outer
        with inner:
            assert get_active_frame_graph() is inner
        assert get_active_frame_graph() is outer
    assert get_active_frame_graph() is default_graph


def test_frame_hard_034_framegraph_context_exception_restores_previous_graph() -> None:
    """ID: FRAME_HARD_034_framegraph_context_exception_restores_previous_graph."""
    default_graph = get_active_frame_graph()
    outer = FrameGraph()
    inner = FrameGraph()

    with pytest.raises(RuntimeError, match="boom"):
        with outer:
            with inner:
                raise RuntimeError("boom")
    assert get_active_frame_graph() is default_graph

    error = r"FrameGraph\.__exit__: no matching context-local entry"
    with pytest.raises(RuntimeError, match=error):
        inner.__exit__(None, None, None)
    assert get_active_frame_graph() is default_graph

    async def reject_inherited_entry_exit() -> None:
        async def child() -> None:
            with pytest.raises(RuntimeError, match=error):
                outer.__exit__(None, None, None)
            assert get_active_frame_graph() is outer

        with outer:
            await asyncio.create_task(child())
            assert get_active_frame_graph() is outer

    asyncio.run(reject_inherited_entry_exit())
    assert get_active_frame_graph() is default_graph


def test_frame_core_004_framegraph_get_or_create_parenting_deterministic() -> None:
    """ID: FRAME_CORE_004_framegraph_get_or_create_parenting_deterministic."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot_1 = g.get_or_create_frame("robot", parent=world)
        robot_2 = g.get_or_create_frame("robot", parent=world)
        assert robot_1 is robot_2
        assert robot_1.parent is world

        sensor = world.child("sensor")
        assert sensor.parent is world
        assert g.get_frame("sensor") is sensor


def test_frame_hard_002_framegraph_cross_graph_parenting_refused() -> None:
    """ID: FRAME_HARD_002_framegraph_cross_graph_parenting_refused."""
    g1 = FrameGraph()
    g2 = FrameGraph()
    with g1:
        world = g1.get_or_create_frame("world")
    with g2:
        with pytest.raises(ValueError, match="different FrameGraph"):
            g2.get_or_create_frame("robot", parent=world)
        robot = g2.get_or_create_frame("robot")
        with pytest.raises(ValueError, match="different FrameGraph"):
            robot.reparent(world)


def test_frame_hard_003_framegraph_freeze_refuses_mutation() -> None:
    """ID: FRAME_HARD_003_framegraph_freeze_refuses_mutation."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot = g.get_or_create_frame("robot", parent=world)
        g.freeze()

        with pytest.raises(ValueError, match="frozen"):
            g.get_or_create_frame("camera", parent=world)
        with pytest.raises(ValueError, match="frozen"):
            robot.reparent(None)
        with pytest.raises(ValueError, match="frozen"):
            g.rename_frame("robot", "base_link")
        with pytest.raises(ValueError, match="frozen"):
            g.remove_frame("robot")


def test_frame_hard_006_rename_replace_refuses_ancestry_overlap_non_destructive() -> None:
    """ID: FRAME_HARD_006_rename_replace_refuses_ancestry_overlap_non_destructive."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot = g.get_or_create_frame("robot", parent=world)
        sensor = g.get_or_create_frame("sensor", parent=robot)

        with pytest.raises(ValueError, match="ancestor/descendant"):
            g.rename_frame(robot, "world", on_conflict="replace")
        assert g.get_frame("world") is world
        assert g.get_frame("robot") is robot
        assert g.get_frame("sensor") is sensor
        assert robot.parent is world
        assert sensor.parent is robot

        with pytest.raises(ValueError, match="ancestor/descendant"):
            g.rename_frame(world, "robot", on_conflict="replace")
        assert g.get_frame("world") is world
        assert g.get_frame("robot") is robot
        assert g.get_frame("sensor") is sensor
        assert robot.parent is world
        assert sensor.parent is robot


def test_frame_hard_007_public_framegraph_type_mismatch_fails_deterministically() -> None:
    """ID: FRAME_HARD_007_public_framegraph_type_mismatch_fails_deterministically."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        with pytest.raises(TypeError, match="must be Frame"):
            g.get_or_create_frame("robot", parent=123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be Frame"):
            g.rename_frame(123, "robot")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be Frame"):
            g.remove_frame(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be Frame"):
            world.reparent(123)  # type: ignore[arg-type]


def test_frame_core_008_reparent_conflict_policy_error_vs_replace_disjoint() -> None:
    """ID: FRAME_CORE_008_reparent_conflict_policy_error_vs_replace_disjoint."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        arm = g.get_or_create_frame("arm", parent=world)
        base = g.get_or_create_frame("base", parent=world)
        camera = g.get_or_create_frame("camera", parent=arm)

        with pytest.raises(ValueError, match="already has parent"):
            g.reparent_frame(camera, base, on_conflict="error")
        assert camera.parent is arm

        g.reparent_frame(camera, base, on_conflict="replace")
        assert camera.parent is base


def test_frame_core_009_remove_frame_subtree_modes_deterministic() -> None:
    """ID: FRAME_CORE_009_remove_frame_subtree_modes_deterministic."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot = g.get_or_create_frame("robot", parent=world)
        sensor = g.get_or_create_frame("sensor", parent=robot)
        g.remove_frame(robot, subtree=False)
        assert g.get_frame("robot") is None
        assert g.get_frame("sensor") is sensor
        assert sensor.parent is None

    g2 = FrameGraph()
    with g2:
        world2 = g2.get_or_create_frame("world")
        robot2 = g2.get_or_create_frame("robot", parent=world2)
        sensor2 = g2.get_or_create_frame("sensor", parent=robot2)
        g2.remove_frame(robot2, subtree=True)
        assert g2.get_frame("robot") is None
        assert g2.get_frame("sensor") is None
        assert sensor2.parent is None


def test_frame_hard_010_reparent_cycle_refused_with_conflict_policy() -> None:
    """ID: FRAME_HARD_010_reparent_cycle_refused_with_conflict_policy."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot = g.get_or_create_frame("robot", parent=world)
        sensor = g.get_or_create_frame("sensor", parent=robot)
        with pytest.raises(ValueError, match="cycle"):
            g.reparent_frame(world, sensor, on_conflict="replace")


def test_frame_hard_011_reparent_replace_refuses_ancestry_overlap_non_destructive() -> None:
    """ID: FRAME_HARD_011_reparent_replace_refuses_ancestry_overlap_non_destructive."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot = g.get_or_create_frame("robot", parent=world)
        sensor = g.get_or_create_frame("sensor", parent=robot)
        with pytest.raises(ValueError, match="cycle"):
            g.reparent_frame(robot, sensor, on_conflict="replace")
        assert robot.parent is world
        assert sensor.parent is robot
        assert g.get_frame("world") is world
        assert g.get_frame("robot") is robot
        assert g.get_frame("sensor") is sensor


def test_frame_hard_012_topology_public_type_validation_no_attributeerror_leak() -> None:
    """ID: FRAME_HARD_012_topology_public_type_validation_no_attributeerror_leak."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        with pytest.raises(TypeError, match="must be Frame"):
            g.reparent_frame(123, None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be Frame"):
            g.reparent_frame(world, 123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be Frame"):
            world.reparent(123, on_conflict="replace")  # type: ignore[arg-type]


def test_frame_hard_013_mutators_reject_unregistered_same_graph_frame_objects() -> None:
    """ID: FRAME_HARD_013_mutators_reject_unregistered_same_graph_frame_objects."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        robot = g.get_or_create_frame("robot", parent=world)
        ghost = Frame(name="ghost", graph=g)

        with pytest.raises(ValueError, match="not registered"):
            g.reparent_frame(ghost, world)
        with pytest.raises(ValueError, match="not registered"):
            g.rename_frame(ghost, "ghost2")
        with pytest.raises(ValueError, match="not registered"):
            g.remove_frame(ghost)

        assert world.children == (robot,)
        assert robot.parent is world
        assert g.get_frame("ghost") is None
        assert g.get_frame("ghost2") is None


def test_frame_hard_014_get_or_create_rejects_unregistered_parent_frame_object() -> None:
    """ID: FRAME_HARD_014_get_or_create_rejects_unregistered_parent_frame_object."""
    g = FrameGraph()
    with g:
        world = g.get_or_create_frame("world")
        ghost_parent = Frame(name="ghost_parent", graph=g)

        with pytest.raises(ValueError, match="not registered"):
            g.get_or_create_frame("camera", parent=ghost_parent)

        assert g.get_frame("camera") is None
        assert g.get_frame("world") is world
        assert world.parent is None
