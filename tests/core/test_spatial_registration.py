from __future__ import annotations

import gc
import weakref
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisObject
from tal.frames import FrameGraph
from tal.spatial import Pose, Position, Rotation, bind_pose, solve_pose_path_transform
from tal.spatial.metadata import (
    get_edge_motion_class,
    get_frame_inertial_status,
    set_expressed_in,
)


def _pose(
    *,
    topology: str = "static",
    rep: str = "components",
    lazy: bool = False,
    graph: FrameGraph | None = None,
    x: float = 1.0,
    samples: int = 2,
) -> Pose:
    sequence_dim = None if topology == "static" else "sample"
    param_coord = "time" if topology == "dynamic" else None
    sample_coords = {} if sequence_dim is None else {"sample": list(range(samples))}
    if param_coord is not None:
        sample_coords[param_coord] = ("sample", np.arange(samples, dtype=float))
    quat = [0.0, 0.0, 0.0, 1.0] if sequence_dim is None else [[0.0, 0.0, 0.0, 1.0]] * samples
    xyz = [x, 0.0, 0.0] if sequence_dim is None else [[x, 0.0, 0.0]] * samples
    rotation_dims = ("quat",) if sequence_dim is None else ("sample", "quat")
    position_dims = ("axis",) if sequence_dim is None else ("sample", "axis")
    rotation = Rotation(AnalysisObject.from_data(
        xr.DataArray(
            quat,
            dims=rotation_dims,
            coords={**sample_coords, "quat": ["x", "y", "z", "w"]},
            name="rotation",
        ),
        sequence_dim=sequence_dim,
        core_dims=("quat",),
        param_coord=param_coord,
    ))
    position = Position(AnalysisObject.from_data(
        xr.DataArray(
            xyz,
            dims=position_dims,
            coords={**sample_coords, "axis": ["x", "y", "z"]},
            name="position",
        ),
        sequence_dim=sequence_dim,
        core_dims=("axis",),
        param_coord=param_coord,
    ))
    pose = Pose.from_components(
        rotation,
        position,
        parent="world",
        child="body",
        graph=graph,
    ).to_rep(rep)
    if not lazy:
        return pose
    chunks = {"sample": 1} if sequence_dim is not None else {}
    return Pose(pose.as_dataset(copy="none").chunk(chunks), graph=graph)


def _topology(graph: FrameGraph, names: tuple[str, ...]) -> tuple[tuple[str, str | None] | None, ...]:
    result = []
    for name in names:
        frame = graph.get_frame(name)
        result.append(None if frame is None else (frame.id, frame.parent.id if frame.parent else None))
    return tuple(result)


def _with_basis(pose: Pose, basis: str) -> Pose:
    ds = set_expressed_in(
        pose.as_dataset(copy="none"),
        expressed_in=basis,
        validate=False,
        owner="test",
    )
    return Pose._from_unvalidated(ds).with_graph(pose.graph)


def _task_recorder(tasks: list[object]) -> Callable[..., None]:
    def record(key: object, *_: object) -> None:
        tasks.append(key)

    return record


@pytest.mark.parametrize(
    ("topology", "samples"),
    [("static", 1), ("dynamic", 1), ("dynamic", 2)],
)
@pytest.mark.parametrize("rep", ["components", "matrix"])
@pytest.mark.parametrize("lazy", [False, True])
def test_spatial_core_129c_001_pose_register_static_dynamic_identity(
    topology, samples, rep, lazy
) -> None:
    """ID: SPATIAL_CORE_129C_001_pose_register_static_dynamic_identity."""
    graph = FrameGraph()
    pose = _pose(topology=topology, rep=rep, lazy=lazy, graph=graph, samples=samples)
    before = pose.as_dataset(copy="deep")
    tasks: list[object] = []
    with Callback(pretask=_task_recorder(tasks)):
        registered = pose.register()
    assert registered is pose
    assert tasks == []
    xr.testing.assert_identical(pose.as_dataset(copy="none"), before)
    parent, child = pose.frames.resolve()
    assert parent is graph.get_frame("world")
    assert child is graph.get_frame("body")
    assert child is not None and child.parent is parent
    assert get_edge_motion_class(child, parent) == "unknown"
    assert get_frame_inertial_status(parent) == "unknown"


def test_spatial_hard_129c_002_registration_preflight_is_atomic_and_lazy() -> None:
    """ID: SPATIAL_HARD_129C_002_registration_preflight_is_atomic_and_lazy."""
    for samples in (1, 2):
        graph = FrameGraph()
        exact = _pose(topology="exact", lazy=True, graph=graph, samples=samples)
        before = exact.as_dataset(copy="deep")
        tasks: list[object] = []
        with (
            Callback(pretask=_task_recorder(tasks)),
            pytest.raises(ValueError, match="requires a declared param_coord"),
        ):
            exact.register()
        assert tasks == []
        assert _topology(graph, ("world", "body")) == (None, None)
        xr.testing.assert_identical(exact.as_dataset(copy="none"), before)

    invalid_policy_cases = (
        (_pose(graph=None), object(), TypeError),
        (_pose(topology="exact", lazy=True, graph=FrameGraph()), object(), TypeError),
        (_pose(graph=None), "merge", ValueError),
        (_pose(topology="exact", lazy=True, graph=FrameGraph()), "merge", ValueError),
    )
    for value, policy, error in invalid_policy_cases:
        tasks = []
        with (
            Callback(pretask=_task_recorder(tasks)),
            pytest.raises(error, match="on_conflict"),
        ):
            value.register(on_conflict=policy)
        assert tasks == []

    cases = (
        (_pose(graph=None), {}, ValueError, "associated with a FrameGraph"),
        (Pose(_pose(graph=graph), parent=None), {}, ValueError, "requires a nonempty parent"),
        (Pose(_pose(graph=graph), child=None), {}, ValueError, "requires a nonempty child"),
        (_pose(graph=graph).frames.retag(child="world"), {}, ValueError, "must be distinct"),
        (_with_basis(_pose(graph=graph), "map"), {}, ValueError, "expressed_in must match"),
        (_pose(graph=graph), {"on_conflict": object()}, TypeError, "on_conflict"),
        (_pose(graph=graph), {"on_conflict": "merge"}, ValueError, "on_conflict"),
    )
    for value, kwargs, error, message in cases:
        current = value.graph
        snapshot = () if current is None else _topology(current, ("world", "body"))
        with pytest.raises(error, match=message):
            value.register(**kwargs)
        if current is not None:
            assert _topology(current, ("world", "body")) == snapshot


def test_spatial_hard_129c_003_registration_conflict_replace_and_reparent_policy() -> None:
    """ID: SPATIAL_HARD_129C_003_registration_conflict_replace_and_reparent_policy."""
    graph = FrameGraph()
    first = _pose(graph=graph, x=1.0)
    first.register()
    before = solve_pose_path_transform("body", "world", graph=graph).as_dataset(copy="deep")
    replacement = _pose(graph=graph, x=2.0)
    with pytest.raises(ValueError, match="already has a bound Pose provider"):
        replacement.register()
    xr.testing.assert_identical(
        solve_pose_path_transform("body", "world", graph=graph).as_dataset(copy="none"),
        before,
    )
    assert replacement.register(on_conflict="replace") is replacement
    position, _ = solve_pose_path_transform("body", "world", graph=graph).decompose()
    np.testing.assert_allclose(position.to_dataarray(), [2.0, 0.0, 0.0])

    other = FrameGraph()
    root = other.get_or_create_frame("other")
    other.get_or_create_frame("body", parent=root)
    value = _pose(graph=other)
    snapshot = _topology(other, ("world", "other", "body"))
    with pytest.raises(ValueError, match="refusing reparent"):
        value.register(on_conflict="replace")
    assert _topology(other, ("world", "other", "body")) == snapshot

    frozen = FrameGraph()
    frozen.freeze()
    with pytest.raises(ValueError, match="frozen"):
        _pose(graph=frozen).register()
    assert _topology(frozen, ("world", "body")) == (None, None)


def test_spatial_core_129c_004_registered_provider_is_nonowning_and_isolated() -> None:
    """ID: SPATIAL_CORE_129C_004_registered_provider_is_nonowning_and_isolated."""
    retained_graph = FrameGraph()
    retained = _pose(graph=retained_graph)
    retained_ds = retained.as_dataset(copy="none")
    retained_ref = weakref.ref(retained)
    retained.register()
    retained_ds["position"].data[0] = 3.0
    position, _ = solve_pose_path_transform(
        "body", "world", graph=retained_graph
    ).decompose()
    np.testing.assert_allclose(position.to_dataarray(), [3.0, 0.0, 0.0])
    del retained
    gc.collect()
    assert retained_ref() is None

    graph = FrameGraph()
    pose = _pose(graph=graph, lazy=True)
    closed: list[str] = []
    pose.as_dataset(copy="none").set_close(lambda: closed.append("source"))
    source_ref = weakref.ref(pose)
    source_ds = pose.as_dataset(copy="none")
    pose.register()
    source_ds.attrs["caller"] = {"changed": True}
    result = solve_pose_path_transform("body", "world", graph=graph)
    assert "caller" not in result.as_dataset(copy="none").attrs
    result.close()
    graph.get_frame("body").remove(subtree=True)
    assert closed == []
    pose.close()
    pose.close()
    assert closed == ["source"]

    del result
    del source_ds
    del pose
    gc.collect()
    assert source_ref() is None


def test_spatial_core_129c_005_bind_pose_retains_callable_and_exact_boundary() -> None:
    """ID: SPATIAL_CORE_129C_005_bind_pose_retains_callable_and_exact_boundary."""
    graph = FrameGraph()
    exact = _pose(topology="exact")
    calls: list[tuple[str, str]] = []

    def provider(child, parent):
        calls.append((child.id, parent.id))
        return exact

    child = bind_pose(graph, "world", "body", provider)
    assert calls == []
    solved = solve_pose_path_transform(child, child.parent, graph=graph)
    assert calls == [("body", "world")]
    xr.testing.assert_allclose(solved.as_dataset(), exact.as_dataset())
