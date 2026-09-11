from __future__ import annotations

import copy

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisObject
from tal.frames import Frame, FrameGraph
from tal.frames.registry import (
    get_edge_to_parent_runtime_ext,
    set_edge_to_parent_runtime_ext,
)
from tal.spatial import (
    PathSolveOptions,
    Pose,
    Position,
    Rotation,
    bind_pose,
    solve_pose_path_transform,
    solve_rotation_path_transform,
)
from tal.spatial.metadata import get_expressed_in, set_expressed_in
from tal.utils.frame_ops import frame_retag
from tal.utils.frame_schema import get_frames


def _pose(x=1., *, lazy=False, rep="components"):
    r = Rotation(AnalysisObject.from_data(
        xr.DataArray([[0., 0., 0., 1.]], dims=("sample", "q"),
                     coords={"sample": [0], "q": ["x", "y", "z", "w"]}, name="r"),
        sequence_dim="sample", core_dims=("q",)))
    p = Position(AnalysisObject.from_data(
        xr.DataArray([[x, 0., 0.]], dims=("sample", "axis"),
                     coords={"sample": [0], "axis": ["x", "y", "z"]}, name="p"),
        sequence_dim="sample", core_dims=("axis",)))
    value = Pose.from_components(r, p).to_rep(rep)
    return Pose(value.as_dataset().chunk({"sample": 1})) if lazy else value


def _topology_snapshot(graph, names):
    snapshot = []
    for name in names:
        frame = graph.get_frame(name)
        if frame is None:
            snapshot.append((name, None, ()))
            continue
        parent_id = None if frame.parent is None else frame.parent.id
        snapshot.append((name, parent_id, tuple(child.id for child in frame.children)))
    return tuple(snapshot)


def _with_basis(value, expressed_in: str):
    ds = set_expressed_in(
        value.as_dataset(copy="none"),
        expressed_in=expressed_in,
        validate=False,
        owner="test",
    )
    return value.__class__(ds)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("rep", ["components", "matrix"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["pose", "rotation"])
def test_bound_paths_match_explicit_resolvers(dynamic, rep, reverse, kind):
    """ID: BIND_127H_001_bound_paths_match_explicit_resolvers."""
    graph = FrameGraph()
    edge = _pose(rep=rep)
    calls = []

    def provider(child, parent):
        calls.append((child.id, parent.id))
        return edge

    bind_pose(graph, "world", "body", provider if dynamic else edge)
    bind_pose(graph, "body", "sensor", provider if dynamic else edge)
    assert calls == []
    src, dst = ("world", "sensor") if reverse else ("sensor", "world")
    solver = solve_pose_path_transform if kind == "pose" else solve_rotation_path_transform
    expected = solver(src, dst, opts=PathSolveOptions(graph=graph), **{
        f"edge_{kind}_fn": lambda *_: edge if kind == "pose" else edge.decompose()[1],
    })
    result = solver(src, dst, graph=graph)
    xr.testing.assert_allclose(result.as_dataset(), expected.as_dataset())
    assert get_frames(result.as_dataset()) == (dst, src)
    assert len(calls) == (2 if dynamic else 0)


@pytest.mark.parametrize(
    "failure",
    [
        "payload",
        "signature",
        "tags",
        "same",
        "cycle",
        "parent",
        "foreign",
        "unregistered",
        "frozen",
        "policy",
        "policy_type",
        "empty",
        "endpoint_type",
        "replace_parent",
        "graph_type",
    ],
)
def test_binding_preflight_is_atomic(failure):
    """ID: BIND_127H_002_preflight_failures_are_atomic."""
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    body = bind_pose(graph, world, "body", _pose())
    parent, child, provider, policy = "new_parent", "new_child", _pose(), "error"
    if failure == "payload": provider = object()
    if failure == "signature": provider = lambda: None
    if failure == "tags": provider = frame_retag(provider, parent="wrong", child="new_child")
    if failure == "same": parent = child
    if failure == "cycle": parent, child = body, world
    if failure == "parent": child = body
    if failure == "foreign": child = FrameGraph().get_or_create_frame("foreign")
    if failure == "unregistered": child = Frame(name="ghost", graph=graph)
    if failure == "frozen": graph.freeze()
    if failure == "policy": policy = "merge"
    if failure == "policy_type": policy = object()
    if failure == "empty": child = "  "
    if failure == "endpoint_type": child = object()
    if failure == "replace_parent": child, policy = body, "replace"
    names = ("world", "body", "new_parent", "new_child")
    before = _topology_snapshot(graph, names)
    before_pose = solve_pose_path_transform(body, world, graph=graph).as_dataset(copy="deep")
    with pytest.raises((ValueError, TypeError), match="spatial.bind_pose"):
        bind_pose(object() if failure == "graph_type" else graph, parent, child, provider, on_conflict=policy)
    assert _topology_snapshot(graph, names) == before
    after_pose = solve_pose_path_transform(body, world, graph=graph).as_dataset(copy="none")
    xr.testing.assert_identical(after_pose, before_pose)


def test_replacement_and_frozen_edge_writes():
    """ID: BIND_127H_003_replacement_preserves_other_extensions."""
    graph = FrameGraph()
    body = bind_pose(graph, "world", "body", _pose())
    set_edge_to_parent_runtime_ext(body, "other", "retained")
    names = ("world", "body")
    before = _topology_snapshot(graph, names)
    before_pose = solve_pose_path_transform(body, "world", graph=graph).as_dataset(copy="deep")
    with pytest.raises(ValueError, match="already has"):
        bind_pose(graph, "world", body, _pose(2.))
    assert _topology_snapshot(graph, names) == before
    xr.testing.assert_identical(
        solve_pose_path_transform(body, "world", graph=graph).as_dataset(copy="none"),
        before_pose,
    )
    bind_pose(graph, "world", body, _pose(2.), on_conflict="replace")
    assert get_edge_to_parent_runtime_ext(body, "other") == "retained"
    np.testing.assert_allclose(solve_pose_path_transform(body, "world", graph=graph).decompose()[0].as_dataset()["p"], [[2., 0., 0.]])
    graph.freeze()
    before = _topology_snapshot(graph, names)
    with pytest.raises(ValueError, match="frozen"):
        bind_pose(graph, "world", body, _pose(), on_conflict="replace")
    with pytest.raises(ValueError, match="frozen"):
        set_edge_to_parent_runtime_ext(body, "other", "changed")
    assert _topology_snapshot(graph, names) == before
    assert get_edge_to_parent_runtime_ext(body, "other") == "retained"


@pytest.mark.parametrize("operation", ["detach", "reparent", "remove", "subtree"])
def test_edge_lifecycle_clears_providers(operation):
    """ID: BIND_127H_004_lifecycle_clears_affected_edge_storage."""
    graph = FrameGraph()
    body = bind_pose(graph, "world", "body", _pose())
    sensor = bind_pose(graph, body, "sensor", _pose())
    world = body.parent
    if operation == "detach":
        body.reparent(None)
        body.reparent(world)
    elif operation == "reparent":
        body.reparent(graph.get_or_create_frame("other"), on_conflict="replace")
    elif operation == "remove":
        body.remove(subtree=False)
    else:
        body.remove(subtree=True)

    if operation == "subtree":
        assert graph.get_frame("body") is None
        assert graph.get_frame("sensor") is None
        return
    if operation == "remove":
        assert graph.get_frame("body") is None
        assert sensor.parent is None
        sensor.reparent(world)
        parent = world
    else:
        parent = body.parent
        solved = solve_pose_path_transform(sensor, body, graph=graph)
        assert get_frames(solved.as_dataset(copy="none")) == (body.id, sensor.id)
    with pytest.raises(ValueError, match="missing bound Pose provider"):
        solve_pose_path_transform(body if operation != "remove" else sensor, parent, graph=graph)


@pytest.mark.parametrize("dynamic", [False, True])
def test_rename_uses_current_edge_identifiers(dynamic):
    """ID: BIND_127H_005_static_and_dynamic_rename_semantics."""
    graph = FrameGraph()
    pose = frame_retag(_pose(), parent="world", child="body")
    provider = (lambda child, parent: frame_retag(_pose(), parent=parent.id, child=child.id)) if dynamic else pose
    body = bind_pose(graph, "world", "body", provider)
    body.rename("renamed")
    body.parent.rename("root")
    out = solve_pose_path_transform(body, body.parent)
    assert get_frames(out.as_dataset()) == ("root", "renamed")
    if dynamic:
        bind_pose(graph, body.parent, body, lambda *_: pose, on_conflict="replace")
        with pytest.raises(ValueError, match="framed edge payload"):
            solve_pose_path_transform(body, body.parent)


@pytest.mark.parametrize("dynamic", [False, True])
def test_edge_providers_use_canonical_parent_basis_across_renames(dynamic):
    """ID: BIND_127H_019_edge_providers_use_canonical_parent_basis."""
    graph = FrameGraph()

    def provider(child, parent):
        value = frame_retag(_pose(), parent=parent.id, child=child.id)
        return _with_basis(value, parent.id)

    world = graph.get_or_create_frame("world")
    body_value = _with_basis(frame_retag(_pose(), parent="world", child="body"), "world")
    body = bind_pose(graph, world, "body", provider if dynamic else body_value)
    sensor_value = _with_basis(frame_retag(_pose(), parent="body", child="sensor"), "body")
    sensor = bind_pose(graph, body, "sensor", provider if dynamic else sensor_value)
    world.rename("root")
    body.rename("base")

    for src, dst in ((sensor, world), (world, sensor)):
        out = solve_pose_path_transform(src, dst, graph=graph)
        assert get_frames(out.as_dataset(copy="none")) == (dst.id, src.id)
        assert get_expressed_in(out.as_dataset(copy="none"), owner="test") == dst.id
        rotation = solve_rotation_path_transform(src, dst, graph=graph)
        assert get_frames(rotation.as_dataset(copy="none")) == (dst.id, src.id)
        assert get_expressed_in(rotation.as_dataset(copy="none"), owner="test") == dst.id


@pytest.mark.parametrize("provider_kind", ["static", "dynamic", "pose", "rotation"])
def test_edge_providers_reject_third_frame_basis(provider_kind):
    """ID: BIND_127H_020_edge_providers_reject_third_frame_basis."""
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    body = graph.get_or_create_frame("body", parent=world)
    pose = _with_basis(frame_retag(_pose(), parent="world", child="body"), "map")

    if provider_kind == "static":
        before = _topology_snapshot(graph, ("world", "body"))
        with pytest.raises(ValueError, match="expressed_in must match parent='world'"):
            bind_pose(graph, world, body, pose)
        assert _topology_snapshot(graph, ("world", "body")) == before
        with pytest.raises(ValueError, match="missing bound Pose provider"):
            solve_pose_path_transform(body, world, graph=graph)
        return
    if provider_kind == "dynamic":
        bind_pose(graph, world, body, lambda *_: pose)
        resolver_kwargs = {}
    elif provider_kind == "pose":
        resolver_kwargs = {"edge_pose_fn": lambda *_: pose}
    else:
        rotation = _with_basis(
            frame_retag(pose.decompose()[1], parent="world", child="body"),
            "map",
        )
        resolver_kwargs = {"edge_rotation_fn": lambda *_: rotation}

    solver = solve_rotation_path_transform if provider_kind == "rotation" else solve_pose_path_transform
    with pytest.raises(ValueError, match="expressed_in must match parent='world'"):
        solver(body, world, graph=graph, **resolver_kwargs)


@pytest.mark.parametrize("dynamic", [False, True])
def test_provider_errors_do_not_mutate_graph(dynamic):
    """ID: BIND_127H_006_callback_errors_preserve_classification_and_state."""
    graph = FrameGraph()

    def provider(*args):
        raise TypeError("inside callback")

    class UninspectableProvider:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, child):
            return child

    callback = provider if dynamic else UninspectableProvider()
    bind_pose(graph, "world", "body", callback)
    before = _topology_snapshot(graph, ("world", "body"))
    with pytest.raises(ValueError if dynamic else TypeError, match="spatial.path_solve.pose"):
        solve_pose_path_transform("body", "world", graph=graph)
    assert _topology_snapshot(graph, ("world", "body")) == before


@pytest.mark.parametrize("provider_source", ["bound", "explicit"])
def test_user_provider_callbacks_envelope_nested_tal_signature_errors(provider_source):
    """ID: BIND_127H_030_user_callbacks_do_not_leak_internal_signature_markers."""

    class UninspectableResolver:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, child):
            return child

    graph = FrameGraph()

    def nested_failure(child, parent):
        return solve_pose_path_transform(
            child,
            parent,
            graph=graph,
            edge_pose_fn=UninspectableResolver(),
        )

    if provider_source == "bound":
        body = bind_pose(graph, "world", "body", nested_failure)
        kwargs = {}
        expected = "provider failed for edge"
    else:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        kwargs = {"edge_pose_fn": nested_failure}
        expected = "edge_pose_fn failed for edge"
    with pytest.raises(ValueError, match=expected) as exc_info:
        solve_pose_path_transform(body, body.parent, graph=graph, **kwargs)
    assert type(exc_info.value) is ValueError
    assert type(exc_info.value.__cause__) is TypeError


@pytest.mark.parametrize(
    ("solver", "resolver_arg"),
    [
        (solve_pose_path_transform, "edge_pose_fn"),
        (solve_rotation_path_transform, "edge_rotation_fn"),
    ],
)
def test_uninspectable_c_callback_typeerror_is_conservative_invocation_misuse(
    solver,
    resolver_arg,
):
    """ID: BIND_127H_031_uninspectable_c_boundary_typeerror_is_public_typeerror."""
    graph = FrameGraph()
    world = graph.get_or_create_frame("world")
    body = graph.get_or_create_frame("body", parent=world)

    with pytest.raises(TypeError, match=rf"{resolver_arg} must be callable") as exc_info:
        solver(body, world, graph=graph, **{resolver_arg: np.ndarray})

    assert type(exc_info.value) is TypeError
    assert type(exc_info.value.__cause__) is TypeError


@pytest.mark.parametrize("lazy", [False, True])
def test_static_binding_preserves_payload_ownership(lazy):
    """ID: BIND_127H_007_metadata_isolation_and_lazy_payload_ownership."""
    graph = FrameGraph()
    pose = frame_retag(_pose(lazy=lazy), parent="world", child="body")
    source = pose.as_dataset(copy="none")
    attrs = copy.deepcopy(source.attrs)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        bind_pose(graph, "world", "body", pose)
        result = solve_pose_path_transform("body", "world", graph=graph)
    assert source.attrs == attrs
    assert tasks == []
    for name, var in source.data_vars.items():
        actual = result.as_dataset(copy="none")[name]
        if lazy:
            assert actual.chunks is not None
        else:
            assert np.shares_memory(var.data, actual.data)


def _nested_metadata_slots(ds, *, var_name, coord_name):
    return (
        ds.attrs["nested"],
        ds.encoding["nested"],
        ds[var_name].attrs["nested"],
        ds[var_name].encoding["nested"],
        ds.coords[coord_name].attrs["nested"],
        ds.coords[coord_name].encoding["nested"],
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_static_binding_isolates_nested_metadata(lazy):
    """ID: BIND_127H_018_static_binding_isolates_nested_metadata."""
    graph = FrameGraph()
    pose = frame_retag(_pose(lazy=lazy), parent="world", child="body")
    source = pose.as_dataset(copy="none")
    var_name = next(iter(source.data_vars))
    coord_name = "sample"
    source.attrs["nested"] = {"items": []}
    source.encoding["nested"] = {"items": []}
    source[var_name].attrs["nested"] = {"items": []}
    source[var_name].encoding["nested"] = {"items": []}
    source.coords[coord_name].attrs["nested"] = {"items": []}
    source.coords[coord_name].encoding["nested"] = {"items": []}

    bind_pose(graph, "world", "body", pose)
    source_slots = _nested_metadata_slots(source, var_name=var_name, coord_name=coord_name)
    for value in source_slots:
        value["items"].append("source")

    resolved = solve_pose_path_transform("body", "world", graph=graph).as_dataset(copy="none")
    resolved_slots = _nested_metadata_slots(resolved, var_name=var_name, coord_name=coord_name)
    assert all(left is not right for left, right in zip(source_slots, resolved_slots, strict=True))
    assert all(value == {"items": []} for value in resolved_slots)
    for value in resolved_slots:
        value["items"].append("result")

    repeated = solve_pose_path_transform("body", "world", graph=graph).as_dataset(copy="none")
    repeated_slots = _nested_metadata_slots(repeated, var_name=var_name, coord_name=coord_name)
    assert all(left is not right for left, right in zip(resolved_slots, repeated_slots, strict=True))
    assert all(value == {"items": []} for value in repeated_slots)
    assert all(value == {"items": ["source"]} for value in source_slots)


@pytest.mark.parametrize("solver", [solve_pose_path_transform, solve_rotation_path_transform, Pose.solve_path_transform, Rotation.solve_path_transform])
@pytest.mark.parametrize("identity", [False, True])
def test_graph_selection_and_resolver_precedence(solver, identity):
    """ID: BIND_127H_008_graph_selection_and_resolver_precedence."""
    from tests._path_options_helpers import FalseyPathSolveOptions

    graph = FrameGraph()
    pose = _pose()
    calls = []
    body = bind_pose(graph, "world", "body", lambda *_: calls.append("bound") or pose)
    world = body.parent
    src = world if identity else body
    other = FrameGraph()
    with other:
        expected = solver(src, world)
        actual = solver(src.id, world.id, graph=graph, opts=FalseyPathSolveOptions())
        xr.testing.assert_allclose(actual.as_dataset(), expected.as_dataset())
    with graph:
        xr.testing.assert_allclose(solver(src.id, world.id).as_dataset(), expected.as_dataset())
    calls.clear()
    kind = "rotation" if solver in (solve_rotation_path_transform, Rotation.solve_path_transform) else "pose"
    explicit = pose.decompose()[1] if kind == "rotation" else pose
    solver(src.id, world.id, graph=graph, **{f"edge_{kind}_fn": lambda *_: explicit})
    assert calls == []
    for options in [PathSolveOptions(graph=graph), FalseyPathSolveOptions(graph=other)]:
        with pytest.raises(ValueError, match="graph and opts.graph"):
            solver(src, world, graph=graph, opts=options)
    with pytest.raises(TypeError, match="opts must be PathSolveOptions"):
        solver(src, world, graph=object(), opts={})
    if identity:
        result = solver(
            src,
            world,
            graph=graph,
            opts=PathSolveOptions(strict=False),
            **{f"edge_{kind}_fn": object()},
        )
        assert result.graph is graph
    else:
        with pytest.raises(ValueError, match="strict=False"):
            solver(src, world, graph=graph, opts=PathSolveOptions(strict=False))


@pytest.mark.parametrize("kind", ["pose", "rotation"])
def test_missing_provider_and_invalid_dynamic_result(kind):
    """ID: BIND_127H_009_missing_edge_and_dynamic_payload_diagnostics."""
    graph = FrameGraph()
    body = bind_pose(graph, "world", "body", _pose())
    sensor = graph.get_or_create_frame("sensor", parent=body)
    solver = solve_pose_path_transform if kind == "pose" else solve_rotation_path_transform
    with pytest.raises(ValueError, match="missing bound Pose provider.*child='sensor', parent='body'"):
        solver(sensor, "world", graph=graph)
    bind_pose(graph, body, sensor, lambda *_: object())
    with pytest.raises(ValueError, match="Pose-coercible"):
        solver(sensor, "world", graph=graph)


@pytest.mark.parametrize("kind,operation", [("position", "to_frame"), ("position", "express_in"), ("pose", "express_in"), ("rotation", "express_in")])
@pytest.mark.parametrize("identity", [False, True])
def test_spatial_delegators_select_bound_graph(kind, operation, identity):
    """ID: BIND_127H_010_spatial_delegators_and_identity_conflicts."""
    graph = FrameGraph()
    edge = _pose()
    bind_pose(graph, "world", "body", edge)
    position, rotation = edge.decompose()
    value = {"position": position, "rotation": rotation, "pose": edge}[kind]
    value = frame_retag(value, parent="body", child="probe")
    dst = "body" if identity else "world"
    resolver_kind = "pose" if kind == "pose" or operation == "to_frame" else "rotation"
    method = getattr(value, operation)
    expected = method(dst, graph=graph, **{f"edge_{resolver_kind}_fn": lambda *_: edge if resolver_kind == "pose" else rotation})
    result = method(dst, graph=graph, opts=PathSolveOptions())
    xr.testing.assert_allclose(result.as_dataset(), expected.as_dataset())
    with pytest.raises(ValueError, match="graph and opts.graph"):
        method(dst, graph=graph, opts=PathSolveOptions(graph=graph))


def test_bound_paths_preserve_strict_labeled_alignment():
    """ID: BIND_127H_011_strict_path_alignment_is_preserved."""
    graph = FrameGraph()
    bind_pose(graph, "world", "body", _pose())
    mismatched = Pose(_pose().as_dataset().assign_coords(sample=[1]))
    bind_pose(graph, "body", "sensor", mismatched)
    with pytest.raises(ValueError, match="spatial.path_solve.pose"):
        solve_pose_path_transform("sensor", "world", graph=graph)


@pytest.mark.parametrize("tags", [("world", None), (None, "body")])
@pytest.mark.parametrize("provider_kind", ["static", "dynamic", "explicit_pose", "explicit_rotation"])
def test_present_matching_provider_frame_tags_are_independently_accepted(tags, provider_kind):
    """ID: BIND_127H_023_present_provider_frame_tags_match_independently."""
    parent, child = tags
    graph = FrameGraph()
    pose = frame_retag(_pose(), parent=parent, child=child, validate=True)
    if provider_kind in {"static", "dynamic"}:
        provider = pose if provider_kind == "static" else lambda *_: pose
        bound = bind_pose(graph, "world", "body", provider)
        result = solve_pose_path_transform(bound, bound.parent, graph=graph)
    else:
        world = graph.get_or_create_frame("world")
        body = graph.get_or_create_frame("body", parent=world)
        if provider_kind == "explicit_pose":
            result = solve_pose_path_transform(body, world, graph=graph, edge_pose_fn=lambda *_: pose)
        else:
            rotation = frame_retag(pose.decompose()[1], parent=parent, child=child, validate=True)
            result = solve_rotation_path_transform(body, world, graph=graph, edge_rotation_fn=lambda *_: rotation)
    assert get_frames(result.as_dataset(copy="none")) == ("world", "body")


@pytest.mark.parametrize("tags", [("wrong", None), (None, "wrong")])
def test_present_mismatching_provider_frame_tag_fails(tags):
    """ID: BIND_127H_024_present_provider_frame_tag_mismatch_fails."""
    pose = frame_retag(_pose(), parent=tags[0], child=tags[1], validate=True)
    with pytest.raises(ValueError, match="spatial.bind_pose: framed edge payload must match each present tag"):
        bind_pose(FrameGraph(), "world", "body", pose)


@pytest.mark.parametrize("rep", ["components", "matrix"])
@pytest.mark.parametrize("lazy", [False, True])
def test_binding_retains_owned_representation(rep, lazy):
    """ID: BIND_127H_016_binding_preserves_representation_and_payload_sharing."""
    value = _pose(rep=rep, lazy=lazy)
    original = value.as_dataset(copy="none")
    before = value.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        child = bind_pose(FrameGraph(), "world", "body", value)
        xr.testing.assert_identical(value.as_dataset(), before)
        if not lazy:
            if rep == "components":
                original["p"].data[0, 0] = 3.0
            else:
                original["pose_matrix"].data[0, 0, 3] = 3.0
        result = solve_pose_path_transform(child, child.parent)
    assert tasks == []
    expected = 1.0 if lazy else 3.0
    position, _ = result.decompose()
    np.testing.assert_allclose(position.to_dataarray().compute(), [[expected, 0.0, 0.0]])


@pytest.mark.parametrize("dynamic", [False, True])
def test_bound_lazy_provider_resource_remains_caller_owned(dynamic):
    """ID: BIND_127H_021_bound_lazy_provider_resource_remains_caller_owned."""
    graph = FrameGraph()
    pose = _pose(lazy=True)
    replacement = _pose(lazy=True)
    closed = []
    pose.as_dataset(copy="none").set_close(lambda: closed.append("source"))
    replacement.as_dataset(copy="none").set_close(lambda: closed.append("replacement"))
    provider = (lambda *_: pose) if dynamic else pose
    child = bind_pose(graph, "world", "body", provider)
    result = solve_pose_path_transform(child, child.parent)
    position, _ = result.decompose()
    np.testing.assert_allclose(position.to_dataarray().compute(), [[1.0, 0.0, 0.0]])
    result.close()
    next_provider = (lambda *_: replacement) if dynamic else replacement
    bind_pose(graph, child.parent, child, next_provider, on_conflict="replace")
    child.remove(subtree=True)
    assert closed == []
    pose.close()
    pose.close()
    replacement.close()
    replacement.close()
    assert closed == ["source", "replacement"]


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("rep", ["components", "matrix"])
def test_bound_provider_paths_defer_numeric_execution(dynamic, rep):
    """ID: BIND_127H_017_bound_path_payload_execution_stays_deferred."""
    graph = FrameGraph()
    value = _pose(rep=rep, lazy=True)
    calls, tasks = [], []

    def provider(child, parent):
        calls.append((child.id, parent.id))
        return value

    with Callback(pretask=lambda *args: tasks.append(args[0])):
        bind_pose(graph, "world", "body", provider if dynamic else value)
        bind_pose(graph, "body", "sensor", provider if dynamic else value)
        assert calls == []
        result = solve_pose_path_transform("sensor", "world", graph=graph)
        assert tasks == []
        position, rotation = result.decompose()
        assert position.to_dataarray().chunks is not None
        assert rotation.to_dataarray().chunks is not None
        np.testing.assert_allclose(position.to_dataarray().compute(), [[2., 0., 0.]])
    assert len(calls) == (2 if dynamic else 0)
    assert tasks
