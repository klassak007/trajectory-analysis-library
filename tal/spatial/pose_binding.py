from __future__ import annotations

from dataclasses import dataclass

from tal.frames import Frame, FrameGraph
from tal.frames.registry import (
    _require_frame_name,
    get_edge_to_parent_runtime_ext,
    set_edge_to_parent_runtime_ext,
)

from .ops.pose_provider_ops import (
    POSE_PROVIDER_KEY,
    BoundPoseProvider,
    prepare_pose_provider,
)


@dataclass(frozen=True)
class _BindingPlan:
    graph: FrameGraph
    parent_id: str
    child_id: str
    provider: BoundPoseProvider


def _endpoint(value: Frame | str, *, graph: FrameGraph, arg: str, owner: str) -> tuple[str, Frame | None]:
    if isinstance(value, Frame):
        graph._assert_owned(value, context=owner)
        return value.id, value
    if not isinstance(value, str):
        raise TypeError(f"{owner}: {arg} must be Frame or non-empty string frame id.")
    name = _require_frame_name(value, owner=owner, arg=arg)
    return name, graph.get_frame(name)


def _require_bindable_edge(
    graph: FrameGraph,
    parent: Frame | None,
    child: Frame | None,
    *,
    parent_id: str,
    child_id: str,
    on_conflict: str,
    owner: str,
) -> None:
    graph._require_attachable(
        child,
        parent,
        frame_id=child_id,
        parent_id=parent_id,
        allow_reparent=False,
        on_conflict="error",
        context=owner,
    )
    if child is None or child.parent is None:
        return
    if on_conflict == "error" and get_edge_to_parent_runtime_ext(child, POSE_PROVIDER_KEY, owner=owner) is not None:
        raise ValueError(f"{owner}: edge already has a bound Pose provider; use on_conflict='replace'.")


def _plan_binding(graph, parent, child, provider, *, on_conflict: str, owner: str) -> _BindingPlan:
    if not isinstance(graph, FrameGraph):
        raise TypeError(f"{owner}: graph must be FrameGraph.")
    graph._assert_mutable(context=owner)
    if on_conflict not in ("error", "replace"):
        raise ValueError(f"{owner}: on_conflict must be 'error' or 'replace'.")
    parent_id, parent_frame = _endpoint(parent, graph=graph, arg="parent", owner=owner)
    child_id, child_frame = _endpoint(child, graph=graph, arg="child", owner=owner)
    _require_bindable_edge(
        graph,
        parent_frame,
        child_frame,
        parent_id=parent_id,
        child_id=child_id,
        on_conflict=on_conflict,
        owner=owner,
    )
    normalized = prepare_pose_provider(provider, child_id=child_id, parent_id=parent_id, owner=owner)
    return _BindingPlan(graph, parent_id, child_id, normalized)


def bind_pose(graph: FrameGraph, parent: Frame | str, child: Frame | str, provider: object, *, on_conflict: str = "error") -> Frame:
    """Bind a Pose value or callable provider to a parent-to-child graph edge.

    Parameters
    ----------
    graph : FrameGraph
        Mutable graph owning the edge.
    parent, child : Frame or str
        Registered frames or frame IDs. Missing string IDs are created after validation.
    provider : object
        Pose-coercible child-to-parent value, or ``provider(child, parent)`` returning one.
    on_conflict : str, optional
        Defaults to ``'error'``. Use ``'replace'`` to replace an existing provider.
        Does not permit reparenting.

    Returns
    -------
    Frame
        Bound child frame.

    Raises
    ------
    TypeError
        If graph, endpoint types, or an inspectable callback signature are invalid.
    ValueError
        If graph mutation, topology, provider layout, or frame tags conflict.

    Notes
    -----
    Supported validation failures leave topology and providers unchanged. Static
    Dataset, variable, and coordinate attrs and encodings are isolated without
    copying owned payloads or converting representation. Later path results do
    not alias the stored metadata. Callables are never run during binding;
    resolution checks their results against current frame IDs.
    For uninspectable callbacks, a boundary-only ``TypeError`` follows the path
    solver's conservative invocation-misuse policy.
    Static bindings survive renames; callbacks returning stale frame tags fail.
    Each present ``parent`` or ``child`` tag must match the corresponding edge;
    the graph supplies omitted tags. Provider values must be represented in the
    edge parent: an explicit ``expressed_in`` value must equal ``parent``.
    Re-express third-frame values before binding. Lazy provider payloads are
    non-owning aliases; keep their owning AO open until all dependent path
    results have been computed. Graph
    replacement and removal do not close provider resources.
    Binding does not infer motion or inertial support. Pose alignment remains
    by dimension names and coordinate labels through existing spatial operations.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.frames import FrameGraph
    >>> from tal.spatial import Pose, Position, Rotation, bind_pose, solve_pose_path_transform
    >>> rotation = Rotation(AnalysisObject.from_data(
    ...     xr.DataArray([0., 0., 0., 1.], dims='q', coords={'q': ['x', 'y', 'z', 'w']}, name='r'),
    ...     core_dims=('q',)))
    >>> position = Position(AnalysisObject.from_data(
    ...     xr.DataArray([1., 0., 0.], dims='axis', coords={'axis': ['x', 'y', 'z']}, name='p'),
    ...     core_dims=('axis',)))
    >>> graph = FrameGraph()
    >>> child = bind_pose(graph, 'world', 'body', Pose.from_components(rotation, position))
    >>> child.parent.id
    'world'
    >>> result = solve_pose_path_transform('body', 'world', graph=graph)
    >>> isinstance(result, Pose)
    True
    """
    owner = "spatial.bind_pose"
    plan = _plan_binding(graph, parent, child, provider, on_conflict=on_conflict, owner=owner)
    parent_frame = plan.graph.get_or_create_frame(plan.parent_id)
    child_frame = plan.graph.get_or_create_frame(plan.child_id, parent=parent_frame)
    set_edge_to_parent_runtime_ext(child_frame, POSE_PROVIDER_KEY, plan.provider, owner=owner)
    return child_frame
