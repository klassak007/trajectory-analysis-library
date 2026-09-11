from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.schema_read import read_param_coord_name, read_roles
from tal.core.schema_validate import validate_schema_structure
from tal.frames import Frame, FrameGraph
from tal.frames.registry import (
    _require_frame_name,
    get_edge_to_parent_runtime_ext,
    set_edge_to_parent_runtime_ext,
)
from tal.utils.frame_schema import get_frames

from .association import associated_graph
from .ops.pose_provider_ops import (
    POSE_PROVIDER_KEY,
    BoundPoseProvider,
    prepare_pose_provider,
)

if TYPE_CHECKING:
    from .pose import Pose

_ProviderTopology = Literal["static", "dynamic", "exact"]
_ConflictPolicy = Literal["error", "replace"]


@dataclass(frozen=True)
class _BindingPlan:
    graph: FrameGraph
    parent_id: str
    child_id: str
    provider: BoundPoseProvider


def _require_conflict_policy(value: object, *, owner: str) -> _ConflictPolicy:
    if not isinstance(value, str):
        raise TypeError(f"{owner}: on_conflict must be 'error' or 'replace'.")
    if value == "error":
        return "error"
    if value == "replace":
        return "replace"
    raise ValueError(f"{owner}: on_conflict must be 'error' or 'replace'.")


def _provider_topology(pose: Pose) -> _ProviderTopology:
    ds = analysis_object_dataset(pose)
    validate_schema_structure(ds)
    _, sequence_dim, _, _ = read_roles(ds)
    param_coord = read_param_coord_name(ds)
    if sequence_dim is None:
        return "static"
    return "dynamic" if param_coord is not None else "exact"


def _registration_endpoint(value: str | None, *, role: str, owner: str) -> str:
    if value is None:
        raise ValueError(f"{owner}: Pose requires a nonempty {role} frame declaration.")
    return _require_frame_name(value, owner=owner, arg=role)


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


def _plan_binding(
    graph,
    parent,
    child,
    provider,
    *,
    on_conflict: _ConflictPolicy,
    owner: str,
) -> _BindingPlan:
    if not isinstance(graph, FrameGraph):
        raise TypeError(f"{owner}: graph must be FrameGraph.")
    graph._assert_mutable(context=owner)
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


def _commit_binding(plan: _BindingPlan, *, owner: str) -> Frame:
    parent_frame = plan.graph.get_or_create_frame(plan.parent_id)
    child_frame = plan.graph.get_or_create_frame(plan.child_id, parent=parent_frame)
    set_edge_to_parent_runtime_ext(child_frame, POSE_PROVIDER_KEY, plan.provider, owner=owner)
    return child_frame


def _register_pose(pose: Pose, *, on_conflict: str = "error") -> Pose:
    """Register one associated Pose through the shared edge-binding owner."""
    owner = "spatial.pose.register"
    policy = _require_conflict_policy(on_conflict, owner=owner)
    graph = associated_graph(pose)
    if graph is None:
        raise ValueError(f"{owner}: Pose must be associated with a FrameGraph before registration.")
    parent, child = get_frames(analysis_object_dataset(pose))
    parent_id = _registration_endpoint(parent, role="parent", owner=owner)
    child_id = _registration_endpoint(child, role="child", owner=owner)
    if parent_id == child_id:
        raise ValueError(f"{owner}: parent and child frame declarations must be distinct.")
    if _provider_topology(pose) == "exact":
        raise ValueError(
            f"{owner}: sequence-backed Pose registration requires a declared param_coord; "
            "use bind_pose(...) for an exact unparameterized provider."
        )
    plan = _plan_binding(
        graph,
        parent_id,
        child_id,
        pose,
        on_conflict=policy,
        owner=owner,
    )
    _commit_binding(plan, owner=owner)
    return pose


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
        If graph, endpoint, conflict-policy types, or an inspectable callback
        signature are invalid.
    ValueError
        If the conflict-policy value, graph mutation, topology, provider
        layout, or frame tags are invalid.

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
    policy = _require_conflict_policy(on_conflict, owner=owner)
    plan = _plan_binding(graph, parent, child, provider, on_conflict=policy, owner=owner)
    return _commit_binding(plan, owner=owner)
