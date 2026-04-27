from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tal.frames import (
    Frame,
    FrameGraph,
    FramePath,
    find_path,
    get_active_frame_graph,
)
from tal.utils.frame_schema import get_frames

from ..metadata import (
    EDGE_MOTION_CLASS_VALUES,
    FRAME_INERTIAL_STATUS_VALUES,
    get_edge_motion_class,
    get_frame_inertial_status,
    get_instantaneous_inertial,
)
from ..metadata.roles import get_kinematics_kind
from ..path_solve import KinematicsPathSupportOptions, PathSolveOptions


@dataclass(frozen=True)
class KinematicsPathSupportContext:
    graph: FrameGraph
    src_parent: Frame
    src_child: Frame | None
    dst: Frame
    path: FramePath
    operation: str
    edge_classes: tuple[str, ...]
    support: KinematicsPathSupportOptions


def _require_callable(value: object, *, owner: str, name: str):
    if callable(value):
        return value
    raise TypeError(f"{owner}: {name} must be callable.")


def _require_status(value: object, *, owner: str, what: str) -> str:
    if isinstance(value, str) and value.strip() in FRAME_INERTIAL_STATUS_VALUES:
        return value.strip()
    raise ValueError(f"{owner}: {what} must be one of {FRAME_INERTIAL_STATUS_VALUES!r}.")


def _require_motion_class(value: object, *, owner: str, what: str) -> str:
    if isinstance(value, str) and value.strip() in EDGE_MOTION_CLASS_VALUES:
        return value.strip()
    raise ValueError(f"{owner}: {what} must be one of {EDGE_MOTION_CLASS_VALUES!r}.")


def _resolve_graph(opts: PathSolveOptions | None, *, dst: Frame | str, owner: str) -> FrameGraph:
    if opts is not None and opts.graph is not None:
        if isinstance(opts.graph, FrameGraph):
            return opts.graph
        raise TypeError(f"{owner}: opts.graph must be FrameGraph or None.")
    if isinstance(dst, Frame):
        graph = dst._graph
        if isinstance(graph, FrameGraph):
            return graph
        raise ValueError(f"{owner}: dst frame is not bound to a valid FrameGraph.")
    return get_active_frame_graph()


def _resolve_endpoint(graph: FrameGraph, value: Frame | str, *, owner: str, arg: str) -> Frame:
    if isinstance(value, Frame):
        if value._graph is not graph:
            raise ValueError(f"{owner}: {arg} frame belongs to a different FrameGraph.")
        if graph.get_frame(value.id) is not value:
            raise ValueError(f"{owner}: {arg} frame {value.id!r} is not registered in graph.")
        return value
    if isinstance(value, str) and value.strip():
        resolved = graph.get_frame(value.strip())
        if isinstance(resolved, Frame):
            return resolved
        raise ValueError(f"{owner}: {arg} frame {value.strip()!r} is not registered in graph.")
    raise TypeError(f"{owner}: {arg} must be Frame or non-empty string frame id.")


def _resolve_source_frames(source, graph: FrameGraph, *, owner: str) -> tuple[Frame, Frame | None]:
    parent_id, child_id = get_frames(source.unsafe_data)
    if parent_id is None:
        raise ValueError(f"{owner}: source requires parent frame metadata.")
    src_parent = _resolve_endpoint(graph, parent_id, owner=owner, arg="source parent")
    if child_id is None:
        src_child = None
    else:
        resolved_child = graph.get_frame(child_id)
        src_child = resolved_child if isinstance(resolved_child, Frame) else None
    return src_parent, src_child


def _operation_kind(source, *, owner: str) -> str:
    kind = get_kinematics_kind(source.unsafe_data, owner=owner)
    if kind is None:
        raise ValueError(f"{owner}: source kinematics kind metadata is required.")
    if "velocity" in kind:
        return "velocity"
    if "acceleration" in kind:
        return "acceleration"
    raise ValueError(f"{owner}: unsupported kinematics kind {kind!r}.")


def _resolve_support(opts: PathSolveOptions | None, *, owner: str) -> KinematicsPathSupportOptions:
    if opts is None or opts.kinematics_support is None:
        return KinematicsPathSupportOptions()
    support = opts.kinematics_support
    if isinstance(support, KinematicsPathSupportOptions):
        return support
    raise TypeError(f"{owner}: opts.kinematics_support must be KinematicsPathSupportOptions or None.")


def _require_inertial_role_support(
    source,
    *,
    src_parent: Frame,
    src_child: Frame | None,
    frame_status_fn: Callable[[Frame], object],
    owner: str,
) -> None:
    for role in get_instantaneous_inertial(source.unsafe_data, owner=owner):
        target = src_parent if role == "parent" else src_child
        if target is None:
            raise ValueError(f"{owner}: inertial role {role!r} requires registered source child frame.")
        status = _require_status(frame_status_fn(target), owner=owner, what=f"inertial status for {role} role")
        if status != "inertial":
            raise ValueError(f"{owner}: required inertial support for role {role!r} is missing.")


def _resolve_edge_classes(
    path: FramePath,
    *,
    edge_class_fn: Callable[[Frame, Frame], object],
    owner: str,
) -> tuple[str, ...]:
    classes = tuple(
        _require_motion_class(
            edge_class_fn(step.child, step.parent),
            owner=owner,
            what=f"edge motion class ({step.child.id!r}, {step.parent.id!r})",
        )
        for step in path.steps
    )
    if any(cls == "unknown" for cls in classes):
        raise ValueError(f"{owner}: unknown edge motion class is insufficient for kinematic to_frame support.")
    return classes


def _require_operation_resolvers(
    *,
    operation: str,
    edge_classes: tuple[str, ...],
    support: KinematicsPathSupportOptions,
    owner: str,
) -> None:
    if operation == "velocity" and any(cls in {"galilean", "dynamic"} for cls in edge_classes):
        if support.edge_velocity_fn is None:
            raise ValueError(f"{owner}: edge_velocity_fn is required for galilean/dynamic velocity support.")
    if operation == "acceleration" and any(cls == "dynamic" for cls in edge_classes):
        if support.edge_acceleration_fn is None:
            raise ValueError(f"{owner}: edge_acceleration_fn is required for dynamic acceleration support.")


def resolve_kinematics_path_support(
    source,
    *,
    dst: Frame | str,
    opts: PathSolveOptions | None,
    owner: str,
) -> KinematicsPathSupportContext:
    graph = _resolve_graph(opts, dst=dst, owner=owner)
    src_parent, src_child = _resolve_source_frames(source, graph, owner=owner)
    dst_frame = _resolve_endpoint(graph, dst, owner=owner, arg="dst")
    operation = _operation_kind(source, owner=owner)
    path = find_path(src_parent, dst_frame)
    support = _resolve_support(opts, owner=owner)
    edge_class_fn = (
        _require_callable(support.edge_motion_class_fn, owner=owner, name="opts.kinematics_support.edge_motion_class_fn")
        if support.edge_motion_class_fn is not None
        else (lambda child, parent: get_edge_motion_class(child, parent, owner=owner))
    )
    frame_status_fn = (
        _require_callable(
            support.frame_inertial_status_fn,
            owner=owner,
            name="opts.kinematics_support.frame_inertial_status_fn",
        )
        if support.frame_inertial_status_fn is not None
        else (lambda frame: get_frame_inertial_status(frame, owner=owner))
    )
    _require_inertial_role_support(
        source,
        src_parent=src_parent,
        src_child=src_child,
        frame_status_fn=frame_status_fn,
        owner=owner,
    )
    edge_classes = _resolve_edge_classes(path, edge_class_fn=edge_class_fn, owner=owner)
    _require_operation_resolvers(operation=operation, edge_classes=edge_classes, support=support, owner=owner)
    return KinematicsPathSupportContext(
        graph=graph,
        src_parent=src_parent,
        src_child=src_child,
        dst=dst_frame,
        path=path,
        operation=operation,
        edge_classes=edge_classes,
        support=support,
    )


__all__ = [
    "KinematicsPathSupportContext",
    "resolve_kinematics_path_support",
]
