from __future__ import annotations

from dataclasses import dataclass

from tal.core.dataset_ownership import analysis_object_dataset
from tal.frames import (
    Frame,
    FrameGraph,
    FramePath,
    find_path,
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
from .edge_resolver_ops import (
    PreparedEdgeResolver,
    _call_prepared_edge_callback,
    _call_prepared_frame_callback,
    _prepare_required_edge_callback,
    _prepare_required_frame_callback,
    _PreparedPathCallback,
)
from .path_configuration import SelectedPathConfiguration, resolve_path_endpoint
from .pose_provider_ops import require_bound_pose_provider


@dataclass(frozen=True)
class KinematicsPathSupportContext:
    graph: FrameGraph
    src_parent: Frame
    src_child: Frame | None
    dst: Frame
    path: FramePath
    operation: str
    edge_classes: tuple[str, ...]
    motion_callback: _PreparedPathCallback | None
    motion_payloads: tuple[object | None, ...]
    finalized: bool = False


@dataclass(frozen=True)
class _PreparedSupportCallbacks:
    edge_class: _PreparedPathCallback | None
    frame_status: _PreparedPathCallback | None


def _require_status(value: object, *, owner: str, what: str) -> str:
    if isinstance(value, str) and value.strip() in FRAME_INERTIAL_STATUS_VALUES:
        return value.strip()
    raise ValueError(f"{owner}: {what} must be one of {FRAME_INERTIAL_STATUS_VALUES!r}.")


def _require_motion_class(value: object, *, owner: str, what: str) -> str:
    if isinstance(value, str) and value.strip() in EDGE_MOTION_CLASS_VALUES:
        return value.strip()
    raise ValueError(f"{owner}: {what} must be one of {EDGE_MOTION_CLASS_VALUES!r}.")


def _resolve_source_frames(source, graph: FrameGraph, *, owner: str) -> tuple[Frame, Frame | None]:
    parent_id, child_id = get_frames(analysis_object_dataset(source))
    if parent_id is None:
        raise ValueError(f"{owner}: source requires parent frame metadata.")
    src_parent = resolve_path_endpoint(parent_id, graph=graph, owner=owner, arg="source parent")
    if child_id is None:
        src_child = None
    else:
        resolved_child = graph.get_frame(child_id)
        src_child = resolved_child if isinstance(resolved_child, Frame) else None
    return src_parent, src_child


def _operation_kind(source, *, owner: str) -> str:
    kind = get_kinematics_kind(analysis_object_dataset(source), owner=owner)
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
    inertial_roles,
    *,
    src_parent: Frame,
    src_child: Frame | None,
    frame_status: _PreparedPathCallback | None,
    owner: str,
) -> None:
    for role in inertial_roles:
        target = src_parent if role == "parent" else src_child
        if target is None:
            raise ValueError(f"{owner}: inertial role {role!r} requires registered source child frame.")
        value = (
            get_frame_inertial_status(target, owner=owner)
            if frame_status is None
            else _call_prepared_frame_callback(frame_status, target, owner=owner)
        )
        status = _require_status(value, owner=owner, what=f"inertial status for {role} role")
        if status != "inertial":
            raise ValueError(f"{owner}: required inertial support for role {role!r} is missing.")


def _resolve_edge_classes(
    path: FramePath,
    *,
    edge_class: _PreparedPathCallback | None,
    owner: str,
) -> tuple[str, ...]:
    classes = tuple(
        _require_motion_class(
            get_edge_motion_class(step.child, step.parent, owner=owner)
            if edge_class is None
            else _call_prepared_edge_callback(edge_class, step.child, step.parent, owner=owner),
            owner=owner,
            what=f"edge motion class ({step.child.id!r}, {step.parent.id!r})",
        )
        for step in path.steps
    )
    if any(cls == "unknown" for cls in classes):
        raise ValueError(f"{owner}: unknown edge motion class is insufficient for kinematic to_frame support.")
    return classes


def _prepare_motion_callback(
    *,
    operation: str,
    edge_classes: tuple[str, ...],
    support: KinematicsPathSupportOptions,
    owner: str,
) -> _PreparedPathCallback | None:
    velocity_required = operation == "velocity" and any(
        cls in {"galilean", "dynamic"} for cls in edge_classes
    )
    acceleration_required = operation == "acceleration" and any(cls == "dynamic" for cls in edge_classes)
    if not velocity_required and not acceleration_required:
        return None
    name = "edge_velocity_fn" if velocity_required else "edge_acceleration_fn"
    value = getattr(support, name)
    if value is None:
        detail = "galilean/dynamic velocity" if velocity_required else "dynamic acceleration"
        raise ValueError(f"{owner}: {name} is required for {detail} support.")
    return _prepare_required_edge_callback(
        value,
        owner=owner,
        arg=f"opts.kinematics_support.{name}",
    )


def _prepare_support_callbacks(
    support: KinematicsPathSupportOptions,
    *,
    needs_frame_status: bool,
    owner: str,
) -> _PreparedSupportCallbacks:
    edge_class = None
    if support.edge_motion_class_fn is not None:
        edge_class = _prepare_required_edge_callback(
            support.edge_motion_class_fn,
            owner=owner,
            arg="opts.kinematics_support.edge_motion_class_fn",
        )
    frame_status = None
    if needs_frame_status and support.frame_inertial_status_fn is not None:
        frame_status = _prepare_required_frame_callback(
            support.frame_inertial_status_fn,
            owner=owner,
            arg="opts.kinematics_support.frame_inertial_status_fn",
        )
    return _PreparedSupportCallbacks(edge_class, frame_status)


def _coerce_motion_payload(payload: object, *, operation: str, owner: str):
    if operation == "velocity":
        from ..velocity import Velocity

        if isinstance(payload, Velocity):
            return payload
        raise ValueError(f"{owner}: edge_velocity_fn must return Velocity payloads.")
    from ..acceleration import Acceleration

    if isinstance(payload, Acceleration):
        return payload
    raise ValueError(f"{owner}: edge_acceleration_fn must return Acceleration payloads.")


def _motion_required(operation: str, motion_class: str) -> bool:
    if operation == "velocity":
        return motion_class in {"galilean", "dynamic"}
    return motion_class == "dynamic"


def _acquire_motion_payloads(
    path: FramePath,
    edge_classes: tuple[str, ...],
    *,
    operation: str,
    callback: _PreparedPathCallback | None,
    owner: str,
) -> tuple[object | None, ...]:
    payloads: list[object | None] = []
    for step, motion_class in zip(path.steps, edge_classes, strict=True):
        if not _motion_required(operation, motion_class):
            payloads.append(None)
            continue
        if callback is None:
            raise RuntimeError("motion support preflight did not prepare its callback")
        payload = _call_prepared_edge_callback(callback, step.child, step.parent, owner=owner)
        payloads.append(_coerce_motion_payload(payload, operation=operation, owner=owner))
    return tuple(payloads)


def finalize_kinematics_path_support(
    context: KinematicsPathSupportContext,
    payloads: tuple[object | None, ...],
) -> KinematicsPathSupportContext:
    return KinematicsPathSupportContext(
        graph=context.graph,
        src_parent=context.src_parent,
        src_child=context.src_child,
        dst=context.dst,
        path=context.path,
        operation=context.operation,
        edge_classes=context.edge_classes,
        motion_callback=context.motion_callback,
        motion_payloads=payloads,
        finalized=True,
    )

def _require_bound_path_providers(
    path: FramePath,
    prepared_resolver: PreparedEdgeResolver,
    *,
    owner: str,
) -> None:
    if prepared_resolver.resolver is not None:
        return
    for step in path.steps:
        require_bound_pose_provider(step.child, step.parent, owner=owner)


def _resolve_support_path(
    source,
    *,
    dst: Frame | str,
    graph: FrameGraph,
    owner: str,
) -> tuple[Frame, Frame | None, Frame, FramePath, str]:
    src_parent, src_child = _resolve_source_frames(source, graph, owner=owner)
    dst_frame = resolve_path_endpoint(dst, graph=graph, owner=owner, arg="dst")
    operation = _operation_kind(source, owner=owner)
    try:
        path = find_path(src_parent, dst_frame)
    except ValueError as exc:
        raise ValueError(f"{owner}: {exc}") from exc
    return src_parent, src_child, dst_frame, path, operation


def _resolve_support_policy(
    source,
    path: FramePath,
    *,
    opts: PathSolveOptions,
    src_parent: Frame,
    src_child: Frame | None,
    operation: str,
    owner: str,
) -> tuple[tuple[str, ...], _PreparedPathCallback | None]:
    support = _resolve_support(opts, owner=owner)
    inertial_roles = get_instantaneous_inertial(analysis_object_dataset(source), owner=owner)
    callbacks = _prepare_support_callbacks(
        support,
        needs_frame_status=bool(inertial_roles),
        owner=owner,
    )
    _require_inertial_role_support(
        inertial_roles,
        src_parent=src_parent,
        src_child=src_child,
        frame_status=callbacks.frame_status,
        owner=owner,
    )
    edge_classes = _resolve_edge_classes(path, edge_class=callbacks.edge_class, owner=owner)
    motion_callback = _prepare_motion_callback(
        operation=operation,
        edge_classes=edge_classes,
        support=support,
        owner=owner,
    )
    return edge_classes, motion_callback


def resolve_kinematics_path_support(
    source,
    *,
    dst: Frame | str,
    configuration: SelectedPathConfiguration,
    prepared_resolver: PreparedEdgeResolver,
    owner: str,
) -> KinematicsPathSupportContext:
    opts, graph = configuration.options, configuration.graph
    src_parent, src_child, dst_frame, path, operation = _resolve_support_path(
        source,
        dst=dst,
        graph=graph,
        owner=owner,
    )
    _require_bound_path_providers(path, prepared_resolver, owner=owner)
    edge_classes, motion_callback = _resolve_support_policy(
        source,
        path,
        opts=opts,
        src_parent=src_parent,
        src_child=src_child,
        operation=operation,
        owner=owner,
    )
    motion_payloads = _acquire_motion_payloads(
        path,
        edge_classes,
        operation=operation,
        callback=motion_callback,
        owner=owner,
    )
    return KinematicsPathSupportContext(
        graph=graph,
        src_parent=src_parent,
        src_child=src_child,
        dst=dst_frame,
        path=path,
        operation=operation,
        edge_classes=edge_classes,
        motion_callback=motion_callback,
        motion_payloads=motion_payloads,
    )


__all__ = [
    "KinematicsPathSupportContext",
    "finalize_kinematics_path_support",
    "resolve_kinematics_path_support",
]
