from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.analysis_object import AnalysisObject
from tal.core.schema_errors import SchemaError
from tal.frames import Frame, FrameGraph, find_path, fold_path, get_active_frame_graph
from tal.utils.frame_schema import get_frames, set_frames

from ..policies.frame import is_framed
from ..metadata import set_pose_rep, set_rotation_rep
from ..pose import Pose
from .pose_ops import _pose_compose_with_owner, _pose_inverse_with_owner
from ..position import Position
from ..rotation import Rotation, _rotation_compose_with_owner, _rotation_inverse_with_owner

_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")
_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")


def _require_frame_id(value: object, *, owner: str, arg: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{owner}: {arg} must be Frame or non-empty string frame id.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{owner}: {arg} must be Frame or non-empty string frame id.")
    return cleaned


def _require_graph_binding(frame: Frame, *, owner: str, arg: str) -> FrameGraph:
    graph = frame._graph
    if isinstance(graph, FrameGraph):
        return graph
    raise ValueError(f"{owner}: {arg} frame is not bound to a valid FrameGraph.")


def _require_registered_frame(frame: Frame, *, graph: FrameGraph, owner: str, arg: str) -> Frame:
    if graph._frames.get(frame.id) is frame:
        return frame
    raise ValueError(f"{owner}: {arg} frame {frame.id!r} is not registered in graph.")


def _resolve_graph(src: object, dst: object, *, graph: FrameGraph | None, owner: str) -> FrameGraph:
    if graph is not None:
        if not isinstance(graph, FrameGraph):
            raise TypeError(f"{owner}: opts.graph must be FrameGraph or None, got {type(graph).__name__}.")
        if isinstance(src, Frame) and _require_graph_binding(src, owner=owner, arg="src") is not graph:
            raise ValueError(f"{owner}: src frame belongs to a different FrameGraph.")
        if isinstance(dst, Frame) and _require_graph_binding(dst, owner=owner, arg="dst") is not graph:
            raise ValueError(f"{owner}: dst frame belongs to a different FrameGraph.")
        return graph
    src_graph = _require_graph_binding(src, owner=owner, arg="src") if isinstance(src, Frame) else None
    dst_graph = _require_graph_binding(dst, owner=owner, arg="dst") if isinstance(dst, Frame) else None
    if src_graph is not None and dst_graph is not None and src_graph is not dst_graph:
        raise ValueError(f"{owner}: src and dst belong to different FrameGraph instances.")
    if src_graph is not None:
        return src_graph
    if dst_graph is not None:
        return dst_graph
    return get_active_frame_graph()


def _resolve_endpoint(value: object, *, graph: FrameGraph, owner: str, arg: str) -> Frame:
    if isinstance(value, Frame):
        if _require_graph_binding(value, owner=owner, arg=arg) is not graph:
            raise ValueError(f"{owner}: {arg} frame belongs to a different FrameGraph.")
        return _require_registered_frame(value, graph=graph, owner=owner, arg=arg)
    frame_id = _require_frame_id(value, owner=owner, arg=arg)
    resolved = graph.get_frame(frame_id)
    if resolved is None:
        raise ValueError(f"{owner}: {arg} frame {frame_id!r} is not registered in graph.")
    return _require_registered_frame(resolved, graph=graph, owner=owner, arg=arg)


def _require_callable(value: object, *, owner: str, arg: str) -> Callable[[Frame, Frame], object]:
    if callable(value):
        return value  # type: ignore[return-value]
    raise TypeError(f"{owner}: {arg} must be callable(child, parent).")


def _require_resolver_signature(
    resolver: Callable[[Frame, Frame], object],
    *,
    owner: str,
    arg: str,
) -> bool:
    try:
        signature = inspect.signature(resolver)
    except (TypeError, ValueError):
        return False
    try:
        signature.bind(object(), object())
    except TypeError as exc:
        raise TypeError(f"{owner}: {arg} must be callable(child, parent).") from exc
    return True


def _is_invocation_signature_typeerror(exc: TypeError) -> bool:
    traceback_obj = exc.__traceback__
    if traceback_obj is None:
        return True
    return traceback_obj.tb_next is None


def _call_edge_resolver(
    resolver: Callable[[Frame, Frame], object],
    child: Frame,
    parent: Frame,
    *,
    owner: str,
    arg: str,
    signature_checked: bool,
) -> object:
    try:
        return resolver(child, parent)
    except TypeError as exc:
        if not signature_checked and _is_invocation_signature_typeerror(exc):
            raise TypeError(f"{owner}: {arg} must be callable(child, parent).") from exc
        raise ValueError(f"{owner}: {arg} failed for edge (child={child.id!r}, parent={parent.id!r}).") from exc
    except Exception as exc:
        raise ValueError(f"{owner}: {arg} failed for edge (child={child.id!r}, parent={parent.id!r}).") from exc


def _require_strict_supported(strict: object, *, owner: str) -> None:
    if not isinstance(strict, bool):
        raise TypeError(f"{owner}: opts.strict must be bool, got {type(strict).__name__}.")
    if not strict:
        raise ValueError(f"{owner}: opts.strict=False is not supported.")


def _normalize_rotation_edge(
    payload: object,
    *,
    child: Frame,
    parent: Frame,
    owner: str,
    strict: bool,
) -> Rotation:
    try:
        rotation = payload if isinstance(payload, Rotation) else Rotation(payload)
        rotation = rotation.as_quat(validate=False)
    except (TypeError, ValueError, SchemaError) as exc:
        raise ValueError(f"{owner}: edge resolver must return Rotation-coercible payload.") from exc
    source = analysis_object_dataset(rotation)
    edge_parent, edge_child = get_frames(source)
    if strict and is_framed(edge_parent, edge_child) and (edge_parent, edge_child) != (parent.id, child.id):
        raise ValueError(
            f"{owner}: framed edge payload must match (parent={parent.id!r}, child={child.id!r}); "
            f"got {(edge_parent, edge_child)!r}."
        )
    cleared = set_frames(source, parent=None, child=None, validate=False)
    return Rotation._from_unvalidated(cleared)


def _normalize_pose_edge(
    payload: object,
    *,
    child: Frame,
    parent: Frame,
    owner: str,
    strict: bool,
) -> Pose:
    try:
        pose = payload if isinstance(payload, Pose) else Pose(payload)
        pose = pose.as_components(validate=False)
    except (TypeError, ValueError, SchemaError) as exc:
        raise ValueError(f"{owner}: edge resolver must return Pose-coercible payload.") from exc
    source = analysis_object_dataset(pose)
    edge_parent, edge_child = get_frames(source)
    if strict and is_framed(edge_parent, edge_child) and (edge_parent, edge_child) != (parent.id, child.id):
        raise ValueError(
            f"{owner}: framed edge payload must match (parent={parent.id!r}, child={child.id!r}); "
            f"got {(edge_parent, edge_child)!r}."
        )
    cleared = set_frames(source, parent=None, child=None, validate=False)
    return Pose._from_unvalidated(cleared)


def _rotation_identity(*, parent: str, child: str, owner: str) -> Rotation:
    arr = xr.DataArray(
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=float),
        dims=("sample", "quat"),
        coords={"sample": [0], "quat": list(_QUAT_LABELS)},
        name="rotation",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="rotation"), sequence_dim="sample", core_dims=("quat",), validate=True)
    ds = set_rotation_rep(analysis_object_dataset(ao), rep="quat", validate=False, owner=owner)
    ds = set_frames(ds, parent=parent, child=child, validate=False)
    return Rotation._from_validated(ds)


def _pose_identity(*, parent: str, child: str, owner: str) -> Pose:
    arr = xr.DataArray(
        np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        dims=("sample", "axis"),
        coords={"sample": [0], "axis": list(_XYZ_LABELS)},
        name="position",
    )
    ao = AnalysisObject.from_data(arr.to_dataset(name="position"), sequence_dim="sample", core_dims=("axis",), validate=True)
    position = Position(ao)
    rotation = _rotation_identity(parent=parent, child=child, owner=owner)
    pose = Pose.from_components(rotation, position, validate=False).as_components(validate=False)
    ds = set_pose_rep(analysis_object_dataset(pose), rep="components", validate=False, owner=owner)
    ds = set_frames(ds, parent=parent, child=child, validate=False)
    return Pose._from_validated(ds)


def _compose_rotation_acc(acc: Rotation | None, value: Rotation, *, owner: str) -> Rotation:
    if acc is None:
        return value
    return _rotation_compose_with_owner(acc, value, validate=False, owner=owner)


def _compose_pose_acc(acc: Pose | None, value: Pose, *, owner: str) -> Pose:
    if acc is None:
        return value
    return _pose_compose_with_owner(acc, value, validate=False, owner=owner)


def solve_rotation_path_transform_impl(
    src: object,
    dst: object,
    *,
    edge_rotation_fn: object,
    graph: FrameGraph | None,
    strict: object,
    owner: str = "spatial.path_solve.rotation",
) -> Rotation:
    _require_strict_supported(strict, owner=owner)
    resolver = _require_callable(edge_rotation_fn, owner=owner, arg="edge_rotation_fn")
    signature_checked = _require_resolver_signature(resolver, owner=owner, arg="edge_rotation_fn")
    resolved_graph = _resolve_graph(src, dst, graph=graph, owner=owner)
    src_frame = _resolve_endpoint(src, graph=resolved_graph, owner=owner, arg="src")
    dst_frame = _resolve_endpoint(dst, graph=resolved_graph, owner=owner, arg="dst")
    if src_frame is dst_frame:
        return _rotation_identity(parent=dst_frame.id, child=src_frame.id, owner=owner)
    try:
        path = find_path(src_frame, dst_frame)
    except ValueError as exc:
        raise ValueError(f"{owner}: {exc}") from exc
    result = fold_path(
        path,
        edge_value_fn=lambda child, parent: _normalize_rotation_edge(
            _call_edge_resolver(
                resolver,
                child,
                parent,
                owner=owner,
                arg="edge_rotation_fn",
                signature_checked=signature_checked,
            ),
            child=child,
            parent=parent,
            owner=owner,
            strict=True,
        ),
        compose=lambda acc, value: _compose_rotation_acc(acc, value, owner=owner),
        inverse=lambda value: _rotation_inverse_with_owner(value, validate=False, owner=owner),
        identity=lambda: None,
    )
    if result is None:
        return _rotation_identity(parent=dst_frame.id, child=src_frame.id, owner=owner)
    out = set_rotation_rep(analysis_object_dataset(result), rep="quat", validate=False, owner=owner)
    out = set_frames(out, parent=dst_frame.id, child=src_frame.id, validate=False)
    return Rotation._from_validated(out)


def solve_pose_path_transform_impl(
    src: object,
    dst: object,
    *,
    edge_pose_fn: object,
    graph: FrameGraph | None,
    strict: object,
    owner: str = "spatial.path_solve.pose",
) -> Pose:
    _require_strict_supported(strict, owner=owner)
    resolver = _require_callable(edge_pose_fn, owner=owner, arg="edge_pose_fn")
    signature_checked = _require_resolver_signature(resolver, owner=owner, arg="edge_pose_fn")
    resolved_graph = _resolve_graph(src, dst, graph=graph, owner=owner)
    src_frame = _resolve_endpoint(src, graph=resolved_graph, owner=owner, arg="src")
    dst_frame = _resolve_endpoint(dst, graph=resolved_graph, owner=owner, arg="dst")
    if src_frame is dst_frame:
        return _pose_identity(parent=dst_frame.id, child=src_frame.id, owner=owner)
    try:
        path = find_path(src_frame, dst_frame)
    except ValueError as exc:
        raise ValueError(f"{owner}: {exc}") from exc
    result = fold_path(
        path,
        edge_value_fn=lambda child, parent: _normalize_pose_edge(
            _call_edge_resolver(
                resolver,
                child,
                parent,
                owner=owner,
                arg="edge_pose_fn",
                signature_checked=signature_checked,
            ),
            child=child,
            parent=parent,
            owner=owner,
            strict=True,
        ),
        compose=lambda acc, value: _compose_pose_acc(acc, value, owner=owner),
        inverse=lambda value: _pose_inverse_with_owner(value, validate=False, owner=owner),
        identity=lambda: None,
    )
    if result is None:
        return _pose_identity(parent=dst_frame.id, child=src_frame.id, owner=owner)
    out = set_pose_rep(analysis_object_dataset(result), rep="components", validate=False, owner=owner)
    out = set_frames(out, parent=dst_frame.id, child=src_frame.id, validate=False)
    return Pose._from_validated(out)


__all__ = [
    "solve_pose_path_transform_impl",
    "solve_rotation_path_transform_impl",
]
