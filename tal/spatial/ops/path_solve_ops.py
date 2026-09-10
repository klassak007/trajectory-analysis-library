from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.schema_errors import SchemaError
from tal.frames import Frame, find_path, fold_path
from tal.utils.frame_schema import set_frames

from ..association import attach_spatial_association
from ..metadata import set_expressed_in, set_pose_rep, set_rotation_rep
from ..pose import Pose
from ..position import Position
from ..rotation import (
    Rotation,
    _rotation_compose_with_owner,
    _rotation_inverse_with_owner,
)
from .edge_resolver_ops import (
    PreparedEdgeResolver,
    call_prepared_edge_resolver,
    prepare_edge_resolver,
)
from .path_configuration import (
    PathConfiguration,
    ResolvedPathEndpointPlan,
    SelectedPathConfiguration,
    require_strict_path_policy,
    resolve_path_endpoint_plan,
)
from .pose_ops import _pose_compose_with_owner, _pose_inverse_with_owner
from .pose_provider_ops import (
    normalize_edge_provider_dataset,
    normalize_pose_provider,
    resolve_bound_pose,
)

_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")
_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")


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
    _ = strict
    cleared = normalize_edge_provider_dataset(
        analysis_object_dataset(rotation),
        child_id=child.id,
        parent_id=parent.id,
        owner=owner,
    )
    return Rotation._from_unvalidated(cleared)


def _normalize_pose_edge(
    payload: object,
    *,
    child: Frame,
    parent: Frame,
    owner: str,
    strict: bool,
) -> Pose:
    return normalize_pose_provider(
        payload, child_id=child.id, parent_id=parent.id, owner=owner,
    )


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
    ds = set_expressed_in(ds, expressed_in=parent, validate=False, owner=owner)
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
    ds = set_expressed_in(ds, expressed_in=parent, validate=False, owner=owner)
    return Pose._from_validated(ds)


def _compose_rotation_acc(acc: Rotation | None, value: Rotation, *, owner: str) -> Rotation:
    if acc is None:
        return value
    return _rotation_compose_with_owner(acc, value, validate=False, owner=owner)


def _compose_pose_acc(acc: Pose | None, value: Pose, *, owner: str) -> Pose:
    if acc is None:
        return value
    return _pose_compose_with_owner(acc, value, validate=False, owner=owner)


def _resolve_edge_value(
    prepared: PreparedEdgeResolver,
    child,
    parent,
    *,
    kind: str,
    owner: str,
):
    if prepared.resolver is None:
        pose = resolve_bound_pose(child, parent, owner=owner)
        return pose if kind == "pose" else pose.decompose(validate=False)[1].as_quat(validate=False)
    payload = call_prepared_edge_resolver(
        prepared,
        child,
        parent,
        owner=owner,
    )
    normalize = _normalize_pose_edge if kind == "pose" else _normalize_rotation_edge
    return normalize(payload, child=child, parent=parent, owner=owner, strict=True)


def _resolved_path(endpoints: ResolvedPathEndpointPlan, *, owner: str):
    try:
        path = find_path(endpoints.source, endpoints.destination)
    except ValueError as exc:
        raise ValueError(f"{owner}: {exc}") from exc
    return path


def _associated_rotation_identity(
    endpoints: ResolvedPathEndpointPlan,
    *,
    owner: str,
) -> Rotation:
    identity = _rotation_identity(
        parent=endpoints.destination.id,
        child=endpoints.source.id,
        owner=owner,
    )
    return attach_spatial_association(identity, endpoints.association)


def _finalize_rotation_path(
    result: Rotation,
    endpoints: ResolvedPathEndpointPlan,
    *,
    owner: str,
) -> Rotation:
    out = set_rotation_rep(
        analysis_object_dataset(result),
        rep="quat",
        validate=False,
        owner=owner,
    )
    out = set_frames(
        out,
        parent=endpoints.destination.id,
        child=endpoints.source.id,
        validate=False,
    )
    out = set_expressed_in(
        out,
        expressed_in=endpoints.destination.id,
        validate=False,
        owner=owner,
    )
    return attach_spatial_association(Rotation._from_validated(out), endpoints.association)


def _fold_rotation_path(path, prepared: PreparedEdgeResolver, *, owner: str) -> Rotation | None:
    result = fold_path(
        path,
        edge_value_fn=lambda child, parent: _resolve_edge_value(
            prepared,
            child,
            parent,
            kind="rotation",
            owner=owner,
        ),
        compose=lambda acc, value: _compose_rotation_acc(acc, value, owner=owner),
        inverse=lambda value: _rotation_inverse_with_owner(value, validate=False, owner=owner),
        identity=lambda: None,
    )
    return result


def solve_rotation_path_transform_impl(
    src: object,
    dst: object,
    *,
    edge_rotation_fn: object,
    configuration: PathConfiguration | SelectedPathConfiguration,
    owner: str = "spatial.path_solve.rotation",
    prepared_resolver: PreparedEdgeResolver | None = None,
) -> Rotation:
    endpoints = resolve_path_endpoint_plan(
        configuration,
        src=src,
        dst=dst,
        owner=owner,
    )
    if endpoints.is_identity:
        return _associated_rotation_identity(endpoints, owner=owner)
    require_strict_path_policy(endpoints.configuration.options.strict, owner=owner)
    prepared = prepared_resolver or prepare_edge_resolver(
        edge_rotation_fn,
        owner=owner,
        arg="edge_rotation_fn",
    )
    result = _fold_rotation_path(_resolved_path(endpoints, owner=owner), prepared, owner=owner)
    if result is None:
        return _associated_rotation_identity(endpoints, owner=owner)
    return _finalize_rotation_path(result, endpoints, owner=owner)


def _associated_pose_identity(
    endpoints: ResolvedPathEndpointPlan,
    *,
    owner: str,
) -> Pose:
    identity = _pose_identity(
        parent=endpoints.destination.id,
        child=endpoints.source.id,
        owner=owner,
    )
    return attach_spatial_association(identity, endpoints.association)


def _finalize_pose_path(
    result: Pose,
    endpoints: ResolvedPathEndpointPlan,
    *,
    owner: str,
) -> Pose:
    out = set_pose_rep(
        analysis_object_dataset(result),
        rep="components",
        validate=False,
        owner=owner,
    )
    out = set_frames(
        out,
        parent=endpoints.destination.id,
        child=endpoints.source.id,
        validate=False,
    )
    out = set_expressed_in(
        out,
        expressed_in=endpoints.destination.id,
        validate=False,
        owner=owner,
    )
    return attach_spatial_association(Pose._from_validated(out), endpoints.association)


def _fold_pose_path(path, prepared: PreparedEdgeResolver, *, owner: str) -> Pose | None:
    result = fold_path(
        path,
        edge_value_fn=lambda child, parent: _resolve_edge_value(
            prepared,
            child,
            parent,
            kind="pose",
            owner=owner,
        ),
        compose=lambda acc, value: _compose_pose_acc(acc, value, owner=owner),
        inverse=lambda value: _pose_inverse_with_owner(value, validate=False, owner=owner),
        identity=lambda: None,
    )
    return result


def solve_pose_path_transform_impl(
    src: object,
    dst: object,
    *,
    edge_pose_fn: object,
    configuration: PathConfiguration | SelectedPathConfiguration,
    owner: str = "spatial.path_solve.pose",
    prepared_resolver: PreparedEdgeResolver | None = None,
) -> Pose:
    endpoints = resolve_path_endpoint_plan(
        configuration,
        src=src,
        dst=dst,
        owner=owner,
    )
    if endpoints.is_identity:
        return _associated_pose_identity(endpoints, owner=owner)
    require_strict_path_policy(endpoints.configuration.options.strict, owner=owner)
    prepared = prepared_resolver or prepare_edge_resolver(
        edge_pose_fn,
        owner=owner,
        arg="edge_pose_fn",
    )
    result = _fold_pose_path(_resolved_path(endpoints, owner=owner), prepared, owner=owner)
    if result is None:
        return _associated_pose_identity(endpoints, owner=owner)
    return _finalize_pose_path(result, endpoints, owner=owner)


__all__ = [
    "solve_pose_path_transform_impl",
    "solve_rotation_path_transform_impl",
]
