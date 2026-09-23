from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.schema_errors import SchemaError
from tal.frames import Frame, FramePath, find_path, fold_path
from tal.utils.frame_schema import set_frames

from ..association import attach_spatial_association, finalize_spatial_from_source
from ..metadata import get_pose_rep, set_expressed_in, set_pose_rep, set_rotation_rep
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
from .path_execution import (
    PreparedPosePathExecution,
    execute_pose_path,
    prepare_pose_path_execution,
)
from .path_query_ops import (
    CompletePathQuery,
    execute_path_query,
    prepare_path_query,
    require_path_temporal_options,
)
from .path_query_output import PathBasisResult, finalize_path_query_output
from .path_query_plan import PathOutputRequest
from .pose_ops import _pose_compose_with_owner, _pose_inverse_with_owner
from .pose_provider_ops import (
    normalize_edge_provider_dataset,
    normalize_pose_provider,
    require_bound_pose_provider,
    resolve_bound_pose,
    resolve_bound_pose_with_representation,
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


def _normalize_pose_edge_with_representation(
    payload: object,
    *,
    child: Frame,
    parent: Frame,
    owner: str,
) -> tuple[Pose, str]:
    try:
        source = payload if isinstance(payload, Pose) else Pose(payload)
        representation = get_pose_rep(analysis_object_dataset(source), owner=owner)
    except (TypeError, ValueError, SchemaError) as exc:
        raise ValueError(f"{owner}: edge resolver must return Pose-coercible payload.") from exc
    value = _normalize_pose_edge(
        source, child=child, parent=parent, owner=owner, strict=True,
    )
    return value, representation


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
) -> Rotation | Pose:
    if prepared.resolver is None:
        pose = resolve_bound_pose(child, parent, owner=owner)
        return pose if kind == "pose" else pose.decompose(validate=False)[1].as_quat(validate=False)
    payload = call_prepared_edge_resolver(
        prepared,
        child,
        parent,
        owner=owner,
    )
    if kind != "pose":
        return _normalize_rotation_edge(
            payload,
            child=child,
            parent=parent,
            owner=owner,
            strict=True,
        )
    return _normalize_pose_edge(payload, child=child, parent=parent, owner=owner, strict=True)


def _resolve_pose_value_with_representation(
    prepared: PreparedEdgeResolver,
    child: Frame,
    parent: Frame,
    *,
    owner: str,
) -> tuple[Pose, str]:
    if prepared.resolver is None:
        return resolve_bound_pose_with_representation(child, parent, owner=owner)
    payload = call_prepared_edge_resolver(prepared, child, parent, owner=owner)
    return _normalize_pose_edge_with_representation(
        payload, child=child, parent=parent, owner=owner,
    )


def _resolved_path(endpoints: ResolvedPathEndpointPlan, *, owner: str):
    try:
        path = find_path(endpoints.source, endpoints.destination)
    except ValueError as exc:
        raise ValueError(f"{owner}: {exc}") from exc
    return path


def _preflight_bound_path(path: FramePath, prepared: PreparedEdgeResolver, *, owner: str) -> None:
    if prepared.resolver is not None:
        return
    for step in path.steps:
        require_bound_pose_provider(step.child, step.parent, owner=owner)


def _acquire_path_values(
    path: FramePath,
    prepared: PreparedEdgeResolver,
    *,
    kind: str,
    owner: str,
) -> tuple[Rotation | Pose, ...]:
    return tuple(
        _resolve_edge_value(
            prepared,
            step.child,
            step.parent,
            kind=kind,
            owner=owner,
        )
        for step in path.steps
    )


def _acquire_pose_path_values(
    path: FramePath,
    prepared: PreparedEdgeResolver,
    *,
    owner: str,
) -> tuple[tuple[Pose, str], ...]:
    return tuple(
        _resolve_pose_value_with_representation(
            prepared, step.child, step.parent, owner=owner,
        )
        for step in path.steps
    )


def _path_value_map(path: FramePath, values: tuple[Rotation | Pose, ...]):
    return {
        (step.child.id, step.parent.id): value
        for step, value in zip(path.steps, values, strict=True)
    }


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


def _fold_rotation_path(path: FramePath, values: tuple[Rotation | Pose, ...], *, owner: str) -> Rotation | None:
    value_map = _path_value_map(path, values)
    result = fold_path(
        path,
        edge_value_fn=lambda child, parent: value_map[(child.id, parent.id)],
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
    query: object | None = None,
    caller: object | None = None,
    owner: str = "spatial.path_solve.rotation",
    prepared_resolver: PreparedEdgeResolver | None = None,
) -> Rotation | PathBasisResult:
    request = _path_output_request(caller)
    endpoints = resolve_path_endpoint_plan(
        configuration,
        src=src,
        dst=dst,
        owner=owner,
    )
    if endpoints.is_identity:
        return _basis_result(_associated_rotation_identity(endpoints, owner=owner), request)
    require_strict_path_policy(endpoints.configuration.options.strict, owner=owner)
    prepared = prepared_resolver or prepare_edge_resolver(
        edge_rotation_fn,
        owner=owner,
        arg="edge_rotation_fn",
    )
    path = _resolved_path(endpoints, owner=owner)
    temporal = require_path_temporal_options(endpoints.configuration.options.temporal, owner=owner)
    _preflight_bound_path(path, prepared, owner=owner)
    values = _acquire_path_values(path, prepared, kind="rotation", owner=owner)
    query_plan = prepare_path_query(
        values, query=query, caller=request.caller, temporal=temporal, owner=owner, basis=request.basis,
    )
    complete = execute_path_query(query_plan, owner=owner)
    result = _fold_rotation_path(path, complete.values, owner=owner)
    if result is None:
        return _basis_result(_associated_rotation_identity(endpoints, owner=owner), request)
    final = _finalize_rotation_path(result, endpoints, owner=owner)
    final = _verify_completed_query_result(
        final,
        output_plan=query_plan.output_plan,
        validate=query_plan.result_validate,
    )
    return _basis_result(final, request, query_plan.caller_output_plan)


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


def _fold_pose_path(path: FramePath, values: tuple[Rotation | Pose, ...], *, owner: str) -> Pose | None:
    value_map = _path_value_map(path, values)
    result = fold_path(
        path,
        edge_value_fn=lambda child, parent: value_map[(child.id, parent.id)],
        compose=lambda acc, value: _compose_pose_acc(acc, value, owner=owner),
        inverse=lambda value: _pose_inverse_with_owner(value, validate=False, owner=owner),
        identity=lambda: None,
    )
    return result


def _basis_result(value, request, output_plan=None):
    if request.basis:
        return PathBasisResult(value, request.caller, output_plan)
    return value


def _path_output_request(caller: object | None) -> PathOutputRequest:
    if isinstance(caller, PathOutputRequest):
        return caller
    prototype = caller if isinstance(caller, Position) else Pose
    return PathOutputRequest(caller, prototype, True)


def _verify_completed_query_result(
    value: Pose | Position | Rotation,
    *,
    output_plan,
    validate: bool,
):
    source = analysis_object_dataset(value)
    completed = finalize_path_query_output(source, plan=output_plan)
    if completed is source:
        return value
    return finalize_spatial_from_source(value, type(value), completed, validate=validate)


def _apply_completed_position_path(
    transform: Pose,
    execution: PreparedPosePathExecution,
    endpoints: ResolvedPathEndpointPlan,
    *,
    owner: str,
) -> Position:
    topology = execution.query.topology
    if topology is None or topology.caller is None:
        raise ValueError(f"{owner}: Position path result is missing caller topology.")
    from ..policies.wrap import wrap_like
    from .pose_apply_ops import _pose_apply_with_owner

    caller = topology.caller.ao
    applied = _pose_apply_with_owner(
        transform, caller, validate=False, owner=owner,
        association=endpoints.association,
    )
    dataset = finalize_path_query_output(
        analysis_object_dataset(applied),
        plan=execution.query.output_plan,
    )
    result = wrap_like(caller, dataset, validate=execution.query.result_validate)
    return attach_spatial_association(result, endpoints.association)


def _finalize_prepared_pose_path(
    complete: CompletePathQuery | Pose | Position,
    execution: PreparedPosePathExecution,
    path: FramePath,
    endpoints: ResolvedPathEndpointPlan,
    *,
    owner: str,
) -> Pose | Position:
    if isinstance(complete, Position):
        return complete
    if isinstance(complete, Pose):
        if execution.kind.startswith("batched-"):
            return complete
        return _finalize_pose_path(complete, endpoints, owner=owner)
    result = _fold_pose_path(path, complete.values, owner=owner)
    if result is None:
        return _associated_pose_identity(endpoints, owner=owner)
    transform = _finalize_pose_path(result, endpoints, owner=owner)
    if execution.query.output_intent == "position":
        topology = execution.query.topology
        if topology is None or topology.caller is None:
            return transform
        return _apply_completed_position_path(
            transform, execution, endpoints, owner=owner,
        )
    return _verify_completed_query_result(
        transform,
        output_plan=execution.query.output_plan,
        validate=execution.query.result_validate,
    )


def solve_pose_path_transform_impl(
    src: object,
    dst: object,
    *,
    edge_pose_fn: object,
    configuration: PathConfiguration | SelectedPathConfiguration,
    query: object | None = None,
    caller: object | None = None,
    owner: str = "spatial.path_solve.pose",
    prepared_resolver: PreparedEdgeResolver | None = None,
) -> Pose | Position | PathBasisResult:
    output_request = _path_output_request(caller)
    endpoints = resolve_path_endpoint_plan(
        configuration,
        src=src,
        dst=dst,
        owner=owner,
    )
    if endpoints.is_identity:
        return _basis_result(_associated_pose_identity(endpoints, owner=owner), output_request)
    require_strict_path_policy(endpoints.configuration.options.strict, owner=owner)
    prepared = prepared_resolver or prepare_edge_resolver(
        edge_pose_fn,
        owner=owner,
        arg="edge_pose_fn",
    )
    path = _resolved_path(endpoints, owner=owner)
    temporal = require_path_temporal_options(endpoints.configuration.options.temporal, owner=owner)
    _preflight_bound_path(path, prepared, owner=owner)
    acquired = _acquire_pose_path_values(path, prepared, owner=owner)
    values = tuple(item[0] for item in acquired)
    query_plan = prepare_path_query(
        values,
        query=query,
        caller=output_request.caller,
        temporal=temporal,
        owner=owner,
        source_representations=tuple(item[1] for item in acquired),
        result_context=endpoints.association,
        result_prototype=output_request.prototype,
        result_validate=output_request.validate,
        basis=output_request.basis,
    )
    execution = prepare_pose_path_execution(path, query_plan)
    complete = execute_pose_path(execution, owner=owner)
    final = _finalize_prepared_pose_path(
        complete, execution, path, endpoints, owner=owner,
    )
    return _basis_result(final, output_request, query_plan.caller_output_plan)


__all__ = [
    "solve_pose_path_transform_impl",
    "solve_rotation_path_transform_impl",
]
