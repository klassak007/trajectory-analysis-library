from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

from tal.frames import Frame, FramePath, find_path

from ..temporal.options import PoseTemporalOptions
from .frame_owner_common import require_source_parent, source_expressed_in_id
from .kinematics_family_frame_ops import (
    FamilyFrameRequest,
    PreparedFramePathRequest,
    prepare_frame_path_request,
)
from .kinematics_path_coupling_ops import (
    _express_motion_in_selected_basis,
    apply_vector_path_coupling,
)
from .kinematics_path_support_ops import (
    KinematicsPathSupportContext,
    finalize_kinematics_path_support,
    resolve_kinematics_path_support,
)
from .path_configuration import resolve_path_endpoint, resolve_path_endpoint_plan
from .path_query_ops import complete_path_query, require_path_temporal_options
from .path_solve_ops import (
    _finalize_pose_path,
    _finalize_rotation_path,
    _fold_pose_path,
    _fold_rotation_path,
    _preflight_bound_path,
    _resolve_edge_value,
)

_EdgeKey = tuple[str, str]


@dataclass(frozen=True)
class _KinematicProviderPlan:
    """Complete provider wrappers and paths discovered before evaluation."""

    support: KinematicsPathSupportContext
    source_basis_path: FramePath | None
    motion_basis_paths: tuple[FramePath | None, ...]
    edge_keys: tuple[_EdgeKey, ...]
    edge_values: tuple[object, ...]
    motion_values: tuple[object, ...]
    temporal: PoseTemporalOptions


@dataclass(frozen=True)
class _EvaluatedProviders:
    edge_values: tuple[tuple[_EdgeKey, object], ...]
    motion_payloads: tuple[object | None, ...]


def _find_basis_path(
    value,
    target: Frame,
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
) -> FramePath | None:
    basis_id = source_expressed_in_id(value, owner=owner)
    if basis_id == target.id:
        return None
    basis = resolve_path_endpoint(
        basis_id,
        graph=prepared.configuration.graph,
        owner=owner,
        arg="provider basis",
    )
    try:
        path = find_path(basis, target)
    except ValueError as exc:
        raise ValueError(f"{owner}: {exc}") from exc
    _preflight_bound_path(path, prepared.resolver, owner=owner)
    return path


def _source_basis_path(
    request: FamilyFrameRequest,
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
) -> FramePath | None:
    parent_id, _ = require_source_parent(request.source, owner=owner)
    parent = resolve_path_endpoint(
        parent_id,
        graph=prepared.configuration.graph,
        owner=owner,
        arg="source parent",
    )
    return _find_basis_path(request.source, parent, prepared, owner=owner)


def _motion_basis_paths(
    support: KinematicsPathSupportContext,
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
) -> tuple[FramePath | None, ...]:
    return tuple(
        None
        if payload is None
        else _find_basis_path(payload, support.src_parent, prepared, owner=owner)
        for payload in support.motion_payloads
    )


def _unique_path_steps(paths: tuple[FramePath, ...]):
    seen: set[_EdgeKey] = set()
    steps = chain.from_iterable(path.steps for path in paths)
    for step in steps:
        key = (step.child.id, step.parent.id)
        if key not in seen:
            seen.add(key)
            yield key, step


def _acquire_pose_values(
    paths: tuple[FramePath, ...],
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
) -> tuple[tuple[_EdgeKey, ...], tuple[object, ...]]:
    cached: dict[_EdgeKey, object] = {}
    ordered: list[_EdgeKey] = []
    for key, step in _unique_path_steps(paths):
        if key in cached:
            continue
        cached[key] = _resolve_edge_value(
            prepared.resolver,
            step.child,
            step.parent,
            kind="pose",
            owner=owner,
        )
        ordered.append(key)
    return tuple(ordered), tuple(cached[key] for key in ordered)


def _build_provider_plan(
    request: FamilyFrameRequest,
    prepared: PreparedFramePathRequest,
    *,
    temporal: PoseTemporalOptions,
    owner: str,
) -> _KinematicProviderPlan:
    source_path = _source_basis_path(request, prepared, owner=owner)
    support = resolve_kinematics_path_support(
        request.source,
        dst=request.dst,
        configuration=prepared.configuration,
        prepared_resolver=prepared.resolver,
        owner=f"{owner}.support",
    )
    motion_paths = _motion_basis_paths(support, prepared, owner=f"{owner}.support")
    paths = (support.path,) + (() if source_path is None else (source_path,))
    paths += tuple(path for path in motion_paths if path is not None)
    edge_keys, edge_values = _acquire_pose_values(
        paths,
        prepared,
        owner=owner,
    )
    motion_values = tuple(value for value in support.motion_payloads if value is not None)
    return _KinematicProviderPlan(
        support,
        source_path,
        motion_paths,
        edge_keys,
        edge_values,
        motion_values,
        temporal,
    )


def _restore_motion_slots(
    template: tuple[object | None, ...],
    values: tuple[object, ...],
) -> tuple[object | None, ...]:
    source = iter(values)
    return tuple(next(source) if item is not None else None for item in template)


def _evaluate_provider_plan(
    request: FamilyFrameRequest,
    plan: _KinematicProviderPlan,
    *,
    owner: str,
) -> _EvaluatedProviders:
    complete = complete_path_query(
        (*plan.edge_values, *plan.motion_values),
        query=None,
        caller=request.source,
        temporal=plan.temporal,
        owner=owner,
    )
    split = len(plan.edge_values)
    edges = tuple(zip(plan.edge_keys, complete.values[:split], strict=True))
    motion = _restore_motion_slots(plan.support.motion_payloads, complete.values[split:])
    return _EvaluatedProviders(edges, motion)


def _path_values(
    path: FramePath,
    values: tuple[tuple[_EdgeKey, object], ...],
) -> tuple[object, ...]:
    by_edge = dict(values)
    return tuple(by_edge[(step.child.id, step.parent.id)] for step in path.steps)


def _solve_pose(
    path: FramePath,
    values: tuple[tuple[_EdgeKey, object], ...],
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
):
    endpoints = resolve_path_endpoint_plan(
        prepared.configuration,
        src=path.nodes[0],
        dst=path.nodes[-1],
        owner=owner,
    )
    folded = _fold_pose_path(path, _path_values(path, values), owner=owner)
    if folded is None:
        raise RuntimeError("non-identity kinematic provider path folded to identity")
    return _finalize_pose_path(folded, endpoints, owner=owner)


def _solve_rotation(
    path: FramePath,
    values: tuple[tuple[_EdgeKey, object], ...],
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
):
    poses = _path_values(path, values)
    rotations = tuple(pose.decompose(validate=False)[1] for pose in poses)
    endpoints = resolve_path_endpoint_plan(
        prepared.configuration,
        src=path.nodes[0],
        dst=path.nodes[-1],
        owner=owner,
    )
    folded = _fold_rotation_path(path, rotations, owner=owner)
    if folded is None:
        raise RuntimeError("non-identity motion-basis path folded to identity")
    return _finalize_rotation_path(folded, endpoints, owner=owner)


def _express_motion_payload(
    payload,
    path: FramePath | None,
    plan: _KinematicProviderPlan,
    evaluated: _EvaluatedProviders,
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
):
    if payload is None or path is None:
        return payload
    solved = _solve_rotation(path, evaluated.edge_values, prepared, owner=owner)
    path_request = PreparedFramePathRequest(
        prepared.configuration,
        prepared.resolver,
        solved=solved,
    )
    return _express_motion_in_selected_basis(
        payload,
        dst=plan.support.src_parent,
        prepared_path=path_request,
        owner=owner,
    )


def _finalize_support(
    plan: _KinematicProviderPlan,
    evaluated: _EvaluatedProviders,
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
) -> KinematicsPathSupportContext:
    payloads = tuple(
        _express_motion_payload(
            payload,
            path,
            plan,
            evaluated,
            prepared,
            owner=owner,
        )
        for payload, path in zip(
            evaluated.motion_payloads,
            plan.motion_basis_paths,
            strict=True,
        )
    )
    return finalize_kinematics_path_support(plan.support, payloads)


def complete_kinematics_path_request(
    request: FamilyFrameRequest,
    prepared: PreparedFramePathRequest,
    *,
    owner: str,
) -> PreparedFramePathRequest:
    """Freeze, evaluate, and solve every provider required by one request."""
    temporal = require_path_temporal_options(
        prepared.configuration.options.temporal,
        owner=owner,
    )
    plan = _build_provider_plan(
        request,
        prepared,
        temporal=temporal,
        owner=owner,
    )
    evaluated = _evaluate_provider_plan(request, plan, owner=owner)
    support = _finalize_support(
        plan,
        evaluated,
        prepared,
        owner=f"{owner}.support",
    )
    solved = _solve_pose(plan.support.path, evaluated.edge_values, prepared, owner=owner)
    source_basis = None
    if plan.source_basis_path is not None:
        source_basis = _solve_pose(
            plan.source_basis_path,
            evaluated.edge_values,
            prepared,
            owner=owner,
        )
    return PreparedFramePathRequest(
        prepared.configuration,
        prepared.resolver,
        support,
        solved,
        source_basis,
    )


def prepare_vector_path_request(
    request: FamilyFrameRequest,
    prepared: PreparedFramePathRequest | None,
    *,
    owner: str,
    resolver_arg: str,
    with_support: bool = False,
) -> PreparedFramePathRequest:
    result = prepared or prepare_frame_path_request(
        request,
        owner=owner,
        resolver_arg=resolver_arg,
    )
    if not with_support or (result.support is not None and result.support.finalized):
        return result
    return complete_kinematics_path_request(request, result, owner=owner)


def couple_vector_path(
    source,
    value,
    request: FamilyFrameRequest,
    *,
    prepared: PreparedFramePathRequest,
    owner: str,
):
    support = prepared.support
    if support is None or not support.finalized:
        raise RuntimeError("kinematic path support was not finalized")
    return apply_vector_path_coupling(
        source,
        value,
        context=support,
        edge_pose_resolver=prepared.resolver,
        configuration=prepared.configuration,
        owner=f"{owner}.support",
    )


__all__ = [
    "complete_kinematics_path_request",
    "couple_vector_path",
    "prepare_vector_path_request",
]
