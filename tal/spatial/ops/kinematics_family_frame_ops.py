from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tal.core.dataset_ownership import analysis_object_dataset
from tal.frames import Frame, FrameGraph

from ..metadata import (
    get_acceleration_rep,
    get_expressed_in,
    get_instantaneous_inertial,
    get_velocity_rep,
    set_expressed_in,
    set_instantaneous_inertial,
)
from ..policies.wrap import wrap_like
from .edge_resolver_ops import PreparedEdgeResolver, prepare_edge_resolver
from .frame_owner_common import (
    dst_frame_id,
    require_source_parent,
    source_expressed_in_id,
)
from .path_configuration import (
    PathConfiguration,
    SelectedPathConfiguration,
    require_strict_path_policy,
    resolve_path_configuration,
    select_path_graph,
)

if TYPE_CHECKING:
    from ..acceleration import Acceleration
    from ..path_solve import PathSolveOptions
    from ..velocity import Velocity
    from .kinematics_path_support_ops import KinematicsPathSupportContext


@dataclass(frozen=True)
class FamilyFrameRequest:
    source: object
    dst: Frame | str
    edge_fn: object
    opts: PathSolveOptions | None
    validate: bool
    owner: str
    graph: FrameGraph | None = None
    selected_configuration: SelectedPathConfiguration | None = field(
        default=None,
        repr=False,
    )
    configuration: PathConfiguration | SelectedPathConfiguration = field(init=False)

    def __post_init__(self) -> None:
        configuration = self.selected_configuration
        if configuration is None:
            configuration = resolve_path_configuration(
                self.opts,
                graph=self.graph,
                owner=self.owner,
                participants=(self.source,),
            )
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "opts", configuration.options)


@dataclass(frozen=True)
class PreparedFramePathRequest:
    configuration: SelectedPathConfiguration
    resolver: PreparedEdgeResolver
    support: KinematicsPathSupportContext | None = None
    solved: object | None = None
    source_basis: object | None = None


def prepare_frame_path_request(
    request: FamilyFrameRequest,
    *,
    owner: str,
    resolver_arg: str,
) -> PreparedFramePathRequest:
    require_strict_path_policy(request.configuration.options.strict, owner=owner)
    resolver = prepare_edge_resolver(request.edge_fn, owner=owner, arg=resolver_arg)
    configuration = select_path_graph(request.configuration, src=None, dst=request.dst, owner=owner)
    return PreparedFramePathRequest(configuration, resolver)


def velocity_family_parts(source: Velocity, *, validate: bool, owner: str):
    rep = get_velocity_rep(analysis_object_dataset(source), owner=owner)
    return rep, source.linear(validate=validate), source.angular(validate=validate)


def acceleration_family_parts(source: Acceleration, *, validate: bool, owner: str):
    rep = get_acceleration_rep(analysis_object_dataset(source), owner=owner)
    return rep, source.linear(validate=validate), source.angular(validate=validate)


def _prepare_family_path_request(
    request: FamilyFrameRequest,
    *,
    operation: str,
) -> PreparedFramePathRequest | None:
    src_parent, _ = require_source_parent(request.source, owner=request.owner)
    current = src_parent
    resolver_arg = "edge_pose_fn"
    if operation == "express_in":
        current = source_expressed_in_id(request.source, owner=request.owner)
        resolver_arg = "edge_rotation_fn"
    if dst_frame_id(request.dst, owner=request.owner) == current:
        return None
    prepared = prepare_frame_path_request(request, owner=request.owner, resolver_arg=resolver_arg)
    if operation == "to_frame":
        from .kinematics_dynamic_path_ops import complete_kinematics_path_request

        return complete_kinematics_path_request(request, prepared, owner=request.owner)
    solved = _solve_family_path(request, prepared, operation=operation)
    return PreparedFramePathRequest(prepared.configuration, prepared.resolver, solved=solved)


def _solve_family_path(
    request: FamilyFrameRequest,
    prepared: PreparedFramePathRequest,
    *,
    operation: str,
):
    from ..path_solve import (
        _solve_pose_path_transform_with_owner,
        _solve_rotation_path_transform_with_owner,
    )

    src_parent, _ = require_source_parent(request.source, owner=request.owner)
    source = src_parent if operation == "to_frame" else source_expressed_in_id(request.source, owner=request.owner)
    solve = _solve_pose_path_transform_with_owner if operation == "to_frame" else _solve_rotation_path_transform_with_owner
    edge_arg = {"edge_pose_fn": request.edge_fn} if operation == "to_frame" else {"edge_rotation_fn": request.edge_fn}
    return solve(
        source,
        request.dst,
        configuration=prepared.configuration,
        prepared_resolver=prepared.resolver,
        caller=request.source,
        owner=request.owner,
        **edge_arg,
    )


def _restore_rep(value, *, source_rep: str, validate: bool):
    if source_rep == "vector6":
        return value.to_rep("vector6", validate=validate)
    return value


def _finalize_family_relation(source, out, *, expressed_in: str, validate: bool, owner: str):
    inertial = get_instantaneous_inertial(analysis_object_dataset(source), owner=owner)
    ds = set_expressed_in(analysis_object_dataset(out), expressed_in=expressed_in, validate=False, owner=owner)
    ds = set_instantaneous_inertial(
        ds,
        instantaneous_inertial=inertial,
        validate=False,
        owner=owner,
    )
    return wrap_like(out, ds, validate=validate)


def run_family_pair_operation(
    request: FamilyFrameRequest,
    *,
    parts_resolver,
    member_runner,
    compose: Callable[..., object],
    operation: str,
    prepared_path: PreparedFramePathRequest | None = None,
):
    rep, linear, angular = parts_resolver(
        request.source,
        validate=request.validate,
        owner=request.owner,
    )
    if prepared_path is None:
        prepared_path = _prepare_family_path_request(request, operation=operation)
    linear_out = member_runner(
        linear, request, owner=f"{request.owner}.linear", prepared_path=prepared_path,
    )
    angular_out = member_runner(
        angular, request, owner=f"{request.owner}.angular", prepared_path=prepared_path,
    )
    out = compose(linear_out, angular_out, validate=request.validate)
    out = _restore_rep(out, source_rep=rep, validate=request.validate)
    expressed_in = get_expressed_in(analysis_object_dataset(linear_out), owner=request.owner)
    return _finalize_family_relation(
        request.source,
        out,
        expressed_in=expressed_in,
        validate=request.validate,
        owner=request.owner,
    )


__all__ = [
    "FamilyFrameRequest",
    "PreparedFramePathRequest",
    "acceleration_family_parts",
    "prepare_frame_path_request",
    "run_family_pair_operation",
    "velocity_family_parts",
]
