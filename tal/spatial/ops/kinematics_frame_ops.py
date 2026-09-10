from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from tal.core.dataset_ownership import analysis_object_dataset
from tal.frames import Frame, FrameGraph
from tal.utils.frame_schema import set_frames

from ..association import (
    SpatialAssociationPlan,
    attach_spatial_association,
)
from ..policies.wrap import wrap_like
from .edge_resolver_ops import (
    PreparedEdgeResolver,
    _PathCallbackSignatureError,
    _raise_public_signature_error,
)
from .frame_owner_common import (
    clear_framing,
    dst_frame_id,
    require_source_parent,
    source_expressed_in_id,
)
from .kinematics_family_frame_ops import (
    FamilyFrameRequest,
    PreparedFramePathRequest,
    acceleration_family_parts,
    prepare_frame_path_request,
    run_family_pair_operation,
    velocity_family_parts,
)
from .kinematics_frame_finalize_ops import (
    finalize_identity_vector_expression as _finalize_identity_vector_expression,
)
from .kinematics_frame_finalize_ops import (
    finalize_vector_expression as _finalize_vector_expression,
)
from .kinematics_frame_finalize_ops import (
    finalize_vector_frame_result as _finalize_vector_frame_result,
)
from .kinematics_frame_finalize_ops import (
    with_relation_semantics as _with_relation_semantics,
)
from .path_configuration import SelectedPathConfiguration, select_identity_graph

if TYPE_CHECKING:
    from ..acceleration import Acceleration, AngularAcceleration, LinearAcceleration
    from ..path_solve import PathSolveOptions
    from ..velocity import AngularVelocity, LinearVelocity, Velocity


def _raise_owner_error(exc: Exception, *, owner: str) -> NoReturn:
    if isinstance(exc, _PathCallbackSignatureError):
        _raise_public_signature_error(exc, owner=owner)
    text = str(exc)
    if text.startswith(f"{owner}:"):
        raise exc
    raise type(exc)(f"{owner}: {text}") from exc


def _canonicalize_vector_source_basis(
    source,
    request: FamilyFrameRequest,
    *,
    src_parent: str,
    src_child: str | None,
    src_expressed_in: str,
    configuration: SelectedPathConfiguration,
    prepared_resolver: PreparedEdgeResolver,
    solve_pose,
    pose_apply,
    owner: str,
):
    if src_expressed_in == src_parent:
        return source
    source_in_basis = wrap_like(
        source,
        set_frames(
            analysis_object_dataset(source),
            parent=src_expressed_in,
            child=src_child,
            validate=False,
        ),
        validate=False,
    )
    basis_to_parent = solve_pose(
        src_expressed_in,
        src_parent,
        edge_pose_fn=request.edge_fn,
        configuration=configuration,
        prepared_resolver=prepared_resolver,
        owner=owner,
    )
    canonical = pose_apply(
        basis_to_parent,
        source_in_basis,
        validate=False,
        owner=owner,
        association=SpatialAssociationPlan(configuration.graph),
    )
    ds = _with_relation_semantics(
        source,
        analysis_object_dataset(canonical),
        expressed_in=src_parent,
        owner=owner,
    )
    return wrap_like(source, ds, validate=False)


def _run_vector_to_frame(
    source,
    request: FamilyFrameRequest,
    *,
    owner: str,
    prepared_path: PreparedFramePathRequest | None = None,
):
    from ..path_solve import _solve_pose_path_transform_with_owner

    source._enforce_invariants(owner=owner)
    src_parent, src_child = require_source_parent(source, owner=owner)
    src_expressed_in = source_expressed_in_id(source, owner=owner)
    dst_id = dst_frame_id(request.dst, owner=owner)
    if dst_id == src_parent:
        selected = select_identity_graph(
            request.configuration,
            src=src_parent,
            dst=request.dst,
            owner=owner,
        )
        ds = _with_relation_semantics(
            source,
            analysis_object_dataset(source),
            expressed_in=src_expressed_in,
            owner=owner,
        )
        result = wrap_like(source, ds, validate=request.validate)
        return attach_spatial_association(result, SpatialAssociationPlan(selected))
    return _run_vector_to_frame_non_identity(
        source,
        request,
        src_parent=src_parent,
        src_child=src_child,
        src_expressed_in=src_expressed_in,
        solve_pose=_solve_pose_path_transform_with_owner,
        prepared_path=prepared_path,
        owner=owner,
    )


def _couple_vector_path(
    source,
    prepared,
    request: FamilyFrameRequest,
    *,
    configuration: SelectedPathConfiguration,
    prepared_resolver: PreparedEdgeResolver,
    owner: str,
):
    from .kinematics_path_coupling_ops import apply_vector_path_coupling
    from .kinematics_path_support_ops import resolve_kinematics_path_support

    support = resolve_kinematics_path_support(
        source,
        dst=request.dst,
        configuration=configuration,
        prepared_resolver=prepared_resolver,
        owner=f"{owner}.support",
    )
    return apply_vector_path_coupling(
        source,
        prepared,
        context=support,
        edge_pose_resolver=prepared_resolver,
        configuration=configuration,
        owner=f"{owner}.support",
    )


def _prepared_path_or_resolve(request, prepared_path, *, owner: str, resolver_arg: str):
    if prepared_path is not None:
        return prepared_path
    return prepare_frame_path_request(request, owner=owner, resolver_arg=resolver_arg)


def _apply_solved_vector_path(
    source,
    prepared,
    request,
    *,
    src_parent: str,
    configuration: SelectedPathConfiguration,
    prepared_resolver: PreparedEdgeResolver,
    solve_pose,
    owner: str,
):
    from .pose_apply_ops import _pose_apply_with_owner

    solved = solve_pose(
        src_parent,
        request.dst,
        edge_pose_fn=request.edge_fn,
        configuration=configuration,
        prepared_resolver=prepared_resolver,
        owner=owner,
    )
    out = _pose_apply_with_owner(
        solved,
        prepared,
        validate=False,
        owner=owner,
        association=SpatialAssociationPlan(configuration.graph),
    )
    return _finalize_vector_frame_result(
        source,
        out,
        src_parent=src_parent,
        request=request,
        owner=owner,
        graph=configuration.graph,
    )


def _run_vector_to_frame_non_identity(
    source,
    request: FamilyFrameRequest,
    *,
    src_parent: str,
    src_child: str | None,
    src_expressed_in: str,
    solve_pose,
    prepared_path: PreparedFramePathRequest | None,
    owner: str,
):
    from .pose_apply_ops import _pose_apply_with_owner

    path_request = _prepared_path_or_resolve(
        request, prepared_path, owner=owner, resolver_arg="edge_pose_fn",
    )
    configuration = path_request.configuration
    prepared_resolver = path_request.resolver
    prepared = _canonicalize_vector_source_basis(
        source,
        request,
        src_parent=src_parent,
        src_child=src_child,
        src_expressed_in=src_expressed_in,
        solve_pose=solve_pose,
        pose_apply=_pose_apply_with_owner,
        configuration=configuration,
        prepared_resolver=prepared_resolver,
        owner=owner,
    )
    prepared = _couple_vector_path(
        source,
        prepared,
        request,
        configuration=configuration,
        prepared_resolver=prepared_resolver,
        owner=owner,
    )
    return _apply_solved_vector_path(
        source,
        prepared,
        request,
        src_parent=src_parent,
        configuration=configuration,
        prepared_resolver=prepared_resolver,
        solve_pose=solve_pose,
        owner=owner,
    )


def _run_vector_express_in(
    source,
    request: FamilyFrameRequest,
    *,
    owner: str,
    prepared_path: PreparedFramePathRequest | None = None,
):
    source._enforce_invariants(owner=owner)
    src_parent, src_child = require_source_parent(source, owner=owner)
    src_expressed_in = source_expressed_in_id(source, owner=owner)
    dst_id = dst_frame_id(request.dst, owner=owner)
    if dst_id == src_expressed_in:
        selected = select_identity_graph(
            request.configuration,
            src=src_expressed_in,
            dst=request.dst,
            owner=owner,
        )
        return _finalize_identity_vector_expression(
            source,
            destination=dst_id,
            request=request,
            graph=selected,
            owner=owner,
        )
    return _apply_vector_expression(
        source,
        request,
        relation=(src_parent, src_child),
        source_basis=src_expressed_in,
        destination=dst_id,
        prepared_path=prepared_path,
        owner=owner,
    )


def _apply_vector_expression(
    source,
    request: FamilyFrameRequest,
    *,
    relation: tuple[str, str | None],
    source_basis: str,
    destination: str,
    prepared_path: PreparedFramePathRequest | None,
    owner: str,
):
    from ..path_solve import _solve_rotation_path_transform_with_owner
    from .rotation_apply_ops import _rotation_apply_with_owner

    path_request = _prepared_path_or_resolve(
        request, prepared_path, owner=owner, resolver_arg="edge_rotation_fn"
    )
    solved = _solve_rotation_path_transform_with_owner(
        source_basis,
        request.dst,
        edge_rotation_fn=request.edge_fn,
        configuration=path_request.configuration,
        prepared_resolver=path_request.resolver,
        owner=owner,
    )
    out = _rotation_apply_with_owner(
        clear_framing(solved, owner=owner),
        source,
        validate=False,
        owner=owner,
        association=SpatialAssociationPlan(path_request.configuration.graph),
    )
    return _finalize_vector_expression(
        source, out,
        relation=relation,
        destination=destination,
        request=request,
        graph=path_request.configuration.graph, owner=owner,
    )


def to_frame_linear_velocity(
    source: LinearVelocity,
    *,
    dst: Frame | str,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> LinearVelocity:
    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner, graph)
    try:
        return _run_vector_to_frame(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def to_frame_angular_velocity(
    source: AngularVelocity,
    *,
    dst: Frame | str,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> AngularVelocity:
    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner, graph)
    try:
        return _run_vector_to_frame(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def to_frame_linear_acceleration(
    source: LinearAcceleration,
    *,
    dst: Frame | str,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> LinearAcceleration:
    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner, graph)
    try:
        return _run_vector_to_frame(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def to_frame_angular_acceleration(
    source: AngularAcceleration,
    *,
    dst: Frame | str,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> AngularAcceleration:
    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner, graph)
    try:
        return _run_vector_to_frame(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def to_frame_velocity_family(
    source: Velocity,
    *,
    dst: Frame | str,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> Velocity:
    from ..velocity import Velocity

    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner, graph)
    try:
        return run_family_pair_operation(
            request,
            parts_resolver=velocity_family_parts,
            member_runner=_run_vector_to_frame,
            compose=Velocity.from_linear_angular,
            operation="to_frame",
        )
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def to_frame_acceleration_family(
    source: Acceleration,
    *,
    dst: Frame | str,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> Acceleration:
    from ..acceleration import Acceleration

    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner, graph)
    try:
        return run_family_pair_operation(
            request,
            parts_resolver=acceleration_family_parts,
            member_runner=_run_vector_to_frame,
            compose=Acceleration.from_linear_angular,
            operation="to_frame",
        )
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def express_in_linear_velocity(
    source: LinearVelocity,
    *,
    dst: Frame | str,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> LinearVelocity:
    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner, graph)
    try:
        return _run_vector_express_in(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def express_in_angular_velocity(
    source: AngularVelocity,
    *,
    dst: Frame | str,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> AngularVelocity:
    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner, graph)
    try:
        return _run_vector_express_in(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def express_in_linear_acceleration(
    source: LinearAcceleration,
    *,
    dst: Frame | str,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> LinearAcceleration:
    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner, graph)
    try:
        return _run_vector_express_in(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def express_in_angular_acceleration(
    source: AngularAcceleration,
    *,
    dst: Frame | str,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> AngularAcceleration:
    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner, graph)
    try:
        return _run_vector_express_in(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def express_in_velocity_family(
    source: Velocity,
    *,
    dst: Frame | str,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> Velocity:
    from ..velocity import Velocity

    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner, graph)
    try:
        return run_family_pair_operation(
            request,
            parts_resolver=velocity_family_parts,
            member_runner=_run_vector_express_in,
            compose=Velocity.from_linear_angular,
            operation="express_in",
        )
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


def express_in_acceleration_family(
    source: Acceleration,
    *,
    dst: Frame | str,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> Acceleration:
    from ..acceleration import Acceleration

    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner, graph)
    try:
        return run_family_pair_operation(
            request,
            parts_resolver=acceleration_family_parts,
            member_runner=_run_vector_express_in,
            compose=Acceleration.from_linear_angular,
            operation="express_in",
        )
    except (TypeError, ValueError) as exc:
        _raise_owner_error(exc, owner=owner)


__all__ = [
    "express_in_acceleration_family",
    "express_in_angular_acceleration",
    "express_in_angular_velocity",
    "express_in_linear_acceleration",
    "express_in_linear_velocity",
    "express_in_velocity_family",
    "to_frame_acceleration_family",
    "to_frame_angular_acceleration",
    "to_frame_angular_velocity",
    "to_frame_linear_acceleration",
    "to_frame_linear_velocity",
    "to_frame_velocity_family",
]
