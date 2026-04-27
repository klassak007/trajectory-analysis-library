from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from tal.frames import Frame
from tal.utils.frame_schema import get_frames, set_frames

from ..metadata import (
    get_acceleration_rep,
    get_expressed_in,
    get_instantaneous_inertial,
    get_velocity_rep,
    set_expressed_in,
    set_instantaneous_inertial,
)
from ..policies.wrap import wrap_like
from .frame_owner_common import clear_framing, dst_frame_id, require_source_parent, source_expressed_in_id

if TYPE_CHECKING:
    from ..acceleration import Acceleration, AngularAcceleration, LinearAcceleration
    from ..path_solve import PathSolveOptions
    from ..velocity import AngularVelocity, LinearVelocity, Velocity


@dataclass(frozen=True)
class FamilyFrameRequest:
    source: object
    dst: Frame | str
    edge_fn: object
    opts: object | None
    validate: bool
    owner: str


def _wrap_owner_error(exc: Exception, *, owner: str) -> Exception:
    text = str(exc)
    if text.startswith(f"{owner}:"):
        return exc
    return type(exc)(f"{owner}: {text}")


def _with_relation_semantics(source, ds, *, expressed_in: str, owner: str):
    inertial = get_instantaneous_inertial(source.unsafe_data, owner=owner)
    out = set_expressed_in(ds, expressed_in=expressed_in, validate=False, owner=owner)
    return set_instantaneous_inertial(
        out,
        instantaneous_inertial=inertial,
        validate=False,
        owner=owner,
    )


def _canonicalize_vector_source_basis(
    source,
    request: FamilyFrameRequest,
    *,
    src_parent: str,
    src_child: str | None,
    src_expressed_in: str,
    solve_pose,
    pose_apply,
    owner: str,
):
    if src_expressed_in == src_parent:
        return source
    source_in_basis = wrap_like(
        source,
        set_frames(
            source.unsafe_data,
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
        opts=request.opts,
        owner=owner,
    )
    canonical = pose_apply(basis_to_parent, source_in_basis, validate=False, owner=owner)
    ds = _with_relation_semantics(
        source,
        canonical.unsafe_data,
        expressed_in=src_parent,
        owner=owner,
    )
    return wrap_like(source, ds, validate=False)


def _run_vector_to_frame(source, request: FamilyFrameRequest, *, owner: str):
    from ..path_solve import _solve_pose_path_transform_with_owner

    source._enforce_invariants(owner=owner)
    src_parent, src_child = require_source_parent(source, owner=owner)
    src_expressed_in = source_expressed_in_id(source, owner=owner)
    dst_id = dst_frame_id(request.dst, owner=owner)
    if dst_id == src_parent:
        ds = _with_relation_semantics(
            source,
            source.unsafe_data,
            expressed_in=src_expressed_in,
            owner=owner,
        )
        return wrap_like(source, ds, validate=request.validate)
    return _run_vector_to_frame_non_identity(
        source,
        request,
        src_parent=src_parent,
        src_child=src_child,
        src_expressed_in=src_expressed_in,
        solve_pose=_solve_pose_path_transform_with_owner,
        owner=owner,
    )


def _run_vector_to_frame_non_identity(
    source,
    request: FamilyFrameRequest,
    *,
    src_parent: str,
    src_child: str | None,
    src_expressed_in: str,
    solve_pose,
    owner: str,
):
    from .kinematics_path_coupling_ops import apply_vector_path_coupling
    from .kinematics_path_support_ops import resolve_kinematics_path_support
    from .pose_apply_ops import _pose_apply_with_owner

    prepared = _canonicalize_vector_source_basis(
        source,
        request,
        src_parent=src_parent,
        src_child=src_child,
        src_expressed_in=src_expressed_in,
        solve_pose=solve_pose,
        pose_apply=_pose_apply_with_owner,
        owner=owner,
    )
    support = resolve_kinematics_path_support(
        source,
        dst=request.dst,
        opts=request.opts,
        owner=f"{owner}.support",
    )
    prepared = apply_vector_path_coupling(
        source,
        prepared,
        context=support,
        edge_pose_fn=request.edge_fn,
        opts=request.opts,
        owner=f"{owner}.support",
    )
    solved = solve_pose(
        src_parent,
        request.dst,
        edge_pose_fn=request.edge_fn,
        opts=request.opts,
        owner=owner,
    )
    out = _pose_apply_with_owner(solved, prepared, validate=False, owner=owner)
    out_parent, _ = get_frames(out.unsafe_data)
    expressed = src_parent if out_parent is None else out_parent
    ds = _with_relation_semantics(source, out.unsafe_data, expressed_in=expressed, owner=owner)
    return wrap_like(source, ds, validate=request.validate)


def _run_vector_express_in(source, request: FamilyFrameRequest, *, owner: str):
    from ..path_solve import _solve_rotation_path_transform_with_owner
    from .rotation_apply_ops import _rotation_apply_with_owner

    source._enforce_invariants(owner=owner)
    src_parent, src_child = require_source_parent(source, owner=owner)
    src_expressed_in = source_expressed_in_id(source, owner=owner)
    dst_id = dst_frame_id(request.dst, owner=owner)
    if dst_id == src_expressed_in:
        ds = _with_relation_semantics(
            source,
            source.unsafe_data,
            expressed_in=dst_id,
            owner=owner,
        )
        return wrap_like(source, ds, validate=request.validate)
    solved = _solve_rotation_path_transform_with_owner(
        src_expressed_in,
        request.dst,
        edge_rotation_fn=request.edge_fn,
        opts=request.opts,
        owner=owner,
    )
    out = _rotation_apply_with_owner(
        clear_framing(solved),
        source,
        validate=False,
        owner=owner,
    )
    ds = set_frames(out.unsafe_data, parent=src_parent, child=src_child, validate=False)
    ds = _with_relation_semantics(source, ds, expressed_in=dst_id, owner=owner)
    return wrap_like(source, ds, validate=request.validate)


def _velocity_family_parts(source: Velocity, *, validate: bool, owner: str):
    rep = get_velocity_rep(source.unsafe_data, owner=owner)
    return rep, source.linear(validate=validate), source.angular(validate=validate)


def _acceleration_family_parts(source: Acceleration, *, validate: bool, owner: str):
    rep = get_acceleration_rep(source.unsafe_data, owner=owner)
    return rep, source.linear(validate=validate), source.angular(validate=validate)


def _restore_rep(value, *, source_rep: str, validate: bool):
    if source_rep == "vector6":
        return value.to_rep("vector6", validate=validate)
    return value


def _finalize_family_relation_semantics(
    *,
    source,
    out,
    expressed_in: str,
    validate: bool,
    owner: str,
):
    inertial = get_instantaneous_inertial(source.unsafe_data, owner=owner)
    ds = set_expressed_in(out.unsafe_data, expressed_in=expressed_in, validate=False, owner=owner)
    ds = set_instantaneous_inertial(
        ds,
        instantaneous_inertial=inertial,
        validate=False,
        owner=owner,
    )
    return wrap_like(out, ds, validate=validate)


def _run_family_pair_operation(
    request: FamilyFrameRequest,
    *,
    parts_resolver,
    member_runner,
    compose: Callable[..., object],
):
    rep, linear, angular = parts_resolver(
        request.source,
        validate=request.validate,
        owner=request.owner,
    )
    linear_out = member_runner(linear, request, owner=f"{request.owner}.linear")
    angular_out = member_runner(angular, request, owner=f"{request.owner}.angular")
    out = compose(linear_out, angular_out, validate=request.validate)
    out = _restore_rep(out, source_rep=rep, validate=request.validate)
    expressed_in = get_expressed_in(linear_out.unsafe_data, owner=request.owner)
    return _finalize_family_relation_semantics(
        source=request.source,
        out=out,
        expressed_in=expressed_in,
        validate=request.validate,
        owner=request.owner,
    )


def to_frame_linear_velocity(
    source: LinearVelocity,
    *,
    dst: Frame | str,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> LinearVelocity:
    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner)
    try:
        return _run_vector_to_frame(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def to_frame_angular_velocity(
    source: AngularVelocity,
    *,
    dst: Frame | str,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> AngularVelocity:
    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner)
    try:
        return _run_vector_to_frame(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def to_frame_linear_acceleration(
    source: LinearAcceleration,
    *,
    dst: Frame | str,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> LinearAcceleration:
    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner)
    try:
        return _run_vector_to_frame(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def to_frame_angular_acceleration(
    source: AngularAcceleration,
    *,
    dst: Frame | str,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> AngularAcceleration:
    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner)
    try:
        return _run_vector_to_frame(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def to_frame_velocity_family(
    source: Velocity,
    *,
    dst: Frame | str,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> Velocity:
    from ..velocity import Velocity

    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner)
    try:
        return _run_family_pair_operation(
            request,
            parts_resolver=_velocity_family_parts,
            member_runner=_run_vector_to_frame,
            compose=Velocity.from_linear_angular,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def to_frame_acceleration_family(
    source: Acceleration,
    *,
    dst: Frame | str,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> Acceleration:
    from ..acceleration import Acceleration

    request = FamilyFrameRequest(source, dst, edge_pose_fn, opts, validate, owner)
    try:
        return _run_family_pair_operation(
            request,
            parts_resolver=_acceleration_family_parts,
            member_runner=_run_vector_to_frame,
            compose=Acceleration.from_linear_angular,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def express_in_linear_velocity(
    source: LinearVelocity,
    *,
    dst: Frame | str,
    edge_rotation_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> LinearVelocity:
    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner)
    try:
        return _run_vector_express_in(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def express_in_angular_velocity(
    source: AngularVelocity,
    *,
    dst: Frame | str,
    edge_rotation_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> AngularVelocity:
    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner)
    try:
        return _run_vector_express_in(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def express_in_linear_acceleration(
    source: LinearAcceleration,
    *,
    dst: Frame | str,
    edge_rotation_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> LinearAcceleration:
    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner)
    try:
        return _run_vector_express_in(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def express_in_angular_acceleration(
    source: AngularAcceleration,
    *,
    dst: Frame | str,
    edge_rotation_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> AngularAcceleration:
    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner)
    try:
        return _run_vector_express_in(source, request, owner=owner)
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def express_in_velocity_family(
    source: Velocity,
    *,
    dst: Frame | str,
    edge_rotation_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> Velocity:
    from ..velocity import Velocity

    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner)
    try:
        return _run_family_pair_operation(
            request,
            parts_resolver=_velocity_family_parts,
            member_runner=_run_vector_express_in,
            compose=Velocity.from_linear_angular,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def express_in_acceleration_family(
    source: Acceleration,
    *,
    dst: Frame | str,
    edge_rotation_fn,
    opts: PathSolveOptions | None,
    validate: bool,
    owner: str,
) -> Acceleration:
    from ..acceleration import Acceleration

    request = FamilyFrameRequest(source, dst, edge_rotation_fn, opts, validate, owner)
    try:
        return _run_family_pair_operation(
            request,
            parts_resolver=_acceleration_family_parts,
            member_runner=_run_vector_express_in,
            compose=Acceleration.from_linear_angular,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


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
