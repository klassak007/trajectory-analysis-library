from __future__ import annotations

from typing import TYPE_CHECKING

from tal.frames import Frame
from tal.utils.frame_schema import get_frames, set_frames

from tal.core.dataset_ownership import analysis_object_dataset

from ..metadata import get_pose_rep, get_rotation_rep, set_expressed_in
from ..policies.wrap import wrap_like
from .frame_owner_common import clear_framing, dst_frame_id, require_source_parent, source_expressed_in_id

if TYPE_CHECKING:
    from ..path_solve import PathSolveOptions
    from ..pose import Pose
    from ..position import Position
    from ..rotation import Rotation


def _finalize_same_relation(source, out_ds, *, expressed_in: str, validate: bool, owner: str):
    parent, child = get_frames(analysis_object_dataset(source))
    ds = set_frames(out_ds, parent=parent, child=child, validate=False)
    ds = set_expressed_in(
        ds,
        expressed_in=expressed_in,
        validate=False,
        owner=owner,
    )
    return wrap_like(source, ds, validate=validate)


def position_express_in(
    position: "Position",
    dst: Frame | str,
    *,
    edge_rotation_fn,
    opts: "PathSolveOptions | None",
    validate: bool,
) -> "Position":
    owner = "spatial.position.express_in"
    from ..path_solve import (
        _coerce_path_solve_options,
        _solve_rotation_path_transform_with_owner,
    )
    from .rotation_apply_ops import _rotation_apply_with_owner

    opts = _coerce_path_solve_options(opts, owner=owner)
    position._enforce_invariants(owner=owner)
    _ = require_source_parent(position, owner=owner)
    src_expressed_in = source_expressed_in_id(position, owner=owner)
    dst_id = dst_frame_id(dst, owner=owner)
    if dst_id == src_expressed_in:
        return _finalize_same_relation(
            position,
            analysis_object_dataset(position),
            expressed_in=dst_id,
            validate=validate,
            owner=owner,
        )
    basis = _solve_rotation_path_transform_with_owner(
        src_expressed_in,
        dst,
        edge_rotation_fn=edge_rotation_fn,
        opts=opts,
        owner=owner,
    )
    rotated = _rotation_apply_with_owner(
        clear_framing(basis),
        position,
        validate=False,
        owner=owner,
    )
    return _finalize_same_relation(
        position,
        analysis_object_dataset(rotated),
        expressed_in=dst_id,
        validate=validate,
        owner=owner,
    )


def rotation_express_in(
    rotation: "Rotation",
    dst: Frame | str,
    *,
    edge_rotation_fn,
    opts: "PathSolveOptions | None",
    validate: bool,
) -> "Rotation":
    owner = "spatial.rotation.express_in"
    from ..path_solve import (
        _coerce_path_solve_options,
        _solve_rotation_path_transform_with_owner,
    )
    from ..rotation import _rotation_compose_with_owner, _rotation_inverse_with_owner

    opts = _coerce_path_solve_options(opts, owner=owner)
    rotation._enforce_invariants(owner=owner)
    _ = require_source_parent(rotation, owner=owner)
    src_expressed_in = source_expressed_in_id(rotation, owner=owner)
    dst_id = dst_frame_id(dst, owner=owner)
    source = analysis_object_dataset(rotation)
    src_rep = get_rotation_rep(source, owner=owner)
    if dst_id == src_expressed_in:
        return _finalize_same_relation(
            rotation, source, expressed_in=dst_id, validate=validate, owner=owner,
        )
    basis = clear_framing(
        _solve_rotation_path_transform_with_owner(
            src_expressed_in,
            dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            owner=owner,
        ).as_quat(validate=False)
    )
    value = clear_framing(rotation.as_quat(validate=False))
    left = _rotation_compose_with_owner(basis, value, validate=False, owner=owner)
    right = _rotation_inverse_with_owner(basis, validate=False, owner=owner)
    out = _rotation_compose_with_owner(left, right, validate=False, owner=owner)
    if src_rep == "matrix":
        out = out.as_matrix(validate=False)
    return _finalize_same_relation(
        rotation,
        analysis_object_dataset(out),
        expressed_in=dst_id,
        validate=validate,
        owner=owner,
    )


def pose_express_in(
    pose: "Pose",
    dst: Frame | str,
    *,
    edge_pose_fn,
    opts: "PathSolveOptions | None",
    validate: bool,
) -> "Pose":
    owner = "spatial.pose.express_in"
    from ..path_solve import _coerce_path_solve_options

    opts = _coerce_path_solve_options(opts, owner=owner)
    pose._enforce_invariants(owner=owner)
    _ = require_source_parent(pose, owner=owner)
    src_expressed_in = source_expressed_in_id(pose, owner=owner)
    dst_id = dst_frame_id(dst, owner=owner)
    source = analysis_object_dataset(pose)
    src_rep = get_pose_rep(source, owner=owner)
    if dst_id == src_expressed_in:
        return _finalize_same_relation(
            pose,
            source,
            expressed_in=dst_id,
            validate=validate,
            owner=owner,
        )
    basis = _solve_pose_basis_rotation(
        src_expressed_in,
        dst,
        edge_pose_fn=edge_pose_fn,
        opts=opts,
        owner=owner,
    )
    out = _reexpress_pose_components(pose, basis, owner=owner)
    if src_rep == "matrix":
        out = out.as_matrix(validate=False)
    return _finalize_same_relation(
        pose,
        analysis_object_dataset(out),
        expressed_in=dst_id,
        validate=validate,
        owner=owner,
    )


def _solve_pose_basis_rotation(
    src: str,
    dst: Frame | str,
    *,
    edge_pose_fn,
    opts: "PathSolveOptions | None",
    owner: str,
):
    from ..path_solve import _solve_pose_path_transform_with_owner

    basis_pose = _solve_pose_path_transform_with_owner(
        src,
        dst,
        edge_pose_fn=edge_pose_fn,
        opts=opts,
        owner=owner,
    )
    _, basis_rotation = basis_pose.decompose(validate=False)
    return clear_framing(basis_rotation.as_quat(validate=False))


def _reexpress_pose_components(pose: "Pose", basis, *, owner: str):
    from ..pose import Pose
    from ..rotation import _rotation_compose_with_owner, _rotation_inverse_with_owner
    from .rotation_apply_ops import _rotation_apply_with_owner

    translation, rotation = pose.as_components(validate=False).decompose(validate=False)
    translation_out = _rotation_apply_with_owner(basis, translation, validate=False, owner=owner)
    rotation_quat = clear_framing(rotation.as_quat(validate=False))
    left = _rotation_compose_with_owner(basis, rotation_quat, validate=False, owner=owner)
    right = _rotation_inverse_with_owner(basis, validate=False, owner=owner)
    rotation_out = _rotation_compose_with_owner(left, right, validate=False, owner=owner)
    return Pose.from_components(rotation_out, translation_out, validate=False)


__all__ = [
    "pose_express_in",
    "position_express_in",
    "rotation_express_in",
]
