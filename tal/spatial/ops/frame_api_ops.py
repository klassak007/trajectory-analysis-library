from __future__ import annotations

from typing import TYPE_CHECKING

from tal.frames import Frame
from tal.utils.frame_schema import get_frames

from tal.core.dataset_ownership import analysis_object_dataset

from ..policies.wrap import wrap_like

if TYPE_CHECKING:
    from ..path_solve import PathSolveOptions
    from ..pose import Pose
    from ..position import Position
    from ..rotation import Rotation


def _require_position_parent(position: "Position", *, owner: str) -> str:
    parent, _ = get_frames(analysis_object_dataset(position))
    if parent is None:
        raise ValueError(f"{owner}: Position.to_frame requires framed input with parent frame id.")
    return parent


def _coerce_frame_id(value: object) -> str | None:
    if isinstance(value, Frame):
        return value.id
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def position_to_frame(
    position: "Position",
    dst: Frame | str,
    *,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    validate: bool,
) -> "Position":
    owner = "spatial.position.to_frame"
    from ..path_solve import _solve_pose_path_transform_with_owner
    from .pose_apply_ops import _pose_apply_with_owner

    position._enforce_invariants(owner=owner)
    source_parent = _require_position_parent(position, owner=owner)
    destination = _coerce_frame_id(dst)
    solved = _solve_pose_path_transform_with_owner(
        source_parent,
        dst,
        edge_pose_fn=edge_pose_fn,
        opts=opts,
        owner=owner,
    )
    if destination == source_parent:
        return wrap_like(position, analysis_object_dataset(position), validate=validate)
    result = _pose_apply_with_owner(
        solved,
        position,
        validate=False,
        owner=owner,
    )
    return wrap_like(position, analysis_object_dataset(result), validate=validate)


def rotation_class_solve_path_transform(
    cls: type["Rotation"],
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_rotation_fn,
    opts: PathSolveOptions | None,
    validate: bool,
) -> "Rotation":
    owner = "spatial.rotation.solve_path_transform"
    from ..path_solve import _solve_rotation_path_transform_with_owner

    solved = _solve_rotation_path_transform_with_owner(
        src,
        dst,
        edge_rotation_fn=edge_rotation_fn,
        opts=opts,
        owner=owner,
    )
    if validate:
        return cls._from_validated(analysis_object_dataset(solved))
    return cls._from_unvalidated(analysis_object_dataset(solved))


def pose_class_solve_path_transform(
    cls: type["Pose"],
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_pose_fn,
    opts: PathSolveOptions | None,
    validate: bool,
) -> "Pose":
    owner = "spatial.pose.solve_path_transform"
    from ..path_solve import _solve_pose_path_transform_with_owner

    solved = _solve_pose_path_transform_with_owner(
        src,
        dst,
        edge_pose_fn=edge_pose_fn,
        opts=opts,
        owner=owner,
    )
    if validate:
        return cls._from_validated(analysis_object_dataset(solved))
    return cls._from_unvalidated(analysis_object_dataset(solved))


__all__ = [
    "pose_class_solve_path_transform",
    "position_to_frame",
    "rotation_class_solve_path_transform",
]
