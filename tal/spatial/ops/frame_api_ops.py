from __future__ import annotations

from typing import TYPE_CHECKING

from tal.core.dataset_ownership import analysis_object_dataset
from tal.frames import Frame, FrameGraph
from tal.utils.frame_schema import get_frames

from ..association import (
    SpatialAssociationPlan,
    attach_spatial_association,
    finalize_spatial_from_source,
)
from ..policies.wrap import wrap_like
from .frame_owner_common import dst_frame_id
from .path_configuration import (
    resolve_path_configuration,
    select_identity_graph,
    select_path_graph,
)

if TYPE_CHECKING:
    from ..path_solve import PathSolveOptions
    from ..pose import Pose
    from ..position import Position
    from ..rotation import Rotation


def _require_position_parent(position: Position, *, owner: str) -> str:
    parent, _ = get_frames(analysis_object_dataset(position))
    if parent is None:
        raise ValueError(f"{owner}: Position.to_frame requires framed input with parent frame id.")
    return parent


def _identity_position_to_frame(
    position: Position,
    dst: Frame | str,
    *,
    source_parent: str,
    configuration,
    validate: bool,
    owner: str,
) -> Position:
    selected = select_identity_graph(
        configuration,
        src=source_parent,
        dst=dst,
        owner=owner,
    )
    result = wrap_like(position, analysis_object_dataset(position), validate=validate)
    return attach_spatial_association(result, SpatialAssociationPlan(selected))


def _transform_position_to_frame(
    position: Position,
    dst: Frame | str,
    *,
    source_parent: str,
    edge_pose_fn,
    configuration,
    validate: bool,
    owner: str,
) -> Position:
    from ..path_solve import _solve_pose_path_transform_with_owner
    from .pose_apply_ops import _pose_apply_with_owner

    selected = select_path_graph(configuration, src=source_parent, dst=dst, owner=owner)
    association = SpatialAssociationPlan(selected.graph)
    solved = _solve_pose_path_transform_with_owner(
        source_parent,
        dst,
        edge_pose_fn=edge_pose_fn,
        configuration=selected,
        caller=position,
        owner=owner,
    )
    result = _pose_apply_with_owner(
        solved,
        position,
        validate=False,
        owner=owner,
        association=association,
    )
    out = wrap_like(position, analysis_object_dataset(result), validate=validate)
    return attach_spatial_association(out, association)


def position_to_frame(
    position: Position,
    dst: Frame | str,
    *,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
) -> Position:
    owner = "spatial.position.to_frame"
    configuration = resolve_path_configuration(
        opts,
        graph=graph,
        owner=owner,
        participants=(position,),
    )
    destination = dst_frame_id(dst, owner=owner)
    position._enforce_invariants(owner=owner)
    source_parent = _require_position_parent(position, owner=owner)
    if destination == source_parent:
        return _identity_position_to_frame(
            position,
            dst,
            source_parent=source_parent,
            configuration=configuration,
            validate=validate,
            owner=owner,
        )
    return _transform_position_to_frame(
        position,
        dst,
        source_parent=source_parent,
        edge_pose_fn=edge_pose_fn,
        configuration=configuration,
        validate=validate,
        owner=owner,
    )


def rotation_class_solve_path_transform(
    cls: type[Rotation],
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    query=None,
    opts: PathSolveOptions | None,
    validate: bool,
) -> Rotation:
    owner = "spatial.rotation.solve_path_transform"
    from ..path_solve import _solve_rotation_path_transform_with_owner

    solved = _solve_rotation_path_transform_with_owner(
        src,
        dst,
        edge_rotation_fn=edge_rotation_fn,
        graph=graph,
        query=query,
        opts=opts,
        owner=owner,
    )
    return finalize_spatial_from_source(
        solved,
        cls,
        analysis_object_dataset(solved),
        validate=validate,
    )


def pose_class_solve_path_transform(
    cls: type[Pose],
    src: Frame | str,
    dst: Frame | str,
    *,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    query=None,
    opts: PathSolveOptions | None,
    validate: bool,
) -> Pose:
    owner = "spatial.pose.solve_path_transform"
    from ..path_solve import _solve_pose_path_transform_with_owner

    solved = _solve_pose_path_transform_with_owner(
        src,
        dst,
        edge_pose_fn=edge_pose_fn,
        graph=graph,
        query=query,
        opts=opts,
        owner=owner,
    )
    return finalize_spatial_from_source(
        solved,
        cls,
        analysis_object_dataset(solved),
        validate=validate,
    )


__all__ = [
    "pose_class_solve_path_transform",
    "position_to_frame",
    "rotation_class_solve_path_transform",
]
