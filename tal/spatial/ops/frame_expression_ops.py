from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tal.core.dataset_ownership import analysis_object_dataset
from tal.frames import Frame, FrameGraph
from tal.utils.frame_schema import get_frames, set_frames

from ..association import (
    SpatialAssociationPlan,
    attach_spatial_association,
)
from ..metadata import get_pose_rep, get_rotation_rep, set_expressed_in
from ..policies.wrap import wrap_like
from .frame_owner_common import (
    clear_framing,
    dst_frame_id,
    require_source_parent,
    source_expressed_in_id,
)
from .path_configuration import (
    PathConfiguration,
    SelectedPathConfiguration,
    resolve_path_configuration,
    select_identity_graph,
    select_path_graph,
)

if TYPE_CHECKING:
    from ..path_solve import PathSolveOptions
    from ..pose import Pose
    from ..position import Position
    from ..rotation import Rotation


@dataclass(frozen=True)
class _ExpressionContext:
    configuration: PathConfiguration
    source_basis: str
    destination: str


def _finalize_same_relation(
    source,
    out_ds,
    *,
    expressed_in: str,
    validate: bool,
    owner: str,
    association: SpatialAssociationPlan,
):
    parent, child = get_frames(analysis_object_dataset(source))
    ds = set_frames(out_ds, parent=parent, child=child, validate=False)
    ds = set_expressed_in(
        ds,
        expressed_in=expressed_in,
        validate=False,
        owner=owner,
    )
    result = wrap_like(source, ds, validate=validate)
    return attach_spatial_association(result, association)


def _prepare_expression(
    source,
    dst: Frame | str,
    *,
    graph: FrameGraph | None,
    opts: PathSolveOptions | None,
    owner: str,
) -> _ExpressionContext:
    configuration = resolve_path_configuration(
        opts,
        graph=graph,
        owner=owner,
        participants=(source,),
    )
    source._enforce_invariants(owner=owner)
    _ = require_source_parent(source, owner=owner)
    return _ExpressionContext(
        configuration=configuration,
        source_basis=source_expressed_in_id(source, owner=owner),
        destination=dst_frame_id(dst, owner=owner),
    )


def _identity_expression(source, dst, context, *, validate: bool, owner: str):
    selected = select_identity_graph(
        context.configuration,
        src=context.source_basis,
        dst=dst,
        owner=owner,
    )
    return _finalize_same_relation(
        source,
        analysis_object_dataset(source),
        expressed_in=context.destination,
        validate=validate,
        owner=owner,
        association=SpatialAssociationPlan(selected),
    )


def position_express_in(
    position: Position,
    dst: Frame | str,
    *,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
) -> Position:
    owner = "spatial.position.express_in"
    from ..path_solve import (
        _solve_rotation_path_transform_with_owner,
    )
    from .rotation_apply_ops import _rotation_apply_with_owner

    context = _prepare_expression(position, dst, graph=graph, opts=opts, owner=owner)
    if context.destination == context.source_basis:
        return _identity_expression(position, dst, context, validate=validate, owner=owner)
    selected_config = select_path_graph(
        context.configuration,
        src=context.source_basis,
        dst=dst,
        owner=owner,
    )
    basis = _solve_rotation_path_transform_with_owner(
        context.source_basis,
        dst,
        edge_rotation_fn=edge_rotation_fn,
        configuration=selected_config,
        owner=owner,
    )
    rotated = _rotation_apply_with_owner(
        clear_framing(basis, owner=owner),
        position,
        validate=False,
        owner=owner,
        association=SpatialAssociationPlan(selected_config.graph),
    )
    return _finalize_same_relation(
        position,
        analysis_object_dataset(rotated),
        expressed_in=context.destination,
        validate=validate,
        owner=owner,
        association=SpatialAssociationPlan(selected_config.graph),
    )


def _conjugate_rotation(rotation, basis, *, association, owner: str):
    from ..rotation import _rotation_compose_with_owner, _rotation_inverse_with_owner

    value = clear_framing(rotation.as_quat(validate=False), owner=owner)
    left = _rotation_compose_with_owner(
        basis,
        value,
        validate=False,
        owner=owner,
        association=association,
    )
    right = _rotation_inverse_with_owner(basis, validate=False, owner=owner)
    return _rotation_compose_with_owner(
        left,
        right,
        validate=False,
        owner=owner,
        association=association,
    )


def rotation_express_in(
    rotation: Rotation,
    dst: Frame | str,
    *,
    edge_rotation_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
) -> Rotation:
    owner = "spatial.rotation.express_in"
    from ..path_solve import (
        _solve_rotation_path_transform_with_owner,
    )

    context = _prepare_expression(rotation, dst, graph=graph, opts=opts, owner=owner)
    source = analysis_object_dataset(rotation)
    src_rep = get_rotation_rep(source, owner=owner)
    if context.destination == context.source_basis:
        return _identity_expression(rotation, dst, context, validate=validate, owner=owner)
    selected_config = select_path_graph(
        context.configuration,
        src=context.source_basis,
        dst=dst,
        owner=owner,
    )
    basis = clear_framing(
        _solve_rotation_path_transform_with_owner(
            context.source_basis,
            dst,
            edge_rotation_fn=edge_rotation_fn,
            configuration=selected_config,
            owner=owner,
        ).as_quat(validate=False),
        owner=owner,
    )
    association = SpatialAssociationPlan(selected_config.graph)
    out = _conjugate_rotation(rotation, basis, association=association, owner=owner)
    if src_rep == "matrix":
        out = out.as_matrix(validate=False)
    return _finalize_same_relation(
        rotation,
        analysis_object_dataset(out),
        expressed_in=context.destination,
        validate=validate,
        owner=owner,
        association=SpatialAssociationPlan(selected_config.graph),
    )


def pose_express_in(
    pose: Pose,
    dst: Frame | str,
    *,
    edge_pose_fn=None,
    graph: FrameGraph | None = None,
    opts: PathSolveOptions | None,
    validate: bool,
) -> Pose:
    owner = "spatial.pose.express_in"

    context = _prepare_expression(pose, dst, graph=graph, opts=opts, owner=owner)
    source = analysis_object_dataset(pose)
    src_rep = get_pose_rep(source, owner=owner)
    if context.destination == context.source_basis:
        return _identity_expression(pose, dst, context, validate=validate, owner=owner)
    selected_config = select_path_graph(
        context.configuration,
        src=context.source_basis,
        dst=dst,
        owner=owner,
    )
    basis = _solve_pose_basis_rotation(
        context.source_basis,
        dst,
        edge_pose_fn=edge_pose_fn,
        configuration=selected_config,
        owner=owner,
    )
    out = _reexpress_pose_components(
        pose,
        basis,
        owner=owner,
        association=SpatialAssociationPlan(selected_config.graph),
    )
    if src_rep == "matrix":
        out = out.as_matrix(validate=False)
    return _finalize_same_relation(
        pose,
        analysis_object_dataset(out),
        expressed_in=context.destination,
        validate=validate,
        owner=owner,
        association=SpatialAssociationPlan(selected_config.graph),
    )


def _solve_pose_basis_rotation(
    src: str,
    dst: Frame | str,
    *,
    edge_pose_fn=None,
    configuration: PathConfiguration | SelectedPathConfiguration,
    owner: str,
):
    from ..path_solve import _solve_pose_path_transform_with_owner

    basis_pose = _solve_pose_path_transform_with_owner(
        src,
        dst,
        edge_pose_fn=edge_pose_fn,
        configuration=configuration,
        owner=owner,
    )
    _, basis_rotation = basis_pose.decompose(validate=False)
    return clear_framing(basis_rotation.as_quat(validate=False), owner=owner)


def _reexpress_pose_components(
    pose: Pose,
    basis,
    *,
    owner: str,
    association: SpatialAssociationPlan,
):
    from ..pose import Pose
    from ..rotation import _rotation_compose_with_owner, _rotation_inverse_with_owner
    from .rotation_apply_ops import _rotation_apply_with_owner

    translation, rotation = pose.as_components(validate=False).decompose(validate=False)
    translation_out = _rotation_apply_with_owner(
        basis,
        translation,
        validate=False,
        owner=owner,
        association=association,
    )
    rotation_quat = clear_framing(rotation.as_quat(validate=False), owner=owner)
    left = _rotation_compose_with_owner(
        basis,
        rotation_quat,
        validate=False,
        owner=owner,
        association=association,
    )
    right = _rotation_inverse_with_owner(basis, validate=False, owner=owner)
    rotation_out = _rotation_compose_with_owner(
        left,
        right,
        validate=False,
        owner=owner,
        association=association,
    )
    return Pose.from_components(
        rotation_out,
        translation_out,
        expressed_in=None,
        graph=association.graph,
        validate=False,
    )


__all__ = [
    "pose_express_in",
    "position_express_in",
    "rotation_express_in",
]
