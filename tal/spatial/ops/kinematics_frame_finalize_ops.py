from __future__ import annotations

from tal.core.dataset_ownership import analysis_object_dataset
from tal.frames import FrameGraph
from tal.utils.frame_schema import get_frames, set_frames

from ..association import SpatialAssociationPlan, attach_spatial_association
from ..metadata import (
    get_instantaneous_inertial,
    set_expressed_in,
    set_instantaneous_inertial,
)
from ..policies.wrap import wrap_like
from .kinematics_family_frame_ops import FamilyFrameRequest


def with_relation_semantics(source, ds, *, expressed_in: str, owner: str):
    """Apply preserved kinematic relation metadata to one result Dataset."""
    inertial = get_instantaneous_inertial(analysis_object_dataset(source), owner=owner)
    out = set_expressed_in(ds, expressed_in=expressed_in, validate=False, owner=owner)
    return set_instantaneous_inertial(
        out,
        instantaneous_inertial=inertial,
        validate=False,
        owner=owner,
    )


def finalize_vector_frame_result(
    source,
    out,
    *,
    src_parent: str,
    request: FamilyFrameRequest,
    owner: str,
    graph: FrameGraph,
):
    """Finalize one kinematic frame-change result."""
    out_parent, _ = get_frames(out_ds := analysis_object_dataset(out))
    expressed = src_parent if out_parent is None else out_parent
    ds = with_relation_semantics(source, out_ds, expressed_in=expressed, owner=owner)
    result = wrap_like(source, ds, validate=request.validate)
    return attach_spatial_association(result, SpatialAssociationPlan(graph))


def finalize_vector_expression(
    source,
    out,
    *,
    relation: tuple[str, str | None],
    destination: str,
    request: FamilyFrameRequest,
    graph: FrameGraph,
    owner: str,
):
    """Finalize one kinematic representation-basis change."""
    ds = set_frames(
        analysis_object_dataset(out),
        parent=relation[0],
        child=relation[1],
        validate=False,
    )
    ds = with_relation_semantics(source, ds, expressed_in=destination, owner=owner)
    result = wrap_like(source, ds, validate=request.validate)
    return attach_spatial_association(result, SpatialAssociationPlan(graph))


def finalize_identity_vector_expression(
    source,
    *,
    destination: str,
    request: FamilyFrameRequest,
    graph: FrameGraph | None,
    owner: str,
):
    """Finalize a no-op kinematic representation-basis request."""
    ds = with_relation_semantics(
        source,
        analysis_object_dataset(source),
        expressed_in=destination,
        owner=owner,
    )
    result = wrap_like(source, ds, validate=request.validate)
    return attach_spatial_association(result, SpatialAssociationPlan(graph))


__all__ = [
    "finalize_identity_vector_expression",
    "finalize_vector_expression",
    "finalize_vector_frame_result",
    "with_relation_semantics",
]
