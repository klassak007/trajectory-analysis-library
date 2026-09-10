from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tal.core.dataset_ownership import (
    analysis_object_dataset,
    metadata_isolated_dataset,
)
from tal.core.schema_errors import SchemaError
from tal.frames import Frame
from tal.frames.registry import get_edge_to_parent_runtime_ext
from tal.utils.frame_schema import get_frames, set_frames

from ..metadata import get_expressed_in
from ..metadata.relation import clear_expressed_in
from .edge_resolver_ops import (
    PreparedEdgeResolver,
    call_prepared_edge_resolver,
    prepare_edge_resolver,
)

if TYPE_CHECKING:
    from ..pose import Pose

POSE_PROVIDER_KEY = "spatial.pose_provider"


@dataclass(frozen=True)
class BoundPoseProvider:
    value: object
    signature_checked: bool | None


def _require_edge_metadata(
    source,
    *,
    child_id: str,
    parent_id: str,
    owner: str,
) -> None:
    edge_parent, edge_child = get_frames(source)
    parent_mismatch = edge_parent is not None and edge_parent != parent_id
    child_mismatch = edge_child is not None and edge_child != child_id
    if parent_mismatch or child_mismatch:
        raise ValueError(
            f"{owner}: framed edge payload must match each present tag for "
            f"(parent={parent_id!r}, child={child_id!r}); "
            f"got {(edge_parent, edge_child)!r}."
        )
    expressed_in = get_expressed_in(source, owner=owner)
    if expressed_in is not None and expressed_in != parent_id:
        raise ValueError(
            f"{owner}: edge provider expressed_in must match parent={parent_id!r}; "
            f"got {expressed_in!r}. Re-express the value in the edge parent first."
        )


def normalize_edge_provider_dataset(
    source,
    *,
    child_id: str,
    parent_id: str,
    owner: str,
):
    """Return one name-neutral, parent-basis edge Dataset."""
    _require_edge_metadata(
        source,
        child_id=child_id,
        parent_id=parent_id,
        owner=owner,
    )
    isolated = metadata_isolated_dataset(source, owner=owner)
    isolated = clear_expressed_in(isolated, validate=False, owner=owner)
    return set_frames(isolated, parent=None, child=None, validate=False)


def normalize_pose_provider(
    payload: object,
    *,
    child_id: str,
    parent_id: str,
    owner: str,
    components: bool = True,
) -> Pose:
    from ..pose import Pose

    try:
        pose = payload if isinstance(payload, Pose) else Pose(payload)
        if components:
            pose = pose.as_components(validate=False)
    except (TypeError, ValueError, SchemaError) as exc:
        raise ValueError(f"{owner}: edge resolver must return Pose-coercible payload.") from exc
    cleared = normalize_edge_provider_dataset(
        analysis_object_dataset(pose),
        child_id=child_id,
        parent_id=parent_id,
        owner=owner,
    )
    return Pose._from_unvalidated(cleared)


def prepare_pose_provider(value: object, *, child_id: str, parent_id: str, owner: str) -> BoundPoseProvider:
    if callable(value):
        prepared = prepare_edge_resolver(value, owner=owner, arg="provider")
        return BoundPoseProvider(prepared.resolver, prepared.signature_checked)
    pose = normalize_pose_provider(value, child_id=child_id, parent_id=parent_id, owner=owner, components=False)
    return BoundPoseProvider(pose, None)


def require_bound_pose_provider(child: Frame, parent: Frame, *, owner: str) -> BoundPoseProvider:
    """Require one bound provider without invoking or normalizing it."""
    provider = get_edge_to_parent_runtime_ext(child, POSE_PROVIDER_KEY, owner=owner)
    if provider is None:
        raise ValueError(f"{owner}: missing bound Pose provider for edge (child={child.id!r}, parent={parent.id!r}).")
    if not isinstance(provider, BoundPoseProvider):
        raise TypeError(f"{owner}: malformed bound Pose provider for edge (child={child.id!r}, parent={parent.id!r}).")
    return provider


def resolve_bound_pose(child: Frame, parent: Frame, *, owner: str) -> Pose:
    provider = require_bound_pose_provider(child, parent, owner=owner)
    value = provider.value
    if provider.signature_checked is not None:
        prepared = PreparedEdgeResolver(value, "provider", provider.signature_checked)
        value = call_prepared_edge_resolver(
            prepared,
            child,
            parent,
            owner=owner,
        )
    return normalize_pose_provider(value, child_id=child.id, parent_id=parent.id, owner=owner)
