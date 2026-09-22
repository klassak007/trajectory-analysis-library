"""Spatial metadata projection and one-commit composite finalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.component_ops.types import ComponentSpec
from tal.core.dataset_ownership import _prepare_ordered_resource_action
from tal.core.orchestration.composite_commit import (
    _commit_composite_result,
    _ComponentRegistryCommit,
    _CompositeCommitSpec,
)
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec
from tal.core.schema import UNSET
from tal.core.schema_update import _schema_projection_view

from ..association import SpatialAssociationPlan
from ..construction import SpatialConstructionPlan

_TARGET_EXTENSIONS = frozenset({"components", "frames", "spatial"})


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _equal_metadata(left: object, right: object) -> bool:
    if left is right:
        return True
    try:
        result = left == right
        return result if isinstance(result, bool) else False
    except Exception:  # noqa: BLE001 - indeterminate opaque metadata is unequal.
        return False


def _retain_unknown_extension(
    selected: dict[str, object],
    *,
    name: str,
    value: object,
    owner: str,
) -> None:
    if name in _TARGET_EXTENSIONS:
        return
    if name not in selected:
        selected[name] = value
        return
    if not _equal_metadata(selected[name], value):
        raise ValueError(
            f"{owner}: spatial inputs contain conflicting tal.ext.{name} metadata."
        )


def _unknown_extensions(
    sources: Sequence[xr.Dataset],
    *,
    owner: str,
) -> dict[str, object]:
    selected: dict[str, object] = {}
    for source in sources:
        ext = _mapping(_mapping(source.attrs.get("tal")).get("ext"))
        for name, value in ext.items():
            _retain_unknown_extension(
                selected,
                name=name,
                value=value,
                owner=owner,
            )
    return selected


def _spatial_block(
    *,
    representation: str,
    construction: SpatialConstructionPlan,
    kinematics_kind: str | None,
) -> dict[str, object]:
    block: dict[str, object] = {
        "representation": {"rep": representation},
    }
    if kinematics_kind is not None:
        block["roles"] = {"kinematics_kind": kinematics_kind}
    if construction.expressed_in is not UNSET and construction.expressed_in is not None:
        block["relation"] = {"expressed_in": construction.expressed_in}
    return block


def project_paired_spatial_metadata(
    candidate: xr.Dataset,
    *,
    sources: Sequence[xr.Dataset],
    representation: str,
    construction: SpatialConstructionPlan,
    kinematics_kind: str | None,
    owner: str,
) -> xr.Dataset:
    """Project target-owned spatial metadata without copying opaque extensions."""
    ext = _unknown_extensions(sources, owner=owner)
    if construction.parent is not None or construction.child is not None:
        frames: dict[str, str] = {}
        if construction.parent is not None:
            frames["parent"] = construction.parent
        if construction.child is not None:
            frames["child"] = construction.child
        ext["frames"] = frames
    ext["spatial"] = _spatial_block(
        representation=representation,
        construction=construction,
        kinematics_kind=kinematics_kind,
    )
    return _schema_projection_view(candidate, candidate, extensions=ext)


def commit_spatial_composite(
    candidate: xr.Dataset,
    *,
    schema: CoreSchemaFinalizeSpec,
    components: tuple[tuple[str, ComponentSpec], ...] | None,
    prototype: AnalysisObject | type[AnalysisObject],
    association: SpatialAssociationPlan,
    resource_sources: Sequence[xr.Dataset],
    validate: bool,
    owner: str,
    metadata_isolated: bool = False,
) -> AnalysisObject:
    """Commit one already assembled spatial composite result."""
    spec = _CompositeCommitSpec(
        owner=owner,
        validate=validate,
        schema=schema,
        components=(
            _ComponentRegistryCommit(action="prune")
            if components is None
            else _ComponentRegistryCommit(action="replace", entries=components)
        ),
        prototype=prototype,
        result_context=association,
        resource_action=_prepare_ordered_resource_action(resource_sources),
        isolate_non_schema_metadata=not metadata_isolated,
    )
    return _commit_composite_result(candidate, spec=spec)


__all__: list[str] = []
