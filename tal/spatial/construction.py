from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.schema import UNSET, UnsetType
from tal.core.typed_lifecycle import _finish_typed_promotion, _prepare_typed_promotion
from tal.frames import FrameGraph
from tal.utils.frame_schema import get_frames, set_frames

from .association import (
    SpatialAssociationMixin,
    SpatialAssociationPlan,
    attach_spatial_association,
    require_graph_override,
    resolve_passive_association,
)
from .metadata import get_expressed_in, set_expressed_in


@dataclass(frozen=True)
class SpatialConstructionOverrides:
    """Type-checked public construction declarations."""

    parent: str | None | UnsetType
    child: str | None | UnsetType
    expressed_in: str | None | UnsetType
    graph: FrameGraph | None | UnsetType


@dataclass(frozen=True)
class SpatialConstructionPlan:
    """Resolved frame declarations and passive association for construction."""

    parent: str | None
    child: str | None
    expressed_in: str | None | UnsetType
    association: SpatialAssociationPlan


@dataclass(frozen=True)
class _SpatialSourceDeclarations:
    """Compatible declarations inherited from spatial construction inputs."""

    parent: str | None
    child: str | None
    expressed_in: str | None


def _resolve_basis_update(
    inherited: str | None,
    requested: str | None | UnsetType,
    *,
    next_parent: str | None,
    owner: str,
) -> str | None | UnsetType:
    if requested is None:
        return None
    effective = inherited or next_parent
    if requested is UNSET:
        return inherited if inherited is not None and inherited != next_parent else UNSET
    if effective is not None and requested != effective:
        raise ValueError(
            f"{owner}: expressed_in={requested!r} conflicts with effective basis "
            f"{effective!r}; re-express the source explicitly."
        )
    return requested


def _normalize_declaration(
    value: object,
    *,
    field: str,
    owner: str,
) -> str | None | UnsetType:
    if value is UNSET or value is None:
        return value
    if not isinstance(value, str):
        raise TypeError(f"{owner}: {field} must be a non-empty string, None, or UNSET.")
    cleaned = value.strip()
    if cleaned:
        return cleaned
    raise ValueError(f"{owner}: {field} must be a non-empty string, None, or UNSET.")


def _shared_source_frame_value(
    sources: tuple[object, ...],
    *,
    index: int,
    field: str,
    owner: str,
) -> str | None:
    selected: str | None = None
    for source in sources:
        candidate = get_frames(analysis_object_dataset(source))[index]
        if candidate is None:
            continue
        if selected is None:
            selected = candidate
            continue
        if candidate != selected:
            raise ValueError(
                f"{owner}: spatial input frame tags must match exactly for {field}; "
                f"got {selected!r} and {candidate!r}."
            )
    return selected


def _shared_source_basis(
    sources: tuple[object, ...],
    *,
    owner: str,
) -> str | None:
    selected: str | None = None
    for source in sources:
        ds = analysis_object_dataset(source)
        parent, _ = get_frames(ds)
        candidate = get_expressed_in(ds, owner=owner) or parent
        if candidate is None:
            continue
        if selected is None:
            selected = candidate
            continue
        if candidate != selected:
            raise ValueError(
                f"{owner}: spatial inputs use different effective expressed_in bases; "
                f"got {selected!r} and {candidate!r}."
            )
    return selected


def _resolve_frame_value(
    current: str | None,
    requested: str | None | UnsetType,
    *,
    field: str,
    owner: str,
) -> str | None:
    if requested is None:
        return None
    if requested is UNSET:
        return current
    if current is not None and requested != current:
        raise ValueError(
            f"{owner}: {field}={requested!r} conflicts with existing {field}={current!r}; "
            "retag or transform the source explicitly."
        )
    return requested


def _resolve_source_declarations(
    sources: tuple[object, ...],
    *,
    owner: str,
) -> _SpatialSourceDeclarations:
    """Validate source compatibility before applying output declarations."""
    return _SpatialSourceDeclarations(
        parent=_shared_source_frame_value(
            sources, index=0, field="parent", owner=owner
        ),
        child=_shared_source_frame_value(
            sources, index=1, field="child", owner=owner
        ),
        expressed_in=_shared_source_basis(sources, owner=owner),
    )


def preflight_spatial_construction(
    *,
    parent: str | None | UnsetType = UNSET,
    child: str | None | UnsetType = UNSET,
    expressed_in: str | None | UnsetType = UNSET,
    graph: FrameGraph | None | UnsetType = UNSET,
    owner: str,
) -> SpatialConstructionOverrides:
    """Validate public declarations before source ownership or payload work."""
    parent_value = _normalize_declaration(parent, field="parent", owner=owner)
    child_value = _normalize_declaration(child, field="child", owner=owner)
    basis_value = _normalize_declaration(expressed_in, field="expressed_in", owner=owner)
    graph_value = require_graph_override(graph, owner=owner)
    return SpatialConstructionOverrides(
        parent=parent_value,
        child=child_value,
        expressed_in=basis_value,
        graph=graph_value,
    )


def prepare_spatial_construction(
    sources: tuple[object, ...],
    *,
    overrides: SpatialConstructionOverrides,
    owner: str,
) -> SpatialConstructionPlan:
    """Resolve inherited declarations without payload or topology work."""
    association = resolve_passive_association(
        sources,
        graph=overrides.graph,
        owner=owner,
    )
    declarations = _resolve_source_declarations(sources, owner=owner)
    next_parent = _resolve_frame_value(
        declarations.parent,
        overrides.parent,
        field="parent",
        owner=owner,
    )
    next_child = _resolve_frame_value(
        declarations.child,
        overrides.child,
        field="child",
        owner=owner,
    )
    basis_update = _resolve_basis_update(
        declarations.expressed_in,
        overrides.expressed_in,
        next_parent=next_parent,
        owner=owner,
    )
    return SpatialConstructionPlan(
        parent=next_parent,
        child=next_child,
        expressed_in=basis_update,
        association=association,
    )


def apply_spatial_construction(
    ds: xr.Dataset,
    *,
    plan: SpatialConstructionPlan,
    owner: str,
) -> xr.Dataset:
    """Apply one resolved construction plan at the metadata boundary."""
    out = set_frames(
        ds,
        parent=plan.parent,
        child=plan.child,
        validate=False,
    )
    if plan.expressed_in is UNSET:
        return out
    return set_expressed_in(
        out,
        expressed_in=plan.expressed_in,
        validate=False,
        owner=owner,
    )


def initialize_spatial_typed(
    instance: object,
    data: object,
    *,
    parent: str | None | UnsetType = UNSET,
    child: str | None | UnsetType = UNSET,
    expressed_in: str | None | UnsetType = UNSET,
    graph: FrameGraph | None | UnsetType = UNSET,
    owner: str,
) -> None:
    """Run one typed spatial constructor through the shared lifecycle."""
    overrides = preflight_spatial_construction(
        parent=parent,
        child=child,
        expressed_in=expressed_in,
        graph=graph,
        owner=owner,
    )
    source = coerce_analysis_object_input(data, owner=owner)
    plan = prepare_spatial_construction(
        (source,),
        overrides=overrides,
        owner=owner,
    )
    instance._init_typed(source, options=plan)
    attach_spatial_association(instance, plan.association)


def prepare_spatial_factory_dataset(
    source: AnalysisObject,
    *,
    owner: str,
) -> xr.Dataset:
    """Prepare one already-owned source for a single-source spatial factory."""
    return _prepare_typed_promotion(source, owner=owner)


def finish_spatial_factory_promotion(
    source: AnalysisObject,
    result: AnalysisObject,
) -> AnalysisObject:
    """Couple source lifetime after a spatial factory lifecycle succeeds."""
    _finish_typed_promotion(source, analysis_object_dataset(result))
    return result


def initialize_spatial_configuration(
    instance: object,
    data: object,
    *,
    overrides: SpatialConstructionOverrides,
    owner: str,
    coerce_source: Callable[..., object],
    pre_enforce: Callable[..., None] | None = None,
    post_enforce: Callable[..., xr.Dataset] | None = None,
) -> None:
    """Run a configuration-style spatial constructor through one lifecycle."""
    source = coerce_source(data, owner=owner)
    plan = prepare_spatial_construction((source,), overrides=overrides, owner=owner)
    instance._bind_dataset(_prepare_typed_promotion(source, owner=owner))
    instance._normalize_metadata(owner=owner)
    prepared = apply_spatial_construction(
        analysis_object_dataset(instance),
        plan=plan,
        owner=owner,
    )
    instance._bind_dataset(prepared)
    if pre_enforce is not None:
        pre_enforce(analysis_object_dataset(instance), owner=owner)
    instance._enforce_invariants(owner=owner)
    if post_enforce is not None:
        checked = post_enforce(analysis_object_dataset(instance), owner=owner)
        instance._bind_dataset(checked)
    _finish_typed_promotion(source, analysis_object_dataset(instance))
    attach_spatial_association(instance, plan.association)


def initialize_declared_spatial_configuration(
    instance: object,
    data: object,
    *,
    parent: str | None | UnsetType,
    child: str | None | UnsetType,
    expressed_in: str | None | UnsetType,
    graph: FrameGraph | None | UnsetType,
    owner: str,
    coerce_source: Callable[..., object],
    pre_enforce: Callable[..., None] | None = None,
    post_enforce: Callable[..., xr.Dataset] | None = None,
) -> None:
    """Preflight declarations, then run one configuration lifecycle."""
    overrides = preflight_spatial_construction(
        parent=parent,
        child=child,
        expressed_in=expressed_in,
        graph=graph,
        owner=owner,
    )
    initialize_spatial_configuration(
        instance,
        data,
        overrides=overrides,
        owner=owner,
        coerce_source=coerce_source,
        pre_enforce=pre_enforce,
        post_enforce=post_enforce,
    )


class SpatialTypedConstructionMixin(SpatialAssociationMixin):
    """Shared constructor boundary for lifecycle-backed spatial types."""

    SPATIAL_CONSTRUCTION_OWNER: str

    def __init__(
        self,
        data: object,
        *,
        parent: str | None | UnsetType = UNSET,
        child: str | None | UnsetType = UNSET,
        expressed_in: str | None | UnsetType = UNSET,
        graph: FrameGraph | None | UnsetType = UNSET,
    ) -> None:
        initialize_spatial_typed(
            self,
            data,
            parent=parent,
            child=child,
            expressed_in=expressed_in,
            graph=graph,
            owner=self.SPATIAL_CONSTRUCTION_OWNER,
        )


class SpatialConfigurationConstructionMixin(SpatialAssociationMixin):
    """Shared constructor boundary for configuration-style spatial types."""

    SPATIAL_CONSTRUCTION_OWNER: str
    SPATIAL_SOURCE_COERCER: Callable[..., object]
    SPATIAL_PRE_ENFORCE: Callable[..., None] | None = None
    SPATIAL_POST_ENFORCE: Callable[..., xr.Dataset] | None = None

    def __init__(
        self,
        data: AnalysisObject | xr.Dataset | xr.DataArray,
        *,
        parent: str | None | UnsetType = UNSET,
        child: str | None | UnsetType = UNSET,
        expressed_in: str | None | UnsetType = UNSET,
        graph: FrameGraph | None | UnsetType = UNSET,
    ) -> None:
        cls = type(self)
        initialize_declared_spatial_configuration(
            self,
            data,
            parent=parent,
            child=child,
            expressed_in=expressed_in,
            graph=graph,
            owner=cls.SPATIAL_CONSTRUCTION_OWNER,
            coerce_source=cls.SPATIAL_SOURCE_COERCER,
            pre_enforce=cls.SPATIAL_PRE_ENFORCE,
            post_enforce=cls.SPATIAL_POST_ENFORCE,
        )


__all__ = [
    "SpatialConfigurationConstructionMixin",
    "SpatialConstructionOverrides",
    "SpatialConstructionPlan",
    "SpatialTypedConstructionMixin",
    "apply_spatial_construction",
    "finish_spatial_factory_promotion",
    "initialize_declared_spatial_configuration",
    "initialize_spatial_configuration",
    "initialize_spatial_typed",
    "preflight_spatial_construction",
    "prepare_spatial_construction",
    "prepare_spatial_factory_dataset",
]
