from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import xarray as xr

from tal.core.dataset_ownership import (
    analysis_object_dataset,
    couple_dataset_resource,
    metadata_isolated_dataset,
)
from tal.core.schema import UNSET, UnsetType
from tal.frames import FrameGraph
from tal.utils.frame_schema import get_frames

from .metadata import get_expressed_in, set_expressed_in

if TYPE_CHECKING:
    from tal.core.analysis_object import AnalysisObject


@dataclass(frozen=True)
class SpatialAssociationPlan:
    """Resolved wrapper-local graph identity for one spatial result."""

    graph: FrameGraph | None


def require_graph_override(
    graph: FrameGraph | None | UnsetType,
    *,
    owner: str,
) -> FrameGraph | None | UnsetType:
    if graph is UNSET or graph is None or isinstance(graph, FrameGraph):
        return graph
    raise TypeError(f"{owner}: graph must be FrameGraph, None, or UNSET.")


def require_graph_association(
    graph: object,
    *,
    owner: str,
) -> FrameGraph | None:
    """Validate the required-value graph boundary used by ``with_graph``."""
    if graph is None or isinstance(graph, FrameGraph):
        return graph
    raise TypeError(f"{owner}: graph must be FrameGraph or None.")


def associated_graph(value: object) -> FrameGraph | None:
    """Return a spatial wrapper's passive graph association, if any."""
    if isinstance(value, SpatialAssociationMixin):
        return value.graph
    return None


def resolve_passive_association(
    values: tuple[object, ...],
    *,
    graph: FrameGraph | None | UnsetType = UNSET,
    owner: str,
) -> SpatialAssociationPlan:
    """Resolve an explicit override or exact shared wrapper association."""
    override = require_graph_override(graph, owner=owner)
    if override is not UNSET:
        return SpatialAssociationPlan(override)
    selected: FrameGraph | None = None
    for value in values:
        candidate = associated_graph(value)
        if candidate is None:
            continue
        if selected is None:
            selected = candidate
            continue
        if candidate is not selected:
            raise ValueError(
                f"{owner}: spatial inputs are associated with different FrameGraph instances; "
                "supply an explicit graph-capable boundary or reassociate an input."
            )
    return SpatialAssociationPlan(selected)


def attach_spatial_association(value: object, plan: SpatialAssociationPlan):
    """Attach one resolved association after successful spatial finalization."""
    if not isinstance(value, SpatialAssociationMixin):
        raise TypeError(
            "spatial.association: result must implement the spatial association boundary."
        )
    value._spatial_graph = plan.graph
    return value


def finalize_spatial_as(
    cls: type,
    ds: xr.Dataset,
    *,
    validate: bool,
    association: SpatialAssociationPlan,
):
    """Finalize an explicit spatial result type and attach association once."""
    result = cls._from_validated(ds) if validate else cls._from_unvalidated(ds)
    return attach_spatial_association(result, association)


def preserve_spatial_basis(
    source: object,
    ds: xr.Dataset,
    *,
    owner: str,
) -> xr.Dataset:
    """Preserve a source's effective basis on one rebuilt spatial Dataset."""
    source_ds = analysis_object_dataset(source)
    source_parent, _ = get_frames(source_ds)
    effective_basis = get_expressed_in(source_ds, owner=owner) or source_parent
    target_parent, _ = get_frames(ds)
    explicit_basis = None if effective_basis == target_parent else effective_basis
    return set_expressed_in(
        ds,
        expressed_in=explicit_basis,
        validate=False,
        owner=owner,
    )


def finalize_spatial_from_source(
    source: object,
    cls: type,
    ds: xr.Dataset,
    *,
    validate: bool,
):
    """Finalize a cross-type spatial result with its source association."""
    plan = SpatialAssociationPlan(associated_graph(source))
    prepared = preserve_spatial_basis(
        source,
        ds,
        owner=f"spatial.{cls.__name__.lower()}.finalize",
    )
    return finalize_spatial_as(cls, prepared, validate=validate, association=plan)


class SpatialAssociationMixin:
    """Private spatial wrapper state and same-source propagation hook."""

    _spatial_graph: FrameGraph | None

    @property
    def graph(self) -> FrameGraph | None:
        """Return this spatial value's associated frame graph, if any.

        Returns
        -------
        FrameGraph or None
            The exact passively associated graph. Association does not imply
            registration and is not stored in xarray metadata.
        """
        return getattr(self, "_spatial_graph", None)

    def with_graph(self, graph: FrameGraph | None) -> Self:
        """Return a metadata-isolated owning alias associated with ``graph``.

        Parameters
        ----------
        graph : FrameGraph or None
            Exact graph to remember, or ``None`` to clear the association.

        Returns
        -------
        Self
            A distinct wrapper with isolated metadata and shared eligible
            payload buffers or lazy graphs.

        Raises
        ------
        TypeError
            If ``graph`` is neither a ``FrameGraph`` nor ``None``.

        Notes
        -----
        This method does not create frames or register providers. If the source
        owns a lazy backend resource, the aliases share one coupled lifetime.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.frames import FrameGraph
        >>> from tal.spatial import Position
        >>> data = xr.Dataset(
        ...     {"position": (("sample", "axis"), [[0.0, 0.0, 0.0]])},
        ...     coords={"sample": [0], "axis": ["x", "y", "z"]},
        ... )
        >>> ao = AnalysisObject.from_data(
        ...     data, sequence_dim="sample", core_dims=("axis",), validate=True,
        ... )
        >>> position = Position(ao, parent="world")
        >>> graph = FrameGraph()
        >>> associated = position.with_graph(graph)
        >>> associated.graph is graph and position.graph is None
        True
        """
        owner = f"{self.__class__.__name__}.with_graph"
        plan = SpatialAssociationPlan(require_graph_association(graph, owner=owner))
        source_ds = analysis_object_dataset(self)
        isolated = metadata_isolated_dataset(source_ds, owner=owner)
        result = super()._rewrap_dataset(isolated, validate=False)
        attach_spatial_association(result, plan)
        couple_dataset_resource(source_ds, analysis_object_dataset(result))
        return result

    def _rewrap_dataset(self, ds: xr.Dataset, *, validate: bool) -> AnalysisObject:
        result = super()._rewrap_dataset(ds, validate=validate)
        return attach_spatial_association(
            result,
            SpatialAssociationPlan(self.graph),
        )

    def _prepare_result_rewrap_context(
        self,
        values: tuple[object, ...],
        *,
        owner: str,
    ) -> object:
        return resolve_passive_association(values, owner=owner)

    def _apply_result_rewrap_context(
        self,
        result: AnalysisObject,
        *,
        context: object | None,
    ) -> AnalysisObject:
        if not isinstance(context, SpatialAssociationPlan):
            return result
        return attach_spatial_association(result, context)


__all__ = [
    "SpatialAssociationMixin",
    "SpatialAssociationPlan",
    "associated_graph",
    "attach_spatial_association",
    "finalize_spatial_as",
    "finalize_spatial_from_source",
    "preserve_spatial_basis",
    "require_graph_association",
    "require_graph_override",
    "resolve_passive_association",
]
