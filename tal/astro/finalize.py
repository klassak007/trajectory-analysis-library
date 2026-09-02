from __future__ import annotations

import xarray as xr

from tal.core import AnalysisObject
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema

from .direction import TopocentricDirection
from .metadata import normalize_topocentric_metadata
from .orchestration import AstroDirectionRuntimeContext


def _schema_spec(
    context: AstroDirectionRuntimeContext,
    *,
    core_dim: str,
) -> CoreSchemaFinalizeSpec:
    observer = context.observer
    preserve_observer_sequence = context.output_sequence_dim == observer.sequence_dim
    return CoreSchemaFinalizeSpec(
        sequence_dim=context.output_sequence_dim,
        batch_dims=context.output_batch_dims,
        core_dims=(core_dim,),
        param_name=observer.param_coord if preserve_observer_sequence else None,
        size_name=observer.sequence_size_coord if preserve_observer_sequence else None,
    )


def _optional_sources(context: AstroDirectionRuntimeContext) -> tuple[xr.DataArray, ...]:
    sources: list[xr.DataArray] = []
    if context.observer.data is not None:
        sources.append(context.observer.data)
    sources.append(context.time.coord)
    return tuple(sources)


def finalize_topocentric_direction(
    context: AstroDirectionRuntimeContext,
    ds: xr.Dataset,
    *,
    core_dim: str,
    backend: str = "astropy",
    time_scale: str | None = None,
    validate: bool = True,
    owner: str,
) -> TopocentricDirection:
    """Finalize a topocentric direction dataset through shared core owners."""
    source = AnalysisObject._from_validated(context.observer.ds)
    finalized = finalize_with_schema(
        source,
        ds,
        spec=_schema_spec(context, core_dim=core_dim),
        validate=validate,
        owner=owner,
        optional_sources=_optional_sources(context),
    )
    normalized = normalize_topocentric_metadata(
        analysis_object_dataset(finalized),
        backend=backend,
        time_scale=time_scale or context.time.scale,
        validate=False,
        owner=owner,
    )
    if validate:
        return TopocentricDirection._from_validated(normalized)
    return TopocentricDirection._from_unvalidated(normalized)


__all__ = ["finalize_topocentric_direction"]
