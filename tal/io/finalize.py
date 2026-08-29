from __future__ import annotations

import xarray as xr

from tal.core.orchestration.context import DatasetContextOptions, resolve_dataset_context
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema


def _optional_sources(ds: xr.Dataset) -> tuple[xr.DataArray, ...]:
    return tuple(ds[name] for name in ds.data_vars)


def _resolve_finalize_spec(source_ao: "AnalysisObject", *, owner: str) -> CoreSchemaFinalizeSpec:
    context = resolve_dataset_context(
        source_ao,
        owner=owner,
        options=DatasetContextOptions(require_roles=True, require_sequence_dim=True),
    )
    if context.sequence_dim is None:
        raise ValueError(f"{owner}: persisted schema must declare sequence_dim for AO-direct reads.")
    return CoreSchemaFinalizeSpec(
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        core_dims=context.core_dims,
        param_name=context.param_coord,
        size_name=context.sequence_size_coord,
    )


def resolve_finalize_source_for_cls(
    cls: type["AnalysisObject"],
    ds: xr.Dataset,
    *,
    owner: str,
) -> "AnalysisObject":
    from tal.core.analysis_object import AnalysisObject

    if not issubclass(cls, AnalysisObject):
        raise TypeError(f"{owner}: cls must be AnalysisObject or subclass; got {cls!r}.")
    try:
        base = coerce_analysis_object_input(ds, owner=owner)
    except Exception as exc:
        raise ValueError(f"{owner}: invalid persisted schema payload.") from exc
    return base


def _construct_loaded_cls(
    cls: type["AnalysisObject"],
    finalized: "AnalysisObject",
) -> "AnalysisObject":
    from tal.core.analysis_object import AnalysisObject

    if cls is AnalysisObject:
        return finalized
    return cls(finalized.unsafe_data)


def finalize_loaded_dataset(
    cls: type["AnalysisObject"],
    ds: xr.Dataset,
    *,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    base_ao = resolve_finalize_source_for_cls(cls, ds, owner=owner)
    spec = _resolve_finalize_spec(base_ao, owner=owner)
    finalized = finalize_with_schema(
        base_ao,
        base_ao.unsafe_data,
        spec=spec,
        validate=validate,
        owner=owner,
        optional_sources=_optional_sources(base_ao.unsafe_data),
    )
    return _construct_loaded_cls(cls, finalized)


__all__ = ["finalize_loaded_dataset", "resolve_finalize_source_for_cls"]
