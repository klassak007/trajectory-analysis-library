from __future__ import annotations

import numpy as np

from ..core.analysis_object import AnalysisObject
from ..core.orchestration.context import (
    DatasetContext,
    DatasetContextOptions,
    resolve_dataset_context,
)
from ..core.orchestration.inputs import coerce_operand


def is_scalar_component(value: object) -> bool:
    return bool(np.isscalar(value))


def coerce_component_or_scalar(
    value: object,
    *,
    owner: str,
    label: str,
) -> AnalysisObject | None:
    out = coerce_operand(
        value,
        owner=owner,
        label=label,
        role="component",
        allow_scalar=True,
        return_scalar_none=True,
    )
    return out if isinstance(out, AnalysisObject) else None


def build_reference_component_context(
    ao: AnalysisObject,
    *,
    owner: str,
) -> DatasetContext:
    context = resolve_dataset_context(
        ao,
        owner=owner,
        options=DatasetContextOptions(
            require_roles=True,
            require_sequence_dim=False,
            select_numeric_var=True,
            require_single_numeric_var=True,
            require_semantic_dims_in_var=True,
        ),
    )
    if context.data is None or context.var_name is None:
        raise ValueError(f"{owner}: reference component context is incomplete.")
    return context


def promote_scalar_component(
    value: object,
    *,
    reference: DatasetContext,
    owner: str,
) -> AnalysisObject:
    if reference.data is None or reference.var_name is None:
        raise ValueError(f"{owner}: reference component context is incomplete.")
    try:
        filled = (reference.data * 0) + value
    except Exception as exc:  # pragma: no cover - backend/type specific error branch
        raise ValueError(f"{owner}: scalar component must be numeric and broadcast-compatible.") from exc
    ds = filled.rename(reference.var_name).to_dataset()
    if reference.param_coord and reference.param_coord in reference.ds.coords and reference.param_coord not in ds.coords:
        ds = ds.assign_coords({reference.param_coord: reference.ds.coords[reference.param_coord]})
    if (
        reference.sequence_size_coord
        and reference.sequence_size_coord in reference.ds.coords
        and reference.sequence_size_coord not in ds.coords
    ):
        ds = ds.assign_coords({reference.sequence_size_coord: reference.ds.coords[reference.sequence_size_coord]})
    return AnalysisObject.from_data(
        ds,
        sequence_dim=reference.sequence_dim,
        batch_dims=reference.batch_dims,
        core_dims=reference.core_dims,
        param_coord=reference.param_coord,
        sequence_size_coord=reference.sequence_size_coord,
        validate=True,
    )


__all__ = [
    "build_reference_component_context",
    "coerce_component_or_scalar",
    "is_scalar_component",
    "promote_scalar_component",
]
