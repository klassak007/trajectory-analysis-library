from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .. import validity_values
from ..orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from .types import CombineContext

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


@dataclass(frozen=True)
class CombineFinalizationPlan:
    """Result wrapper and opaque domain context for one combine operation."""

    source: AnalysisObject
    rewrap_context: object | None


def prepare_combine_finalization(
    contexts: list[CombineContext],
    *,
    owner: str,
    source_ao: AnalysisObject | None = None,
) -> CombineFinalizationPlan:
    """Resolve result context before alignment or numerical combine work."""
    if not contexts:
        raise ValueError(f"{owner}: expected at least one input.")
    source = contexts[0].ao if source_ao is None else source_ao
    values = tuple(context.ao for context in contexts)
    context = source._prepare_result_rewrap_context(values, owner=owner)
    return CombineFinalizationPlan(source, context)


def finalize_combine_output(
    plan: CombineFinalizationPlan,
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, ...],
    param_coord: str | None,
    sequence_size_coord: str | None,
    validate: bool,
) -> "AnalysisObject":
    size_name = sequence_size_coord
    candidate = ds
    if sequence_dim is not None:
        candidate, size_name = _normalize_sequence_size_coord(
            candidate,
            sequence_dim=sequence_dim,
            sequence_size_coord=size_name,
        )
    spec = CoreSchemaFinalizeSpec(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
        param_name=param_coord,
        size_name=size_name,
    )
    result = finalize_with_schema(
        plan.source,
        candidate,
        spec=spec,
        validate=validate,
        owner="finalize_combine_output",
        clear_returns_unvalidated=True,
    )
    return plan.source._apply_result_rewrap_context(
        result,
        context=plan.rewrap_context,
    )


def apply_outer_fill(
    ds: xr.Dataset,
    *,
    dim: str,
    labels: object,
    fill_value: float | int | None | Mapping[str, float | int | None],
) -> xr.Dataset:
    if not isinstance(fill_value, Mapping):
        return ds.reindex({dim: labels}, fill_value=fill_value)
    out = ds.reindex({dim: labels}, fill_value=np.nan)
    for name, value in fill_value.items():
        if name not in out.variables:
            continue
        v = out[name].reindex({dim: labels}, fill_value=value)
        if name in out.coords:
            out = out.assign_coords({name: v})
        else:
            out[name] = v
    return out


def _normalize_sequence_size_coord(
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    sequence_size_coord: str | None,
) -> tuple[xr.Dataset, str | None]:
    if sequence_dim is None or sequence_size_coord is None or sequence_size_coord not in ds.coords:
        return ds, sequence_size_coord
    try:
        validated = validity_values.require_valid_sequence_size_values(
            ds.coords[sequence_size_coord],
            sequence_size_coord=sequence_size_coord,
            sequence_len=int(ds.sizes.get(sequence_dim, 0)),
            owner="finalize_combine_output",
        )
    except ValueError:
        return ds.drop_vars(sequence_size_coord, errors="ignore"), None
    return ds.assign_coords({sequence_size_coord: validated}), sequence_size_coord


__all__ = [
    "CombineFinalizationPlan",
    "apply_outer_fill",
    "finalize_combine_output",
    "prepare_combine_finalization",
]
