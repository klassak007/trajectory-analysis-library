from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from ..orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
from ..param_engine.validity_mask import validate_sequence_size_values
from .types import CombineContext


def finalize_combine_output(
    context: CombineContext,
    ds: xr.Dataset,
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, ...],
    param_coord: str | None,
    sequence_size_coord: str | None,
    validate: bool,
    source_ao: "AnalysisObject | None" = None,
) -> "AnalysisObject":
    source = _resolve_finalize_source(context, source_ao=source_ao)
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
    return finalize_with_schema(
        source,
        candidate,
        spec=spec,
        validate=validate,
        owner="finalize_combine_output",
        clear_returns_unvalidated=True,
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
        validated = validate_sequence_size_values(
            ds.coords[sequence_size_coord],
            sequence_size_coord=sequence_size_coord,
            sequence_len=int(ds.sizes.get(sequence_dim, 0)),
            owner="finalize_combine_output",
        )
    except ValueError:
        return ds.drop_vars(sequence_size_coord, errors="ignore"), None
    return ds.assign_coords({sequence_size_coord: validated}), sequence_size_coord


def _resolve_finalize_source(
    context: CombineContext,
    *,
    source_ao: "AnalysisObject | None",
) -> "AnalysisObject":
    if source_ao is None:
        return context.ao
    return source_ao
