from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import xarray as xr

from ..core.analysis_object import AnalysisObject
from ..core.orchestration.schema_finalize import (
    CoreSchemaFinalizeSpec,
    finalize_with_schema,
)


@dataclass(frozen=True)
class ArrayFinalizeSpec:
    output_var_name: str
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    param_name: str | None
    size_name: str | None


def _to_output_dataset(result: xr.DataArray, *, output_var_name: str) -> xr.Dataset:
    arr = result if result.name == output_var_name else result.rename(output_var_name)
    return arr.to_dataset(name=output_var_name)


def finalize_array_result(
    source_ao: AnalysisObject,
    result: xr.DataArray,
    *,
    spec: ArrayFinalizeSpec,
    optional_sources: Sequence[xr.DataArray],
    owner: str,
    validate: bool,
) -> AnalysisObject:
    ds = _to_output_dataset(result, output_var_name=spec.output_var_name)
    core_spec = CoreSchemaFinalizeSpec(
        sequence_dim=spec.sequence_dim,
        batch_dims=spec.batch_dims,
        core_dims=spec.core_dims,
        param_name=spec.param_name,
        size_name=spec.size_name,
    )
    return finalize_with_schema(
        source_ao,
        ds,
        spec=core_spec,
        validate=validate,
        owner=owner,
        optional_sources=optional_sources,
    )


__all__ = [
    "ArrayFinalizeSpec",
    "finalize_array_result",
]
