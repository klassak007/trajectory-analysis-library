from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from ..param_engine import ParamMapOptions
from ..param_engine.prepared import PreparedParamEvaluation, prepare_param_evaluation
from .types import ParamRuntimeContext


def _without_sequence_size_coordinate(
    value: xr.DataArray,
    *,
    name: str | None,
) -> xr.DataArray:
    if name is None or name not in value.coords:
        return value
    return value.drop_vars(name)


def prepare_runtime_param_evaluation(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    options: ParamMapOptions,
    param_kind: str,
    query_dim: str,
    reuse: Sequence[PreparedParamEvaluation] = (),
) -> PreparedParamEvaluation:
    """Prepare evaluation after removing source-only validity metadata."""
    size_name = context.sequence_size_coord
    return prepare_param_evaluation(
        param=_without_sequence_size_coordinate(context.spec.coord, name=size_name),
        query=query,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        batch_coords=context.batch_coords,
        valid_mask=_without_sequence_size_coordinate(context.valid_mask, name=size_name),
        options=options,
        param_kind=param_kind,
        query_dim=query_dim,
        reuse=reuse,
    )


__all__ = ["prepare_runtime_param_evaluation"]
