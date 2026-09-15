from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ..param_engine import (
    ParamMapOptions,
)
from ..param_engine.map_apply import apply_param_map_with_batch_dims
from ..param_engine.prepared import PreparedParamEvaluation
from .finalize import finalize_param_output
from .guards import (
    assert_query_dim_safe,
    assert_reserved_metadata_safe,
    mark_reserved_coord,
)
from .runtime_prepare import prepare_runtime_param_evaluation
from .types import ParamEvalOptions, ParamRuntimeContext

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


def _apply_map_dataset(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    sequence_size_coord: str | None,
    param_map,
) -> xr.Dataset:
    out_vars: dict[str, xr.DataArray] = {}
    for name, var in ds.data_vars.items():
        if sequence_dim not in var.dims:
            out_vars[str(name)] = var
            continue
        if not np.issubdtype(np.dtype(var.dtype), np.number):
            raise TypeError(f"param at/resample: non-numeric sequence variable {name!r} is not supported.")
        if sequence_size_coord in var.coords:
            var = var.drop_vars(sequence_size_coord)
        out_vars[str(name)] = apply_param_map_with_batch_dims(
            var,
            param_map=param_map,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
        )
    out = xr.Dataset(data_vars=out_vars)
    sequence_coords = [
        name for name, coord in ds.coords.items() if sequence_dim in coord.dims
    ]
    base = ds.drop_vars([*ds.data_vars, *sequence_coords], errors="ignore")
    collisions = tuple(name for name in base.coords if name in out.coords)
    if collisions:
        out = out.drop_vars(collisions)
    return out.assign_coords(base.coords)


def evaluate_param(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamEvalOptions,
    validate: bool,
    prepared: PreparedParamEvaluation | None = None,
) -> AnalysisObject:
    """Evaluate AO data on a parameter query grid.

    Parameters
    ----------
    context : ParamRuntimeContext
        Resolved runtime context/payload used by this orchestration boundary.
    query : xr.DataArray | np.ndarray | Sequence[float] | float, optional
        Query coordinate/grid used for parameter evaluation.
    opts : ParamEvalOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    assert_query_dim_safe(
        context.ds,
        sequence_dim=context.sequence_dim,
        query_dim=opts.query_dim,
        owner="param at/resample",
    )
    assert_reserved_metadata_safe(
        context.ds,
        reserved=("valid", "sample_index"),
        param_name=context.spec.name,
        owner="param at/resample",
    )
    evaluation = prepare_runtime_param_evaluation(
        context,
        query=query,
        options=ParamMapOptions(method=opts.method, duplicate_policy=opts.duplicate_policy),
        param_kind=context.param_kind,
        query_dim=opts.query_dim,
        reuse=() if prepared is None else (prepared,),
    )
    grid = evaluation.grid
    pmap = evaluation.param_map
    ds_out = _apply_map_dataset(
        context.ds,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        sequence_size_coord=context.sequence_size_coord,
        param_map=pmap,
    )
    ds_out = ds_out.assign_coords({"valid": mark_reserved_coord(pmap.valid, name="valid")})
    return finalize_param_output(
        context,
        ds_out,
        query=grid.values,
        query_dim=opts.query_dim,
        valid_query=pmap.valid,
        validate=validate,
        trajectory=(grid.stacked_dims is None),
    )


__all__ = ["evaluate_param"]
