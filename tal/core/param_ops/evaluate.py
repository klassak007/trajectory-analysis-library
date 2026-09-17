from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ..param_engine import (
    ParamMapOptions,
)
from ..param_engine.map_apply import apply_param_map_with_batch_dims
from ..param_engine.prepared import PreparedParamEvaluation
from ..param_engine.query_topology import (
    QueryOutputPlan,
    generated_query_coordinate_names,
    preflight_query_output_namespace,
)
from .finalize import assign_sampled_query_coordinate, finalize_param_output
from .guards import (
    assert_query_dim_safe,
    assert_reserved_metadata_safe,
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


def _preflight_evaluation_request(
    context: ParamRuntimeContext,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamEvalOptions,
    output_intent: Literal["grid", "trajectory"],
) -> QueryOutputPlan:
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
    trajectory = output_intent == "trajectory" or not isinstance(query, xr.DataArray) or (
        len(set(query.dims) - set(context.batch_dims)) <= 1
    )
    return preflight_query_output_namespace(
        context.ds,
        query,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        owner="param at/resample",
        intent=output_intent,
        generated_names=generated_query_coordinate_names(
            operation="evaluate",
            param_name=context.spec.name,
            size_name=context.sequence_size_coord,
            trajectory=trajectory,
            mapped_dataset=True,
        ),
    )


def evaluate_param(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamEvalOptions,
    validate: bool,
    prepared: PreparedParamEvaluation | None = None,
    output_intent: Literal["grid", "trajectory"] = "grid",
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
    output_plan = _preflight_evaluation_request(context, query, opts, output_intent)
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
    output_plan = replace(output_plan, topology=evaluation.query_topology)
    ds_out = _apply_map_dataset(
        context.ds,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        sequence_size_coord=context.sequence_size_coord,
        param_map=pmap,
    )
    if output_intent == "grid" and grid.stacked_dims is not None:
        ds_out = assign_sampled_query_coordinate(
            ds_out, query=grid.values, query_dim=grid.query_dim, name=context.spec.name,
        )
    return finalize_param_output(
        context,
        ds_out,
        query=grid.values,
        query_dim=opts.query_dim,
        valid_query=pmap.valid,
        query_topology=evaluation.query_topology,
        validate=validate,
        trajectory=(output_intent == "trajectory" or grid.stacked_dims is None),
        owner="param at/resample",
        output_plan=output_plan,
    )


__all__ = ["evaluate_param"]
