from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
from .finalize import (
    _finalize_prepared_param_output,
    _prepare_param_output_dataset,
    assign_sampled_query_coordinate,
)
from .guards import (
    assert_query_dim_safe,
    assert_reserved_metadata_safe,
)
from .runtime_prepare import prepare_runtime_param_evaluation
from .types import ParamEvalOptions, ParamRuntimeContext

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


@dataclass(frozen=True)
class _ParamDatasetPart:
    """One mapped Dataset and its already prepared output declarations."""

    dataset: xr.Dataset
    evaluation: PreparedParamEvaluation
    output_plan: QueryOutputPlan
    trajectory: bool


def _apply_map_dataset(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    sequence_size_coord: str | None,
    param_map,
    data_vars: tuple[str, ...] | None = None,
) -> xr.Dataset:
    out_vars: dict[str, xr.DataArray] = {}
    names = tuple(ds.data_vars) if data_vars is None else data_vars
    for name in names:
        var = ds[name]
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
    owner: str,
) -> QueryOutputPlan:
    assert_query_dim_safe(
        context.ds,
        sequence_dim=context.sequence_dim,
        query_dim=opts.query_dim,
        owner=owner,
    )
    assert_reserved_metadata_safe(
        context.ds,
        reserved=("valid", "sample_index"),
        param_name=context.spec.name,
        owner=owner,
    )
    trajectory = output_intent == "trajectory" or not isinstance(query, xr.DataArray) or (
        len(set(query.dims) - set(context.batch_dims)) <= 1
    )
    return preflight_query_output_namespace(
        context.ds,
        query,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        owner=owner,
        intent=output_intent,
        generated_names=generated_query_coordinate_names(
            operation="evaluate",
            param_name=context.spec.name,
            size_name=context.sequence_size_coord,
            trajectory=trajectory,
            mapped_dataset=True,
        ),
    )


def _mapped_param_dataset(
    context: ParamRuntimeContext,
    *,
    evaluation: PreparedParamEvaluation,
    data_vars: tuple[str, ...] | None,
) -> xr.Dataset:
    return _apply_map_dataset(
        context.ds,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        sequence_size_coord=context.sequence_size_coord,
        param_map=evaluation.param_map,
        data_vars=data_vars,
    )


def _prepare_param_dataset_part(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamEvalOptions,
    prepared: PreparedParamEvaluation | None = None,
    output_intent: Literal["grid", "trajectory"] = "grid",
    data_vars: tuple[str, ...] | None = None,
    owner: str = "param at/resample",
    copy_schema: bool = True,
) -> _ParamDatasetPart:
    output_plan = _preflight_evaluation_request(
        context, query, opts, output_intent, owner
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
    output_plan = replace(output_plan, topology=evaluation.query_topology)
    ds_out = _mapped_param_dataset(
        context,
        evaluation=evaluation,
        data_vars=data_vars,
    )
    if output_intent == "grid" and grid.stacked_dims is not None:
        ds_out = assign_sampled_query_coordinate(
            ds_out, query=grid.values, query_dim=grid.query_dim, name=context.spec.name,
        )
    trajectory = output_intent == "trajectory" or grid.stacked_dims is None
    ds_out = _prepare_param_output_dataset(
        context,
        ds_out,
        query=grid.values,
        query_dim=opts.query_dim,
        valid_query=evaluation.param_map.valid,
        query_topology=evaluation.query_topology,
        trajectory=trajectory,
        owner=owner,
        output_plan=output_plan,
        copy_schema=copy_schema,
    )
    return _ParamDatasetPart(ds_out, evaluation, output_plan, trajectory)


def evaluate_param(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamEvalOptions,
    validate: bool,
    prepared: PreparedParamEvaluation | None = None,
    output_intent: Literal["grid", "trajectory"] = "grid",
) -> AnalysisObject:
    """Evaluate AO data on a parameter query grid."""
    part = _prepare_param_dataset_part(
        context,
        query=query,
        opts=opts,
        prepared=prepared,
        output_intent=output_intent,
    )
    return _finalize_prepared_param_output(
        context,
        part.dataset,
        validate=validate,
        trajectory=part.trajectory,
        output_plan=part.output_plan,
        query_topology=part.evaluation.query_topology,
    )


__all__ = ["evaluate_param"]
