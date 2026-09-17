from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.param_engine import ParamMapOptions
from tal.core.param_engine.prepared import PreparedParamEvaluation
from tal.core.param_engine.query_topology import (
    QueryOutputPlan,
    generated_query_coordinate_names,
    preflight_query_output_namespace,
)
from tal.core.param_ops.guards import (
    assert_query_dim_safe,
    assert_reserved_metadata_safe,
)
from tal.core.param_ops.runtime_prepare import prepare_runtime_param_evaluation
from tal.core.param_ops.types import ParamRuntimeContext

from ..temporal.options import RotationTemporalOptions, resolve_rotation_method
from .rotation_temporal_types import RotationTemporalRequest

if TYPE_CHECKING:
    from ..rotation import Rotation


def build_quat_param_map(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: RotationTemporalOptions,
    method: Literal["nearest", "linear"],
    prepared: PreparedParamEvaluation | None,
) -> PreparedParamEvaluation:
    return prepare_runtime_param_evaluation(
        context,
        query=query,
        options=ParamMapOptions(method=method, duplicate_policy=opts.duplicate_policy),
        param_kind=context.param_kind,
        query_dim=opts.query_dim,
        reuse=() if prepared is None else (prepared,),
    )


def resolve_rotation_runtime(
    request: RotationTemporalRequest,
    *,
    source: Rotation,
) -> ParamRuntimeContext:
    runtime_source = source if request.on is None else source.set_param_coord(name=request.on, validate=False)
    context = resolve_param_runtime_context(
        runtime_source,
        on=request.on,
        sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        sequence_size_coord=request.sequence_size_coord,
    )
    assert_query_dim_safe(
        context.ds,
        sequence_dim=context.sequence_dim,
        query_dim=request.opts.query_dim,
        owner=request.owner,
    )
    assert_reserved_metadata_safe(
        context.ds,
        reserved=("valid", "sample_index"),
        param_name=context.spec.name,
        owner=request.owner,
    )
    return context


def prepare_rotation_temporal_evaluation(
    request: RotationTemporalRequest,
) -> tuple[ParamRuntimeContext, PreparedParamEvaluation, QueryOutputPlan]:
    context = resolve_rotation_runtime(request, source=request.rotation)
    output_plan = preflight_query_output_namespace(
        context.ds,
        request.query,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        owner=request.owner,
        intent="trajectory",
        generated_names=generated_query_coordinate_names(
            operation="evaluate",
            param_name=context.spec.name,
            size_name=context.sequence_size_coord,
            trajectory=True,
            mapped_dataset=False,
        ),
    )
    method = resolve_rotation_method(request.opts)
    evaluation = build_quat_param_map(
        context,
        query=request.query,
        opts=request.opts,
        method="nearest" if method == "nearest" else "linear",
        prepared=request.prepared,
    )
    return context, evaluation, replace(output_plan, topology=evaluation.query_topology)


__all__: list[str] = []
