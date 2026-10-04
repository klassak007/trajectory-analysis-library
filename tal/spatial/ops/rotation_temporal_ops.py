from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.indexing import (
    capture_index_topology,
    restore_index_topology,
    without_index_topology,
)
from tal.core.param_engine import ParamMap
from tal.core.param_engine.blocking import (
    LogicalRowBlock,
    assemble_logical_blocks,
    prepare_logical_row_blocks,
    select_logical_block,
)
from tal.core.param_engine.map_apply import (
    align_param_map_application,
    empty_mapped_value,
    gather_along_sequence,
    gather_sequence_block,
)
from tal.core.param_engine.prepared import PreparedParamEvaluation
from tal.core.param_engine.query_output_verify import verify_query_output_plan
from tal.core.param_engine.query_topology import (
    QueryOutputPlan,
)
from tal.core.param_ops.finalize import finalize_param_output
from tal.core.param_ops.types import ParamRuntimeContext

from ..kernels.rotation_interp_backends import (
    ROTATION_INTERP_BACKEND_AUTO,
    slerp_quat_backend,
)
from ..metadata import get_rotation_rep
from ..temporal.options import RotationTemporalOptions, resolve_rotation_method
from .rotation_temporal_plan import (
    build_quat_param_map,
    prepare_rotation_temporal_evaluation,
    resolve_rotation_runtime,
)
from .rotation_temporal_types import RotationTemporalRequest
from .temporal_structural_validity import (
    declare_no_usable_sequence_rows,
    has_no_usable_sequence_rows,
)

if TYPE_CHECKING:
    from ..rotation import Rotation


def _normalize_quat_block(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    finite = np.isfinite(norms[..., 0]) & (norms[..., 0] > 0.0)
    normalized = np.full_like(out, np.nan, dtype=np.float64)
    np.divide(out, norms, out=normalized, where=finite[..., None])
    return normalized


def _order_query_then_core(values: xr.DataArray, *, query_dim: str, core_dim: str) -> xr.DataArray:
    outer = [dim for dim in values.dims if dim not in {query_dim, core_dim}]
    return values.transpose(*(outer + [query_dim, core_dim]))


def _broadcast_query_da(values: xr.DataArray, *, template: xr.DataArray, core_dim: str) -> xr.DataArray:
    target = template.isel({core_dim: 0}, drop=True)
    snapshot = capture_index_topology(target, dims=tuple(target.dims))
    values = without_index_topology(values, dims=tuple(target.dims))
    target = without_index_topology(target, dims=tuple(target.dims))
    broadcast = values.broadcast_like(target)
    return restore_index_topology(broadcast, snapshot)  # type: ignore[return-value]


def _build_base_output_dataset(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    values: xr.DataArray,
) -> xr.Dataset:
    coords = {
        name: coord
        for name, coord in context.ds.coords.items()
        if context.sequence_dim not in coord.dims and name not in values.coords
    }
    result = xr.Dataset(data_vars={var_name: values})
    return result.assign_coords(coords) if coords else result


def _require_single_payload_var(ds: xr.Dataset, *, owner: str) -> str:
    names = tuple(str(name) for name in ds.data_vars)
    if len(names) == 1:
        return names[0]
    raise ValueError(
        f"{owner}: auxiliary payload vars are not supported for typed Rotation.param.*; "
        f"expected exactly one payload variable, got {names!r}. "
        "Interpolate auxiliary vars via AnalysisObject.param.* first."
    )


def _rotation_temporal_payload(
    context: ParamRuntimeContext,
    *,
    var_name: str,
) -> xr.DataArray:
    source = context.ds[var_name]
    if context.sequence_size_coord in source.coords:
        return source.drop_vars(context.sequence_size_coord)
    return source


def _evaluate_quat_nearest_linear(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    quat_dim: str,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: RotationTemporalOptions,
    method: Literal["nearest", "linear"],
    prepared: PreparedParamEvaluation | None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, str]:
    evaluation = build_quat_param_map(
        context,
        query=query,
        opts=opts,
        method=method,
        prepared=prepared,
    )
    query_values = evaluation.grid.values
    pmap = evaluation.param_map
    source = _rotation_temporal_payload(context, var_name=var_name)
    left = gather_along_sequence(
        source,
        pmap.i0,
        sequence_dim=context.sequence_dim,
        query_dim=pmap.query_dim,
        owner="spatial.rotation.temporal",
    )
    left = _order_query_then_core(left, query_dim=pmap.query_dim, core_dim=quat_dim)
    valid = _broadcast_query_da(pmap.valid, template=left, core_dim=quat_dim)
    if method == "nearest":
        return left.where(valid, np.nan), query_values, valid, pmap.query_dim
    right = gather_along_sequence(
        source,
        pmap.i1,
        sequence_dim=context.sequence_dim,
        query_dim=pmap.query_dim,
        owner="spatial.rotation.temporal",
    )
    blended = (1.0 - pmap.alpha) * left + pmap.alpha * right
    normalized = _normalize_linear_quat(blended, source=source, quat_dim=quat_dim)
    normalized = _order_query_then_core(normalized, query_dim=pmap.query_dim, core_dim=quat_dim)
    return normalized.where(valid, np.nan), query_values, valid, pmap.query_dim


def _normalize_linear_quat(
    blended: xr.DataArray,
    *,
    source: xr.DataArray,
    quat_dim: str,
) -> xr.DataArray:
    normalized = xr.apply_ufunc(
        _normalize_quat_block,
        blended,
        input_core_dims=[[quat_dim]],
        output_core_dims=[[quat_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {quat_dim: 4}, "allow_rechunk": True},
    )
    return normalized.assign_coords({quat_dim: source.coords[quat_dim]})


def _evaluate_quat_slerp(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    quat_dim: str,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: RotationTemporalOptions,
    prepared: PreparedParamEvaluation | None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, str]:
    evaluation = build_quat_param_map(
        context,
        query=query,
        opts=opts,
        method="linear",
        prepared=prepared,
    )
    query_values = evaluation.grid.values
    pmap = evaluation.param_map
    source = _rotation_temporal_payload(context, var_name=var_name)
    out = _apply_mapped_slerp_blocks(
        source=source,
        pmap=pmap,
        sequence_dim=context.sequence_dim,
        quat_dim=quat_dim,
    )
    valid = _broadcast_query_da(pmap.valid, template=out, core_dim=quat_dim)
    out = out.assign_coords({quat_dim: source.coords[quat_dim]})
    out = _order_query_then_core(out, query_dim=pmap.query_dim, core_dim=quat_dim)
    return out.where(valid, np.nan), query_values, valid, pmap.query_dim


def _gather_mapped_quat(
    source: xr.DataArray,
    indexer: xr.DataArray,
    *,
    sequence_dim: str,
    query_dim: str,
    quat_dim: str,
) -> xr.DataArray:
    gathered = gather_sequence_block(
        source,
        indexer,
        sequence_dim=sequence_dim,
        query_dim=query_dim,
    )
    return _order_query_then_core(gathered, query_dim=query_dim, core_dim=quat_dim)


def _mapped_slerp_block(
    source: xr.DataArray,
    pmap: ParamMap,
    block: LogicalRowBlock,
    *,
    sequence_dim: str,
    quat_dim: str,
) -> xr.DataArray:
    query_dim = pmap.query_dim
    source = select_logical_block(source, block)
    indexes = tuple(select_logical_block(value, block) for value in (pmap.i0, pmap.i1))
    gathered = tuple(
        _gather_mapped_quat(
            source,
            indexer,
            sequence_dim=sequence_dim,
            query_dim=query_dim,
            quat_dim=quat_dim,
        )
        for indexer in indexes
    )
    alpha = select_logical_block(pmap.alpha, block)
    valid = select_logical_block(pmap.valid, block)
    alpha = _broadcast_query_da(alpha, template=gathered[0], core_dim=quat_dim)
    valid = _broadcast_query_da(valid, template=gathered[0], core_dim=quat_dim)
    return _slerp_one_block(*gathered, alpha, valid, query_dim=query_dim, quat_dim=quat_dim)


def _slerp_output_template(
    source: xr.DataArray,
    pmap: ParamMap,
    *,
    sequence_dim: str,
    quat_dim: str,
) -> xr.DataArray:
    query_dim = pmap.query_dim
    logical_dims = tuple(dim for dim in source.dims if dim not in {sequence_dim, quat_dim})
    logical_dims += tuple(dim for dim in pmap.i0.dims if dim not in logical_dims)
    unindexed_source = without_index_topology(source, dims=logical_dims)
    unindexed_map = without_index_topology(pmap.i0, dims=logical_dims)
    seed = unindexed_source.isel({sequence_dim: 0}, drop=True)
    template = xr.broadcast(seed, unindexed_map)[0]
    ordered = _order_query_then_core(template, query_dim=query_dim, core_dim=quat_dim)
    prototype = np.broadcast_to(np.empty((), dtype=np.float64), ordered.shape)
    return ordered.copy(data=prototype)


def _apply_mapped_slerp_blocks(
    *,
    source: xr.DataArray,
    pmap: ParamMap,
    sequence_dim: str,
    quat_dim: str,
) -> xr.DataArray:
    source, pmap = align_param_map_application(
        source,
        pmap,
        sequence_dim=sequence_dim,
        owner="spatial.rotation.temporal",
    )
    plan = prepare_logical_row_blocks(
        source,
        pmap.i0,
        excluded_dims=frozenset({sequence_dim, quat_dim}),
        fastest_dim=pmap.query_dim,
    )
    blocks = (
        _mapped_slerp_block(
            source,
            pmap,
            block,
            sequence_dim=sequence_dim,
            quat_dim=quat_dim,
        )
        for block in plan.blocks
    )
    template = _slerp_output_template(
        source,
        pmap,
        sequence_dim=sequence_dim,
        quat_dim=quat_dim,
    )
    return assemble_logical_blocks(
        blocks,
        plan=plan,
        template=template,
        index_sources=(source, pmap.i0),
        owner="spatial.rotation.temporal",
    )


def _slerp_one_block(
    q0: xr.DataArray,
    q1: xr.DataArray,
    alpha: xr.DataArray,
    valid: xr.DataArray,
    *,
    query_dim: str,
    quat_dim: str,
) -> xr.DataArray:
    return xr.apply_ufunc(
        slerp_quat_backend,
        q0,
        q1,
        alpha,
        valid,
        kwargs={"backend": ROTATION_INTERP_BACKEND_AUTO},
        input_core_dims=[[query_dim, quat_dim], [query_dim, quat_dim], [query_dim], [query_dim]],
        output_core_dims=[[query_dim, quat_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {quat_dim: 4}, "allow_rechunk": True},
    )


def _finalize_rotation_temporal_output(
    source: Rotation,
    context: ParamRuntimeContext,
    *,
    var_name: str,
    values: xr.DataArray,
    query: xr.DataArray,
    valid_query: xr.DataArray,
    query_dim: str,
    source_rep: str,
    validate: bool,
    output_plan: QueryOutputPlan,
) -> Rotation:
    ds_out = _build_base_output_dataset(context, var_name=var_name, values=values)
    intermediate_plan = (
        replace(output_plan, source_index_groups=()) if source_rep != "quat" else output_plan
    )
    evaluated = finalize_param_output(
        context,
        ds_out,
        query=query,
        query_dim=query_dim,
        valid_query=valid_query,
        query_topology=output_plan.topology,
        validate=False,
        trajectory=True,
        owner=output_plan.owner,
        output_plan=intermediate_plan,
    )
    result = evaluated if isinstance(evaluated, source.__class__) else source.__class__(analysis_object_dataset(evaluated))
    if source_rep != "quat":
        result = result.to_rep(source_rep, validate=False)
    final = source._rewrap_dataset(
        analysis_object_dataset(result),
        validate=validate,
    )
    if output_plan.topology is not None:
        verify_query_output_plan(
            analysis_object_dataset(final), plan=output_plan, topology=output_plan.topology,
        )
    return final


def _evaluate_empty_rotation(
    request: RotationTemporalRequest,
    *,
    source: Rotation,
    source_rep: str,
    context: ParamRuntimeContext,
    evaluation: PreparedParamEvaluation,
    output_plan: QueryOutputPlan,
) -> Rotation:
    var_name = _require_single_payload_var(
        context.ds,
        owner=request.owner,
    )
    values = empty_mapped_value(
        _rotation_temporal_payload(context, var_name=var_name),
        param_map=evaluation.param_map,
        sequence_dim=context.sequence_dim,
    )
    outer = [
        dim
        for dim in values.dims
        if dim not in {*context.core_dims, evaluation.param_map.query_dim}
    ]
    values = values.transpose(
        *outer,
        evaluation.param_map.query_dim,
        *context.core_dims,
    )
    result = _finalize_rotation_temporal_output(
        source,
        context,
        var_name=var_name,
        values=values,
        query=evaluation.grid.values,
        valid_query=evaluation.param_map.valid,
        query_dim=evaluation.param_map.query_dim,
        source_rep=source_rep,
        validate=request.validate,
        output_plan=output_plan,
    )
    declared = declare_no_usable_sequence_rows(
        analysis_object_dataset(result), owner=request.owner,
    )
    return source._rewrap_dataset(declared, validate=request.validate)


def _evaluate_rotation_request(
    request: RotationTemporalRequest,
    *,
    context: ParamRuntimeContext,
    var_name: str,
    prepared: PreparedParamEvaluation,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, str]:
    return _evaluate_quat_temporal_part(
        context,
        var_name=var_name,
        quat_dim=context.core_dims[0],
        query=request.query,
        opts=request.opts,
        prepared=prepared,
    )


def _evaluate_quat_temporal_part(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    quat_dim: str,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: RotationTemporalOptions,
    prepared: PreparedParamEvaluation | None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, str]:
    """Evaluate one quaternion part without constructing a typed wrapper."""
    method = resolve_rotation_method(opts)
    if method == "slerp":
        return _evaluate_quat_slerp(
            context,
            var_name=var_name,
            quat_dim=quat_dim,
            query=query,
            opts=opts,
            prepared=prepared,
        )
    return _evaluate_quat_nearest_linear(
        context,
        var_name=var_name,
        quat_dim=quat_dim,
        query=query,
        opts=opts,
        method=method,
        prepared=prepared,
    )


def _run_rotation_temporal_request(request: RotationTemporalRequest) -> Rotation:
    source = request.rotation
    source_ds = analysis_object_dataset(source)
    _require_single_payload_var(source_ds, owner=request.owner)
    source_rep = get_rotation_rep(source_ds, owner=request.owner)
    source_context, evaluation, output_plan = prepare_rotation_temporal_evaluation(request)
    if has_no_usable_sequence_rows(source_ds, owner=request.owner) or evaluation.has_no_rows:
        return _evaluate_empty_rotation(
            request,
            source=source,
            source_rep=source_rep,
            context=source_context,
            evaluation=evaluation,
            output_plan=output_plan,
        )
    context = source_context
    if source_rep != "quat":
        quat_source = source.as_quat(validate=False)
        context = resolve_rotation_runtime(request, source=quat_source)
    var_name = _require_single_payload_var(context.ds, owner=request.owner)
    out_values, query_values, valid_query, query_dim = _evaluate_rotation_request(
        request,
        context=context,
        var_name=var_name,
        prepared=evaluation,
    )
    return _finalize_rotation_temporal_output(
        source,
        context,
        var_name=var_name,
        values=out_values,
        query=query_values,
        valid_query=valid_query,
        query_dim=query_dim,
        source_rep=source_rep,
        validate=request.validate,
        output_plan=output_plan,
    )


def rotation_param_at(
    rotation: Rotation,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    on: str | None,
    opts: RotationTemporalOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
    prepared: PreparedParamEvaluation | None = None,
) -> Rotation:
    request = RotationTemporalRequest(
        rotation=rotation,
        query=query,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
        prepared=prepared,
    )
    try:
        return _run_rotation_temporal_request(request)
    except (TypeError, ValueError) as exc:
        text = str(exc)
        if text.startswith(f"{owner}:"):
            raise
        raise type(exc)(f"{owner}: {text}") from exc


def rotation_param_resample_to(
    rotation: Rotation,
    *,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float,
    on: str | None,
    opts: RotationTemporalOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> Rotation:
    return rotation_param_at(
        rotation,
        query=grid,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )


__all__ = ["rotation_param_at", "rotation_param_resample_to"]
