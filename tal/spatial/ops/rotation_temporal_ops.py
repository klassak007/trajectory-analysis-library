from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.param_engine import ParamMapOptions, build_param_map, normalize_query_grid
from tal.core.param_engine.map_apply import gather_along_sequence
from tal.core.param_ops.finalize import finalize_param_output
from tal.core.param_ops.guards import assert_query_dim_safe, assert_reserved_metadata_safe
from tal.core.param_ops.types import ParamRuntimeContext

from ..kernels.rotation_interp_backends import ROTATION_INTERP_BACKEND_SCIPY, slerp_quat_backend
from ..metadata import get_rotation_rep
from ..temporal.options import RotationTemporalOptions, resolve_rotation_method

if TYPE_CHECKING:
    from ..rotation import Rotation


@dataclass(frozen=True)
class RotationTemporalRequest:
    rotation: Rotation
    query: xr.DataArray | np.ndarray | Sequence[float] | float
    on: str | None
    opts: RotationTemporalOptions
    validate: bool
    sequence_dim: str | None
    batch_dims: Sequence[str] | None
    sequence_size_coord: str | None
    owner: str


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
    return values.broadcast_like(target)


def _build_base_output_dataset(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    values: xr.DataArray,
) -> xr.Dataset:
    coords = {
        name: coord
        for name, coord in context.ds.coords.items()
        if context.sequence_dim not in coord.dims
    }
    return xr.Dataset(data_vars={var_name: values}, coords=coords)


def _require_single_payload_var(ds: xr.Dataset, *, owner: str) -> str:
    names = tuple(str(name) for name in ds.data_vars.keys())
    if len(names) == 1:
        return names[0]
    raise ValueError(
        f"{owner}: auxiliary payload vars are not supported for typed Rotation.param.*; "
        f"expected exactly one payload variable, got {names!r}. "
        "Interpolate auxiliary vars via AnalysisObject.param.* first."
    )


def _build_quat_param_map(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: RotationTemporalOptions,
    method: Literal["nearest", "linear"],
) -> tuple[xr.DataArray, object]:
    grid = normalize_query_grid(
        query,
        query_dim=opts.query_dim,
        batch_dims=context.batch_dims,
        batch_coords=context.batch_coords,
    )
    pmap = build_param_map(
        param=context.spec.coord,
        query=grid.values,
        sequence_dim=context.sequence_dim,
        query_dim=grid.query_dim,
        valid_mask=context.valid_mask,
        options=ParamMapOptions(method=method, duplicate_policy=opts.duplicate_policy),
    )
    return grid.values, pmap


def _evaluate_quat_nearest_linear(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    quat_dim: str,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: RotationTemporalOptions,
    method: Literal["nearest", "linear"],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, str]:
    query_values, pmap = _build_quat_param_map(context, query=query, opts=opts, method=method)
    source = context.ds[var_name]
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
    normalized = xr.apply_ufunc(
        _normalize_quat_block,
        blended,
        input_core_dims=[[quat_dim]],
        output_core_dims=[[quat_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {quat_dim: 4}},
    )
    normalized = normalized.assign_coords({quat_dim: source.coords[quat_dim]})
    normalized = _order_query_then_core(normalized, query_dim=pmap.query_dim, core_dim=quat_dim)
    return normalized.where(valid, np.nan), query_values, valid, pmap.query_dim


def _evaluate_quat_slerp(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    quat_dim: str,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: RotationTemporalOptions,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, str]:
    query_values, pmap = _build_quat_param_map(context, query=query, opts=opts, method="linear")
    source = context.ds[var_name]
    q0 = gather_along_sequence(
        source,
        pmap.i0,
        sequence_dim=context.sequence_dim,
        query_dim=pmap.query_dim,
        owner="spatial.rotation.temporal",
    )
    q0 = _order_query_then_core(q0, query_dim=pmap.query_dim, core_dim=quat_dim)
    q1 = gather_along_sequence(
        source,
        pmap.i1,
        sequence_dim=context.sequence_dim,
        query_dim=pmap.query_dim,
        owner="spatial.rotation.temporal",
    )
    q1 = _order_query_then_core(q1, query_dim=pmap.query_dim, core_dim=quat_dim)
    alpha = _broadcast_query_da(pmap.alpha, template=q0, core_dim=quat_dim)
    valid = _broadcast_query_da(pmap.valid, template=q0, core_dim=quat_dim)
    out = xr.apply_ufunc(
        slerp_quat_backend,
        q0,
        q1,
        alpha,
        valid,
        kwargs={"backend": ROTATION_INTERP_BACKEND_SCIPY},
        input_core_dims=[[pmap.query_dim, quat_dim], [pmap.query_dim, quat_dim], [pmap.query_dim], [pmap.query_dim]],
        output_core_dims=[[pmap.query_dim, quat_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {quat_dim: 4}},
    )
    out = out.assign_coords({quat_dim: source.coords[quat_dim]})
    out = _order_query_then_core(out, query_dim=pmap.query_dim, core_dim=quat_dim)
    return out.where(valid, np.nan), query_values, valid, pmap.query_dim


def _resolve_runtime(request: RotationTemporalRequest, *, source: Rotation) -> ParamRuntimeContext:
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
) -> Rotation:
    ds_out = _build_base_output_dataset(context, var_name=var_name, values=values)
    evaluated = finalize_param_output(
        context,
        ds_out,
        query=query,
        query_dim=query_dim,
        valid_query=valid_query,
        validate=False,
        trajectory=True,
    )
    result = evaluated if isinstance(evaluated, source.__class__) else source.__class__(evaluated.unsafe_data)
    if source_rep != "quat":
        result = result.to_rep(source_rep, validate=False)
    if validate:
        return source.__class__._from_validated(result.unsafe_data)
    return source.__class__._from_unvalidated(result.unsafe_data)


def _run_rotation_temporal_request(request: RotationTemporalRequest) -> Rotation:
    source = request.rotation
    _require_single_payload_var(source.unsafe_data, owner=request.owner)
    source_rep = get_rotation_rep(source.unsafe_data, owner=request.owner)
    quat_source = source.as_quat(validate=False)
    context = _resolve_runtime(request, source=quat_source)
    var_name = _require_single_payload_var(context.ds, owner=request.owner)
    quat_dim = context.core_dims[0]
    method = resolve_rotation_method(request.opts)
    if method == "slerp":
        out_values, query_values, valid_query, query_dim = _evaluate_quat_slerp(
            context,
            var_name=var_name,
            quat_dim=quat_dim,
            query=request.query,
            opts=request.opts,
        )
    else:
        out_values, query_values, valid_query, query_dim = _evaluate_quat_nearest_linear(
            context,
            var_name=var_name,
            quat_dim=quat_dim,
            query=request.query,
            opts=request.opts,
            method=method,
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
