"""Unfinalized numerical parts for Pose temporal evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from tal.core.component_ops.types import ComponentSpec
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.lazy import payload_chunks_for_dim
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec
from tal.core.param_engine.map_apply import apply_param_map_with_batch_dims
from tal.core.param_ops.evaluate import _ParamDatasetPart, _prepare_param_dataset_part
from tal.core.param_ops.guards import mark_generated_size_coord
from tal.core.param_ops.types import ParamRuntimeContext
from tal.core.schema_read import read_roles
from tal.core.schema_validate.finalize import transfer_dataarray_metadata
from tal.core.validity_finalize import assign_sequence_size_from_valid_mask
from tal.utils.xarray_namespace import dataset_namespace_names, unique_temp_dim

from ..metadata import get_pose_rep
from ..temporal.options import as_rotation_method, resolve_rotation_method
from .pose_component_ops import resolve_pose_component_specs
from .pose_kernel_adapters import apply_matrix_to_components_kernel
from .pose_layout import enforce_pose_layout
from .pose_matrix_validation import (
    prepare_pose_matrix_for_conversion,
    select_pose_matrix_payload,
)
from .pose_temporal_types import PoseTemporalRequest
from .rotation_temporal_ops import _evaluate_quat_temporal_part
from .temporal_structural_validity import has_no_usable_sequence_rows

_XYZ = ("x", "y", "z")
_QUAT = ("x", "y", "z", "w")


@dataclass(frozen=True)
class PoseTemporalParts:
    """One carrier and its unwrapped typed numerical parts."""

    carrier: xr.Dataset
    position: xr.DataArray
    rotation: xr.DataArray
    schema: CoreSchemaFinalizeSpec
    components: tuple[tuple[str, ComponentSpec], ...] | None
    representation: str
    position_var: str
    rotation_var: str
    position_dim: str
    rotation_dim: str
    matrix_core_dims: tuple[str, str] | None
    has_no_usable_rows: bool
    output_plan: object
    query_topology: object


@dataclass(frozen=True)
class _PoseSourcePlan:
    representation: str
    rotation: xr.DataArray | None
    position_var: str
    rotation_var: str
    position_dim: str
    rotation_dim: str
    matrix_core_dims: tuple[str, str] | None
    selected_vars: tuple[str, ...]


def _available_dim(base: str, ds: xr.Dataset) -> str:
    if base not in ds.dims:
        return base
    index = 2
    while f"{base}_{index}" in ds.dims:
        index += 1
    return f"{base}_{index}"


def _matrix_source_spec(
    ds: xr.Dataset,
    *,
    owner: str,
) -> tuple[str, str, str, tuple[str, str]]:
    _, _, _, core_dims = read_roles(ds)
    if len(core_dims) != 2:
        raise ValueError(f"{owner}: matrix Pose requires exactly two core dimensions.")
    row_dim, col_dim = core_dims
    matrix_var = select_pose_matrix_payload(
        ds,
        core_dims=(row_dim, col_dim),
        owner=owner,
    )
    pos_dim = _available_dim("axis", ds)
    quat_dim = _available_dim("quat", ds)
    return matrix_var, pos_dim, quat_dim, (row_dim, col_dim)


def _matrix_source_parts(
    ds: xr.Dataset,
    *,
    matrix_var: str,
    pos_dim: str,
    quat_dim: str,
    matrix_core: tuple[str, str],
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    row_dim, col_dim = matrix_core
    prepared = prepare_pose_matrix_for_conversion(ds[[matrix_var]], owner=owner)
    position, rotation = apply_matrix_to_components_kernel(
        prepared,
        row_dim=row_dim,
        col_dim=col_dim,
        pos_dim=pos_dim,
        quat_dim=quat_dim,
    )
    return position, rotation


def _component_source_parts(
    ds: xr.Dataset,
    *,
    owner: str,
) -> tuple[xr.DataArray, str, str, str]:
    _, _, _, core_dims = read_roles(ds)
    (pos_dim, pos_var), (_quat_dim, rot_var) = resolve_pose_component_specs(
        ds, owner=owner, core_dims=core_dims
    )
    return ds[rot_var], pos_var, rot_var, pos_dim


def _trajectory_dim(
    values: xr.DataArray,
    *,
    query_dim: str,
    context: ParamRuntimeContext,
    carrier: xr.Dataset,
) -> xr.DataArray:
    out = (
        values.rename({query_dim: context.sequence_dim})
        if query_dim != context.sequence_dim
        else values
    )
    if context.sequence_dim in carrier.coords:
        out = out.assign_coords({context.sequence_dim: carrier.coords[context.sequence_dim]})
    return out


def _empty_vector(
    valid: xr.DataArray,
    *,
    dim: str,
    values: tuple[float, ...],
    lazy_reference: xr.DataArray,
) -> xr.DataArray:
    labels = _XYZ if len(values) == 3 else _QUAT
    valid_chunks = getattr(valid.data, "chunks", None)
    reference_chunks = getattr(lazy_reference.data, "chunks", None)
    if valid_chunks is None and reference_chunks is None:
        basis = xr.DataArray(np.asarray(values), dims=dim, coords={dim: list(labels)})
        result = xr.ones_like(valid, dtype=np.float64) * basis
        result.attrs = {}
        result.encoding = {}
        return result
    import dask.array as da

    template = valid.expand_dims({dim: list(labels)}, axis=-1)
    chunks = tuple(
        payload_chunks_for_dim(valid, dim=logical_dim)
        if valid_chunks is not None
        else (int(valid.sizes[logical_dim]),)
        for logical_dim in valid.dims
    ) + ((len(values),),)
    basis = da.from_array(np.asarray(values), chunks=(len(values),))
    data = da.broadcast_to(basis, template.shape, chunks=chunks)
    return xr.DataArray(data, dims=template.dims, coords=template.coords)


def _empty_payload_topology(
    carrier: xr.Dataset,
    *,
    context: ParamRuntimeContext,
) -> xr.DataArray:
    valid = carrier.coords["valid"]
    missing = tuple(dim for dim in context.batch_dims if dim not in valid.dims)
    if missing:
        valid = valid.expand_dims({dim: carrier.coords[dim] for dim in missing})
    dims = tuple(dim for dim in (*context.batch_dims, context.sequence_dim) if dim in valid.dims)
    return valid.transpose(*dims)


def _retain_variable_metadata(
    result: xr.DataArray,
    source: xr.DataArray,
) -> xr.DataArray:
    return transfer_dataarray_metadata(source, result)


def _ensure_no_rows_size(
    carrier: _ParamDatasetPart,
    *,
    context: ParamRuntimeContext,
) -> tuple[_ParamDatasetPart, str]:
    name = context.sequence_size_coord or unique_temp_dim(
        f"{context.sequence_dim}_size", taken_dims=dataset_namespace_names(carrier.dataset),
    )
    zeros = xr.DataArray(
        np.zeros(
            tuple(int(carrier.dataset.sizes[dim]) for dim in context.batch_dims),
            dtype=np.int64,
        ),
        dims=context.batch_dims,
    )
    dataset = carrier.dataset.assign_coords({name: zeros})
    return replace(carrier, dataset=mark_generated_size_coord(dataset, name=name)), name


def _resolve_output_validity(
    carrier: _ParamDatasetPart,
    *,
    context: ParamRuntimeContext,
    no_rows: bool,
) -> tuple[_ParamDatasetPart, str | None]:
    if no_rows:
        return _ensure_no_rows_size(carrier, context=context)
    size_name = context.sequence_size_coord
    if size_name is None:
        return carrier, None
    valid = carrier.dataset.coords["valid"]
    dataset = carrier.dataset.drop_vars(size_name, errors="ignore")
    dataset = dataset.assign_coords(valid=valid)
    dataset, kept = assign_sequence_size_from_valid_mask(
        dataset,
        valid=valid,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        sequence_size_coord=size_name,
    )
    if kept is not None:
        dataset = mark_generated_size_coord(dataset, name=kept)
    return replace(carrier, dataset=dataset), kept


def _map_position(
    source: xr.DataArray,
    *,
    context: ParamRuntimeContext,
    part,
) -> xr.DataArray:
    mapped = apply_param_map_with_batch_dims(
        source,
        param_map=part.evaluation.param_map,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
    )
    return _retain_variable_metadata(_trajectory_dim(
        mapped,
        query_dim=part.evaluation.param_map.query_dim,
        context=context,
        carrier=part.dataset,
    ), source)


def _rotation_part(
    source: xr.DataArray,
    *,
    context: ParamRuntimeContext,
    request: PoseTemporalRequest,
    quat_dim: str,
    carrier,
) -> xr.DataArray:
    rotation_context = replace(
        context,
        ds=source.to_dataset(name="rotation"),
        core_dims=(quat_dim,),
    )
    opts = as_rotation_method(
        request.opts.rotation_opts,
        method=resolve_rotation_method(request.opts.rotation_opts),
    )
    values, _, _, query_dim = _evaluate_quat_temporal_part(
        rotation_context,
        var_name="rotation",
        quat_dim=quat_dim,
        query=request.query,
        opts=opts,
        prepared=None if request.prepared is None else request.prepared.rotation,
    )
    return _retain_variable_metadata(
        _trajectory_dim(
            values, query_dim=query_dim, context=context, carrier=carrier.dataset
        ),
        source,
    )


def _prepare_source_plan(
    ds: xr.Dataset,
    *,
    context: ParamRuntimeContext,
    owner: str,
    validate: bool,
) -> _PoseSourcePlan:
    enforce_pose_layout(
        ds,
        owner=owner,
        schema_prepared=True,
        allow_matrix_auxiliary=not validate,
    )
    representation = get_pose_rep(ds, owner=owner)
    if representation == "components":
        rotation, pos_var, rot_var, pos_dim = _component_source_parts(
            ds, owner=owner
        )
        quat_dim = next(dim for dim in rotation.dims if dim in context.core_dims)
        return _PoseSourcePlan(
            representation=representation,
            rotation=rotation,
            position_var=pos_var,
            rotation_var=rot_var,
            position_dim=pos_dim,
            rotation_dim=quat_dim,
            matrix_core_dims=None,
            selected_vars=tuple(name for name in ds.data_vars if name != rot_var),
        )
    matrix_var, pos_dim, quat_dim, matrix_core = _matrix_source_spec(
        ds, owner=owner
    )
    return _PoseSourcePlan(
        representation=representation,
        rotation=None,
        position_var=matrix_var,
        rotation_var=matrix_var,
        position_dim=pos_dim,
        rotation_dim=quat_dim,
        matrix_core_dims=matrix_core,
        selected_vars=tuple(name for name in ds.data_vars if name != matrix_var),
    )


def _prepare_carrier(
    ds: xr.Dataset,
    *,
    context: ParamRuntimeContext,
    request: PoseTemporalRequest,
    source: _PoseSourcePlan,
) -> tuple[_ParamDatasetPart, str | None, bool]:
    carrier = _prepare_param_dataset_part(
        context,
        query=request.query,
        opts=request.opts.position_opts,
        prepared=None if request.prepared is None else request.prepared.position,
        output_intent="trajectory",
        data_vars=source.selected_vars,
        owner=request.owner,
        copy_schema=False,
    )
    no_rows = (
        has_no_usable_sequence_rows(ds, owner=request.owner)
        or carrier.evaluation.has_no_rows
    )
    carrier, size_name = _resolve_output_validity(
        carrier, context=context, no_rows=no_rows,
    )
    return carrier, size_name, no_rows


def _empty_payload_parts(
    ds: xr.Dataset,
    *,
    context: ParamRuntimeContext,
    carrier: _ParamDatasetPart,
    source: _PoseSourcePlan,
) -> tuple[xr.DataArray, xr.DataArray]:
    topology = _empty_payload_topology(carrier.dataset, context=context)
    position = (
        carrier.dataset[source.position_var]
        if source.representation == "components"
        else _empty_vector(
            topology,
            dim=source.position_dim,
            values=(0.0, 0.0, 0.0),
            lazy_reference=ds[source.position_var],
        )
    )
    rotation = _empty_vector(
        topology,
        dim=source.rotation_dim,
        values=(0.0, 0.0, 0.0, 1.0),
        lazy_reference=position,
    )
    if source.rotation is not None:
        rotation = _retain_variable_metadata(rotation, source.rotation)
    return position, rotation


def _mapped_matrix_parts(
    ds: xr.Dataset,
    *,
    context: ParamRuntimeContext,
    carrier: _ParamDatasetPart,
    request: PoseTemporalRequest,
    source: _PoseSourcePlan,
) -> tuple[xr.DataArray, xr.DataArray]:
    if source.matrix_core_dims is None:
        raise RuntimeError(f"{request.owner}: matrix core dimensions were not prepared.")
    position, rotation = _matrix_source_parts(
        ds,
        matrix_var=source.position_var,
        pos_dim=source.position_dim,
        quat_dim=source.rotation_dim,
        matrix_core=source.matrix_core_dims,
        owner=request.owner,
    )
    mapped = _map_position(position, context=context, part=carrier)
    rotated = _rotation_part(
        rotation, context=context, request=request,
        quat_dim=source.rotation_dim, carrier=carrier,
    )
    return mapped, rotated


def _mapped_payload_parts(
    ds: xr.Dataset,
    *,
    context: ParamRuntimeContext,
    carrier: _ParamDatasetPart,
    request: PoseTemporalRequest,
    source: _PoseSourcePlan,
) -> tuple[xr.DataArray, xr.DataArray]:
    if source.representation != "components":
        return _mapped_matrix_parts(
            ds, context=context, carrier=carrier, request=request, source=source,
        )
    if source.rotation is None:
        raise RuntimeError(f"{request.owner}: rotation source was not prepared.")
    rotation = _rotation_part(
        source.rotation, context=context, request=request,
        quat_dim=source.rotation_dim, carrier=carrier,
    )
    return carrier.dataset[source.position_var], rotation


def _output_declarations(
    source: _PoseSourcePlan,
    *,
    owner: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ComponentSpec], ...] | None]:
    if source.representation == "components":
        return (
            (source.position_dim, source.rotation_dim),
            (
                ("position", ComponentSpec(source.position_dim, _XYZ, var=source.position_var)),
                ("rotation", ComponentSpec(source.rotation_dim, _QUAT, var=source.rotation_var)),
            ),
        )
    if source.matrix_core_dims is None:
        raise RuntimeError(f"{owner}: matrix core dimensions were not prepared.")
    return source.matrix_core_dims, None


def _prepare_context(
    request: PoseTemporalRequest,
    *,
    on: str | None,
) -> ParamRuntimeContext:
    return resolve_param_runtime_context(
        request.pose,
        on=on,
        sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        sequence_size_coord=request.sequence_size_coord,
        allow_declared_param_override=on is not None,
    )


def _output_schema(
    context: ParamRuntimeContext,
    *,
    core_dims: tuple[str, ...],
    size_name: str | None,
) -> CoreSchemaFinalizeSpec:
    return CoreSchemaFinalizeSpec(
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        core_dims=core_dims,
        param_name=context.spec.name,
        size_name=size_name,
    )


def prepare_pose_temporal_parts(
    request: PoseTemporalRequest,
    *,
    on: str | None,
) -> PoseTemporalParts:
    source = request.pose
    ds = analysis_object_dataset(source)
    context = _prepare_context(request, on=on)
    source_plan = _prepare_source_plan(
        ds,
        context=context,
        owner=request.owner,
        validate=request.validate,
    )
    carrier, size_name, no_rows = _prepare_carrier(
        ds, context=context, request=request, source=source_plan,
    )
    position, rotation = (
        _empty_payload_parts(
            ds, context=context, carrier=carrier, source=source_plan,
        )
        if no_rows
        else _mapped_payload_parts(
            ds, context=context, carrier=carrier, request=request, source=source_plan,
        )
    )
    core_dims, components = _output_declarations(
        source_plan, owner=request.owner,
    )
    return PoseTemporalParts(
        carrier=carrier.dataset,
        position=position,
        rotation=rotation,
        schema=_output_schema(
            context,
            core_dims=core_dims,
            size_name=size_name,
        ),
        components=components,
        representation=source_plan.representation,
        position_var=source_plan.position_var,
        rotation_var=source_plan.rotation_var,
        position_dim=source_plan.position_dim,
        rotation_dim=source_plan.rotation_dim,
        matrix_core_dims=source_plan.matrix_core_dims,
        has_no_usable_rows=no_rows,
        output_plan=carrier.output_plan,
        query_topology=carrier.evaluation.query_topology,
    )


__all__: list[str] = []
