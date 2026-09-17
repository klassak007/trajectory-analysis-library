from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.component_ops import read_components
from tal.core.component_ops.runtime_checks import select_component_var
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.param_engine.prepared import PreparedParamEvaluation
from tal.core.param_engine.query_topology import (
    QueryOutputPlan,
    generated_query_coordinate_names,
    preflight_query_output_namespace,
)
from tal.core.param_ops.evaluate import evaluate_param
from tal.core.param_ops.guards import reserved_coord_is_owned
from tal.core.schema_read import read_roles

from ..metadata import get_pose_rep
from ..temporal.options import (
    PoseTemporalOptions,
    as_rotation_method,
    resolve_rotation_method,
)
from .temporal_structural_validity import (
    declare_no_usable_sequence_rows,
    has_no_usable_sequence_rows,
)

if TYPE_CHECKING:
    from ..pose import Pose
    from ..position import Position
    from ..rotation import Rotation


@dataclass(frozen=True)
class PoseTemporalRequest:
    pose: Pose
    query: xr.DataArray | np.ndarray | Sequence[float] | float
    on: str | None
    opts: PoseTemporalOptions
    validate: bool
    sequence_dim: str | None
    batch_dims: Sequence[str] | None
    sequence_size_coord: str | None
    owner: str
    mode: Literal["at", "resample_to"]
    prepared: PreparedPoseEvaluation | None = None


@dataclass(frozen=True)
class PreparedPoseEvaluation:
    """Prepared linear and rotation maps for one Pose request."""

    position: PreparedParamEvaluation
    rotation: PreparedParamEvaluation


def _effective_param_key(request: PoseTemporalRequest) -> str | None:
    if request.on is None:
        return request.opts.on
    if request.opts.on is None or request.opts.on == request.on:
        return request.on
    raise ValueError(
        f"{request.owner}: conflicting on= value between argument ({request.on!r}) "
        f"and opts.on ({request.opts.on!r})."
    )


def _eval_position(
    position: Position,
    request: PoseTemporalRequest,
    *,
    on: str | None,
) -> Position:
    source = position if on is None else position.set_param_coord(name=on, validate=False)
    context = resolve_param_runtime_context(
        source,
        on=on,
        sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        sequence_size_coord=request.sequence_size_coord,
    )
    return evaluate_param(
        context,
        query=request.query,
        opts=request.opts.position_opts,
        validate=False,
        prepared=None if request.prepared is None else request.prepared.position,
        output_intent="trajectory",
    )


def _eval_rotation(
    rotation: Rotation,
    request: PoseTemporalRequest,
    *,
    on: str | None,
) -> tuple[Rotation, QueryOutputPlan]:
    source = rotation if on is None else rotation.set_param_coord(name=on, validate=False)
    plan = _rotation_query_output_plan(source, request, on=on)
    resolved = as_rotation_method(request.opts.rotation_opts, method=resolve_rotation_method(request.opts.rotation_opts))
    if request.prepared is not None:
        from .rotation_temporal_ops import rotation_param_at

        result = rotation_param_at(
            source,
            query=request.query,
            on=on,
            opts=resolved,
            validate=False,
            sequence_dim=request.sequence_dim,
            batch_dims=request.batch_dims,
            sequence_size_coord=request.sequence_size_coord,
            owner=request.owner,
            prepared=request.prepared.rotation,
        )
        return result, plan
    kwargs = {
        "on": on,
        "opts": resolved,
        "validate": False,
        "sequence_dim": request.sequence_dim,
        "batch_dims": request.batch_dims,
        "sequence_size_coord": request.sequence_size_coord,
    }
    if request.mode == "at":
        return source.param.at(request.query, **kwargs), plan
    return source.param.resample_to(request.query, **kwargs), plan


def _rotation_query_output_plan(
    source: Rotation,
    request: PoseTemporalRequest,
    *,
    on: str | None,
) -> QueryOutputPlan:
    context = resolve_param_runtime_context(
        source, on=on, sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        sequence_size_coord=request.sequence_size_coord,
    )
    return preflight_query_output_namespace(
        context.ds, request.query,
        sequence_dim=context.sequence_dim, batch_dims=context.batch_dims,
        owner=request.owner, intent="trajectory",
        generated_names=generated_query_coordinate_names(
            operation="evaluate", param_name=context.spec.name,
            size_name=context.sequence_size_coord, trajectory=True,
            mapped_dataset=False,
        ),
    )


def _eval_payload_carrier(
    source: Pose,
    request: PoseTemporalRequest,
    *,
    on: str | None,
) -> xr.Dataset:
    carrier = AnalysisObject._from_unvalidated(analysis_object_dataset(source))
    if on is not None:
        carrier = carrier.set_param_coord(name=on, validate=False)
    context = resolve_param_runtime_context(
        carrier,
        on=on,
        sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        sequence_size_coord=request.sequence_size_coord,
    )
    evaluated = evaluate_param(
        context,
        query=request.query,
        opts=request.opts.position_opts,
        validate=False,
        prepared=None if request.prepared is None else request.prepared.position,
        output_intent="trajectory",
    )
    return analysis_object_dataset(evaluated)


def _overlay_components_payload(
    carrier_ds: xr.Dataset,
    typed_ds: xr.Dataset,
    *,
    owner: str,
) -> xr.Dataset:
    merged = carrier_ds.copy(deep=True)
    carrier_registry = read_components(carrier_ds)
    typed_registry = read_components(typed_ds)
    for component_name in ("position", "rotation"):
        carrier_spec = carrier_registry[component_name]
        typed_spec = typed_registry[component_name]
        carrier_var = select_component_var(
            carrier_ds,
            spec=carrier_spec,
            component_name=component_name,
            owner=owner,
            operand="carrier pose",
        )
        typed_var = select_component_var(
            typed_ds,
            spec=typed_spec,
            component_name=component_name,
            owner=owner,
            operand="typed pose",
        )
        merged[carrier_var] = typed_ds[typed_var]
    return merged


def _without_redundant_rotation_query_coords(
    rotation: Rotation,
    *,
    source: Pose,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    plan: QueryOutputPlan,
) -> Rotation:
    """Let the position/carrier own caller-only labels during recomposition."""
    if not isinstance(query, xr.DataArray):
        return rotation
    source_ds = analysis_object_dataset(source)
    dataset = analysis_object_dataset(rotation)
    remove = tuple(
        name for name in query.coords
        if name in dataset.coords and name not in plan.generated_names
        and name != plan.sequence_dim
        and (name not in source_ds.coords or reserved_coord_is_owned(source_ds, name=name))
    )
    if not remove:
        return rotation
    return rotation._rewrap_dataset(dataset.drop_vars(remove), validate=False)


def _resolve_matrix_payload_var(
    ds: xr.Dataset,
    *,
    owner: str,
    what: str,
) -> str:
    _, _, _, core_dims = read_roles(ds)
    if len(core_dims) != 2:
        raise ValueError(f"{owner}: {what} requires exactly two matrix core dims; got {core_dims!r}.")
    row_dim, col_dim = core_dims
    candidates = [
        str(name)
        for name, var in ds.data_vars.items()
        if np.issubdtype(np.dtype(var.dtype), np.number) and row_dim in var.dims and col_dim in var.dims
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"{owner}: {what} requires one numeric matrix payload var containing core dims "
            f"{(row_dim, col_dim)!r}; found none."
        )
    raise ValueError(
        f"{owner}: {what} is ambiguous; found multiple numeric matrix payload vars "
        f"{candidates!r} containing core dims {(row_dim, col_dim)!r}."
    )


def _matrix_only_pose_source(
    source: Pose,
    *,
    owner: str,
) -> tuple[Pose, str]:
    source_ds = analysis_object_dataset(source)
    matrix_var = _resolve_matrix_payload_var(
        source_ds,
        owner=owner,
        what="pose temporal source matrix payload",
    )
    matrix_only = source_ds[[matrix_var]]
    return source.__class__._from_unvalidated(matrix_only), matrix_var


def _matrix_core_dim_renames(
    source_core: tuple[str, ...],
    typed_core: tuple[str, ...],
    *,
    owner: str,
) -> dict[str, str]:
    if not source_core or not typed_core:
        return {}
    if len(source_core) != len(typed_core):
        raise ValueError(
            f"{owner}: source/typed matrix core dims are incompatible: "
            f"{source_core!r} vs {typed_core!r}."
        )
    return {
        typed_dim: source_dim
        for typed_dim, source_dim in zip(typed_core, source_core, strict=True)
        if typed_dim != source_dim
    }


def _overlay_matrix_payload(
    carrier_ds: xr.Dataset,
    typed_ds: xr.Dataset,
    *,
    source_matrix_var: str,
    owner: str,
) -> xr.Dataset:
    if source_matrix_var not in carrier_ds.data_vars:
        raise ValueError(f"{owner}: carrier dataset is missing source matrix var {source_matrix_var!r}.")
    typed_var = _resolve_matrix_payload_var(
        typed_ds,
        owner=owner,
        what="pose temporal typed matrix payload",
    )
    carrier_da = carrier_ds[source_matrix_var]
    typed_da = typed_ds[typed_var]
    _, source_seq, _, source_core = read_roles(carrier_ds)
    _, typed_seq, _, typed_core = read_roles(typed_ds)
    rename_map = _matrix_core_dim_renames(source_core, typed_core, owner=owner)
    if typed_seq is not None and source_seq is not None and typed_seq != source_seq:
        rename_map[typed_seq] = source_seq
    if rename_map:
        typed_da = typed_da.reset_coords(drop=True).rename(rename_map)
    if set(typed_da.dims) != set(carrier_da.dims):
        raise ValueError(
            f"{owner}: typed matrix payload dims {list(typed_da.dims)!r} are incompatible with "
            f"carrier dims {list(carrier_da.dims)!r}."
        )
    typed_da = typed_da.transpose(*carrier_da.dims)
    for dim in carrier_da.dims:
        if dim in carrier_ds.coords and dim in typed_da.dims:
            typed_da = typed_da.assign_coords({dim: carrier_ds.coords[dim]})
    merged = carrier_ds.copy(deep=True)
    merged[source_matrix_var] = typed_da
    return merged


def _needs_matrix_aux_rebind(
    *,
    validate: bool,
    matrix_template_ds: xr.Dataset | None,
    merged_ds: xr.Dataset,
) -> bool:
    return (not validate) and matrix_template_ds is not None and len(merged_ds.data_vars) > 1


def _require_matrix_aux_rebind_compatibility(
    *,
    matrix_template_ds: xr.Dataset,
    merged_ds: xr.Dataset,
    owner: str,
) -> None:
    template_var = _resolve_matrix_payload_var(
        matrix_template_ds,
        owner=owner,
        what="pose temporal matrix template payload",
    )
    merged_var = _resolve_matrix_payload_var(
        merged_ds,
        owner=owner,
        what="pose temporal merged matrix payload",
    )
    template_shape = tuple(int(size) for size in matrix_template_ds[template_var].shape)
    merged_shape = tuple(int(size) for size in merged_ds[merged_var].shape)
    if len(template_shape) != len(merged_shape):
        raise ValueError(
            f"{owner}: validate=False matrix aux rebind requires matching matrix payload rank; "
            f"template={template_shape!r}, merged={merged_shape!r}."
        )
    if tuple(sorted(template_shape)) != tuple(sorted(merged_shape)):
        raise ValueError(
            f"{owner}: validate=False matrix aux rebind requires matching matrix payload shapes; "
            f"template={template_shape!r}, merged={merged_shape!r}."
        )


def _rewrap_matrix_aux_unvalidated(
    source: Pose,
    *,
    merged_ds: xr.Dataset,
    matrix_template_ds: xr.Dataset,
    owner: str,
) -> Pose:
    _require_matrix_aux_rebind_compatibility(
        matrix_template_ds=matrix_template_ds,
        merged_ds=merged_ds,
        owner=owner,
    )
    out = source._rewrap_dataset(matrix_template_ds, validate=False)
    out._bind_dataset(merged_ds)
    return out


def _rewrap_pose_temporal_output(
    source: Pose,
    *,
    merged_ds: xr.Dataset,
    validate: bool,
) -> Pose:
    return source._rewrap_dataset(merged_ds, validate=validate)


def _empty_matrix_temporal_output(
    source: Pose,
    carrier_ds: xr.Dataset,
    *,
    request: PoseTemporalRequest,
) -> Pose:
    """Keep unavailable matrix rows rigid while validity marks them missing."""
    var_name = _resolve_matrix_payload_var(
        carrier_ds, owner=request.owner, what="pose temporal empty matrix payload",
    )
    _, _, _, core_dims = read_roles(carrier_ds)
    row_dim, col_dim = core_dims
    matrix = carrier_ds[var_name]
    identity = xr.DataArray(
        np.eye(4, dtype=np.float64), dims=(row_dim, col_dim),
        coords={dim: carrier_ds.coords[dim] for dim in core_dims if dim in carrier_ds.coords},
    )
    if "valid" not in carrier_ds.coords:
        raise ValueError(f"{request.owner}: empty matrix result is missing generated validity.")
    filled = xr.where(carrier_ds.coords["valid"], matrix, identity, keep_attrs=True)
    filled = filled.transpose(*matrix.dims)
    filled.encoding = matrix.encoding.copy()
    output = declare_no_usable_sequence_rows(
        carrier_ds.assign({var_name: filled}), owner=request.owner,
    )
    if not request.validate and len(output.data_vars) > 1:
        return _rewrap_matrix_aux_unvalidated(
            source, merged_ds=output, matrix_template_ds=output[[var_name]],
            owner=request.owner,
        )
    return _rewrap_pose_temporal_output(source, merged_ds=output, validate=request.validate)


def _assemble_pose_temporal_payload(
    carrier_ds: xr.Dataset,
    recomposed: Pose,
    *,
    source_rep: str,
    source_matrix_var: str | None,
    owner: str,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    if source_rep != "matrix":
        merged = _overlay_components_payload(
            carrier_ds, analysis_object_dataset(recomposed), owner=owner
        )
        return merged, None
    if source_matrix_var is None:
        raise ValueError(f"{owner}: missing source matrix payload selection.")
    template = analysis_object_dataset(recomposed.as_matrix(validate=False))
    merged = _overlay_matrix_payload(
        carrier_ds, template, source_matrix_var=source_matrix_var, owner=owner
    )
    return merged, template


def _run_pose_temporal_request(request: PoseTemporalRequest) -> Pose:
    source = request.pose
    source_ds = analysis_object_dataset(source)
    source_rep = get_pose_rep(source_ds, owner=request.owner)
    empty_source = has_no_usable_sequence_rows(source_ds, owner=request.owner)
    on = _effective_param_key(request)
    carrier_ds = _eval_payload_carrier(source, request, on=on)
    matrix_source_var: str | None = None
    decompose_source = source
    if source_rep == "matrix":
        decompose_source, matrix_source_var = _matrix_only_pose_source(source, owner=request.owner)
    position, rotation = decompose_source.decompose(validate=False)
    position_out = _eval_position(position, request, on=on)
    rotation_out, rotation_plan = _eval_rotation(rotation, request, on=on)
    rotation_out = _without_redundant_rotation_query_coords(
        rotation_out, source=source, query=request.query, plan=rotation_plan,
    )
    recomposed = source.__class__.from_components(rotation_out, position_out, validate=False)
    if source_rep == "matrix" and empty_source:
        return _empty_matrix_temporal_output(source, carrier_ds, request=request)
    merged_ds, matrix_template_ds = _assemble_pose_temporal_payload(
        carrier_ds,
        recomposed,
        source_rep=source_rep,
        source_matrix_var=matrix_source_var,
        owner=request.owner,
    )
    if empty_source:
        merged_ds = declare_no_usable_sequence_rows(merged_ds, owner=request.owner)
    if _needs_matrix_aux_rebind(
        validate=request.validate,
        matrix_template_ds=matrix_template_ds,
        merged_ds=merged_ds,
    ):
        if matrix_template_ds is None:
            raise ValueError(f"{request.owner}: missing matrix template for validate=False matrix aux rebind.")
        return _rewrap_matrix_aux_unvalidated(
            source,
            merged_ds=merged_ds,
            matrix_template_ds=matrix_template_ds,
            owner=request.owner,
        )
    return _rewrap_pose_temporal_output(
        source,
        merged_ds=merged_ds,
        validate=request.validate,
    )


def pose_param_at(
    pose: Pose,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    on: str | None,
    opts: PoseTemporalOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
    prepared: PreparedPoseEvaluation | None = None,
) -> Pose:
    request = PoseTemporalRequest(
        pose=pose,
        query=query,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
        mode="at",
        prepared=prepared,
    )
    try:
        return _run_pose_temporal_request(request)
    except (TypeError, ValueError) as exc:
        text = str(exc)
        if text.startswith(f"{owner}:"):
            raise
        raise type(exc)(f"{owner}: {text}") from exc


def pose_param_resample_to(
    pose: Pose,
    *,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float,
    on: str | None,
    opts: PoseTemporalOptions,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> Pose:
    request = PoseTemporalRequest(
        pose=pose,
        query=grid,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
        mode="resample_to",
    )
    try:
        return _run_pose_temporal_request(request)
    except (TypeError, ValueError) as exc:
        text = str(exc)
        if text.startswith(f"{owner}:"):
            raise
        raise type(exc)(f"{owner}: {text}") from exc


__all__ = ["PreparedPoseEvaluation", "pose_param_at", "pose_param_resample_to"]
