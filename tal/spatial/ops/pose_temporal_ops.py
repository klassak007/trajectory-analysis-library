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
from tal.core.schema_read import read_roles

from ..metadata import get_pose_rep
from ..temporal.options import (
    PoseTemporalOptions,
    as_rotation_method,
    resolve_rotation_method,
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
    kwargs = {
        "on": on,
        "opts": request.opts.position_opts,
        "validate": False,
        "sequence_dim": request.sequence_dim,
        "batch_dims": request.batch_dims,
        "sequence_size_coord": request.sequence_size_coord,
    }
    if request.mode == "at":
        return source.param.at(request.query, **kwargs)
    return source.param.resample_to(request.query, **kwargs)


def _eval_rotation(
    rotation: Rotation,
    request: PoseTemporalRequest,
    *,
    on: str | None,
) -> Rotation:
    source = rotation if on is None else rotation.set_param_coord(name=on, validate=False)
    resolved = as_rotation_method(request.opts.rotation_opts, method=resolve_rotation_method(request.opts.rotation_opts))
    kwargs = {
        "on": on,
        "opts": resolved,
        "validate": False,
        "sequence_dim": request.sequence_dim,
        "batch_dims": request.batch_dims,
        "sequence_size_coord": request.sequence_size_coord,
    }
    if request.mode == "at":
        return source.param.at(request.query, **kwargs)
    return source.param.resample_to(request.query, **kwargs)


def _eval_payload_carrier(
    source: Pose,
    request: PoseTemporalRequest,
    *,
    on: str | None,
) -> xr.Dataset:
    carrier = AnalysisObject._from_unvalidated(analysis_object_dataset(source))
    if on is not None:
        carrier = carrier.set_param_coord(name=on, validate=False)
    kwargs = {
        "on": on,
        "opts": request.opts.position_opts,
        "validate": False,
        "sequence_dim": request.sequence_dim,
        "batch_dims": request.batch_dims,
        "sequence_size_coord": request.sequence_size_coord,
    }
    if request.mode == "at":
        return analysis_object_dataset(carrier.param.at(request.query, **kwargs))
    return analysis_object_dataset(carrier.param.resample_to(request.query, **kwargs))


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


def _run_pose_temporal_request(request: PoseTemporalRequest) -> Pose:
    source = request.pose
    source_rep = get_pose_rep(analysis_object_dataset(source), owner=request.owner)
    on = _effective_param_key(request)
    carrier_ds = _eval_payload_carrier(source, request, on=on)
    matrix_source_var: str | None = None
    decompose_source = source
    if source_rep == "matrix":
        decompose_source, matrix_source_var = _matrix_only_pose_source(source, owner=request.owner)
    position, rotation = decompose_source.decompose(validate=False)
    position_out = _eval_position(position, request, on=on)
    rotation_out = _eval_rotation(rotation, request, on=on)
    recomposed = source.__class__.from_components(rotation_out, position_out, validate=False)
    matrix_template_ds: xr.Dataset | None = None
    if source_rep == "matrix":
        if matrix_source_var is None:
            raise ValueError(f"{request.owner}: missing source matrix payload selection.")
        typed_matrix = analysis_object_dataset(recomposed.as_matrix(validate=False))
        matrix_template_ds = typed_matrix
        merged_ds = _overlay_matrix_payload(
            carrier_ds,
            typed_matrix,
            source_matrix_var=matrix_source_var,
            owner=request.owner,
        )
    else:
        merged_ds = _overlay_components_payload(
            carrier_ds,
            analysis_object_dataset(recomposed),
            owner=request.owner,
        )
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


__all__ = ["pose_param_at", "pose_param_resample_to"]
