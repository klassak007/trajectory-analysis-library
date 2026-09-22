"""Pose temporal orchestration with one final composite commit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.composite_commit import _prepare_composite_schema_target
from tal.core.param_engine.query_output_verify import verify_query_output_plan
from tal.core.schema_errors import SchemaError
from tal.core.schema_update import source_schema_view
from tal.core.schema_validate.finalize import transfer_dataarray_metadata

from ..association import SpatialAssociationPlan
from ..temporal.options import PoseTemporalOptions
from .composite_finalize import commit_spatial_composite
from .pose_kernel_adapters import apply_components_to_matrix_kernel
from .pose_temporal_parts import PoseTemporalParts, prepare_pose_temporal_parts
from .pose_temporal_types import PoseTemporalRequest, PreparedPoseEvaluation

if TYPE_CHECKING:
    from ..pose import Pose


def _effective_param_key(request: PoseTemporalRequest) -> str | None:
    if request.on is None:
        return request.opts.on
    if request.opts.on is None or request.opts.on == request.on:
        return request.on
    raise ValueError(
        f"{request.owner}: conflicting on= value between argument "
        f"({request.on!r}) and opts.on ({request.opts.on!r})."
    )


def _matrix_payload(
    parts: PoseTemporalParts,
    source: xr.Dataset,
    *,
    owner: str,
) -> xr.DataArray:
    if parts.matrix_core_dims is None:
        raise RuntimeError(f"{owner}: matrix core dimensions were not prepared.")
    row_dim, col_dim = parts.matrix_core_dims
    position = parts.position
    rotation = parts.rotation
    if "valid" in parts.carrier.coords:
        valid = parts.carrier.coords["valid"]
        identity = xr.DataArray(
            [0.0, 0.0, 0.0, 1.0],
            dims=parts.rotation_dim,
            coords={parts.rotation_dim: ["x", "y", "z", "w"]},
        )
        position = position.where(valid, 0.0)
        rotation = rotation.where(valid, identity)
    matrix = apply_components_to_matrix_kernel(
        position,
        rotation,
        pos_dim=parts.position_dim,
        quat_dim=parts.rotation_dim,
        row_dim=row_dim,
        col_dim=col_dim,
        owner=owner,
    )
    source_var = source[parts.position_var]
    matrix = matrix.transpose(*source_var.dims)
    return transfer_dataarray_metadata(source_var, matrix)


def _assembled_variables(
    parts: PoseTemporalParts,
    source: xr.Dataset,
    *,
    owner: str,
) -> dict[str, xr.DataArray]:
    matrix = None
    if parts.representation == "matrix":
        matrix = _matrix_payload(parts, source, owner=owner)
    values: dict[str, xr.DataArray] = {}
    for name in source.data_vars:
        if name == parts.position_var:
            values[name] = parts.position if matrix is None else matrix
        elif name == parts.rotation_var:
            values[name] = parts.rotation
        elif name in parts.carrier.data_vars:
            values[name] = parts.carrier[name]
    return values


def _assemble_candidate(
    parts: PoseTemporalParts,
    source: xr.Dataset,
    *,
    owner: str,
) -> xr.Dataset:
    candidate = xr.Dataset(
        data_vars=_assembled_variables(parts, source, owner=owner),
        coords=parts.carrier.coords,
    )
    return source_schema_view(parts.carrier, candidate)


def _prepare_pose_temporal_candidate(
    request: PoseTemporalRequest,
) -> tuple[xr.Dataset, PoseTemporalParts, xr.Dataset]:
    source = request.pose
    source_ds = analysis_object_dataset(source)
    parts = prepare_pose_temporal_parts(
        request,
        on=_effective_param_key(request),
    )
    candidate = _assemble_candidate(parts, source_ds, owner=request.owner)
    candidate = _prepare_composite_schema_target(
        candidate,
        schema=parts.schema,
        validate=request.validate,
    )
    verify_query_output_plan(
        candidate,
        plan=parts.output_plan,
        topology=parts.query_topology,
    )
    return candidate, parts, source_ds


def _prepare_owned_candidate(
    request: PoseTemporalRequest,
) -> tuple[xr.Dataset, PoseTemporalParts, xr.Dataset]:
    try:
        return _prepare_pose_temporal_candidate(request)
    except SchemaError:
        raise
    except (TypeError, ValueError) as exc:
        if type(exc) not in (TypeError, ValueError):
            raise
        prefix = f"{request.owner}:"
        if str(exc).startswith(prefix):
            raise
        raise type(exc)(f"{prefix} {exc}") from exc


def _run_pose_temporal_request(request: PoseTemporalRequest) -> Pose:
    source = request.pose
    candidate, parts, source_ds = _prepare_owned_candidate(request)
    result = commit_spatial_composite(
        candidate,
        schema=parts.schema,
        components=parts.components,
        prototype=source,
        association=SpatialAssociationPlan(source.graph),
        resource_sources=(source_ds,),
        validate=request.validate,
        owner=request.owner,
    )
    return cast("Pose", result)


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
    return _run_pose_temporal_request(PoseTemporalRequest(
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
    ))


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
    return _run_pose_temporal_request(PoseTemporalRequest(
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
    ))


__all__ = ["PreparedPoseEvaluation", "pose_param_at", "pose_param_resample_to"]
