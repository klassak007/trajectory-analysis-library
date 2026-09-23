from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core.component_ops import read_components
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.finalize import transfer_dataset_attrs
from tal.core.orchestration.runtime_checks import select_single_numeric_var
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec
from tal.core.schema import UNSET, set_param_coord, set_roles, set_validity
from tal.core.schema_read import read_roles, read_sequence_size_coord_name
from tal.utils.frame_schema import set_frames

from ..association import SpatialAssociationPlan
from ..metadata import set_expressed_in, set_pose_rep
from ..pose import Pose
from ..position import Position
from .batched_path_plan import PreparedBatchedPathExecution
from .composite_finalize import commit_spatial_composite
from .path_query_ops import _restore_output_dataset
from .path_query_output import finalize_path_query_output
from .pose_component_ops import resolve_pose_component_specs


def _association(plan: PreparedBatchedPathExecution) -> SpatialAssociationPlan:
    context = plan.finalization.result_context
    if isinstance(context, SpatialAssociationPlan):
        return context
    return SpatialAssociationPlan(None)


def _position_candidate(plan: PreparedBatchedPathExecution, values: np.ndarray) -> xr.Dataset:
    topology = plan.query.topology
    if topology is None or topology.caller is None:
        raise ValueError("batched Position finalization requires caller topology.")
    source = topology.caller.ds
    name = select_single_numeric_var(source, owner="spatial.path_solve.pose", what="Position caller")
    _, sequence_dim, _, core_dims = read_roles(source)
    output = xr.DataArray(values, dims=(*plan.logical_rows.dims, core_dims[0]))
    if plan.finalization.query_dim != sequence_dim:
        output = output.rename({plan.finalization.query_dim: sequence_dim})
    output = output.transpose(*source[name].dims)
    candidate = source[name].copy(data=output.data).to_dataset(name=name)
    candidate = transfer_dataset_attrs(source, candidate, validate=False)
    candidate.encoding = dict(source.encoding)
    candidate = set_frames(
        candidate,
        parent=plan.finalization.parent_frame,
        child=plan.finalization.child_frame,
        validate=False,
    )
    candidate = set_expressed_in(
        candidate,
        expressed_in=None,
        validate=False,
        owner="spatial.path_solve.pose",
    )
    return finalize_path_query_output(candidate, plan=plan.query.output_plan)


def _pose_candidate(
    plan: PreparedBatchedPathExecution,
    translation: np.ndarray,
    quaternion: np.ndarray,
) -> tuple[xr.Dataset, tuple[tuple[str, object], ...]]:
    item = plan.query.items[0]
    if item.projection is None or plan.query.topology is None:
        raise ValueError("batched Pose finalization requires provider topology.")
    source_value = item.projection.value
    source = analysis_object_dataset(source_value)
    _, sequence_dim, _, core_dims = read_roles(source)
    specs = resolve_pose_component_specs(
        source,
        owner="spatial.path_solve.pose",
        core_dims=core_dims,
    )
    (position_dim, position_var), (rotation_dim, rotation_var) = specs
    sequence_coords = [name for name, coord in source.coords.items() if sequence_dim in coord.dims]
    candidate = source.copy(deep=False).drop_vars([*source.data_vars, *sequence_coords], errors="ignore")
    dims = plan.logical_rows.dims
    candidate[position_var] = xr.DataArray(translation, dims=(*dims, position_dim))
    candidate[rotation_var] = xr.DataArray(quaternion, dims=(*dims, rotation_dim))
    candidate = set_roles(
        candidate,
        sequence_dim=plan.finalization.query_dim,
        batch_dims=plan.finalization.batch_dims,
        core_dims=core_dims,
        validate=False,
    )
    candidate = set_param_coord(candidate, name=None, validate=False)
    candidate = set_validity(candidate, sequence_size_coord=None, validate=False)
    candidate = _restore_output_dataset(
        candidate,
        plan.query.topology,
        output_plan=plan.query.output_plan,
    )
    candidate = set_pose_rep(candidate, rep="components", validate=False, owner="spatial.path_solve.pose")
    candidate = set_frames(
        candidate,
        parent=plan.finalization.parent_frame,
        child=plan.finalization.child_frame,
        validate=False,
    )
    candidate = set_expressed_in(
        candidate,
        expressed_in=plan.finalization.expressed_in,
        validate=False,
        owner="spatial.path_solve.pose",
    )
    return candidate, tuple(read_components(source_value).items())


def commit_batched_position(plan: PreparedBatchedPathExecution, values: np.ndarray):
    topology = plan.query.topology
    if topology is None or topology.caller is None:
        raise ValueError("batched Position commit requires caller topology.")
    source = topology.caller.ds
    _, sequence_dim, batch_dims, core_dims = read_roles(source)
    schema = CoreSchemaFinalizeSpec(
        sequence_dim,
        batch_dims,
        core_dims,
        topology.param_name,
        read_sequence_size_coord_name(source),
    )
    return commit_spatial_composite(
        _position_candidate(plan, values),
        schema=schema,
        components=UNSET,
        prototype=plan.query.result_prototype or Position,
        association=_association(plan),
        resource_sources=(),
        validate=plan.query.result_validate,
        owner="spatial.path_solve.pose",
    )


def commit_batched_pose(
    plan: PreparedBatchedPathExecution,
    translation: np.ndarray,
    quaternion: np.ndarray,
):
    candidate, components = _pose_candidate(plan, translation, quaternion)
    _, sequence_dim, batch_dims, core_dims = read_roles(candidate)
    schema = CoreSchemaFinalizeSpec(
        sequence_dim,
        batch_dims,
        core_dims,
        plan.finalization.param_name,
        read_sequence_size_coord_name(candidate),
    )
    return commit_spatial_composite(
        candidate,
        schema=schema,
        components=components,
        prototype=plan.query.result_prototype or Pose,
        association=_association(plan),
        resource_sources=(),
        validate=plan.query.result_validate,
        owner="spatial.path_solve.pose",
    )


__all__ = ["commit_batched_pose", "commit_batched_position"]
