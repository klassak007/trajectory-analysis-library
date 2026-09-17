from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import (
    analysis_object_dataset,
    metadata_isolated_dataset,
)
from tal.core.ordered_dtypes import is_ordered_real_numeric_dtype
from tal.core.schema import set_param_coord, set_roles, set_validity
from tal.core.schema_read import read_roles
from tal.frames import FramePath
from tal.utils.numba_support import _numba_available

from ..kernels.fused_pose_path import fuse_pose_path
from ..kernels.streaming_pose_path import stream_pose_path_blocks
from ..metadata import get_pose_rep
from ..pose import Pose
from ..temporal.options import resolve_rotation_method
from .path_query_ops import (
    CompletePathQuery,
    _raise_path_query_execution_error,
    _restore_output_topology,
    execute_path_query,
    require_path_query_coverage,
)
from .path_query_plan import PreparedPathQuery, PreparedProviderQuery
from .pose_component_ops import resolve_pose_component_specs

PathExecutionKind = Literal["generic", "fused-numba", "scipy-stream"]


@dataclass(frozen=True)
class PreparedPosePathExecution:
    """One immutable Pose path execution choice."""

    path: FramePath
    query: PreparedPathQuery
    kind: PathExecutionKind


@dataclass(frozen=True)
class _PoseArrays:
    translation: np.ndarray
    quaternion: np.ndarray
    position_var: str
    rotation_var: str
    position_dim: str
    rotation_dim: str
    core_dims: tuple[str, ...]
    source: xr.Dataset


def _metadata_is_streamable(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    component_vars: tuple[str, str],
    core_dims: tuple[str, ...],
) -> bool:
    if set(ds.attrs) != {"tal"} or ds.encoding:
        return False
    if any(value.attrs or value.encoding for value in ds.variables.values()):
        return False
    if set(ds.data_vars) != set(component_vars):
        return False
    for name, coord in ds.coords.items():
        if sequence_dim in coord.dims:
            continue
        if name not in core_dims or coord.dims != (name,) or name not in ds.xindexes:
            return False
    return True


def _provider_is_streamable(item: PreparedProviderQuery) -> bool:
    projection = item.projection
    context = item.context
    if item.required.topology != "dynamic" or projection is None or context is None:
        return False
    value = projection.value
    if not isinstance(value, Pose) or get_pose_rep(analysis_object_dataset(value), owner="spatial.path_solve.pose") != "components":
        return False
    if (
        context.param_kind != "numeric"
        or context.batch_dims
        or context.spec.coord.chunks is not None
        or context.valid_mask.chunks is not None
        or len(item.evaluations) != 1
    ):
        return False
    temporal = item.temporal
    if temporal.position_opts.method != "linear" or resolve_rotation_method(temporal.rotation_opts) != "slerp":
        return False
    ds = analysis_object_dataset(value)
    _, sequence_dim, _, core_dims = read_roles(ds)
    (position_dim, position_var), (rotation_dim, rotation_var) = resolve_pose_component_specs(
        ds,
        owner="spatial.path_solve.pose",
        core_dims=core_dims,
    )
    expected = ((sequence_dim, position_dim), (sequence_dim, rotation_dim))
    variables = (ds[position_var], ds[rotation_var])
    payloads_match = all(
        var.dims == dims
        and var.chunks is None
        and is_ordered_real_numeric_dtype(var.dtype)
        for var, dims in zip(variables, expected, strict=True)
    )
    return payloads_match and _metadata_is_streamable(
        ds,
        sequence_dim=sequence_dim,
        component_vars=(position_var, rotation_var),
        core_dims=core_dims,
    )


def _streaming_layout(item: PreparedProviderQuery) -> tuple[str, ...]:
    if item.projection is None:
        return ()
    ds = analysis_object_dataset(item.projection.value)
    _, sequence_dim, _, core_dims = read_roles(ds)
    specs = resolve_pose_component_specs(ds, owner="spatial.path_solve.pose", core_dims=core_dims)
    return (sequence_dim, *core_dims, *(part for spec in specs for part in spec))


def _core_indexes_equal(left: xr.Dataset, right: xr.Dataset, core_dims: tuple[str, ...]) -> bool:
    left_groups = tuple(
        (tuple(names), index)
        for index, names in left.xindexes.group_by_index()
        if any(name in core_dims for name in names)
    )
    right_groups = tuple(
        (tuple(names), index)
        for index, names in right.xindexes.group_by_index()
        if any(name in core_dims for name in names)
    )
    if len(left_groups) != len(right_groups):
        return False
    return all(
        left_names == right_names and left_index.equals(right_index)
        for (left_names, left_index), (right_names, right_index) in zip(left_groups, right_groups, strict=True)
    )


def _streaming_layouts_match(items: tuple[PreparedProviderQuery, ...]) -> bool:
    first_item = items[0]
    if first_item.projection is None:
        return False
    first_layout = _streaming_layout(first_item)
    first_ds = analysis_object_dataset(first_item.projection.value)
    core_dims = read_roles(first_ds)[3]
    for item in items[1:]:
        if item.projection is None or _streaming_layout(item) != first_layout:
            return False
        if not _core_indexes_equal(first_ds, analysis_object_dataset(item.projection.value), core_dims):
            return False
    return True


def _streaming_eligible(plan: PreparedPathQuery) -> bool:
    topology = plan.topology
    if topology is None or topology.caller is not None or topology.batch_dims:
        return False
    if (
        topology.param_kind != "numeric"
        or topology.query.ndim != 1
        or topology.query.chunks is not None
        or np.dtype(topology.query.dtype) != np.dtype(np.float64)
    ):
        return False
    if not plan.items or not all(_provider_is_streamable(item) for item in plan.items):
        return False
    if not _streaming_layouts_match(plan.items):
        return False
    shared = plan.items[0].evaluations[0]
    return all(item.evaluations[0] is shared for item in plan.items[1:])


def prepare_pose_path_execution(
    path: FramePath,
    query: PreparedPathQuery,
) -> PreparedPosePathExecution:
    """Classify one prepared Pose request without inspecting payload values."""
    topology = query.topology
    if topology is None or topology.query.size == 0 or not _streaming_eligible(query):
        kind: PathExecutionKind = "generic"
    elif _numba_available():
        kind = "fused-numba"
    else:
        kind = "scipy-stream"
    return PreparedPosePathExecution(path, query, kind)


def _pose_arrays(item: PreparedProviderQuery) -> _PoseArrays:
    if item.projection is None:
        raise ValueError("spatial.path_solve.pose: streaming provider projection is missing.")
    value = item.projection.value
    ds = analysis_object_dataset(value)
    core_dims = read_roles(ds)[3]
    (position_dim, position_var), (rotation_dim, rotation_var) = resolve_pose_component_specs(
        ds,
        owner="spatial.path_solve.pose",
        core_dims=core_dims,
    )
    translation = np.asarray(ds[position_var].data)
    quaternion = np.asarray(ds[rotation_var].data)
    return _PoseArrays(
        translation,
        quaternion,
        position_var,
        rotation_var,
        position_dim,
        rotation_dim,
        core_dims,
        ds,
    )


def _streaming_dataset(arrays: _PoseArrays, result: tuple[np.ndarray, np.ndarray], query_dim: str) -> xr.Dataset:
    base = metadata_isolated_dataset(arrays.source, owner="spatial.path_solve.pose")
    source_sequence = read_roles(arrays.source)[1]
    position_source = base[arrays.position_var]
    rotation_source = base[arrays.rotation_var]
    sequence_coords: list[object] = []
    if source_sequence is not None:
        sequence_coords = [name for name, coord in base.coords.items() if source_sequence in coord.dims]
    base = base.drop_vars([*base.data_vars, *sequence_coords], errors="ignore")
    base = base.assign(
        {
            arrays.position_var: (
                (query_dim, arrays.position_dim),
                result[0],
                position_source.attrs,
            ),
            arrays.rotation_var: (
                (query_dim, arrays.rotation_dim),
                result[1],
                rotation_source.attrs,
            ),
        }
    )
    base[arrays.position_var].encoding = dict(position_source.encoding)
    base[arrays.rotation_var].encoding = dict(rotation_source.encoding)
    base = set_roles(base, sequence_dim=query_dim, batch_dims=(), core_dims=arrays.core_dims, validate=False)
    base = set_param_coord(base, name=None, validate=False)
    return set_validity(base, sequence_size_coord=None, validate=False)


def _execute_streaming(plan: PreparedPosePathExecution, *, owner: str) -> Pose:
    topology = plan.query.topology
    if topology is None:
        raise ValueError(f"{owner}: streaming query topology is missing.")
    provider_arrays = tuple(_pose_arrays(item) for item in plan.query.items)
    prepared = plan.query.items[0].evaluations[0].param_map
    try:
        result = stream_pose_path_blocks(
            tuple(item.translation for item in provider_arrays),
            tuple(item.quaternion for item in provider_arrays),
            tuple(-1 if step.invert else 1 for step in plan.path.steps),
            i0=np.asarray(prepared.i0.data),
            i1=np.asarray(prepared.i1.data),
            alpha=np.asarray(prepared.alpha.data),
            valid=np.asarray(prepared.valid.data),
        )
    except (TypeError, ValueError) as exc:
        _raise_path_query_execution_error(exc, owner=owner)
    value = Pose._from_unvalidated(_streaming_dataset(provider_arrays[0], result, topology.query_dim))
    return _restore_output_topology(value, topology, owner=owner)  # type: ignore[return-value]


def _execute_fused(plan: PreparedPosePathExecution, *, owner: str) -> Pose:
    topology = plan.query.topology
    if topology is None:
        raise ValueError(f"{owner}: fused query topology is missing.")
    provider_arrays = tuple(_pose_arrays(item) for item in plan.query.items)
    prepared = plan.query.items[0].evaluations[0].param_map
    try:
        result = fuse_pose_path(
            np.stack(tuple(item.translation for item in provider_arrays)),
            np.stack(tuple(item.quaternion for item in provider_arrays)),
            np.asarray(tuple(-1 if step.invert else 1 for step in plan.path.steps), dtype=np.int8),
            i0=np.asarray(prepared.i0.data),
            i1=np.asarray(prepared.i1.data),
            alpha=np.asarray(prepared.alpha.data),
            valid=np.asarray(prepared.valid.data),
            quaternion_dtypes=tuple(item.quaternion.dtype for item in provider_arrays),
        )
    except (TypeError, ValueError) as exc:
        _raise_path_query_execution_error(exc, owner=owner)
    value = Pose._from_unvalidated(_streaming_dataset(provider_arrays[0], result, topology.query_dim))
    return _restore_output_topology(value, topology, owner=owner)  # type: ignore[return-value]


def execute_pose_path(plan: PreparedPosePathExecution, *, owner: str) -> CompletePathQuery | Pose:
    """Dispatch one frozen Pose request to its accepted executor."""
    require_path_query_coverage(plan.query, owner=owner)
    if plan.kind == "fused-numba":
        return _execute_fused(plan, owner=owner)
    if plan.kind == "scipy-stream":
        return _execute_streaming(plan, owner=owner)
    return execute_path_query(plan.query, owner=owner, coverage_checked=True)


__all__ = ["PreparedPosePathExecution", "execute_pose_path", "prepare_pose_path_execution"]
