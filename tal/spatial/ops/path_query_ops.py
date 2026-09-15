from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.param_ops.evaluate import evaluate_param
from tal.core.schema import set_param_coord, set_roles, set_validity
from tal.core.schema_read import read_roles, read_sequence_size_coord_name

from ..pose import Pose
from ..rotation import Rotation
from ..temporal.options import (
    PoseTemporalOptions,
    _validate_pose_temporal_options,
)
from .path_query_plan import (
    PreparedPathQuery,
    PreparedProviderQuery,
    RequiredProvider,
    prepare_path_query,
)
from .path_query_topology import OutputTopology
from .pose_temporal_ops import PreparedPoseEvaluation, pose_param_at
from .rotation_temporal_ops import rotation_param_at


@dataclass(frozen=True)
class CompletePathQuery:
    """Complete provider set after classification and optional evaluation."""

    providers: tuple[RequiredProvider, ...]
    values: tuple[object, ...]
    topology: OutputTopology | None


def require_path_temporal_options(value: object, *, owner: str) -> PoseTemporalOptions:
    """Validate the strict PathSolveOptions.temporal boundary."""
    if value is None:
        return _validate_pose_temporal_options(PoseTemporalOptions(), owner=owner)
    if not isinstance(value, PoseTemporalOptions):
        raise TypeError(f"{owner}: opts.temporal must be PoseTemporalOptions or None.")
    return _validate_pose_temporal_options(value, owner=owner)


def _realize_coverage(valid: xr.DataArray, *, owner: str) -> None:
    """Realize only the combined coordinate/validity boolean for strict coverage."""
    try:
        result = valid.all()
        if result.chunks is not None:
            result = result.compute()
        covered = bool(np.asarray(result.data).all())
    except Exception as exc:
        raise ValueError(f"{owner}: failed to realize temporal coordinate coverage.") from exc
    if not covered:
        raise ValueError(f"{owner}: query lies outside a required provider's closed parameter domain.")


def _require_provider_coverage(
    item: PreparedProviderQuery,
    topology: OutputTopology,
    *,
    owner: str,
) -> None:
    caller_valid = topology.caller.valid_mask if topology.caller is not None else None
    for evaluation in item.evaluations:
        pmap = evaluation.param_map
        valid = pmap.valid if caller_valid is None else (pmap.valid | ~caller_valid.rename({topology.sequence_dim: topology.query_dim}))
        _realize_coverage(valid, owner=owner)


def _drop_sequence_coords(ds: xr.Dataset, *, sequence_dim: str) -> xr.Dataset:
    names = [name for name, coord in ds.coords.items() if sequence_dim in coord.dims]
    return ds.drop_vars(names, errors="ignore")


def _assign_output_coords(ds: xr.Dataset, topology: OutputTopology) -> xr.Dataset:
    out = ds.assign_coords(topology.batch_coords)
    if topology.caller is not None:
        caller_ds = topology.caller.ds
        seq = topology.sequence_dim
        if seq in caller_ds.coords:
            out = out.assign_coords({seq: caller_ds.coords[seq]})
        out = out.assign_coords({topology.param_name: topology.caller.spec.coord})
        size_name = topology.caller.sequence_size_coord
        if size_name is not None:
            out = out.assign_coords({size_name: caller_ds.coords[size_name]})
        return out
    query = topology.query.reset_coords(drop=True)
    if topology.query_dim != topology.sequence_dim:
        query = query.rename({topology.query_dim: topology.sequence_dim})
    if query.ndim == 1:
        return out.assign_coords({topology.sequence_dim: query})
    axis = topology.query.coords.get(topology.query_dim)
    if axis is None:
        axis = xr.DataArray(np.arange(topology.query.sizes[topology.query_dim]), dims=topology.query_dim)
    if topology.query_dim != topology.sequence_dim:
        axis = axis.rename({topology.query_dim: topology.sequence_dim})
    return out.assign_coords(
        {
            topology.sequence_dim: axis,
            topology.param_name: query,
        }
    )


def _restore_output_dataset(value: object, topology: OutputTopology) -> xr.Dataset:
    ds = analysis_object_dataset(value)
    _, sequence_dim, _, core_dims = read_roles(ds)
    if sequence_dim is None:
        ds = ds.expand_dims({topology.sequence_dim: int(topology.query.sizes[topology.query_dim])})
    else:
        ds = _drop_sequence_coords(ds, sequence_dim=sequence_dim)
        if sequence_dim != topology.sequence_dim:
            ds = ds.rename({sequence_dim: topology.sequence_dim})
    ds = _assign_output_coords(ds, topology)
    leading = (*topology.batch_dims, topology.sequence_dim)
    ordered = (*leading, *(dim for dim in ds.dims if dim not in leading))
    ds = ds.transpose(*ordered)
    old_size = read_sequence_size_coord_name(ds)
    if old_size is not None and (topology.caller is None or old_size != topology.caller.sequence_size_coord):
        ds = ds.drop_vars(old_size, errors="ignore")
    ds = set_roles(ds, sequence_dim=topology.sequence_dim, batch_dims=topology.batch_dims, core_dims=core_dims, validate=False)
    ds = set_param_coord(ds, name=topology.param_name, validate=False)
    size_name = topology.caller.sequence_size_coord if topology.caller is not None else None
    ds = set_validity(ds, sequence_size_coord=size_name, validate=False)
    return ds


def _restore_output_topology(
    value: object,
    topology: OutputTopology,
    *,
    owner: str,
) -> object:
    try:
        ds = _restore_output_dataset(value, topology)
        return value.__class__._from_unvalidated(ds)
    except (KeyError, TypeError, ValueError) as exc:
        text = str(exc)
        if text.startswith(f"{owner}:"):
            raise
        raise ValueError(f"{owner}: failed to restore query output topology; {text}") from exc


def _raise_path_query_execution_error(exc: TypeError | ValueError, *, owner: str) -> None:
    """Translate a provider execution failure through the public solver owner."""
    if str(exc).startswith(f"{owner}:"):
        raise exc
    raise type(exc)(f"{owner}: {exc}") from exc


def _evaluate_dynamic(
    item: PreparedProviderQuery,
    topology: OutputTopology,
    *,
    owner: str,
) -> object:
    projection = item.projection
    context = item.context
    if projection is None or context is None:
        raise ValueError(f"{owner}: dynamic provider plan is incomplete.")
    aligned = projection.value
    effective = item.temporal
    try:
        if isinstance(aligned, Pose):
            evaluated = pose_param_at(
                aligned,
                query=topology.query,
                on=effective.on,
                opts=effective,
                validate=False,
                sequence_dim=None,
                batch_dims=None,
                sequence_size_coord=None,
                owner=owner,
                prepared=PreparedPoseEvaluation(item.evaluations[0], item.evaluations[-1]),
            )
        elif isinstance(aligned, Rotation):
            evaluated = rotation_param_at(
                aligned,
                query=topology.query,
                on=effective.on,
                opts=effective.rotation_opts,
                validate=False,
                sequence_dim=None,
                batch_dims=None,
                sequence_size_coord=None,
                owner=owner,
                prepared=item.evaluations[0],
            )
        else:
            evaluated = evaluate_param(
                context,
                query=topology.query,
                opts=effective.position_opts,
                validate=False,
                prepared=item.evaluations[0],
            )
    except (TypeError, ValueError) as exc:
        _raise_path_query_execution_error(exc, owner=owner)
    return _restore_output_topology(evaluated, topology, owner=owner)


def _broadcast_static(
    item: PreparedProviderQuery,
    topology: OutputTopology,
    *,
    owner: str,
) -> object:
    if item.projection is None:
        raise ValueError(f"{owner}: static provider plan is incomplete.")
    return _restore_output_topology(item.projection.value, topology, owner=owner)


def require_path_query_coverage(plan: PreparedPathQuery, *, owner: str) -> None:
    """Resolve strict coverage from the maps already owned by one plan."""
    if plan.topology is None:
        return
    for item in plan.items:
        if item.required.topology == "dynamic":
            _require_provider_coverage(item, plan.topology, owner=owner)


def execute_path_query(
    plan: PreparedPathQuery,
    *,
    owner: str,
    coverage_checked: bool = False,
) -> CompletePathQuery:
    """Execute the accepted generic provider pipeline from one frozen plan."""
    if plan.topology is None:
        return CompletePathQuery(plan.providers, tuple(item.required.value for item in plan.items), None)
    if not coverage_checked:
        require_path_query_coverage(plan, owner=owner)
    evaluated = tuple(
        _evaluate_dynamic(item, plan.topology, owner=owner)
        if item.required.topology == "dynamic"
        else _broadcast_static(item, plan.topology, owner=owner)
        for item in plan.items
    )
    return CompletePathQuery(plan.providers, evaluated, plan.topology)


def complete_path_query(
    values: Sequence[object],
    *,
    query: object | None,
    caller: object | None,
    temporal: PoseTemporalOptions,
    owner: str,
) -> CompletePathQuery:
    """Prepare and execute one complete transform-provider set."""
    plan = prepare_path_query(
        values,
        query=query,
        caller=caller,
        temporal=temporal,
        owner=owner,
    )
    return execute_path_query(plan, owner=owner)


__all__ = [
    "CompletePathQuery",
    "OutputTopology",
    "RequiredProvider",
    "complete_path_query",
    "execute_path_query",
    "prepare_path_query",
    "require_path_query_coverage",
    "require_path_temporal_options",
]
