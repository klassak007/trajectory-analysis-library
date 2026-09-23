from __future__ import annotations

from dataclasses import dataclass, replace

import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.param_engine.query_output_verify import verify_query_output_plan
from tal.core.param_engine.query_topology import (
    QueryOutputPlan,
    attach_query_coordinates,
    preflight_query_output_namespace,
    prepare_query_topology,
)
from tal.core.schema_read import read_roles

from .path_query_topology import OutputTopology


@dataclass(frozen=True)
class PathBasisResult:
    """A verified intermediate transform and its caller's final declaration."""

    value: object
    caller: object
    output_plan: QueryOutputPlan | None


def finalize_basis_application(value: xr.Dataset, basis: PathBasisResult | None, caller: object) -> xr.Dataset:
    """Verify the completed caller, leaving family members to their own owners."""
    if basis is None or basis.caller is not caller:
        return value
    return finalize_path_query_output(value, plan=basis.output_plan)


def _public_query(
    topology: OutputTopology,
    *,
    include_batch: bool,
) -> xr.DataArray:
    query = topology.query
    if topology.query_dim != topology.sequence_dim:
        query = query.rename({topology.query_dim: topology.sequence_dim})
    if not include_batch:
        return query
    missing = tuple(dim for dim in topology.batch_dims if dim not in query.dims)
    if missing:
        query = query.expand_dims(
            {dim: int(topology.batch_coords.sizes[dim]) for dim in missing}
        )
    query = query.transpose(*topology.batch_dims, topology.sequence_dim)
    return query.assign_coords(topology.batch_coords)


def _output_source(
    value: object,
    topology: OutputTopology,
    *,
    retain_sequence_coords: bool,
) -> xr.Dataset:
    source = analysis_object_dataset(value)
    sequence_dim = read_roles(source)[1]
    if sequence_dim is None:
        return source
    if not retain_sequence_coords:
        names = tuple(
            name for name, coord in source.coords.items()
            if sequence_dim in coord.dims
        )
        source = source.drop_vars(names, errors="ignore")
    if sequence_dim == topology.sequence_dim:
        return source
    return source.rename_dims({sequence_dim: topology.sequence_dim})


def prepare_path_query_output_plan(
    topology: OutputTopology,
    *,
    source: object,
    retain_sequence_coords: bool,
    owner: str,
) -> QueryOutputPlan:
    """Preflight one path result's public query/output namespace."""
    query = _public_query(topology, include_batch=not retain_sequence_coords)
    projected_source = _output_source(
        source,
        topology,
        retain_sequence_coords=retain_sequence_coords,
    )
    generated = (
        ()
        if topology.param_name == topology.sequence_dim
        else (topology.param_name,)
    )
    plan = preflight_query_output_namespace(
        projected_source,
        None if retain_sequence_coords else query,
        sequence_dim=topology.sequence_dim,
        batch_dims=topology.batch_dims,
        owner=owner,
        intent="grid",
        generated_names=generated,
        retain_sequence_coords=retain_sequence_coords,
    )
    query_topology = prepare_query_topology(
        query,
        query_dim=topology.sequence_dim,
        stacked_dims=None,
    )
    return replace(plan, topology=query_topology)


def finalize_path_query_output(
    value: xr.Dataset,
    *,
    plan: QueryOutputPlan | None,
) -> xr.Dataset:
    """Restore caller coordinates and verify one assembled path result."""
    if plan is None or plan.topology is None:
        return value
    coordinate_names = set(plan.topology.coordinates.coordinates)
    coordinate_names.update(
        name
        for snapshot in plan.topology.coordinates.indexes
        for name in snapshot.coordinates
    )
    restored = value
    if not coordinate_names <= set(value.coords):
        restored = attach_query_coordinates(
            value,
            topology=plan.topology,
            plan=plan,
            owner=plan.owner,
        )
    verify_query_output_plan(restored, plan=plan, topology=plan.topology)
    return restored


__all__: list[str] = []
