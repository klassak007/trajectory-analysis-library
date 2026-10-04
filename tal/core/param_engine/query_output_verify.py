"""Final checks for public parameter-query output topology."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import xarray as xr

from ..orchestration.indexing import index_group_for_coordinate
from ..schema_read import read_roles, read_validity

if TYPE_CHECKING:
    from .query_topology import QueryOutputPlan, QueryTopologyPlan


def verify_query_output_plan(
    value: xr.Dataset | xr.DataArray,
    *,
    plan: QueryOutputPlan,
    topology: QueryTopologyPlan,
) -> None:
    """Check the selected output intent against assembled public topology."""
    if plan.topology is not topology:
        raise ValueError(f"{plan.owner}: output plan does not match the prepared query topology.")
    if plan.intent == "index":
        if not isinstance(value, xr.DataArray) or set(plan.consumed_names) & set(value.coords):
            raise ValueError(f"{plan.owner}: index output retains consumed runtime metadata.")
    elif isinstance(value, xr.Dataset) and not plan.protected_data_vars <= set(value.data_vars):
        raise ValueError(f"{plan.owner}: output lost a surviving data variable.")
    _verify_generated_coordinates(value, plan=plan, topology=topology)
    _verify_output_indexes(value, plan=plan, topology=topology)
    if isinstance(value, xr.Dataset):
        _verify_output_schema(value, plan=plan)
    sizes = dict(zip(topology.dims, topology.sizes, strict=True))
    if plan.intent == "trajectory" and plan.query_only_dims:
        expected = prod(sizes[dim] for dim in plan.query_only_dims)
        if value.sizes.get(plan.sequence_dim) != expected:
            raise ValueError(f"{plan.owner}: typed output sequence disagrees with the query topology.")
        return
    for dim in topology.stacked_dims or ():
        if value.sizes.get(dim) != sizes[dim]:
            raise ValueError(f"{plan.owner}: restored query dimension {dim!r} has the wrong size.")


def _verify_output_schema(value: xr.Dataset, *, plan: QueryOutputPlan) -> None:
    declared, sequence_dim, batch_dims, core_dims = read_roles(value)
    if declared:
        roles = ({sequence_dim} if sequence_dim is not None else set()) | set(batch_dims) | set(core_dims)
        if not roles <= set(value.dims):
            raise ValueError(f"{plan.owner}: output roles reference absent dimensions.")
    validity_declared, size_name, _layout = read_validity(value)
    if validity_declared and size_name not in value.coords:
        raise ValueError(f"{plan.owner}: output validity references absent coordinates.")


def _verify_generated_coordinates(
    value: xr.Dataset | xr.DataArray,
    *,
    plan: QueryOutputPlan,
    topology: QueryTopologyPlan,
) -> None:
    required = plan.generated_names - plan.optional_generated_names
    missing = required - set(value.coords)
    if missing:
        raise ValueError(f"{plan.owner}: output lost generated coordinates {sorted(missing)!r}.")
    query_dims = (
        (plan.sequence_dim,)
        if plan.intent == "trajectory" or topology.stacked_dims is None
        else topology.stacked_dims
    )
    for name in required:
        if not set(query_dims) <= set(value.coords[name].dims):
            raise ValueError(f"{plan.owner}: generated coordinate {name!r} has the wrong topology.")


def _caller_index_groups(
    *,
    plan: QueryOutputPlan,
    topology: QueryTopologyPlan,
) -> tuple[tuple[tuple[str, ...], xr.Index], ...]:
    if plan.intent == "trajectory" or (plan.intent == "grid" and topology.stacked_dims is None):
        return ()
    rename = (
        {dim: topology.query_dim for dim in plan.query_only_dims if dim != topology.query_dim}
        if topology.stacked_dims is None else {}
    )
    captured = (
        (index, coordinates)
        for snapshot in topology.coordinates.indexes
        for index, coordinates in snapshot.coordinates.xindexes.group_by_index()
    )
    projected = (
        _project_caller_index(
            index, coordinates, rename=rename,
            skip_batch=topology.stacked_dims is None, plan=plan,
        )
        for index, coordinates in captured
    )
    return tuple(group for group in projected if group is not None)


def _project_caller_index(
    index: xr.Index,
    coordinates: dict[object, xr.Variable],
    *,
    rename: dict[str, str],
    skip_batch: bool,
    plan: QueryOutputPlan,
) -> tuple[tuple[str, ...], xr.Index] | None:
    if skip_batch and any(set(coord.dims) & set(plan.batch_dims) for coord in coordinates.values()):
        return None  # Source batch labels take precedence after exact alignment.
    names = tuple(str(rename.get(name, name)) for name in coordinates)
    try:
        projected = index.rename(rename, rename) if rename else index
    except Exception as exc:
        raise ValueError(f"{plan.owner}: caller index topology cannot be renamed safely.") from exc
    return names, projected


def _verify_output_indexes(
    value: xr.Dataset | xr.DataArray,
    *,
    plan: QueryOutputPlan,
    topology: QueryTopologyPlan,
) -> None:
    caller_groups = _caller_index_groups(plan=plan, topology=topology)
    caller_names = {name for names, _ in caller_groups for name in names}
    for names, index in caller_groups:
        _require_index_group(value, names=names, index=index, owner=plan.owner)
    for names, index in plan.source_index_groups:
        if set(names) & caller_names:
            continue
        _require_index_group(value, names=names, index=index, owner=plan.owner)


def _require_index_group(
    value: xr.Dataset | xr.DataArray,
    *,
    names: tuple[str, ...],
    index: xr.Index,
    owner: str,
) -> None:
    actual = index_group_for_coordinate(value, names[0])
    try:
        matches = actual is not None and actual[0] == names and bool(actual[1].equals(index))
    except Exception:  # noqa: BLE001 - an indeterminate index is not preserved.
        matches = False
    if not matches:
        raise ValueError(f"{owner}: output lost index topology {names!r}.")
