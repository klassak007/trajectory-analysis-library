from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import numpy as np
import xarray as xr

from tal.utils.xarray_namespace import dataarray_namespace_names, unique_temp_dim

from ..orchestration.indexing import (
    ResultCoordinateSnapshot,
    capture_result_coordinates,
    coordinate_variables_compatible,
    index_group_for_coordinate,
    require_compatible_shared_batch_index_types,
    restore_result_coordinates,
    sequence_dependent_coordinate_names,
    without_index_topology,
)
from ..schema import _transfer_dataset_attrs_for_finalize
from ..schema_read import read_sequence_size_coord_name
from ..schema_validate.finalize import transfer_dataarray_metadata


@dataclass(frozen=True)
class QueryTopologyPlan:
    """Private labeled-query topology retained across internal stacking."""

    query_dim: str
    stacked_dims: tuple[str, ...] | None
    dims: tuple[str, ...]
    sizes: tuple[int, ...]
    coordinates: ResultCoordinateSnapshot

    @property
    def logical_row_count(self) -> int:
        return int(prod(self.sizes))

    @property
    def has_no_rows(self) -> bool:
        return self.logical_row_count == 0


@dataclass(frozen=True)
class QueryOutputPlan:
    """Private output intent and namespace ownership for one query operation."""

    intent: Literal["grid", "trajectory", "index"]
    owner: str
    sequence_dim: str
    batch_dims: tuple[str, ...]
    query_only_dims: tuple[str, ...]
    generated_names: frozenset[str]
    optional_generated_names: frozenset[str]
    consumed_names: frozenset[str]
    protected_data_vars: frozenset[str]
    protected_dims: frozenset[str]
    source_coord_names: frozenset[str]
    sampled_source_coord_names: frozenset[str]
    source_index_groups: tuple[tuple[tuple[str, ...], xr.Index], ...]
    topology: QueryTopologyPlan | None = None


def generated_query_coordinate_names(
    *,
    operation: Literal["evaluate", "select", "index"],
    param_name: str | None,
    size_name: str | None,
    trajectory: bool,
    mapped_dataset: bool,
) -> tuple[str, ...]:
    """Describe coordinates emitted by the shared parameter output owners."""
    if operation == "index":
        return ()
    names = [param_name] if param_name is not None else []
    if operation == "select":
        names.extend(("sample_index", "valid"))
    elif mapped_dataset or size_name is not None:
        names.append("valid")
    if trajectory and size_name is not None:
        names.append(size_name)
    return tuple(names)


def _coordinates_compatible(
    left: xr.Dataset | xr.DataArray,
    right: xr.Dataset | xr.DataArray,
    *,
    name: str,
) -> bool:
    left_coord = left.coords.variables[name]
    right_coord = right.coords.variables[name]
    try:
        left_group = index_group_for_coordinate(left, name)
        right_group = index_group_for_coordinate(right, name)
        if left_group is not None or right_group is not None:
            return (left_group is not None and right_group is not None
                    and left_group[0] == right_group[0]
                    and bool(left_group[1].equals(right_group[1]))
                    and coordinate_variables_compatible(left_coord, right_coord, indexed=True))
        return coordinate_variables_compatible(left_coord, right_coord)
    except Exception:  # noqa: BLE001 - uncertain equality cannot authorize replacement.
        return False


def _preflight_query_name_claim(
    source: xr.Dataset | xr.DataArray,
    query: xr.DataArray,
    *,
    plan: QueryOutputPlan,
    name: str,
) -> None:
    owner = plan.owner
    group = index_group_for_coordinate(query, name)
    consumed_query_axis = (
        plan.intent == "trajectory" and name == plan.sequence_dim and name in query.dims
    )
    sampled_sequence_axis = (
        name == plan.sequence_dim and name in source.coords
        and source.coords[name].dims == (plan.sequence_dim,)
    )
    if name in plan.generated_names and (name in query.dims or group is not None) and not (
        sampled_sequence_axis or consumed_query_axis
    ):
        raise ValueError(
            f"{owner}: query axis or index {name!r} conflicts with generated output metadata; "
            "rename the caller query axis or index."
        )
    if name in plan.protected_data_vars:
        raise ValueError(f"{owner}: query name {name!r} collides with an output data variable.")
    if name in plan.protected_dims:
        raise ValueError(f"{owner}: query name {name!r} collides with a surviving core dimension.")
    if (
        plan.intent == "trajectory" and name == plan.sequence_dim and not consumed_query_axis
        and name in query.coords and query.coords[name].dims
    ):
        raise ValueError(f"{owner}: query coordinate {name!r} collides with the positional output sequence.")
    if (
        name in query.dims and name in source.coords and name not in plan.batch_dims
        and plan.sequence_dim not in source.coords[name].dims and name not in plan.consumed_names
    ):
        raise ValueError(f"{owner}: query dimension {name!r} collides with a surviving source coordinate.")


def _preflight_one_query_name(
    source: xr.Dataset | xr.DataArray,
    query: xr.DataArray,
    *,
    plan: QueryOutputPlan,
    name: str,
) -> None:
    _preflight_query_name_claim(source, query, plan=plan, name=name)
    if name not in query.coords or name not in source.coords:
        return
    if name in plan.generated_names or name in plan.consumed_names:
        return
    source_coord = source.coords[name]
    if name == plan.sequence_dim and not query.coords[name].dims and name not in query.dims:
        return
    if plan.sequence_dim in source_coord.dims:
        return
    if name in plan.batch_dims and not query.coords[name].dims and name not in query.dims:
        return
    if name in plan.batch_dims and source_coord.dims == (name,) and query.coords[name].dims == (name,):
        return
    if not _coordinates_compatible(source, query, name=name):
        raise ValueError(f"{plan.owner}: query coordinate {name!r} conflicts with a surviving source coordinate.")


def _surviving_source_names(
    source: xr.Dataset | xr.DataArray,
    *,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    intent: Literal["grid", "trajectory", "index"],
    consumed_names: tuple[str, ...],
    retain_sequence_coords: bool,
) -> tuple[frozenset[str], frozenset[str], frozenset[str], tuple[tuple[tuple[str, ...], xr.Index], ...]]:
    data_vars = frozenset(str(name) for name in source.data_vars) if isinstance(source, xr.Dataset) and intent != "index" else frozenset()
    coords = frozenset(
        str(name) for name, coord in source.coords.items()
        if (retain_sequence_coords or sequence_dim not in coord.dims)
        and name not in consumed_names
    )
    dimensions = frozenset(
        str(dim)
        for variable in (
            *(source.data_vars.values() if isinstance(source, xr.Dataset) and intent != "index" else ()),
            *(source.coords[name] for name in coords),
        )
        for dim in variable.dims
        if dim not in (sequence_dim, *batch_dims)
    )
    groups = tuple(
        (tuple(str(name) for name in names), index.copy(deep=False))
        for index, names in source.xindexes.group_by_index()
        if set(names) <= coords
        and all(sequence_dim not in variable.dims for variable in names.values())
    )
    return data_vars, dimensions, coords, groups


def preflight_query_output_namespace(
    source: xr.Dataset | xr.DataArray,
    query: object,
    *,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    owner: str,
    intent: Literal["grid", "trajectory", "index"] = "grid",
    generated_names: tuple[str, ...] = (),
    consumed_names: tuple[str, ...] = (),
    retain_sequence_coords: bool = False,
) -> QueryOutputPlan:
    """Reject deterministic caller/output name conflicts before mapping."""
    data_vars, protected_dims, surviving_coords, source_groups = _surviving_source_names(
        source, sequence_dim=sequence_dim, batch_dims=batch_dims, intent=intent,
        consumed_names=consumed_names, retain_sequence_coords=retain_sequence_coords,
    )
    sampled_coords = frozenset(
        name for name in sequence_dependent_coordinate_names(source, sequence_dim=sequence_dim)
        if retain_sequence_coords and name not in consumed_names
    )
    query_only_dims = tuple(dim for dim in query.dims if dim not in batch_dims) if isinstance(query, xr.DataArray) else ()
    size_name = read_sequence_size_coord_name(source) if isinstance(source, xr.Dataset) else None
    plan = QueryOutputPlan(
        intent=intent,
        owner=owner,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        query_only_dims=query_only_dims,
        generated_names=frozenset(generated_names) | sampled_coords,
        optional_generated_names=frozenset((size_name,)) if size_name in generated_names else frozenset(),
        consumed_names=frozenset(consumed_names),
        protected_data_vars=data_vars,
        protected_dims=protected_dims,
        source_coord_names=surviving_coords,
        sampled_source_coord_names=sampled_coords,
        source_index_groups=source_groups,
    )
    if not isinstance(query, xr.DataArray):
        return plan
    require_compatible_shared_batch_index_types(source, query, batch_dims=batch_dims, owner=owner)
    if intent == "trajectory":
        _reject_transform_label_projection(query, plan=plan, owner=owner)
    for name in (*query.dims, *query.coords):
        _preflight_one_query_name(source, query, plan=plan, name=str(name))
    return plan


def prepare_query_topology(
    query: xr.DataArray,
    *,
    query_dim: str,
    stacked_dims: tuple[str, ...] | None,
) -> QueryTopologyPlan:
    """Capture one normalized query's public topology without reading data."""
    dims = tuple(query.dims)
    return QueryTopologyPlan(
        query_dim=query_dim,
        stacked_dims=stacked_dims,
        dims=dims,
        sizes=tuple(int(query.sizes[dim]) for dim in dims),
        coordinates=capture_result_coordinates(
            query,
            output_dims=dims,
            owner="normalize_query_grid",
        ),
    )


def reshape_query_data(
    data: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    lazy: bool,
) -> object:
    """Reshape query metadata without asking Dask to reshape an empty product."""
    if lazy and 0 in shape:
        import dask.array as da

        return da.empty(shape, dtype=dtype, chunks=tuple(max(size, 1) for size in shape))
    return data.reshape(shape)


def _reshape_dataarray(
    value: xr.DataArray,
    *,
    plan: QueryTopologyPlan,
) -> xr.DataArray:
    variable = value.variable
    axis = variable.dims.index(plan.query_dim)
    stacked_dims = plan.stacked_dims or ()
    size_by_dim = dict(zip(plan.dims, plan.sizes, strict=True))
    dims = (*variable.dims[:axis], *stacked_dims, *variable.dims[axis + 1 :])
    shape = (
        *variable.shape[:axis],
        *(size_by_dim[dim] for dim in stacked_dims),
        *variable.shape[axis + 1 :],
    )
    data = variable.data
    if variable.chunks is not None and plan.has_no_rows:
        chunks = tuple(
            variable.chunks[index]
            if dim == plan.query_dim
            else (int(variable.sizes[dim]),)
            for index, dim in enumerate(variable.dims)
        )
        data = data.rechunk(chunks)
    target = xr.DataArray(
        xr.Variable(dims, data.reshape(shape)),
        name=value.name,
    )
    return transfer_dataarray_metadata(value, target)


def _coordinate_source(
    value: xr.Dataset,
    *,
    query_dim: str,
) -> xr.Dataset:
    projected = without_index_topology(value, dims=(query_dim,))
    names = tuple(
        name for name, coord in projected.coords.items() if query_dim in coord.dims
    )
    return projected.drop_vars(names, errors="ignore")


def _caller_coordinates_for_result(
    out: xr.Dataset,
    *,
    plan: QueryTopologyPlan,
    generated_names: set[str],
    owner: str,
) -> ResultCoordinateSnapshot:
    caller = restore_result_coordinates(xr.Dataset(), plan.coordinates)
    assert isinstance(caller, xr.Dataset)
    protected_dims = set(out.dims) - set(plan.stacked_dims or ())
    skip: set[str] = set()
    for name in caller.coords:
        if name in generated_names:
            skip.add(str(name))
            continue
        if name in out.data_vars:
            raise ValueError(f"{owner}: query coordinate {name!r} collides with an output data variable.")
        if name in protected_dims and name not in out.coords:
            raise ValueError(f"{owner}: query coordinate {name!r} collides with a surviving core dimension.")
        if name in out.coords and not _coordinates_compatible(out, caller, name=str(name)):
            raise ValueError(f"{owner}: query coordinate {name!r} conflicts with a surviving output coordinate.")
        if name in out.coords:
            skip.add(str(name))
    for _, coordinates in caller.xindexes.group_by_index():
        names = {str(name) for name in coordinates}
        if names & skip and names - skip:
            raise ValueError(f"{owner}: query index group {sorted(names)!r} conflicts with output coordinates.")
    remaining = caller.drop_vars(tuple(skip), errors="ignore")
    return capture_result_coordinates(
        remaining,
        output_dims=tuple(out.dims),
        owner=owner,
    )


def _restore_stacked_dataset(
    value: xr.Dataset,
    *,
    plan: QueryTopologyPlan,
    owner: str,
) -> xr.Dataset:
    stacked_dims = plan.stacked_dims or ()
    output_dims = tuple(
        dim
        for source_dim in value.dims
        for dim in (stacked_dims if source_dim == plan.query_dim else (source_dim,))
    )
    source_coordinates = capture_result_coordinates(
        _coordinate_source(value, query_dim=plan.query_dim),
        output_dims=output_dims,
        owner=owner,
    )
    data_vars = {
        name: _reshape_dataarray(data_array, plan=plan)
        if plan.query_dim in data_array.dims
        else data_array.copy(deep=False)
        for name, data_array in value.data_vars.items()
    }
    generated_coords = {
        name: _reshape_dataarray(coord, plan=plan).variable
        if plan.query_dim in coord.dims
        else coord.variable.copy(deep=False)
        for name, coord in value.coords.items()
        if name not in (plan.query_dim, *stacked_dims)
    }
    out = xr.Dataset(data_vars=data_vars, coords=generated_coords)
    out.encoding = dict(value.encoding)
    out = _transfer_dataset_attrs_for_finalize(value, out, validate=False)
    out = restore_result_coordinates(out, source_coordinates)
    generated_names = {
        str(name) for name, coord in value.coords.items()
        if plan.query_dim in coord.dims
    }
    caller_coordinates = _caller_coordinates_for_result(
        out,
        plan=plan,
        generated_names=generated_names,
        owner=owner,
    )
    return restore_result_coordinates(out, caller_coordinates)


def restore_query_topology(
    value: xr.Dataset | xr.DataArray,
    *,
    plan: QueryTopologyPlan,
    owner: str,
) -> xr.Dataset | xr.DataArray:
    """Restore labeled stacked-query topology, including an empty product."""
    if plan.stacked_dims is None or plan.query_dim not in value.dims:
        return value
    if isinstance(value, xr.Dataset):
        return _restore_stacked_dataset(
            value,
            plan=plan,
            owner=owner,
        )
    temp_name = unique_temp_dim(
        "__tal_query_value",
        taken_dims=(
            *dataarray_namespace_names(value),
            *tuple(str(name) for name in plan.coordinates.coordinates),
            *(
                str(name)
                for snapshot in plan.coordinates.indexes
                for name in snapshot.coordinates
            ),
        ),
    )
    restored = _restore_stacked_dataset(
        value.to_dataset(name=temp_name),
        plan=plan,
        owner=owner,
    )
    return restored[temp_name].rename(value.name)


def _reject_transform_label_projection(
    caller: xr.Dataset | xr.DataArray,
    *,
    plan: QueryOutputPlan,
    owner: str,
) -> None:
    flattened = set(plan.query_only_dims)
    for index, names in caller.xindexes.group_by_index():
        if not isinstance(index, xr.indexes.CoordinateTransformIndex):
            continue
        if any(flattened.intersection(caller.coords.variables[name].dims) for name in names):
            raise ValueError(
                f"{owner}: flattening transform-backed query labels requires coordinate evaluation; "
                "materialize those labels explicitly before the typed query."
            )


def _trajectory_caller_coordinate(
    coord: xr.DataArray,
    *,
    topology: QueryTopologyPlan,
    plan: QueryOutputPlan,
) -> xr.DataArray:
    flattened = set(plan.query_only_dims)
    if not flattened.intersection(coord.dims):
        return coord
    leading = tuple(dim for dim in plan.batch_dims if dim in topology.dims)
    dims = (*leading, *plan.query_only_dims)
    size_by_dim = dict(zip(topology.dims, topology.sizes, strict=True))
    fillers = tuple(
        xr.DataArray(np.arange(size_by_dim[dim]), dims=(dim,))
        for dim in dims if dim not in coord.dims
    )
    expanded = xr.broadcast(coord, *fillers)[0].transpose(*dims)
    shape = tuple(size_by_dim[dim] for dim in leading) + (int(prod(size_by_dim[dim] for dim in plan.query_only_dims)),)
    projected = xr.DataArray(
        reshape_query_data(
            expanded.variable.data,
            shape=shape,
            dtype=expanded.dtype,
            lazy=expanded.chunks is not None,
        ),
        dims=(*leading, plan.sequence_dim),
        name=coord.name,
    )
    return transfer_dataarray_metadata(coord, projected)


def attach_trajectory_query_coordinates(
    value: xr.Dataset,
    *,
    topology: QueryTopologyPlan,
    plan: QueryOutputPlan,
    owner: str,
) -> xr.Dataset:
    """Attach caller labels after typed output metadata has been generated."""
    caller = restore_result_coordinates(xr.Dataset(), topology.coordinates)
    assert isinstance(caller, xr.Dataset)
    _reject_transform_label_projection(caller, plan=plan, owner=owner)
    out = value
    for name in caller.coords:
        if name == plan.sequence_dim and name in topology.dims and plan.intent == "trajectory":
            continue  # The input lane is consumed; the output sequence remains positional.
        if name in plan.generated_names or name in plan.consumed_names:
            continue
        if name in (*plan.batch_dims, plan.sequence_dim) and not caller.coords.variables[name].dims and name not in topology.dims:
            continue
        out = _attach_one_trajectory_coordinate(
            out, caller=caller, name=str(name), topology=topology, plan=plan, owner=owner
        )
    return out


def _attach_one_trajectory_coordinate(
    out: xr.Dataset,
    *,
    caller: xr.Dataset,
    name: str,
    topology: QueryTopologyPlan,
    plan: QueryOutputPlan,
    owner: str,
) -> xr.Dataset:
    if name in out.data_vars or (name == plan.sequence_dim and name not in plan.batch_dims):
        raise ValueError(f"{owner}: query coordinate {name!r} collides with typed output topology.")
    if name in plan.batch_dims and name in out.xindexes and name in caller.xindexes:
        return out  # The source batch index owns labels after the query was reindexed to it.
    projected = _trajectory_caller_coordinate(caller.coords[name], topology=topology, plan=plan)
    if name not in out.coords:
        return out.assign_coords({name: projected})
    if name not in plan.source_coord_names:
        return out  # The current query has already contributed this coordinate.
    candidate = (
        xr.Dataset(coords={name: projected})
        if set(plan.query_only_dims).intersection(caller.coords.variables[name].dims)
        else caller
    )
    if not _coordinates_compatible(out, candidate, name=name):
        raise ValueError(f"{owner}: query coordinate {name!r} conflicts with a surviving output coordinate.")
    return out


__all__: list[str] = []
