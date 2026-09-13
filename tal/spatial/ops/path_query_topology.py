from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.indexing import (
    isel_rows,
    lane_index_groups,
    require_exact_lane_indexes,
    require_unique_lane_indexes,
)
from tal.core.param_engine import normalize_query_grid
from tal.core.param_ops.types import ParamRuntimeContext
from tal.core.schema import set_roles
from tal.core.schema_read import read_roles, read_sequence_size_coord_name
from tal.utils.xarray_namespace import (
    dataarray_namespace_names,
    dataset_namespace_names,
    unique_temp_dim,
)

XarrayObject = xr.Dataset | xr.DataArray


@dataclass(frozen=True)
class _ProviderBatchProjection:
    """One provider after metadata-only batch-topology projection."""

    value: object
    param_on: str | None


@dataclass(frozen=True)
class OutputTopology:
    """One request-local query grid and its public output topology."""

    query: xr.DataArray
    query_dim: str
    sequence_dim: str
    param_name: str
    param_kind: str
    batch_dims: tuple[str, ...]
    batch_coords: xr.Coordinates
    batch_sources: tuple[tuple[str, XarrayObject], ...]
    caller: ParamRuntimeContext | None
    provider_values: tuple[_ProviderBatchProjection, ...] = ()


@dataclass(frozen=True)
class _DirectEvaluationPlan:
    query: xr.DataArray
    query_dim: str
    batch_dims: tuple[str, ...]
    provider_values: tuple[_ProviderBatchProjection, ...]


def _batch_coordinate_names(
    value: XarrayObject,
    *,
    batch_dims: tuple[str, ...],
) -> tuple[object, ...]:
    allowed = set(batch_dims)
    return tuple(
        name
        for name, coord in value.coords.items()
        if coord.dims and set(coord.dims).issubset(allowed)
    )


def batch_coordinates(
    value: XarrayObject,
    *,
    batch_dims: tuple[str, ...],
) -> xr.Coordinates:
    """Select batch-only coordinates while preserving public xarray indexes."""
    selected_names = _batch_coordinate_names(value, batch_dims=batch_dims)
    selected = set(selected_names)
    variables = {name: value.coords[name].variable for name in selected_names}
    indexes: dict[object, xr.Index] = {}
    for index, coordinates in value.xindexes.group_by_index():
        if not coordinates or not set(coordinates).issubset(selected):
            continue
        copied = index.copy(deep=False)
        variables.update(copied.create_variables(coordinates))
        indexes.update({name: copied for name in coordinates})
    return xr.Coordinates(variables, indexes=indexes)


def _without_batch_indexes(
    value: XarrayObject,
    *,
    batch_dims: tuple[str, ...],
) -> XarrayObject:
    allowed = set(batch_dims)
    names: list[object] = []
    for _, coordinates in value.xindexes.group_by_index():
        if coordinates and all(
            variable.dims and set(variable.dims).issubset(allowed)
            for variable in coordinates.values()
        ):
            names.extend(coordinates)
    return value.drop_vars(names) if names else value


def without_batch_coordinates(
    value: xr.DataArray,
    *,
    batch_dims: tuple[str, ...],
) -> xr.DataArray:
    names = _batch_coordinate_names(value, batch_dims=batch_dims)
    return value.drop_vars(names) if names else value


def _provider_batch_dims(value: object) -> tuple[str, ...]:
    return read_roles(analysis_object_dataset(value))[2]


def _require_matching_batch(
    source: XarrayObject,
    target: XarrayObject,
    *,
    dim: str,
    owner: str,
) -> None:
    indexed = require_exact_lane_indexes(
        source,
        target,
        lane_dim=dim,
        owner=owner,
        what="provider batch",
    )
    if not indexed and int(source.sizes[dim]) != int(target.sizes[dim]):
        raise ValueError(f"{owner}: unindexed provider batch {dim!r} must match by size.")


def _query_batch_dims(query: object) -> tuple[str, ...]:
    if not isinstance(query, xr.DataArray) or query.ndim <= 1:
        return ()
    return tuple(str(dim) for dim in query.dims[:-1])


def _provider_surviving_names(value: object) -> set[str]:
    ds = analysis_object_dataset(value)
    sequence_dim = read_roles(ds)[1]
    names = {str(name) for name in ds.data_vars}
    names.update(str(dim) for dim in ds.dims if dim != sequence_dim)
    names.update(
        str(name)
        for name, coord in ds.coords.items()
        if sequence_dim is None or sequence_dim not in coord.dims
    )
    return names


def _target_batch_names(
    target: XarrayObject | None,
    *,
    batch_dims: tuple[str, ...],
) -> set[str]:
    names = set(batch_dims)
    if target is not None:
        names.update(str(name) for name in _batch_coordinate_names(target, batch_dims=batch_dims))
    return names


def _consumed_provider_names(value: object) -> set[str]:
    ds = analysis_object_dataset(value)
    sequence_dim = read_roles(ds)[1]
    if sequence_dim is None:
        return set()
    names = {
        str(name)
        for name, coord in ds.coords.items()
        if sequence_dim in coord.dims
    }
    names.add(sequence_dim)
    size_name = read_sequence_size_coord_name(ds)
    if size_name is not None:
        names.add(size_name)
    return names


def _rename_consumed_batch_conflicts(
    value: object,
    *,
    target_names: set[str],
    param_on: str | None,
) -> _ProviderBatchProjection:
    collisions = sorted(_consumed_provider_names(value) & target_names)
    if not collisions:
        return _ProviderBatchProjection(value, param_on)
    taken = set(dataset_namespace_names(analysis_object_dataset(value))) | target_names
    rename_map: dict[str, str] = {}
    for name in collisions:
        replacement = unique_temp_dim(f"__tal_provider_{name}", taken_dims=tuple(taken))
        rename_map[name] = replacement
        taken.add(replacement)
    renamed = value.rename(rename_map, validate=False)
    return _ProviderBatchProjection(renamed, rename_map.get(param_on, param_on))


def _provider_namespace_collisions(
    value: object,
    *,
    target_names: set[str],
    shared_batch_dims: set[str],
    shared_index_names: set[str],
) -> set[str]:
    ds = analysis_object_dataset(value)
    collisions = set(target_names) & {str(name) for name in ds.data_vars}
    collisions.update(
        name for name in target_names if name in ds.dims and name not in shared_batch_dims
    )
    for name in target_names & {str(name) for name in ds.coords}:
        if name in shared_index_names:
            continue
        collisions.add(name)
    return collisions


def _shared_batch_index_names(value: object, dims: tuple[str, ...]) -> set[str]:
    ds = analysis_object_dataset(value)
    return {
        str(name)
        for dim in dims
        for group in lane_index_groups(ds, lane_dim=dim)
        for name in group.coordinate_names
    }


def _require_public_sequence_safe(
    target: XarrayObject | None,
    values: Sequence[object],
    *,
    batch_dims: tuple[str, ...],
    sequence_dim: str,
    owner: str,
) -> None:
    collisions = {
        sequence_dim
        for value in values
        if sequence_dim in _provider_surviving_names(value)
    }
    if sequence_dim in _target_batch_names(target, batch_dims=batch_dims):
        collisions.add(sequence_dim)
    if isinstance(target, xr.DataArray):
        final_dim = str(target.dims[-1]) if target.ndim else None
        if sequence_dim in target.coords and sequence_dim != final_dim:
            collisions.add(sequence_dim)
    if collisions:
        raise ValueError(
            f"{owner}: query_dim {sequence_dim!r} collides with surviving "
            "provider or query batch topology. Choose a distinct temporal query_dim."
        )


def _transient_namespace(
    query: object,
    values: Sequence[object],
) -> set[str]:
    names = (
        set(dataarray_namespace_names(query))
        if isinstance(query, xr.DataArray)
        else set()
    )
    for value in values:
        names.update(dataset_namespace_names(analysis_object_dataset(value)))
    return names


def _surviving_namespace(
    target: XarrayObject | None,
    values: Sequence[object],
    *,
    batch_dims: tuple[str, ...],
    public_query_dim: str,
) -> set[str]:
    names = _target_batch_names(target, batch_dims=batch_dims)
    names.add(public_query_dim)
    for value in values:
        names.update(_provider_surviving_names(value))
    return names


def _validate_provider_batch_shape(
    value: object,
    target: XarrayObject | None,
    *,
    batch_dims: tuple[str, ...],
    owner: str,
) -> tuple[str, ...]:
    ds = analysis_object_dataset(value)
    provider_dims = _provider_batch_dims(value)
    present = tuple(dim for dim in provider_dims if dim in batch_dims)
    if tuple(dim for dim in batch_dims if dim in present) != present:
        raise ValueError(f"{owner}: provider batch dimensions must follow query order.")
    if target is not None:
        for dim in present:
            _require_matching_batch(ds, target, dim=dim, owner=owner)
    extras = tuple(dim for dim in provider_dims if dim not in batch_dims)
    bad = tuple(dim for dim in extras if int(ds.sizes[dim]) != 1)
    if bad:
        raise ValueError(
            f"{owner}: provider-only batch dims {list(bad)!r} would create Cartesian expansion."
        )
    return present


def _project_provider_batch(
    value: object,
    target: XarrayObject | None,
    *,
    target_names: set[str],
    batch_dims: tuple[str, ...],
    param_on: str | None,
    owner: str,
) -> _ProviderBatchProjection:
    try:
        present = _validate_provider_batch_shape(
            value, target, batch_dims=batch_dims, owner=owner,
        )
        item = _squeeze_provider_batches(value, target_dims=batch_dims)
        projection = _rename_consumed_batch_conflicts(
            item,
            target_names=target_names,
            param_on=param_on,
        )
        collisions = _provider_namespace_collisions(
            projection.value,
            target_names=target_names,
            shared_batch_dims=set(present),
            shared_index_names=_shared_batch_index_names(projection.value, present),
        )
        if collisions:
            raise ValueError(
                f"{owner}: query batch topology collides with surviving provider names "
                f"{sorted(collisions)!r}. Rename the query dimensions or coordinates."
            )
        return projection
    except (KeyError, TypeError, ValueError) as exc:
        text = str(exc)
        if text.startswith(f"{owner}:"):
            raise
        raise ValueError(f"{owner}: failed to project provider topology; {text}") from exc


def project_provider_batches(
    values: Sequence[object],
    target: XarrayObject | None,
    *,
    batch_dims: tuple[str, ...],
    param_on: str | None,
    owner: str,
) -> tuple[_ProviderBatchProjection, ...]:
    """Validate and project providers to effective pre-evaluation topology."""
    target_names = _target_batch_names(target, batch_dims=batch_dims)
    if target is not None:
        for dim in batch_dims:
            require_unique_lane_indexes(target, lane_dim=dim, owner=owner)
    return tuple(
        _project_provider_batch(
            value,
            target,
            target_names=target_names,
            batch_dims=batch_dims,
            param_on=param_on,
            owner=owner,
        )
        for value in values
    )


def _normalize_direct_query(
    query: object,
    *,
    batch_dims: tuple[str, ...],
    internal_dim: str,
    param_kind: str,
    owner: str,
) -> xr.DataArray:
    evaluation_query = (
        without_batch_coordinates(query, batch_dims=batch_dims)
        if isinstance(query, xr.DataArray)
        else query
    )
    try:
        grid = normalize_query_grid(
            evaluation_query,
            query_dim=internal_dim,
            batch_dims=batch_dims,
            param_kind=param_kind,
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{owner}: {exc}") from exc
    return grid.values


def _direct_param_name(
    target: XarrayObject | None,
    values: Sequence[object],
    normalized: xr.DataArray,
    *,
    batch_dims: tuple[str, ...],
    public_query_dim: str,
) -> str:
    if normalized.ndim <= 1:
        return public_query_dim
    names = _surviving_namespace(
        target,
        values,
        batch_dims=batch_dims,
        public_query_dim=public_query_dim,
    )
    return unique_temp_dim(
        f"{public_query_dim}_value",
        taken_dims=tuple(names),
    )


def _prepare_direct_evaluation(
    query: object,
    values: Sequence[object],
    *,
    param_kind: str,
    param_on: str | None,
    public_query_dim: str,
    owner: str,
) -> _DirectEvaluationPlan:
    batch_dims = _query_batch_dims(query)
    target = query if isinstance(query, xr.DataArray) else None
    projected = project_provider_batches(
        values,
        target,
        batch_dims=batch_dims,
        param_on=param_on,
        owner=owner,
    )
    projected_values = tuple(item.value for item in projected)
    _require_public_sequence_safe(
        target,
        projected_values,
        batch_dims=batch_dims,
        sequence_dim=public_query_dim,
        owner=owner,
    )
    names = _transient_namespace(query, projected_values)
    internal_dim = unique_temp_dim("__tal_path_query", taken_dims=tuple(names))
    normalized = _normalize_direct_query(
        query,
        batch_dims=batch_dims,
        internal_dim=internal_dim,
        param_kind=param_kind,
        owner=owner,
    )
    return _DirectEvaluationPlan(normalized, internal_dim, batch_dims, projected)


def build_direct_topology(
    query: object,
    values: Sequence[object],
    *,
    param_kind: str,
    param_on: str | None,
    public_query_dim: str,
    owner: str,
) -> OutputTopology:
    """Build one query-owned direct-solve topology without payload work."""
    evaluation = _prepare_direct_evaluation(
        query,
        values,
        param_kind=param_kind,
        param_on=param_on,
        public_query_dim=public_query_dim,
        owner=owner,
    )
    coords = (
        batch_coordinates(query, batch_dims=evaluation.batch_dims)
        if isinstance(query, xr.DataArray)
        else xr.Coordinates()
    )
    param_name = _direct_param_name(
        query if isinstance(query, xr.DataArray) else None,
        tuple(item.value for item in evaluation.provider_values),
        evaluation.query,
        batch_dims=evaluation.batch_dims,
        public_query_dim=public_query_dim,
    )
    source = query if isinstance(query, xr.DataArray) else evaluation.query
    sources = tuple((dim, source) for dim in evaluation.batch_dims)
    topology = OutputTopology(
        evaluation.query,
        evaluation.query_dim,
        public_query_dim,
        param_name,
        param_kind,
        evaluation.batch_dims,
        coords,
        sources,
        None,
    )
    provider_values = align_projected_provider_batches(
        evaluation.provider_values,
        topology,
        owner=owner,
    )
    return replace(topology, provider_values=provider_values)


def _squeeze_provider_batches(
    value: object,
    *,
    target_dims: tuple[str, ...],
) -> object:
    ds = analysis_object_dataset(value)
    original_batch_dims = _provider_batch_dims(value)
    extras = tuple(dim for dim in original_batch_dims if dim not in target_dims)
    for dim in extras:
        ds = isel_rows(ds, dim=dim, rows=np.asarray([0], dtype=np.intp))
        ds = ds.isel({dim: 0}, drop=True)
    if not extras:
        return value
    _, sequence_dim, _, core_dims = read_roles(ds)
    ds = set_roles(
        ds,
        sequence_dim=sequence_dim,
        batch_dims=tuple(dim for dim in original_batch_dims if dim not in extras),
        core_dims=core_dims,
        validate=False,
    )
    return value.__class__._from_unvalidated(ds)


def _align_projected_provider(
    projection: _ProviderBatchProjection,
    topology: OutputTopology,
    *,
    owner: str,
) -> _ProviderBatchProjection:
    try:
        value = projection.value
        ds = analysis_object_dataset(value)
        present = _provider_batch_dims(value)
        sources = dict(topology.batch_sources)
        ds = _without_batch_indexes(ds, batch_dims=topology.batch_dims)
        missing = tuple(dim for dim in topology.batch_dims if dim not in present)
        if missing:
            ds = ds.expand_dims(
                {dim: int(sources[dim].sizes[dim]) for dim in missing}
            )
        _, sequence_dim, _, core_dims = read_roles(ds)
        ds = set_roles(
            ds,
            sequence_dim=sequence_dim,
            batch_dims=topology.batch_dims,
            core_dims=core_dims,
            validate=False,
        )
        aligned = value.__class__._from_unvalidated(ds)
        return _ProviderBatchProjection(aligned, projection.param_on)
    except (KeyError, TypeError, ValueError) as exc:
        text = str(exc)
        if text.startswith(f"{owner}:"):
            raise
        raise ValueError(f"{owner}: failed to project provider batch topology; {text}") from exc


def align_projected_provider_batches(
    values: Sequence[_ProviderBatchProjection],
    topology: OutputTopology,
    *,
    owner: str,
) -> tuple[_ProviderBatchProjection, ...]:
    """Broadcast validated provider projections to the query-owned topology."""
    return tuple(
        _align_projected_provider(value, topology, owner=owner)
        for value in values
    )


__all__ = [
    "OutputTopology",
    "align_projected_provider_batches",
    "batch_coordinates",
    "build_direct_topology",
    "project_provider_batches",
    "without_batch_coordinates",
]
