from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.param_engine import normalize_query_grid
from tal.utils.xarray_namespace import (
    dataarray_namespace_names,
    dataset_namespace_names,
    unique_temp_dim,
)

from .path_query_topology import (
    OutputTopology,
    _provider_surviving_names,
    _ProviderBatchProjection,
    _target_batch_names,
    align_projected_provider_batches,
    batch_coordinates,
    project_provider_batches,
    without_batch_coordinates,
)

XarrayObject = xr.Dataset | xr.DataArray


@dataclass(frozen=True)
class _DirectEvaluationPlan:
    query: xr.DataArray
    query_dim: str
    batch_dims: tuple[str, ...]
    provider_values: tuple[_ProviderBatchProjection, ...]


def _query_batch_dims(query: object) -> tuple[str, ...]:
    if not isinstance(query, xr.DataArray) or query.ndim <= 1:
        return ()
    return tuple(str(dim) for dim in query.dims[:-1])


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


def _transient_namespace(query: object, values: Sequence[object]) -> set[str]:
    names = set(dataarray_namespace_names(query)) if isinstance(query, xr.DataArray) else set()
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
    if isinstance(target, xr.DataArray):
        consumed_axis = str(target.dims[-1]) if target.ndim else None
        names.update(
            str(name)
            for name in dataarray_namespace_names(target)
            if str(name) != consumed_axis
        )
    names.add(public_query_dim)
    for value in values:
        names.update(_provider_surviving_names(value))
    return names


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
    return unique_temp_dim(f"{public_query_dim}_value", taken_dims=tuple(names))


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
        values, target, batch_dims=batch_dims, param_on=param_on, owner=owner,
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
    topology = OutputTopology(
        evaluation.query,
        evaluation.query_dim,
        public_query_dim,
        param_name,
        param_kind,
        evaluation.batch_dims,
        coords,
        tuple((dim, source) for dim in evaluation.batch_dims),
        None,
    )
    providers = align_projected_provider_batches(
        evaluation.provider_values, topology, owner=owner,
    )
    return replace(
        topology,
        provider_values=providers,
        native_provider_values=evaluation.provider_values,
    )


__all__ = ["build_direct_topology"]
