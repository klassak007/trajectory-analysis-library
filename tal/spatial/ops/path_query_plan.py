from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.param_engine import ParamMapOptions, normalize_query_grid
from tal.core.param_engine.prepared import PreparedParamEvaluation
from tal.core.param_ops.runtime_prepare import prepare_runtime_param_evaluation
from tal.core.param_ops.types import ParamRuntimeContext
from tal.utils.xarray_namespace import dataset_namespace_names, unique_temp_dim

from ..pose import Pose
from ..position import Position
from ..rotation import Rotation
from ..temporal.options import PoseTemporalOptions, resolve_rotation_method
from .path_query_topology import (
    OutputTopology,
    _ProviderBatchProjection,
    align_projected_provider_batches,
    batch_coordinates,
    build_direct_topology,
    project_provider_batches,
    without_batch_coordinates,
)
from .provider_topology import ProviderTopology, classify_provider_topology

PathOutputIntent = Literal["pose", "position", "unsupported"]


@dataclass(frozen=True)
class RequiredProvider:
    """One acquired provider occurrence in path order."""

    value: object
    topology: ProviderTopology
    source_representation: str | None = None


@dataclass(frozen=True)
class PreparedProviderQuery:
    """One provider after topology and parameter preparation."""

    required: RequiredProvider
    projection: _ProviderBatchProjection | None
    context: ParamRuntimeContext | None
    temporal: PoseTemporalOptions
    evaluations: tuple[PreparedParamEvaluation, ...] = ()


@dataclass(frozen=True)
class PreparedPathQuery:
    """Immutable provider/query plan created before payload evaluation."""

    providers: tuple[RequiredProvider, ...]
    items: tuple[PreparedProviderQuery, ...]
    topology: OutputTopology | None
    output_intent: PathOutputIntent
    result_context: object | None = None


def provider_context(
    value: object,
    temporal: PoseTemporalOptions,
    *,
    owner: str,
) -> ParamRuntimeContext:
    try:
        source = value if temporal.on is None else value.set_param_coord(name=temporal.on, validate=False)
        return resolve_param_runtime_context(source, on=temporal.on)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{owner}: could not resolve provider parameter coordinate; {exc}") from exc


def _query_kind(query: object, *, owner: str) -> str:
    dtype = getattr(query, "dtype", None)
    if dtype is None:
        try:
            dtype = np.asarray(query).dtype
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{owner}: query values must be numeric or datetime-like.") from exc
    return "datetime64" if np.issubdtype(np.dtype(dtype), np.datetime64) else "numeric"


def _one_param_kind(contexts: Sequence[ParamRuntimeContext], *, owner: str) -> str:
    kinds = {context.param_kind for context in contexts}
    if len(kinds) == 1:
        return next(iter(kinds))
    raise ValueError(f"{owner}: dynamic providers use mixed numeric and datetime64 parameter domains.")


def _object_topology(
    caller: object,
    temporal: PoseTemporalOptions,
    contexts: Sequence[ParamRuntimeContext],
    values: Sequence[object],
    *,
    owner: str,
) -> OutputTopology:
    try:
        context = resolve_param_runtime_context(caller)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: a dynamic object request requires a caller param_coord; {exc}") from exc
    if contexts and context.param_kind != _one_param_kind(contexts, owner=owner):
        raise ValueError(f"{owner}: caller and providers use mixed parameter domains.")
    names = dataset_namespace_names(analysis_object_dataset(caller))
    query_dim = temporal.position_opts.query_dim
    if query_dim in names:
        query_dim = unique_temp_dim("__tal_path_query", taken_dims=names)
    evaluation_query = without_batch_coordinates(context.spec.coord, batch_dims=context.batch_dims)
    grid = normalize_query_grid(
        evaluation_query,
        query_dim=query_dim,
        batch_dims=context.batch_dims,
        param_kind=context.param_kind,
    )
    topology = OutputTopology(
        grid.values,
        grid.query_dim,
        context.sequence_dim,
        context.spec.name,
        context.param_kind,
        context.batch_dims,
        batch_coordinates(context.ds, batch_dims=context.batch_dims),
        tuple((dim, context.ds) for dim in context.batch_dims),
        context,
    )
    projected = project_provider_batches(
        values,
        context.ds,
        batch_dims=context.batch_dims,
        param_on=temporal.on,
        owner=owner,
    )
    return replace(topology, provider_values=align_projected_provider_batches(projected, topology, owner=owner))


def _direct_topology(
    query: object,
    values: Sequence[object],
    contexts: Sequence[ParamRuntimeContext],
    temporal: PoseTemporalOptions,
    *,
    owner: str,
) -> OutputTopology:
    param_kind = _one_param_kind(contexts, owner=owner) if contexts else _query_kind(query, owner=owner)
    return build_direct_topology(
        query,
        values,
        param_kind=param_kind,
        param_on=temporal.on,
        public_query_dim=temporal.position_opts.query_dim,
        owner=owner,
    )


def map_options(value: object, temporal: PoseTemporalOptions) -> tuple[ParamMapOptions, ...]:
    position = temporal.position_opts
    rotation = temporal.rotation_opts
    rotation_method = resolve_rotation_method(rotation)
    rotation_map = ParamMapOptions(
        method="nearest" if rotation_method == "nearest" else "linear",
        duplicate_policy=rotation.duplicate_policy,
    )
    if isinstance(value, Rotation):
        return (rotation_map,)
    position_map = ParamMapOptions(method=position.method, duplicate_policy=position.duplicate_policy)
    if not isinstance(value, Pose) or position_map == rotation_map:
        return (position_map,)
    return position_map, rotation_map


def _effective_temporal(
    temporal: PoseTemporalOptions,
    projection: _ProviderBatchProjection,
    *,
    query_dim: str,
) -> PoseTemporalOptions:
    position = replace(temporal.position_opts, query_dim=query_dim)
    rotation = replace(temporal.rotation_opts, query_dim=query_dim)
    return replace(
        temporal,
        position_opts=position,
        rotation_opts=rotation,
        on=projection.param_on,
    )


def _prepare_provider_context(
    required: RequiredProvider,
    projection: _ProviderBatchProjection,
    topology: OutputTopology,
    temporal: PoseTemporalOptions,
    *,
    owner: str,
) -> PreparedProviderQuery:
    effective = _effective_temporal(temporal, projection, query_dim=topology.query_dim)
    if required.topology != "dynamic":
        return PreparedProviderQuery(required, projection, None, effective)
    context = provider_context(projection.value, effective, owner=owner)
    return PreparedProviderQuery(required, projection, context, effective)


def _prepare_provider_evaluations(
    item: PreparedProviderQuery,
    topology: OutputTopology,
    reuse: list[PreparedParamEvaluation],
    defer_backend_selection: bool,
) -> PreparedProviderQuery:
    context = item.context
    if context is None or item.projection is None:
        return item
    evaluations: list[PreparedParamEvaluation] = []
    for options in map_options(item.projection.value, item.temporal):
        evaluation = prepare_runtime_param_evaluation(
            context,
            query=topology.query,
            options=options,
            param_kind=context.param_kind,
            query_dim=topology.query_dim,
            reuse=reuse,
            defer_backend_selection=defer_backend_selection,
        )
        evaluations.append(evaluation)
        if all(evaluation is not candidate for candidate in reuse):
            reuse.append(evaluation)
    return replace(item, evaluations=tuple(evaluations))


def _preflight_batched_metadata(plan: PreparedPathQuery) -> bool:
    topology = plan.topology
    if topology is None or not topology.batch_dims:
        return False
    from .batched_path_plan import classify_batched_path_metadata

    classification = classify_batched_path_metadata(plan)
    return classification.map_is_lazy


def _required_providers(
    values: Sequence[object],
    representations: Sequence[str | None] | None,
    *,
    owner: str,
) -> tuple[RequiredProvider, ...]:
    resolved = (None,) * len(values) if representations is None else representations
    if len(resolved) != len(values):
        raise ValueError(f"{owner}: provider representation count does not match provider count.")
    return tuple(
        RequiredProvider(value, classify_provider_topology(value), representation)
        for value, representation in zip(values, resolved, strict=True)
    )


def _evaluate_providers(
    items: tuple[PreparedProviderQuery, ...],
    topology: OutputTopology,
    *,
    defer_backend_selection: bool,
) -> tuple[PreparedProviderQuery, ...]:
    reuse: list[PreparedParamEvaluation] = []
    return tuple(
        _prepare_provider_evaluations(
            item, topology, reuse, defer_backend_selection,
        )
        for item in items
    )


def prepare_path_query(
    values: Sequence[object],
    *,
    query: object | None,
    caller: object | None,
    temporal: PoseTemporalOptions,
    owner: str,
    source_representations: Sequence[str | None] | None = None,
    result_context: object | None = None,
) -> PreparedPathQuery:
    """Classify topology and prepare reusable maps before payload work."""
    providers = _required_providers(
        values,
        source_representations,
        owner=owner,
    )
    output_intent: PathOutputIntent
    if isinstance(caller, Position):
        output_intent = "position"
    elif caller is None:
        output_intent = "pose"
    else:
        output_intent = "unsupported"
    has_dynamic = any(item.topology == "dynamic" for item in providers)
    has_exact = any(item.topology == "exact" for item in providers)
    if has_exact and (has_dynamic or query is not None):
        raise ValueError(f"{owner}: exact providers cannot be combined with dynamic providers or an explicit query.")
    if has_dynamic and caller is None and query is None:
        raise ValueError(f"{owner}: dynamic direct path solving requires explicit query=.")
    if not (has_dynamic or (query is not None and not has_exact)):
        items = tuple(PreparedProviderQuery(item, None, None, temporal) for item in providers)
        return PreparedPathQuery(providers, items, None, output_intent, result_context)
    contexts = tuple(provider_context(item.value, temporal, owner=owner) for item in providers if item.topology == "dynamic")
    topology = (
        _object_topology(caller, temporal, contexts, values, owner=owner)
        if caller is not None
        else _direct_topology(query, values, contexts, temporal, owner=owner)
    )
    items = tuple(
        _prepare_provider_context(item, projection, topology, temporal, owner=owner)
        for item, projection in zip(providers, topology.provider_values, strict=True)
    )
    prepared = PreparedPathQuery(providers, items, topology, output_intent, result_context)
    defer_backend_selection = _preflight_batched_metadata(prepared)
    evaluated = _evaluate_providers(
        items,
        topology,
        defer_backend_selection=defer_backend_selection,
    )
    return replace(prepared, items=evaluated)


__all__ = [
    "PreparedPathQuery",
    "PreparedProviderQuery",
    "RequiredProvider",
    "map_options",
    "prepare_path_query",
    "provider_context",
]
