from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.param_engine import ParamMapOptions, build_param_map, normalize_query_grid
from tal.core.param_ops.types import ParamRuntimeContext
from tal.core.schema import set_param_coord, set_roles, set_validity
from tal.core.schema_read import read_roles, read_sequence_size_coord_name
from tal.utils.xarray_namespace import (
    dataset_namespace_names,
    unique_temp_dim,
)

from ..pose import Pose
from ..rotation import Rotation
from ..temporal.options import (
    PoseTemporalOptions,
    _validate_pose_temporal_options,
    resolve_rotation_method,
)
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


@dataclass(frozen=True)
class RequiredProvider:
    """One acquired provider occurrence in path order."""

    value: object
    topology: ProviderTopology


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


def _provider_context(
    value,
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


def _require_one_param_kind(contexts: Sequence[ParamRuntimeContext], *, owner: str) -> str:
    kinds = {context.param_kind for context in contexts}
    if len(kinds) == 1:
        return next(iter(kinds))
    raise ValueError(f"{owner}: dynamic providers use mixed numeric and datetime64 parameter domains.")


def _topology_from_context(
    context: ParamRuntimeContext,
) -> xr.Coordinates:
    return batch_coordinates(context.ds, batch_dims=context.batch_dims)


def _object_topology(
    caller,
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
    if contexts and context.param_kind != _require_one_param_kind(contexts, owner=owner):
        raise ValueError(f"{owner}: caller and providers use mixed parameter domains.")
    names = dataset_namespace_names(analysis_object_dataset(caller))
    query_dim = temporal.position_opts.query_dim
    if query_dim in names:
        query_dim = unique_temp_dim("__tal_path_query", taken_dims=names)
    evaluation_query = without_batch_coordinates(
        context.spec.coord,
        batch_dims=context.batch_dims,
    )
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
        _topology_from_context(context),
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
    aligned = align_projected_provider_batches(projected, topology, owner=owner)
    return replace(topology, provider_values=aligned)


def _direct_topology(
    query: object,
    values: Sequence[object],
    contexts: Sequence[ParamRuntimeContext],
    temporal: PoseTemporalOptions,
    *,
    owner: str,
) -> OutputTopology:
    param_kind = _require_one_param_kind(contexts, owner=owner) if contexts else _query_kind(query, owner=owner)
    return build_direct_topology(
        query,
        values,
        param_kind=param_kind,
        param_on=temporal.on,
        public_query_dim=temporal.position_opts.query_dim,
        owner=owner,
    )


def _map_options(value: object, temporal: PoseTemporalOptions) -> tuple[ParamMapOptions, ...]:
    position = temporal.position_opts
    rotation = temporal.rotation_opts
    rotation_method = resolve_rotation_method(rotation)
    rotation_map = ParamMapOptions(
        method="nearest" if rotation_method == "nearest" else "linear",
        duplicate_policy=rotation.duplicate_policy,
    )
    if isinstance(value, Rotation):
        return (rotation_map,)
    if not isinstance(value, Pose):
        return (ParamMapOptions(method=position.method, duplicate_policy=position.duplicate_policy),)
    position_map = ParamMapOptions(method=position.method, duplicate_policy=position.duplicate_policy)
    return (position_map,) if position_map == rotation_map else (position_map, rotation_map)


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
    value: object,
    topology: OutputTopology,
    temporal: PoseTemporalOptions,
    *,
    owner: str,
) -> None:
    context = _provider_context(value, temporal, owner=owner)
    caller_valid = topology.caller.valid_mask if topology.caller is not None else None
    for options in _map_options(value, temporal):
        pmap = build_param_map(
            param=context.spec.coord,
            query=topology.query,
            sequence_dim=context.sequence_dim,
            query_dim=topology.query_dim,
            valid_mask=context.valid_mask,
            options=options,
            param_kind=context.param_kind,
        )
        valid = pmap.valid if caller_valid is None else (pmap.valid | ~caller_valid.rename({topology.sequence_dim: topology.query_dim}))
        _realize_coverage(valid, owner=owner)


def _temporal_for_query_dim(temporal: PoseTemporalOptions, query_dim: str) -> PoseTemporalOptions:
    position = replace(temporal.position_opts, query_dim=query_dim)
    rotation = replace(temporal.rotation_opts, query_dim=query_dim)
    return replace(temporal, position_opts=position, rotation_opts=rotation)


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


def _evaluate_dynamic(
    projection: _ProviderBatchProjection,
    topology: OutputTopology,
    temporal: PoseTemporalOptions,
    *,
    owner: str,
) -> object:
    aligned = projection.value
    effective = replace(
        _temporal_for_query_dim(temporal, topology.query_dim),
        on=projection.param_on,
    )
    _require_provider_coverage(aligned, topology, effective, owner=owner)
    try:
        if isinstance(aligned, Pose):
            evaluated = aligned.param.at(topology.query, opts=effective, validate=False)
        elif isinstance(aligned, Rotation):
            evaluated = aligned.param.at(
                topology.query,
                on=effective.on,
                opts=effective.rotation_opts,
                validate=False,
            )
        else:
            evaluated = aligned.param.at(
                topology.query,
                on=effective.on,
                opts=effective.position_opts,
                validate=False,
            )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{owner}: {exc}") from exc
    return _restore_output_topology(evaluated, topology, owner=owner)


def _broadcast_static(
    projection: _ProviderBatchProjection,
    topology: OutputTopology,
    *,
    owner: str,
) -> object:
    return _restore_output_topology(projection.value, topology, owner=owner)


def complete_path_query(
    values: Sequence[object],
    *,
    query: object | None,
    caller: object | None,
    temporal: PoseTemporalOptions,
    owner: str,
) -> CompletePathQuery:
    """Classify and evaluate one complete transform-provider set."""
    providers = tuple(RequiredProvider(value, classify_provider_topology(value)) for value in values)
    has_dynamic = any(item.topology == "dynamic" for item in providers)
    has_exact = any(item.topology == "exact" for item in providers)
    if has_exact and (has_dynamic or query is not None):
        raise ValueError(f"{owner}: exact providers cannot be combined with dynamic providers or an explicit query.")
    if has_dynamic and caller is None and query is None:
        raise ValueError(f"{owner}: dynamic direct path solving requires explicit query=.")
    temporal_mode = has_dynamic or (query is not None and not has_exact)
    if not temporal_mode:
        return CompletePathQuery(providers, tuple(values), None)
    contexts = tuple(
        _provider_context(item.value, temporal, owner=owner)
        for item in providers
        if item.topology == "dynamic"
    )
    topology = (
        _object_topology(caller, temporal, contexts, values, owner=owner)
        if caller is not None
        else _direct_topology(query, values, contexts, temporal, owner=owner)
    )
    evaluated = tuple(
        _evaluate_dynamic(projected, topology, temporal, owner=owner)
        if item.topology == "dynamic"
        else _broadcast_static(projected, topology, owner=owner)
        for item, projected in zip(
            providers,
            topology.provider_values,
            strict=True,
        )
    )
    return CompletePathQuery(providers, evaluated, topology)


__all__ = [
    "CompletePathQuery",
    "OutputTopology",
    "RequiredProvider",
    "complete_path_query",
    "require_path_temporal_options",
]
