"""Shared complete and selected Dataset ingress owners."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr

from .ao_internal import finalize_structural
from .component_ops.registry import _read_registry_from_dataset
from .component_ops.rewrite import plan_component_update
from .dataset_ownership import (
    analysis_object_dataset,
    couple_dataset_resource,
    isolate_external_dataset,
    metadata_isolated_dataset,
)
from .schema import (
    UNSET,
    _is_bootstrap_schema,
    _SchemaUpdatePlan,
    _validate_existing_schema_envelope,
)
from .schema_errors import SchemaError
from .schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from .schema_update import (
    commit_ingress_target,
    prepare_ingress_target,
    source_schema_view,
)
from .schema_validate import prepare_schema_validation
from .schema_validate.finalize import finalize_validated_schema

if TYPE_CHECKING:
    from .analysis_object import AnalysisObject
    from .component_ops.types import ComponentSpec


@dataclass(frozen=True)
class SelectedOutputPlan:
    """Immutable names and topology for one ordered structural projection."""

    variables: tuple[str, ...]
    dimensions: tuple[str, ...]
    coordinates: tuple[Hashable, ...]
    indexes: tuple[tuple[Hashable, ...], ...]


def _require_validate(value: object, *, owner: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{owner}: validate must be a bool.")


def _optional_declared_name(value: object, *, field: str, owner: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise TypeError(f"{owner}: {field} must be None or a non-empty string.")


def _role_sequence(value: object, *, field: str, owner: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{owner}: {field} must be a sequence of dimension names.")
    names = tuple(value)
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError(f"{owner}: {field} must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{owner}: {field} contains duplicate dimensions.")
    return names


def _selection_names(value: object, *, owner: str) -> tuple[str, ...]:
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = tuple(value)
    else:
        raise TypeError(f"{owner}: names must be a non-empty string or ordered sequence of names.")
    if any(type(name) is not str for name in names):
        raise TypeError(f"{owner}: every selected name must be a string.")
    if not names or any(not name for name in names):
        raise ValueError(f"{owner}: select one or more non-empty data-variable names.")
    if len(set(names)) != len(names):
        raise ValueError(f"{owner}: duplicate data-variable names are not allowed.")
    return names


def _require_membership(ds: xr.Dataset, names: tuple[str, ...], *, owner: str) -> None:
    for name in names:
        if name not in ds.data_vars:
            kind = "coordinate" if name in ds.coords else "dimension" if name in ds.dims else "unknown"
            raise ValueError(f"{owner}: {name!r} is not a data variable ({kind}).")


def _preflight_source(ds: xr.Dataset, *, owner: str) -> Mapping[str, ComponentSpec]:
    if "tal" in ds.attrs:
        payload = ds.attrs.get("tal")
        if isinstance(payload, Mapping) and not _is_bootstrap_schema(payload):
            prepare_schema_validation(ds)
        else:
            _validate_existing_schema_envelope(ds)
    return _read_registry_from_dataset(ds, owner=owner)


def _validated_result(ds: xr.Dataset, *, validate: bool) -> xr.Dataset:
    prepared = prepare_schema_validation(ds)
    return finalize_validated_schema(ds, prepared.tal) if validate else ds


def _bind_external(cls: type[AnalysisObject], ds: xr.Dataset, *, validate: bool) -> AnalysisObject:
    owned = isolate_external_dataset(ds)
    if validate:
        return cls._from_validated(owned)
    return cls._from_unvalidated(owned, schema_prepared=True)


def complete_ingress(
    cls: type[AnalysisObject],
    data: xr.Dataset | xr.DataArray,
    *,
    plan: _SchemaUpdatePlan,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    """Apply either complete-target or overlay core semantics before one copy."""
    _require_validate(validate, owner=owner)
    source = cls._normalized_ingress_dataset(data)
    registry = _preflight_source(source, owner=owner)
    target = prepare_ingress_target(source, source, plan)
    component = plan_component_update(target, owner=owner, validated_registry=registry)
    updated = commit_ingress_target(source, source, plan, component_update=component)
    return _bind_external(cls, _validated_result(updated, validate=validate), validate=validate)


def _overlay_plan(
    *,
    sequence_dim: str | None,
    batch_dims: Sequence[str],
    core_dims: Sequence[str],
    param_coord: str | None,
    sequence_size_coord: str | None,
    layout: str,
    owner: str,
) -> _SchemaUpdatePlan:
    for field, value in (
        ("sequence_dim", sequence_dim), ("param_coord", param_coord),
        ("sequence_size_coord", sequence_size_coord),
    ):
        _optional_declared_name(value, field=field, owner=owner)
    batch = _role_sequence(batch_dims, field="batch_dims", owner=owner)
    core = _role_sequence(core_dims, field="core_dims", owner=owner)
    if layout != "left_packed":
        raise ValueError(f"{owner}: layout must be 'left_packed'.")
    names = batch + core + ((sequence_dim,) if sequence_dim is not None else ())
    if len(set(names)) != len(names):
        raise ValueError(f"{owner}: sequence, batch, and core dimensions must be disjoint.")
    if param_coord is not None and param_coord == sequence_size_coord:
        raise ValueError(f"{owner}: parameter and size coordinates must be distinct.")
    required = [name for name, value in (
        ("param_coord", param_coord), ("sequence_size_coord", sequence_size_coord)
    ) if value is not None]
    if sequence_dim is None and required:
        raise ValueError(
            "from_data requires sequence_dim when schema-bearing arguments are provided. "
            f"Missing sequence_dim with: {', '.join(required)}."
        )
    roles_declared = sequence_dim is not None or bool(batch) or bool(core)
    return _SchemaUpdatePlan(
        sequence_dim=sequence_dim if sequence_dim is not None else UNSET,
        batch_dims=batch if roles_declared else UNSET,
        core_dims=core if roles_declared else UNSET,
        param_coord=param_coord if param_coord is not None else UNSET,
        sequence_size_coord=sequence_size_coord if sequence_size_coord is not None else UNSET,
        layout=layout,
    )


def overlay_ingress(
    cls: type[AnalysisObject],
    data: xr.Dataset | xr.DataArray,
    *,
    sequence_dim: str | None,
    batch_dims: Sequence[str],
    core_dims: Sequence[str],
    param_coord: str | None,
    sequence_size_coord: str | None,
    layout: str,
    validate: bool,
) -> AnalysisObject:
    """Retain `from_data`'s value-based overlay semantics through shared ingress."""
    owner = "AnalysisObject.from_data"
    _require_validate(validate, owner=owner)
    plan = _overlay_plan(
        sequence_dim=sequence_dim, batch_dims=batch_dims, core_dims=core_dims,
        param_coord=param_coord, sequence_size_coord=sequence_size_coord,
        layout=layout, owner=owner,
    )
    return complete_ingress(cls, data, plan=plan, validate=validate, owner=owner)


def _semantic_roles(ds: xr.Dataset, plan: _SchemaUpdatePlan | None) -> tuple[str, ...]:
    if plan is None:
        _, sequence, batch, core = read_roles(ds)
    else:
        sequence, batch, core = plan.sequence_dim, plan.batch_dims, plan.core_dims
    return ((sequence,) if isinstance(sequence, str) else ()) + tuple(batch) + tuple(core)


def _semantic_carriers(ds: xr.Dataset, plan: _SchemaUpdatePlan | None) -> tuple[str, ...]:
    if plan is None:
        names = (read_param_coord_name(ds), read_sequence_size_coord_name(ds))
    else:
        names = (plan.param_coord, plan.sequence_size_coord)
    return tuple(name for name in names if isinstance(name, str) and name in ds.coords)


def _has_role_carrier(ds: xr.Dataset, role: str, *, allowed: set[str], semantic: tuple[str, ...]) -> bool:
    if role in ds.coords and ds.coords[role].dims == (role,):
        return True
    if any(role in ds.coords[name].dims and set(ds.coords[name].dims) <= allowed for name in semantic):
        return True
    return any(
        role in variable.dims and set(variable.dims) <= allowed
        for _, group in ds.xindexes.group_by_index()
        for variable in group.values()
    )


def _planned_dimensions(
    ds: xr.Dataset, names: tuple[str, ...], *, plan: _SchemaUpdatePlan | None, owner: str
) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(dim for name in names for dim in ds[name].dims))
    roles = _semantic_roles(ds, plan)
    allowed = set(selected) | set(roles)
    retained = list(selected)
    semantic = _semantic_carriers(ds, plan)
    for role in roles:
        if role in retained:
            continue
        if _has_role_carrier(ds, role, allowed=allowed, semantic=semantic):
            retained.append(role)
        elif plan is not None:
            raise ValueError(f"{owner}: declared role dimension {role!r} has no selected variable or carrier.")
    return tuple(retained)


def _plan_selection(
    ds: xr.Dataset, names: tuple[str, ...], *, plan: _SchemaUpdatePlan | None, owner: str
) -> SelectedOutputPlan:
    dims = _planned_dimensions(ds, names, plan=plan, owner=owner)
    frozen = set(dims)
    coordinates: list[Hashable] = []
    groups: list[tuple[Hashable, ...]] = []
    indexed: set[Hashable] = set()
    for _, group in ds.xindexes.group_by_index():
        group_names = tuple(group)
        group_dims = {dim for var in group.values() for dim in var.dims}
        if group_dims & frozen and not group_dims <= frozen:
            raise ValueError(f"{owner}: native index group {group_names!r} cannot be selected partially.")
        if group_dims <= frozen:
            groups.append(group_names)
            indexed.update(group_names)
    for name, coord in ds.coords.items():
        if name in names:
            raise ValueError(f"{owner}: coordinate {name!r} would reclassify a selected data variable.")
        if set(coord.dims) <= frozen and (name not in ds.xindexes or name in indexed):
            coordinates.append(name)
    return SelectedOutputPlan(names, dims, tuple(coordinates), tuple(groups))


def _project_selected(ds: xr.Dataset, plan: SelectedOutputPlan) -> xr.Dataset:
    variables = {name: ds.data_vars[name].variable for name in plan.variables}
    coordinates = {name: ds.coords[name].variable for name in plan.coordinates}
    indexes: dict[Hashable, xr.Index] = {}
    for group in plan.indexes:
        copied = ds.xindexes[group[0]].copy(deep=False)
        rebuilt = copied.create_variables({name: coordinates[name] for name in group})
        coordinates.update(rebuilt)
        indexes.update({name: copied for name in group})
    result = xr.Dataset(variables, coords=xr.Coordinates(coordinates, indexes=indexes))
    result.encoding = dict(ds.encoding)
    return result


def selected_external_ingress(
    cls: type[AnalysisObject],
    data: xr.Dataset | xr.DataArray,
    *,
    names: object,
    plan: _SchemaUpdatePlan,
    validate: bool,
) -> AnalysisObject:
    owner = "AnalysisLayoutSpec.wrap"
    _require_validate(validate, owner=owner)
    selected = _selection_names(names, owner=owner)
    source = cls._normalized_ingress_dataset(data)
    if isinstance(data, xr.DataArray):
        raise TypeError(f"{owner}: data_vars is only supported for Dataset input.")
    _require_membership(source, selected, owner=owner)
    registry = _preflight_source(source, owner=owner)
    selection = _plan_selection(source, selected, plan=plan, owner=owner)
    projected = _project_selected(source, selection)
    target = prepare_ingress_target(source, projected, plan)
    component = plan_component_update(target, owner=owner, validated_registry=registry)
    updated = commit_ingress_target(source, projected, plan, component_update=component)
    return _bind_external(cls, _validated_result(updated, validate=validate), validate=validate)


def select_owned_variables(source: AnalysisObject, names: object, *, validate: bool) -> AnalysisObject:
    owner = f"{type(source).__name__}.select_vars"
    _require_validate(validate, owner=owner)
    selected = _selection_names(names, owner=owner)
    dataset = analysis_object_dataset(source)
    _require_membership(dataset, selected, owner=owner)
    registry = _preflight_source(dataset, owner=owner)
    selection = _plan_selection(dataset, selected, plan=None, owner=owner)
    projected = source_schema_view(dataset, _project_selected(dataset, selection))
    isolated = metadata_isolated_dataset(projected, owner=owner)
    try:
        result = finalize_structural(
            source, isolated, validate=validate, validated_registry=registry
        )
    except SchemaError:
        raise
    except (TypeError, ValueError) as error:
        raise type(error)(f"{owner}: {error}") from error
    couple_dataset_resource(dataset, analysis_object_dataset(result))
    return result
