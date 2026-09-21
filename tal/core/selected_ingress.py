"""Copy-neutral planning for ordered Dataset data-variable projections."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr

from .component_ops.rewrite import plan_component_update
from .schema import _SchemaUpdatePlan
from .schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from .schema_update import commit_ingress_target, prepare_ingress_target

if TYPE_CHECKING:
    from .component_ops.types import ComponentSpec


@dataclass(frozen=True)
class SelectedOutputPlan:
    """Immutable names and topology for one ordered structural projection."""

    variables: tuple[str, ...]
    dimensions: tuple[str, ...]
    coordinates: tuple[Hashable, ...]
    indexes: tuple[tuple[Hashable, ...], ...]


@dataclass(frozen=True)
class PreparedSelectedIngress:
    """Copy-neutral selected target ready for one schema-owned commit."""

    source: xr.Dataset
    projected: xr.Dataset
    target: xr.Dataset
    schema_plan: _SchemaUpdatePlan
    component_update: object


def selection_names(value: object, *, owner: str) -> tuple[str, ...]:
    """Normalize one ordered public data-variable selection."""
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = tuple(value)
    else:
        raise TypeError(
            f"{owner}: names must be a non-empty string or ordered sequence of names."
        )
    if any(type(name) is not str for name in names):
        raise TypeError(f"{owner}: every selected name must be a string.")
    if not names or any(not name for name in names):
        raise ValueError(f"{owner}: select one or more non-empty data-variable names.")
    if len(set(names)) != len(names):
        raise ValueError(f"{owner}: duplicate data-variable names are not allowed.")
    return names


def require_data_variables(
    ds: xr.Dataset,
    names: tuple[str, ...],
    *,
    owner: str,
) -> None:
    """Require every requested name to identify a data variable."""
    for name in names:
        if name in ds.data_vars:
            continue
        kind = (
            "coordinate"
            if name in ds.coords
            else "dimension"
            if name in ds.dims
            else "unknown"
        )
        raise ValueError(f"{owner}: {name!r} is not a data variable ({kind}).")


def _semantic_roles(
    ds: xr.Dataset,
    plan: _SchemaUpdatePlan | None,
) -> tuple[str, ...]:
    if plan is None:
        _, sequence, batch, core = read_roles(ds)
    else:
        sequence, batch, core = plan.sequence_dim, plan.batch_dims, plan.core_dims
    return ((sequence,) if isinstance(sequence, str) else ()) + tuple(batch) + tuple(core)


def _semantic_carriers(
    ds: xr.Dataset,
    plan: _SchemaUpdatePlan | None,
) -> tuple[str, ...]:
    if plan is None:
        names = (read_param_coord_name(ds), read_sequence_size_coord_name(ds))
    else:
        names = (plan.param_coord, plan.sequence_size_coord)
    return tuple(name for name in names if isinstance(name, str) and name in ds.coords)


def _has_role_carrier(
    ds: xr.Dataset,
    role: str,
    *,
    allowed: set[str],
    semantic: tuple[str, ...],
) -> bool:
    if role in ds.coords and ds.coords[role].dims == (role,):
        return True
    if any(
        role in ds.coords[name].dims and set(ds.coords[name].dims) <= allowed
        for name in semantic
    ):
        return True
    return any(
        role in variable.dims and set(variable.dims) <= allowed
        for _, group in ds.xindexes.group_by_index()
        for variable in group.values()
    )


def _planned_dimensions(
    ds: xr.Dataset,
    names: tuple[str, ...],
    *,
    plan: _SchemaUpdatePlan | None,
    owner: str,
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
            raise ValueError(
                f"{owner}: declared role dimension {role!r} has no selected variable or carrier."
            )
    return tuple(retained)


def plan_selected_output(
    ds: xr.Dataset,
    names: tuple[str, ...],
    *,
    plan: _SchemaUpdatePlan | None,
    owner: str,
) -> SelectedOutputPlan:
    """Plan coordinates and complete public index groups for a projection."""
    dims = _planned_dimensions(ds, names, plan=plan, owner=owner)
    frozen = set(dims)
    coordinates: list[Hashable] = []
    groups: list[tuple[Hashable, ...]] = []
    indexed: set[Hashable] = set()
    for _, group in ds.xindexes.group_by_index():
        group_names = tuple(group)
        group_dims = {dim for var in group.values() for dim in var.dims}
        if group_dims & frozen and not group_dims <= frozen:
            raise ValueError(
                f"{owner}: native index group {group_names!r} cannot be selected partially."
            )
        if group_dims <= frozen:
            groups.append(group_names)
            indexed.update(group_names)
    for name, coord in ds.coords.items():
        if name in names:
            raise ValueError(
                f"{owner}: coordinate {name!r} would reclassify a selected data variable."
            )
        if set(coord.dims) <= frozen and (name not in ds.xindexes or name in indexed):
            coordinates.append(name)
    return SelectedOutputPlan(names, dims, tuple(coordinates), tuple(groups))


def project_selected_output(
    ds: xr.Dataset,
    plan: SelectedOutputPlan,
) -> xr.Dataset:
    """Apply a copy-neutral selected-output plan."""
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


def prepare_selected_ingress(
    source: xr.Dataset,
    names: tuple[str, ...],
    *,
    schema_plan: _SchemaUpdatePlan,
    validated_registry: Mapping[str, ComponentSpec],
    owner: str,
) -> PreparedSelectedIngress:
    """Validate and project one selected ingress without copying metadata."""
    require_data_variables(source, names, owner=owner)
    selection = plan_selected_output(
        source,
        names,
        plan=schema_plan,
        owner=owner,
    )
    projected = project_selected_output(source, selection)
    target = prepare_ingress_target(source, projected, schema_plan)
    component = plan_component_update(
        target,
        owner=owner,
        validated_registry=validated_registry,
    )
    return PreparedSelectedIngress(
        source=source,
        projected=projected,
        target=target,
        schema_plan=schema_plan,
        component_update=component,
    )


def commit_prepared_selected_ingress(prepared: PreparedSelectedIngress) -> xr.Dataset:
    """Commit a previously validated selected-ingress plan exactly once."""
    return commit_ingress_target(
        prepared.source,
        prepared.projected,
        prepared.schema_plan,
        component_update=prepared.component_update,
    )


__all__ = [
    "PreparedSelectedIngress",
    "SelectedOutputPlan",
    "commit_prepared_selected_ingress",
    "plan_selected_output",
    "prepare_selected_ingress",
    "project_selected_output",
    "require_data_variables",
    "selection_names",
]
