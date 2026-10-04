"""Shared complete and selected Dataset ingress owners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from .schema_update import (
    commit_ingress_target,
    prepare_ingress_target,
    source_schema_view,
)
from .schema_validate import prepare_schema_validation
from .schema_validate.finalize import finalize_validated_schema
from .selected_ingress import (
    commit_prepared_selected_ingress,
    plan_selected_output,
    prepare_selected_ingress,
    project_selected_output,
    require_data_variables,
    selection_names,
)

if TYPE_CHECKING:
    from .analysis_object import AnalysisObject
    from .component_ops.types import ComponentSpec


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
    selected = selection_names(names, owner=owner)
    source = cls._normalized_ingress_dataset(data)
    if isinstance(data, xr.DataArray):
        raise TypeError(f"{owner}: data_vars is only supported for Dataset input.")
    require_data_variables(source, selected, owner=owner)
    registry = _preflight_source(source, owner=owner)
    prepared = prepare_selected_ingress(
        source,
        selected,
        schema_plan=plan,
        validated_registry=registry,
        owner=owner,
    )
    updated = commit_prepared_selected_ingress(prepared)
    return _bind_external(cls, _validated_result(updated, validate=validate), validate=validate)


def select_owned_variables(source: AnalysisObject, names: object, *, validate: bool) -> AnalysisObject:
    owner = f"{type(source).__name__}.select_vars"
    _require_validate(validate, owner=owner)
    selected = selection_names(names, owner=owner)
    dataset = analysis_object_dataset(source)
    require_data_variables(dataset, selected, owner=owner)
    registry = _preflight_source(dataset, owner=owner)
    selection = plan_selected_output(dataset, selected, plan=None, owner=owner)
    projected = source_schema_view(dataset, project_selected_output(dataset, selection))
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
