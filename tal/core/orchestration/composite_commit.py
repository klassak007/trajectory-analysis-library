"""Private single-commit boundary for already assembled composite results."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, NoReturn

import xarray as xr

from ..analysis_object import AnalysisObject
from ..component_ops.options import validate_component_registry_options
from ..component_ops.registry import (
    _encode_registry_payload,
    _read_registry_from_dataset,
)
from ..component_ops.types import ComponentRegistryOptions, ComponentSpec
from ..dataset_ownership import analysis_object_dataset
from ..schema import UNSET, _SchemaUpdatePlan
from ..schema_errors import SchemaError, _schema_error_with_context
from ..schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from ..schema_update import commit_ingress_target, prepare_commit_target
from ..schema_validate import prepare_schema_validation, validate_schema_structure
from .schema_finalize import CoreSchemaFinalizeSpec

_ComponentAction = Literal["preserve", "replace", "prune"]
_ResourceAction = Callable[[AnalysisObject], None]


@dataclass(frozen=True)
class _ComponentRegistryCommit:
    """Immutable component-registry intent for one final commit."""

    action: _ComponentAction = "preserve"
    entries: tuple[tuple[str, ComponentSpec], ...] = ()

    def __post_init__(self) -> None:
        if self.action not in {"preserve", "replace", "prune"}:
            raise ValueError(f"component commit action is invalid: {self.action!r}.")
        if self.action != "replace" and self.entries:
            raise ValueError(f"component commit action {self.action!r} cannot carry entries.")
        names = tuple(name for name, _ in self.entries)
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("component commit names must be non-empty strings.")
        if len(names) != len(set(names)):
            raise ValueError("component commit names must be unique.")


@dataclass(frozen=True)
class _CompositeCommitSpec:
    """Narrow declaration for committing one already assembled Dataset."""

    owner: str
    validate: bool
    schema: CoreSchemaFinalizeSpec
    components: _ComponentRegistryCommit
    prototype: AnalysisObject
    result_context: object | None = None
    resource_action: _ResourceAction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.owner, str) or not self.owner:
            raise TypeError("composite commit owner must be a non-empty string.")
        if not isinstance(self.prototype, AnalysisObject):
            raise TypeError("composite commit prototype must be an AnalysisObject.")
        if self.resource_action is not None and not callable(self.resource_action):
            raise TypeError("composite commit resource action must be callable.")
        if self.schema.sequence_dim is None and (
            self.schema.param_name is not None or self.schema.size_name is not None
        ):
            raise ValueError(
                "composite commit parameter and validity declarations require a sequence dimension."
            )


def _schema_update_plan(spec: CoreSchemaFinalizeSpec) -> _SchemaUpdatePlan:
    sequence_dim = spec.sequence_dim
    return _SchemaUpdatePlan(
        sequence_dim=sequence_dim,
        batch_dims=spec.batch_dims,
        core_dims=spec.core_dims,
        param_coord=spec.param_name if sequence_dim is not None else None,
        sequence_size_coord=spec.size_name if sequence_dim is not None else None,
        complete_target=True,
    )


def _registry_mapping(commit: _ComponentRegistryCommit) -> dict[str, ComponentSpec]:
    return dict(commit.entries)


def _validate_planned_registry(
    candidate: xr.Dataset,
    *,
    commit: _ComponentRegistryCommit,
    owner: str,
) -> Mapping[str, ComponentSpec]:
    if commit.action == "preserve":
        return _read_registry_from_dataset(candidate, owner=owner)
    if commit.action == "prune":
        return {}
    registry = _registry_mapping(commit)
    declared, sequence_dim, _, core_dims = read_roles(candidate)
    if not declared:
        raise ValueError(f"{owner}: component replacement requires declared roles.")
    validate_component_registry_options(
        ComponentRegistryOptions(registry=registry, replace=True),
        ds=candidate,
        sequence_dim=sequence_dim,
        core_dims=core_dims,
        owner=owner,
    )
    return registry


def _component_update(
    commit: _ComponentRegistryCommit,
    registry: Mapping[str, ComponentSpec],
) -> object:
    if commit.action == "preserve":
        return UNSET
    if commit.action == "prune":
        return None
    return _encode_registry_payload(registry)


def _verify_schema(ds: xr.Dataset, spec: _CompositeCommitSpec) -> None:
    declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    expected_declared = bool(
        spec.schema.sequence_dim is not None
        or spec.schema.batch_dims
        or spec.schema.core_dims
    )
    actual = (declared, sequence_dim, batch_dims, core_dims)
    expected = (
        expected_declared,
        spec.schema.sequence_dim,
        spec.schema.batch_dims,
        spec.schema.core_dims,
    )
    if actual != expected:
        raise ValueError(f"{spec.owner}: committed core roles changed: expected {expected!r}, got {actual!r}.")
    if read_param_coord_name(ds) != spec.schema.param_name:
        raise ValueError(f"{spec.owner}: committed parameter declaration changed.")
    if read_sequence_size_coord_name(ds) != spec.schema.size_name:
        raise ValueError(f"{spec.owner}: committed validity declaration changed.")


def _verify_result(result: AnalysisObject, spec: _CompositeCommitSpec) -> None:
    ds = analysis_object_dataset(result)
    if spec.validate:
        prepare_schema_validation(ds)
    else:
        validate_schema_structure(ds)
    _verify_schema(ds, spec)
    actual = _read_registry_from_dataset(ds, owner=spec.owner)
    if spec.components.action == "replace" and actual != _registry_mapping(spec.components):
        raise ValueError(f"{spec.owner}: committed component registry changed.")
    if spec.components.action == "prune" and actual:
        raise ValueError(f"{spec.owner}: committed component registry was not pruned.")


def _apply_result_context(result: AnalysisObject, spec: _CompositeCommitSpec) -> AnalysisObject:
    applied = spec.prototype._apply_result_rewrap_context(
        result,
        context=spec.result_context,
    )
    if applied is not result:
        raise ValueError(f"{spec.owner}: result context must not create another wrapper.")
    return result


def _raise_owned_preflight_failure(
    error: Exception,
    *,
    owner: str,
) -> NoReturn:
    if isinstance(error, SchemaError):
        raise _schema_error_with_context(error, context=owner) from error
    if type(error) not in (TypeError, ValueError):
        raise error
    prefix = f"{owner}:"
    if str(error).startswith(prefix):
        raise error
    raise type(error)(f"{prefix} {error}") from error


def _prepare_commit(
    candidate: xr.Dataset,
    spec: _CompositeCommitSpec,
) -> tuple[_SchemaUpdatePlan, Mapping[str, ComponentSpec]]:
    schema_plan = _schema_update_plan(spec.schema)
    try:
        provisional = prepare_commit_target(
            candidate,
            candidate,
            schema_plan,
            validate=spec.validate,
        )
        registry = _validate_planned_registry(
            provisional,
            commit=spec.components,
            owner=spec.owner,
        )
    except Exception as exc:  # noqa: BLE001 -- only TAL-owned preflight is translated
        _raise_owned_preflight_failure(exc, owner=spec.owner)
    return schema_plan, registry


def _commit_result(candidate: xr.Dataset, spec: _CompositeCommitSpec) -> AnalysisObject:
    schema_plan, registry = _prepare_commit(candidate, spec)
    committed = commit_ingress_target(
        candidate,
        candidate,
        schema_plan,
        component_update=_component_update(spec.components, registry),
    )
    result = spec.prototype._rewrap_dataset(
        committed,
        validate=spec.validate,
        schema_prepared=True,
    )
    _verify_result(result, spec)
    result = _apply_result_context(result, spec)
    if spec.resource_action is not None:
        spec.resource_action(result)
    return result


def _commit_composite_result(
    candidate: xr.Dataset,
    *,
    spec: _CompositeCommitSpec,
) -> AnalysisObject:
    """Commit one operation-assembled candidate without planning or assembly."""
    if not isinstance(candidate, xr.Dataset):
        raise TypeError(
            f"{spec.owner}: composite candidate must be xr.Dataset; "
            f"got {type(candidate).__name__}."
        )
    return _commit_result(candidate, spec)


__all__: list[str] = []
