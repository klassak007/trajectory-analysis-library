"""Copy-neutral schema-update preparation and single-copy commit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import xarray as xr

from .schema import (
    SCHEMA_VERSION,
    UNSET,
    _apply_writer,
    _copy_tal,
    _require_dataset,
    _SchemaUpdatePlan,
)
from .schema_validate import prepare_schema_validation
from .schema_validate.finalize import _replace_dataset_attrs_with_tal


def _canonical_name(value: Any) -> Any:
    return str.__str__(value) if isinstance(value, str) else value


def _canonical_dims(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [_canonical_name(item) for item in value]
    return value


def _update_roles(core: dict[str, Any], plan: _SchemaUpdatePlan) -> None:
    values = (plan.sequence_dim, plan.batch_dims, plan.core_dims)
    if all(value is UNSET for value in values):
        return
    previous = core.get("roles")
    roles = dict(previous) if isinstance(previous, Mapping) else {}
    roles.setdefault("batch_dims", [])
    roles.setdefault("core_dims", [])
    if plan.sequence_dim is not UNSET:
        if plan.sequence_dim is None:
            roles.pop("sequence_dim", None)
        else:
            roles["sequence_dim"] = _canonical_name(plan.sequence_dim)
    if plan.batch_dims is not UNSET:
        roles["batch_dims"] = _canonical_dims(plan.batch_dims)
    if plan.core_dims is not UNSET:
        roles["core_dims"] = _canonical_dims(plan.core_dims)
    core["roles"] = roles


def _update_optional(core: dict[str, Any], plan: _SchemaUpdatePlan) -> None:
    if plan.param_coord is not UNSET:
        if plan.param_coord is None:
            core.pop("param_coord", None)
        else:
            core["param_coord"] = {"name": _canonical_name(plan.param_coord)}
    if plan.sequence_size_coord is not UNSET:
        if plan.sequence_size_coord is None:
            core.pop("validity", None)
        else:
            core["validity"] = {
                "sequence_size_coord": _canonical_name(plan.sequence_size_coord),
                "layout": _canonical_name(plan.layout),
            }


def _updated_core(tal: dict[str, Any], plan: _SchemaUpdatePlan, *, owned: bool) -> None:
    original = tal.get("core")
    core = (
        original if owned and isinstance(original, dict)
        else dict(original) if isinstance(original, Mapping) else {}
    )
    if plan.complete_target:
        core.clear()
    _update_roles(core, plan)
    _update_optional(core, plan)
    if plan.complete_target and plan.sequence_dim is None and not plan.batch_dims and not plan.core_dims:
        core.pop("roles", None)
    tal["core"] = core


def _apply_to_tal(tal: dict[str, Any], plan: _SchemaUpdatePlan, *, owned: bool) -> None:
    tal["version"] = SCHEMA_VERSION
    _updated_core(tal, plan, owned=owned)


def _schema_view(source: xr.Dataset, output: xr.Dataset, tal: Mapping[str, Any]) -> xr.Dataset:
    return _replace_dataset_attrs_with_tal(
        output, ordinary_attrs=source.attrs, tal=tal, isolate_tal=False
    )


def source_schema_view(source: xr.Dataset, output: xr.Dataset) -> xr.Dataset:
    """Attach source schema to a projection without invoking copy protocols."""
    tal = source.attrs.get("tal")
    return _schema_view(source, output, tal if isinstance(tal, Mapping) else {})


def prepare_ingress_target(
    source: xr.Dataset, output: xr.Dataset, plan: _SchemaUpdatePlan
) -> xr.Dataset:
    """Validate an output target without copying the source extension graph."""
    tal = source.attrs.get("tal")
    provisional = dict(tal) if isinstance(tal, Mapping) else {}
    _apply_to_tal(provisional, plan, owned=False)
    candidate = _schema_view(source, output, provisional)
    prepare_schema_validation(candidate)
    return candidate


def _replace_extension_owned(tal: dict[str, Any], *, name: str, value: object | None) -> None:
    ext = tal.get("ext")
    if value is None and not isinstance(ext, dict):
        return
    if not isinstance(ext, dict):
        ext = {}
        tal["ext"] = ext
    if value is None:
        ext.pop(name, None)
    else:
        ext[name] = value


def commit_ingress_target(
    source: xr.Dataset,
    output: xr.Dataset,
    plan: _SchemaUpdatePlan,
    *,
    component_update: object = UNSET,
) -> xr.Dataset:
    """Copy source schema once and commit its planned core/component result."""
    tal = _copy_tal(source)
    _apply_to_tal(tal, plan, owned=True)
    if component_update is not UNSET:
        _replace_extension_owned(tal, name="components", value=component_update)
    return _schema_view(source, output, tal)


def replace_extension_namespace(ds: xr.Dataset, *, name: str, value: object | None) -> xr.Dataset:
    """Replace one extension namespace through the schema mutation owner."""
    tal = _copy_tal(ds)
    _replace_extension_owned(tal, name=name, value=value)
    return _schema_view(ds, ds, tal)


def apply_schema_update(ds: xr.Dataset, plan: _SchemaUpdatePlan, *, validate: bool) -> xr.Dataset:
    """Apply existing public writer policy through the same update rules."""
    candidate = _require_dataset(ds, owner="apply_schema_update")
    tal = _copy_tal(candidate)
    _apply_to_tal(tal, plan, owned=True)
    return _apply_writer(candidate, tal, validate=validate)
