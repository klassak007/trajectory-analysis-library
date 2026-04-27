from __future__ import annotations

import json
from collections.abc import Mapping

import xarray as xr

from .options import COMPONENTS_SCHEMA_VERSION, require_supported_components_version
from ..schema import merge_schema
from ..schema_read import read_roles
from .types import ComponentSpec

_COMPONENTS_NAMESPACE = "components"
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def _read_components_block(ds: xr.Dataset, *, owner: str) -> Mapping[str, object] | None:
    tal = ds.attrs.get("tal")
    if tal is None:
        return None
    if not isinstance(tal, Mapping):
        raise ValueError(f"{owner}: tal schema payload must be a mapping.")
    ext = tal.get("ext")
    if ext is None:
        return None
    if not isinstance(ext, Mapping):
        raise ValueError(f"{owner}: tal.ext must be a mapping.")
    if _COMPONENTS_NAMESPACE not in ext:
        return None
    block = ext.get(_COMPONENTS_NAMESPACE)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}: tal.ext.components must be a mapping.")
    return block


def _is_json_scalar(value: object) -> bool:
    if not isinstance(value, _JSON_SCALAR_TYPES):
        return False
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _decode_entry_or_none(name: object, payload: object) -> tuple[str, ComponentSpec] | None:
    if not isinstance(name, str) or not name or not isinstance(payload, Mapping):
        return None
    core_dim = payload.get("core_dim")
    labels = payload.get("labels")
    var = payload.get("var")
    if not isinstance(core_dim, str) or not core_dim or not isinstance(labels, list) or not labels:
        return None
    if var is not None and (not isinstance(var, str) or not var):
        return None
    out_labels: list[object] = []
    seen: set[object] = set()
    for label in labels:
        if not _is_json_scalar(label) or label in seen:
            return None
        seen.add(label)
        out_labels.append(label)
    return name, ComponentSpec(core_dim=core_dim, labels=tuple(out_labels), var=var)


def _decode_prunable_registry(block: Mapping[str, object], *, owner: str) -> dict[str, ComponentSpec]:
    require_supported_components_version(
        block.get("version"),
        owner=owner,
        path="tal.ext.components.version",
    )
    registry = block.get("registry")
    if not isinstance(registry, Mapping):
        return {}
    out: dict[str, ComponentSpec] = {}
    for name, payload in registry.items():
        decoded = _decode_entry_or_none(name, payload)
        if decoded is None:
            continue
        key, spec = decoded
        if key in out:
            continue
        out[key] = spec
    return out


def _normalize_rename_map(rename_map: Mapping[str, str] | None) -> dict[str, str]:
    if not isinstance(rename_map, Mapping):
        return {}
    out: dict[str, str] = {}
    for old, new in rename_map.items():
        if isinstance(old, str) and isinstance(new, str):
            out[old] = new
    return out


def _rewrite_registry_entries(
    registry: Mapping[str, ComponentSpec],
    *,
    ds: xr.Dataset,
    core_dims: tuple[str, ...],
    rename_map: Mapping[str, str],
) -> dict[str, ComponentSpec]:
    out: dict[str, ComponentSpec] = {}
    used: dict[str, set[object]] = {}
    for name, spec in registry.items():
        dim = rename_map.get(spec.core_dim, spec.core_dim)
        if dim not in core_dims or dim not in ds.coords:
            continue
        var_name = spec.var
        if var_name is not None:
            var_name = rename_map.get(var_name, var_name)
        if var_name is not None and var_name not in ds.data_vars:
            continue
        index = ds.get_index(dim)
        if index.has_duplicates:
            continue
        available = set(index.tolist())
        if any(label not in available for label in spec.labels):
            continue
        seen = used.setdefault(dim, set())
        if any(label in seen for label in spec.labels):
            continue
        seen.update(spec.labels)
        out[name] = ComponentSpec(core_dim=dim, labels=spec.labels, var=var_name)
    return out


def _components_patch(registry: Mapping[str, ComponentSpec]) -> Mapping[str, object]:
    if not registry:
        return {"ext": {_COMPONENTS_NAMESPACE: None}}
    entries: dict[str, object] = {}
    for name, spec in registry.items():
        entry: dict[str, object] = {
            "core_dim": spec.core_dim,
            "labels": list(spec.labels),
        }
        if spec.var is not None:
            entry["var"] = spec.var
        entries[name] = entry
    return {"ext": {_COMPONENTS_NAMESPACE: {"version": COMPONENTS_SCHEMA_VERSION, "registry": entries}}}


def _apply_components_patch(ds: xr.Dataset, *, registry: Mapping[str, ComponentSpec]) -> xr.Dataset:
    cleared = merge_schema(ds, patch={"ext": {_COMPONENTS_NAMESPACE: None}}, validate=False)
    if not registry:
        return cleared
    return merge_schema(cleared, patch=_components_patch(registry), validate=False)


def rewrite_component_registry_after_structure(
    ds: xr.Dataset,
    *,
    rename_map: Mapping[str, str] | None,
    owner: str,
) -> xr.Dataset:
    block = _read_components_block(ds, owner=owner)
    if block is None:
        return ds
    source_registry = _decode_prunable_registry(block, owner=owner)
    if not source_registry:
        return _apply_components_patch(ds, registry={})
    roles_declared, _, _, core_dims = read_roles(ds)
    if not roles_declared:
        return _apply_components_patch(ds, registry={})
    remap = _normalize_rename_map(rename_map)
    rewritten = _rewrite_registry_entries(source_registry, ds=ds, core_dims=core_dims, rename_map=remap)
    return _apply_components_patch(ds, registry=rewritten)


__all__ = ["rewrite_component_registry_after_structure"]
