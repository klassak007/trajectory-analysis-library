from __future__ import annotations

import json
from collections.abc import Mapping

import xarray as xr

from .types import (
    ComponentComposeOptions,
    ComponentExtractOptions,
    ComponentPatchOptions,
    ComponentRegistryOptions,
    ComponentSpec,
)

_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))
COMPONENTS_SCHEMA_VERSION = 1


def coerce_component_registry_options(opts: object | None, *, owner: str) -> ComponentRegistryOptions:
    if opts is None:
        raise TypeError(f"{owner}: opts must be ComponentRegistryOptions.")
    if not isinstance(opts, ComponentRegistryOptions):
        raise TypeError(f"{owner}: opts must be ComponentRegistryOptions.")
    return opts


def coerce_component_extract_options(opts: object | None, *, owner: str) -> ComponentExtractOptions:
    if opts is None:
        return ComponentExtractOptions()
    if not isinstance(opts, ComponentExtractOptions):
        raise TypeError(f"{owner}: opts must be ComponentExtractOptions or None.")
    validate_component_extract_options(opts, owner=owner)
    return opts


def coerce_component_patch_options(opts: object | None, *, owner: str) -> ComponentPatchOptions:
    if opts is None:
        raise TypeError(f"{owner}: opts must be ComponentPatchOptions.")
    if not isinstance(opts, ComponentPatchOptions):
        raise TypeError(f"{owner}: opts must be ComponentPatchOptions.")
    validate_component_patch_options(opts, owner=owner)
    return opts


def coerce_component_compose_options(opts: object | None, *, owner: str) -> ComponentComposeOptions:
    if opts is None:
        raise TypeError(f"{owner}: opts must be ComponentComposeOptions.")
    if not isinstance(opts, ComponentComposeOptions):
        raise TypeError(f"{owner}: opts must be ComponentComposeOptions.")
    validate_component_compose_options(opts, owner=owner)
    return opts


def require_supported_components_version(
    version: object,
    *,
    owner: str,
    path: str = "tal.ext.components.version",
) -> int:
    if isinstance(version, bool) or not isinstance(version, int) or version != COMPONENTS_SCHEMA_VERSION:
        raise ValueError(
            f"{owner}: {path} must be integer {COMPONENTS_SCHEMA_VERSION}; got {version!r}."
        )
    return version


def _require_nonempty_name(value: object, *, field: str, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner}: {field} must be a non-empty string.")
    return value


def _require_optional_output_var(output_var: object, *, owner: str) -> str | None:
    if output_var is None:
        return None
    return _require_nonempty_name(output_var, field="opts.output_var", owner=owner)


def _validate_extract_names(names: object, *, owner: str) -> tuple[str, ...] | None:
    if names is None:
        return None
    if not isinstance(names, tuple):
        raise TypeError(f"{owner}: opts.names must be a tuple[str, ...] when provided.")
    out: list[str] = []
    seen: set[str] = set()
    for idx, raw in enumerate(names):
        name = _require_nonempty_name(raw, field=f"opts.names[{idx}]", owner=owner)
        if name in seen:
            raise ValueError(f"{owner}: opts.names must contain unique component names.")
        seen.add(name)
        out.append(name)
    return tuple(out)


def validate_component_extract_options(
    opts: ComponentExtractOptions,
    *,
    owner: str,
) -> None:
    _validate_extract_names(opts.names, owner=owner)
    _require_optional_output_var(opts.output_var, owner=owner)


def validate_component_patch_options(
    opts: ComponentPatchOptions,
    *,
    owner: str,
) -> None:
    if opts.on_overlap not in {"error", "replace"}:
        raise ValueError(f"{owner}: opts.on_overlap must be 'error' or 'replace'.")
    _require_optional_output_var(opts.output_var, owner=owner)


def _validate_compose_registry(
    registry: object,
    *,
    owner: str,
) -> None:
    if not isinstance(registry, Mapping):
        raise TypeError(f"{owner}: opts.registry must be a mapping[str, ComponentSpec].")
    if not registry:
        raise ValueError(f"{owner}: opts.registry must not be empty.")
    names: set[str] = set()
    for raw_name, raw_spec in registry.items():
        name = _require_nonempty_name(raw_name, field="component name", owner=owner)
        if name in names:
            raise ValueError(f"{owner}: duplicate component name {name!r}.")
        names.add(name)
        spec = _require_component_spec(raw_spec, name=name, owner=owner)
        _require_nonempty_name(spec.core_dim, field=f"registry[{name!r}].core_dim", owner=owner)
        _validate_component_labels(spec.labels, name=name, owner=owner)
        if spec.var is not None:
            _require_nonempty_name(spec.var, field=f"registry[{name!r}].var", owner=owner)


def validate_component_compose_options(
    opts: ComponentComposeOptions,
    *,
    owner: str,
) -> None:
    _validate_compose_registry(opts.registry, owner=owner)
    _require_optional_output_var(opts.output_var, owner=owner)


def _require_component_spec(value: object, *, name: str, owner: str) -> ComponentSpec:
    if not isinstance(value, ComponentSpec):
        raise TypeError(f"{owner}: registry[{name!r}] must be ComponentSpec.")
    return value


def _require_json_scalar(label: object, *, name: str, index: int, owner: str) -> object:
    if not isinstance(label, _JSON_SCALAR_TYPES):
        raise ValueError(
            f"{owner}: registry[{name!r}].labels[{index}] must be a JSON scalar "
            "(str/int/float/bool/null)."
        )
    try:
        json.dumps(label, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{owner}: registry[{name!r}].labels[{index}] must be JSON-serializable without NaN/Inf."
        ) from exc
    return label


def _validate_component_labels(labels: object, *, name: str, owner: str) -> tuple[object, ...]:
    if not isinstance(labels, tuple) or not labels:
        raise ValueError(f"{owner}: registry[{name!r}].labels must be a non-empty tuple.")
    out: list[object] = []
    seen: set[object] = set()
    for idx, raw in enumerate(labels):
        label = _require_json_scalar(raw, name=name, index=idx, owner=owner)
        if label in seen:
            raise ValueError(f"{owner}: registry[{name!r}].labels must be unique.")
        seen.add(label)
        out.append(label)
    return tuple(out)


def _require_component_var_exists(
    var_name: str | None,
    *,
    ds: xr.Dataset,
    name: str,
    owner: str,
) -> None:
    if var_name is None:
        return
    _require_nonempty_name(var_name, field=f"registry[{name!r}].var", owner=owner)
    if var_name not in ds.data_vars:
        raise ValueError(f"{owner}: registry[{name!r}].var {var_name!r} is not a dataset variable.")


def _require_component_core_dim(
    core_dim: object,
    *,
    core_dims: tuple[str, ...],
    ds: xr.Dataset,
    name: str,
    owner: str,
) -> str:
    dim = _require_nonempty_name(core_dim, field=f"registry[{name!r}].core_dim", owner=owner)
    if dim not in core_dims:
        raise ValueError(f"{owner}: registry[{name!r}].core_dim {dim!r} is not in declared core_dims {core_dims!r}.")
    if dim not in ds.dims:
        raise ValueError(f"{owner}: registry[{name!r}].core_dim {dim!r} is not a dataset dimension.")
    if dim not in ds.coords:
        raise ValueError(f"{owner}: registry[{name!r}] requires explicit coordinate labels for dim {dim!r}.")
    return dim


def _require_labels_present_on_dim(
    labels: tuple[object, ...],
    *,
    ds: xr.Dataset,
    core_dim: str,
    name: str,
    owner: str,
) -> None:
    index = ds.get_index(core_dim)
    if index.has_duplicates:
        raise ValueError(f"{owner}: registry[{name!r}] requires unique coordinate labels on dim {core_dim!r}.")
    available = set(index.tolist())
    missing = [label for label in labels if label not in available]
    if missing:
        raise ValueError(
            f"{owner}: registry[{name!r}] labels {missing!r} are not present on core dim {core_dim!r}."
        )


def _require_disjoint_labels_by_core_dim(
    *,
    name: str,
    core_dim: str,
    labels: tuple[object, ...],
    used: dict[str, set[object]],
    owner: str,
) -> None:
    seen = used.setdefault(core_dim, set())
    overlap = [label for label in labels if label in seen]
    if overlap:
        raise ValueError(
            f"{owner}: registry[{name!r}] overlaps existing component labels on dim {core_dim!r}: {overlap!r}."
        )
    seen.update(labels)


def validate_component_registry_options(
    opts: ComponentRegistryOptions,
    *,
    ds: xr.Dataset,
    sequence_dim: str | None,
    core_dims: tuple[str, ...],
    owner: str,
) -> None:
    if sequence_dim is not None:
        _require_nonempty_name(sequence_dim, field="sequence_dim", owner=owner)
    if not isinstance(opts.registry, Mapping):
        raise TypeError(f"{owner}: opts.registry must be a mapping[str, ComponentSpec].")
    if not isinstance(opts.replace, bool):
        raise ValueError(f"{owner}: opts.replace must be a bool.")

    names: set[str] = set()
    used_by_dim: dict[str, set[object]] = {}
    for raw_name, raw_spec in opts.registry.items():
        name = _require_nonempty_name(raw_name, field="component name", owner=owner)
        if name in names:
            raise ValueError(f"{owner}: duplicate component name {name!r}.")
        names.add(name)
        spec = _require_component_spec(raw_spec, name=name, owner=owner)
        core_dim = _require_component_core_dim(spec.core_dim, core_dims=core_dims, ds=ds, name=name, owner=owner)
        labels = _validate_component_labels(spec.labels, name=name, owner=owner)
        _require_labels_present_on_dim(labels, ds=ds, core_dim=core_dim, name=name, owner=owner)
        _require_component_var_exists(spec.var, ds=ds, name=name, owner=owner)
        _require_disjoint_labels_by_core_dim(
            name=name,
            core_dim=core_dim,
            labels=labels,
            used=used_by_dim,
            owner=owner,
        )


__all__ = [
    "COMPONENTS_SCHEMA_VERSION",
    "coerce_component_compose_options",
    "coerce_component_extract_options",
    "coerce_component_patch_options",
    "coerce_component_registry_options",
    "require_supported_components_version",
    "validate_component_compose_options",
    "validate_component_extract_options",
    "validate_component_patch_options",
    "validate_component_registry_options",
]
