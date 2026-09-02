from __future__ import annotations

from collections.abc import Mapping

import xarray as xr

from ..analysis_object import AnalysisObject
from ..dataset_ownership import analysis_object_dataset
from ..orchestration.finalize import finalize_like, rewrap_unvalidated_like
from ..orchestration.inputs import coerce_analysis_object_input
from ..schema import merge_schema, validate_schema
from ..schema_read import read_roles
from .options import (
    COMPONENTS_SCHEMA_VERSION,
    coerce_component_registry_options,
    require_supported_components_version,
    validate_component_registry_options,
)
from .types import ComponentRegistryOptions, ComponentSpec

_COMPONENTS_NAMESPACE = "components"


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
    block = ext.get(_COMPONENTS_NAMESPACE)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}: tal.ext.components must be a mapping.")
    return block


def _require_declared_roles(ds: xr.Dataset, *, owner: str) -> tuple[str | None, tuple[str, ...]]:
    try:
        roles_declared, sequence_dim, _, core_dims = read_roles(ds)
    except ValueError as exc:
        raise ValueError(f"{owner}: unable to read declared roles: {exc}") from exc
    if not roles_declared:
        raise ValueError(f"{owner}: declared roles are required.")
    return sequence_dim, core_dims


def _decode_component_spec_payload(
    payload: object,
    *,
    name: str,
    owner: str,
) -> ComponentSpec:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{owner}: tal.ext.components.registry[{name!r}] must be a mapping.")
    unknown = [key for key in payload if key not in {"core_dim", "labels", "var"}]
    if unknown:
        raise ValueError(f"{owner}: unsupported keys in tal.ext.components.registry[{name!r}]: {unknown!r}.")
    if "core_dim" not in payload or "labels" not in payload:
        raise ValueError(f"{owner}: tal.ext.components.registry[{name!r}] requires core_dim and labels.")
    labels = payload["labels"]
    if not isinstance(labels, list):
        raise ValueError(f"{owner}: tal.ext.components.registry[{name!r}].labels must be a list.")
    var = payload.get("var")
    if var is not None and not isinstance(var, str):
        raise ValueError(f"{owner}: tal.ext.components.registry[{name!r}].var must be a string when provided.")
    return ComponentSpec(core_dim=payload["core_dim"], labels=tuple(labels), var=var)


def _decode_registry_payload(
    block: Mapping[str, object],
    *,
    owner: str,
) -> dict[str, ComponentSpec]:
    require_supported_components_version(
        block.get("version"),
        owner=owner,
        path="tal.ext.components.version",
    )
    registry = block.get("registry")
    if not isinstance(registry, Mapping):
        raise ValueError(f"{owner}: tal.ext.components.registry must be a mapping.")
    out: dict[str, ComponentSpec] = {}
    for raw_name, raw_payload in registry.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"{owner}: component names must be non-empty strings.")
        if raw_name in out:
            raise ValueError(f"{owner}: duplicate component name {raw_name!r}.")
        out[raw_name] = _decode_component_spec_payload(raw_payload, name=raw_name, owner=owner)
    return out


def _validate_registry_against_dataset(
    registry: Mapping[str, ComponentSpec],
    *,
    ds: xr.Dataset,
    owner: str,
) -> None:
    sequence_dim, core_dims = _require_declared_roles(ds, owner=owner)
    validate_component_registry_options(
        ComponentRegistryOptions(registry=registry, replace=True),
        ds=ds,
        sequence_dim=sequence_dim,
        core_dims=core_dims,
        owner=owner,
    )


def _read_registry_from_dataset(ds: xr.Dataset, *, owner: str) -> dict[str, ComponentSpec]:
    block = _read_components_block(ds, owner=owner)
    if block is None:
        return {}
    decoded = _decode_registry_payload(block, owner=owner)
    _validate_registry_against_dataset(decoded, ds=ds, owner=owner)
    return decoded


def _encode_registry_payload(registry: Mapping[str, ComponentSpec]) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": COMPONENTS_SCHEMA_VERSION,
        "registry": {},
    }
    entries = payload["registry"]
    assert isinstance(entries, dict)
    for name, spec in registry.items():
        entry: dict[str, object] = {
            "core_dim": spec.core_dim,
            "labels": list(spec.labels),
        }
        if spec.var is not None:
            entry["var"] = spec.var
        entries[name] = entry
    return payload


def _component_registry_patch(registry: Mapping[str, ComponentSpec]) -> Mapping[str, object]:
    if not registry:
        return {"ext": {_COMPONENTS_NAMESPACE: None}}
    return {"ext": {_COMPONENTS_NAMESPACE: _encode_registry_payload(registry)}}


def _apply_registry_patch(ds: xr.Dataset, *, registry: Mapping[str, ComponentSpec]) -> xr.Dataset:
    cleared = merge_schema(ds, patch={"ext": {_COMPONENTS_NAMESPACE: None}}, validate=False)
    if not registry:
        return cleared
    return merge_schema(cleared, patch=_component_registry_patch(registry), validate=False)


def _coerce_source_ao(value: object, *, owner: str) -> AnalysisObject:
    return coerce_analysis_object_input(value, owner=owner)


def _rewrap_component_result(
    source: AnalysisObject,
    ds: xr.Dataset,
    *,
    validate: bool,
) -> AnalysisObject:
    if validate:
        validated = validate_schema(ds)
        return finalize_like(source, validated, validate=True, owner="components.define")
    return rewrap_unvalidated_like(source, ds, owner="components.define")


def _merge_registry(
    existing: Mapping[str, ComponentSpec],
    additions: Mapping[str, ComponentSpec],
    *,
    owner: str,
) -> dict[str, ComponentSpec]:
    merged = dict(existing)
    for name, spec in additions.items():
        if name in merged:
            raise ValueError(f"{owner}: component name {name!r} already exists with opts.replace=False.")
        merged[name] = spec
    return merged


def define_components(
    ao: object,
    *,
    opts: ComponentRegistryOptions,
    validate: bool = True,
) -> AnalysisObject:
    """Define or update component registry metadata on an AO.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    opts : ComponentRegistryOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec, define_components, read_components
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> tagged = define_components(ao, opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
    >>> read_components(tagged)["xy"].labels
    ('x', 'y')
    """
    owner = "components.define"
    source = _coerce_source_ao(ao, owner=owner)
    options = coerce_component_registry_options(opts, owner=owner)
    source_ds = analysis_object_dataset(source)
    sequence_dim, core_dims = _require_declared_roles(source_ds, owner=owner)
    validate_component_registry_options(
        options,
        ds=source_ds,
        sequence_dim=sequence_dim,
        core_dims=core_dims,
        owner=owner,
    )

    if options.replace:
        target = dict(options.registry)
    else:
        existing = _read_registry_from_dataset(source_ds, owner=owner)
        target = _merge_registry(existing, options.registry, owner=owner)
        validate_component_registry_options(
            ComponentRegistryOptions(registry=target, replace=True),
            ds=source_ds,
            sequence_dim=sequence_dim,
            core_dims=core_dims,
            owner=owner,
        )

    ds_out = _apply_registry_patch(source_ds, registry=target)
    return _rewrap_component_result(source, ds_out, validate=validate)


def read_components(ao: object) -> Mapping[str, ComponentSpec]:
    """Read component registry metadata from an AO.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.

    Returns
    -------
    Mapping[str, ComponentSpec]
        Mapping-like result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec, read_components
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... ).components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
    >>> sorted(read_components(ao))
    ['xy']
    """
    owner = "components.read"
    source = _coerce_source_ao(ao, owner=owner)
    return dict(_read_registry_from_dataset(analysis_object_dataset(source), owner=owner))


__all__ = [
    "define_components",
    "read_components",
]
