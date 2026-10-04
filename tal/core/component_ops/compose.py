from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import xarray as xr

from ..analysis_object import AnalysisObject
from ..dataset_ownership import analysis_object_dataset
from ..combine_ops import CoreConcatOptions, MergeOptions, concat_core, merge
from ..orchestration.inputs import coerce_analysis_object_input
from ..schema import merge_schema
from .options import coerce_component_compose_options
from .registry import define_components
from .runtime_checks import (
    require_declared_roles_with_sequence,
    require_exact_label_set,
    require_explicit_unique_labels,
    select_component_var,
)
from .types import ComponentComposeOptions, ComponentRegistryOptions, ComponentSpec


@dataclass(frozen=True)
class _ComposeEntry:
    name: str
    spec: ComponentSpec
    ao: AnalysisObject
    target_var: str
    sequence_dim: str
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]


def _normalize_component_mapping(
    components: Mapping[str, object],
    *,
    owner: str,
) -> tuple[tuple[str, object], ...]:
    if not isinstance(components, Mapping):
        raise TypeError(f"{owner}: components must be a mapping[str, AO-like].")
    items = tuple(components.items())
    if not items:
        raise ValueError(f"{owner}: components mapping must not be empty.")
    seen: set[str] = set()
    for raw_name, _ in items:
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"{owner}: component names must be non-empty strings.")
        if raw_name in seen:
            raise ValueError(f"{owner}: duplicate component name {raw_name!r} in components mapping.")
        seen.add(raw_name)
    return items


def _require_exact_name_set(
    names: tuple[str, ...],
    *,
    registry: Mapping[str, ComponentSpec],
    owner: str,
) -> None:
    registry_names = set(registry.keys())
    provided = set(names)
    missing = [name for name in registry if name not in provided]
    extra = [name for name in names if name not in registry_names]
    if not missing and not extra:
        return
    parts: list[str] = []
    if missing:
        parts.append(f"missing={missing!r}")
    if extra:
        parts.append(f"extra={extra!r}")
    raise ValueError(f"{owner}: component names must exactly match opts.registry keys ({', '.join(parts)}).")


def _single_var_component_input(
    source: AnalysisObject,
    *,
    data: xr.DataArray,
    var_name: str,
) -> AnalysisObject:
    ds = data.to_dataset(name=var_name)
    tal = analysis_object_dataset(source).attrs.get("tal")
    if isinstance(tal, Mapping):
        ds = merge_schema(ds, patch=dict(tal), validate=False)
    ds = merge_schema(ds, patch={"ext": {"components": None}}, validate=False)
    return AnalysisObject._from_unvalidated(ds)


def _prepare_entry(
    name: str,
    raw_component: object,
    *,
    spec: ComponentSpec,
    owner: str,
) -> _ComposeEntry:
    source = coerce_analysis_object_input(raw_component, owner=owner)
    source_ds = analysis_object_dataset(source)
    sequence_dim, batch_dims, core_dims = require_declared_roles_with_sequence(
        source_ds,
        owner=owner,
        operand=name,
    )
    var_name = select_component_var(source_ds, spec=spec, component_name=name, owner=owner)
    data = source_ds[var_name]
    labels = require_explicit_unique_labels(data, dim=spec.core_dim, owner=owner, operand=f"component {name!r}")
    require_exact_label_set(labels, expected=spec.labels, owner=owner, component_name=name)
    selected = data.sel({spec.core_dim: list(spec.labels)})
    target_var = spec.var if spec.var is not None else var_name
    if target_var != var_name:
        selected = selected.rename(target_var)
    prepared = _single_var_component_input(source, data=selected, var_name=target_var)
    return _ComposeEntry(
        name=name,
        spec=spec,
        ao=prepared,
        target_var=target_var,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
    )


def _prepare_entries(
    components: tuple[tuple[str, object], ...],
    *,
    registry: Mapping[str, ComponentSpec],
    owner: str,
) -> tuple[_ComposeEntry, ...]:
    by_name = dict(components)
    ordered_names = tuple(str(name) for name in registry.keys())
    _require_exact_name_set(tuple(by_name.keys()), registry=registry, owner=owner)
    out: list[_ComposeEntry] = []
    for name in ordered_names:
        out.append(_prepare_entry(name, by_name[name], spec=registry[name], owner=owner))
    return tuple(out)


def _require_shared_roles(entries: tuple[_ComposeEntry, ...], *, owner: str) -> None:
    first = entries[0]
    expected = (first.sequence_dim, first.batch_dims, first.core_dims)
    for entry in entries[1:]:
        actual = (entry.sequence_dim, entry.batch_dims, entry.core_dims)
        if actual == expected:
            continue
        raise ValueError(f"{owner}: all component payload roles must match exactly; got {actual!r} vs {expected!r}.")


def _group_entries(entries: tuple[_ComposeEntry, ...]) -> list[tuple[tuple[str, str], list[_ComposeEntry]]]:
    groups: dict[tuple[str, str], list[_ComposeEntry]] = {}
    order: list[tuple[str, str]] = []
    for entry in entries:
        key = (entry.target_var, entry.spec.core_dim)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(entry)
    return [(key, groups[key]) for key in order]


def _compose_group(
    *,
    key: tuple[str, str],
    entries: list[_ComposeEntry],
    validate: bool,
) -> AnalysisObject:
    target_var, core_dim = key
    labels = tuple(label for entry in entries for label in entry.spec.labels)
    return concat_core(
        [entry.ao for entry in entries],
        opts=CoreConcatOptions(core_dim=core_dim, core_labels=labels, output_var=target_var),
        validate=validate,
    )


def _merge_group_outputs(outputs: list[AnalysisObject], *, validate: bool) -> AnalysisObject:
    if len(outputs) == 1:
        return outputs[0]
    return merge(
        outputs,
        opts=MergeOptions(
            batch_join="exact",
            sequence_join="exact",
            compat="no_conflicts",
            combine_attrs="override",
        ),
        validate=validate,
    )


def _apply_output_var_policy(
    ao: AnalysisObject,
    *,
    output_var: str | None,
    owner: str,
    validate: bool,
) -> AnalysisObject:
    if output_var is None:
        return ao
    ds = analysis_object_dataset(ao)
    if len(ds.data_vars) != 1:
        raise ValueError(f"{owner}: opts.output_var requires composed output to have exactly one data variable.")
    current = str(next(iter(ds.data_vars)))
    if current == output_var:
        return ao
    return ao.rename({current: output_var}, validate=validate)


def compose_components(
    components: Mapping[str, object],
    *,
    opts: ComponentComposeOptions,
    validate: bool = True,
) -> AnalysisObject:
    """Compose named component payloads into one AO output.

    Parameters
    ----------
    components : Mapping[str, object]
        Operand/component input consumed by this operation.
    opts : ComponentComposeOptions, optional
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
    >>> from tal.core.component_ops import ComponentComposeOptions, ComponentSpec, compose_components
    >>> x = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0]])}, coords={"sample": [0], "axis": ["x"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> y = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[2.0]])}, coords={"sample": [0], "axis": ["y"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> opts = ComponentComposeOptions({"x": ComponentSpec("axis", ("x",)), "y": ComponentSpec("axis", ("y",))})
    >>> out = compose_components({"x": x, "y": y}, opts=opts)
    >>> out.as_dataset()["vec"].sel(axis="y").item()
    2.0
    """
    owner = "components.compose"
    options = coerce_component_compose_options(opts, owner=owner)
    items = _normalize_component_mapping(components, owner=owner)
    entries = _prepare_entries(items, registry=options.registry, owner=owner)
    _require_shared_roles(entries, owner=owner)
    groups = _group_entries(entries)
    composed_groups = [_compose_group(key=key, entries=group, validate=validate) for key, group in groups]
    composed = _merge_group_outputs(composed_groups, validate=validate)
    with_registry = define_components(
        composed,
        opts=ComponentRegistryOptions(registry=dict(options.registry), replace=True),
        validate=validate,
    )
    return _apply_output_var_policy(
        with_registry,
        output_var=options.output_var,
        owner=owner,
        validate=validate,
    )


__all__ = ["compose_components"]
