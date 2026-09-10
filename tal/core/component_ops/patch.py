from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import xarray as xr

from ..analysis_object import AnalysisObject
from ..combine_ops.overlay_core import overlay_core
from ..combine_ops.types import CoreOverlayOptions
from ..dataset_ownership import analysis_object_dataset
from ..orchestration.finalize import finalize_like
from ..orchestration.inputs import coerce_analysis_object_input
from ..schema import merge_schema
from .options import coerce_component_patch_options
from .registry import read_components
from .runtime_checks import (
    require_declared_roles_with_sequence,
    require_exact_label_set,
    require_explicit_unique_labels,
    select_component_var,
    select_patch_component_var,
)
from .types import ComponentPatchOptions, ComponentSpec


@dataclass(frozen=True)
class _PatchEntry:
    name: str
    spec: ComponentSpec
    patch: AnalysisObject
    base_var: str
    patch_var: str
    labels: tuple[object, ...]


@dataclass(frozen=True)
class _PatchFinalizationPlan:
    source: AnalysisObject
    rewrap_context: object | None


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
    out: list[tuple[str, object]] = []
    for raw_name, raw_value in items:
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"{owner}: component names must be non-empty strings.")
        out.append((raw_name, raw_value))
    return tuple(out)


def _prepare_patch_entries(
    base: AnalysisObject,
    *,
    components: tuple[tuple[str, object], ...],
    registry: Mapping[str, ComponentSpec],
    owner: str,
) -> tuple[_PatchEntry, ...]:
    base_ds = analysis_object_dataset(base)
    seq_dim, batch_dims, core_dims = require_declared_roles_with_sequence(
        base_ds,
        owner=owner,
        operand="base",
    )
    out: list[_PatchEntry] = []
    for name, raw_patch in components:
        if name not in registry:
            raise ValueError(f"{owner}: unknown component name {name!r}.")
        out.append(
            _prepare_patch_entry(
                base_ds,
                name=name,
                raw_patch=raw_patch,
                spec=registry[name],
                base_roles=(seq_dim, batch_dims, core_dims),
                owner=owner,
            )
        )
    return tuple(out)


def _prepare_patch_entry(
    base_ds: xr.Dataset,
    *,
    name: str,
    raw_patch: object,
    spec: ComponentSpec,
    base_roles: tuple[str, tuple[str, ...], tuple[str, ...]],
    owner: str,
) -> _PatchEntry:
    patch = coerce_analysis_object_input(raw_patch, owner=owner)
    patch_ds = analysis_object_dataset(patch)
    patch_roles = require_declared_roles_with_sequence(patch_ds, owner=owner, operand=name)
    base_var = select_component_var(base_ds, spec=spec, component_name=name, owner=owner, operand="base")
    patch_var = select_patch_component_var(
        patch_ds,
        base_var=base_var,
        base_spec=spec,
        patch_spec=read_components(patch).get(name),
        component_name=name,
        owner=owner,
    )
    _require_compatible_patch_roles(
        base_ds,
        patch_ds,
        base_var=base_var,
        patch_var=patch_var,
        base_roles=base_roles,
        patch_roles=patch_roles,
        component_name=name,
        owner=owner,
    )
    labels = _require_patch_labels(
        patch_ds,
        patch_var=patch_var,
        spec=spec,
        component_name=name,
        owner=owner,
    )
    return _PatchEntry(name, spec, patch, base_var, patch_var, labels)


def _require_patch_labels(
    ds: xr.Dataset,
    *,
    patch_var: str,
    spec: ComponentSpec,
    component_name: str,
    owner: str,
) -> tuple[object, ...]:
    labels = require_explicit_unique_labels(
        ds[patch_var],
        dim=spec.core_dim,
        owner=owner,
        operand=f"component {component_name!r} patch",
    )
    require_exact_label_set(
        labels,
        expected=spec.labels,
        owner=owner,
        component_name=component_name,
        kind="patch labels",
    )
    return labels


def _active_core_roles(
    ds: xr.Dataset,
    *,
    var_name: str,
    core_dims: tuple[str, ...],
) -> tuple[str, ...]:
    dims = ds[var_name].dims
    return tuple(dim for dim in core_dims if dim in dims)


def _require_compatible_patch_roles(
    base_ds: xr.Dataset,
    patch_ds: xr.Dataset,
    *,
    base_var: str,
    patch_var: str,
    base_roles: tuple[str, tuple[str, ...], tuple[str, ...]],
    patch_roles: tuple[str, tuple[str, ...], tuple[str, ...]],
    component_name: str,
    owner: str,
) -> None:
    if patch_roles[:2] != base_roles[:2]:
        raise ValueError(
            f"{owner}: component {component_name!r} patch sequence and batch roles "
            "must match the base exactly."
        )
    base_core = _active_core_roles(base_ds, var_name=base_var, core_dims=base_roles[2])
    patch_core = _active_core_roles(patch_ds, var_name=patch_var, core_dims=patch_roles[2])
    if patch_core != base_core:
        raise ValueError(
            f"{owner}: component {component_name!r} selected-variable core roles "
            f"must match the base exactly; expected {base_core!r}, got {patch_core!r}."
        )


def _require_output_var_policy(
    entries: tuple[_PatchEntry, ...],
    *,
    output_var: str | None,
    owner: str,
) -> str | None:
    if output_var is None:
        return None
    vars_touched = {entry.base_var for entry in entries}
    if len(vars_touched) != 1:
        raise ValueError(f"{owner}: opts.output_var requires all patched components to target one base variable.")
    return next(iter(vars_touched))


def _group_entries(entries: tuple[_PatchEntry, ...]) -> list[tuple[tuple[str, str], list[_PatchEntry]]]:
    groups: dict[tuple[str, str], list[_PatchEntry]] = {}
    order: list[tuple[str, str]] = []
    for entry in entries:
        key = (entry.base_var, entry.spec.core_dim)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(entry)
    return [(key, groups[key]) for key in order]


def _strip_components_ext(ds: xr.Dataset) -> xr.Dataset:
    return merge_schema(ds, patch={"ext": {"components": None}}, validate=False)


def _single_var_ao(
    ds: xr.Dataset,
    *,
    var_name: str,
    owner: str,
) -> AnalysisObject:
    source = AnalysisObject._from_unvalidated(ds, schema_prepared=True)
    subset = _strip_components_ext(ds[[var_name]])
    return finalize_like(
        source,
        subset,
        validate=False,
        owner=owner,
    )


def _apply_patch_group(
    ds: xr.Dataset,
    *,
    key: tuple[str, str],
    entries: list[_PatchEntry],
    on_overlap: str,
    owner: str,
) -> xr.Dataset:
    base_var, core_dim = key
    base_input = _single_var_ao(ds, var_name=base_var, owner=owner)
    patches = [
        _single_var_ao(
            analysis_object_dataset(entry.patch),
            var_name=entry.patch_var,
            owner=owner,
        )
        for entry in entries
    ]
    patched = overlay_core(
        base_input,
        patches,
        opts=CoreOverlayOptions(core_dim=core_dim, on_overlap=on_overlap, output_var=base_var),
        validate=False,
    )
    return ds.assign({base_var: analysis_object_dataset(patched)[base_var]})


def _prepare_patch_finalization(
    source: AnalysisObject,
    entries: tuple[_PatchEntry, ...],
    *,
    owner: str,
) -> _PatchFinalizationPlan:
    values = (source, *(entry.patch for entry in entries))
    context = source._prepare_result_rewrap_context(values, owner=owner)
    return _PatchFinalizationPlan(source, context)


def _finalize_patch_output(
    plan: _PatchFinalizationPlan,
    ds: xr.Dataset,
    *,
    output_var: str | None,
    renamed_target: str | None,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    result = finalize_like(plan.source, ds, validate=validate, owner=owner)
    if output_var is not None:
        assert renamed_target is not None
        result = result.rename({renamed_target: output_var}, validate=validate)
    return plan.source._apply_result_rewrap_context(
        result,
        context=plan.rewrap_context,
    )


def patch_components(
    base: object,
    components: Mapping[str, object],
    *,
    opts: ComponentPatchOptions,
    validate: bool = True,
) -> AnalysisObject:
    """Patch component payloads into a base AO using registry semantics.

    Parameters
    ----------
    base : object
        Input dataset/source value processed by this operation.
    components : Mapping[str, object]
        Operand/component input consumed by this operation.
    opts : ComponentPatchOptions, optional
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
    >>> from tal.core.component_ops import ComponentPatchOptions, ComponentRegistryOptions, ComponentSpec
    >>> from tal.core.component_ops import patch_components
    >>> base = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... ).components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
    >>> patch = AnalysisObject.from_data(
    ...     xr.Dataset({"vec": (("sample", "axis"), [[9.0, 8.0]])}, coords={"sample": [0], "axis": ["x", "y"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> out = patch_components(base, {"xy": patch}, opts=ComponentPatchOptions(on_overlap="replace"))
    >>> out.as_dataset()["vec"].sel(axis="y").item()
    8.0
    """
    owner = "components.patch"
    source = coerce_analysis_object_input(base, owner=owner)
    options = coerce_component_patch_options(opts, owner=owner)
    items = _normalize_component_mapping(components, owner=owner)
    registry = read_components(source)
    entries = _prepare_patch_entries(source, components=items, registry=registry, owner=owner)
    renamed_target = _require_output_var_policy(entries, output_var=options.output_var, owner=owner)
    finalization = _prepare_patch_finalization(source, entries, owner=owner)
    out_ds = analysis_object_dataset(source)
    for key, grouped in _group_entries(entries):
        out_ds = _apply_patch_group(
            out_ds,
            key=key,
            entries=grouped,
            on_overlap=options.on_overlap,
            owner=owner,
        )
    return _finalize_patch_output(
        finalization,
        out_ds,
        output_var=options.output_var,
        renamed_target=renamed_target,
        validate=validate,
        owner=owner,
    )


__all__ = ["patch_components"]
