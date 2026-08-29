from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec, define_components, read_components
from tal.core.orchestration.alignment import align_exact_for_plan
from tal.core.orchestration.context import resolve_semantic_topology_from_dataset
from tal.core.orchestration.topology import (
    STRICT_NON_CORE_POLICY,
    TopologyOperand,
    TopologyPolicy,
    resolve_binary_topology,
)
from tal.core.orchestration.runtime_checks import require_exact_labels
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.core.schema import copy_dataset_attrs


@dataclass(frozen=True)
class PairAssemblyOptions:
    left_what: str
    right_what: str
    pair_what: str
    left_component_name: str
    right_component_name: str
    left_expected_labels: tuple[str, ...]
    right_expected_labels: tuple[str, ...]


def shared_optional_name(
    left: str | None,
    right: str | None,
    *,
    field: str,
    owner: str,
    allow_one_sided_inherit: bool = False,
) -> str | None:
    if left == right:
        return left
    if allow_one_sided_inherit:
        if left is None:
            return right
        if right is None:
            return left
    raise ValueError(f"{owner}: {field} mismatch across inputs ({left!r} vs {right!r}).")


def component_var_names(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
    opts: PairAssemblyOptions,
) -> tuple[str, str]:
    left_var = str(next(iter(left_ds.data_vars)))
    right_var = str(next(iter(right_ds.data_vars)))
    if left_var == right_var:
        raise ValueError(
            f"{owner}: {opts.left_what}/{opts.right_what} variable names must be distinct; rename one operand first."
        )
    return left_var, right_var


def resolve_paired_roles(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
    left_what: str,
    right_what: str,
) -> tuple[str, str]:
    left_declared, _, _, core_left = read_roles(left_ds)
    right_declared, _, _, core_right = read_roles(right_ds)
    if not left_declared:
        raise ValueError(f"{owner}: {left_what} requires declared roles.")
    if not right_declared:
        raise ValueError(f"{owner}: {right_what} requires declared roles.")
    if len(core_left) != 1 or len(core_right) != 1:
        raise ValueError(f"{owner}: {left_what} and {right_what} must each declare exactly one core dim.")
    left_dim = core_left[0]
    right_dim = core_right[0]
    if left_dim == right_dim:
        raise ValueError(f"{owner}: {left_what}/{right_what} core dims must be distinct.")
    return left_dim, right_dim


def resolve_pair_registry(
    ds: xr.Dataset,
    *,
    owner: str,
    pair_what: str,
    left_component_name: str,
    right_component_name: str,
) -> Mapping[str, ComponentSpec]:
    registry = read_components(ds)
    expected = {left_component_name, right_component_name}
    names = set(registry.keys())
    if names != expected:
        raise ValueError(
            f"{owner}: {pair_what} components layout requires registry names exactly {sorted(expected)!r}; "
            f"got {sorted(names)!r}."
        )
    return registry


def resolve_component_spec(
    spec: ComponentSpec,
    *,
    component_name: str,
    core_dims: tuple[str, ...],
    expected_labels: tuple[str, ...],
    owner: str,
    pair_what: str,
) -> tuple[str, str]:
    core_dim = spec.core_dim
    if core_dim not in core_dims:
        raise ValueError(
            f"{owner}: {pair_what} component {component_name!r} core dim {core_dim!r} must be declared in roles.core_dims."
        )
    labels = tuple(spec.labels)
    require_exact_labels(labels, expected=expected_labels, owner=owner, what=f"{pair_what} component {component_name!r}")
    if spec.var is None:
        raise ValueError(f"{owner}: {pair_what} component {component_name!r} must declare spec.var in components layout.")
    return core_dim, spec.var


def merge_component_payloads(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    left_var: str,
    right_var: str,
    owner: str,
) -> xr.Dataset:
    try:
        return xr.merge(
            [left_ds[[left_var]], right_ds[[right_var]]],
            join="exact",
            compat="equals",
            combine_attrs="drop_conflicts",
        )
    except ValueError as exc:
        raise ValueError(f"{owner}: {left_var}/{right_var} coordinates must align exactly: {exc}") from exc


def _component_topology_operand(
    ds: xr.Dataset,
    *,
    var_name: str,
    core_dim: str,
    index: int,
    owner: str,
    what: str,
    policy: TopologyPolicy,
) -> TopologyOperand:
    return TopologyOperand(
        index=index,
        data=ds[var_name],
        semantic=resolve_semantic_topology_from_dataset(
            ds,
            var_name=var_name,
            core_dims=(core_dim,),
            owner=owner,
            what=what,
            allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
            allow_missing_batch_dims=policy.mode == "semantic_broadcast",
        ),
        param_coord=read_param_coord_name(ds),
    )


def _rewrap_aligned_component_dataset(
    aligned: xr.DataArray,
    *,
    source_ds: xr.Dataset,
    var_name: str,
) -> xr.Dataset:
    out = aligned.to_dataset(name=var_name)
    return copy_dataset_attrs(source_ds, out, validate=False)


def align_paired_component_payloads(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    left_var: str,
    right_var: str,
    left_dim: str,
    right_dim: str,
    owner: str,
    opts: PairAssemblyOptions,
    policy: TopologyPolicy | None = None,
) -> tuple[xr.Dataset, xr.Dataset, str | None, tuple[str, ...]]:
    resolved_policy = policy if policy is not None else STRICT_NON_CORE_POLICY
    what = f"{opts.pair_what} component assembly"
    plan = resolve_binary_topology(
        _component_topology_operand(
            left_ds,
            var_name=left_var,
            core_dim=left_dim,
            index=0,
            owner=owner,
            what=opts.left_what,
            policy=resolved_policy,
        ),
        _component_topology_operand(
            right_ds,
            var_name=right_var,
            core_dim=right_dim,
            index=1,
            owner=owner,
            what=opts.right_what,
            policy=resolved_policy,
        ),
        owner=owner,
        what=what,
        policy=resolved_policy,
    )
    left_da, right_da = align_exact_for_plan(
        plan,
        owner=owner,
        what=what,
    )
    return (
        _rewrap_aligned_component_dataset(left_da, source_ds=left_ds, var_name=left_var),
        _rewrap_aligned_component_dataset(right_da, source_ds=right_ds, var_name=right_var),
        plan.sequence_dim,
        plan.batch_dims,
    )


def build_base_analysis_object(
    merged: xr.Dataset,
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, str],
    param_coord: str | None,
    sequence_size_coord: str | None,
    validate: bool,
) -> AnalysisObject:
    param_name = param_coord if sequence_dim is not None else None
    size_name = sequence_size_coord if sequence_dim is not None else None
    kwargs: dict[str, object] = {
        "batch_dims": batch_dims,
        "core_dims": core_dims,
        "param_coord": param_name,
        "sequence_size_coord": size_name,
        "validate": validate,
    }
    if sequence_dim is not None:
        kwargs["sequence_dim"] = sequence_dim
    return AnalysisObject.from_data(
        merged,
        **kwargs,
    )


def resolve_paired_optional_coord_names(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
    allow_one_sided_inherit: bool,
) -> tuple[str | None, str | None]:
    param_name = shared_optional_name(
        read_param_coord_name(left_ds),
        read_param_coord_name(right_ds),
        field="param_coord",
        owner=owner,
        allow_one_sided_inherit=allow_one_sided_inherit,
    )
    size_name = shared_optional_name(
        read_sequence_size_coord_name(left_ds),
        read_sequence_size_coord_name(right_ds),
        field="sequence_size_coord",
        owner=owner,
        allow_one_sided_inherit=allow_one_sided_inherit,
    )
    return param_name, size_name


def build_paired_components_dataset(
    *,
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    left_dim: str,
    right_dim: str,
    left_var: str,
    right_var: str,
    owner: str,
    validate: bool,
    opts: PairAssemblyOptions,
    policy: TopologyPolicy | None = None,
) -> xr.Dataset:
    allow_one_sided_inherit = policy is not None and policy.mode == "semantic_broadcast"
    param_name, size_name = resolve_paired_optional_coord_names(
        left_ds,
        right_ds,
        owner=owner,
        allow_one_sided_inherit=allow_one_sided_inherit,
    )
    merged = merge_component_payloads(
        left_ds,
        right_ds,
        left_var=left_var,
        right_var=right_var,
        owner=owner,
    )
    base = build_base_analysis_object(
        merged,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=(left_dim, right_dim),
        param_coord=param_name,
        sequence_size_coord=size_name,
        validate=validate,
    )
    registry: Mapping[str, ComponentSpec] = {
        opts.left_component_name: ComponentSpec(core_dim=left_dim, labels=opts.left_expected_labels, var=left_var),
        opts.right_component_name: ComponentSpec(core_dim=right_dim, labels=opts.right_expected_labels, var=right_var),
    }
    with_registry = define_components(
        base,
        opts=ComponentRegistryOptions(registry=registry, replace=True),
        validate=validate,
    )
    return with_registry.unsafe_data


def clear_component_registry(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    source = AnalysisObject._from_unvalidated(ds)
    try:
        cleaned = define_components(
            source,
            opts=ComponentRegistryOptions(registry={}, replace=True),
            validate=False,
        )
    except ValueError as exc:
        raise ValueError(f"{owner}: failed to clear component registry metadata: {exc}") from exc
    return cleaned.unsafe_data


__all__ = [
    "PairAssemblyOptions",
    "align_paired_component_payloads",
    "build_paired_components_dataset",
    "clear_component_registry",
    "component_var_names",
    "merge_component_payloads",
    "resolve_component_spec",
    "resolve_paired_optional_coord_names",
    "resolve_pair_registry",
    "resolve_paired_roles",
    "shared_optional_name",
]
