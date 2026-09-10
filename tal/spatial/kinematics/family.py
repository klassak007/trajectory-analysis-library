from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.component_ops import ComponentExtractOptions, extract_components
from tal.core.component_ops.runtime_checks import require_component_numeric_var
from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.alignment_intent import select_topology_policy_with_intents
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.orchestration.runtime_checks import (
    require_declared_roles_with_sequence,
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_single_core_dim_with_length,
    select_single_numeric_var,
)
from tal.core.orchestration.topology import (
    SEMANTIC_NON_CORE_POLICY,
    STRICT_NON_CORE_POLICY,
    TopologyPolicy,
)
from tal.core.schema_read import validate_schema_if_needed
from tal.utils.frame_schema import get_frames, set_frames
from tal.utils.topology_operation_families import (
    operation_intent_support_for_operation_family,
)

from ..association import finalize_spatial_as, finalize_spatial_from_source
from ..construction import (
    SpatialConstructionOverrides,
    apply_spatial_construction,
    finish_spatial_factory_promotion,
    preflight_spatial_construction,
    prepare_spatial_construction,
    prepare_spatial_factory_dataset,
)
from ..metadata import normalize_kinematic_relation_semantics
from .paired_components import (
    PairAssemblyOptions,
    align_paired_component_payloads,
    build_paired_components_dataset,
    clear_component_registry,
    component_var_names,
    resolve_component_spec,
    resolve_pair_registry,
    resolve_paired_roles,
)
from .vector6_ops import (
    Vector6FamilyOptions,
    enforce_vector6_layout_invariants,
    pack_linear_angular_to_vector6_dataset,
    unpack_vector6_to_linear_angular_datasets,
)

SetRepFn = Callable[[xr.Dataset, str, bool, str], xr.Dataset]
GetRepFn = Callable[[xr.Dataset, str], str]
SetKindFn = Callable[[xr.Dataset, str, bool, str], xr.Dataset]
NormalizeKindFn = Callable[[xr.Dataset, str, bool, str], xr.Dataset]
ValidateRolesFn = Callable[[xr.Dataset, str], None]
ExtractLinearFn = Callable[[xr.Dataset, str], tuple[xr.Dataset, xr.Dataset]]


@dataclass(frozen=True)
class KinematicsFamilyConfig:
    family_name: str
    family_kind: str
    linear_kind: str
    angular_kind: str
    linear_class_name: str
    angular_class_name: str
    family_class_name: str
    xyz_labels: tuple[str, str, str]
    set_linear_rep: SetRepFn
    set_angular_rep: SetRepFn
    set_family_rep: SetRepFn
    get_linear_rep: GetRepFn
    get_angular_rep: GetRepFn
    get_family_rep: GetRepFn
    set_kinematics_kind: SetKindFn
    normalize_kinematics_kind: NormalizeKindFn
    validate_spatial_roles: ValidateRolesFn
    vector6_opts: Vector6FamilyOptions
    pair_opts: PairAssemblyOptions


@dataclass(frozen=True)
class KinematicsClasses:
    linear_cls: type
    angular_cls: type
    family_cls: type


def coerce_source(value: object, *, owner: str) -> AnalysisObject:
    return coerce_analysis_object_input(value, owner=owner)


def coerce_typed_operand(value: object, *, expected_cls: type, owner: str, label: str) -> Any:
    if isinstance(value, expected_cls):
        return value
    try:
        return expected_cls(value)
    except TypeError as exc:
        raise TypeError(
            f"{owner}: {label} operand must be {expected_cls.__name__}, AnalysisObject, xr.Dataset, or xr.DataArray."
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{owner}: {label} operand is not a valid {expected_cls.__name__}: {exc}") from exc


def normalize_typed_metadata(ds: xr.Dataset, *, rep_getter: GetRepFn, rep_setter: SetRepFn, expected_kind: str, cfg: KinematicsFamilyConfig, owner: str) -> xr.Dataset:
    rep = rep_getter(ds, owner)
    out = rep_setter(ds, rep, False, owner)
    out = cfg.normalize_kinematics_kind(out, expected_kind, False, owner)
    return normalize_kinematic_relation_semantics(
        out,
        validate=False,
        owner=owner,
    )


def _enforce_xyz_cart_invariant(ds: xr.Dataset, *, owner: str, what: str, rep_getter: GetRepFn) -> None:
    select_single_numeric_var(ds, owner=owner, what=what)
    core_dim = require_single_core_dim_with_length(ds, expected_length=3, owner=owner, what=what)
    labels = require_explicit_unique_dim_labels(ds, dim=core_dim, owner=owner, what=what)
    require_exact_labels(labels, expected=("x", "y", "z"), owner=owner, what=f"{what} core")
    rep = rep_getter(ds, owner)
    if rep != "cart":
        raise ValueError(f"{owner}: {what} supports only cart representation; got {rep!r}.")


def _resolve_family_component_specs(
    ds: xr.Dataset,
    *,
    core_dims: tuple[str, ...],
    cfg: KinematicsFamilyConfig,
    owner: str,
) -> tuple[str, str, str, str]:
    registry = resolve_pair_registry(
        ds,
        owner=owner,
        pair_what=cfg.family_class_name,
        left_component_name=cfg.pair_opts.left_component_name,
        right_component_name=cfg.pair_opts.right_component_name,
    )
    left_name = cfg.pair_opts.left_component_name
    right_name = cfg.pair_opts.right_component_name
    left_dim, left_var = resolve_component_spec(
        registry[left_name],
        component_name=left_name,
        core_dims=core_dims,
        expected_labels=cfg.pair_opts.left_expected_labels,
        owner=owner,
        pair_what=cfg.family_class_name,
    )
    right_dim, right_var = resolve_component_spec(
        registry[right_name],
        component_name=right_name,
        core_dims=core_dims,
        expected_labels=cfg.pair_opts.right_expected_labels,
        owner=owner,
        pair_what=cfg.family_class_name,
    )
    if left_dim == right_dim:
        raise ValueError(
            f"{owner}: {cfg.family_class_name} {left_name}/{right_name} components must use distinct core dims."
        )
    return left_dim, left_var, right_dim, right_var


def _validate_family_component_axis_labels(
    ds: xr.Dataset,
    *,
    left_dim: str,
    right_dim: str,
    cfg: KinematicsFamilyConfig,
    owner: str,
) -> None:
    left_name = cfg.pair_opts.left_component_name
    right_name = cfg.pair_opts.right_component_name
    left_labels = require_explicit_unique_dim_labels(ds, dim=left_dim, owner=owner, what=f"{cfg.family_class_name} {left_name}")
    right_labels = require_explicit_unique_dim_labels(ds, dim=right_dim, owner=owner, what=f"{cfg.family_class_name} {right_name}")
    require_exact_labels(
        left_labels,
        expected=cfg.pair_opts.left_expected_labels,
        owner=owner,
        what=f"{cfg.family_class_name} {left_name} core dim {left_dim!r}",
    )
    require_exact_labels(
        right_labels,
        expected=cfg.pair_opts.right_expected_labels,
        owner=owner,
        what=f"{cfg.family_class_name} {right_name} core dim {right_dim!r}",
    )


def enforce_linear_invariants(ds: xr.Dataset, *, owner: str, cfg: KinematicsFamilyConfig) -> None:
    candidate = validate_schema_if_needed(ds)
    cfg.validate_spatial_roles(candidate, owner=owner)
    _ = get_frames(candidate)
    _enforce_xyz_cart_invariant(
        candidate,
        owner=owner,
        what=cfg.linear_class_name,
        rep_getter=cfg.get_linear_rep,
    )


def enforce_angular_invariants(ds: xr.Dataset, *, owner: str, cfg: KinematicsFamilyConfig) -> None:
    candidate = validate_schema_if_needed(ds)
    cfg.validate_spatial_roles(candidate, owner=owner)
    _ = get_frames(candidate)
    _enforce_xyz_cart_invariant(
        candidate,
        owner=owner,
        what=cfg.angular_class_name,
        rep_getter=cfg.get_angular_rep,
    )


def _enforce_components_layout_invariants(ds: xr.Dataset, *, cfg: KinematicsFamilyConfig, owner: str) -> None:
    sequence_dim, batch_dims, core_dims = require_declared_roles_with_sequence(ds, owner=owner, what=cfg.family_class_name)
    if len(core_dims) != 2:
        raise ValueError(
            f"{owner}: {cfg.family_class_name} components layout requires exactly two core dims; got {core_dims!r}."
        )
    left_name = cfg.pair_opts.left_component_name
    right_name = cfg.pair_opts.right_component_name
    left_dim, left_var, right_dim, right_var = _resolve_family_component_specs(
        ds,
        core_dims=core_dims,
        cfg=cfg,
        owner=owner,
    )
    required_non_core_dims = (sequence_dim, *batch_dims)
    require_component_numeric_var(
        ds,
        component_name=left_name,
        var_name=left_var,
        required_dims=required_non_core_dims + (left_dim,),
        owner=owner,
        operand=cfg.family_class_name,
    )
    require_component_numeric_var(
        ds,
        component_name=right_name,
        var_name=right_var,
        required_dims=required_non_core_dims + (right_dim,),
        owner=owner,
        operand=cfg.family_class_name,
    )
    _validate_family_component_axis_labels(ds, left_dim=left_dim, right_dim=right_dim, cfg=cfg, owner=owner)


def enforce_family_invariants(ds: xr.Dataset, *, cfg: KinematicsFamilyConfig, owner: str) -> None:
    candidate = validate_schema_if_needed(ds)
    cfg.validate_spatial_roles(candidate, owner=owner)
    _ = get_frames(candidate)
    rep = cfg.get_family_rep(candidate, owner)
    if rep == "components":
        _enforce_components_layout_invariants(candidate, cfg=cfg, owner=owner)
        return
    if rep == "vector6":
        enforce_vector6_layout_invariants(candidate, owner=owner, opts=cfg.vector6_opts)
        return
    raise ValueError(f"{owner}: unsupported {cfg.family_name} representation {rep!r}.")


def build_family_dataset(
    linear_ds: xr.Dataset,
    angular_ds: xr.Dataset,
    *,
    cfg: KinematicsFamilyConfig,
    owner: str,
    validate: bool,
    policy: TopologyPolicy,
) -> xr.Dataset:
    seq_linear, batch_linear, left_dim, right_dim, left_var, right_var, aligned_linear_ds, aligned_angular_ds = (
        _resolve_aligned_family_component_payloads(
            linear_ds,
            angular_ds,
            cfg=cfg,
            owner=owner,
            policy=policy,
        )
    )
    return build_paired_components_dataset(
        left_ds=aligned_linear_ds,
        right_ds=aligned_angular_ds,
        sequence_dim=seq_linear,
        batch_dims=batch_linear,
        left_dim=left_dim,
        right_dim=right_dim,
        left_var=left_var,
        right_var=right_var,
        owner=owner,
        validate=validate,
        opts=cfg.pair_opts,
        policy=policy,
    )


def _resolve_aligned_family_component_payloads(
    linear_ds: xr.Dataset,
    angular_ds: xr.Dataset,
    *,
    cfg: KinematicsFamilyConfig,
    owner: str,
    policy: TopologyPolicy,
) -> tuple[str | None, tuple[str, ...], str, str, str, str, xr.Dataset, xr.Dataset]:
    left_dim, right_dim = resolve_paired_roles(
        linear_ds,
        angular_ds,
        owner=owner,
        left_what=cfg.pair_opts.left_what,
        right_what=cfg.pair_opts.right_what,
    )
    left_var, right_var = component_var_names(
        linear_ds,
        angular_ds,
        owner=owner,
        opts=cfg.pair_opts,
    )
    aligned_linear_ds, aligned_angular_ds, sequence_dim, batch_dims = align_paired_component_payloads(
        linear_ds,
        angular_ds,
        left_var=left_var,
        right_var=right_var,
        left_dim=left_dim,
        right_dim=right_dim,
        owner=owner,
        opts=cfg.pair_opts,
        policy=policy,
    )
    return (
        sequence_dim,
        batch_dims,
        left_dim,
        right_dim,
        left_var,
        right_var,
        aligned_linear_ds,
        aligned_angular_ds,
    )


def family_from_linear_angular(
    linear: object,
    angular: object,
    *,
    cfg: KinematicsFamilyConfig,
    classes: KinematicsClasses,
    owner: str,
    validate: bool,
    overrides: SpatialConstructionOverrides | None = None,
):
    linear_value = coerce_typed_operand(linear, expected_cls=classes.linear_cls, owner=owner, label="linear")
    angular_value = coerce_typed_operand(angular, expected_cls=classes.angular_cls, owner=owner, label="angular")
    construction = overrides or preflight_spatial_construction(owner=owner)
    plan = prepare_spatial_construction(
        (linear_value, angular_value), overrides=construction, owner=owner
    )
    linear_ds = analysis_object_dataset(linear_value)
    angular_ds = analysis_object_dataset(angular_value)
    selection = select_topology_policy_with_intents(
        (linear_value, angular_value),
        owner=owner,
        operation_family=f"spatial.{cfg.family_name}.components",
        support=operation_intent_support_for_operation_family(
            f"spatial.{cfg.family_name}.components",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    ds = build_family_dataset(
        linear_ds,
        angular_ds,
        cfg=cfg,
        owner=owner,
        validate=validate,
        policy=selection.policy,
    )
    ds = normalize_typed_metadata(ds, rep_getter=cfg.get_family_rep, rep_setter=cfg.set_family_rep, expected_kind=cfg.family_kind, cfg=cfg, owner=owner)
    ds = apply_spatial_construction(ds, plan=plan, owner=owner)
    return finalize_spatial_as(
        classes.family_cls, ds, validate=validate, association=plan.association
    )


def family_from_vector6(
    data: object,
    *,
    cfg: KinematicsFamilyConfig,
    classes: KinematicsClasses,
    owner: str,
    validate: bool,
    overrides: SpatialConstructionOverrides | None = None,
):
    source_ao = coerce_source(data, owner=owner)
    construction = overrides or preflight_spatial_construction(owner=owner)
    plan = prepare_spatial_construction(
        (source_ao,), overrides=construction, owner=owner
    )
    source = prepare_spatial_factory_dataset(source_ao, owner=owner)
    ds = cfg.set_family_rep(source, "vector6", False, owner)
    ds = cfg.set_kinematics_kind(ds, cfg.family_kind, False, owner)
    ds = apply_spatial_construction(ds, plan=plan, owner=owner)
    result = finalize_spatial_as(
        classes.family_cls, ds, validate=validate, association=plan.association
    )
    return finish_spatial_factory_promotion(source_ao, result)


def family_to_rep(value, rep: Literal["components", "vector6"], *, cfg: KinematicsFamilyConfig, classes: KinematicsClasses, owner: str, validate: bool):
    source = analysis_object_dataset(value)
    current_rep = cfg.get_family_rep(source, owner)
    target_rep = cfg.get_family_rep(cfg.set_family_rep(source, rep, False, owner), owner)
    if target_rep == current_rep:
        return finalize_spatial_from_source(value, classes.family_cls, source, validate=validate)
    if target_rep == "components":
        return family_as_components(value, cfg=cfg, classes=classes, owner=owner, validate=validate)
    return family_as_vector6(value, cfg=cfg, classes=classes, owner=owner, validate=validate)


def family_as_components(value, *, cfg: KinematicsFamilyConfig, classes: KinematicsClasses, owner: str, validate: bool):
    source = analysis_object_dataset(value)
    rep = cfg.get_family_rep(source, owner)
    if rep == "components":
        return finalize_spatial_from_source(value, classes.family_cls, source, validate=validate)
    linear_ds, angular_ds = unpack_vector6_to_linear_angular_datasets(
        source,
        owner=owner,
        opts=cfg.vector6_opts,
        set_linear_rep=lambda ds, _validate, _owner: cfg.set_linear_rep(ds, "cart", False, _owner),
        set_angular_rep=lambda ds, _validate, _owner: cfg.set_angular_rep(ds, "cart", False, _owner),
    )
    linear = finalize_spatial_from_source(value, classes.linear_cls, linear_ds, validate=False)
    angular = finalize_spatial_from_source(value, classes.angular_cls, angular_ds, validate=False)
    out = family_from_linear_angular(
        linear,
        angular,
        cfg=cfg,
        classes=classes,
        owner=f"{owner}.from_linear_angular",
        validate=False,
    )
    return finalize_spatial_from_source(
        value, classes.family_cls, analysis_object_dataset(out), validate=validate
    )


def family_as_vector6(value, *, cfg: KinematicsFamilyConfig, classes: KinematicsClasses, owner: str, validate: bool):
    source = analysis_object_dataset(value)
    rep = cfg.get_family_rep(source, owner)
    if rep == "vector6":
        return finalize_spatial_from_source(value, classes.family_cls, source, validate=validate)
    components = finalize_spatial_from_source(value, classes.family_cls, source, validate=False)
    linear = family_linear(components, cfg=cfg, classes=classes, owner=f"{owner}.linear", validate=False)
    angular = family_angular(components, cfg=cfg, classes=classes, owner=f"{owner}.angular", validate=False)
    selection = select_topology_policy_with_intents(
        (linear, angular),
        owner=owner,
        operation_family=f"spatial.{cfg.family_name}.vector6",
        support=operation_intent_support_for_operation_family(
            f"spatial.{cfg.family_name}.vector6",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    policy = selection.policy
    out_ds = pack_linear_angular_to_vector6_dataset(
        analysis_object_dataset(linear),
        analysis_object_dataset(angular),
        owner=owner,
        opts=cfg.vector6_opts,
        set_spatial_rep=lambda ds, _validate, _owner: cfg.set_family_rep(ds, "vector6", False, _owner),
        policy=policy,
    )
    return finalize_spatial_from_source(value, classes.family_cls, out_ds, validate=validate)


def _finalize_component_extract(ds: xr.Dataset, *, rep_setter: SetRepFn, kind: str, cfg: KinematicsFamilyConfig, owner: str) -> xr.Dataset:
    out = clear_component_registry(ds, owner=owner)
    out = rep_setter(out, "cart", False, owner)
    return cfg.set_kinematics_kind(out, kind, False, owner)


def family_linear(value, *, cfg: KinematicsFamilyConfig, classes: KinematicsClasses, owner: str, validate: bool):
    value_ds = analysis_object_dataset(value)
    rep = cfg.get_family_rep(value_ds, owner)
    if rep == "vector6":
        linear_ds, _ = unpack_vector6_to_linear_angular_datasets(
            value_ds,
            owner=owner,
            opts=cfg.vector6_opts,
            set_linear_rep=lambda ds, _validate, _owner: cfg.set_linear_rep(ds, "cart", False, _owner),
            set_angular_rep=lambda ds, _validate, _owner: cfg.set_angular_rep(ds, "cart", False, _owner),
        )
        return finalize_spatial_from_source(value, classes.linear_cls, linear_ds, validate=validate)
    source = AnalysisObject._from_validated(value_ds) if validate else AnalysisObject._from_unvalidated(value_ds)
    extracted = extract_components(source, opts=ComponentExtractOptions(names=(cfg.pair_opts.left_component_name,)), validate=validate)
    ds = _finalize_component_extract(
        analysis_object_dataset(extracted[cfg.pair_opts.left_component_name]),
        rep_setter=cfg.set_linear_rep,
        kind=cfg.linear_kind,
        cfg=cfg,
        owner=owner,
    )
    parent, child = get_frames(value_ds)
    ds = set_frames(ds, parent=parent, child=child, validate=False)
    return finalize_spatial_from_source(value, classes.linear_cls, ds, validate=validate)


def family_angular(value, *, cfg: KinematicsFamilyConfig, classes: KinematicsClasses, owner: str, validate: bool):
    value_ds = analysis_object_dataset(value)
    rep = cfg.get_family_rep(value_ds, owner)
    if rep == "vector6":
        _, angular_ds = unpack_vector6_to_linear_angular_datasets(
            value_ds,
            owner=owner,
            opts=cfg.vector6_opts,
            set_linear_rep=lambda ds, _validate, _owner: cfg.set_linear_rep(ds, "cart", False, _owner),
            set_angular_rep=lambda ds, _validate, _owner: cfg.set_angular_rep(ds, "cart", False, _owner),
        )
        return finalize_spatial_from_source(value, classes.angular_cls, angular_ds, validate=validate)
    source = AnalysisObject._from_validated(value_ds) if validate else AnalysisObject._from_unvalidated(value_ds)
    extracted = extract_components(source, opts=ComponentExtractOptions(names=(cfg.pair_opts.right_component_name,)), validate=validate)
    ds = _finalize_component_extract(
        analysis_object_dataset(extracted[cfg.pair_opts.right_component_name]),
        rep_setter=cfg.set_angular_rep,
        kind=cfg.angular_kind,
        cfg=cfg,
        owner=owner,
    )
    parent, child = get_frames(value_ds)
    ds = set_frames(ds, parent=parent, child=child, validate=False)
    return finalize_spatial_from_source(value, classes.angular_cls, ds, validate=validate)


__all__ = [
    "KinematicsClasses",
    "KinematicsFamilyConfig",
    "coerce_source",
    "enforce_angular_invariants",
    "enforce_family_invariants",
    "enforce_linear_invariants",
    "family_angular",
    "family_as_components",
    "family_as_vector6",
    "family_from_linear_angular",
    "family_from_vector6",
    "family_linear",
    "family_to_rep",
    "normalize_typed_metadata",
]
