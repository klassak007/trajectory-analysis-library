from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core import AnalysisObject, read_components
from tal.core.orchestration.alignment import align_exact_for_plan, non_core_dims
from tal.core.orchestration.context import resolve_semantic_topology_from_dataset
from tal.core.orchestration.topology import (
    STRICT_NON_CORE_POLICY,
    TopologyPolicy,
    TopologyOperand,
    resolve_binary_topology,
)
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_single_core_dim_with_length,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.schema_read import (
    read_param_coord_name,
    read_sequence_size_coord_name,
    validate_schema_if_needed,
)
from tal.utils.frame_schema import get_frames, set_frames

from ..conversion.finalize import allocate_free_dim_name as _allocate_dim_name
from ..policies.frame import resolve_components_shared_frames
from ..kernels.kinematics_vector6_kernels import (
    pack_vector6_kernel,
    unpack_vector6_angular_kernel,
    unpack_vector6_linear_kernel,
)
from ..metadata import set_kinematics_kind
from .paired_components import clear_component_registry, shared_optional_name
from .paired_components import resolve_paired_optional_coord_names

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")
_VECTOR6_LABELS: tuple[str, str, str, str, str, str] = (
    "linear_x",
    "linear_y",
    "linear_z",
    "angular_x",
    "angular_y",
    "angular_z",
)


@dataclass(frozen=True)
class Vector6FamilyOptions:
    family_what: str
    family_kind: str
    linear_kind: str
    angular_kind: str
    vector_var_name: str
    linear_var_name: str
    angular_var_name: str
    vector_dim_candidates: tuple[str, ...]
    linear_dim_candidates: tuple[str, ...]
    angular_dim_candidates: tuple[str, ...]


VELOCITY_VECTOR6_OPTS = Vector6FamilyOptions(
    family_what="Velocity",
    family_kind="velocity",
    linear_kind="linear_velocity",
    angular_kind="angular_velocity",
    vector_var_name="velocity",
    linear_var_name="linear_velocity",
    angular_var_name="angular_velocity",
    vector_dim_candidates=("velocity_axis", "vector6", "spatial6"),
    linear_dim_candidates=("linear_axis", "linear_xyz", "linear"),
    angular_dim_candidates=("angular_axis", "angular_xyz", "angular"),
)

ACCELERATION_VECTOR6_OPTS = Vector6FamilyOptions(
    family_what="Acceleration",
    family_kind="acceleration",
    linear_kind="linear_acceleration",
    angular_kind="angular_acceleration",
    vector_var_name="acceleration",
    linear_var_name="linear_acceleration",
    angular_var_name="angular_acceleration",
    vector_dim_candidates=("acceleration_axis", "vector6", "spatial6"),
    linear_dim_candidates=("linear_axis", "linear_xyz", "linear"),
    angular_dim_candidates=("angular_axis", "angular_xyz", "angular"),
)

RepWriter = Callable[[xr.Dataset, bool, str], xr.Dataset]


def vector6_labels() -> tuple[str, str, str, str, str, str]:
    return _VECTOR6_LABELS


def _source_optional_coord_names(ds: xr.Dataset) -> tuple[str | None, str | None]:
    return read_param_coord_name(ds), read_sequence_size_coord_name(ds)


def _vector6_var_and_dim(ds: xr.Dataset, *, owner: str, opts: Vector6FamilyOptions) -> tuple[str, str]:
    var_name = select_single_numeric_var(ds, owner=owner, what=opts.family_what)
    core_dim = require_single_core_dim_with_length(ds, expected_length=6, owner=owner, what=opts.family_what)
    require_var_contains_dims(
        ds,
        var_name=var_name,
        required_dims=(core_dim,),
        owner=owner,
        what=opts.family_what,
    )
    labels = require_explicit_unique_dim_labels(ds, dim=core_dim, owner=owner, what=opts.family_what)
    require_exact_labels(labels, expected=_VECTOR6_LABELS, owner=owner, what=f"{opts.family_what} vector6 core")
    return var_name, core_dim


def enforce_vector6_layout_invariants(ds: xr.Dataset, *, owner: str, opts: Vector6FamilyOptions) -> None:
    candidate = validate_schema_if_needed(ds)
    from tal.core.schema_read import read_roles

    declared, _, _, _ = read_roles(candidate)
    if not declared:
        raise ValueError(f"{owner}: {opts.family_what} requires declared roles.")
    _vector6_var_and_dim(candidate, owner=owner, opts=opts)
    try:
        registry = read_components(candidate)
    except ValueError as exc:
        raise ValueError(f"{owner}: invalid component registry state for vector6 layout: {exc}") from exc
    if registry:
        raise ValueError(f"{owner}: {opts.family_what} vector6 layout must not carry component registry entries.")


def _vector3_var_and_dim(ds: xr.Dataset, *, owner: str, what: str) -> tuple[str, str]:
    var_name = select_single_numeric_var(ds, owner=owner, what=what)
    core_dim = require_single_core_dim_with_length(ds, expected_length=3, owner=owner, what=what)
    require_var_contains_dims(
        ds,
        var_name=var_name,
        required_dims=(core_dim,),
        owner=owner,
        what=what,
    )
    labels = require_explicit_unique_dim_labels(ds, dim=core_dim, owner=owner, what=what)
    require_exact_labels(labels, expected=_XYZ_LABELS, owner=owner, what=f"{what} core")
    return var_name, core_dim


def _wrap_pack_kernel(linear: np.ndarray, angular: np.ndarray, *, owner: str) -> np.ndarray:
    try:
        return pack_vector6_kernel(linear, angular)
    except ValueError as exc:
        raise ValueError(f"{owner}: components->vector6 kernel failed.") from exc


def _wrap_unpack_linear_kernel(values: np.ndarray, *, owner: str) -> np.ndarray:
    try:
        return unpack_vector6_linear_kernel(values)
    except ValueError as exc:
        raise ValueError(f"{owner}: vector6->linear kernel failed.") from exc


def _wrap_unpack_angular_kernel(values: np.ndarray, *, owner: str) -> np.ndarray:
    try:
        return unpack_vector6_angular_kernel(values)
    except ValueError as exc:
        raise ValueError(f"{owner}: vector6->angular kernel failed.") from exc


def _build_vector6_base_dataset(
    packed: xr.DataArray,
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    core_dim: str,
    param_coord: str | None,
    sequence_size_coord: str | None,
    owner: str,
) -> xr.Dataset:
    param_name = param_coord if sequence_dim is not None else None
    size_name = sequence_size_coord if sequence_dim is not None else None
    kwargs: dict[str, object] = {
        "batch_dims": batch_dims,
        "core_dims": (core_dim,),
        "param_coord": param_name,
        "sequence_size_coord": size_name,
        "validate": False,
    }
    if sequence_dim is not None:
        kwargs["sequence_dim"] = sequence_dim
    base = AnalysisObject.from_data(
        packed.to_dataset(name=packed.name or "vector6"),
        **kwargs,
    )
    try:
        return clear_component_registry(analysis_object_dataset(base), owner=owner)
    except ValueError as exc:
        raise ValueError(f"{owner}: failed to clear component registry for vector6 layout: {exc}") from exc


def _build_vector3_base_dataset(
    component: xr.DataArray,
    *,
    var_name: str,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    core_dim: str,
    param_coord: str | None,
    sequence_size_coord: str | None,
    owner: str,
) -> xr.Dataset:
    param_name = param_coord if sequence_dim is not None else None
    size_name = sequence_size_coord if sequence_dim is not None else None
    kwargs: dict[str, object] = {
        "batch_dims": batch_dims,
        "core_dims": (core_dim,),
        "param_coord": param_name,
        "sequence_size_coord": size_name,
        "validate": False,
    }
    if sequence_dim is not None:
        kwargs["sequence_dim"] = sequence_dim
    base = AnalysisObject.from_data(
        component.to_dataset(name=var_name),
        **kwargs,
    )
    try:
        return clear_component_registry(analysis_object_dataset(base), owner=owner)
    except ValueError as exc:
        raise ValueError(f"{owner}: failed to clear component registry for vec3 layout: {exc}") from exc


def _pack_aligned_vector6_dataarray(
    linear_ds: xr.Dataset,
    angular_ds: xr.Dataset,
    *,
    owner: str,
    opts: Vector6FamilyOptions,
    policy: TopologyPolicy,
) -> tuple[xr.DataArray, str | None, tuple[str, ...]]:
    linear_da, angular_da, linear_dim, angular_dim, sequence_dim, batch_dims = _resolve_aligned_pack_operands(
        linear_ds,
        angular_ds,
        owner=owner,
        opts=opts,
        policy=policy,
    )
    vector_dim = _allocate_dim_name(
        existing_dims=set(linear_da.dims) | set(angular_da.dims),
        candidates=opts.vector_dim_candidates,
        base=opts.vector_dim_candidates[0],
        owner=owner,
        what=f"{opts.family_what} vector6",
    )
    kernel = partial(_wrap_pack_kernel, owner=owner)
    packed = xr.apply_ufunc(
        kernel,
        linear_da,
        angular_da,
        input_core_dims=[[linear_dim], [angular_dim]],
        output_core_dims=[[vector_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.result_type(linear_da.dtype, angular_da.dtype)],
        dask_gufunc_kwargs={"output_sizes": {vector_dim: 6}},
    )
    return (
        packed.assign_coords({vector_dim: list(_VECTOR6_LABELS)}).rename(opts.vector_var_name),
        sequence_dim,
        batch_dims,
    )


def _resolve_aligned_pack_operands(
    linear_ds: xr.Dataset,
    angular_ds: xr.Dataset,
    *,
    owner: str,
    opts: Vector6FamilyOptions,
    policy: TopologyPolicy,
) -> tuple[xr.DataArray, xr.DataArray, str, str, str | None, tuple[str, ...]]:
    linear_var, linear_dim = _vector3_var_and_dim(linear_ds, owner=owner, what=f"{opts.family_what} linear component")
    angular_var, angular_dim = _vector3_var_and_dim(angular_ds, owner=owner, what=f"{opts.family_what} angular component")
    plan = resolve_binary_topology(
        TopologyOperand(
            index=0,
            data=linear_ds[linear_var],
            semantic=resolve_semantic_topology_from_dataset(
                linear_ds,
                var_name=linear_var,
                core_dims=(linear_dim,),
                owner=owner,
                what=f"{opts.family_what} linear component",
                allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
                allow_missing_batch_dims=policy.mode == "semantic_broadcast",
            ),
            param_coord=read_param_coord_name(linear_ds),
        ),
        TopologyOperand(
            index=1,
            data=angular_ds[angular_var],
            semantic=resolve_semantic_topology_from_dataset(
                angular_ds,
                var_name=angular_var,
                core_dims=(angular_dim,),
                owner=owner,
                what=f"{opts.family_what} angular component",
                allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
                allow_missing_batch_dims=policy.mode == "semantic_broadcast",
            ),
            param_coord=read_param_coord_name(angular_ds),
        ),
        owner=owner,
        what=f"{opts.family_what.lower()} vector6 pack",
        policy=policy,
    )
    linear_da, angular_da = align_exact_for_plan(
        plan,
        owner=owner,
        what=f"{opts.family_what.lower()} vector6 pack",
    )
    return linear_da, angular_da, linear_dim, angular_dim, plan.sequence_dim, plan.batch_dims


def _finalize_vector6_dataset(
    packed: xr.DataArray,
    *,
    linear_ds: xr.Dataset,
    angular_ds: xr.Dataset,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    owner: str,
    opts: Vector6FamilyOptions,
    set_spatial_rep: RepWriter,
    allow_one_sided_inherit: bool,
) -> xr.Dataset:
    vector_dim = str(packed.dims[-1])
    param_name, size_name = resolve_paired_optional_coord_names(
        linear_ds,
        angular_ds,
        owner=owner,
        allow_one_sided_inherit=allow_one_sided_inherit,
    )
    out_ds = _build_vector6_base_dataset(
        packed,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dim=vector_dim,
        param_coord=param_name,
        sequence_size_coord=size_name,
        owner=owner,
    )
    out_ds = set_spatial_rep(out_ds, False, owner)
    out_ds = set_kinematics_kind(out_ds, kind=opts.family_kind, validate=False, owner=owner)
    parent, child = resolve_components_shared_frames(
        linear_ds,
        angular_ds,
        owner=owner,
        left_name="linear",
        right_name="angular",
    )
    return set_frames(out_ds, parent=parent, child=child, validate=False)


def pack_linear_angular_to_vector6_dataset(
    linear_ds: xr.Dataset,
    angular_ds: xr.Dataset,
    *,
    owner: str,
    opts: Vector6FamilyOptions,
    set_spatial_rep: RepWriter,
    policy: TopologyPolicy | None = None,
) -> xr.Dataset:
    linear_ds = validate_schema_if_needed(linear_ds)
    angular_ds = validate_schema_if_needed(angular_ds)
    resolved_policy = policy if policy is not None else STRICT_NON_CORE_POLICY
    packed, sequence_dim, batch_dims = _pack_aligned_vector6_dataarray(
        linear_ds,
        angular_ds,
        owner=owner,
        opts=opts,
        policy=resolved_policy,
    )
    allow_one_sided_inherit = resolved_policy.mode == "semantic_broadcast"
    return _finalize_vector6_dataset(
        packed,
        linear_ds=linear_ds,
        angular_ds=angular_ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        owner=owner,
        opts=opts,
        set_spatial_rep=set_spatial_rep,
        allow_one_sided_inherit=allow_one_sided_inherit,
    )


def _unpack_vector6_component_dataarrays(
    vector_ds: xr.Dataset,
    *,
    owner: str,
    opts: Vector6FamilyOptions,
) -> tuple[xr.DataArray, str, str]:
    vector_var, vector_dim = _vector6_var_and_dim(vector_ds, owner=owner, opts=opts)
    vector_da = vector_ds[vector_var]
    occupied = set(non_core_dims(vector_da, core_dims=(vector_dim,)))
    linear_dim = _allocate_dim_name(
        existing_dims=occupied,
        candidates=opts.linear_dim_candidates,
        base=opts.linear_dim_candidates[0],
        owner=owner,
        what="linear vec3",
    )
    occupied.add(linear_dim)
    angular_dim = _allocate_dim_name(
        existing_dims=occupied,
        candidates=opts.angular_dim_candidates,
        base=opts.angular_dim_candidates[0],
        owner=owner,
        what="angular vec3",
    )
    linear_da = xr.apply_ufunc(
        partial(_wrap_unpack_linear_kernel, owner=owner),
        vector_da,
        input_core_dims=[[vector_dim]],
        output_core_dims=[[linear_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[vector_da.dtype],
        dask_gufunc_kwargs={"output_sizes": {linear_dim: 3}},
    ).assign_coords({linear_dim: list(_XYZ_LABELS)}).rename(opts.linear_var_name)
    angular_da = xr.apply_ufunc(
        partial(_wrap_unpack_angular_kernel, owner=owner),
        vector_da,
        input_core_dims=[[vector_dim]],
        output_core_dims=[[angular_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[vector_da.dtype],
        dask_gufunc_kwargs={"output_sizes": {angular_dim: 3}},
    ).assign_coords({angular_dim: list(_XYZ_LABELS)}).rename(opts.angular_var_name)
    return linear_da, angular_da, vector_dim


def _finalize_unpacked_components(
    linear_da: xr.DataArray,
    angular_da: xr.DataArray,
    vector_ds: xr.Dataset,
    *,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    owner: str,
    opts: Vector6FamilyOptions,
    set_linear_rep: RepWriter,
    set_angular_rep: RepWriter,
) -> tuple[xr.Dataset, xr.Dataset]:
    param_name, size_name = _source_optional_coord_names(vector_ds)
    linear_dim = str(linear_da.dims[-1])
    angular_dim = str(angular_da.dims[-1])
    linear_ds = _build_vector3_base_dataset(
        linear_da,
        var_name=opts.linear_var_name,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dim=linear_dim,
        param_coord=param_name,
        sequence_size_coord=size_name,
        owner=owner,
    )
    angular_ds = _build_vector3_base_dataset(
        angular_da,
        var_name=opts.angular_var_name,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dim=angular_dim,
        param_coord=param_name,
        sequence_size_coord=size_name,
        owner=owner,
    )
    parent, child = get_frames(vector_ds)
    linear_ds = set_kinematics_kind(set_linear_rep(linear_ds, False, owner), kind=opts.linear_kind, validate=False, owner=owner)
    angular_ds = set_kinematics_kind(set_angular_rep(angular_ds, False, owner), kind=opts.angular_kind, validate=False, owner=owner)
    linear_ds = set_frames(linear_ds, parent=parent, child=child, validate=False)
    angular_ds = set_frames(angular_ds, parent=parent, child=child, validate=False)
    return linear_ds, angular_ds


def unpack_vector6_to_linear_angular_datasets(
    vector_ds: xr.Dataset,
    *,
    owner: str,
    opts: Vector6FamilyOptions,
    set_linear_rep: RepWriter,
    set_angular_rep: RepWriter,
) -> tuple[xr.Dataset, xr.Dataset]:
    candidate = validate_schema_if_needed(vector_ds)
    from tal.core.schema_read import read_roles

    declared, sequence_dim, batch_dims, _ = read_roles(candidate)
    if not declared:
        raise ValueError(f"{owner}: {opts.family_what} requires declared roles.")
    linear_da, angular_da, _ = _unpack_vector6_component_dataarrays(candidate, owner=owner, opts=opts)
    return _finalize_unpacked_components(
        linear_da,
        angular_da,
        candidate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        owner=owner,
        opts=opts,
        set_linear_rep=set_linear_rep,
        set_angular_rep=set_angular_rep,
    )


__all__ = [
    "ACCELERATION_VECTOR6_OPTS",
    "VELOCITY_VECTOR6_OPTS",
    "Vector6FamilyOptions",
    "enforce_vector6_layout_invariants",
    "pack_linear_angular_to_vector6_dataset",
    "unpack_vector6_to_linear_angular_datasets",
    "vector6_labels",
]
