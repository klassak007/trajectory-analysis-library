from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.orchestration.alignment import align_exact_for_plan
from tal.core.orchestration.alignment_intent import select_topology_policy_with_intents
from tal.core.orchestration.context import resolve_semantic_topology_from_dataset
from tal.core.orchestration.topology import (
    SEMANTIC_NON_CORE_POLICY,
    STRICT_NON_CORE_POLICY,
    TopologyPolicy,
    TopologyOperand,
    resolve_binary_topology,
)
from tal.core.orchestration.runtime_checks import resolve_single_numeric_var_single_core_dim
from tal.core.orchestration.finalize import transfer_dataset_attrs
from tal.core.schema_read import read_param_coord_name, validate_schema_if_needed
from tal.utils.frame_schema import set_frames
from tal.utils.topology_operation_families import operation_intent_support_for_operation_family

from ..acceleration import Acceleration, AngularAcceleration, LinearAcceleration
from ..policies.frame import resolve_apply_output_frames
from ..kinematics.vector6_ops import (
    ACCELERATION_VECTOR6_OPTS,
    VELOCITY_VECTOR6_OPTS,
    pack_linear_angular_to_vector6_dataset,
    unpack_vector6_to_linear_angular_datasets,
)
from ..metadata import (
    get_acceleration_rep,
    get_velocity_rep,
    set_acceleration_rep,
    set_angular_acceleration_rep,
    set_angular_velocity_rep,
    set_linear_acceleration_rep,
    set_linear_velocity_rep,
    set_velocity_rep,
)
from ..position import Position
from ..kernels.rotation_apply_kernels import rotate_vec3_kernel
from ..policies.wrap import wrap_like
from ..velocity import AngularVelocity, LinearVelocity, Velocity

if TYPE_CHECKING:
    from ..rotation import Rotation

_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")

_VECTOR_TARGET_TYPES = (
    Position,
    LinearVelocity,
    AngularVelocity,
    LinearAcceleration,
    AngularAcceleration,
)

_SPATIAL_TARGET_TYPES = (
    Velocity,
    Acceleration,
)

_SUPPORTED_TARGET_TYPES = _VECTOR_TARGET_TYPES + _SPATIAL_TARGET_TYPES


def _topology_operand(
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


def _set_linear_velocity_cart_rep(ds: xr.Dataset, validate: bool, owner: str) -> xr.Dataset:
    return set_linear_velocity_rep(ds, rep="cart", validate=validate, owner=owner)


def _set_angular_velocity_cart_rep(ds: xr.Dataset, validate: bool, owner: str) -> xr.Dataset:
    return set_angular_velocity_rep(ds, rep="cart", validate=validate, owner=owner)


def _set_velocity_vector6_rep(ds: xr.Dataset, validate: bool, owner: str) -> xr.Dataset:
    return set_velocity_rep(ds, rep="vector6", validate=validate, owner=owner)


def _set_linear_acceleration_cart_rep(ds: xr.Dataset, validate: bool, owner: str) -> xr.Dataset:
    return set_linear_acceleration_rep(ds, rep="cart", validate=validate, owner=owner)


def _set_angular_acceleration_cart_rep(ds: xr.Dataset, validate: bool, owner: str) -> xr.Dataset:
    return set_angular_acceleration_rep(ds, rep="cart", validate=validate, owner=owner)


def _set_acceleration_vector6_rep(ds: xr.Dataset, validate: bool, owner: str) -> xr.Dataset:
    return set_acceleration_rep(ds, rep="vector6", validate=validate, owner=owner)


def _wrap_rotation_apply_kernel(values: np.ndarray, quat: np.ndarray, *, owner: str) -> np.ndarray:
    try:
        return rotate_vec3_kernel(values, quat)
    except ValueError as exc:
        raise ValueError(f"{owner}: rotation apply vector kernel failed.") from exc


def _velocity_components_for_apply(target: Velocity, *, owner: str) -> tuple[LinearVelocity, AngularVelocity, str]:
    rep = get_velocity_rep(target.unsafe_data, owner=owner)
    if rep == "components":
        return target.linear(validate=False), target.angular(validate=False), rep
    linear_ds, angular_ds = unpack_vector6_to_linear_angular_datasets(
        target.unsafe_data,
        owner=owner,
        opts=VELOCITY_VECTOR6_OPTS,
        set_linear_rep=_set_linear_velocity_cart_rep,
        set_angular_rep=_set_angular_velocity_cart_rep,
    )
    return LinearVelocity._from_unvalidated(linear_ds), AngularVelocity._from_unvalidated(angular_ds), rep


def _acceleration_components_for_apply(
    target: Acceleration,
    *,
    owner: str,
) -> tuple[LinearAcceleration, AngularAcceleration, str]:
    rep = get_acceleration_rep(target.unsafe_data, owner=owner)
    if rep == "components":
        return target.linear(validate=False), target.angular(validate=False), rep
    linear_ds, angular_ds = unpack_vector6_to_linear_angular_datasets(
        target.unsafe_data,
        owner=owner,
        opts=ACCELERATION_VECTOR6_OPTS,
        set_linear_rep=_set_linear_acceleration_cart_rep,
        set_angular_rep=_set_angular_acceleration_cart_rep,
    )
    return LinearAcceleration._from_unvalidated(linear_ds), AngularAcceleration._from_unvalidated(angular_ds), rep


def _velocity_output_for_rep(
    linear_out: LinearVelocity,
    angular_out: AngularVelocity,
    *,
    rep: str,
    owner: str,
) -> Velocity:
    out = Velocity.from_linear_angular(linear_out, angular_out, validate=False)
    if rep == "components":
        return out
    selection = select_topology_policy_with_intents(
        (linear_out, angular_out),
        owner=owner,
        operation_family="spatial.rotation.apply",
        support=operation_intent_support_for_operation_family(
            "spatial.rotation.apply",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    policy = selection.policy
    out_ds = pack_linear_angular_to_vector6_dataset(
        out.linear(validate=False).unsafe_data,
        out.angular(validate=False).unsafe_data,
        owner=owner,
        opts=VELOCITY_VECTOR6_OPTS,
        set_spatial_rep=_set_velocity_vector6_rep,
        policy=policy,
    )
    return Velocity._from_unvalidated(out_ds)


def _acceleration_output_for_rep(
    linear_out: LinearAcceleration,
    angular_out: AngularAcceleration,
    *,
    rep: str,
    owner: str,
) -> Acceleration:
    out = Acceleration.from_linear_angular(linear_out, angular_out, validate=False)
    if rep == "components":
        return out
    selection = select_topology_policy_with_intents(
        (linear_out, angular_out),
        owner=owner,
        operation_family="spatial.rotation.apply",
        support=operation_intent_support_for_operation_family(
            "spatial.rotation.apply",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    policy = selection.policy
    out_ds = pack_linear_angular_to_vector6_dataset(
        out.linear(validate=False).unsafe_data,
        out.angular(validate=False).unsafe_data,
        owner=owner,
        opts=ACCELERATION_VECTOR6_OPTS,
        set_spatial_rep=_set_acceleration_vector6_rep,
        policy=policy,
    )
    return Acceleration._from_unvalidated(out_ds)


def _apply_to_vector_target(
    rotation: "Rotation",
    target: object,
    *,
    validate: bool,
    owner: str,
) -> object:
    target_ds = validate_schema_if_needed(target.unsafe_data)
    quat_rotation = rotation.as_quat(validate=False)
    quat_ds = validate_schema_if_needed(quat_rotation.unsafe_data)
    selection = select_topology_policy_with_intents(
        (rotation, target),
        owner=owner,
        operation_family="spatial.rotation.apply",
        support=operation_intent_support_for_operation_family(
            "spatial.rotation.apply",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    policy = selection.policy
    target_var, target_dim, quat_dim, target_da, quat_da = _resolve_vector_apply_inputs(
        target_ds,
        quat_ds,
        owner=owner,
        target_name=type(target).__name__,
        policy=policy,
    )
    rotated = _apply_vector_rotation_kernel(
        target_da,
        quat_da,
        target_dim=target_dim,
        quat_dim=quat_dim,
        owner=owner,
    )
    out_ds = rotated.to_dataset(name=target_var)
    out_ds = transfer_dataset_attrs(target_ds, out_ds, validate=False)
    parent, child = resolve_apply_output_frames(rotation.unsafe_data, target_ds, owner=owner)
    out_ds = set_frames(out_ds, parent=parent, child=child, validate=False)
    return wrap_like(target, out_ds, validate=validate)


def _resolve_vector_apply_inputs(
    target_ds: xr.Dataset,
    quat_ds: xr.Dataset,
    *,
    owner: str,
    target_name: str,
    policy: TopologyPolicy,
) -> tuple[str, str, str, xr.DataArray, xr.DataArray]:
    target_var, target_dim = resolve_single_numeric_var_single_core_dim(
        target_ds,
        owner=owner,
        what=target_name,
    )
    quat_var, quat_dim = resolve_single_numeric_var_single_core_dim(
        quat_ds,
        owner=owner,
        what="Rotation quaternion layout",
    )
    plan = resolve_binary_topology(
        _topology_operand(
            target_ds,
            var_name=target_var,
            core_dim=target_dim,
            index=0,
            owner=owner,
            what=target_name,
            policy=policy,
        ),
        _topology_operand(
            quat_ds,
            var_name=quat_var,
            core_dim=quat_dim,
            index=1,
            owner=owner,
            what="Rotation quaternion layout",
            policy=policy,
        ),
        owner=owner,
        what="rotation apply",
        policy=policy,
    )
    target_da, quat_da = align_exact_for_plan(plan, owner=owner, what="rotation apply")
    return target_var, target_dim, quat_dim, target_da, quat_da


def _apply_vector_rotation_kernel(
    target_da: xr.DataArray,
    quat_da: xr.DataArray,
    *,
    target_dim: str,
    quat_dim: str,
    owner: str,
) -> xr.DataArray:
    kernel = partial(_wrap_rotation_apply_kernel, owner=owner)
    try:
        rotated = xr.apply_ufunc(
            kernel,
            target_da,
            quat_da,
            input_core_dims=[[target_dim], [quat_dim]],
            output_core_dims=[[target_dim]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {target_dim: 3}},
        )
    except ValueError as exc:
        raise ValueError(f"{owner}: rotation apply failed after alignment: {exc}") from exc
    return rotated.assign_coords({target_dim: list(_XYZ_LABELS)})


def _apply_to_spatial_target(
    rotation: "Rotation",
    target: object,
    *,
    validate: bool,
    owner: str,
) -> object:
    if isinstance(target, Velocity):
        linear_in, angular_in, rep = _velocity_components_for_apply(target, owner=owner)
        linear_out = _apply_to_vector_target(rotation, linear_in, validate=False, owner=owner)
        angular_out = _apply_to_vector_target(rotation, angular_in, validate=False, owner=owner)
        out = _velocity_output_for_rep(linear_out, angular_out, rep=rep, owner=owner)
        if validate:
            return Velocity._from_validated(out.unsafe_data)
        return out
    if isinstance(target, Acceleration):
        linear_in, angular_in, rep = _acceleration_components_for_apply(target, owner=owner)
        linear_out = _apply_to_vector_target(rotation, linear_in, validate=False, owner=owner)
        angular_out = _apply_to_vector_target(rotation, angular_in, validate=False, owner=owner)
        out = _acceleration_output_for_rep(linear_out, angular_out, rep=rep, owner=owner)
        if validate:
            return Acceleration._from_validated(out.unsafe_data)
        return out
    raise TypeError(f"{owner}: unsupported spatial target type {type(target).__name__!r}.")


def _rotation_apply_with_owner(
    rotation: "Rotation",
    target: object,
    *,
    validate: bool,
    owner: str,
) -> object:
    if not isinstance(target, _SUPPORTED_TARGET_TYPES):
        raise TypeError(
            f"{owner}: target must be Position, LinearVelocity, AngularVelocity, "
            "LinearAcceleration, AngularAcceleration, Velocity, or Acceleration."
        )
    rotation._enforce_invariants(owner=owner)
    target._enforce_invariants(owner=owner)
    _ = resolve_apply_output_frames(rotation.unsafe_data, target.unsafe_data, owner=owner)
    if isinstance(target, _VECTOR_TARGET_TYPES):
        return _apply_to_vector_target(rotation, target, validate=validate, owner=owner)
    return _apply_to_spatial_target(rotation, target, validate=validate, owner=owner)


def rotation_apply(rotation: "Rotation", target: object, *, validate: bool) -> object:
    return _rotation_apply_with_owner(
        rotation,
        target,
        validate=validate,
        owner="spatial.rotation.apply",
    )


__all__ = ["rotation_apply"]
