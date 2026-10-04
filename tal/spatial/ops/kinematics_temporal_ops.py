from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.orchestration.runtime_checks import require_var_contains_dims, select_single_numeric_var
from tal.core.param_ops.types import ParamRuntimeContext
from tal.utils.frame_schema import get_frames, set_frames

from ..association import finalize_spatial_from_source
from ..kernels.kinematics_temporal_kernels import (
    cumulative_simpson_kernel,
    cumulative_trapezoid_kernel,
    finite_difference_one_sided_kernel,
    local_poly_first_derivative_kernel,
)
from ..metadata import (
    get_expressed_in,
    get_instantaneous_inertial,
    get_angular_acceleration_rep,
    get_angular_velocity_rep,
    get_linear_acceleration_rep,
    get_linear_velocity_rep,
    get_position_rep,
    set_expressed_in,
    set_instantaneous_inertial,
    set_angular_acceleration_rep,
    set_angular_velocity_rep,
    set_kinematics_kind,
    set_linear_acceleration_rep,
    set_linear_velocity_rep,
    set_position_rep,
)
from ..metadata.roles import set_position_intent
from ..temporal.options import (
    KinematicsDerivativeOptions,
    KinematicsIntegralOptions,
    coerce_kinematics_derivative_options,
    coerce_kinematics_integral_options,
    require_supported_derivative_options,
    require_supported_integral_options,
)

if TYPE_CHECKING:
    from ..acceleration import AngularAcceleration, LinearAcceleration
    from ..position import Position
    from ..velocity import AngularVelocity, LinearVelocity

RepGetter = Callable[..., str]
RepSetter = Callable[..., xr.Dataset]


@dataclass(frozen=True)
class DerivativeRequest:
    source: object
    on: str | None
    opts: object | None
    validate: bool
    sequence_dim: str | None
    batch_dims: Sequence[str] | None
    sequence_size_coord: str | None
    owner: str


@dataclass(frozen=True)
class IntegralRequest:
    source: object
    on: str | None
    opts: object | None
    validate: bool
    sequence_dim: str | None
    batch_dims: Sequence[str] | None
    sequence_size_coord: str | None
    owner: str


@dataclass(frozen=True)
class TemporalOutputSpec:
    target_cls: type
    source_rep_getter: RepGetter
    target_rep_setter: RepSetter
    target_kind: str | None
    position_intent: str | None = None


def _derivative_kernel_kwargs(opts: KinematicsDerivativeOptions) -> tuple[Callable[..., np.ndarray], dict[str, int]]:
    if opts.method == "local_poly":
        return local_poly_first_derivative_kernel, {"window": int(opts.window), "poly_order": int(opts.poly_order)}
    return finite_difference_one_sided_kernel, {}


def _integral_kernel_kwargs(opts: KinematicsIntegralOptions) -> tuple[Callable[..., np.ndarray], dict[str, float]]:
    if opts.method == "simpson":
        return cumulative_simpson_kernel, {"initial_value": float(opts.initial_value)}
    return cumulative_trapezoid_kernel, {"initial_value": float(opts.initial_value)}


def _runtime_source(source: object, *, on: str | None):
    if on is None:
        return source
    return source.set_param_coord(name=on, validate=False)


def _resolve_runtime(
    source: object,
    *,
    on: str | None,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
) -> ParamRuntimeContext:
    return resolve_param_runtime_context(
        source,
        on=on,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
    )


def _resolve_payload_var_and_core(
    context: ParamRuntimeContext,
    *,
    owner: str,
    what: str,
) -> tuple[str, str]:
    var_name = select_single_numeric_var(context.ds, owner=owner, what=what)
    if len(context.core_dims) != 1:
        raise ValueError(f"{owner}: {what} requires exactly one core dim; got {context.core_dims!r}.")
    core_dim = context.core_dims[0]
    require_var_contains_dims(
        context.ds,
        var_name=var_name,
        required_dims=(context.sequence_dim, core_dim),
        owner=owner,
        what=what,
    )
    return var_name, core_dim


def _apply_derivative_operator(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    core_dim: str,
    opts: KinematicsDerivativeOptions,
) -> xr.DataArray:
    source = context.ds[var_name]
    source_no_core = source.isel({core_dim: 0}, drop=True)
    param = context.spec.coord.broadcast_like(source_no_core)
    valid = context.valid_mask.broadcast_like(source_no_core)
    kernel, kernel_kwargs = _derivative_kernel_kwargs(opts)
    out = xr.apply_ufunc(
        kernel,
        source,
        param,
        valid,
        kwargs=kernel_kwargs,
        input_core_dims=[[context.sequence_dim, core_dim], [context.sequence_dim], [context.sequence_dim]],
        output_core_dims=[[context.sequence_dim, core_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {core_dim: int(source.sizes[core_dim])}},
    )
    out = out.transpose(*source.dims)
    if core_dim in source.coords and source.coords[core_dim].dims == (core_dim,):
        out = out.assign_coords({core_dim: source.coords[core_dim]})
    return out


def _apply_integral_operator(
    context: ParamRuntimeContext,
    *,
    var_name: str,
    core_dim: str,
    opts: KinematicsIntegralOptions,
) -> xr.DataArray:
    source = context.ds[var_name]
    source_no_core = source.isel({core_dim: 0}, drop=True)
    param = context.spec.coord.broadcast_like(source_no_core)
    valid = context.valid_mask.broadcast_like(source_no_core)
    kernel, kernel_kwargs = _integral_kernel_kwargs(opts)
    out = xr.apply_ufunc(
        kernel,
        source,
        param,
        valid,
        kwargs=kernel_kwargs,
        input_core_dims=[[context.sequence_dim, core_dim], [context.sequence_dim], [context.sequence_dim]],
        output_core_dims=[[context.sequence_dim, core_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {core_dim: int(source.sizes[core_dim])}},
    )
    out = out.transpose(*source.dims)
    if core_dim in source.coords and source.coords[core_dim].dims == (core_dim,):
        out = out.assign_coords({core_dim: source.coords[core_dim]})
    return out


def _replace_payload(
    ds: xr.Dataset,
    *,
    var_name: str,
    values: xr.DataArray,
) -> xr.Dataset:
    out = ds.copy(deep=False)
    out[var_name] = values
    return out


def _apply_output_metadata(
    ds: xr.Dataset,
    *,
    source_ds: xr.Dataset,
    spec: TemporalOutputSpec,
    owner: str,
) -> xr.Dataset:
    rep = spec.source_rep_getter(source_ds, owner=owner)
    out = spec.target_rep_setter(ds, rep=rep, validate=False, owner=owner)
    out = set_kinematics_kind(out, kind=spec.target_kind, validate=False, owner=owner)
    out = set_position_intent(out, intent=spec.position_intent, validate=False, owner=owner)
    expressed_in = get_expressed_in(source_ds, owner=owner)
    out = set_expressed_in(out, expressed_in=expressed_in, validate=False, owner=owner)
    inertial = get_instantaneous_inertial(source_ds, owner=owner)
    out = set_instantaneous_inertial(
        out,
        instantaneous_inertial=(inertial if spec.target_kind is not None else None),
        validate=False,
        owner=owner,
    )
    parent, child = get_frames(source_ds)
    return set_frames(out, parent=parent, child=child, validate=False)


def _wrap_owner_error(exc: Exception, *, owner: str) -> Exception:
    text = str(exc)
    if text.startswith(f"{owner}:"):
        return exc
    return type(exc)(f"{owner}: {text}")


def _run_derivative_request(
    request: DerivativeRequest,
    *,
    spec: TemporalOutputSpec,
):
    opts = coerce_kinematics_derivative_options(request.opts, owner=request.owner)
    require_supported_derivative_options(opts, owner=request.owner)
    runtime_source = _runtime_source(request.source, on=request.on)
    context = _resolve_runtime(
        runtime_source,
        on=request.on,
        sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        sequence_size_coord=request.sequence_size_coord,
    )
    var_name, core_dim = _resolve_payload_var_and_core(
        context,
        owner=request.owner,
        what="kinematics derivative source",
    )
    values = _apply_derivative_operator(context, var_name=var_name, core_dim=core_dim, opts=opts)
    out = _replace_payload(context.ds, var_name=var_name, values=values)
    finalized = _apply_output_metadata(out, source_ds=context.ds, spec=spec, owner=request.owner)
    return finalize_spatial_from_source(
        request.source,
        spec.target_cls,
        finalized,
        validate=request.validate,
    )


def _run_integral_request(
    request: IntegralRequest,
    *,
    spec: TemporalOutputSpec,
):
    opts = coerce_kinematics_integral_options(request.opts, owner=request.owner)
    require_supported_integral_options(opts, owner=request.owner)
    runtime_source = _runtime_source(request.source, on=request.on)
    context = _resolve_runtime(
        runtime_source,
        on=request.on,
        sequence_dim=request.sequence_dim,
        batch_dims=request.batch_dims,
        sequence_size_coord=request.sequence_size_coord,
    )
    var_name, core_dim = _resolve_payload_var_and_core(
        context,
        owner=request.owner,
        what="kinematics integral source",
    )
    values = _apply_integral_operator(context, var_name=var_name, core_dim=core_dim, opts=opts)
    out = _replace_payload(context.ds, var_name=var_name, values=values)
    finalized = _apply_output_metadata(out, source_ds=context.ds, spec=spec, owner=request.owner)
    return finalize_spatial_from_source(
        request.source,
        spec.target_cls,
        finalized,
        validate=request.validate,
    )


def _position_to_linear_velocity_spec() -> TemporalOutputSpec:
    from ..velocity import LinearVelocity

    return TemporalOutputSpec(
        target_cls=LinearVelocity,
        source_rep_getter=get_position_rep,
        target_rep_setter=set_linear_velocity_rep,
        target_kind="linear_velocity",
        position_intent=None,
    )


def _linear_velocity_to_linear_acceleration_spec() -> TemporalOutputSpec:
    from ..acceleration import LinearAcceleration

    return TemporalOutputSpec(
        target_cls=LinearAcceleration,
        source_rep_getter=get_linear_velocity_rep,
        target_rep_setter=set_linear_acceleration_rep,
        target_kind="linear_acceleration",
        position_intent=None,
    )


def _angular_velocity_to_angular_acceleration_spec() -> TemporalOutputSpec:
    from ..acceleration import AngularAcceleration

    return TemporalOutputSpec(
        target_cls=AngularAcceleration,
        source_rep_getter=get_angular_velocity_rep,
        target_rep_setter=set_angular_acceleration_rep,
        target_kind="angular_acceleration",
        position_intent=None,
    )


def _linear_velocity_to_position_spec() -> TemporalOutputSpec:
    from ..position import Position

    return TemporalOutputSpec(
        target_cls=Position,
        source_rep_getter=get_linear_velocity_rep,
        target_rep_setter=set_position_rep,
        target_kind=None,
        position_intent="delta",
    )


def _linear_acceleration_to_linear_velocity_spec() -> TemporalOutputSpec:
    from ..velocity import LinearVelocity

    return TemporalOutputSpec(
        target_cls=LinearVelocity,
        source_rep_getter=get_linear_acceleration_rep,
        target_rep_setter=set_linear_velocity_rep,
        target_kind="linear_velocity",
        position_intent=None,
    )


def _angular_acceleration_to_angular_velocity_spec() -> TemporalOutputSpec:
    from ..velocity import AngularVelocity

    return TemporalOutputSpec(
        target_cls=AngularVelocity,
        source_rep_getter=get_angular_acceleration_rep,
        target_rep_setter=set_angular_velocity_rep,
        target_kind="angular_velocity",
        position_intent=None,
    )


def differentiate_position_to_linear_velocity(
    position: Position,
    *,
    on: str | None,
    opts: object | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> LinearVelocity:
    request = DerivativeRequest(
        source=position,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    try:
        return _run_derivative_request(request, spec=_position_to_linear_velocity_spec())
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def differentiate_linear_velocity_to_linear_acceleration(
    velocity: LinearVelocity,
    *,
    on: str | None,
    opts: object | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> LinearAcceleration:
    request = DerivativeRequest(
        source=velocity,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    try:
        return _run_derivative_request(request, spec=_linear_velocity_to_linear_acceleration_spec())
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def differentiate_angular_velocity_to_angular_acceleration(
    velocity: AngularVelocity,
    *,
    on: str | None,
    opts: object | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> AngularAcceleration:
    request = DerivativeRequest(
        source=velocity,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    try:
        return _run_derivative_request(request, spec=_angular_velocity_to_angular_acceleration_spec())
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def integrate_linear_velocity_to_position(
    velocity: LinearVelocity,
    *,
    on: str | None,
    opts: object | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> Position:
    request = IntegralRequest(
        source=velocity,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    try:
        return _run_integral_request(request, spec=_linear_velocity_to_position_spec())
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def integrate_linear_acceleration_to_linear_velocity(
    acceleration: LinearAcceleration,
    *,
    on: str | None,
    opts: object | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> LinearVelocity:
    request = IntegralRequest(
        source=acceleration,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    try:
        return _run_integral_request(request, spec=_linear_acceleration_to_linear_velocity_spec())
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


def integrate_angular_acceleration_to_angular_velocity(
    acceleration: AngularAcceleration,
    *,
    on: str | None,
    opts: object | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
) -> AngularVelocity:
    request = IntegralRequest(
        source=acceleration,
        on=on,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
        owner=owner,
    )
    try:
        return _run_integral_request(request, spec=_angular_acceleration_to_angular_velocity_spec())
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


__all__ = [
    "differentiate_angular_velocity_to_angular_acceleration",
    "differentiate_linear_velocity_to_linear_acceleration",
    "differentiate_position_to_linear_velocity",
    "integrate_angular_acceleration_to_angular_velocity",
    "integrate_linear_acceleration_to_linear_velocity",
    "integrate_linear_velocity_to_position",
]
