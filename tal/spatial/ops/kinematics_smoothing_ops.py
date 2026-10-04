from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset

from ..kernels.kinematics_temporal_kernels import (
    gaussian_partial_renorm_kernel,
    local_poly_smooth_kernel,
    moving_average_partial_renorm_kernel,
)
from ..metadata import (
    get_angular_acceleration_rep,
    get_angular_velocity_rep,
    get_linear_acceleration_rep,
    get_linear_velocity_rep,
    get_position_intent,
    get_position_rep,
    set_angular_acceleration_rep,
    set_angular_velocity_rep,
    set_linear_acceleration_rep,
    set_linear_velocity_rep,
    set_position_rep,
)
from ..temporal.options import (
    KinematicsSmoothingOptions,
    coerce_kinematics_smoothing_options,
    require_supported_smoothing_options,
)
from .kinematics_temporal_ops import (
    TemporalOutputSpec,
    _apply_output_metadata,
    _replace_payload,
    _resolve_payload_var_and_core,
    _resolve_runtime,
    _runtime_source,
    _wrap_owner_error,
)


@dataclass(frozen=True)
class SmoothingRequest:
    source: object
    on: str | None
    opts: object | None
    validate: bool
    sequence_dim: str | None
    batch_dims: Sequence[str] | None
    sequence_size_coord: str | None
    owner: str


def _apply_smoothing_operator(
    context,
    *,
    var_name: str,
    core_dim: str,
    opts: KinematicsSmoothingOptions,
) -> xr.DataArray:
    source = context.ds[var_name]
    source_no_core = source.isel({core_dim: 0}, drop=True)
    param = context.spec.coord.broadcast_like(source_no_core)
    valid = context.valid_mask.broadcast_like(source_no_core)
    if opts.method == "local_poly":
        kernel = local_poly_smooth_kernel
        kernel_kwargs = {"window": int(opts.window), "poly_order": int(opts.poly_order)}
    elif opts.method == "gaussian":
        kernel = gaussian_partial_renorm_kernel
        kernel_kwargs = {"window": int(opts.window), "sigma": float(opts.sigma)}
    else:
        kernel = moving_average_partial_renorm_kernel
        kernel_kwargs = {"window": int(opts.window)}
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


def _smoothing_spec_for_source(source: object, *, owner: str) -> TemporalOutputSpec:
    from ..acceleration import AngularAcceleration, LinearAcceleration
    from ..position import Position
    from ..velocity import AngularVelocity, LinearVelocity

    if isinstance(source, Position):
        intent = get_position_intent(analysis_object_dataset(source), owner=owner)
        return TemporalOutputSpec(Position, get_position_rep, set_position_rep, None, intent)
    if isinstance(source, LinearVelocity):
        return TemporalOutputSpec(LinearVelocity, get_linear_velocity_rep, set_linear_velocity_rep, "linear_velocity", None)
    if isinstance(source, AngularVelocity):
        return TemporalOutputSpec(AngularVelocity, get_angular_velocity_rep, set_angular_velocity_rep, "angular_velocity", None)
    if isinstance(source, LinearAcceleration):
        return TemporalOutputSpec(
            LinearAcceleration, get_linear_acceleration_rep, set_linear_acceleration_rep, "linear_acceleration", None
        )
    if isinstance(source, AngularAcceleration):
        return TemporalOutputSpec(
            AngularAcceleration, get_angular_acceleration_rep, set_angular_acceleration_rep, "angular_acceleration", None
        )
    raise TypeError(f"{owner}: smooth is supported only on typed Position/Velocity/Acceleration classes.")


def _run_smoothing_request(request: SmoothingRequest):
    opts = coerce_kinematics_smoothing_options(request.opts, owner=request.owner)
    require_supported_smoothing_options(opts, owner=request.owner)
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
        what="kinematics smoothing source",
    )
    values = _apply_smoothing_operator(context, var_name=var_name, core_dim=core_dim, opts=opts)
    out = _replace_payload(context.ds, var_name=var_name, values=values)
    spec = _smoothing_spec_for_source(request.source, owner=request.owner)
    return _apply_output_metadata(out, source_ds=context.ds, spec=spec, owner=request.owner), spec


def smooth_kinematics_like(
    source: object,
    *,
    on: str | None,
    opts: object | None,
    validate: bool,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
    owner: str,
):
    from ..association import finalize_spatial_from_source

    request = SmoothingRequest(source, on, opts, validate, sequence_dim, batch_dims, sequence_size_coord, owner)
    try:
        finalized, spec = _run_smoothing_request(request)
        return finalize_spatial_from_source(
            source,
            spec.target_cls,
            finalized,
            validate=validate,
        )
    except (TypeError, ValueError) as exc:
        raise _wrap_owner_error(exc, owner=owner) from exc


__all__ = ["smooth_kinematics_like"]
