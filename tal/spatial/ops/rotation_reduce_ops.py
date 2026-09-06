from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.finalize import transfer_dataset_attrs
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.orchestration.schema_finalize import (
    CoreSchemaFinalizeSpec,
    finalize_with_schema,
)
from tal.core.reducer_ops.api import (
    assemble_reduced_dataset,
    reduce_analysis_object,
    resolve_reducer_request,
)
from tal.core.reducer_ops.types import DimLike, WeightInput
from tal.core.reducer_ops.validity import (
    apply_structural_mask,
    reduce_missing_on_valid_prefix,
    resolve_structural_valid_mask,
)
from tal.core.reducer_ops.weights import coerce_aligned_weights, validate_weight_values
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.utils.xarray_namespace import unique_temp_dim

from ..kernels.rotation_mean_kernels import quat_mean_kernel
from ..policies.wrap import wrap_as
from .quat_role_dim_ops import resolve_quat_dim_with_role_fallback

if TYPE_CHECKING:
    from ..rotation import Rotation


_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")


def _resolve_quat_payload(rotation: Rotation, *, owner: str) -> tuple[xr.Dataset, str, str, xr.DataArray]:
    source = analysis_object_dataset(rotation.as_quat(validate=False))
    var_name = select_single_numeric_var(source, owner=owner, what="Rotation.mean")
    quat_dim = resolve_quat_dim_with_role_fallback(source, var_name=var_name, owner=owner, what="Rotation.mean")
    require_var_contains_dims(source, var_name=var_name, required_dims=(quat_dim,), owner=owner, what="Rotation.mean")
    labels = require_explicit_unique_dim_labels(source, dim=quat_dim, owner=owner, what="Rotation.mean")
    require_exact_labels(labels, expected=_QUAT_LABELS, owner=owner, what="Rotation.mean quaternion labels")
    return source, var_name, quat_dim, source[var_name]


def _resolve_sequence_size_coord(
    ds: xr.Dataset,
) -> str | None:
    declared, sequence_dim, _, _ = read_roles(ds)
    if not declared or sequence_dim is None:
        return None
    return read_sequence_size_coord_name(ds)


def _prepare_reduce_payload(
    rotation: Rotation,
    *,
    owner: str,
) -> tuple[xr.Dataset, str, str, xr.DataArray, xr.DataArray | None, xr.DataArray]:
    ds, var_name, quat_dim, data = _resolve_quat_payload(rotation, owner=owner)
    _, sequence_dim, _, _ = read_roles(ds)
    sequence_size_coord = _resolve_sequence_size_coord(ds)
    mask = resolve_structural_valid_mask(
        ds,
        sequence_dim=sequence_dim,
        sequence_size_coord=sequence_size_coord,
        var=data,
        owner=owner,
    )
    masked = apply_structural_mask(data, mask=mask)
    return ds, var_name, quat_dim, data, mask, masked


def _broadcast_weight_to_payload(
    weight: xr.DataArray,
    *,
    masked: xr.DataArray,
    quat_dim: str,
    owner: str,
) -> xr.DataArray:
    payload_rows = masked.isel({quat_dim: 0}, drop=True)
    try:
        return weight.broadcast_like(payload_rows)
    except ValueError as exc:
        raise ValueError(
            f"{owner}: weighted Rotation.mean weights are not broadcast-compatible with payload rows."
        ) from exc


def _prepare_weight_for_reduce(
    masked: xr.DataArray,
    *,
    reduce_dims: tuple[str, ...],
    quat_dim: str,
    weights: WeightInput,
    mask: xr.DataArray | None,
    skipna: bool,
    owner: str,
) -> xr.DataArray:
    aligned_weight = coerce_aligned_weights(
        masked,
        reduce_dims=reduce_dims,
        weights=weights,
        op="mean",
        mask=mask,
        owner=owner,
    )
    if aligned_weight is None:
        unit = xr.ones_like(
            masked.isel({quat_dim: 0}, drop=True),
            dtype=np.float64,
        )
        return _broadcast_weight_to_payload(
            unit,
            masked=masked,
            quat_dim=quat_dim,
            owner=owner,
        )
    validated = validate_weight_values(
        aligned_weight,
        mask=mask,
        payload_has_entries=masked.size != 0,
        skipna=skipna,
        owner=owner,
    )
    if skipna:
        validated = validated.fillna(0)
    return _broadcast_weight_to_payload(validated, masked=masked, quat_dim=quat_dim, owner=owner)


def _stack_reduce_dims(
    masked: xr.DataArray,
    weight: xr.DataArray,
    *,
    reduce_dims: tuple[str, ...],
    owner: str,
) -> tuple[xr.DataArray, xr.DataArray, str]:
    reduce_axis = unique_temp_dim("__rotation_reduce__", taken_dims=tuple(masked.dims) + tuple(weight.dims))
    try:
        return masked.stack({reduce_axis: reduce_dims}), weight.stack({reduce_axis: reduce_dims}), reduce_axis
    except ValueError as exc:
        raise ValueError(f"{owner}: failed to stack reduction dims {reduce_dims!r} for Rotation.mean.") from exc


def _execute_quat_reduce(
    masked: xr.DataArray,
    weight: xr.DataArray,
    *,
    dim: str | tuple[str, ...],
    quat_dim: str,
) -> xr.DataArray:
    if masked.size == 0:
        # TAL owns empty quaternion means. Native reduction supplies the lazy
        # output topology without entering Dask's zero-sized gufunc core path.
        template = masked.sum(dim=dim, keep_attrs=True)
        return xr.full_like(template, np.nan, dtype=np.float64)
    reduced = xr.apply_ufunc(
        quat_mean_kernel,
        masked,
        weight,
        input_core_dims=[[dim, quat_dim], [dim]],
        output_core_dims=[[quat_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {quat_dim: 4}},
    )
    return reduced.assign_coords({quat_dim: list(_QUAT_LABELS)})


def _execute_multi_dim_quat_reduce(
    masked: xr.DataArray,
    weight: xr.DataArray,
    *,
    reduce_dims: tuple[str, ...],
    quat_dim: str,
    owner: str,
) -> xr.DataArray:
    if masked.size == 0:
        return _execute_quat_reduce(masked, weight, dim=reduce_dims, quat_dim=quat_dim)
    stacked_masked, stacked_weight, reduce_axis = _stack_reduce_dims(
        masked,
        weight,
        reduce_dims=reduce_dims,
        owner=owner,
    )
    return _execute_quat_reduce(stacked_masked, stacked_weight, dim=reduce_axis, quat_dim=quat_dim)


def _finalize_rotation_reduce(
    rotation: Rotation,
    ds: xr.Dataset,
    *,
    var_name: str,
    reduced: xr.DataArray,
    source: xr.DataArray,
    reduce_dims: tuple[str, ...],
    owner: str,
    validate: bool,
) -> Rotation:
    reduced_arr = reduced if reduced.name == var_name else reduced.rename(var_name)
    _, sequence_dim, _, _ = read_roles(ds)
    candidate = assemble_reduced_dataset(
        ds,
        {var_name: reduced_arr},
        reduce_dims=reduce_dims,
        sequence_dim=sequence_dim,
        param_coord=read_param_coord_name(ds) if sequence_dim is not None else None,
        sequence_size_coord=_resolve_sequence_size_coord(ds),
    )
    candidate = transfer_dataset_attrs(ds, candidate, validate=False)
    spec = _resolve_rotation_reduce_spec(
        ds,
        candidate=candidate,
        reduce_dims=reduce_dims,
    )
    return finalize_with_schema(
        rotation,
        candidate,
        spec=spec,
        validate=validate,
        owner=owner,
        optional_sources=(source, *tuple(ds.coords.values())),
    )


def _apply_rotation_reduce_missing_policy(
    reduced: xr.DataArray,
    source: xr.DataArray,
    *,
    reduce_dims: tuple[str, ...],
    mask: xr.DataArray | None,
    skipna: bool,
) -> xr.DataArray:
    if skipna:
        return reduced
    poison = reduce_missing_on_valid_prefix(
        source,
        reduce_dims=reduce_dims,
        mask=mask,
    )
    return reduced.where(~poison)


def _resolve_rotation_reduce_spec(
    source_ds: xr.Dataset,
    *,
    candidate: xr.Dataset,
    reduce_dims: tuple[str, ...],
) -> CoreSchemaFinalizeSpec:
    declared, sequence_dim, batch_dims, core_dims = read_roles(source_ds)
    reduced = set(reduce_dims)
    sequence_removed = sequence_dim is not None and (
        sequence_dim in reduced or sequence_dim not in candidate.dims
    )
    if not declared or sequence_removed:
        return CoreSchemaFinalizeSpec(sequence_dim=None, batch_dims=(), core_dims=(), param_name=None, size_name=None)
    kept_batch = tuple(
        dim for dim in batch_dims if dim in candidate.dims and dim not in reduced
    )
    kept_core = tuple(
        dim for dim in core_dims if dim in candidate.dims and dim not in reduced
    )
    return CoreSchemaFinalizeSpec(
        sequence_dim=sequence_dim,
        batch_dims=kept_batch,
        core_dims=kept_core,
        param_name=(
            read_param_coord_name(source_ds)
            if sequence_dim is not None
            else None
        ),
        size_name=(
            read_sequence_size_coord_name(source_ds)
            if sequence_dim is not None
            else None
        ),
    )


def _reduce_one_dim(
    rotation: Rotation,
    *,
    dim: str,
    requested_dims: tuple[str, ...],
    skipna: bool,
    weights: WeightInput,
    owner: str,
    validate: bool,
) -> Rotation:
    ds, var_name, quat_dim, source, mask, masked = _prepare_reduce_payload(rotation, owner=owner)
    prepared_weight = _prepare_weight_for_reduce(
        masked,
        reduce_dims=(dim,),
        quat_dim=quat_dim,
        weights=weights,
        mask=mask,
        skipna=skipna,
        owner=owner,
    )
    reduced = _execute_quat_reduce(masked, prepared_weight, dim=dim, quat_dim=quat_dim)
    reduced = _apply_rotation_reduce_missing_policy(
        reduced,
        source,
        reduce_dims=(dim,),
        mask=mask,
        skipna=skipna,
    )
    return _finalize_rotation_reduce(
        rotation,
        ds,
        var_name=var_name,
        reduced=reduced,
        source=source,
        reduce_dims=requested_dims,
        owner=owner,
        validate=validate,
    )


def _reduce_multi_dim(
    rotation: Rotation,
    *,
    reduce_dims: tuple[str, ...],
    requested_dims: tuple[str, ...],
    skipna: bool,
    weights: WeightInput,
    owner: str,
    validate: bool,
) -> Rotation:
    ds, var_name, quat_dim, source, mask, masked = _prepare_reduce_payload(rotation, owner=owner)
    prepared_weight = _prepare_weight_for_reduce(
        masked,
        reduce_dims=reduce_dims,
        quat_dim=quat_dim,
        weights=weights,
        mask=mask,
        skipna=skipna,
        owner=owner,
    )
    reduced = _execute_multi_dim_quat_reduce(
        masked,
        prepared_weight,
        reduce_dims=reduce_dims,
        quat_dim=quat_dim,
        owner=owner,
    )
    reduced = _apply_rotation_reduce_missing_policy(
        reduced,
        source,
        reduce_dims=reduce_dims,
        mask=mask,
        skipna=skipna,
    )
    return _finalize_rotation_reduce(
        rotation,
        ds,
        var_name=var_name,
        reduced=reduced,
        source=source,
        reduce_dims=requested_dims,
        owner=owner,
        validate=validate,
    )


def _reduce_structural_dims(
    rotation: Rotation,
    *,
    reduce_dims: tuple[str, ...],
    owner: str,
    validate: bool,
) -> Rotation:
    """Finalize requested topology without invoking the quaternion kernel."""
    return reduce_analysis_object(
        rotation,
        op="mean",
        dim=reduce_dims,
        skipna=True,
        ddof=0,
        weights=None,
        owner=owner,
        validate=validate,
    )


def rotation_mean(
    rotation: Rotation,
    *,
    dim: DimLike = None,
    skipna: bool = True,
    weights: WeightInput = None,
    validate: bool = True,
    owner: str = "spatial.rotation.mean",
) -> Rotation:
    source = analysis_object_dataset(rotation)
    request = resolve_reducer_request(
        rotation,
        source,
        op="mean",
        dim=dim,
        weights=weights,
        owner=owner,
    )
    if not request.reduce_dims:
        return wrap_as(rotation.__class__, source, validate=validate)
    if not request.active_reduce_dims:
        return _reduce_structural_dims(
            rotation,
            reduce_dims=request.reduce_dims,
            owner=owner,
            validate=validate,
        )
    if len(request.active_reduce_dims) > 1:
        return _reduce_multi_dim(
            rotation,
            reduce_dims=request.active_reduce_dims,
            requested_dims=request.reduce_dims,
            skipna=skipna,
            weights=weights,
            owner=owner,
            validate=validate,
        )
    return _reduce_one_dim(
        rotation,
        dim=request.active_reduce_dims[0],
        requested_dims=request.reduce_dims,
        skipna=skipna,
        weights=weights,
        owner=owner,
        validate=validate,
    )


def _mean_method(
    self: Rotation,
    dim: DimLike = None,
    *,
    skipna: bool = True,
    weights: WeightInput = None,
    validate: bool = True,
) -> Rotation:
    """Mean method."""
    return rotation_mean(
        self,
        dim=dim,
        skipna=skipna,
        weights=weights,
        validate=validate,
        owner="spatial.rotation.mean",
    )


def install_rotation_reducer_methods(cls: type) -> None:
    cls.mean = _mean_method


__all__ = ["install_rotation_reducer_methods", "rotation_mean"]
