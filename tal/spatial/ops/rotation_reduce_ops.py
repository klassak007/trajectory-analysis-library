from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.reducer_ops.dims import resolve_reduce_dims
from tal.core.reducer_ops.types import DimLike, WeightInput
from tal.core.reducer_ops.validity import apply_structural_mask, reduce_missing_on_valid_prefix, resolve_structural_valid_mask
from tal.core.reducer_ops.weights import coerce_aligned_weights, validate_weight_values
from tal.core.orchestration.schema_finalize import (
    CoreSchemaFinalizeSpec,
    finalize_with_schema,
)
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.schema import merge_schema
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.utils.xarray_namespace import unique_temp_dim

from .quat_role_dim_ops import resolve_quat_dim_with_role_fallback
from ..policies.wrap import wrap_as

if TYPE_CHECKING:
    from ..rotation import Rotation


_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")


def _quat_mean_kernel(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if array.shape[-1] != 4:
        raise ValueError("rotation mean kernel requires final quaternion axis length 4.")
    n = int(array.shape[-2])
    batch = array.shape[:-2]
    target_weight_shape = batch + (n,)
    try:
        w = np.broadcast_to(weight, target_weight_shape)
    except ValueError as exc:
        raise ValueError(
            "spatial.rotation.mean: quaternion mean kernel requires weights broadcastable to "
            f"shape {target_weight_shape!r}, got {weight.shape!r}."
        ) from exc
    flat_q = array.reshape((-1, n, 4))
    flat_w = w.reshape((-1, n))
    out = np.full((flat_q.shape[0], 4), np.nan, dtype=np.float64)
    for index in range(flat_q.shape[0]):
        q_row = flat_q[index]
        w_row = flat_w[index]
        valid = np.isfinite(w_row) & np.all(np.isfinite(q_row), axis=1)
        if not np.any(valid):
            continue
        qv = q_row[valid]
        wv = w_row[valid]
        total = float(wv.sum())
        if not np.isfinite(total) or total <= 0:
            continue
        gram = (qv[:, :, None] * qv[:, None, :] * wv[:, None, None]).sum(axis=0)
        eigvals, eigvecs = np.linalg.eigh(gram)
        quat = eigvecs[:, int(np.argmax(eigvals))]
        norm = float(np.linalg.norm(quat))
        if norm <= 0 or not np.isfinite(norm):
            continue
        quat = quat / norm
        if quat[3] < 0:
            quat = -quat
        out[index] = quat
    return out.reshape(batch + (4,))


def _resolve_quat_payload(rotation: "Rotation", *, owner: str) -> tuple[xr.Dataset, str, str, xr.DataArray]:
    source = rotation.as_quat(validate=False).unsafe_data
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
    rotation: "Rotation",
    *,
    owner: str,
) -> tuple[xr.Dataset, str, str, xr.DataArray, xr.DataArray | None, xr.DataArray]:
    ds, var_name, quat_dim, data = _resolve_quat_payload(rotation, owner=owner)
    _, sequence_dim, _, _ = read_roles(ds)
    sequence_size_coord = _resolve_sequence_size_coord(ds)
    mask = resolve_structural_valid_mask(ds, sequence_dim=sequence_dim, sequence_size_coord=sequence_size_coord, var=data)
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
        owner=owner,
    )
    if aligned_weight is None:
        aligned_weight = xr.ones_like(masked.isel({quat_dim: 0}, drop=True), dtype=np.float64)
    validated = validate_weight_values(aligned_weight, mask=mask, skipna=skipna, owner=owner)
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
    dim: str,
    quat_dim: str,
) -> xr.DataArray:
    reduced = xr.apply_ufunc(
        _quat_mean_kernel,
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


def _finalize_rotation_reduce(
    rotation: "Rotation",
    ds: xr.Dataset,
    *,
    var_name: str,
    reduced: xr.DataArray,
    source: xr.DataArray,
    reduce_dims: tuple[str, ...],
    mask: xr.DataArray | None,
    skipna: bool,
    owner: str,
    validate: bool,
) -> "Rotation":
    if not skipna:
        poison = reduce_missing_on_valid_prefix(source, reduce_dims=reduce_dims, mask=mask)
        reduced = reduced.where(~poison)
    candidate = _build_rotation_reduce_dataset(ds, var_name=var_name, reduced=reduced)
    spec = _resolve_rotation_reduce_spec(ds, reduced=reduced, reduce_dims=reduce_dims)
    return finalize_with_schema(
        rotation,
        candidate,
        spec=spec,
        validate=validate,
        owner=owner,
        optional_sources=(source,),
    )


def _copy_non_tal_attrs(source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    attrs = {name: value for name, value in source.attrs.items() if name != "tal"}
    if not attrs:
        return target
    return target.assign_attrs(attrs)


def _source_ext_patch(ds: xr.Dataset) -> Mapping[str, object] | None:
    tal = ds.attrs.get("tal")
    if not isinstance(tal, Mapping):
        return None
    ext = tal.get("ext")
    if not isinstance(ext, Mapping):
        return None
    return {"ext": dict(ext)}


def _build_rotation_reduce_dataset(
    source_ds: xr.Dataset,
    *,
    var_name: str,
    reduced: xr.DataArray,
) -> xr.Dataset:
    reduced_arr = reduced if reduced.name == var_name else reduced.rename(var_name)
    candidate = _copy_non_tal_attrs(source_ds, reduced_arr.to_dataset(name=var_name))
    ext_patch = _source_ext_patch(source_ds)
    if ext_patch is None:
        return candidate
    return merge_schema(candidate, ext_patch, validate=False)


def _resolve_rotation_reduce_spec(
    source_ds: xr.Dataset,
    *,
    reduced: xr.DataArray,
    reduce_dims: tuple[str, ...],
) -> CoreSchemaFinalizeSpec:
    declared, sequence_dim, batch_dims, core_dims = read_roles(source_ds)
    if not declared or sequence_dim is None or sequence_dim not in reduced.dims:
        return CoreSchemaFinalizeSpec(sequence_dim=None, batch_dims=(), core_dims=(), param_name=None, size_name=None)
    reduced_set = set(reduce_dims)
    kept_batch = tuple(dim for dim in batch_dims if dim in reduced.dims and dim not in reduced_set and dim != sequence_dim)
    kept_core = tuple(dim for dim in core_dims if dim in reduced.dims and dim != sequence_dim)
    return CoreSchemaFinalizeSpec(
        sequence_dim=sequence_dim,
        batch_dims=kept_batch,
        core_dims=kept_core,
        param_name=read_param_coord_name(source_ds),
        size_name=read_sequence_size_coord_name(source_ds),
    )


def _reduce_one_dim(
    rotation: "Rotation",
    *,
    dim: str,
    skipna: bool,
    weights: WeightInput,
    owner: str,
    validate: bool,
) -> "Rotation":
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
    return _finalize_rotation_reduce(
        rotation,
        ds,
        var_name=var_name,
        reduced=reduced,
        source=source,
        reduce_dims=(dim,),
        mask=mask,
        skipna=skipna,
        owner=owner,
        validate=validate,
    )


def _reduce_multi_dim(
    rotation: "Rotation",
    *,
    reduce_dims: tuple[str, ...],
    skipna: bool,
    weights: WeightInput,
    owner: str,
    validate: bool,
) -> "Rotation":
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
    stacked_masked, stacked_weight, reduce_axis = _stack_reduce_dims(
        masked,
        prepared_weight,
        reduce_dims=reduce_dims,
        owner=owner,
    )
    reduced = _execute_quat_reduce(stacked_masked, stacked_weight, dim=reduce_axis, quat_dim=quat_dim)
    return _finalize_rotation_reduce(
        rotation,
        ds,
        var_name=var_name,
        reduced=reduced,
        source=source,
        reduce_dims=reduce_dims,
        mask=mask,
        skipna=skipna,
        owner=owner,
        validate=validate,
    )


def rotation_mean(
    rotation: "Rotation",
    *,
    dim: DimLike = None,
    skipna: bool = True,
    weights: WeightInput = None,
    validate: bool = True,
    owner: str = "spatial.rotation.mean",
) -> "Rotation":
    reduce_dims = resolve_reduce_dims(
        rotation.unsafe_data,
        dim=dim,
        component_dims=rotation._required_component_dims_for_reduce(),
        owner=owner,
    )
    if isinstance(weights, np.ndarray) and len(reduce_dims) > 1:
        raise ValueError(f"{owner}: ndarray weights are only valid for single-dim reduction.")
    if not reduce_dims:
        return wrap_as(rotation.__class__, rotation.unsafe_data, validate=validate)
    if len(reduce_dims) > 1:
        return _reduce_multi_dim(
            rotation,
            reduce_dims=reduce_dims,
            skipna=skipna,
            weights=weights,
            owner=owner,
            validate=validate,
        )
    reduce_dim = reduce_dims[0]
    if reduce_dim not in rotation.unsafe_data.dims:
        return wrap_as(rotation.__class__, rotation.unsafe_data, validate=validate)
    return _reduce_one_dim(
        rotation,
        dim=reduce_dim,
        skipna=skipna,
        weights=weights,
        owner=owner,
        validate=validate,
    )


def _mean_method(
    self: "Rotation",
    dim: DimLike = None,
    *,
    skipna: bool = True,
    weights: WeightInput = None,
    validate: bool = True,
) -> "Rotation":
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
