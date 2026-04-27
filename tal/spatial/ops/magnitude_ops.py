from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.core.var_naming import default_datavar_name
from tal.linalg import norm as linalg_norm
from tal.linalg.array import Array
from tal.linalg.finalize import ArrayFinalizeSpec, finalize_array_result

from .quat_role_dim_ops import resolve_quat_dim_with_role_fallback

if TYPE_CHECKING:
    from ..rotation import Rotation

_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")


def _coerce_array_output(value: object, *, owner: str) -> Array:
    if not isinstance(value, Array):
        raise TypeError(f"{owner}: expected Array output; got {type(value).__name__}.")
    if type(value) is Array:
        return value
    return Array._from_unvalidated(value.unsafe_data)


def spatial_vector_norm(
    value: object,
    *,
    ord: int | float | None = 2,
    owner: str,
) -> Array:
    try:
        out = linalg_norm(value, ord=ord)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{owner}: {exc}") from exc
    return _coerce_array_output(out, owner=owner)


def spatial_vector_magnitude(
    value: object,
    *,
    owner: str,
) -> Array:
    return spatial_vector_norm(value, ord=2, owner=owner)


def _resolve_rotation_quat_payload(
    rotation: "Rotation",
    *,
    owner: str,
) -> tuple[AnalysisObject, xr.Dataset, str, str, xr.DataArray]:
    try:
        quat = rotation.as_quat(validate=False)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{owner}: {exc}") from exc
    source_ao = AnalysisObject._from_unvalidated(quat.unsafe_data)
    source_ds = source_ao.unsafe_data
    var_name = select_single_numeric_var(source_ds, owner=owner, what="Rotation magnitude")
    quat_dim = resolve_quat_dim_with_role_fallback(
        source_ds,
        var_name=var_name,
        owner=owner,
        what="Rotation magnitude",
    )
    require_var_contains_dims(
        source_ds,
        var_name=var_name,
        required_dims=(quat_dim,),
        owner=owner,
        what="Rotation magnitude",
    )
    labels = require_explicit_unique_dim_labels(
        source_ds,
        dim=quat_dim,
        owner=owner,
        what="Rotation magnitude",
    )
    require_exact_labels(
        labels,
        expected=_QUAT_LABELS,
        owner=owner,
        what="Rotation magnitude quaternion labels",
    )
    return source_ao, source_ds, var_name, quat_dim, source_ds[var_name]


def _rotation_angle_magnitude_kernel(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[-1] != 4:
        raise ValueError("spatial.rotation.magnitude: expected trailing quaternion dim length 4.")
    norms = np.linalg.norm(arr, axis=-1)
    out = np.full(norms.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(norms) & (norms > 0.0)
    if not np.any(valid):
        return out
    normalized = np.zeros_like(arr, dtype=np.float64)
    normalized[valid] = arr[valid] / norms[valid][..., None]
    vec_norm = np.linalg.norm(normalized[..., :3], axis=-1)
    scalar = np.abs(normalized[..., 3])
    angles = 2.0 * np.arctan2(vec_norm, scalar)
    out[valid] = np.clip(angles[valid], 0.0, np.pi)
    return out


def _compute_rotation_angle_magnitude(
    payload: xr.DataArray,
    *,
    quat_dim: str,
) -> xr.DataArray:
    return xr.apply_ufunc(
        _rotation_angle_magnitude_kernel,
        payload,
        input_core_dims=[[quat_dim]],
        output_core_dims=[[]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _finalize_rotation_angle_magnitude(
    source_ao: AnalysisObject,
    source_ds: xr.Dataset,
    source_var: xr.DataArray,
    result: xr.DataArray,
    *,
    owner: str,
) -> Array:
    declared, sequence_dim, batch_dims, _ = read_roles(source_ds)
    if not declared:
        raise ValueError(f"{owner}: rotation magnitude requires declared roles.")
    spec = ArrayFinalizeSpec(
        output_var_name=default_datavar_name(),
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=(),
        param_name=read_param_coord_name(source_ds),
        size_name=read_sequence_size_coord_name(source_ds),
    )
    finalized = finalize_array_result(
        source_ao,
        result,
        spec=spec,
        optional_sources=(source_var,),
        owner=owner,
        validate=True,
    )
    return Array._from_validated(finalized.unsafe_data)


def rotation_angle_magnitude(
    rotation: "Rotation",
    *,
    owner: str,
) -> Array:
    source_ao, source_ds, _, quat_dim, payload = _resolve_rotation_quat_payload(rotation, owner=owner)
    result = _compute_rotation_angle_magnitude(payload, quat_dim=quat_dim)
    return _finalize_rotation_angle_magnitude(
        source_ao,
        source_ds,
        payload,
        result,
        owner=owner,
    )


def _position_norm_method(
    self,
    *,
    ord: int | float | None = 2,
) -> Array:
    """Position norm method."""
    return spatial_vector_norm(self, ord=ord, owner="spatial.position.norm")


def _position_magnitude_method(self) -> Array:
    """Position magnitude method."""
    return spatial_vector_magnitude(self, owner="spatial.position.magnitude")


def install_position_magnitude_methods(cls: type) -> None:
    cls.norm = _position_norm_method
    cls.magnitude = _position_magnitude_method


def _linear_velocity_norm_method(
    self,
    *,
    ord: int | float | None = 2,
) -> Array:
    """Linear velocity norm method."""
    return spatial_vector_norm(self, ord=ord, owner="spatial.linear_velocity.norm")


def _linear_velocity_magnitude_method(self) -> Array:
    """Linear velocity magnitude method."""
    return spatial_vector_magnitude(self, owner="spatial.linear_velocity.magnitude")


def _angular_velocity_norm_method(
    self,
    *,
    ord: int | float | None = 2,
) -> Array:
    """Angular velocity norm method."""
    return spatial_vector_norm(self, ord=ord, owner="spatial.angular_velocity.norm")


def _angular_velocity_magnitude_method(self) -> Array:
    """Angular velocity magnitude method."""
    return spatial_vector_magnitude(self, owner="spatial.angular_velocity.magnitude")


def install_velocity_magnitude_methods(linear_cls: type, angular_cls: type) -> None:
    linear_cls.norm = _linear_velocity_norm_method
    linear_cls.magnitude = _linear_velocity_magnitude_method
    angular_cls.norm = _angular_velocity_norm_method
    angular_cls.magnitude = _angular_velocity_magnitude_method


def _linear_acceleration_norm_method(
    self,
    *,
    ord: int | float | None = 2,
) -> Array:
    """Linear acceleration norm method."""
    return spatial_vector_norm(self, ord=ord, owner="spatial.linear_acceleration.norm")


def _linear_acceleration_magnitude_method(self) -> Array:
    """Linear acceleration magnitude method."""
    return spatial_vector_magnitude(self, owner="spatial.linear_acceleration.magnitude")


def _angular_acceleration_norm_method(
    self,
    *,
    ord: int | float | None = 2,
) -> Array:
    """Angular acceleration norm method."""
    return spatial_vector_norm(self, ord=ord, owner="spatial.angular_acceleration.norm")


def _angular_acceleration_magnitude_method(self) -> Array:
    """Angular acceleration magnitude method."""
    return spatial_vector_magnitude(self, owner="spatial.angular_acceleration.magnitude")


def install_acceleration_magnitude_methods(linear_cls: type, angular_cls: type) -> None:
    linear_cls.norm = _linear_acceleration_norm_method
    linear_cls.magnitude = _linear_acceleration_magnitude_method
    angular_cls.norm = _angular_acceleration_norm_method
    angular_cls.magnitude = _angular_acceleration_magnitude_method


def _rotation_norm_method(self) -> Array:
    """Rotation norm method."""
    return rotation_angle_magnitude(self, owner="spatial.rotation.norm")


def _rotation_magnitude_method(self) -> Array:
    """Rotation magnitude method."""
    return rotation_angle_magnitude(self, owner="spatial.rotation.magnitude")


def install_rotation_magnitude_methods(cls: type) -> None:
    cls.norm = _rotation_norm_method
    cls.magnitude = _rotation_magnitude_method


__all__ = [
    "install_acceleration_magnitude_methods",
    "install_position_magnitude_methods",
    "install_rotation_magnitude_methods",
    "install_velocity_magnitude_methods",
    "rotation_angle_magnitude",
    "spatial_vector_magnitude",
    "spatial_vector_norm",
]
