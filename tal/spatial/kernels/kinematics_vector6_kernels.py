from __future__ import annotations

import numpy as np


def _require_last_dim(values: np.ndarray, *, expected: int, owner: str, what: str) -> None:
    if values.ndim < 1:
        raise ValueError(f"{owner}: {what} must have at least one dimension.")
    if int(values.shape[-1]) != expected:
        raise ValueError(f"{owner}: {what} last dimension must have length {expected}.")


def pack_vector6_kernel(linear: np.ndarray, angular: np.ndarray) -> np.ndarray:
    owner = "spatial.kinematics.vector6.kernel.pack"
    _require_last_dim(linear, expected=3, owner=owner, what="linear vector")
    _require_last_dim(angular, expected=3, owner=owner, what="angular vector")
    if linear.shape[:-1] != angular.shape[:-1]:
        raise ValueError(
            f"{owner}: linear/angular non-core shape mismatch "
            f"({linear.shape[:-1]!r} vs {angular.shape[:-1]!r})."
        )
    return np.concatenate((linear, angular), axis=-1)


def unpack_vector6_linear_kernel(values: np.ndarray) -> np.ndarray:
    owner = "spatial.kinematics.vector6.kernel.unpack_linear"
    _require_last_dim(values, expected=6, owner=owner, what="vector6")
    return values[..., :3]


def unpack_vector6_angular_kernel(values: np.ndarray) -> np.ndarray:
    owner = "spatial.kinematics.vector6.kernel.unpack_angular"
    _require_last_dim(values, expected=6, owner=owner, what="vector6")
    return values[..., 3:]


__all__ = [
    "pack_vector6_kernel",
    "unpack_vector6_angular_kernel",
    "unpack_vector6_linear_kernel",
]
