from __future__ import annotations

import numpy as np

from ..kernels.fixed_size_backends import (
    matrix_to_quat_block_backend,
    quat_compose_block_backend,
    quat_inverse_block_backend,
    quat_to_matrix_block_backend,
)


def wrap_quat_to_matrix_backend(values: np.ndarray) -> np.ndarray:
    try:
        return quat_to_matrix_block_backend(values)
    except ValueError as exc:
        raise ValueError(f"spatial.rotation.kernel.quat_to_matrix: {exc}") from exc


def wrap_matrix_to_quat_backend(values: np.ndarray) -> np.ndarray:
    try:
        return matrix_to_quat_block_backend(values)
    except ValueError as exc:
        raise ValueError(f"spatial.rotation.kernel.matrix_to_quat: {exc}") from exc


def wrap_compose_quat_kernel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    owner: str,
) -> np.ndarray:
    try:
        return quat_compose_block_backend(left, right)
    except ValueError as exc:
        if owner.startswith("spatial.path_solve."):
            raise ValueError(f"{owner}: compose kernel failed after alignment.") from exc
        raise ValueError(
            f"{owner}: compose kernel failed after alignment: "
            f"spatial.rotation.kernel.compose: {exc}"
        ) from exc


def wrap_inverse_quat_kernel(values: np.ndarray, *, owner: str) -> np.ndarray:
    try:
        return quat_inverse_block_backend(values)
    except ValueError as exc:
        if owner.startswith("spatial.path_solve."):
            raise ValueError(f"{owner}: inverse kernel failed.") from exc
        raise ValueError(
            f"{owner}: inverse kernel failed: spatial.rotation.kernel.inverse: {exc}"
        ) from exc


__all__ = [
    "wrap_compose_quat_kernel",
    "wrap_inverse_quat_kernel",
    "wrap_matrix_to_quat_backend",
    "wrap_quat_to_matrix_backend",
]
