from __future__ import annotations

import numpy as np

from tal.utils.block_rows import BlockInputSpec, BlockRows, prepare_block_rows

from .rotation_mean_kernels import quat_mean_kernel

ROTATION_MEAN_BACKEND_NUMBA = "numba"
ROTATION_MEAN_BACKEND_NUMPY = "numpy"
_OWNER = "spatial.rotation.mean_backend"
_QUAT_SIZE = 4


def _as_float_array(value: object, *, name: str, owner: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{owner}: {name} could not be coerced to float64.") from exc


def _broadcast_weights(weights: object, *, target_shape: tuple[int, ...], owner: str) -> np.ndarray:
    weight = _as_float_array(weights, name="weights", owner=owner)
    try:
        return np.ascontiguousarray(np.broadcast_to(weight, target_shape))
    except ValueError as exc:
        raise ValueError(f"{owner}: weights must be broadcastable to {target_shape!r}.") from exc


def prepare_quat_mean_rows(values: object, weights: object, *, owner: str) -> BlockRows:
    array = _as_float_array(values, name="values", owner=owner)
    if array.ndim < 2:
        raise ValueError(f"{owner}: values must include reduce and quaternion dimensions.")
    if array.shape[-1] != _QUAT_SIZE:
        raise ValueError(f"{owner}: values must have trailing quaternion dim length 4.")
    weight = _broadcast_weights(weights, target_shape=tuple(int(size) for size in array.shape[:-1]), owner=owner)
    return prepare_block_rows(
        (array, weight),
        (BlockInputSpec("values", 2, np.float64), BlockInputSpec("weights", 1, np.float64)),
        output_core_shape=(_QUAT_SIZE,),
        owner=owner,
    )


def quat_mean_block_backend(
    values: object,
    weights: object,
    *,
    backend: str = ROTATION_MEAN_BACKEND_NUMPY,
) -> np.ndarray:
    if backend == ROTATION_MEAN_BACKEND_NUMPY:
        prepare_quat_mean_rows(values, weights, owner=_OWNER)
        return quat_mean_kernel(np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64))
    if backend == ROTATION_MEAN_BACKEND_NUMBA:
        from .rotation_mean_numba_backends import quat_mean_block_numba

        return quat_mean_block_numba(values, weights, owner=_OWNER)
    raise ValueError(f"{_OWNER}: unsupported rotation mean backend {backend!r}.")


__all__ = [
    "ROTATION_MEAN_BACKEND_NUMBA",
    "ROTATION_MEAN_BACKEND_NUMPY",
    "prepare_quat_mean_rows",
    "quat_mean_block_backend",
]
