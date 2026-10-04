from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .fixed_size_backends import (
    SPATIAL_FIXED_BACKEND_SCIPY,
    pose_compose_translation_block_backend,
    pose_inverse_translation_block_backend,
    quat_compose_block_backend,
    quat_inverse_block_backend,
)
from .rotation_interp_backends import ROTATION_INTERP_BACKEND_SCIPY, slerp_quat_backend
from .rotation_interp_blocks import SLERP_BLOCK_SIZE as _QUERY_BLOCK_SIZE


@dataclass(frozen=True)
class _MapArrays:
    i0: np.ndarray
    i1: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray


def _interpolate_translation(
    values: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    left = values[i0]
    right = values[i1]
    return (1.0 - alpha[:, None]) * left + alpha[:, None] * right


def _interpolate_quaternion(
    values: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    return slerp_quat_backend(
        values[i0][None, :, :],
        values[i1][None, :, :],
        alpha[None, :],
        valid[None, :],
        backend=ROTATION_INTERP_BACKEND_SCIPY,
    )[0]


def _orient_block(
    translation: np.ndarray,
    quaternion: np.ndarray,
    *,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    if direction == 1:
        return translation, quaternion
    if direction != -1:
        raise ValueError("spatial.path_solve.pose: path direction must be +1 or -1.")
    inverse_q = quat_inverse_block_backend(
        quaternion,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    inverse_t = pose_inverse_translation_block_backend(
        translation,
        quaternion,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    return inverse_t, inverse_q


def _compose_block(
    accumulated_t: np.ndarray,
    accumulated_q: np.ndarray,
    edge_t: np.ndarray,
    edge_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    translation = pose_compose_translation_block_backend(
        accumulated_t,
        edge_t,
        edge_q,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    quaternion = quat_compose_block_backend(
        accumulated_q,
        edge_q,
        backend=SPATIAL_FIXED_BACKEND_SCIPY,
    )
    return translation, quaternion


def _validate_inputs(
    translations: Sequence[np.ndarray],
    quaternions: Sequence[np.ndarray],
    directions: Sequence[int],
) -> None:
    if not translations or len(translations) != len(quaternions) or len(translations) != len(directions):
        raise ValueError("spatial.path_solve.pose: streaming edge inputs must have one value per path step.")
    for translation, quaternion in zip(translations, quaternions, strict=True):
        if translation.ndim != 2 or translation.shape[1] != 3:
            raise ValueError("spatial.path_solve.pose: streaming translation must have shape (sample, 3).")
        if quaternion.ndim != 2 or quaternion.shape[1] != 4:
            raise ValueError("spatial.path_solve.pose: streaming quaternion must have shape (sample, 4).")
        if translation.shape[0] != quaternion.shape[0]:
            raise ValueError("spatial.path_solve.pose: streaming component sample sizes must match.")


def _stream_edge(
    translation: np.ndarray,
    quaternion: np.ndarray,
    *,
    direction: int,
    edge: int,
    mapping: _MapArrays,
    out_t: np.ndarray,
    out_q: np.ndarray,
) -> None:
    query_size = int(mapping.valid.size)
    max_index = int(translation.shape[0] - 1)
    for start in range(0, query_size, _QUERY_BLOCK_SIZE):
        stop = min(start + _QUERY_BLOCK_SIZE, query_size)
        active = np.asarray(mapping.valid[start:stop], dtype=bool)
        if not np.any(active):
            continue
        selected = np.flatnonzero(active)
        left = np.clip(mapping.i0[start:stop][selected], 0, max_index)
        right = np.clip(mapping.i1[start:stop][selected], 0, max_index)
        fraction = mapping.alpha[start:stop][selected]
        edge_t = _interpolate_translation(translation, left, right, fraction)
        edge_q = _interpolate_quaternion(quaternion, left, right, fraction, np.ones(selected.size, dtype=bool))
        edge_t, edge_q = _orient_block(edge_t, edge_q, direction=direction)
        target = start + selected
        if edge == 0:
            out_t[target], out_q[target] = edge_t, edge_q
        else:
            out_t[target], out_q[target] = _compose_block(out_t[target], out_q[target], edge_t, edge_q)


def stream_pose_path_blocks(
    translations: Sequence[np.ndarray],
    quaternions: Sequence[np.ndarray],
    directions: Sequence[int],
    *,
    i0: np.ndarray,
    i1: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate and fold one eager path with bounded edge-local state."""
    _validate_inputs(translations, quaternions, directions)
    query_size = int(valid.size)
    out_t = np.full((query_size, 3), np.nan, dtype=np.float64)
    out_q = np.full((query_size, 4), np.nan, dtype=np.float64)
    mapping = _MapArrays(i0, i1, alpha, valid)
    for edge, (translation, quaternion) in enumerate(zip(translations, quaternions, strict=True)):
        _stream_edge(
            translation,
            quaternion,
            direction=int(directions[edge]),
            edge=edge,
            mapping=mapping,
            out_t=out_t,
            out_q=out_q,
        )
    return out_t, out_q


__all__ = ["stream_pose_path_blocks"]
