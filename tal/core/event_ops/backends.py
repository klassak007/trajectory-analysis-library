from __future__ import annotations

import numpy as np

EVENT_BOUNDARY_BACKEND_NUMPY_BLOCK = "numpy_block"
EVENT_INTERVALS_BACKEND_NUMPY_BLOCK = "numpy_block"
EVENT_BOUNDARY_BACKEND_NUMBA = "numba"
EVENT_INTERVALS_BACKEND_NUMBA = "numba"


def boundary_bounded_block_backend(
    mask_block: np.ndarray,
    valid_block: np.ndarray,
    clock_block: np.ndarray,
    *,
    include_initial: bool,
    emit_triggers: bool,
    dedupe_atol: float,
    max_events: int,
    owner: str,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if backend == EVENT_BOUNDARY_BACKEND_NUMPY_BLOCK:
        from .numpy_backends import boundary_bounded_block_numpy

        return boundary_bounded_block_numpy(
            mask_block,
            valid_block,
            clock_block,
            include_initial=include_initial,
            emit_triggers=emit_triggers,
            dedupe_atol=dedupe_atol,
            max_events=max_events,
            owner=owner,
        )
    if backend == EVENT_BOUNDARY_BACKEND_NUMBA:
        from .numba_backends import boundary_bounded_block_numba

        return boundary_bounded_block_numba(
            mask_block,
            valid_block,
            clock_block,
            include_initial=include_initial,
            emit_triggers=emit_triggers,
            dedupe_atol=dedupe_atol,
            max_events=max_events,
            owner=owner,
        )
    raise ValueError(f"{owner}: unsupported boundary backend {backend!r}.")


def intervals_bounded_block_backend(
    time_block: np.ndarray,
    edge_block: np.ndarray,
    before_block: np.ndarray,
    after_block: np.ndarray,
    *,
    max_segments: int,
    owner: str,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if backend == EVENT_INTERVALS_BACKEND_NUMPY_BLOCK:
        from .numpy_backends import intervals_bounded_block_numpy

        return intervals_bounded_block_numpy(
            time_block,
            edge_block,
            before_block,
            after_block,
            max_segments=max_segments,
            owner=owner,
        )
    if backend == EVENT_INTERVALS_BACKEND_NUMBA:
        from .numba_backends import intervals_bounded_block_numba

        return intervals_bounded_block_numba(
            time_block,
            edge_block,
            before_block,
            after_block,
            max_segments=max_segments,
            owner=owner,
        )
    raise ValueError(f"{owner}: unsupported intervals backend {backend!r}.")


__all__ = [
    "EVENT_BOUNDARY_BACKEND_NUMBA",
    "EVENT_BOUNDARY_BACKEND_NUMPY_BLOCK",
    "EVENT_INTERVALS_BACKEND_NUMBA",
    "EVENT_INTERVALS_BACKEND_NUMPY_BLOCK",
    "boundary_bounded_block_backend",
    "intervals_bounded_block_backend",
]
