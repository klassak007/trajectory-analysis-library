from __future__ import annotations

import numpy as np

EVENT_BOUNDARY_BACKEND_NUMPY_ROW = "numpy_row"
EVENT_INTERVALS_BACKEND_NUMPY_ROW = "numpy_row"


def boundary_bounded_row_backend(
    mask_row: np.ndarray,
    valid_row: np.ndarray,
    clock_row: np.ndarray,
    *,
    include_initial: bool,
    emit_triggers: bool,
    dedupe_atol: float,
    max_events: int,
    owner: str,
    backend: str = EVENT_BOUNDARY_BACKEND_NUMPY_ROW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if backend != EVENT_BOUNDARY_BACKEND_NUMPY_ROW:
        raise ValueError(f"{owner}: unsupported boundary backend {backend!r}.")
    from .boundary import _bounded_row_kernel

    return _bounded_row_kernel(
        mask_row,
        valid_row,
        clock_row,
        include_initial=include_initial,
        emit_triggers=emit_triggers,
        dedupe_atol=dedupe_atol,
        max_events=max_events,
        owner=owner,
    )


def intervals_bounded_row_backend(
    time_row: np.ndarray,
    edge_row: np.ndarray,
    before_row: np.ndarray,
    after_row: np.ndarray,
    *,
    max_segments: int,
    owner: str,
    backend: str = EVENT_INTERVALS_BACKEND_NUMPY_ROW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if backend != EVENT_INTERVALS_BACKEND_NUMPY_ROW:
        raise ValueError(f"{owner}: unsupported intervals backend {backend!r}.")
    from .intervals import _bounded_row_kernel

    return _bounded_row_kernel(
        time_row,
        edge_row,
        before_row,
        after_row,
        max_segments=max_segments,
        owner=owner,
    )


__all__ = [
    "EVENT_BOUNDARY_BACKEND_NUMPY_ROW",
    "EVENT_INTERVALS_BACKEND_NUMPY_ROW",
    "boundary_bounded_row_backend",
    "intervals_bounded_row_backend",
]
