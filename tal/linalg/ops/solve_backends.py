from __future__ import annotations

import numpy as np

LSTSQ_BACKEND_NUMPY_ROW = "numpy_row"


def _lstsq_numpy_row(a: np.ndarray, b: np.ndarray, *, rcond: float | None) -> np.ndarray:
    return np.linalg.lstsq(a, b, rcond=rcond)[0]


def lstsq_solution_backend(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rcond: float | None,
    backend: str = LSTSQ_BACKEND_NUMPY_ROW,
) -> np.ndarray:
    if backend != LSTSQ_BACKEND_NUMPY_ROW:
        raise ValueError(f"linalg.solve: unsupported lstsq backend {backend!r}.")
    return _lstsq_numpy_row(a, b, rcond=rcond)


__all__ = ["LSTSQ_BACKEND_NUMPY_ROW", "lstsq_solution_backend"]
