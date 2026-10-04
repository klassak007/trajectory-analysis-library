from __future__ import annotations

import numpy as np


def is_monotonic(values: np.ndarray, *, order: str) -> bool:
    """Return whether numeric values satisfy the adapter time-order policy."""
    if values.size <= 1:
        return True
    if order == "strict":
        return bool(np.all(values[1:] > values[:-1]))
    return bool(np.all(values[1:] >= values[:-1]))


__all__ = ["is_monotonic"]
