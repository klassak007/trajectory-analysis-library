"""NumPy buffer preparation for SciPy spatial kernels."""

from __future__ import annotations

import numpy as np


def writable_scipy_vectors(values: np.ndarray) -> np.ndarray:
    """Return a writable vector buffer required by ``Rotation.apply``."""
    if values.flags.writeable:
        return values
    return values.copy()


__all__ = ["writable_scipy_vectors"]
