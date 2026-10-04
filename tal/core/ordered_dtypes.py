"""Neutral dtype predicates for ordered real numeric domains."""

from __future__ import annotations

import numpy as np

ORDERED_REAL_NUMERIC_KINDS = frozenset({"i", "u", "f"})


def is_ordered_real_numeric_dtype(dtype: object) -> bool:
    """Return whether ``dtype`` has total-order parameter semantics."""
    try:
        resolved = np.dtype(dtype)
    except TypeError:
        return False
    return resolved.kind in ORDERED_REAL_NUMERIC_KINDS


def is_integral_dtype(dtype: object) -> bool:
    """Return whether ``dtype`` is a signed or unsigned integer dtype."""
    try:
        resolved = np.dtype(dtype)
    except TypeError:
        return False
    return resolved.kind in {"i", "u"}


def is_float64_exact_integer(value: object) -> bool:
    """Return whether an integer value round-trips exactly through float64."""
    integer = int(value)
    converted = float(integer)
    return np.isfinite(converted) and int(converted) == integer


__all__ = [
    "ORDERED_REAL_NUMERIC_KINDS",
    "is_float64_exact_integer",
    "is_integral_dtype",
    "is_ordered_real_numeric_dtype",
]
