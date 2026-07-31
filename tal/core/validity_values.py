"""Schema-agnostic sequence-size value normalization."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

SequenceSizeFailure = Literal[
    "unsupported_dtype",
    "chunked",
    "non_finite",
    "non_integer",
    "out_of_range",
]

_REAL_COUNT_KINDS = frozenset({"i", "u", "f"})


class SequenceSizeValueError(ValueError):
    """Report a normalized sequence-size value-contract failure."""

    def __init__(self, reason: SequenceSizeFailure, detail: str) -> None:
        self.reason = reason
        super().__init__(detail)


def normalize_sequence_size_values(
    size: xr.DataArray,
    *,
    sequence_len: int,
) -> xr.DataArray:
    """Validate eager sequence sizes and return canonical int64 counts.

    Chunked coordinates fail before materialization. Supported values use real
    integer or floating dtypes, are finite and integer-valued, and lie within
    the closed interval from zero through ``sequence_len``.
    """
    dtype = np.dtype(size.dtype)
    if dtype.kind not in _REAL_COUNT_KINDS:
        raise SequenceSizeValueError(
            "unsupported_dtype",
            f"dtype must be a real numeric count dtype; got {str(dtype)!r}",
        )
    if getattr(size.data, "chunks", None) is not None:
        raise SequenceSizeValueError(
            "chunked",
            "chunked sequence_size_coord uses an explicit lazy-safe fail-fast boundary; "
            "compute or rechunk that coordinate explicitly before this operation",
        )
    values = np.asarray(size.data)
    if dtype.kind == "f" and np.any(~np.isfinite(values)):
        raise SequenceSizeValueError("non_finite", "values must be finite")
    if dtype.kind == "f" and np.any(np.trunc(values) != values):
        raise SequenceSizeValueError("non_integer", "values must be integers")
    if np.any(values < 0) or np.any(values > sequence_len):
        raise SequenceSizeValueError(
            "out_of_range",
            f"values must be within [0, {sequence_len}]",
        )
    return size.copy(data=values.astype("int64", copy=False))


def require_valid_sequence_size_values(
    size: xr.DataArray,
    *,
    sequence_size_coord: str,
    sequence_len: int,
    owner: str,
) -> xr.DataArray:
    """Normalize sequence sizes with an operation-owned error envelope."""
    try:
        return normalize_sequence_size_values(size, sequence_len=sequence_len)
    except SequenceSizeValueError as exc:
        raise ValueError(
            f"{owner}: invalid sequence_size_coord {sequence_size_coord!r}: {exc}."
        ) from exc


__all__ = [
    "SequenceSizeFailure",
    "SequenceSizeValueError",
    "normalize_sequence_size_values",
    "require_valid_sequence_size_values",
]
