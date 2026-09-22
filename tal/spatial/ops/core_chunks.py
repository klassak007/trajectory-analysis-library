"""Metadata-only Dask preparation for fixed-size spatial core axes."""

from __future__ import annotations

import xarray as xr

from tal.core.orchestration.lazy import payload_chunks_for_dim


def single_core_chunk(values: xr.DataArray, *, dim: str) -> xr.DataArray:
    """Coalesce one fixed-size core axis without changing outer chunks."""
    chunks = payload_chunks_for_dim(values, dim=dim)
    if chunks is None or len(chunks) <= 1:
        return values
    return values.chunk({dim: -1})


__all__ = ["single_core_chunk"]
