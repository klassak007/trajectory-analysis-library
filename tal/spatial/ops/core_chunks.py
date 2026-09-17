"""Metadata-only Dask preparation for fixed-size spatial core axes."""

from __future__ import annotations

import xarray as xr


def single_core_chunk(values: xr.DataArray, *, dim: str) -> xr.DataArray:
    """Coalesce one fixed-size core axis without changing outer chunks."""
    if values.chunks is None or len(values.chunksizes[dim]) <= 1:
        return values
    return values.chunk({dim: -1})


__all__ = ["single_core_chunk"]
