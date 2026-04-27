from __future__ import annotations

import numpy as np
import xarray as xr


def batch_coord(obj: xr.Dataset | xr.DataArray, *, dim: str) -> xr.DataArray:
    if dim in obj.coords and obj.coords[dim].dims == (dim,):
        return obj.coords[dim]
    size = int(obj.sizes.get(dim, 0))
    return xr.DataArray(np.arange(size, dtype="int64"), dims=[dim], name=dim)


__all__ = ["batch_coord"]
