from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from tal.utils.numba_support import _numba_available

from .backends import (
    _PARAM_BACKEND_AUTO,
    PARAM_BOUNDS_BACKEND_NUMBA,
    PARAM_BOUNDS_BACKEND_NUMPY_BLOCK,
    PARAM_MAP_BACKEND_NUMBA,
    PARAM_MAP_BACKEND_NUMPY_BLOCK,
)
from .types import ParamMapOptions


@dataclass(frozen=True)
class _ParamMapExecutionOptions(ParamMapOptions):
    backend: str | None = None


def _with_map_backend(
    options: ParamMapOptions,
    backend: str | None,
) -> ParamMapOptions:
    if backend is None:
        return options
    return _ParamMapExecutionOptions(
        method=options.method,
        duplicate_policy=options.duplicate_policy,
        backend=backend,
    )


def _param_map_backend(options: ParamMapOptions) -> str | None:
    return options.backend if isinstance(options, _ParamMapExecutionOptions) else None


def _float_numba_compatible(*values: xr.DataArray) -> bool:
    return all(
        np.dtype(value.dtype).kind == "f" and np.dtype(value.dtype).itemsize in {4, 8}
        for value in values
    )


def select_map_backend(
    param: xr.DataArray,
    valid: xr.DataArray,
    query: xr.DataArray,
    requested: str | None,
) -> str:
    lazy = any(value.chunks is not None for value in (param, valid, query))
    if requested == _PARAM_BACKEND_AUTO:
        return _PARAM_BACKEND_AUTO if lazy else PARAM_MAP_BACKEND_NUMPY_BLOCK
    if requested is not None:
        return requested
    if lazy:
        return _PARAM_BACKEND_AUTO
    if _float_numba_compatible(param, query) and _numba_available():
        return PARAM_MAP_BACKEND_NUMBA
    return PARAM_MAP_BACKEND_NUMPY_BLOCK


def select_bounds_backend(
    param: xr.DataArray,
    valid: xr.DataArray,
    start: xr.DataArray,
    stop: xr.DataArray,
) -> str:
    if any(value.chunks is not None for value in (param, valid, start, stop)):
        return _PARAM_BACKEND_AUTO
    if _float_numba_compatible(param, start, stop) and _numba_available():
        return PARAM_BOUNDS_BACKEND_NUMBA
    return PARAM_BOUNDS_BACKEND_NUMPY_BLOCK


__all__: list[str] = []
