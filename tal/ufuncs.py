"""AO-aware wrappers for xarray/NumPy universal functions.

The functions in this module accept ordinary scalar/array inputs as well as
``AnalysisObject`` operands. When an AO participates in the operation, TAL
preserves the wrapped dataset schema where the resulting dimensional structure
still supports it.

Examples
--------
>>> import xarray as xr
>>> from tal.core import AnalysisObject
>>> from tal import ufuncs
>>> ao = AnalysisObject.from_data(
...     xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
...     sequence_dim="sample",
...     core_dims=(),
...     validate=True,
... )
>>> out = ufuncs.add(ao, 1.0)
>>> out.as_dataset()["value"].values.tolist()
[2.0, 3.0]
"""

from __future__ import annotations

from tal.core.ufunc_ops import BINARY_UFUNC_NAMES, UNARY_UFUNC_NAMES, apply_binary_ufunc, apply_unary_ufunc


def _build_unary_wrapper(name: str):
    def wrapper(x: object) -> object:
        return apply_unary_ufunc(name, x, owner=f"tal.ufuncs.{name}")

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = f"TAL AO-aware wrapper for ``xarray.ufuncs.{name}``."
    return wrapper


def _build_binary_wrapper(name: str):
    def wrapper(x1: object, x2: object) -> object:
        return apply_binary_ufunc(name, x1, x2, owner=f"tal.ufuncs.{name}")

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = f"TAL AO-aware wrapper for ``xarray.ufuncs.{name}``."
    return wrapper


for _name in UNARY_UFUNC_NAMES:
    globals()[_name] = _build_unary_wrapper(_name)
for _name in BINARY_UFUNC_NAMES:
    globals()[_name] = _build_binary_wrapper(_name)

__all__ = list(UNARY_UFUNC_NAMES) + list(BINARY_UFUNC_NAMES)
