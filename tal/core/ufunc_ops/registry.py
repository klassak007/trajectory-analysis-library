from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import xarray as xr

UfuncCallable = Callable[..., object]
UfuncFamily = Literal["arithmetic_binary", "comparison", "logical", "passthrough"]

_WRAPPER_TYPE_NAMES = frozenset({"_unary_ufunc", "_binary_ufunc"})
_COMPARISON_UFUNCS = frozenset({"equal", "not_equal", "greater", "greater_equal", "less", "less_equal"})
_LOGICAL_CONDITION_UFUNCS = frozenset({"logical_and", "logical_or", "logical_xor", "logical_not"})
_ARITHMETIC_BINARY_UFUNCS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "true_divide",
        "floor_divide",
        "mod",
        "remainder",
        "power",
        "pow",
        "float_power",
    }
)


def _is_xarray_ufunc_wrapper(value: object) -> bool:
    return type(value).__name__ in _WRAPPER_TYPE_NAMES


def _collect_xarray_ufuncs() -> dict[str, UfuncCallable]:
    out: dict[str, UfuncCallable] = {}
    for name in dir(xr.ufuncs):
        if name.startswith("_"):
            continue
        value = getattr(xr.ufuncs, name)
        if _is_xarray_ufunc_wrapper(value):
            out[name] = value
    return out


XARRAY_UFUNCS = _collect_xarray_ufuncs()
UNARY_UFUNC_NAMES = tuple(sorted(name for name, fn in XARRAY_UFUNCS.items() if type(fn).__name__ == "_unary_ufunc"))
BINARY_UFUNC_NAMES = tuple(
    sorted(name for name, fn in XARRAY_UFUNCS.items() if type(fn).__name__ == "_binary_ufunc")
)


def get_xarray_ufunc(name: str, *, owner: str) -> UfuncCallable:
    try:
        return XARRAY_UFUNCS[name]
    except KeyError as exc:
        raise ValueError(f"{owner}: unknown xarray ufunc {name!r}.") from exc


def classify_ufunc_family(name: str, *, owner: str) -> UfuncFamily:
    _ = get_xarray_ufunc(name, owner=owner)
    if name in _COMPARISON_UFUNCS:
        return "comparison"
    if name in _LOGICAL_CONDITION_UFUNCS:
        return "logical"
    if name in _ARITHMETIC_BINARY_UFUNCS:
        return "arithmetic_binary"
    return "passthrough"


def all_registered_ufunc_names() -> tuple[str, ...]:
    return tuple(sorted(XARRAY_UFUNCS))


__all__ = [
    "BINARY_UFUNC_NAMES",
    "UfuncCallable",
    "UfuncFamily",
    "UNARY_UFUNC_NAMES",
    "XARRAY_UFUNCS",
    "all_registered_ufunc_names",
    "classify_ufunc_family",
    "get_xarray_ufunc",
]
