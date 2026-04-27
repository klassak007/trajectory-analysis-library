from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import numpy as np
import xarray as xr

ReducerOp = Literal[
    "mean",
    "sum",
    "std",
    "var",
    "median",
    "min",
    "max",
    "count",
    "any",
    "all",
]

DimLike: TypeAlias = str | Sequence[str] | None
WeightInput: TypeAlias = xr.DataArray | np.ndarray | Mapping[str, xr.DataArray | np.ndarray] | None

NUMERIC_REDUCER_OPS = frozenset({"mean", "sum", "std", "var", "median", "min", "max", "count"})
BOOLEAN_REDUCER_OPS = frozenset({"any", "all"})
SUPPORTED_WEIGHTED_OPS = frozenset({"mean", "sum"})

NUMERIC_KINDS = frozenset({"i", "u", "f", "c"})
NUMERIC_OR_BOOL_KINDS = frozenset({"b", "i", "u", "f", "c"})


def require_supported_op(op: str, *, owner: str) -> ReducerOp:
    allowed = NUMERIC_REDUCER_OPS | BOOLEAN_REDUCER_OPS
    if op in allowed:
        return op  # type: ignore[return-value]
    raise ValueError(f"{owner}: reducer op {op!r} is not supported.")


def weighted_supported(op: ReducerOp) -> bool:
    return op in SUPPORTED_WEIGHTED_OPS


__all__ = [
    "BOOLEAN_REDUCER_OPS",
    "DimLike",
    "NUMERIC_KINDS",
    "NUMERIC_OR_BOOL_KINDS",
    "NUMERIC_REDUCER_OPS",
    "ReducerOp",
    "SUPPORTED_WEIGHTED_OPS",
    "WeightInput",
    "require_supported_op",
    "weighted_supported",
]
