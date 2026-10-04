from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ResolvedReducerRequest:
    """Metadata-only reducer request resolved before numerical execution."""

    reducer: ReducerOp
    reduce_dims: tuple[str, ...]
    eligible_names: tuple[str, ...]
    active_reduce_dims: tuple[str, ...]

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
    "NUMERIC_KINDS",
    "NUMERIC_OR_BOOL_KINDS",
    "NUMERIC_REDUCER_OPS",
    "SUPPORTED_WEIGHTED_OPS",
    "DimLike",
    "ReducerOp",
    "ResolvedReducerRequest",
    "WeightInput",
    "require_supported_op",
    "weighted_supported",
]
