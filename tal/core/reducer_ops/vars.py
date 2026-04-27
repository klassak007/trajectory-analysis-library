from __future__ import annotations

import xarray as xr

from .types import BOOLEAN_REDUCER_OPS, NUMERIC_KINDS, NUMERIC_OR_BOOL_KINDS, ReducerOp


def _kind_allowed(kind: str, *, op: ReducerOp) -> bool:
    if op in BOOLEAN_REDUCER_OPS:
        return kind in NUMERIC_OR_BOOL_KINDS
    return kind in NUMERIC_KINDS


def select_eligible_var_names(ds: xr.Dataset, *, op: ReducerOp, owner: str) -> tuple[str, ...]:
    names = tuple(name for name, da in ds.data_vars.items() if _kind_allowed(da.dtype.kind, op=op))
    if names:
        return names
    if op in BOOLEAN_REDUCER_OPS:
        raise ValueError(f"{owner}: reducer {op!r} requires at least one numeric or boolean data variable.")
    raise ValueError(f"{owner}: reducer {op!r} requires at least one numeric data variable.")


__all__ = ["select_eligible_var_names"]
