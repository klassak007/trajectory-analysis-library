from __future__ import annotations

from numbers import Real

import numpy as np

from ..plan import ArrayOperandContext


def require_vector_core_dim(
    op: ArrayOperandContext,
    *,
    owner: str,
    operand_label: str,
) -> str:
    if len(op.core_dims) != 1:
        raise ValueError(
            f"{owner}: {operand_label} must have exactly one core dim; got {op.core_dims!r}."
        )
    return op.core_dims[0]


def require_matching_vector_core_dim(
    left_op: ArrayOperandContext,
    right_op: ArrayOperandContext,
    *,
    owner: str,
) -> str:
    left_dim = require_vector_core_dim(left_op, owner=owner, operand_label="left operand")
    right_dim = require_vector_core_dim(right_op, owner=owner, operand_label="right operand")
    if left_dim != right_dim:
        raise ValueError(
            f"{owner}: dot requires matching vector core dim names; got {left_dim!r} and {right_dim!r}."
        )
    return left_dim


def coerce_norm_ord(value: int | float | None, *, owner: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{owner}: ord must be int | float | None.")
    coerced: int | float = value
    if isinstance(value, np.integer):
        coerced = int(value)
    elif isinstance(value, np.floating):
        coerced = float(value)
    if np.isnan(float(coerced)):
        raise ValueError(f"{owner}: ord must not be NaN.")
    return coerced


__all__ = [
    "coerce_norm_ord",
    "require_matching_vector_core_dim",
    "require_vector_core_dim",
]
