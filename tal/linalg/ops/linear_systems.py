from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import xarray as xr

from ...core.orchestration.lazy import fail_if_chunked_boundary, is_chunked_dataarray
from ..plan import ArrayOperandContext

T = TypeVar("T")


def enforce_unchunked_linear_system_inputs(
    *arrays: xr.DataArray,
    owner: str,
    message: str,
) -> None:
    chunked = any(is_chunked_dataarray(arr) for arr in arrays)
    fail_if_chunked_boundary(chunked, owner=owner, message=message)


def require_matrix_core_dims(
    op: ArrayOperandContext,
    *,
    owner: str,
    operand_label: str = "left operand",
) -> tuple[str, str]:
    if len(op.core_dims) != 2:
        raise ValueError(f"{owner}: {operand_label} must have exactly two core dims; got {op.core_dims!r}.")
    row_dim, col_dim = op.core_dims
    if row_dim == col_dim:
        raise ValueError(f"{owner}: {operand_label} core dims must be distinct; got {op.core_dims!r}.")
    return row_dim, col_dim


def require_square_matrix(
    op: ArrayOperandContext,
    *,
    row_dim: str,
    col_dim: str,
    owner: str,
    message: str | None = None,
) -> None:
    row_size = op.data.sizes[row_dim]
    col_size = op.data.sizes[col_dim]
    if row_size == col_size:
        return
    if message is None:
        message = (
            f"{owner}: matrix must be square on dims {row_dim!r}, {col_dim!r}; "
            f"got sizes {row_size} and {col_size}."
        )
    raise ValueError(message)


def run_with_linalgerror_normalization(
    fn: Callable[[], T],
    *,
    owner: str,
    message: str,
) -> T:
    try:
        return fn()
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{owner}: {message}") from exc


__all__ = [
    "enforce_unchunked_linear_system_inputs",
    "require_matrix_core_dims",
    "require_square_matrix",
    "run_with_linalgerror_normalization",
]
