from __future__ import annotations

"""Shared lazy-boundary helpers for orchestration call sites.

This module owns chunked-data preflight checks and fail-fast boundaries used by
orchestration layers in param/combine operations.
"""

from collections.abc import Sequence

import xarray as xr


def is_chunked_dataarray(da: xr.DataArray) -> bool:
    """Return ``True`` when a DataArray is backed by chunked storage.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray value processed by this operation.

    Returns
    -------
    bool
        Boolean result indicating whether the requested condition is satisfied.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return getattr(da.data, "chunks", None) is not None


def is_chunked_variable(var: xr.Variable | xr.DataArray) -> bool:
    """Return ``True`` when a Variable/DataArray is backed by chunks.

    Parameters
    ----------
    var : xr.Variable | xr.DataArray
        Validation/diagnostic metadata used for deterministic error reporting.

    Returns
    -------
    bool
        Boolean result indicating whether the requested condition is satisfied.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return getattr(var.data, "chunks", None) is not None


def fail_if_chunked_boundary(
    condition: bool,
    *,
    owner: str,
    message: str,
) -> None:
    """Raise deterministic TAL-owned boundary errors for chunked paths.

    Parameters
    ----------
    condition : bool
        Condition/expression used for event or mask evaluation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    message : str, optional
        Validation/diagnostic metadata used for deterministic error reporting.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if condition:
        raise ValueError(f"{owner}: {message}")


def require_unchunked_dataarray(
    da: xr.DataArray,
    *,
    owner: str,
    field: str,
    guidance: str,
) -> None:
    """Enforce unchunked input for one boundary field.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray value processed by this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    field : str, optional
        Label/name selection used by this operation.
    guidance : str, optional
        Validation/diagnostic metadata used for deterministic error reporting.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    fail_if_chunked_boundary(
        is_chunked_dataarray(da),
        owner=owner,
        message=f"{field}; {guidance}",
    )


def _resolve_context_field(context: object, *, field: str) -> xr.DataArray:
    value = context
    for part in field.split("."):
        value = getattr(value, part)
    if not isinstance(value, xr.DataArray):
        raise TypeError(f"orchestration.lazy: field {field!r} did not resolve to xr.DataArray.")
    return value


def require_unchunked_auto_grid_sources(
    contexts: Sequence[object],
    *,
    owner: str,
    fields: Sequence[str],
) -> None:
    """Fail fast for auto-grid joins when source param/valid inputs are chunked.

    Parameters
    ----------
    contexts : Sequence[object]
        Resolved runtime context/payload used by this orchestration boundary.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    fields : Sequence[str], optional
        Label/name selection used by this operation.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    for context in contexts:
        for field in fields:
            fail_if_chunked_boundary(
                is_chunked_dataarray(_resolve_context_field(context, field=field)),
                owner=owner,
                message=(
                    "grid=None with join in {'outer','inner','domain','exact'} "
                    "is not supported for chunked param/valid inputs; pass an explicit grid."
                ),
            )


__all__ = [
    "fail_if_chunked_boundary",
    "is_chunked_dataarray",
    "is_chunked_variable",
    "require_unchunked_auto_grid_sources",
    "require_unchunked_dataarray",
]
