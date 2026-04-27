from __future__ import annotations

import numpy as np
import xarray as xr

from ..orchestration.lazy import fail_if_chunked_boundary, is_chunked_dataarray
from .types import WhenOptions


def selected_when_mask(
    effective: xr.DataArray,
    *,
    valid_mask: xr.DataArray,
    inside: bool,
) -> xr.DataArray:
    """Return canonical when-selection mask for inside/complement semantics.

    Parameters
    ----------
    effective : xr.DataArray
        Validity/mask payload used by this operation.
    valid_mask : xr.DataArray, optional
        Validity/mask payload used by this operation.
    inside : bool, optional
        Validity/mask payload used by this operation.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if inside:
        return effective.astype(bool)
    return (valid_mask.astype(bool) & ~effective.astype(bool)).astype(bool)


def enforce_when_on_empty(
    selected: xr.DataArray,
    *,
    opts: WhenOptions,
    owner: str,
    layout: str,
    unit: str,
    chunked_message: str,
) -> None:
    """Enforce deterministic ``on_empty='error'`` behavior for when layouts.

    Parameters
    ----------
    selected : xr.DataArray
        Validity/mask payload used by this operation.
    opts : WhenOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.
    layout : str, optional
        Layout/strategy selector controlling output organization.
    unit : str, optional
        Validation/diagnostic metadata used for deterministic error reporting.
    chunked_message : str, optional
        Diagnostic message used when chunked inputs violate boundary constraints.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if opts.on_empty != "error":
        return
    fail_if_chunked_boundary(
        is_chunked_dataarray(selected),
        owner=owner,
        message=chunked_message,
    )
    if not bool(np.any(selected.data)):
        raise ValueError(
            f"{owner}: no selected {unit} for opts.layout={layout!r} and opts.inside={opts.inside!r}."
        )


__all__ = ["enforce_when_on_empty", "selected_when_mask"]
