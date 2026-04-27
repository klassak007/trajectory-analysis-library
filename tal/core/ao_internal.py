from __future__ import annotations

from collections.abc import Mapping

import xarray as xr


def finalize_structural(
    ao,
    ds: xr.Dataset,
    *,
    validate: bool,
    rename_map: Mapping[str, str] | None = None,
):
    """Centralized wrapper for AO structural finalization internals.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    rename_map : Mapping[str, str] | None, optional
        Injective mapping used to rewrite identifiers/labels.

    Returns
    -------
    object
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return ao._finalize_structural(ds, validate=validate, rename_map=rename_map)


def from_unvalidated_like(ao, ds: xr.Dataset):
    """Centralized wrapper for AO unvalidated rewrap internals.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    ds : xr.Dataset
        Input dataset/source value processed by this operation.

    Returns
    -------
    object
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return ao.__class__._from_unvalidated(ds)


__all__ = ["finalize_structural", "from_unvalidated_like"]
