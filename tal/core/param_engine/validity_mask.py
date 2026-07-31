from __future__ import annotations

import numpy as np
import xarray as xr

from .. import validity_values
from .schema_resolve import _resolve_schema_context, _resolve_schema_context_validated
from .types import ParamCoordSpec


def _sequence_index(ds: xr.Dataset, sequence_dim: str) -> xr.DataArray:
    n = int(ds.sizes.get(sequence_dim, 0))
    if sequence_dim in ds.coords and ds.coords[sequence_dim].dims == (sequence_dim,):
        coord = ds.coords[sequence_dim]
    else:
        coord = xr.DataArray(np.arange(n, dtype="int64"), dims=[sequence_dim], name=sequence_dim)
    values = np.arange(n, dtype="int64")
    return xr.DataArray(values, dims=[sequence_dim], coords={sequence_dim: coord})


def _resolved_param_coord(context, spec: ParamCoordSpec) -> xr.DataArray:
    name = context.param_name or spec.name
    if name not in context.ds.coords:
        raise ValueError(
            "resolve_param_valid_mask: resolved param coordinate "
            f"{name!r} not found in dataset coords."
        )
    return context.ds.coords[name]


def _mask_from_sequence_size(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    sequence_size_coord: str,
) -> xr.DataArray:
    size = ds.coords[sequence_size_coord]
    size = validity_values.require_valid_sequence_size_values(
        size,
        sequence_size_coord=sequence_size_coord,
        sequence_len=int(ds.sizes.get(sequence_dim, 0)),
        owner="resolve_param_valid_mask",
    )
    idx = _sequence_index(ds, sequence_dim)
    if not batch_dims:
        size_n = int(np.asarray(size.data, dtype="int64").item())
        return idx < size_n
    expanded = size.expand_dims({sequence_dim: idx.coords[sequence_dim]})
    mask = idx < expanded
    return mask.transpose(*batch_dims, sequence_dim)


def finite_param_mask(param: xr.DataArray) -> xr.DataArray:
    """Build fallback validity mask from finite/not-null parameter values.

    Parameters
    ----------
    param : xr.DataArray
        Parameter-domain input used for temporal evaluation/alignment.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if np.issubdtype(param.dtype, np.number):
        return xr.apply_ufunc(np.isfinite, param, dask="allowed")
    return param.notnull()


def _mask_from_context(context, spec: ParamCoordSpec) -> xr.DataArray:
    if context.sequence_size_coord is None:
        return finite_param_mask(_resolved_param_coord(context, spec))
    return _mask_from_sequence_size(
        context.ds,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        sequence_size_coord=context.sequence_size_coord,
    )


def resolve_param_valid_mask(
    ds: xr.Dataset,
    *,
    spec: ParamCoordSpec,
    sequence_size_coord: str | None = None,
) -> xr.DataArray:
    context = _resolve_schema_context(
        ds,
        explicit_sequence_dim=spec.sequence_dim,
        explicit_batch_dims=spec.batch_dims,
        explicit_param_name=spec.name,
        explicit_sequence_size_coord=sequence_size_coord,
    )
    return _mask_from_context(context, spec)


def _resolve_param_valid_mask_validated(
    ds: xr.Dataset,
    *,
    spec: ParamCoordSpec,
    sequence_size_coord: str | None = None,
) -> xr.DataArray:
    context = _resolve_schema_context_validated(
        ds,
        explicit_sequence_dim=spec.sequence_dim,
        explicit_batch_dims=spec.batch_dims,
        explicit_param_name=spec.name,
        explicit_sequence_size_coord=sequence_size_coord,
    )
    return _mask_from_context(context, spec)


__all__ = ["finite_param_mask", "resolve_param_valid_mask"]
