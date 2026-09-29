from __future__ import annotations

import numpy as np
import xarray as xr

from ..ordered_dtypes import is_ordered_real_numeric_dtype
from ..validity_mask import resolve_structural_valid_mask_base
from .schema_resolve import _resolve_schema_context, _resolve_schema_context_validated
from .types import ParamCoordSpec


def _validate_param_mask_dtype(param: xr.DataArray, *, owner: str) -> bool:
    if is_ordered_real_numeric_dtype(param.dtype):
        return True
    if np.issubdtype(np.dtype(param.dtype), np.datetime64):
        return False
    raise ValueError(
        f"{owner}: param coordinate must have an ordered real numeric or datetime64 dtype, "
        f"got {param.dtype!r}."
    )


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
    mask = resolve_structural_valid_mask_base(
        ds,
        sequence_dim=sequence_dim,
        sequence_size_coord=sequence_size_coord,
        owner="resolve_param_valid_mask",
    )
    assert mask is not None
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
    if _validate_param_mask_dtype(param, owner="finite_param_mask"):
        return xr.apply_ufunc(np.isfinite, param, dask="allowed")
    return param.notnull()


def _mask_from_context(context, spec: ParamCoordSpec) -> xr.DataArray:
    param = _resolved_param_coord(context, spec)
    if context.sequence_size_coord is None:
        return finite_param_mask(param)
    _validate_param_mask_dtype(param, owner="resolve_param_valid_mask")
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
    allow_declared_param_override: bool = False,
) -> xr.DataArray:
    context = _resolve_schema_context_validated(
        ds,
        explicit_sequence_dim=spec.sequence_dim,
        explicit_batch_dims=spec.batch_dims,
        explicit_param_name=spec.name,
        explicit_sequence_size_coord=sequence_size_coord,
        allow_declared_param_override=allow_declared_param_override,
    )
    return _mask_from_context(context, spec)


__all__ = ["finite_param_mask", "resolve_param_valid_mask"]
