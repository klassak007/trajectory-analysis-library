from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from ..param_engine import ParamMapOptions, apply_param_map, build_param_map, normalize_query_grid
from .finalize import finalize_param_output
from .guards import assert_query_dim_safe, assert_reserved_metadata_safe, mark_reserved_coord
from .types import ParamEvalOptions, ParamRuntimeContext


def _apply_map_dataset(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    param_map,
) -> xr.Dataset:
    out_vars: dict[str, xr.DataArray] = {}
    for name, var in ds.data_vars.items():
        if sequence_dim not in var.dims:
            out_vars[str(name)] = var
            continue
        if not np.issubdtype(np.dtype(var.dtype), np.number):
            raise TypeError(f"param at/resample: non-numeric sequence variable {name!r} is not supported.")
        out_vars[str(name)] = apply_param_map(var, param_map=param_map, sequence_dim=sequence_dim)
    base_coords = {cname: coord for cname, coord in ds.coords.items() if sequence_dim not in coord.dims}
    return xr.Dataset(data_vars=out_vars, coords=base_coords)


def evaluate_param(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamEvalOptions,
    validate: bool,
) -> "AnalysisObject":
    """Evaluate AO data on a parameter query grid.

    Parameters
    ----------
    context : ParamRuntimeContext
        Resolved runtime context/payload used by this orchestration boundary.
    query : xr.DataArray | np.ndarray | Sequence[float] | float, optional
        Query coordinate/grid used for parameter evaluation.
    opts : ParamEvalOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    assert_query_dim_safe(
        context.ds,
        sequence_dim=context.sequence_dim,
        query_dim=opts.query_dim,
        owner="param at/resample",
    )
    assert_reserved_metadata_safe(
        context.ds,
        reserved=("valid", "sample_index"),
        param_name=context.spec.name,
        owner="param at/resample",
    )
    grid = normalize_query_grid(
        query,
        query_dim=opts.query_dim,
        batch_dims=context.batch_dims,
        batch_coords=context.batch_coords,
        param_kind=context.param_kind,
    )
    pmap = build_param_map(
        param=context.spec.coord,
        query=grid.values,
        sequence_dim=context.sequence_dim,
        query_dim=grid.query_dim,
        valid_mask=context.valid_mask,
        options=ParamMapOptions(method=opts.method, duplicate_policy=opts.duplicate_policy),
        param_kind=context.param_kind,
    )
    ds_out = _apply_map_dataset(context.ds, sequence_dim=context.sequence_dim, param_map=pmap)
    ds_out = ds_out.assign_coords({"valid": mark_reserved_coord(pmap.valid, name="valid")})
    return finalize_param_output(
        context,
        ds_out,
        query=grid.values,
        query_dim=opts.query_dim,
        valid_query=pmap.valid,
        validate=validate,
        trajectory=(grid.stacked_dims is None),
    )


__all__ = ["evaluate_param"]
