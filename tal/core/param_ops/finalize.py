from __future__ import annotations

import numpy as np
import xarray as xr

from ..dataset_ownership import analysis_object_dataset
from ..ao_internal import finalize_structural
from ..orchestration.finalize import transfer_dataset_attrs
from ..validity_finalize import assign_sequence_size_from_valid_mask
from .types import ParamRuntimeContext


def _positional_coord(size: int) -> np.ndarray:
    return np.arange(int(size), dtype="int64")


def _rename_query_dim(
    da: xr.DataArray,
    *,
    query_dim: str,
    sequence_dim: str,
) -> xr.DataArray:
    if query_dim in da.dims and query_dim != sequence_dim:
        return da.rename({query_dim: sequence_dim})
    return da


def _query_attach_values(
    query: xr.DataArray,
    *,
    query_dim: str,
    sequence_dim: str,
    sequence_size: int,
) -> xr.DataArray:
    out = _rename_query_dim(query.reset_coords(drop=True), query_dim=query_dim, sequence_dim=sequence_dim)
    if sequence_dim not in out.dims:
        return out
    if int(out.sizes[sequence_dim]) != int(sequence_size):
        raise ValueError(
            "finalize_param_output: query length does not match output sequence length after normalization."
        )
    out = out.drop_vars(sequence_dim, errors="ignore")
    return out.assign_coords({sequence_dim: _positional_coord(sequence_size)})


def finalize_param_output(
    context: ParamRuntimeContext,
    ds: xr.Dataset,
    *,
    query: xr.DataArray | None,
    query_dim: str,
    valid_query: xr.DataArray | None,
    validate: bool,
    trajectory: bool,
) -> "AnalysisObject":
    ds_out = transfer_dataset_attrs(context.ds, ds, validate=False)
    if query is not None and query_dim in ds_out.dims and not trajectory:
        ds_out = ds_out.unstack(query_dim)
    if trajectory and query_dim in ds_out.dims and query_dim != context.sequence_dim:
        if context.sequence_dim in ds_out.coords and context.sequence_dim not in ds_out.dims:
            ds_out = ds_out.drop_vars(context.sequence_dim, errors="ignore")
        ds_out = ds_out.rename({query_dim: context.sequence_dim})
    if trajectory and context.sequence_dim in ds_out.dims:
        ds_out = ds_out.assign_coords({context.sequence_dim: _positional_coord(ds_out.sizes[context.sequence_dim])})
    if query is not None and trajectory:
        q = _query_attach_values(
            query,
            query_dim=query_dim,
            sequence_dim=context.sequence_dim,
            sequence_size=int(ds_out.sizes.get(context.sequence_dim, 0)),
        )
        ds_out = ds_out.assign_coords({context.spec.name: q})
    if trajectory and valid_query is not None and context.sequence_size_coord:
        v = _rename_query_dim(valid_query, query_dim=query_dim, sequence_dim=context.sequence_dim)
        ds_out, _ = assign_sequence_size_from_valid_mask(
            ds_out,
            valid=v,
            sequence_dim=context.sequence_dim,
            batch_dims=context.batch_dims,
            sequence_size_coord=context.sequence_size_coord,
        )
    out = finalize_structural(context.ao, ds_out, validate=validate)
    if not trajectory:
        core = analysis_object_dataset(out).attrs.get("tal", {}).get("core", {})
        if isinstance(core, dict) and "validity" in core:
            out = out.set_validity(sequence_size_coord=None, validate=validate)
    return out


__all__ = [
    "finalize_param_output",
]
