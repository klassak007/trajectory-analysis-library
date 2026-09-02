from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from ..dataset_ownership import analysis_object_dataset
from ..ao_internal import finalize_structural
from ..param_engine import build_param_bounds_map
from ..param_engine.map_apply import gather_along_sequence, gather_dataset_along_sequence
from ..validity_layout import sequence_size_from_mask
from ..validity_layout import scalar_int_boundary
from .finalize import finalize_param_output
from .guards import (
    assert_query_dim_safe,
    assert_reserved_metadata_safe,
    dataset_namespace_names,
    mark_reserved_coord,
    unique_temp_dim,
)
from .index import build_index_result
from .options import validate_select_options
from .types import ParamRuntimeContext, ParamSelectOptions


def _is_numeric_dtype(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.number)


def _sequence_var_names(ds: xr.Dataset, *, sequence_dim: str) -> tuple[str, ...]:
    return tuple(str(name) for name, var in ds.data_vars.items() if sequence_dim in var.dims)


def _sequence_coord_names(ds: xr.Dataset, *, sequence_dim: str) -> tuple[str, ...]:
    names: list[str] = []
    for name, coord in ds.coords.items():
        if name == sequence_dim:
            continue
        if sequence_dim in coord.dims:
            names.append(str(name))
    return tuple(names)


def _validate_sequence_vars_numeric(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    op: str,
) -> tuple[str, ...]:
    names = _sequence_var_names(ds, sequence_dim=sequence_dim)
    for name in names:
        if not _is_numeric_dtype(np.dtype(ds[name].dtype)):
            raise TypeError(f"{op}: non-numeric sequence variable {name!r} is not supported.")
    return names


def _mask_selected_vars(
    ds: xr.Dataset,
    *,
    mask: xr.DataArray,
    variable_names: Sequence[str],
) -> xr.Dataset:
    out = ds.copy(deep=False)
    updates: dict[str, xr.DataArray] = {}
    for name in variable_names:
        if name in out.data_vars:
            updates[str(name)] = out[name].where(mask)
    if updates:
        out = out.assign(updates)
    return out


def _mask_sequence_coords(
    ds: xr.Dataset,
    *,
    mask: xr.DataArray,
    coord_names: Sequence[str],
    protected_names: Sequence[str] = (),
) -> xr.Dataset:
    out = ds.copy(deep=False)
    protected = {str(name) for name in protected_names}
    updates: dict[str, xr.DataArray] = {}
    for name in coord_names:
        cname = str(name)
        if cname in protected or cname not in out.coords:
            continue
        updates[cname] = out.coords[cname].where(mask)
    if updates:
        out = out.assign_coords(updates)
    return out


def _set_sequence_coord(ds: xr.Dataset, *, sequence_dim: str) -> xr.Dataset:
    if sequence_dim in ds.coords:
        ds = ds.drop_vars(sequence_dim, errors="ignore")
    return ds.assign_coords({sequence_dim: np.arange(int(ds.sizes.get(sequence_dim, 0)), dtype="int64")})


def _promote_dim_dataarray(
    da: xr.DataArray,
    *,
    from_dim: str,
    to_dim: str,
    size: int,
) -> xr.DataArray:
    if from_dim not in da.dims or from_dim == to_dim:
        return da
    coord = xr.DataArray(np.arange(int(size), dtype="int64"), dims=[from_dim], name=to_dim)
    out = da.assign_coords({to_dim: coord}).swap_dims({from_dim: to_dim})
    return out.drop_vars(from_dim, errors="ignore") if from_dim in out.coords else out


def _promote_dim_dataset(
    ds: xr.Dataset,
    *,
    from_dim: str,
    to_dim: str,
    size: int,
) -> xr.Dataset:
    if from_dim not in ds.dims or from_dim == to_dim:
        return ds
    coord = xr.DataArray(np.arange(int(size), dtype="int64"), dims=[from_dim], name=to_dim)
    out = ds.assign_coords({to_dim: coord}).swap_dims({from_dim: to_dim})
    return out.drop_vars(from_dim, errors="ignore") if from_dim in out.coords else out


def _collapse_scalar_point(
    out: "AnalysisObject",
    *,
    sequence_dim: str,
    validate: bool,
) -> "AnalysisObject":
    if sequence_dim in analysis_object_dataset(out).dims:
        return out.isel({sequence_dim: 0}, drop=True, validate=validate)
    return out


def _slice_out_len(
    context: ParamRuntimeContext,
    *,
    seg_len: xr.DataArray,
    layout: str,
) -> int:
    """Resolve slice output width.

    `layout="padded"` may perform explicit eager scalar reduction for chunked
    inputs to determine the maximum row length.
    """
    if layout == "packed":
        return int(context.ds.sizes.get(context.sequence_dim, 0))
    if layout == "padded":
        if int(seg_len.size) == 0:
            return 0
        return scalar_int_boundary(
            seg_len.max(),
            owner="param sel",
            field="layout='padded' output length",
            allow_chunked_compute=True,
        )
    raise ValueError(f"param sel: unsupported layout {layout!r}.")


def _slice_idx_and_valid(
    *,
    bounds_i0: xr.DataArray,
    seg_len: xr.DataArray,
    valid_mask: xr.DataArray,
    sequence_dim: str,
    qdim: str,
    out_len: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    q = xr.DataArray(np.arange(out_len, dtype="int64"), dims=[qdim])
    idx = (bounds_i0.expand_dims({qdim: out_len}) + q).astype("int64")
    span_valid = q < seg_len.expand_dims({qdim: out_len})
    idx_span = idx.where(span_valid, other=np.int64(0)).astype("int64")
    source_valid = gather_along_sequence(
        valid_mask,
        idx_span,
        sequence_dim=sequence_dim,
        query_dim=qdim,
        owner="param sel",
    )
    return idx, span_valid & source_valid


def _slice_output_dataset(
    context: ParamRuntimeContext,
    *,
    idx: xr.DataArray,
    valid_q: xr.DataArray,
    qdim: str,
    out_len: int,
    seq_vars: Sequence[str],
    seq_coords: Sequence[str],
) -> xr.Dataset:
    idx_safe = idx.where(valid_q, other=np.int64(0)).astype("int64")
    ds = gather_dataset_along_sequence(
        context.ds,
        idx_safe,
        sequence_dim=context.sequence_dim,
        query_dim=qdim,
        owner="param sel",
    )
    ds = _promote_dim_dataset(ds, from_dim=qdim, to_dim=context.sequence_dim, size=out_len)
    valid_seq = _promote_dim_dataarray(valid_q, from_dim=qdim, to_dim=context.sequence_dim, size=out_len)
    idx_seq = _promote_dim_dataarray(idx, from_dim=qdim, to_dim=context.sequence_dim, size=out_len)
    ds = _mask_selected_vars(ds, mask=valid_seq, variable_names=seq_vars)
    ds = _mask_sequence_coords(ds, mask=valid_seq, coord_names=seq_coords)
    ds = _set_sequence_coord(ds, sequence_dim=context.sequence_dim)
    ds = ds.assign_coords(
        {
            "sample_index": mark_reserved_coord(
                idx_seq.where(valid_seq, other=np.int64(-1)).astype("int64"),
                name="sample_index",
            ),
            "valid": mark_reserved_coord(valid_seq, name="valid"),
        }
    )
    if context.sequence_size_coord:
        ds = ds.assign_coords(
            {
                context.sequence_size_coord: sequence_size_from_mask(
                    valid_seq,
                    sequence_dim=context.sequence_dim,
                    batch_dims=context.batch_dims,
                )
            }
        )
    return ds


def _point_select(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamSelectOptions,
    validate: bool,
) -> "AnalysisObject":
    seq_vars = _validate_sequence_vars_numeric(
        context.ds,
        sequence_dim=context.sequence_dim,
        op="param selection",
    )
    seq_coords = _sequence_coord_names(context.ds, sequence_dim=context.sequence_dim)
    result = build_index_result(context, query=query, opts=opts)
    index = result.index.astype("int64")
    valid = result.valid.astype(bool)
    idx = index.where(valid, other=np.int64(0)).astype("int64")
    source_ds = context.ds
    if int(source_ds.sizes.get(context.sequence_dim, 0)) == 0:
        source_ds = source_ds.reindex({context.sequence_dim: [0]}, fill_value=np.nan)
    ds = gather_dataset_along_sequence(
        source_ds,
        idx,
        sequence_dim=context.sequence_dim,
        query_dim=result.grid.query_dim,
        owner="param sel",
    )
    ds = _mask_selected_vars(ds, mask=valid, variable_names=seq_vars)
    ds = _mask_sequence_coords(ds, mask=valid, coord_names=seq_coords)
    sample_index = index.where(valid, other=np.int64(-1)).astype("int64")
    ds = ds.assign_coords(
        {
            "sample_index": mark_reserved_coord(sample_index, name="sample_index"),
            "valid": mark_reserved_coord(valid, name="valid"),
        }
    )
    trajectory = result.grid.stacked_dims is None
    out = finalize_param_output(
        context,
        ds,
        query=result.grid.values,
        query_dim=opts.query_dim,
        valid_query=valid,
        validate=validate,
        trajectory=trajectory,
    )
    if result.scalar_query:
        return _collapse_scalar_point(out, sequence_dim=context.sequence_dim, validate=validate)
    return out


def _slice_select(
    context: ParamRuntimeContext,
    *,
    query: slice,
    opts: ParamSelectOptions,
    validate: bool,
) -> "AnalysisObject":
    seq_vars = _validate_sequence_vars_numeric(
        context.ds,
        sequence_dim=context.sequence_dim,
        op="param selection",
    )
    seq_coords = _sequence_coord_names(context.ds, sequence_dim=context.sequence_dim)
    bounds = build_param_bounds_map(
        param=context.spec.coord,
        start=query.start,
        stop=query.stop,
        sequence_dim=context.sequence_dim,
        valid_mask=context.valid_mask,
        param_kind=context.param_kind,
    )
    bounds_i0 = bounds.i0
    bounds_i1 = bounds.i1
    seg_len = (bounds_i1 - bounds_i0).clip(min=0).astype("int64")
    out_len = _slice_out_len(context, seg_len=seg_len, layout=opts.layout)
    qdim = unique_temp_dim(
        f"{context.sequence_dim}__slice",
        taken_dims=dataset_namespace_names(context.ds),
    )
    idx, valid_q = _slice_idx_and_valid(
        bounds_i0=bounds_i0,
        seg_len=seg_len,
        valid_mask=context.valid_mask,
        sequence_dim=context.sequence_dim,
        qdim=qdim,
        out_len=out_len,
    )
    ds = _slice_output_dataset(
        context,
        idx=idx,
        valid_q=valid_q,
        qdim=qdim,
        out_len=out_len,
        seq_vars=seq_vars,
        seq_coords=seq_coords,
    )
    return finalize_structural(context.ao, ds, validate=validate)


def select_param(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray | np.ndarray | Sequence[float] | float | slice,
    opts: ParamSelectOptions,
    validate: bool,
) -> "AnalysisObject":
    """Param-aware selection for point/list/slice queries.

    Parameters
    ----------
    context : ParamRuntimeContext
        Resolved runtime context/payload used by this orchestration boundary.
    query : xr.DataArray | np.ndarray | Sequence[float] | float | slice, optional
        Query coordinate/grid used for parameter evaluation.
    opts : ParamSelectOptions, optional
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
    validate_select_options(opts, owner="param sel")
    assert_reserved_metadata_safe(
        context.ds,
        reserved=("valid", "sample_index"),
        param_name=context.spec.name,
        owner="param sel",
    )
    if isinstance(query, slice):
        if query.step is not None:
            raise ValueError("param sel: slice step is not supported; pass a start/stop slice without step.")
        return _slice_select(context, query=query, opts=opts, validate=validate)
    assert_query_dim_safe(
        context.ds,
        sequence_dim=context.sequence_dim,
        query_dim=opts.query_dim,
        owner="param sel",
    )
    return _point_select(context, query=query, opts=opts, validate=validate)


__all__ = ["select_param"]
