from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import xarray as xr

from .. import validity_values
from ..orchestration.topology import (
    align_combine_batch_axis,
    allocate_flat_batch_dim_name,
    join_combine_batch_labels,
    join_batch_indices,
    restore_combine_batch_axis,
    stack_combine_batch_axis,
)
from ..param_ops.batch_labels import labels_selectable_from
from ..param_ops.guards import assert_unique_dim_labels
from ..validity_layout import is_left_packed_mask
from .finalize import finalize_combine_output
from .normalize import effective_batch_dims, effective_sequence_dim
from .types import AlignOptions, CombineContext


def _join_sequence_labels(
    contexts: list[CombineContext],
    *,
    sequence_dim: str,
    mode: str,
) -> pd.Index:
    indices = [ctx.ds.get_index(sequence_dim) for ctx in contexts if sequence_dim in ctx.ds.dims]
    return join_batch_indices(indices, mode=mode, owner="align sequence")


def _select_or_reindex_dim(
    ds: xr.Dataset,
    *,
    dim: str,
    labels: pd.Index,
    mode: str,
    fill_value: object,
    owner: str,
) -> xr.Dataset:
    if dim not in ds.dims:
        return ds
    if mode == "outer":
        return ds.reindex({dim: labels}, fill_value=fill_value)
    source = ds.get_index(dim)
    if not labels_selectable_from(source, labels=labels):
        raise ValueError(f"{owner}: labels are not selectable from source index for dim {dim!r}.")
    return ds.sel({dim: labels})


def _batch_labels(contexts: list[CombineContext], *, batch_dims: tuple[str, ...], flat_dim: str, mode: str) -> pd.Index | None:
    if not batch_dims:
        return None
    stacked: list[xr.Dataset] = []
    include: list[bool] = []
    for ctx in contexts:
        has_all = all(dim in ctx.ds.dims for dim in batch_dims)
        has_any = any(dim in ctx.ds.dims for dim in batch_dims)
        if has_any and not has_all:
            raise ValueError("align: partial batch topology is not supported; provide all batch dims or none.")
        stacked.append(
            stack_combine_batch_axis(
                ctx.ds,
                batch_dims=batch_dims,
                flat_dim=flat_dim,
                owner="align",
                singleton_unbatched=False,
            )
        )
        include.append(has_all)
    return join_combine_batch_labels(
        stacked,
        flat_dim=flat_dim,
        mode=mode,
        owner="align batch",
        include=include,
    )


def _assert_unique_axes(
    contexts: list[CombineContext],
    *,
    batch_dims: tuple[str, ...],
    sequence_dim: str | None,
) -> None:
    dims = tuple(batch_dims) + ((sequence_dim,) if sequence_dim else ())
    for ctx in contexts:
        for dim in dims:
            assert_unique_dim_labels(ctx.ds, dim=dim, owner="align")


def _align_batch_context(
    context: CombineContext,
    *,
    batch_dims: tuple[str, ...],
    labels: pd.Index | None,
    flat_dim: str,
    mode: str,
    fill_value: object,
) -> CombineContext:
    if not batch_dims or labels is None:
        return context
    has_all = all(dim in context.ds.dims for dim in batch_dims)
    flat = stack_combine_batch_axis(
        context.ds,
        batch_dims=batch_dims,
        flat_dim=flat_dim,
        owner="align",
        singleton_unbatched=False,
    )
    aligned = align_combine_batch_axis(
        flat,
        flat_dim=flat_dim,
        labels=labels,
        mode=mode,
        fill_value=fill_value,
        owner="align batch",
        unbatched=not has_all,
    )
    ds_out = restore_combine_batch_axis(
        aligned,
        batch_dims=batch_dims,
        flat_dim=flat_dim,
        owner="align",
        drop_unbatched_flat=False,
    )
    return replace(context, ds=ds_out)


def _source_size_values(
    context: CombineContext,
    *,
    sequence_dim: str,
) -> tuple[xr.DataArray, tuple[str, ...]] | None:
    name = context.sequence_size_coord
    if not name or name not in context.ds.coords:
        return None
    coord = context.ds.coords[name]
    expected = context.batch_dims if context.batch_dims else ()
    if tuple(coord.dims) != expected:
        return None
    seq_len = int(context.ds.sizes.get(sequence_dim, 0))
    validated = validity_values.require_valid_sequence_size_values(
        coord,
        sequence_size_coord=name,
        sequence_len=seq_len,
        owner="align",
    )
    return validated, expected


def _rebuild_outer_size_coord(
    context: CombineContext,
    *,
    aligned_ds: xr.Dataset,
    sequence_dim: str,
    labels: pd.Index,
) -> xr.DataArray | None:
    source_index = context.ds.get_index(sequence_dim)
    indexer = source_index.get_indexer(labels)
    present = np.asarray(indexer >= 0, dtype=bool)
    size_info = _source_size_values(context, sequence_dim=sequence_dim)
    seq_len = int(context.ds.sizes.get(sequence_dim, 0))
    if context.batch_dims:
        coords = {dim: aligned_ds.coords[dim] for dim in context.batch_dims}
        size_shape = tuple(int(aligned_ds.sizes[dim]) for dim in context.batch_dims)
    else:
        coords = {}
        size_shape = ()
    if size_info is None:
        sizes = np.full(size_shape if size_shape else (), seq_len, dtype="int64")
    else:
        size_da, _ = size_info
        sizes = np.asarray(size_da.data, dtype="int64")
    idx = np.broadcast_to(indexer, sizes.shape + (indexer.size,))
    available = np.broadcast_to(present, sizes.shape + (present.size,))
    valid_arr = available & (idx >= 0) & (idx < np.expand_dims(sizes, axis=-1))
    valid = xr.DataArray(
        valid_arr,
        dims=context.batch_dims + (sequence_dim,),
        coords={**coords, sequence_dim: labels},
    )
    if not is_left_packed_mask(valid, sequence_dim=sequence_dim):
        return None
    out = valid.astype("int64").sum(dim=sequence_dim)
    if context.batch_dims:
        return out.transpose(*context.batch_dims)
    return xr.DataArray(np.int64(out.data), dims=())


def _repair_outer_sequence_validity(
    context: CombineContext,
    *,
    aligned_ds: xr.Dataset,
    sequence_dim: str,
    labels: pd.Index,
    pad_invalid_outer: bool,
) -> CombineContext:
    name = context.sequence_size_coord
    if name is None:
        return replace(context, ds=aligned_ds)
    if not pad_invalid_outer:
        return replace(context, ds=aligned_ds.drop_vars(name, errors="ignore"), sequence_size_coord=None)
    rebuilt = _rebuild_outer_size_coord(context, aligned_ds=aligned_ds, sequence_dim=sequence_dim, labels=labels)
    if rebuilt is None:
        return replace(context, ds=aligned_ds.drop_vars(name, errors="ignore"), sequence_size_coord=None)
    return replace(context, ds=aligned_ds.assign_coords({name: rebuilt}))


def align_contexts(
    contexts: list[CombineContext],
    *,
    opts: AlignOptions,
) -> list[CombineContext]:
    if not contexts:
        return []
    sequence_dim = effective_sequence_dim(contexts, owner="align", require=False)
    batch_dims = effective_batch_dims(contexts)
    _assert_unique_axes(contexts, batch_dims=batch_dims, sequence_dim=sequence_dim)
    flat_dim = allocate_flat_batch_dim_name([ctx.ds for ctx in contexts], owner="align")
    labels = _batch_labels(contexts, batch_dims=batch_dims, flat_dim=flat_dim, mode=opts.batch_join)
    out = [
        _align_batch_context(
            ctx,
            batch_dims=batch_dims,
            labels=labels,
            flat_dim=flat_dim,
            mode=opts.batch_join,
            fill_value=opts.fill_value,
        )
        for ctx in contexts
    ]
    if sequence_dim is None:
        return out
    seq_labels = _join_sequence_labels(out, sequence_dim=sequence_dim, mode=opts.sequence_join)
    final: list[CombineContext] = []
    for ctx in out:
        ds = _select_or_reindex_dim(
            ctx.ds,
            dim=sequence_dim,
            labels=seq_labels,
            mode=opts.sequence_join,
            fill_value=opts.fill_value,
            owner="align sequence",
        )
        if opts.sequence_join != "outer":
            final.append(replace(ctx, ds=ds))
            continue
        final.append(
            _repair_outer_sequence_validity(
                ctx,
                aligned_ds=ds,
                sequence_dim=sequence_dim,
                labels=seq_labels,
                pad_invalid_outer=opts.pad_invalid_outer,
            )
        )
    return final


def align_many(
    contexts: list[CombineContext],
    *,
    opts: AlignOptions,
    validate: bool,
) -> list["AnalysisObject"]:
    out: list["AnalysisObject"] = []
    aligned = align_contexts(contexts, opts=opts)
    seq = effective_sequence_dim(aligned, owner="align", require=False)
    batch = effective_batch_dims(aligned)
    for ctx in aligned:
        out.append(
            finalize_combine_output(
                ctx,
                ctx.ds,
                sequence_dim=seq,
                batch_dims=batch,
                core_dims=ctx.core_dims,
                param_coord=ctx.param_coord,
                sequence_size_coord=ctx.sequence_size_coord,
                validate=validate,
            )
        )
    return out


def align_pair(
    left: CombineContext,
    right: CombineContext,
    *,
    opts: AlignOptions,
    validate: bool,
) -> tuple["AnalysisObject", "AnalysisObject"]:
    out = align_many([left, right], opts=opts, validate=validate)
    return out[0], out[1]
