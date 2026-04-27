from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
import xarray as xr

from ..param_engine.validity_mask import validate_sequence_size_values
from ..param_ops.batch_labels import join_batch_index, labels_selectable_from
from ..param_ops.guards import assert_unique_dim_labels, dataset_namespace_names
from .align import _repair_outer_sequence_validity
from .finalize import finalize_combine_output
from .metadata import (
    canonicalize_optional_names,
    resolve_core_dims,
    shared_optional_name,
)
from .normalize import effective_batch_dims, effective_sequence_dim
from .options import normalize_batch_labels
from .types import BatchConcatOptions, CombineContext


def _ensure_batch_dim_available(contexts: Sequence[CombineContext], *, batch_dim: str) -> None:
    for ctx in contexts:
        if batch_dim in dataset_namespace_names(ctx.ds):
            raise ValueError(f"concat_batch: batch_dim {batch_dim!r} collides with existing dataset namespace.")


def _sequence_labels(contexts: Sequence[CombineContext], *, sequence_dim: str, mode: str) -> pd.Index:
    for ctx in contexts:
        if sequence_dim in ctx.ds.dims:
            assert_unique_dim_labels(ctx.ds, dim=sequence_dim, owner="concat_batch")
    indices = [ctx.ds.get_index(sequence_dim) for ctx in contexts if sequence_dim in ctx.ds.dims]
    return join_batch_index(indices, mode=mode, owner="concat_batch sequence")


def _align_sequence_only(contexts: list[CombineContext], *, sequence_dim: str, mode: str, fill_value: object) -> list[CombineContext]:
    labels = _sequence_labels(contexts, sequence_dim=sequence_dim, mode=mode)
    out: list[CombineContext] = []
    for ctx in contexts:
        if sequence_dim not in ctx.ds.dims:
            out.append(ctx)
            continue
        if mode == "outer":
            ds = ctx.ds.reindex({sequence_dim: labels}, fill_value=fill_value)
            out.append(
                _repair_outer_sequence_validity(
                    ctx,
                    aligned_ds=ds,
                    sequence_dim=sequence_dim,
                    labels=labels,
                    pad_invalid_outer=True,
                )
            )
            continue
        source = ctx.ds.get_index(sequence_dim)
        if not labels_selectable_from(source, labels=labels):
            raise ValueError("concat_batch: sequence labels are not selectable from source index.")
        out.append(replace(ctx, ds=ctx.ds.sel({sequence_dim: labels})))
    return out


def _batch_coord(ds: xr.Dataset, dim: str) -> xr.DataArray:
    if dim in ds.coords and ds.coords[dim].dims == (dim,):
        return ds.coords[dim]
    return xr.DataArray(np.arange(int(ds.sizes[dim]), dtype="int64"), dims=(dim,), name=dim)


def _batch_union_labels(contexts: list[CombineContext], *, dim: str) -> pd.Index | None:
    indices = [ctx.ds.get_index(dim) for ctx in contexts if dim in ctx.ds.dims]
    if not indices:
        return None
    return join_batch_index(indices, mode="outer", owner="concat_batch batch")


def _normalize_context_batch_topology(
    contexts: list[CombineContext],
    *,
    batch_dims: tuple[str, ...],
    fill_value: object,
) -> list[CombineContext]:
    if not batch_dims:
        return contexts
    labels_by_dim = {dim: _batch_union_labels(contexts, dim=dim) for dim in batch_dims}
    out: list[CombineContext] = []
    for ctx in contexts:
        ds = ctx.ds
        for dim in batch_dims:
            labels = labels_by_dim[dim]
            if labels is None:
                continue
            if dim not in ds.dims:
                ds = ds.expand_dims({dim: labels})
                continue
            assert_unique_dim_labels(ds, dim=dim, owner="concat_batch")
            ds = ds.reindex({dim: labels}, fill_value=fill_value)
        out.append(replace(ctx, ds=ds))
    return out


def _declared_lengths(ctx: CombineContext, *, sequence_dim: str) -> xr.DataArray:
    n = int(ctx.ds.sizes.get(sequence_dim, 0))
    name = ctx.sequence_size_coord
    if name and name in ctx.ds.coords:
        coord = ctx.ds.coords[name]
        if ctx.batch_dims and tuple(coord.dims) == ctx.batch_dims:
            return validate_sequence_size_values(
                coord,
                sequence_size_coord=name,
                sequence_len=n,
                owner="concat_batch",
            ).astype("int64")
        if not ctx.batch_dims and tuple(coord.dims) == ():
            return validate_sequence_size_values(
                coord,
                sequence_size_coord=name,
                sequence_len=n,
                owner="concat_batch",
            ).astype("int64")
    if not ctx.batch_dims:
        return xr.DataArray(np.asarray(n, dtype="int64"), dims=())
    shape = tuple(int(ctx.ds.sizes[dim]) for dim in ctx.batch_dims)
    data = np.full(shape, n, dtype="int64")
    coords = {dim: _batch_coord(ctx.ds, dim) for dim in ctx.batch_dims}
    return xr.DataArray(data, dims=ctx.batch_dims, coords=coords)


def _assign_sequence_size_coord(
    ds: xr.Dataset,
    *,
    contexts: list[CombineContext],
    sequence_dim: str,
    out_batch_dims: tuple[str, ...],
    name: str,
) -> xr.Dataset:
    if not out_batch_dims:
        return ds
    target = {dim: _batch_coord(ds, dim) for dim in out_batch_dims[1:]}

    def _row_for_context(ctx: CombineContext) -> xr.DataArray:
        row = _declared_lengths(ctx, sequence_dim=sequence_dim)
        for dim, labels in target.items():
            if dim not in row.dims:
                row = row.expand_dims({dim: labels})
        if target:
            row = row.reindex({dim: labels for dim, labels in target.items()}, fill_value=0)
            row = row.transpose(*out_batch_dims[1:])
        return row

    rows: list[xr.DataArray] = []
    for ctx in contexts:
        rows.append(_row_for_context(ctx))
    joined = xr.concat(rows, dim=out_batch_dims[0])
    joined = joined.assign_coords({out_batch_dims[0]: _batch_coord(ds, out_batch_dims[0])})
    return ds.assign_coords({name: joined})


def _concat_dataset(
    aligned: list[CombineContext],
    *,
    opts: BatchConcatOptions,
    labels: tuple[object, ...],
) -> xr.Dataset:
    return xr.concat(
        [ctx.ds for ctx in aligned],
        dim=opts.batch_dim,
        join="outer",
        coords="different",
        data_vars="all",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        fill_value=opts.fill_value,
    ).assign_coords({opts.batch_dim: xr.DataArray(list(labels), dims=(opts.batch_dim,))})


def _concat_batch_metadata(
    aligned: list[CombineContext],
    *,
    ds_out: xr.Dataset,
    sequence_dim: str | None,
    out_batch_dims: tuple[str, ...],
) -> tuple[xr.Dataset, str | None, str | None, tuple[str, ...]]:
    ds_out, param_name, size_name = canonicalize_optional_names(
        ds_out,
        sequence_dim=sequence_dim,
        batch_dims=out_batch_dims,
        param_name=shared_optional_name([ctx.param_coord for ctx in aligned]),
        size_name=shared_optional_name([ctx.sequence_size_coord for ctx in aligned]),
    )
    core_dims = resolve_core_dims(
        aligned,
        ds=ds_out,
        sequence_dim=sequence_dim,
        batch_dims=out_batch_dims,
        owner="concat_batch",
    )
    return ds_out, param_name, size_name, core_dims


def concat_batch_contexts(
    contexts: list[CombineContext],
    *,
    opts: BatchConcatOptions,
    validate: bool,
) -> "AnalysisObject":
    _ensure_batch_dim_available(contexts, batch_dim=opts.batch_dim)
    sequence_dim = effective_sequence_dim(contexts, owner="concat_batch", require=False)
    batch_dims = effective_batch_dims(contexts)
    aligned = contexts if sequence_dim is None else _align_sequence_only(
        contexts,
        sequence_dim=sequence_dim,
        mode=opts.sequence_join,
        fill_value=opts.fill_value,
    )
    size_contexts = aligned
    aligned = _normalize_context_batch_topology(
        aligned,
        batch_dims=batch_dims,
        fill_value=opts.fill_value,
    )
    labels = normalize_batch_labels(opts.batch_labels, size=len(aligned))
    ds_out = _concat_dataset(aligned, opts=opts, labels=labels)
    out_batch_dims = (opts.batch_dim,) + batch_dims
    size_name = shared_optional_name([ctx.sequence_size_coord for ctx in aligned])
    if sequence_dim is not None and size_name:
        ds_out = _assign_sequence_size_coord(
            ds_out,
            contexts=size_contexts,
            sequence_dim=sequence_dim,
            out_batch_dims=out_batch_dims,
            name=size_name,
        )
    ds_out, param_name, size_name, core_dims = _concat_batch_metadata(
        aligned,
        ds_out=ds_out,
        sequence_dim=sequence_dim,
        out_batch_dims=out_batch_dims,
    )
    return finalize_combine_output(
        aligned[0],
        ds_out,
        sequence_dim=sequence_dim,
        batch_dims=out_batch_dims if sequence_dim is not None else (),
        core_dims=core_dims,
        param_coord=param_name,
        sequence_size_coord=size_name,
        validate=validate,
    )
