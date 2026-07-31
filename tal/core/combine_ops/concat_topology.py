from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .. import validity_values
from ..orchestration.topology import (
    align_combine_batch_axis,
    allocate_flat_batch_dim_name,
    batch_index_for_dataset,
    join_combine_batch_labels,
    restore_combine_batch_axis,
    stack_combine_batch_axis,
)
from ..param_ops.guards import assert_unique_dim_labels
from .types import CombineContext, SequenceConcatOptions


def flat_dim_name(contexts: list[CombineContext]) -> str:
    return allocate_flat_batch_dim_name([ctx.ds for ctx in contexts], owner="concat_sequence")


def stack_batch(ds: xr.Dataset, *, batch_dims: tuple[str, ...], flat_dim: str) -> xr.Dataset:
    return stack_combine_batch_axis(
        ds,
        batch_dims=batch_dims,
        flat_dim=flat_dim,
        owner="concat_sequence",
        singleton_unbatched=True,
    )


def restore_batch(ds: xr.Dataset, *, batch_dims: tuple[str, ...], flat_dim: str) -> xr.Dataset:
    return restore_combine_batch_axis(
        ds,
        batch_dims=batch_dims,
        flat_dim=flat_dim,
        owner="concat_sequence",
        drop_unbatched_flat=True,
    )


def join_batch_labels(
    stacked: list[xr.Dataset],
    contexts: list[CombineContext],
    *,
    batch_dims: tuple[str, ...],
    flat_dim: str,
    mode: str,
) -> pd.Index:
    include = [not batch_dims or bool(ctx.batch_dims) for ctx in contexts]
    return join_combine_batch_labels(
        stacked,
        flat_dim=flat_dim,
        mode=mode,
        owner="concat_sequence batch",
        include=include,
    )


def align_batch(
    ds: xr.Dataset,
    *,
    flat_dim: str,
    labels: pd.Index,
    mode: str,
    fill_value: object,
    unbatched: bool,
) -> xr.Dataset:
    return align_combine_batch_axis(
        ds,
        flat_dim=flat_dim,
        labels=labels,
        mode=mode,
        fill_value=fill_value,
        owner="concat_sequence",
        unbatched=unbatched,
    )


def _source_lengths(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    flat_dim: str,
    sequence_size_coord: str | None,
) -> np.ndarray:
    n = int(ds.sizes.get(sequence_dim, 0))
    b = int(ds.sizes.get(flat_dim, 0))
    if sequence_size_coord and sequence_size_coord in ds.coords:
        coord = ds.coords[sequence_size_coord]
        if coord.dims == (flat_dim,):
            validated = validity_values.require_valid_sequence_size_values(
                coord,
                sequence_size_coord=sequence_size_coord,
                sequence_len=n,
                owner="concat_sequence",
            )
            return np.asarray(validated.data, dtype="int64")
        if coord.dims == ():
            validated = validity_values.require_valid_sequence_size_values(
                coord,
                sequence_size_coord=sequence_size_coord,
                sequence_len=n,
                owner="concat_sequence",
            )
            scalar = int(np.asarray(validated.data).item())
            return np.full((b,), scalar, dtype="int64")
    return np.full((b,), n, dtype="int64")


def _project_source_lengths(
    *,
    source: pd.Index,
    labels: pd.Index,
    source_lengths: np.ndarray,
    unbatched: bool,
) -> np.ndarray:
    if unbatched:
        scalar = int(source_lengths[0]) if source_lengths.size else 0
        return np.full((int(labels.size),), scalar, dtype="int64")
    out = np.zeros((int(labels.size),), dtype="int64")
    indexer = source.get_indexer(labels)
    present = indexer >= 0
    if np.any(present):
        out[present] = source_lengths[indexer[present]]
    return out


def _aligned_context_row(
    ds: xr.Dataset,
    ctx: CombineContext,
    *,
    labels: pd.Index,
    batch_dims: tuple[str, ...],
    flat_dim: str,
    sequence_dim: str,
    opts: SequenceConcatOptions,
) -> tuple[xr.Dataset, np.ndarray]:
    unbatched = not batch_dims or not ctx.batch_dims
    ds_aligned = align_batch(
        ds,
        flat_dim=flat_dim,
        labels=labels,
        mode=opts.batch_join,
        fill_value=opts.fill_value,
        unbatched=unbatched,
    )
    source = pd.Index([0]) if unbatched else batch_index_for_dataset(
        ds,
        dim=flat_dim,
        owner="concat_sequence",
    )
    source_lengths = _source_lengths(
        ds,
        sequence_dim=sequence_dim,
        flat_dim=flat_dim,
        sequence_size_coord=ctx.sequence_size_coord,
    )
    projected = _project_source_lengths(
        source=source,
        labels=labels,
        source_lengths=source_lengths,
        unbatched=unbatched,
    )
    return ds_aligned, projected


def aligned_sequence_inputs(
    contexts: list[CombineContext],
    *,
    batch_dims: tuple[str, ...],
    flat_dim: str,
    sequence_dim: str,
    opts: SequenceConcatOptions,
) -> tuple[list[xr.Dataset], pd.Index, np.ndarray]:
    stacked = [stack_batch(ctx.ds, batch_dims=batch_dims, flat_dim=flat_dim) for ctx in contexts]
    for ds in stacked:
        assert_unique_dim_labels(ds, dim=flat_dim, owner="concat_sequence")
        assert_unique_dim_labels(ds, dim=sequence_dim, owner="concat_sequence")
    labels = join_batch_labels(
        stacked,
        contexts,
        batch_dims=batch_dims,
        flat_dim=flat_dim,
        mode=opts.batch_join,
    )
    rows = [
        _aligned_context_row(
            ds,
            ctx,
            labels=labels,
            batch_dims=batch_dims,
            flat_dim=flat_dim,
            sequence_dim=sequence_dim,
            opts=opts,
        )
        for ds, ctx in zip(stacked, contexts, strict=True)
    ]
    aligned = [item[0] for item in rows]
    lengths_out = np.vstack([item[1] for item in rows])
    return aligned, labels, lengths_out


__all__ = ["aligned_sequence_inputs", "flat_dim_name", "restore_batch"]
