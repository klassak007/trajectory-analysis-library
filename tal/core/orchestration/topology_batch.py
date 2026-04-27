from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd
import xarray as xr

from ..param_ops.batch_labels import batch_index, join_batch_index, labels_selectable_from
from ..param_ops.batch_topology import (
    BatchFlattenPlan,
    flatten_batch_contexts,
    flatten_query_for_plan,
    restore_dataset_batch_dims,
    stacked_batch_coords_with_labels,
)
from ..param_ops.guards import dataset_namespace_names, unique_temp_dim


def flatten_param_contexts(
    contexts: list["ParamRuntimeContext"],
    *,
    owner: str,
) -> tuple[list["ParamRuntimeContext"], BatchFlattenPlan]:
    _ = owner
    return flatten_batch_contexts(contexts)


def flatten_query_for_batch_plan(
    query: xr.DataArray | object,
    *,
    plan: BatchFlattenPlan,
    owner: str,
) -> xr.DataArray | object:
    if not isinstance(query, xr.DataArray):
        return query
    return flatten_query_for_plan(query, plan=plan, owner=owner)


def restore_dataset_batch_topology(
    ds: xr.Dataset,
    *,
    plan: BatchFlattenPlan,
    owner: str,
) -> xr.Dataset:
    return restore_dataset_batch_dims(ds, plan=plan, owner=owner)


def restore_dataset_multi_batch(
    ds: xr.Dataset,
    *,
    flat_dim: str,
    batch_dims: tuple[str, ...],
    owner: str,
) -> xr.Dataset:
    plan = BatchFlattenPlan(enabled=True, flat_dim=flat_dim, batch_dims=batch_dims)
    return restore_dataset_batch_topology(ds, plan=plan, owner=owner)


def _restore_unbatched_flat_coords(
    source: xr.Dataset,
    out: xr.Dataset,
    *,
    flat_dim: str,
) -> xr.Dataset:
    for name, coord in source.coords.items():
        if tuple(coord.dims) != (flat_dim,) or name in out.coords:
            continue
        out = out.assign_coords({name: coord.isel({flat_dim: 0}, drop=True)})
    return out


def _restore_unbatched_flat_axis(ds: xr.Dataset, *, flat_dim: str) -> xr.Dataset:
    out = ds.isel({flat_dim: 0}, drop=True).drop_vars(flat_dim, errors="ignore")
    out = _restore_unbatched_flat_coords(ds, out, flat_dim=flat_dim)
    return out.drop_vars(flat_dim, errors="ignore")


def batch_index_for_dataset(
    ds: xr.Dataset,
    *,
    dim: str,
    owner: str,
) -> pd.Index:
    try:
        return batch_index(ds, dim=dim)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: failed to resolve batch index for dim {dim!r}.") from exc


def join_batch_indices(
    indices: Sequence[pd.Index],
    *,
    mode: Literal["inner", "outer", "exact"],
    owner: str,
) -> pd.Index:
    return join_batch_index(indices, mode=mode, owner=owner)


def allocate_flat_batch_dim_name(
    datasets: Sequence[xr.Dataset],
    *,
    owner: str,
) -> str:
    _ = owner
    names: set[str] = set()
    for ds in datasets:
        names.update(dataset_namespace_names(ds))
    return unique_temp_dim("__tal_batch__", taken_dims=tuple(sorted(names)))


def stack_combine_batch_axis(
    ds: xr.Dataset,
    *,
    batch_dims: tuple[str, ...],
    flat_dim: str,
    owner: str,
    singleton_unbatched: bool,
) -> xr.Dataset:
    if not batch_dims:
        return ds.expand_dims({flat_dim: [0]}) if singleton_unbatched else ds
    has_all = all(dim in ds.dims for dim in batch_dims)
    has_any = any(dim in ds.dims for dim in batch_dims)
    if has_any and not has_all:
        raise ValueError(f"{owner}: partial batch topology is not supported.")
    if not has_all:
        return ds.expand_dims({flat_dim: [0]}) if singleton_unbatched else ds
    if len(batch_dims) == 1:
        return ds.rename({batch_dims[0]: flat_dim})
    stacked = ds.stack({flat_dim: list(batch_dims)}, create_index=False)
    return stacked.assign_coords(
        stacked_batch_coords_with_labels(
            stacked=stacked,
            batch_dims=batch_dims,
            flat_dim=flat_dim,
            owner=owner,
        )
    )


def restore_combine_batch_axis(
    ds: xr.Dataset,
    *,
    batch_dims: tuple[str, ...],
    flat_dim: str,
    owner: str,
    drop_unbatched_flat: bool,
) -> xr.Dataset:
    if flat_dim not in ds.dims:
        return ds
    if not batch_dims and not drop_unbatched_flat:
        return ds
    if not batch_dims:
        return _restore_unbatched_flat_axis(ds, flat_dim=flat_dim)
    if len(batch_dims) == 1:
        return ds.rename({flat_dim: batch_dims[0]})
    return restore_dataset_multi_batch(
        ds,
        flat_dim=flat_dim,
        batch_dims=batch_dims,
        owner=owner,
    )


def join_combine_batch_labels(
    stacked: Sequence[xr.Dataset],
    *,
    flat_dim: str,
    mode: Literal["inner", "outer", "exact"],
    owner: str,
    include: Sequence[bool] | None = None,
) -> pd.Index:
    if include is not None and len(include) != len(stacked):
        raise ValueError(f"{owner}: include mask must match stacked datasets length.")
    indices = [
        batch_index_for_dataset(ds, dim=flat_dim, owner=owner)
        for i, ds in enumerate(stacked)
        if include is None or include[i]
    ]
    if not indices:
        indices = [batch_index_for_dataset(ds, dim=flat_dim, owner=owner) for ds in stacked]
    return join_batch_indices(indices, mode=mode, owner=owner)


def align_combine_batch_axis(
    ds: xr.Dataset,
    *,
    flat_dim: str,
    labels: pd.Index,
    mode: Literal["inner", "outer", "exact"],
    fill_value: object,
    owner: str,
    unbatched: bool,
) -> xr.Dataset:
    if unbatched:
        base = ds.isel({flat_dim: 0}, drop=True) if flat_dim in ds.dims else ds
        return base.expand_dims({flat_dim: labels})
    if mode == "outer":
        return ds.reindex({flat_dim: labels}, fill_value=fill_value)
    source = batch_index_for_dataset(ds, dim=flat_dim, owner=owner)
    if not labels_selectable_from(source, labels=labels):
        raise ValueError(f"{owner}: batch labels are not selectable from source.")
    return ds.sel({flat_dim: labels})

