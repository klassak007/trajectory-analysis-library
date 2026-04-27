from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .axis_coords import batch_coord
from .guards import dataset_namespace_names, unique_temp_dim

if TYPE_CHECKING:
    from .types import ParamRuntimeContext


@dataclass(frozen=True)
class BatchFlattenPlan:
    enabled: bool
    flat_dim: str
    batch_dims: tuple[str, ...]


_MISSING_LABEL = object()


def _is_missing_label(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _canonicalize_label(value: object) -> object:
    return _MISSING_LABEL if _is_missing_label(value) else value


def _restore_label(value: object) -> object:
    return np.nan if value is _MISSING_LABEL else value


def _restore_coord_values(values: list[object]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        out[idx] = _restore_label(value)
    return out


def composite_batch_labels(arrays: list[list[object]]) -> np.ndarray:
    rows = list(zip(*arrays))
    labels = np.empty(len(rows), dtype=object)
    labels[:] = [tuple(_canonicalize_label(item) for item in row) for row in rows]
    return labels


def _coord_values_for_labels(
    coord: xr.DataArray,
    *,
    owner: str,
    dim: str,
) -> list[object]:
    chunks = getattr(coord.data, "chunks", None)
    if chunks is not None:
        raise ValueError(
            f"{owner}: chunked batch coordinate labels are not supported for dim {dim!r}."
        )
    return coord.to_numpy().reshape(-1).tolist()


def _batch_cols_from_named_flat_coords(
    ds: xr.Dataset,
    *,
    plan: BatchFlattenPlan,
    owner: str,
) -> dict[str, list[object]] | None:
    cols: dict[str, list[object]] = {}
    for dim in plan.batch_dims:
        if dim not in ds.coords or tuple(ds.coords[dim].dims) != (plan.flat_dim,):
            return None
        values = _coord_values_for_labels(ds.coords[dim], owner=owner, dim=dim)
        cols[dim] = [_canonicalize_label(value) for value in values]
    return cols


def _batch_cols_are_unique(cols: dict[str, list[object]], *, batch_dims: tuple[str, ...]) -> bool:
    arrays = [[_canonicalize_label(item) for item in cols[dim]] for dim in batch_dims]
    preview = pd.MultiIndex.from_arrays(arrays, names=list(batch_dims))
    return bool(preview.is_unique)


def _batch_cols_from_composite_labels(
    labels: list[object],
    *,
    plan: BatchFlattenPlan,
    owner: str,
) -> dict[str, list[object]]:
    cols: dict[str, list[object]] = {dim: [] for dim in plan.batch_dims}
    for value in labels:
        if not isinstance(value, tuple) or len(value) != len(plan.batch_dims):
            raise ValueError(
                f"{owner}: failed to restore multi-batch topology; composite batch labels are malformed."
            )
        for dim, item in zip(plan.batch_dims, value, strict=True):
            cols[dim].append(item)
    return cols


def _batch_cols_for_restore(
    ds: xr.Dataset,
    *,
    plan: BatchFlattenPlan,
    owner: str,
) -> dict[str, list[object]]:
    named_cols = _batch_cols_from_named_flat_coords(ds, plan=plan, owner=owner)
    if named_cols is not None and _batch_cols_are_unique(named_cols, batch_dims=plan.batch_dims):
        return named_cols
    labels = _coord_values_for_labels(ds.coords[plan.flat_dim], owner=owner, dim=plan.flat_dim)
    return _batch_cols_from_composite_labels(labels, plan=plan, owner=owner)


def _stack_batch_coords(
    *,
    batch_coords: dict[str, xr.DataArray],
    batch_dims: tuple[str, ...],
    flat_dim: str,
    owner: str,
) -> dict[str, xr.DataArray]:
    base = xr.Dataset({dim: batch_coords[dim] for dim in batch_dims})
    stacked = base.stack({flat_dim: list(batch_dims)}, create_index=False)
    return stacked_batch_coords_with_labels(
        stacked=stacked,
        batch_dims=batch_dims,
        flat_dim=flat_dim,
        owner=owner,
    )


def stacked_batch_coords_with_labels(
    *,
    stacked: xr.Dataset,
    batch_dims: tuple[str, ...],
    flat_dim: str,
    owner: str,
) -> dict[str, xr.DataArray]:
    arrays = [
        _coord_values_for_labels(stacked[dim], owner=owner, dim=dim)
        for dim in batch_dims
    ]
    labels = composite_batch_labels(arrays)
    out = {dim: stacked[dim] for dim in batch_dims}
    out[flat_dim] = xr.DataArray(labels, dims=[flat_dim], name=flat_dim)
    return out


def _flatten_valid_mask(
    valid: xr.DataArray,
    *,
    plan: BatchFlattenPlan,
    labels: xr.DataArray,
    owner: str,
) -> xr.DataArray:
    present = tuple(dim for dim in plan.batch_dims if dim in valid.dims)
    if not present:
        return valid
    if present != plan.batch_dims:
        raise ValueError(
            f"{owner}: validity mask batch topology is not canonical for batch flattening."
        )
    out = valid.stack({plan.flat_dim: list(plan.batch_dims)}, create_index=False)
    return out.assign_coords({plan.flat_dim: labels})


def _flat_dim_for_contexts(contexts: list["ParamRuntimeContext"]) -> str:
    taken: set[str] = set()
    for context in contexts:
        taken.update(dataset_namespace_names(context.ds))
    return unique_temp_dim("__tal_batch__", taken_dims=tuple(sorted(taken)))


def flatten_batch_contexts(
    contexts: list["ParamRuntimeContext"],
) -> tuple[list["ParamRuntimeContext"], BatchFlattenPlan]:
    if not contexts or len(contexts[0].batch_dims) <= 1:
        return contexts, BatchFlattenPlan(False, "", ())
    batch_dims = contexts[0].batch_dims
    flat_dim = _flat_dim_for_contexts(contexts)
    plan = BatchFlattenPlan(True, flat_dim, batch_dims)
    out = [flatten_runtime_context(context, plan=plan, owner="param ops") for context in contexts]
    return out, plan


def flatten_runtime_context(
    context: "ParamRuntimeContext",
    *,
    plan: BatchFlattenPlan,
    owner: str,
) -> "ParamRuntimeContext":
    if not plan.enabled:
        return context
    stacked_coords = _stack_batch_coords(
        batch_coords=context.batch_coords,
        batch_dims=plan.batch_dims,
        flat_dim=plan.flat_dim,
        owner=owner,
    )
    labels = stacked_coords[plan.flat_dim]
    ds_flat = context.ds.stack({plan.flat_dim: list(plan.batch_dims)}, create_index=False)
    ds_flat = ds_flat.assign_coords(stacked_coords)
    valid = _flatten_valid_mask(context.valid_mask, plan=plan, labels=labels, owner=owner)
    spec = replace(context.spec, coord=ds_flat.coords[context.spec.name], batch_dims=(plan.flat_dim,))
    batch_coord = ds_flat.coords[plan.flat_dim] if plan.flat_dim in ds_flat.coords else labels
    return replace(
        context,
        ds=ds_flat,
        spec=spec,
        batch_dims=(plan.flat_dim,),
        valid_mask=valid,
        batch_coords={plan.flat_dim: batch_coord},
    )


def flatten_query_for_plan(
    query: xr.DataArray | np.ndarray | float | int | list[float] | tuple[float, ...],
    *,
    plan: BatchFlattenPlan,
    owner: str,
):
    if not plan.enabled or not isinstance(query, xr.DataArray):
        return query
    present = tuple(dim for dim in plan.batch_dims if dim in query.dims)
    if not present:
        return query
    if present != plan.batch_dims:
        raise ValueError(
            f"{owner}: query includes a partial batch topology for multi-batch alignment; "
            "provide all batch_dims or none."
        )
    coords = {dim: batch_coord(query, dim=dim) for dim in plan.batch_dims}
    stacked_coords = _stack_batch_coords(
        batch_coords=coords,
        batch_dims=plan.batch_dims,
        flat_dim=plan.flat_dim,
        owner=owner,
    )
    out = query.stack({plan.flat_dim: list(plan.batch_dims)}, create_index=False)
    return out.assign_coords(stacked_coords)


def restore_dataset_batch_dims(
    ds: xr.Dataset,
    *,
    plan: BatchFlattenPlan,
    owner: str,
) -> xr.Dataset:
    if not plan.enabled or plan.flat_dim not in ds.dims:
        return ds
    cols = _batch_cols_for_restore(ds, plan=plan, owner=owner)
    arrays = [cols[dim] for dim in plan.batch_dims]
    preview = pd.MultiIndex.from_arrays(arrays, names=list(plan.batch_dims))
    if not preview.is_unique:
        raise ValueError(
            f"{owner}: failed to restore multi-batch topology; duplicate composite batch labels are not representable."
        )
    coords = {
        dim: xr.DataArray(values, dims=[plan.flat_dim], name=dim)
        for dim, values in cols.items()
    }
    try:
        with_coords = ds.assign_coords(coords)
        out = with_coords.set_index({plan.flat_dim: list(plan.batch_dims)}).unstack(plan.flat_dim)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"{owner}: failed to restore multi-batch topology after alignment.") from exc
    restored = {
        dim: xr.DataArray(
            _restore_coord_values(_coord_values_for_labels(out.coords[dim], owner=owner, dim=dim)),
            dims=[dim],
            name=dim,
        )
        for dim in plan.batch_dims
    }
    return out.assign_coords(restored)


__all__ = [
    "BatchFlattenPlan",
    "composite_batch_labels",
    "flatten_batch_contexts",
    "flatten_query_for_plan",
    "flatten_runtime_context",
    "restore_dataset_batch_dims",
    "stacked_batch_coords_with_labels",
]
