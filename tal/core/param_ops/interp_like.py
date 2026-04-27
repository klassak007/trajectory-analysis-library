from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import xarray as xr

from ..orchestration.finalize import restore_and_finalize
from ..orchestration.inputs import query_coord_from_other_input
from ..orchestration.topology import (
    batch_index_for_dataset,
    flatten_param_contexts,
    flatten_query_for_batch_plan,
)
from .batch_labels import labels_selectable_from
from .guards import assert_unique_dim_labels
from .resample import resample_param
from .types import ParamEvalOptions, ParamRuntimeContext


def _assert_unique_batch_labels(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray,
    dim: str,
) -> None:
    assert_unique_dim_labels(context.ds, dim=dim, owner="interp_like")
    assert_unique_dim_labels(query, dim=dim, owner="interp_like")


def _align_inner_batch(
    context: ParamRuntimeContext,
    *,
    query: xr.DataArray,
) -> tuple[ParamRuntimeContext, xr.DataArray]:
    if len(context.batch_dims) != 1:
        raise ValueError("interp_like: internal batch topology must be flattened to one dimension.")
    dim = context.batch_dims[0]
    if dim not in query.dims:
        return context, query
    _assert_unique_batch_labels(context, query=query, dim=dim)
    self_labels = batch_index_for_dataset(context.ds, dim=dim, owner="interp_like")
    other_labels = query.get_index(dim)
    common = self_labels.intersection(other_labels, sort=False)
    if not labels_selectable_from(self_labels, labels=common):
        raise ValueError(
            f"interp_like: batch labels are not representable on source index for dim {dim!r}. "
            "Align batch label representations before interpolation."
        )
    if not labels_selectable_from(other_labels, labels=common):
        raise ValueError(
            f"interp_like: batch labels are not representable on query index for dim {dim!r}. "
            "Align batch label representations before interpolation."
        )
    try:
        ds = context.ds.sel({dim: common})
        query_sel = query.sel({dim: common})
    except KeyError as exc:
        raise ValueError(
            f"interp_like: batch labels are not representable for dim {dim!r}. "
            "Align batch label representations before interpolation."
        ) from exc
    spec_coord = ds.coords[context.spec.name]
    valid = context.valid_mask.sel({dim: common}) if dim in context.valid_mask.dims else context.valid_mask
    batch_coord = ds.coords[dim] if dim in ds.coords else xr.DataArray(common, dims=[dim], name=dim)
    updated = replace(
        context,
        ds=ds,
        spec=replace(context.spec, coord=spec_coord),
        valid_mask=valid,
        batch_coords={dim: batch_coord},
    )
    return updated, query_sel


def _reindex_left_batch(context: ParamRuntimeContext, *, query: xr.DataArray) -> xr.DataArray:
    if len(context.batch_dims) != 1:
        raise ValueError("interp_like: internal batch topology must be flattened to one dimension.")
    dim = context.batch_dims[0]
    if dim not in query.dims:
        return query
    _assert_unique_batch_labels(context, query=query, dim=dim)
    return query.reindex({dim: context.batch_coords[dim]}, fill_value=np.nan)


def interp_like_param(
    context: ParamRuntimeContext,
    *,
    other: "AnalysisObject | xr.Dataset | xr.DataArray",
    opts: ParamEvalOptions,
    batch_join: Literal["inner", "left"],
    validate: bool,
) -> "AnalysisObject":
    """Interpolate onto another object's parameter grid.

    Parameters
    ----------
    context : ParamRuntimeContext
        Resolved runtime context/payload used by this orchestration boundary.
    other : AnalysisObject | xr.Dataset | xr.DataArray, optional
        Secondary operand combined with the receiver/source operand.
    opts : ParamEvalOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    batch_join : Literal['inner', 'left'], optional
        Policy selector controlling alignment/join behavior.
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
    query = query_coord_from_other_input(
        other,
        coord_name=context.spec.name,
        owner="interp_like",
    )
    if not context.batch_dims:
        return resample_param(context, grid=query, opts=opts, validate=validate)
    flattened, plan = flatten_param_contexts([context], owner="interp_like")
    ctx_flat = flattened[0]
    query_flat = flatten_query_for_batch_plan(query, plan=plan, owner="interp_like")
    if batch_join == "inner":
        ctx, query_in = _align_inner_batch(ctx_flat, query=query_flat)
    elif batch_join == "left":
        query_in = _reindex_left_batch(ctx_flat, query=query_flat)
        ctx = ctx_flat
    else:
        raise ValueError("interp_like: batch_join must be 'inner' or 'left'.")
    out = resample_param(ctx, grid=query_in, opts=opts, validate=validate)
    if not plan.enabled:
        return out
    return restore_and_finalize(
        context.ao,
        out.unsafe_data,
        plan=plan,
        validate=validate,
        owner="interp_like",
    )


__all__ = ["interp_like_param"]
