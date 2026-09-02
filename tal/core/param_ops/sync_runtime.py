from __future__ import annotations

"""Runtime orchestration helpers for param synchronization."""

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..dataset_ownership import analysis_object_dataset
from ..ao_internal import finalize_structural
from ..ordered_dtypes import is_float64_exact_integer
from ..param_engine import ParamMapOptions, build_param_map, normalize_query_grid
from ..param_engine.map_apply import gather_along_sequence
from ..validity_finalize import assign_sequence_size_from_valid_mask
from .batch_labels import (
    batch_index,
    has_mixed_null_domain,
    index_equivalent_labels,
    index_matches_labels,
    join_batch_index,
    labels_collapse_under_dtype,
    labels_selectable_from,
    missing_label_mask,
)
from .guards import assert_query_dim_safe, assert_unique_dim_labels, mark_reserved_coord
from .sync_autogrid import build_auto_grid_from_join
from .types import ParamEvalOptions, ParamRuntimeContext

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


def eval_options_from_sync(*, query_dim: str, how: Literal["interp", "nearest", "fill"]) -> ParamEvalOptions:
    if how == "interp":
        return ParamEvalOptions(method="linear", query_dim=query_dim)
    return ParamEvalOptions(method="nearest", query_dim=query_dim)


def grid_from_join(contexts: Sequence[ParamRuntimeContext], *, join: str, tol: float | int) -> xr.DataArray:
    """Build a shared synchronization target grid from join policy.

    Parameters
    ----------
    contexts : Sequence[ParamRuntimeContext]
        Resolved runtime context/payload used by this orchestration boundary.
    join : str, optional
        Policy selector controlling alignment/join behavior.
    tol : float | int, optional
        Numeric tolerance used for matching/alignment logic.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if join in ("left", "override"):
        return contexts[0].spec.coord
    if join == "right":
        return contexts[-1].spec.coord
    return build_auto_grid_from_join(
        contexts,
        join=join,
        tol=tol,
        owner="synchronize_param",
        param_kind=contexts[0].param_kind,
    )


def _join_batch_labels(
    contexts: Sequence[ParamRuntimeContext],
    *,
    dim: str,
    mode: Literal["inner", "outer", "exact"],
) -> pd.Index:
    indices = [batch_index(ctx.ds, dim=dim) for ctx in contexts]
    return join_batch_index(indices, mode=mode, owner="synchronize_param")


def _assert_unique_source_batch_labels(
    contexts: Sequence[ParamRuntimeContext],
    *,
    dim: str,
) -> None:
    for context in contexts:
        assert_unique_dim_labels(context.ds, dim=dim, owner="synchronize_param")


def _source_batch_dtype(context: ParamRuntimeContext, *, dim: str) -> np.dtype:
    if dim in context.ds.coords:
        return np.asarray(context.ds.coords[dim].values).dtype
    return np.dtype("int64")


def _batch_selectability_error(*, dim: str, mode: str) -> ValueError:
    return ValueError(
        "synchronize_param: batch labels are not representable on source index "
        f"for dim {dim!r} under batch_join={mode!r}. "
        "Align batch label representations before synchronization."
    )


def _assert_outer_batch_domain_compatible(
    contexts: Sequence[ParamRuntimeContext],
    *,
    dim: str,
    labels: pd.Index,
) -> None:
    if len(contexts) <= 1:
        return
    source_indices = [batch_index(context.ds, dim=dim) for context in contexts]
    if all(index_matches_labels(source, labels) for source in source_indices):
        return
    if all(index_equivalent_labels(source, labels) for source in source_indices):
        return
    if has_mixed_null_domain(labels):
        raise ValueError(
            "synchronize_param: batch_join='outer' has incompatible mixed batch label domains "
            f"on dim {dim!r}; null and non-null labels cannot be joined safely."
        )
    for context in contexts:
        dtype = _source_batch_dtype(context, dim=dim)
        if labels_collapse_under_dtype(labels, dtype=dtype):
            raise ValueError(
                "synchronize_param: batch_join='outer' has incompatible mixed batch label domains "
                f"on dim {dim!r}; labels collapse under dtype coercion."
            )


def _align_valid_mask_for_batch(
    context: ParamRuntimeContext,
    *,
    dim: str,
    labels: pd.Index,
    mode: Literal["inner", "outer", "exact"],
) -> xr.DataArray:
    valid = context.valid_mask.reset_coords(drop=True)
    if dim in valid.dims:
        if mode == "outer":
            return valid.reindex({dim: labels}, fill_value=False)
        source = valid.get_index(dim)
        if not labels_selectable_from(source, labels=labels):
            raise _batch_selectability_error(dim=dim, mode=mode)
        try:
            return valid.sel({dim: labels})
        except KeyError as exc:
            raise _batch_selectability_error(dim=dim, mode=mode) from exc
    if mode != "outer":
        return valid
    out = valid.expand_dims({dim: labels})
    source = batch_index(context.ds, dim=dim)
    missing = missing_label_mask(labels, source=source, dim=dim)
    return out.where(~missing, other=False)


def _outer_batch_reindex_fill_values(context: ParamRuntimeContext) -> dict[str, object]:
    fills: dict[str, object] = {}
    for name, variable in context.ds.variables.items():
        dtype = np.dtype(variable.dtype)
        if np.issubdtype(dtype, np.datetime64):
            fills[str(name)] = np.datetime64("NaT", "ns")
        elif np.issubdtype(dtype, np.timedelta64):
            fills[str(name)] = np.timedelta64("NaT", "ns")
    return fills


def _align_batch_context(
    context: ParamRuntimeContext,
    *,
    dim: str,
    labels: pd.Index,
    mode: Literal["inner", "outer", "exact"],
) -> ParamRuntimeContext:
    if mode == "outer":
        ds = context.ds.reindex({dim: labels}, fill_value=_outer_batch_reindex_fill_values(context))
    else:
        source = batch_index(context.ds, dim=dim)
        if not labels_selectable_from(source, labels=labels):
            raise _batch_selectability_error(dim=dim, mode=mode)
        try:
            ds = context.ds.sel({dim: labels})
        except KeyError as exc:
            raise _batch_selectability_error(dim=dim, mode=mode) from exc
    spec_coord = ds.coords[context.spec.name]
    valid = _align_valid_mask_for_batch(context, dim=dim, labels=labels, mode=mode)
    batch_coord = ds.coords[dim] if dim in ds.coords else xr.DataArray(labels, dims=[dim], name=dim)
    return replace(context, ds=ds, spec=replace(context.spec, coord=spec_coord), valid_mask=valid, batch_coords={dim: batch_coord})


def align_contexts_batch(
    contexts: Sequence[ParamRuntimeContext],
    *,
    mode: Literal["inner", "outer", "exact"],
) -> list[ParamRuntimeContext]:
    """Align runtime contexts across the active batch axis using batch_join policy.

    Parameters
    ----------
    contexts : Sequence[ParamRuntimeContext]
        Resolved runtime context/payload used by this orchestration boundary.
    mode : Literal['inner', 'outer', 'exact'], optional
        Policy selector controlling alignment/join behavior.

    Returns
    -------
    list[ParamRuntimeContext]
        Ordered collection produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if not contexts[0].batch_dims:
        return list(contexts)
    if len(contexts[0].batch_dims) != 1:
        raise ValueError("synchronize_param: internal batch topology must be flattened to one dimension.")
    dim = contexts[0].batch_dims[0]
    _assert_unique_source_batch_labels(contexts, dim=dim)
    labels = _join_batch_labels(contexts, dim=dim, mode=mode)
    if mode == "outer":
        _assert_outer_batch_domain_compatible(contexts, dim=dim, labels=labels)
    return [_align_batch_context(ctx, dim=dim, labels=labels, mode=mode) for ctx in contexts]


def ensure_shared_topology(contexts: Sequence[ParamRuntimeContext]) -> None:
    first = contexts[0]
    for ctx in contexts[1:]:
        if ctx.sequence_dim != first.sequence_dim:
            raise ValueError("synchronize_param: all inputs must share the same sequence_dim.")
        if ctx.batch_dims != first.batch_dims:
            raise ValueError("synchronize_param: all inputs must share the same batch_dims.")


def ensure_shared_param_kind(contexts: Sequence[ParamRuntimeContext]) -> None:
    first = contexts[0].param_kind
    for ctx in contexts[1:]:
        if ctx.param_kind != first:
            raise ValueError("synchronize_param: all inputs must share the same param coordinate kind.")


def _nearest_tolerance_mask(
    context: ParamRuntimeContext,
    *,
    grid: xr.DataArray,
    query_dim: str,
    tol: float | int,
) -> xr.DataArray:
    assert_query_dim_safe(context.ds, sequence_dim=context.sequence_dim, query_dim=query_dim, owner="synchronize_param")
    q = normalize_query_grid(
        grid,
        query_dim=query_dim,
        batch_dims=context.batch_dims,
        batch_coords=context.batch_coords,
        param_kind=context.param_kind,
    )
    pmap = build_param_map(
        param=context.spec.coord,
        query=q.values,
        sequence_dim=context.sequence_dim,
        query_dim=q.query_dim,
        valid_mask=context.valid_mask,
        options=ParamMapOptions(method="nearest"),
        param_kind=context.param_kind,
    )
    nearest = gather_along_sequence(
        context.spec.coord,
        pmap.i0,
        sequence_dim=context.sequence_dim,
        query_dim=q.query_dim,
        owner="synchronize_param",
    )
    mask = _within_tolerance(nearest=nearest, query=q.values, valid=pmap.valid, tol=tol, param_kind=context.param_kind)
    if q.query_dim != context.sequence_dim and q.query_dim in mask.dims:
        out = mask
        if context.sequence_dim in out.coords and context.sequence_dim not in out.dims:
            out = out.reset_coords(names=context.sequence_dim, drop=True)
        return out.rename({q.query_dim: context.sequence_dim})
    return mask


def _within_tolerance(
    *,
    nearest: xr.DataArray,
    query: xr.DataArray,
    valid: xr.DataArray,
    tol: float | int,
    param_kind: str,
) -> xr.DataArray:
    if param_kind == "datetime64":
        delta = (nearest - query).astype("timedelta64[ns]").astype("int64")
        safe_delta = delta.where(valid, other=0)
        return valid & (xr.apply_ufunc(np.abs, safe_delta, dask="allowed") <= int(tol))
    if np.dtype(nearest.dtype).kind in {"i", "u"} or np.dtype(query.dtype).kind in {"i", "u"}:
        return xr.apply_ufunc(
            _numeric_within_tolerance_block,
            nearest,
            query,
            valid,
            kwargs={"tol": tol},
            vectorize=False,
            dask="parallelized",
            output_dtypes=[bool],
        )
    return valid & (xr.apply_ufunc(np.abs, nearest - query, dask="allowed") <= float(tol))


def _numeric_within_tolerance_block(
    nearest: np.ndarray,
    query: np.ndarray,
    valid: np.ndarray,
    *,
    tol: float | int,
) -> np.ndarray:
    nearest_values, query_values, valid_values = np.broadcast_arrays(nearest, query, valid)
    nearest_integral = nearest_values.dtype.kind in {"i", "u"}
    query_integral = query_values.dtype.kind in {"i", "u"}
    mixed_float_integer = nearest_integral != query_integral
    out = np.zeros(valid_values.shape, dtype=bool)
    for index in np.ndindex(valid_values.shape):
        if not bool(valid_values[index]):
            continue
        left = np.asarray(nearest_values[index]).reshape(()).item()
        right = np.asarray(query_values[index]).reshape(()).item()
        integer_value = left if nearest_integral else right
        if mixed_float_integer and not is_float64_exact_integer(integer_value):
            raise ValueError(
                "synchronize_param: mixed integer/float tolerance comparison would convert "
                f"integer value {int(integer_value)!r} lossily to float64. "
                "Use matching integer parameter/grid dtypes or rescale the parameter domain."
            )
        if np.isfinite(left) and np.isfinite(right):
            out[index] = abs(left - right) <= tol
    return out


def _mask_numeric_sequence(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    mask: xr.DataArray,
    fill_value: float | int,
) -> xr.Dataset:
    out = ds.copy(deep=False)
    updates: dict[str, xr.DataArray] = {}
    for name, var in out.data_vars.items():
        if sequence_dim in var.dims and np.issubdtype(var.dtype, np.number):
            updates[str(name)] = var.where(mask, other=fill_value)
    return out.assign(updates) if updates else out


def _apply_fill_metadata(
    ds: xr.Dataset,
    *,
    context: ParamRuntimeContext,
    mask: xr.DataArray,
) -> xr.Dataset:
    out = ds.assign_coords({"valid": mark_reserved_coord(mask, name="valid")})
    out, _ = assign_sequence_size_from_valid_mask(
        out,
        valid=mask,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        sequence_size_coord=context.sequence_size_coord,
    )
    return out


def apply_fill(
    out: "AnalysisObject",
    *,
    context: ParamRuntimeContext,
    grid: xr.DataArray,
    tol: float | int,
    fill_value: float | int,
    eval_opts: ParamEvalOptions,
    validate: bool,
) -> "AnalysisObject":
    """Apply tolerance-masked nearest fill and finalize structural metadata.

    Parameters
    ----------
    out : AnalysisObject
        Mutable output buffer/state object updated by this runtime operation.
    context : ParamRuntimeContext, optional
        Resolved runtime context/payload used by this orchestration boundary.
    grid : xr.DataArray, optional
        Parameter-domain input used for temporal evaluation/alignment.
    tol : float, optional
        Numeric tolerance used for matching/alignment logic.
    fill_value : float | int, optional
        Behavior flag/policy controlling boundary semantics.
    eval_opts : ParamEvalOptions, optional
        Options controlling policy and execution behavior.
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
    mask = _nearest_tolerance_mask(context, grid=grid, query_dim=eval_opts.query_dim, tol=tol)
    ds = _mask_numeric_sequence(
        analysis_object_dataset(out),
        sequence_dim=context.sequence_dim,
        mask=mask,
        fill_value=fill_value,
    )
    ds = _apply_fill_metadata(ds, context=context, mask=mask)
    return finalize_structural(out, ds, validate=validate)


__all__ = [
    "align_contexts_batch",
    "apply_fill",
    "ensure_shared_topology",
    "ensure_shared_param_kind",
    "eval_options_from_sync",
    "grid_from_join",
]
