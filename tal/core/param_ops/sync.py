from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import xarray as xr

from ..orchestration.finalize import finalize_like, restore_and_finalize
from ..orchestration.inputs import normalize_analysis_object_inputs
from ..orchestration.lazy import require_unchunked_auto_grid_sources
from ..orchestration.resolve import resolve_param_runtime_context
from ..orchestration.topology import (
    flatten_param_contexts,
    flatten_query_for_batch_plan,
)
from .guards import mark_reserved_coord
from .options import coerce_sync_options, resolve_sync_runtime
from .resample import resample_param
from .sync_runtime import (
    align_contexts_batch,
    apply_fill,
    ensure_shared_topology,
    eval_options_from_sync,
    grid_from_join,
)
from .types import ParamRuntimeContext, ParamSyncOptions

if TYPE_CHECKING:
    import numpy as np

    from ..analysis_object import AnalysisObject


def _resolve_sync_contexts(
    aos: Sequence["AnalysisObject"],
    *,
    on: str | None,
    sequence_dim: str | None,
    batch_dims: Sequence[str] | None,
    sequence_size_coord: str | None,
) -> tuple[list[ParamRuntimeContext], list[ParamRuntimeContext], object]:
    contexts = [
        resolve_param_runtime_context(
            ao,
            on=on,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
        )
        for ao in aos
    ]
    ensure_shared_topology(contexts)
    base_contexts = list(contexts)
    contexts, batch_plan = flatten_param_contexts(base_contexts, owner="synchronize_param")
    return base_contexts, contexts, batch_plan


def _is_identity_sync_case(
    *,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float | None,
    options: ParamSyncOptions,
) -> bool:
    return (
        grid is None
        and options.how == "nearest"
        and options.join in {"left", "right", "override"}
    )


def _single_input_target(
    aligned: Sequence[ParamRuntimeContext],
    *,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float | None,
) -> xr.DataArray | np.ndarray | Sequence[float] | float | None:
    if len(aligned) != 1 or grid is not None:
        return grid
    return aligned[0].spec.coord


def _resolve_target_grid(
    aligned: Sequence[ParamRuntimeContext],
    *,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float | None,
    join: str,
    tol: float,
) -> xr.DataArray:
    target = _single_input_target(aligned, grid=grid)
    if target is not None:
        return target
    if join in {"outer", "inner", "domain", "exact"}:
        require_unchunked_auto_grid_sources(
            aligned,
            owner="synchronize_param",
            fields=("spec.coord", "valid_mask"),
        )
    return grid_from_join(aligned, join=join, tol=tol)


def _sync_one(
    src_ctx: ParamRuntimeContext,
    ctx: ParamRuntimeContext,
    *,
    target: xr.DataArray,
    eval_opts,
    how: Literal["interp", "nearest", "fill"],
    tol: float,
    fill_value: float | int,
    batch_plan,
    validate: bool,
) -> "AnalysisObject":
    synced = resample_param(ctx, grid=target, opts=eval_opts, validate=validate)
    if how == "fill":
        synced = apply_fill(
            synced,
            context=ctx,
            grid=target,
            tol=tol,
            fill_value=fill_value,
            eval_opts=eval_opts,
            validate=validate,
        )
    if not batch_plan.enabled:
        return finalize_like(src_ctx.ao, synced.unsafe_data, validate=validate, owner="synchronize_param")
    return restore_and_finalize(
        src_ctx.ao,
        synced.unsafe_data,
        plan=batch_plan,
        validate=validate,
        owner="synchronize_param",
    )


def _sync_identity_one(
    src_ctx: ParamRuntimeContext,
    ctx: ParamRuntimeContext,
    *,
    batch_plan,
    validate: bool,
) -> "AnalysisObject":
    ds = ctx.ds.assign_coords({"valid": mark_reserved_coord(ctx.valid_mask, name="valid")})
    out = finalize_like(ctx.ao, ds, validate=validate, owner="synchronize_param")
    if not batch_plan.enabled:
        return out
    return restore_and_finalize(
        src_ctx.ao,
        out.unsafe_data,
        plan=batch_plan,
        validate=validate,
        owner="synchronize_param",
    )


def synchronize_param(
    aos: Sequence["AnalysisObject | xr.Dataset | xr.DataArray"],
    *,
    on: str | None = None,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float | None = None,
    opts: ParamSyncOptions | None = None,
    validate: bool = True,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
    sequence_size_coord: str | None = None,
) -> list["AnalysisObject"]:
    """Synchronize AO-like inputs onto a shared param grid.

    Parameters
    ----------
    aos : Sequence['AnalysisObject | xr.Dataset | xr.DataArray']
        AO-like inputs consumed by this orchestration boundary.
    on : str | None, optional
        Coordinate/dimension name used as the operation domain.
    grid : xr.DataArray | np.ndarray | Sequence[float] | float | None, optional
        Parameter-domain input used for temporal evaluation/alignment.
    opts : ParamSyncOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : Sequence[str] | None, optional
        Optional override for batch dimensions used by temporal semantics.
    sequence_size_coord : str | None, optional
        Optional sequence-size coordinate used for ragged validity handling.

    Returns
    -------
    list['AnalysisObject']
        Ordered collection produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ParamSyncOptions, synchronize_param
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [0.0, 1.0, 4.0])}, coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> synced = synchronize_param([ao], on="time", grid=[0.0, 1.0], opts=ParamSyncOptions(join="override"))
    >>> synced[0].unsafe_data["value"].values.tolist()
    [0.0, 1.0]
    """
    if not aos:
        raise ValueError("synchronize_param: expected at least one AnalysisObject.")
    ao_inputs = normalize_analysis_object_inputs(aos, owner="synchronize_param", require_nonempty=True)
    options = coerce_sync_options(opts, owner="synchronize_param")
    tol, fill_value = resolve_sync_runtime(options, owner="synchronize_param")
    eval_opts = eval_options_from_sync(query_dim=options.query_dim, how=options.how)
    base_contexts, contexts, batch_plan = _resolve_sync_contexts(
        ao_inputs,
        on=on,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
    )
    aligned = align_contexts_batch(contexts, mode=options.batch_join)
    if len(aligned) == 1 and _is_identity_sync_case(grid=grid, options=options):
        return [_sync_identity_one(base_contexts[0], aligned[0], batch_plan=batch_plan, validate=validate)]
    target = _resolve_target_grid(aligned, grid=grid, join=options.join, tol=tol)
    target = flatten_query_for_batch_plan(target, plan=batch_plan, owner="synchronize_param")
    return [
        _sync_one(
            src_ctx,
            ctx,
            target=target,
            eval_opts=eval_opts,
            how=options.how,
            tol=tol,
            fill_value=fill_value,
            batch_plan=batch_plan,
            validate=validate,
        )
        for src_ctx, ctx in zip(base_contexts, aligned, strict=True)
    ]


def synchronize(
    aos: Sequence["AnalysisObject | xr.Dataset | xr.DataArray"],
    *,
    mode: Literal["param"] = "param",
    on: str | None = None,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float | None = None,
    opts: ParamSyncOptions | None = None,
    validate: bool = True,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
    sequence_size_coord: str | None = None,
) -> list["AnalysisObject"]:
    """Generic synchronization wrapper (param mode only).

    Parameters
    ----------
    aos : Sequence['AnalysisObject | xr.Dataset | xr.DataArray']
        AO-like inputs consumed by this orchestration boundary.
    mode : Literal['param'], optional
        Policy selector controlling alignment/join behavior.
    on : str | None, optional
        Coordinate/dimension name used as the operation domain.
    grid : xr.DataArray | np.ndarray | Sequence[float] | float | None, optional
        Parameter-domain input used for temporal evaluation/alignment.
    opts : ParamSyncOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : Sequence[str] | None, optional
        Optional override for batch dimensions used by temporal semantics.
    sequence_size_coord : str | None, optional
        Optional sequence-size coordinate used for ragged validity handling.

    Returns
    -------
    list['AnalysisObject']
        Ordered collection produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, ParamSyncOptions, synchronize
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"value": ("sample", [0.0, 1.0, 4.0])}, coords={"sample": [0, 1, 2], "time": ("sample", [0.0, 1.0, 2.0])}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> out = synchronize([ao], on="time", grid=[0.0, 2.0], opts=ParamSyncOptions(join="override"))
    >>> out[0].unsafe_data["value"].values.tolist()
    [0.0, 4.0]
    """
    if mode != "param":
        raise ValueError(f"synchronize: unsupported mode {mode!r}; expected 'param'.")
    return synchronize_param(
        aos,
        on=on,
        grid=grid,
        opts=opts,
        validate=validate,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        sequence_size_coord=sequence_size_coord,
    )


__all__ = ["synchronize", "synchronize_param"]
