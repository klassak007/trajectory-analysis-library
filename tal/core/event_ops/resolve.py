from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from ..orchestration.inputs import coerce_analysis_object_input
from ..orchestration.resolve import resolve_param_runtime_context
from ..param_engine.validity_mask import finite_param_mask
from ..param_ops.types import ParamRuntimeContext
from .types import ConditionEvalOptions


@dataclass(frozen=True)
class EventEvalContext:
    """Resolved context payload for condition mask evaluation.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    ao: "AnalysisObject"
    runtime: ParamRuntimeContext
    clock: xr.DataArray
    valid_mask: xr.DataArray
    opts: ConditionEvalOptions

    @property
    def dims(self) -> tuple[str, ...]:
        return self.runtime.batch_dims + (self.runtime.sequence_dim,)


def _canonical_dims(runtime: ParamRuntimeContext) -> tuple[str, ...]:
    return runtime.batch_dims + (runtime.sequence_dim,)


def _assert_dim_labels_match_context(
    da: xr.DataArray,
    *,
    runtime: ParamRuntimeContext,
    owner: str,
    field: str,
) -> None:
    expected_dims = runtime.batch_dims + (runtime.sequence_dim,)
    for dim in expected_dims:
        if dim not in da.dims:
            continue
        if dim not in runtime.ds.dims:
            continue
        if da.get_index(dim).equals(runtime.ds.get_index(dim)):
            continue
        raise ValueError(f"{owner}: {field} labels for batch dim {dim!r} must match runtime labels.")


def _canonicalize_runtime_dim_order(
    da: xr.DataArray,
    *,
    runtime: ParamRuntimeContext,
    owner: str,
    field: str,
) -> xr.DataArray:
    dims = _canonical_dims(runtime)
    if runtime.sequence_dim not in da.dims:
        raise ValueError(f"{owner}: {field} is missing expected dims {[runtime.sequence_dim]!r}.")
    missing = [dim for dim in runtime.batch_dims if dim not in da.dims]
    out = da
    for dim in missing:
        coord = runtime.batch_coords.get(dim)
        if coord is None and dim in runtime.ds.coords:
            coord = runtime.ds.coords[dim]
        if coord is None:
            raise ValueError(f"{owner}: {field} cannot infer labels for missing batch dim {dim!r}.")
        out = out.expand_dims({dim: coord})
    _assert_dim_labels_match_context(out, runtime=runtime, owner=owner, field=field)
    missing_after = [dim for dim in dims if dim not in out.dims]
    if missing_after:
        raise ValueError(f"{owner}: {field} is missing expected dims {missing_after!r}.")
    extra = [dim for dim in out.dims if dim not in dims]
    if extra:
        raise ValueError(f"{owner}: {field} has unexpected dims {extra!r}.")
    if tuple(out.dims) == dims:
        return out
    return out.transpose(*dims)


def resolve_event_valid_mask_by_mode(
    runtime: ParamRuntimeContext,
    *,
    opts: ConditionEvalOptions,
    owner: str,
) -> xr.DataArray:
    """Resolve event validity mask under the selected validity mode policy.

    Parameters
    ----------
    runtime : ParamRuntimeContext
        Resolved runtime context/payload used by this orchestration boundary.
    opts : ConditionEvalOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if opts.validity_mode == "auto":
        return runtime.valid_mask.astype(bool)
    if opts.validity_mode == "prefix_only":
        if runtime.sequence_size_coord is None:
            raise ValueError(
                f"{owner}: opts.validity_mode='prefix_only' requires declared sequence_size_coord validity."
            )
        return runtime.valid_mask.astype(bool)
    if opts.validity_mode == "finite_gather":
        return finite_param_mask(runtime.spec.coord).astype(bool)
    raise ValueError(
        f"{owner}: unsupported validity_mode {opts.validity_mode!r}; "
        "expected one of ['auto', 'prefix_only', 'finite_gather']."
    )


def resolve_event_eval_context(
    ao: "AnalysisObject",
    *,
    opts: ConditionEvalOptions,
    owner: str,
) -> EventEvalContext:
    """Resolve event evaluation context on a chosen clock coordinate.

    Parameters
    ----------
    ao : AnalysisObject
        AnalysisObject-like input value.
    opts : ConditionEvalOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    EventEvalContext
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    ctx_ao = ao if opts.on is None else coerce_analysis_object_input(opts.on, owner=owner)
    runtime = resolve_param_runtime_context(
        ctx_ao,
        on=opts.coord_name,
        sequence_dim=opts.sample_dim,
    )
    clock = _canonicalize_runtime_dim_order(
        runtime.spec.coord,
        runtime=runtime,
        owner=owner,
        field=f"clock coord {runtime.spec.name!r}",
    )
    valid_mask = _canonicalize_runtime_dim_order(
        resolve_event_valid_mask_by_mode(runtime, opts=opts, owner=owner),
        runtime=runtime,
        owner=owner,
        field="resolved validity mask",
    )
    return EventEvalContext(
        ao=ctx_ao,
        runtime=runtime,
        clock=clock,
        valid_mask=valid_mask,
        opts=opts,
    )


__all__ = ["EventEvalContext", "resolve_event_eval_context", "resolve_event_valid_mask_by_mode"]
