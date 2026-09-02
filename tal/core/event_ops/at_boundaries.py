from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from ..dataset_ownership import analysis_object_dataset
from ..orchestration.finalize import finalize_like
from ..orchestration.lazy import fail_if_chunked_boundary, is_chunked_dataarray
from ..param_ops.guards import dataset_namespace_names
from ..param_ops.types import ParamEvalOptions
from .boundary import extract_event_boundaries
from .boundary_select import select_event_boundaries
from .evaluate import evaluate_mask
from .pack import pack_event_table
from .resolve import EventEvalContext, resolve_event_eval_context
from .types import Condition, EventExtractOptions, AtBoundariesOptions

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject

_AT_BOUNDARIES_METADATA = (
    "event_edge_code",
    "event_sample_index_before",
    "event_sample_index_after",
)


def _selection_cap(opts: AtBoundariesOptions) -> int | None:
    if opts.mode == "first":
        return 1
    if opts.mode == "first_n":
        return int(opts.max_events or 0)
    if opts.mode == "all" and opts.max_events is not None:
        return int(opts.max_events)
    return None


def _edge_multiplier(edges: str) -> int:
    return 1 if edges == "all" else 2


def _boundary_extraction_cap(opts: AtBoundariesOptions, *, chunked: bool) -> int | None:
    if not chunked:
        return None
    selection_cap = _selection_cap(opts)
    if selection_cap is None:
        return None
    return max(1, selection_cap * _edge_multiplier(opts.edges))


def _boundary_event_options(opts: AtBoundariesOptions, *, chunked: bool) -> EventExtractOptions:
    return EventExtractOptions(
        eval=opts.eval,
        include_initial=False,
        truth_eval="exact",
        max_events=_boundary_extraction_cap(opts, chunked=chunked),
    )


def _chunked_boundary_inputs(effective: xr.DataArray, *, context: EventEvalContext) -> bool:
    return any(is_chunked_dataarray(da) for da in (effective, context.valid_mask, context.clock))


def _preflight_boundary_extraction(*, chunked: bool, opts: AtBoundariesOptions, owner: str) -> None:
    fail_if_chunked_boundary(
        chunked and opts.max_events is None,
        owner=owner,
        message="chunked event extraction requires opts.max_events for bounded output sizing.",
    )
    fail_if_chunked_boundary(
        chunked and opts.mode == "last",
        owner=owner,
        message="opts.mode='last' requires unchunked boundary selection for exact semantics.",
    )


def _table_event_dim(table: xr.Dataset, *, context: EventEvalContext, owner: str) -> str:
    dims = [dim for dim in table["time"].dims if dim not in context.runtime.batch_dims]
    if len(dims) != 1:
        raise ValueError(f"{owner}: expected exactly one boundary axis, found {dims!r}.")
    return dims[0]


def _param_eval_options(*, query_dim: str, opts: AtBoundariesOptions) -> ParamEvalOptions:
    return ParamEvalOptions(method=opts.eval.ao_interp, duplicate_policy="invalid", query_dim=query_dim)


def _metadata_namespace_safe(ds: xr.Dataset, *, owner: str) -> None:
    names = set(dataset_namespace_names(ds))
    conflicts = sorted(name for name in _AT_BOUNDARIES_METADATA if name in names)
    if conflicts:
        raise ValueError(f"{owner}: boundary metadata names conflict with dataset namespace: {conflicts!r}.")


def _attach_boundary_metadata(
    out: "AnalysisObject",
    *,
    table: xr.Dataset,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    out_ds = analysis_object_dataset(out)
    _metadata_namespace_safe(out_ds, owner=owner)
    ds = out_ds.assign_coords(
        {
            "event_edge_code": table["edge_code"].astype("int8"),
            "event_sample_index_before": table["sample_index_before"].astype("int64"),
            "event_sample_index_after": table["sample_index_after"].astype("int64"),
        }
    )
    return finalize_like(out, ds, validate=validate, owner=owner)


def evaluate_at_boundaries_condition(
    ao: "AnalysisObject",
    condition: Condition,
    *,
    opts: AtBoundariesOptions,
    validate: bool = True,
    owner: str = "events.at_boundaries",
) -> "AnalysisObject":
    """Evaluate an AO at selected enter/exit boundary times for a condition.

    Parameters
    ----------
    ao : AnalysisObject
        AnalysisObject-like input value.
    condition : Condition
        Condition/expression used for event or mask evaluation.
    opts : AtBoundariesOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    context = resolve_event_eval_context(ao, opts=opts.eval, owner=owner)
    effective = evaluate_mask(condition, context=context, owner=owner)
    chunked = _chunked_boundary_inputs(effective, context=context)
    _preflight_boundary_extraction(chunked=chunked, opts=opts, owner=owner)
    boundaries = extract_event_boundaries(
        condition,
        effective_mask=effective,
        context=context,
        opts=_boundary_event_options(opts, chunked=chunked),
        owner=owner,
        emit_triggers=False,
    )
    selected = select_event_boundaries(
        boundaries,
        context=context,
        edges=opts.edges,
        mode=opts.mode,
        max_events=opts.max_events,
        on_empty=opts.on_empty,
        owner=owner,
    )
    table = pack_event_table(selected, context=context, owner=owner)
    boundary_dim = _table_event_dim(table, context=context, owner=owner)
    out = context.ao.param.at(
        table["time"],
        on=opts.eval.coord_name,
        opts=_param_eval_options(query_dim=boundary_dim, opts=opts),
        validate=validate,
        sequence_dim=context.runtime.sequence_dim,
        batch_dims=context.runtime.batch_dims,
        sequence_size_coord=context.runtime.sequence_size_coord,
    )
    if (
        context.runtime.sequence_dim in analysis_object_dataset(out).dims
        and context.runtime.sequence_dim != boundary_dim
    ):
        out = out.rename({context.runtime.sequence_dim: boundary_dim}, validate=validate)
    return _attach_boundary_metadata(out, table=table, validate=validate, owner=owner)


__all__ = ["evaluate_at_boundaries_condition"]
