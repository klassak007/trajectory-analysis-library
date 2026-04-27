from __future__ import annotations

"""Owner for ``around(layout='stacked')`` built on segments around output."""

from typing import TYPE_CHECKING

from .finalize import finalize_event_output
from ..schema_read import read_roles
from .around import evaluate_around_segments_windows
from .resolve import EventEvalContext, resolve_event_eval_context
from .types import AroundOptions, Condition
from .window_stack import stack_event_windows

if TYPE_CHECKING:
    import xarray as xr

    from ..analysis_object import AnalysisObject


def _segments_options(opts: AroundOptions) -> AroundOptions:
    return AroundOptions(
        eval=opts.eval,
        edge=opts.edge,
        pre=opts.pre,
        post=opts.post,
        dt=opts.dt,
        grid=opts.grid,
        layout="segments",
    )


def _around_dims(
    ds: "xr.Dataset",
    *,
    context: EventEvalContext,
    owner: str,
) -> tuple[str, str]:
    roles_declared, sequence_dim, batch_dims, _ = read_roles(ds)
    if not roles_declared or sequence_dim is None:
        raise ValueError(f"{owner}: expected around segments output with declared roles.")
    if len(batch_dims) != len(context.runtime.batch_dims) + 1:
        raise ValueError(f"{owner}: expected exactly one around event axis in batch dims, got {batch_dims!r}.")
    if tuple(batch_dims[:-1]) != context.runtime.batch_dims:
        raise ValueError(f"{owner}: around segments batch dims do not match context batch dims.")
    return str(batch_dims[-1]), str(sequence_dim)


def evaluate_around_stacked_windows(
    ao: "AnalysisObject",
    events_or_condition: Condition | "xr.DataArray" | "xr.Dataset",
    *,
    opts: AroundOptions,
    validate: bool = True,
    owner: str = "events.around",
) -> "AnalysisObject":
    """Evaluate around windows using sequence-stacked event-major layout.

    Parameters
    ----------
    ao : AnalysisObject
        AnalysisObject-like input value.
    events_or_condition : Condition | 'xr.DataArray' | 'xr.Dataset'
        Condition/expression used for event or mask evaluation.
    opts : AroundOptions, optional
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
    segments = evaluate_around_segments_windows(
        ao,
        events_or_condition,
        opts=_segments_options(opts),
        validate=validate,
        owner=owner,
    )
    event_dim, tau_dim = _around_dims(segments.unsafe_data, context=context, owner=owner)
    stacked = stack_event_windows(
        segments.unsafe_data,
        batch_dims=context.runtime.batch_dims,
        event_dim=event_dim,
        tau_dim=tau_dim,
        owner=owner,
    )
    return finalize_event_output(
        segments,
        stacked.ds,
        sequence_dim=stacked.stack_dim,
        batch_dims=context.runtime.batch_dims,
        core_dims=context.runtime.core_dims,
        param_name="window_tau",
        size_name=stacked.size_name,
        validate=validate,
        owner=owner,
    )


__all__ = ["evaluate_around_stacked_windows"]
