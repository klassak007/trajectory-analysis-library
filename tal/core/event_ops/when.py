from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from .when_common import enforce_when_on_empty, selected_when_mask
from .when_segments import evaluate_when_segments_layout
from .when_stream import evaluate_when_stream_layout
from .evaluate import evaluate_mask
from .resolve import EventEvalContext, resolve_event_eval_context
from .types import Condition, WhenOptions

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject

def _apply_mask_layout(
    context: EventEvalContext,
    selected: xr.DataArray,
    *,
    validate: bool,
) -> "AnalysisObject":
    return context.ao.where(selected, drop=False, validate=validate)


def evaluate_when_condition(
    ao: "AnalysisObject",
    condition: Condition,
    *,
    opts: WhenOptions,
    validate: bool = True,
    owner: str = "events.when",
) -> "AnalysisObject":
    """Evaluate condition-driven AO selection under ``WhenOptions``.

    Parameters
    ----------
    ao : AnalysisObject
        AnalysisObject-like input value.
    condition : Condition
        Condition/expression used for event or mask evaluation.
    opts : WhenOptions, optional
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
    if opts.layout == "segments":
        return evaluate_when_segments_layout(
            ao,
            condition,
            opts=opts,
            validate=validate,
            owner=owner,
        )
    if opts.layout == "stream":
        return evaluate_when_stream_layout(
            ao,
            condition,
            opts=opts,
            validate=validate,
            owner=owner,
        )
    if opts.layout != "mask":
        raise NotImplementedError(
            f"{owner}: opts.layout={opts.layout!r} is not supported."
        )
    context = resolve_event_eval_context(ao, opts=opts.eval, owner=owner)
    effective = evaluate_mask(condition, context=context, owner=owner)
    selected = selected_when_mask(
        effective,
        valid_mask=context.valid_mask,
        inside=opts.inside,
    )
    enforce_when_on_empty(
        selected,
        opts=opts,
        owner=owner,
        layout="mask",
        unit="samples",
        chunked_message="opts.on_empty='error' requires unchunked when selection.",
    )
    return _apply_mask_layout(context, selected, validate=validate)


__all__ = ["evaluate_when_condition"]
