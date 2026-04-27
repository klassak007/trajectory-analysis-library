from __future__ import annotations

"""Sequence-concat orchestration driver: plan -> overlap -> pack -> finalize."""

import xarray as xr

from .concat_finalize import finalize_concat_sequence_contexts, postprocess_concat_dataset
from .concat_overlap import validate_concat_overlap
from .concat_pack import build_packed_concat_dataset
from .concat_plan import ConcatSequencePlan, build_concat_sequence_plan
from .types import CombineContext, SequenceConcatOptions


def _build_concat_sequence_output(
    *,
    opts: SequenceConcatOptions,
    plan: ConcatSequencePlan,
) -> tuple[xr.Dataset, str | None, str | None]:
    validate_concat_overlap(plan, opts=opts, owner="concat_sequence")
    out, valid = build_packed_concat_dataset(plan, fill_value=opts.fill_value)
    return postprocess_concat_dataset(out, valid=valid, opts=opts, plan=plan)


def concat_sequence_contexts(
    contexts: list[CombineContext],
    *,
    opts: SequenceConcatOptions,
    validate: bool,
) -> "AnalysisObject":
    """Concatenate sequence segments across inputs under sequence/batch policies.

    Parameters
    ----------
    contexts : list[CombineContext]
        Resolved runtime context/payload used by this orchestration boundary.
    opts : SequenceConcatOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
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
    plan = build_concat_sequence_plan(
        contexts,
        opts=opts,
        owner="concat_sequence",
    )
    out, param_name, size_name = _build_concat_sequence_output(
        opts=opts,
        plan=plan,
    )
    return finalize_concat_sequence_contexts(
        contexts,
        ds=out,
        plan=plan,
        param_name=param_name,
        size_name=size_name,
        validate=validate,
    )
