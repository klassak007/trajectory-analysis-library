from __future__ import annotations

from collections.abc import Sequence

from ..orchestration.inputs import normalize_analysis_object_inputs
from ..orchestration.resolve import (
    effective_batch_dims as _effective_batch_dims,
    effective_sequence_dim as _effective_sequence_dim,
    resolve_combine_contexts,
)
from .types import CombineContext, CombineResolveOptions


def normalize_inputs(items: Sequence[object], *, owner: str) -> list["AnalysisObject"]:
    return normalize_analysis_object_inputs(items, owner=owner, require_nonempty=True)


def resolve_contexts(
    aos: Sequence["AnalysisObject"],
    *,
    resolve: CombineResolveOptions,
) -> list[CombineContext]:
    return resolve_combine_contexts(
        aos,
        require_sequence=resolve.require_sequence,
        owner=resolve.owner,
    )


def effective_sequence_dim(contexts: Sequence[CombineContext], *, owner: str, require: bool) -> str | None:
    return _effective_sequence_dim(contexts, owner=owner, require=require)


def effective_batch_dims(contexts: Sequence[CombineContext]) -> tuple[str, ...]:
    return _effective_batch_dims(contexts)
