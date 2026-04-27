from __future__ import annotations

from dataclasses import dataclass

from tal.core.orchestration.context import DatasetContext, DatasetContextOptions, resolve_dataset_context
from tal.core.orchestration.inputs import coerce_analysis_object_input

from .options import AOVizOptions, VizKind, coerce_viz_options, require_supported_viz_kind


@dataclass(frozen=True)
class VizRuntimeContext:
    source: "AnalysisObject"
    context: DatasetContext
    kind: VizKind
    opts: AOVizOptions


def resolve_viz_runtime(
    value: object,
    *,
    kind: VizKind | str,
    opts: AOVizOptions | None,
    owner: str,
) -> VizRuntimeContext:
    resolved_kind = require_supported_viz_kind(kind, owner=owner)
    source = coerce_analysis_object_input(value, owner=owner)
    options = coerce_viz_options(opts, owner=owner)
    context = resolve_dataset_context(
        source,
        owner=owner,
        options=DatasetContextOptions(require_roles=True, require_sequence_dim=True),
    )
    return VizRuntimeContext(
        source=source,
        context=context,
        kind=resolved_kind,
        opts=options,
    )


__all__ = ["VizRuntimeContext", "resolve_viz_runtime"]
