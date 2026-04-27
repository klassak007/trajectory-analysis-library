from __future__ import annotations

from typing import TYPE_CHECKING

from .types import ReducerOp

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


def _defining_class_for_reducer_method(cls: type, *, op: str) -> type | None:
    for base in cls.__mro__:
        if op in base.__dict__:
            return base
    return None


def _uses_analysis_object_surface_method(
    defining_cls: type,
    *,
    op: str,
    analysis_object_cls: type,
) -> bool:
    ao_method = analysis_object_cls.__dict__.get(op)
    current_method = defining_cls.__dict__.get(op)
    return current_method is ao_method


def resolve_reducer_finalize_source(
    source: "AnalysisObject",
    *,
    op: ReducerOp,
    owner: str,
) -> "AnalysisObject":
    from ..analysis_object import AnalysisObject

    source_cls = source.__class__
    if source_cls is AnalysisObject:
        return source
    defining_cls = _defining_class_for_reducer_method(source_cls, op=op)
    if defining_cls is None:
        raise ValueError(f"{owner}: reducer method {op!r} is not available on {source_cls.__name__}.")
    if _uses_analysis_object_surface_method(defining_cls, op=op, analysis_object_cls=AnalysisObject):
        return AnalysisObject._from_unvalidated(source.unsafe_data)
    return source


__all__ = ["resolve_reducer_finalize_source"]
