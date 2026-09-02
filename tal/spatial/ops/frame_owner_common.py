from __future__ import annotations

from tal.frames import Frame
from tal.utils.frame_schema import get_frames, set_frames

from tal.core.dataset_ownership import analysis_object_dataset

from ..metadata import get_expressed_in


def dst_frame_id(dst: Frame | str, *, owner: str) -> str:
    if isinstance(dst, Frame):
        return dst.id
    if isinstance(dst, str):
        cleaned = dst.strip()
        if cleaned:
            return cleaned
    raise TypeError(f"{owner}: dst must be Frame or non-empty string frame id.")


def require_source_parent(value, *, owner: str) -> tuple[str, str | None]:
    parent, child = get_frames(analysis_object_dataset(value))
    if parent is None:
        raise ValueError(f"{owner}: framed input with parent frame id is required.")
    return parent, child


def source_expressed_in_id(value, *, owner: str) -> str:
    parent, _ = require_source_parent(value, owner=owner)
    expressed = get_expressed_in(analysis_object_dataset(value), owner=owner)
    return parent if expressed is None else expressed


def clear_framing(value):
    ds = set_frames(analysis_object_dataset(value), parent=None, child=None, validate=False)
    return value.__class__._from_unvalidated(ds)


__all__ = [
    "clear_framing",
    "dst_frame_id",
    "require_source_parent",
    "source_expressed_in_id",
]
