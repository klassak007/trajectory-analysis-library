from __future__ import annotations

import xarray as xr

from tal.frames import Frame
from tal.utils.frame_schema import get_frames, set_frames

from tal.core.dataset_ownership import analysis_object_dataset

from ..metadata import get_expressed_in, set_expressed_in
from ..metadata.relation import clear_expressed_in


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


def require_parent_basis_for_inverse(value, *, owner: str) -> None:
    """Require framed inverse inputs to use their relation-parent basis."""
    ds = analysis_object_dataset(value)
    parent, child = get_frames(ds)
    expressed = get_expressed_in(ds, owner=owner)
    if parent is None:
        if child is None and expressed is None:
            return
        raise ValueError(
            f"{owner}: inverse requires a parent frame whenever child or "
            "expressed_in is declared; declare parent or clear the remaining "
            "frame metadata before inversion."
        )
    if expressed == parent:
        return
    raise ValueError(
        f"{owner}: inverse requires expressed_in to equal parent; "
        "call value.express_in(parent).inverse() first."
    )


def frame_inverse_component_datasets(
    source: xr.Dataset,
    components: tuple[xr.Dataset, xr.Dataset],
    *,
    owner: str,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Apply one inverse relation and basis to component Datasets."""
    parent, child = get_frames(source)
    framed: list[xr.Dataset] = []
    for component in components:
        result = set_frames(component, parent=child, child=parent, validate=False)
        framed.append(
            set_expressed_in(
                result,
                expressed_in=child,
                validate=False,
                owner=owner,
            )
        )
    return framed[0], framed[1]


def clear_framing(value, *, owner: str):
    ds = clear_expressed_in(
        analysis_object_dataset(value),
        validate=False,
        owner=owner,
    )
    ds = set_frames(ds, parent=None, child=None, validate=False)
    return value._rewrap_dataset(ds, validate=False)


__all__ = [
    "clear_framing",
    "dst_frame_id",
    "frame_inverse_component_datasets",
    "require_parent_basis_for_inverse",
    "require_source_parent",
    "source_expressed_in_id",
]
