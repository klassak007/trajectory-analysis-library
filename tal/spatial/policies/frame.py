from __future__ import annotations

import xarray as xr

from tal.utils.frame_schema import get_frames


def is_framed(parent: str | None, child: str | None) -> bool:
    return parent is not None or child is not None


def resolve_apply_output_frames(
    transform_ds: xr.Dataset,
    target_ds: xr.Dataset,
    *,
    owner: str,
) -> tuple[str | None, str | None]:
    transform_parent, transform_child = get_frames(transform_ds)
    target_parent, target_child = get_frames(target_ds)
    transform_framed = is_framed(transform_parent, transform_child)
    target_framed = is_framed(target_parent, target_child)
    if not transform_framed and not target_framed:
        return None, None
    if transform_framed and not target_framed:
        return transform_parent, transform_child
    if target_framed and not transform_framed:
        return target_parent, target_child
    if transform_child != target_parent:
        raise ValueError(
            f"{owner}: framed apply requires transform.child == target.parent; "
            f"got {transform_child!r} vs {target_parent!r}."
        )
    return transform_parent, target_child


def resolve_compose_output_frames(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
) -> tuple[str | None, str | None]:
    left_parent, left_child = get_frames(left_ds)
    right_parent, right_child = get_frames(right_ds)
    left_framed = is_framed(left_parent, left_child)
    right_framed = is_framed(right_parent, right_child)
    if not left_framed and not right_framed:
        return None, None
    if left_framed and not right_framed:
        return left_parent, left_child
    if right_framed and not left_framed:
        return right_parent, right_child
    if left_child != right_parent:
        raise ValueError(
            f"{owner}: framed compose requires left.child == right.parent; got {left_child!r} vs {right_parent!r}."
        )
    return left_parent, right_child


def resolve_components_shared_frames(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
    left_name: str,
    right_name: str,
) -> tuple[str | None, str | None]:
    left_parent, left_child = get_frames(left_ds)
    right_parent, right_child = get_frames(right_ds)
    left_framed = is_framed(left_parent, left_child)
    right_framed = is_framed(right_parent, right_child)
    if left_framed and right_framed and (left_parent, left_child) != (right_parent, right_child):
        raise ValueError(
            f"{owner}: {left_name}/{right_name} frame tags must match exactly when both are framed; "
            f"got {(left_parent, left_child)!r} vs {(right_parent, right_child)!r}."
        )
    if left_framed:
        return left_parent, left_child
    if right_framed:
        return right_parent, right_child
    return None, None


def resolve_bidirectional_tip_tail_frames(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
    what: str,
) -> tuple[str | None, str | None]:
    left_parent, left_child = get_frames(left_ds)
    right_parent, right_child = get_frames(right_ds)
    left_framed = is_framed(left_parent, left_child)
    right_framed = is_framed(right_parent, right_child)
    if left_framed and right_framed:
        if left_child is not None and right_parent is not None and left_child == right_parent:
            return left_parent, right_child
        if right_child is not None and left_parent is not None and right_child == left_parent:
            return right_parent, left_child
        raise ValueError(
            f"{owner}: framed {what} requires tip-to-tail compatibility "
            "(left.child == right.parent or right.child == left.parent)."
        )
    if left_framed:
        return left_parent, left_child
    if right_framed:
        return right_parent, right_child
    return None, None


__all__ = [
    "is_framed",
    "resolve_apply_output_frames",
    "resolve_bidirectional_tip_tail_frames",
    "resolve_compose_output_frames",
    "resolve_components_shared_frames",
]
