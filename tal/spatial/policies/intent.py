from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tal.utils.frame_schema import get_frames

from tal.core.dataset_ownership import analysis_object_dataset

from .frame import is_framed, resolve_bidirectional_tip_tail_frames
from ..metadata import get_position_intent

if TYPE_CHECKING:
    from ..position import Position


@dataclass(frozen=True)
class PositionAddPlan:
    output_parent: str | None
    output_child: str | None

def _resolve_unframed_intent(
    left: "Position",
    right: "Position",
    *,
    owner: str,
) -> PositionAddPlan:
    left_delta = get_position_intent(analysis_object_dataset(left), owner=owner) == "delta"
    right_delta = get_position_intent(analysis_object_dataset(right), owner=owner) == "delta"
    if left_delta == right_delta:
        raise ValueError(
            f"{owner}: ambiguous unframed Position addition; mark exactly one operand as displacement via as_delta()."
        )
    return PositionAddPlan(output_parent=None, output_child=None)


def resolve_position_add_intent(
    left: "Position",
    right: "Position",
    *,
    owner: str,
) -> PositionAddPlan:
    left_ds = analysis_object_dataset(left)
    right_ds = analysis_object_dataset(right)
    left_parent, left_child = get_frames(left_ds)
    right_parent, right_child = get_frames(right_ds)
    left_framed = is_framed(left_parent, left_child)
    right_framed = is_framed(right_parent, right_child)

    if left_framed and right_framed:
        chained = resolve_bidirectional_tip_tail_frames(
            left_ds,
            right_ds,
            owner=owner,
            what="Position addition",
        )
        return PositionAddPlan(output_parent=chained[0], output_child=chained[1])
    if left_framed:
        return PositionAddPlan(output_parent=left_parent, output_child=left_child)
    if right_framed:
        return PositionAddPlan(output_parent=right_parent, output_child=right_child)
    return _resolve_unframed_intent(left, right, owner=owner)


__all__ = [
    "PositionAddPlan",
    "resolve_position_add_intent",
]
