from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import GroupingFoundationContext, GroupingFoundationOptions


GroupedLayout = Literal["padded", "stacked"]


@dataclass(frozen=True)
class GroupByOptions:
    """Options for grouped-view construction on top of grouping foundation context.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    preserve_batch: bool = False
    foundation_opts: GroupingFoundationOptions | None = None
    group_dim: str = "group_key"
    member_dim: str = "group_member"
    sequence_index_coord: str = "sequence_index"


@dataclass(frozen=True)
class GroupMaterializeOptions:
    """Options for grouped layout materialization.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    layout: GroupedLayout = "padded"
    group_dim: str | None = None
    member_dim: str | None = None
    sequence_index_coord: str | None = None
    include_empty_groups: bool = True


@dataclass(frozen=True)
class GroupedRuntimePlan:
    """Resolved immutable grouped runtime plan used by grouped wrappers.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    foundation: GroupingFoundationContext
    preserve_batch: bool
    group_dim: str
    member_dim: str
    sequence_index_coord: str
    row_dims: tuple[str, ...]
    stacked_row_dim: str
    group_labels: tuple[object, ...]
    bin_domain_labels: tuple[object, ...] | None
    global_group_rows: tuple[tuple[int, ...], ...]
    per_batch_group_sequence_rows: tuple[tuple[tuple[int, ...], ...], ...] | None
    batch_shape: tuple[int, ...]
