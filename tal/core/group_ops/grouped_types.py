from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import (
    BatchGroupingFoundationContext,
    GroupingFoundationContext,
    GroupingFoundationOptions,
)

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
class BatchGroupReduceOptions:
    """Options for reducers on a batch-only grouped view.

    Parameters
    ----------
    group_dim : str | None, optional
        Override the group dimension name selected by ``GroupByOptions``.
    include_empty_groups : bool, optional
        Include declared empty bin groups in reducer output.

    Examples
    --------
    >>> from tal.core import BatchGroupReduceOptions
    >>> opts = BatchGroupReduceOptions(group_dim="outcome", include_empty_groups=True)
    >>> (opts.group_dim, opts.include_empty_groups)
    ('outcome', True)
    """

    group_dim: str | None = None
    include_empty_groups: bool = False


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


@dataclass(frozen=True)
class BatchGroupedRuntimePlan:
    """Resolved immutable batch-only grouped runtime plan."""

    foundation: BatchGroupingFoundationContext
    group_dim: str
    group_labels: tuple[object, ...]
    bin_domain_labels: tuple[object, ...] | None
    global_group_rows: tuple[tuple[int, ...], ...]
