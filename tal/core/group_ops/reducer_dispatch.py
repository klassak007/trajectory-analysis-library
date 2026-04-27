from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ..analysis_object import AnalysisObject
from ..reducer_ops.finalize_policy import resolve_reducer_finalize_source
from ..reducer_ops.types import DimLike, ReducerOp, WeightInput, require_supported_op
from .grouped_options import coerce_group_materialize_options, resolve_layout_names
from .grouped_types import GroupMaterializeOptions, GroupedLayout
from .materialize import materialize_grouped_view

if TYPE_CHECKING:
    from .accessor import GroupedView


def _coerce_grouped_reducer_materialize_opts(
    opts: GroupMaterializeOptions | None,
    *,
    owner: str,
) -> GroupMaterializeOptions:
    if opts is None:
        return GroupMaterializeOptions(layout="padded", include_empty_groups=False)
    return coerce_group_materialize_options(opts, owner=owner)


def _map_dim_for_canonical_padded(
    dim: DimLike,
    *,
    requested_layout: GroupedLayout,
    stacked_member_dim: str,
    sequence_dim: str,
) -> DimLike:
    if dim is None:
        return sequence_dim
    if requested_layout != "stacked":
        return dim
    if isinstance(dim, str):
        return sequence_dim if dim == stacked_member_dim else dim
    if not isinstance(dim, Sequence):
        return dim
    mapped = tuple(sequence_dim if value == stacked_member_dim else value for value in dim)
    return mapped


def _dispatch_reducer_call(
    ao: AnalysisObject,
    *,
    op: ReducerOp,
    dim: DimLike,
    skipna: bool,
    ddof: int,
    weights: WeightInput,
    validate: bool,
) -> AnalysisObject:
    fn = getattr(ao, op)
    if op in {"std", "var"}:
        return fn(dim=dim, skipna=skipna, ddof=ddof, weights=weights, validate=validate)
    if op in {"mean", "sum", "median", "min", "max"}:
        return fn(dim=dim, skipna=skipna, weights=weights, validate=validate)
    return fn(dim=dim, weights=weights, validate=validate)


def grouped_reduce(
    grouped: "GroupedView",
    *,
    op: str,
    dim: DimLike = None,
    skipna: bool = True,
    ddof: int = 0,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | None = None,
    validate: bool = True,
    owner: str,
) -> AnalysisObject:
    reducer = require_supported_op(op, owner=owner)
    plan = grouped._resolve_plan(owner=owner)
    source_for_finalize = resolve_reducer_finalize_source(
        plan.foundation.ao,
        op=reducer,
        owner=owner,
    )
    materialize_opts = _coerce_grouped_reducer_materialize_opts(opts, owner=owner)
    _, stacked_member_dim, _ = resolve_layout_names(
        group_dim=plan.group_dim,
        member_dim=plan.member_dim,
        sequence_index_coord=plan.sequence_index_coord,
        opts=materialize_opts,
        owner=owner,
    )
    mapped_dim = _map_dim_for_canonical_padded(
        dim,
        requested_layout=materialize_opts.layout,
        stacked_member_dim=stacked_member_dim,
        sequence_dim=plan.foundation.sequence_dim,
    )
    grouped_ao = materialize_grouped_view(
        plan,
        opts=replace(materialize_opts, layout="padded"),
        validate=validate,
        owner=f"{owner}.materialize",
        source_ao=source_for_finalize,
    )
    return _dispatch_reducer_call(
        grouped_ao,
        op=reducer,
        dim=mapped_dim,
        skipna=skipna,
        ddof=ddof,
        weights=weights,
        validate=validate,
    )


__all__ = ["grouped_reduce"]
