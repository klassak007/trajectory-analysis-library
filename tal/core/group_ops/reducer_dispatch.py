from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..analysis_object import AnalysisObject
from ..reducer_ops.api import (
    resolve_reducer_request,
)
from ..reducer_ops.dims import normalize_reduce_dim_input
from ..reducer_ops.finalize_policy import resolve_reducer_finalize_source
from ..reducer_ops.types import (
    DimLike,
    ReducerOp,
    ResolvedReducerRequest,
    WeightInput,
    require_supported_op,
)
from ..reducer_ops.weights import (
    require_no_unsupported_weights,
    require_weight_alignment,
)
from .batch_reduce import (
    BatchRowPartition,
    assemble_batch_partition_outputs,
    resolve_batch_row_partitions,
    select_batch_row_partition,
)
from .grouped_options import (
    ResolvedGroupedReducerOptions,
    resolve_batch_group_reduce_options,
    resolve_sequence_group_reduce_options,
)
from .grouped_types import (
    BatchGroupedRuntimePlan,
    BatchGroupReduceOptions,
    GroupedLayout,
    GroupMaterializeOptions,
)
from .materialize import materialize_grouped_view
from .types import BatchGroupingFoundationContext

if TYPE_CHECKING:
    from .accessor import GroupedView
    from .batch_view import BatchGroupedView


@dataclass(frozen=True)
class _BatchReducerCall:
    op: ReducerOp
    dims: tuple[str, ...]
    skipna: bool
    ddof: int
    weights: WeightInput
    ndarray_dims: tuple[str, ...]
    group_dim: str


@dataclass(frozen=True)
class _BatchReducerExecution:
    options: ResolvedGroupedReducerOptions
    request: ResolvedReducerRequest
    plan: BatchGroupedRuntimePlan
    call: _BatchReducerCall


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


def _resolve_batch_dim_input(
    dim: DimLike,
    *,
    foundation: BatchGroupingFoundationContext,
    group_dim: str,
    owner: str,
) -> tuple[str, ...]:
    primary_dim = foundation.primary_batch_dim
    explicit = normalize_reduce_dim_input(dim, owner=owner)
    if explicit is None:
        explicit = (primary_dim,)
    if primary_dim not in explicit:
        raise ValueError(
            f"{owner}: batch grouping reduction dims must include primary batch "
            f"dimension {primary_dim!r}."
        )
    if group_dim in explicit:
        raise ValueError(f"{owner}: batch grouping cannot reduce its group dimension {group_dim!r}.")
    return explicit


def _resolve_batch_reduce_request(
    grouped: BatchGroupedView,
    source: AnalysisObject,
    *,
    reducer: ReducerOp,
    dim: DimLike,
    weights: WeightInput,
    opts: object | None,
    owner: str,
) -> tuple[ResolvedGroupedReducerOptions, ResolvedReducerRequest]:
    foundation = grouped._foundation
    options = resolve_batch_group_reduce_options(
        opts,
        defaults=grouped._opts,
        ds=foundation.ds,
        owner=owner,
    )
    requested_dims = _resolve_batch_dim_input(
        dim,
        foundation=foundation,
        group_dim=options.group_dim,
        owner=owner,
    )
    request = resolve_reducer_request(
        source,
        foundation.ds,
        op=reducer,
        dim=requested_dims,
        weights=weights,
        owner=owner,
    )
    return options, request


def _call_batch_partition(
    source: AnalysisObject,
    partition: BatchRowPartition,
    *,
    primary_dim: str,
    call: _BatchReducerCall,
) -> AnalysisObject:
    selected, weights = select_batch_row_partition(
        source,
        call.weights,
        primary_dim=primary_dim,
        rows=partition.rows,
        ndarray_dims=call.ndarray_dims,
        group_dim=call.group_dim,
    )
    return _dispatch_reducer_call(
        selected,
        op=call.op,
        dim=call.dims,
        skipna=call.skipna,
        ddof=call.ddof,
        weights=weights,
        validate=False,
    )


def _execute_batch_partitions(
    source: AnalysisObject,
    partitions: tuple[BatchRowPartition, ...],
    *,
    primary_dim: str,
    call: _BatchReducerCall,
) -> tuple[tuple[AnalysisObject, ...], AnalysisObject]:
    empty_partition = next(
        (partition for partition in partitions if not partition.rows),
        None,
    )
    empty = (
        None
        if empty_partition is None
        else _call_batch_partition(
            source,
            empty_partition,
            primary_dim=primary_dim,
            call=call,
        )
    )
    outputs = tuple(
        _partition_output(
            source,
            partition,
            empty=empty,
            primary_dim=primary_dim,
            call=call,
        )
        for partition in partitions
    )
    if outputs:
        return outputs, outputs[0]
    prototype = _call_batch_partition(
        source,
        BatchRowPartition(label=None, rows=()),
        primary_dim=primary_dim,
        call=call,
    )
    return (), prototype


def _partition_output(
    source: AnalysisObject,
    partition: BatchRowPartition,
    *,
    empty: AnalysisObject | None,
    primary_dim: str,
    call: _BatchReducerCall,
) -> AnalysisObject:
    if partition.rows:
        return _call_batch_partition(
            source,
            partition,
            primary_dim=primary_dim,
            call=call,
        )
    assert empty is not None
    return empty


def _resolve_batch_execution(
    grouped: BatchGroupedView,
    source: AnalysisObject,
    *,
    reducer: ReducerOp,
    dim: DimLike,
    skipna: bool,
    ddof: int,
    weights: WeightInput,
    opts: object | None,
    owner: str,
) -> _BatchReducerExecution:
    options, request = _resolve_batch_reduce_request(
        grouped,
        source,
        reducer=reducer,
        dim=dim,
        weights=weights,
        opts=opts,
        owner=owner,
    )
    plan = grouped._resolve_plan(owner=owner)
    foundation = grouped._foundation
    require_weight_alignment(
        foundation.ds,
        names=request.eligible_names,
        reduce_dims=request.reduce_dims,
        weights=weights,
        op=reducer,
        owner=owner,
    )
    call = _BatchReducerCall(
        op=reducer,
        dims=request.reduce_dims,
        skipna=skipna,
        ddof=ddof,
        weights=weights,
        ndarray_dims=request.active_reduce_dims,
        group_dim=options.group_dim,
    )
    return _BatchReducerExecution(options, request, plan, call)


def _batch_grouped_reduce(
    grouped: BatchGroupedView,
    *,
    reducer: ReducerOp,
    dim: DimLike,
    skipna: bool,
    ddof: int,
    weights: WeightInput,
    opts: object | None,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    foundation = grouped._foundation
    source = foundation.ao
    execution = _resolve_batch_execution(
        grouped,
        source,
        reducer=reducer,
        dim=dim,
        skipna=skipna,
        ddof=ddof,
        weights=weights,
        opts=opts,
        owner=owner,
    )
    partitions = resolve_batch_row_partitions(
        execution.plan,
        options=execution.options,
    )
    outputs, prototype = _execute_batch_partitions(
        source,
        partitions,
        primary_dim=foundation.primary_batch_dim,
        call=execution.call,
    )
    finalize_source = resolve_reducer_finalize_source(source, op=reducer, owner=owner)
    return assemble_batch_partition_outputs(
        execution.plan,
        partitions,
        outputs,
        prototype=prototype,
        request=execution.request,
        options=execution.options,
        finalize_source=finalize_source,
        validate=validate,
        owner=owner,
    )


def _sequence_grouped_reduce(
    grouped: GroupedView,
    *,
    reducer: ReducerOp,
    dim: DimLike,
    skipna: bool,
    ddof: int,
    weights: WeightInput,
    opts: object | None,
    validate: bool,
    owner: str,
) -> AnalysisObject:
    options = resolve_sequence_group_reduce_options(
        opts,
        defaults=grouped._opts,
        ds=grouped._foundation.ds,
        owner=owner,
    )
    source_for_finalize = resolve_reducer_finalize_source(
        grouped._foundation.ao,
        op=reducer,
        owner=owner,
    )
    materialize_opts = options.sequence_options
    assert materialize_opts is not None
    assert materialize_opts.member_dim is not None
    plan = grouped._resolve_plan(owner=owner)
    mapped_dim = _map_dim_for_canonical_padded(
        dim,
        requested_layout=materialize_opts.layout,
        stacked_member_dim=materialize_opts.member_dim,
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


def grouped_reduce(
    grouped: GroupedView | BatchGroupedView,
    *,
    op: str,
    dim: DimLike = None,
    skipna: bool = True,
    ddof: int = 0,
    weights: WeightInput = None,
    opts: GroupMaterializeOptions | BatchGroupReduceOptions | None = None,
    validate: bool = True,
    owner: str,
) -> AnalysisObject:
    reducer = require_supported_op(op, owner=owner)
    require_no_unsupported_weights(weights=weights, op=reducer, owner=owner)
    dispatch = _batch_grouped_reduce if isinstance(
        grouped._foundation,
        BatchGroupingFoundationContext,
    ) else _sequence_grouped_reduce
    return dispatch(
        grouped,
        reducer=reducer,
        dim=dim,
        skipna=skipna,
        ddof=ddof,
        weights=weights,
        opts=opts,
        validate=validate,
        owner=owner,
    )


__all__ = ["grouped_reduce"]
