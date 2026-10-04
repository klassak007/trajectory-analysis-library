from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ..param_engine import ParamMap, ParamMapOptions
from ..param_engine.prepared import PreparedParamEvaluation
from ..param_engine.query_topology import (
    QueryOutputPlan,
    generated_query_coordinate_names,
    preflight_query_output_namespace,
)
from .finalize import (
    _finalize_prepared_param_output,
    _prepare_param_output_dataset,
    assign_sampled_query_coordinate,
)
from .guards import assert_query_dim_safe, assert_reserved_metadata_safe
from .options import coerce_eval_options
from .runtime_prepare import prepare_runtime_param_evaluation
from .types import ParamEvalOptions, ParamRuntimeContext

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


@dataclass(frozen=True)
class ParamQueryOptions:
    """Declare query evaluation and output ownership for a domain kernel.

    Parameters
    ----------
    evaluation : ParamEvalOptions, optional
        Interpolation, duplicate policy, and temporary query dimension.
    output_intent : str, optional
        Allowed values: ``'grid'`` and ``'trajectory'``. Grid queries restore caller dimensions; trajectories retain sequence
        semantics. One-dimensional grid queries also produce trajectories.
    mapped_dataset : bool, optional
        Whether the kernel maps the whole Dataset. Selected-payload kernels
        use False and retain the existing selected-payload validity policy.

    Notes
    -----
    Options do not alter the domain kernel or introduce numerical behavior.

    Examples
    --------
    >>> from tal.core.param_ops.query import ParamQueryOptions
    >>> opts = ParamQueryOptions(mapped_dataset=False)
    >>> assert opts.evaluation.method == 'linear'
    """

    evaluation: ParamEvalOptions = field(default_factory=ParamEvalOptions)
    output_intent: Literal["grid", "trajectory"] = "grid"
    mapped_dataset: bool = True


@dataclass(frozen=True)
class ParamQueryPlan:
    """Request-local query state supplied by :func:`prepare_param_query`.

    Read ``query``, ``query_dim``, ``param_map``, and ``trajectory`` to execute
    an aligned domain kernel. Pass the unchanged plan and kernel Dataset to
    :func:`finalize_param_query`. Construction and private state belong to TAL;
    callers must not mutate arrays retained by the plan.
    """

    _context: ParamRuntimeContext
    _evaluation: PreparedParamEvaluation
    _output_plan: QueryOutputPlan
    _options: ParamQueryOptions

    @property
    def query(self) -> xr.DataArray:
        """Return normalized labeled query values for the kernel."""
        return self._evaluation.grid.values

    @property
    def query_dim(self) -> str:
        """Return the temporary kernel query dimension name."""
        return self._evaluation.grid.query_dim

    @property
    def param_map(self) -> ParamMap:
        """Return endpoint indices, interpolation weights, and query validity."""
        return self._evaluation.param_map

    @property
    def trajectory(self) -> bool:
        """Return whether finalization retains TAL sequence semantics."""
        return (
            self._options.output_intent == "trajectory"
            or self._evaluation.grid.stacked_dims is None
        )


def _query_options(opts: ParamQueryOptions | None, *, owner: str) -> ParamQueryOptions:
    out = ParamQueryOptions() if opts is None else opts
    if not isinstance(out, ParamQueryOptions):
        raise TypeError(f"{owner}: opts must be ParamQueryOptions or None.")
    evaluation = coerce_eval_options(out.evaluation, owner=owner)
    if out.output_intent not in ("grid", "trajectory"):
        raise ValueError(f"{owner}: output_intent must be grid or trajectory.")
    if not isinstance(out.mapped_dataset, bool):
        raise TypeError(f"{owner}: mapped_dataset must be bool.")
    return replace(out, evaluation=evaluation)


def _preflight_query(context, query, *, opts: ParamQueryOptions, owner: str):
    assert_query_dim_safe(
        context.ds,
        sequence_dim=context.sequence_dim,
        query_dim=opts.evaluation.query_dim,
        owner=owner,
    )
    assert_reserved_metadata_safe(
        context.ds,
        reserved=("valid", "sample_index"),
        param_name=context.spec.name,
        owner=owner,
    )
    trajectory = (
        opts.output_intent == "trajectory"
        or not isinstance(query, xr.DataArray)
        or (len(set(query.dims) - set(context.batch_dims)) <= 1)
    )
    return preflight_query_output_namespace(
        context.ds,
        query,
        param_name=context.spec.name,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
        owner=owner,
        intent=opts.output_intent,
        generated_names=generated_query_coordinate_names(
            operation="evaluate",
            param_name=context.spec.name,
            size_name=context.sequence_size_coord,
            trajectory=trajectory,
            mapped_dataset=opts.mapped_dataset,
        ),
    )


def _prepare_param_query(context, query, *, opts, owner, prepared=None):
    opts = _query_options(opts, owner=owner)
    output = _preflight_query(context, query, opts=opts, owner=owner)
    evaluation = prepare_runtime_param_evaluation(
        context,
        query=query,
        options=ParamMapOptions(
            method=opts.evaluation.method,
            duplicate_policy=opts.evaluation.duplicate_policy,
        ),
        param_kind=context.param_kind,
        query_dim=opts.evaluation.query_dim,
        reuse=() if prepared is None else (prepared,),
    )
    output = replace(output, topology=evaluation.query_topology)
    return ParamQueryPlan(context, evaluation, output, opts)


def prepare_param_query(
    context: ParamRuntimeContext,
    query: xr.DataArray | np.ndarray | Sequence[float] | float,
    *,
    opts: ParamQueryOptions | None = None,
    owner: str,
) -> ParamQueryPlan:
    """Preflight output names and prepare a labeled domain query kernel.

    Parameters
    ----------
    context : tal.core.param_ops.types.ParamRuntimeContext
        Resolved source from ``resolve_param_runtime_context``.
    query : xarray.DataArray or numpy.ndarray or collections.abc.Sequence or float
        Values to evaluate in the source parameter coordinate.
    opts : ParamQueryOptions or None, optional
        Evaluation and output ownership declarations.
    owner : str
        Operation label for deterministic boundary errors.

    Returns
    -------
    ParamQueryPlan
        Normalized query and parameter map with opaque finalization state.

    Raises
    ------
    TypeError
        If the context or options have unsupported types.
    ValueError
        If options, query labels, or output names conflict.

    Notes
    -----
    Alignment follows dimension names and coordinate labels. Deterministic
    namespace conflicts are checked before map construction. Parameter mapping
    retains TAL's existing eager coordinate boundary; payloads are not computed.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.resolve import resolve_param_runtime_context
    >>> from tal.core.param_ops.query import prepare_param_query
    >>> source = AnalysisObject.from_data(
    ...     xr.Dataset({'v': ('sample', [0., 10.])},
    ...                coords={'time': ('sample', [0., 1.])}),
    ...     sequence_dim='sample', param_coord='time')
    >>> plan = prepare_param_query(resolve_param_runtime_context(source),
    ...                            [0.5], owner='example.query')
    >>> assert plan.trajectory and plan.query_dim == 'query'
    >>> assert plan.param_map.valid.all()
    """
    if not isinstance(context, ParamRuntimeContext):
        raise TypeError(f"{owner}: context must be ParamRuntimeContext.")
    return _prepare_param_query(context, query, opts=opts, owner=owner)


def _assert_kernel_query_names(plan: ParamQueryPlan, dataset: xr.Dataset) -> None:
    caller_names = set(plan._evaluation.query_topology.coordinates.coordinates)
    owner = plan._output_plan.owner
    payload_collisions = caller_names.intersection(dataset.data_vars)
    if payload_collisions:
        name = min(payload_collisions)
        raise ValueError(
            f"{owner}: query coordinate {name!r} collides with an output data variable."
        )
    surviving_dims = set(dataset.dims) - {plan.query_dim, *plan._context.batch_dims}
    dimension_collisions = caller_names.intersection(surviving_dims)
    if dimension_collisions:
        name = min(dimension_collisions)
        raise ValueError(
            f"{owner}: query coordinate {name!r} collides with a surviving core dimension."
        )


def _prepare_query_dataset(plan, dataset, *, copy_schema=True):
    _assert_kernel_query_names(plan, dataset)
    context = plan._context
    if plan._options.output_intent == "grid" and not plan.trajectory:
        dataset = assign_sampled_query_coordinate(
            dataset,
            query=plan.query,
            query_dim=plan.query_dim,
            name=context.spec.name,
        )
    return _prepare_param_output_dataset(
        context,
        dataset,
        query=plan.query,
        query_dim=plan.query_dim,
        valid_query=plan.param_map.valid,
        query_topology=plan._evaluation.query_topology,
        trajectory=plan.trajectory,
        owner=plan._output_plan.owner,
        output_plan=plan._output_plan,
        copy_schema=copy_schema,
    )


def finalize_param_query(
    plan: ParamQueryPlan,
    dataset: xr.Dataset,
    *,
    validate: bool = True,
) -> AnalysisObject:
    """Restore query topology and finalize a domain kernel's output Dataset.

    Parameters
    ----------
    plan : ParamQueryPlan
        Unchanged plan from ``prepare_param_query`` for this kernel.
    dataset : xarray.Dataset
        Kernel output using the plan's temporary query dimension. Preserve
        surviving source coordinates and indexes during kernel assembly.
    validate : bool, optional
        Run full validation after shared schema and topology finalization.

    Returns
    -------
    AnalysisObject
        Schema-consistent output with truthful roles and validity.

    Raises
    ------
    TypeError
        If the plan or Dataset has an unsupported type.
    ValueError
        If output topology or coordinate ownership conflicts with the plan.

    Notes
    -----
    Output-name ownership remains enforced with ``validate=False``. Source
    extension metadata is preserved. Finalization does not compute payloads.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.resolve import resolve_param_runtime_context
    >>> from tal.core.param_ops.query import prepare_param_query, finalize_param_query
    >>> source = AnalysisObject.from_data(
    ...     xr.Dataset({'v': ('sample', [0., 10.])},
    ...                coords={'time': ('sample', [0., 1.])}),
    ...     sequence_dim='sample', param_coord='time')
    >>> plan = prepare_param_query(resolve_param_runtime_context(source),
    ...                            [0.5], owner='example.query')
    >>> result = finalize_param_query(plan, xr.Dataset({'v': ('query', [5.])}))
    >>> assert result.as_dataset()['v'].dims == ('sample',)
    >>> assert result.as_dataset()['time'].data.tolist() == [0.5]
    """
    if not isinstance(plan, ParamQueryPlan) or not isinstance(dataset, xr.Dataset):
        raise TypeError("finalize_param_query: expected ParamQueryPlan and Dataset.")
    output = _prepare_query_dataset(plan, dataset)
    return _finalize_prepared_param_output(
        plan._context,
        output,
        validate=validate,
        trajectory=plan.trajectory,
        output_plan=plan._output_plan,
        query_topology=plan._evaluation.query_topology,
    )


__all__ = [
    "ParamQueryOptions",
    "ParamQueryPlan",
    "finalize_param_query",
    "prepare_param_query",
]
