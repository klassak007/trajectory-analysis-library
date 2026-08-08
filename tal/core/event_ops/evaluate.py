from __future__ import annotations

import numpy as np
import xarray as xr

from ..orchestration.axis_map import resolve_role_axis_map
from ..orchestration.inputs import coerce_analysis_object_input
from ..orchestration.resolve import resolve_param_runtime_context
from ..param_ops import ParamEvalOptions
from tal.utils.xarray_namespace import rename_dims_collision_safe
from .resolve import EventEvalContext
from .types import (
    AndNode,
    CompareNode,
    Condition,
    ConditionNode,
    CoordOperand,
    EvalMask,
    NotNode,
    OrNode,
    VarOperand,
)


def _ensure_numeric(da: xr.DataArray, *, owner: str, field: str) -> None:
    if np.issubdtype(np.dtype(da.dtype), np.number):
        return
    raise ValueError(f"{owner}: {field} must be numeric, got dtype {da.dtype!r}.")


def _align_operand_to_clock(
    operand: xr.DataArray,
    *,
    clock: xr.DataArray,
    owner: str,
    field: str,
) -> xr.DataArray:
    bad = [dim for dim in operand.dims if dim not in clock.dims]
    if bad:
        raise ValueError(f"{owner}: {field} has non-context dims {bad!r}; expected subset of {clock.dims!r}.")
    for dim in operand.dims:
        if not operand.get_index(dim).equals(clock.get_index(dim)):
            raise ValueError(f"{owner}: {field} labels for dim {dim!r} must match context clock labels.")
    try:
        out, _ = xr.broadcast(operand, clock)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{owner}: {field} could not be broadcast to context clock dims {clock.dims!r}."
        ) from exc
    return out.transpose(*clock.dims)


def _resolve_named_operand(
    operand: VarOperand | CoordOperand,
    *,
    context: EventEvalContext,
    owner: str,
) -> xr.DataArray:
    ds = context.runtime.ds
    if isinstance(operand, VarOperand):
        if operand.name not in ds.data_vars:
            raise ValueError(f"{owner}: data variable {operand.name!r} not found on context object.")
        return ds.data_vars[operand.name]
    if operand.name not in ds.coords:
        raise ValueError(f"{owner}: coordinate {operand.name!r} not found on context object.")
    return ds.coords[operand.name]


def _resolve_ao_operand(
    operand: object,
    *,
    context: EventEvalContext,
    owner: str,
) -> xr.DataArray:
    target = coerce_analysis_object_input(operand, owner=owner)
    try:
        target_runtime = resolve_param_runtime_context(target)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: failed to resolve AO operand runtime context.") from exc
    to_target = resolve_role_axis_map(
        context.runtime,
        target_runtime,
        source_clock=context.clock,
        owner=owner,
    )
    query = rename_dims_collision_safe(context.clock, mapping=to_target)
    opts = ParamEvalOptions(method=context.opts.ao_interp)
    try:
        aligned = target.param.at(
            query,
            on=target_runtime.spec.name,
            opts=opts,
            validate=False,
            sequence_dim=target_runtime.sequence_dim,
            batch_dims=target_runtime.batch_dims,
            sequence_size_coord=target_runtime.sequence_size_coord,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: failed to align AO operand on context clock.") from exc
    data_vars = list(aligned.unsafe_data.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            f"{owner}: AO operand must resolve to exactly one numeric data variable, got {len(data_vars)}."
        )
    resolved = aligned.unsafe_data.data_vars[data_vars[0]]
    to_context = {dst: src for src, dst in to_target.items()}
    return rename_dims_collision_safe(resolved, mapping=to_context)


def _broadcast_scalar_operand(
    operand: object,
    *,
    clock: xr.DataArray,
    owner: str,
    field: str,
) -> xr.DataArray:
    scalar_dtype = np.asarray(operand).dtype
    scalar_is_numeric = np.issubdtype(scalar_dtype, np.number)
    scalar_is_timedelta = np.issubdtype(scalar_dtype, np.timedelta64)
    if not scalar_is_numeric or scalar_is_timedelta:
        raise ValueError(f"{owner}: {field} must be numeric, got dtype {scalar_dtype!r}.")
    broadcast = xr.full_like(clock, operand, dtype=scalar_dtype)
    return broadcast.drop_attrs(deep=False).rename(None)


def _resolve_operand(
    operand: object,
    *,
    context: EventEvalContext,
    owner: str,
    field: str,
) -> xr.DataArray:
    if isinstance(operand, (VarOperand, CoordOperand)):
        base = _resolve_named_operand(operand, context=context, owner=owner)
        return _align_operand_to_clock(base, clock=context.clock, owner=owner, field=field)
    if isinstance(operand, xr.DataArray):
        return _align_operand_to_clock(operand, clock=context.clock, owner=owner, field=field)
    if np.isscalar(operand):
        return _broadcast_scalar_operand(operand, clock=context.clock, owner=owner, field=field)
    return _align_operand_to_clock(
        _resolve_ao_operand(operand, context=context, owner=owner),
        clock=context.clock,
        owner=owner,
        field=field,
    )


def _compare(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    op: str,
    context: EventEvalContext,
    owner: str,
) -> EvalMask:
    _ensure_numeric(left, owner=owner, field="left operand")
    _ensure_numeric(right, owner=owner, field="right operand")
    finite = xr.apply_ufunc(np.isfinite, left, dask="allowed") & xr.apply_ufunc(np.isfinite, right, dask="allowed")
    if op in {"eq", "ne"}:
        close = xr.apply_ufunc(
            np.isclose,
            left,
            right,
            kwargs={"rtol": context.opts.eq_rtol, "atol": context.opts.eq_atol, "equal_nan": False},
            dask="allowed",
        )
        truth = close if op == "eq" else ~close
    else:
        truth = {
            "lt": left < right,
            "le": left <= right,
            "gt": left > right,
            "ge": left >= right,
        }[op]
    determinate = finite.astype(bool)
    truth = truth.where(determinate, other=False).astype(bool)
    return EvalMask(truth=truth, determinate=determinate)


def _eval_compare(node: CompareNode, *, context: EventEvalContext, owner: str) -> EvalMask:
    left = _resolve_operand(node.left, context=context, owner=owner, field="left operand")
    right = _resolve_operand(node.right, context=context, owner=owner, field="right operand")
    return _compare(left, right, op=node.op, context=context, owner=owner)


def _eval_not(mask: EvalMask) -> EvalMask:
    determinate = mask.determinate.astype(bool)
    truth = (~mask.truth).where(determinate, other=False).astype(bool)
    return EvalMask(truth=truth, determinate=determinate)


def _eval_and(left: EvalMask, right: EvalMask) -> EvalMask:
    determinate = (
        (left.determinate & right.determinate)
        | (left.determinate & ~left.truth)
        | (right.determinate & ~right.truth)
    ).astype(bool)
    truth = (left.truth & right.truth).where(determinate, other=False).astype(bool)
    return EvalMask(truth=truth, determinate=determinate)


def _eval_or(left: EvalMask, right: EvalMask) -> EvalMask:
    determinate = (
        (left.determinate & right.determinate)
        | (left.determinate & left.truth)
        | (right.determinate & right.truth)
    ).astype(bool)
    truth = (left.truth | right.truth).where(determinate, other=False).astype(bool)
    return EvalMask(truth=truth, determinate=determinate)


def _eval_node(node: ConditionNode, *, context: EventEvalContext, owner: str) -> EvalMask:
    if isinstance(node, CompareNode):
        return _eval_compare(node, context=context, owner=owner)
    if isinstance(node, NotNode):
        return _eval_not(_eval_node(node.child, context=context, owner=owner))
    if isinstance(node, AndNode):
        return _eval_and(
            _eval_node(node.left, context=context, owner=owner),
            _eval_node(node.right, context=context, owner=owner),
        )
    if isinstance(node, OrNode):
        return _eval_or(
            _eval_node(node.left, context=context, owner=owner),
            _eval_node(node.right, context=context, owner=owner),
        )
    raise TypeError(f"{owner}: unsupported condition node type {type(node).__name__}.")


def evaluate_mask(
    condition: Condition,
    *,
    context: EventEvalContext,
    owner: str,
) -> xr.DataArray:
    """Evaluate a condition into the effective context mask.

    Parameters
    ----------
    condition : Condition
        Condition/expression used for event or mask evaluation.
    context : EventEvalContext, optional
        Resolved runtime context/payload used by this orchestration boundary.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    xr.DataArray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if not isinstance(condition, Condition):
        raise TypeError(f"{owner}: condition must be Condition.")
    evaluated = _eval_node(condition.node, context=context, owner=owner)
    effective = (evaluated.truth & evaluated.determinate & context.valid_mask).astype(bool)
    return effective.drop_attrs(deep=False)


__all__ = ["evaluate_mask"]
