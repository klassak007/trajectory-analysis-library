from __future__ import annotations

from ..analysis_object import AnalysisObject
from ..event_ops.types import Condition
from ..event_ops.types import CoordOperand, VarOperand
from .finalize import finalize_binary_ao_result, finalize_unary_ao_result
from .kernel import call_binary_ufunc_kernel, call_unary_ufunc_kernel
from .orchestrate import (
    prepare_binary_ao_context,
    prepare_unary_ao_context,
    require_condition,
    validate_condition_compare_operands,
)
from .registry import classify_ufunc_family, get_xarray_ufunc

_COMPARE_OP_BY_UFUNC = {
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "equal": "eq",
    "not_equal": "ne",
}


def _logical_unary(name: str, value: object, *, owner: str) -> Condition:
    if name != "logical_not":
        raise TypeError(f"{owner}: unary logical ufunc {name!r} is not supported.")
    return ~require_condition(value, owner=owner, side="operand")


def _logical_binary(name: str, left: object, right: object, *, owner: str) -> Condition:
    left_cond = require_condition(left, owner=owner, side="left")
    right_cond = require_condition(right, owner=owner, side="right")
    if name == "logical_and":
        return left_cond & right_cond
    if name == "logical_or":
        return left_cond | right_cond
    if name == "logical_xor":
        return left_cond ^ right_cond
    raise TypeError(f"{owner}: binary logical ufunc {name!r} is not supported.")


def _comparison(name: str, left: object, right: object, *, owner: str) -> Condition:
    validate_condition_compare_operands(left, right, owner=owner)
    op = _COMPARE_OP_BY_UFUNC[name]
    return Condition.compare(left, op, right)


def _should_build_condition_comparison(left: object, right: object) -> bool:
    condition_types = (AnalysisObject, VarOperand, CoordOperand)
    return isinstance(left, condition_types) or isinstance(right, condition_types)


def apply_unary_ufunc(
    name: str,
    value: object,
    *,
    owner: str,
    validate: bool = True,
) -> object:
    family = classify_ufunc_family(name, owner=owner)
    if family == "comparison":
        raise TypeError(f"{owner}: comparison ufunc {name!r} requires two operands.")
    if family == "logical":
        return _logical_unary(name, value, owner=owner)

    ufunc = get_xarray_ufunc(name, owner=owner)
    runtime = prepare_unary_ao_context(value, owner=owner)
    result = call_unary_ufunc_kernel(ufunc, runtime.operand, owner=owner)
    if runtime.source is None:
        return result
    return finalize_unary_ao_result(runtime.source, result, owner=owner, validate=validate)


def apply_binary_ufunc(
    name: str,
    left: object,
    right: object,
    *,
    owner: str,
    validate: bool = True,
) -> object:
    family = classify_ufunc_family(name, owner=owner)
    if family == "comparison":
        if _should_build_condition_comparison(left, right):
            return _comparison(name, left, right, owner=owner)
    if family == "logical":
        return _logical_binary(name, left, right, owner=owner)

    ufunc = get_xarray_ufunc(name, owner=owner)
    runtime = prepare_binary_ao_context(left, right, owner=owner)
    result = call_binary_ufunc_kernel(ufunc, runtime.left, runtime.right, owner=owner)
    if runtime.source is None:
        return result
    return finalize_binary_ao_result(
        runtime.source,
        result,
        owner=owner,
        validate=validate,
        output_core_dims=runtime.output_core_dims,
        rewrap_context=runtime.rewrap_context,
    )


__all__ = [
    "apply_binary_ufunc",
    "apply_unary_ufunc",
]
