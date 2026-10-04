from __future__ import annotations

from collections.abc import Callable

import xarray as xr

from ...core.analysis_object import AnalysisObject
from ...core.dataset_ownership import analysis_object_dataset
from ...core.orchestration.inputs import coerce_operand
from ...core.var_naming import default_datavar_name
from ..array import Array
from ..finalize import ArrayFinalizeSpec, finalize_array_result
from ..plan import ArrayOperandContext, ArrayPlan, build_array_binary_plan
from ..result_type import rewrap_binary_output_array, resolve_binary_output_array_type


def _finalize_source_array(ao: AnalysisObject) -> Array:
    if type(ao) is Array:
        return ao
    return Array._from_unvalidated(analysis_object_dataset(ao))


def coerce_binary_operands(
    left: object,
    right: object,
    *,
    owner: str,
) -> tuple[AnalysisObject, AnalysisObject, type[Array]]:
    output_cls = resolve_binary_output_array_type(left)
    left_out = coerce_operand(left, owner=owner, label="left")
    right_out = coerce_operand(right, owner=owner, label="right")
    if not isinstance(left_out, AnalysisObject) or not isinstance(right_out, AnalysisObject):
        raise TypeError(f"{owner}: binary operands must be AO-like.")
    left_ao = left_out
    right_ao = right_out
    return left_ao, right_ao, output_cls


def build_strict_binary_plan(
    left_ao: AnalysisObject,
    right_ao: AnalysisObject,
    *,
    owner: str,
) -> ArrayPlan:
    return build_array_binary_plan(
        left_ao,
        right_ao,
        owner=owner,
        require_declared_roles=False,
        allowed_core_arity=(1, 2),
        align_mode="exact",
        require_single_numeric_var=True,
    )


def binary_output_var_name(left_var: str, right_var: str) -> str:
    _ = (left_var, right_var)
    return default_datavar_name()


def resolve_elementwise_output_core_dims(
    *,
    plan: ArrayPlan,
    left_op: ArrayOperandContext,
    right_op: ArrayOperandContext,
    owner: str,
) -> tuple[str, ...]:
    if plan.core_policy == "strict":
        if left_op.core_dims == right_op.core_dims:
            return left_op.core_dims
        raise ValueError(
            f"{owner}: elementwise binary ops require matching core_dims under core_policy='strict'; "
            f"got {left_op.core_dims!r} and {right_op.core_dims!r}."
        )
    return plan.output_core_dims


def compute_elementwise_binary(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    operator_fn: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    _ = owner
    out = operator_fn(left, right)
    if output_var_name:
        out = out.rename(output_var_name)
    return out


def _build_finalize_spec(
    *,
    output_var_name: str,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    output_core_dims: tuple[str, ...],
    shared_param_coord: str | None,
    shared_sequence_size_coord: str | None,
) -> ArrayFinalizeSpec:
    return ArrayFinalizeSpec(
        output_var_name=output_var_name,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=output_core_dims,
        param_name=shared_param_coord,
        size_name=shared_sequence_size_coord,
    )


def run_elementwise_binary(
    left: object,
    right: object,
    *,
    owner: str,
    op_name: str,
    operator_fn: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
    validate: bool = True,
) -> Array:
    left_ao, right_ao, output_cls = coerce_binary_operands(left, right, owner=owner)
    plan = build_strict_binary_plan(left_ao, right_ao, owner=owner)
    left_op, right_op = plan.operands
    output_core_dims = resolve_elementwise_output_core_dims(
        plan=plan,
        left_op=left_op,
        right_op=right_op,
        owner=owner,
    )
    output_var_name = binary_output_var_name(left_op.var_name, right_op.var_name)
    out = compute_elementwise_binary(
        left_op.data,
        right_op.data,
        operator_fn=operator_fn,
        output_var_name=output_var_name,
        owner=owner,
    )
    return finalize_binary_output(
        plan=plan,
        left_op=left_op,
        right_op=right_op,
        result=out,
        output_var_name=output_var_name,
        output_core_dims=output_core_dims,
        output_cls=output_cls,
        owner=owner,
        validate=validate,
    )


def finalize_binary_output(
    *,
    plan: ArrayPlan,
    left_op: ArrayOperandContext,
    right_op: ArrayOperandContext,
    result: xr.DataArray,
    output_var_name: str,
    output_core_dims: tuple[str, ...],
    output_cls: type[Array],
    owner: str,
    validate: bool = True,
) -> Array:
    finalized = finalize_array_result(
        _finalize_source_array(left_op.ao),
        result,
        spec=_build_finalize_spec(
            output_var_name=output_var_name,
            sequence_dim=plan.sequence_dim,
            batch_dims=plan.batch_dims,
            output_core_dims=output_core_dims,
            shared_param_coord=plan.shared_param_coord,
            shared_sequence_size_coord=plan.shared_sequence_size_coord,
        ),
        optional_sources=(left_op.data, right_op.data),
        owner=owner,
        validate=validate,
    )
    return rewrap_binary_output_array(finalized, output_cls=output_cls)


__all__ = [
    "binary_output_var_name",
    "build_strict_binary_plan",
    "compute_elementwise_binary",
    "coerce_binary_operands",
    "finalize_binary_output",
    "run_elementwise_binary",
    "resolve_elementwise_output_core_dims",
]
