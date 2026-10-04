from __future__ import annotations

import xarray as xr

from ...core.analysis_object import AnalysisObject
from ...core.dataset_ownership import analysis_object_dataset
from ...core.orchestration.inputs import coerce_operand
from ..array import Array
from ..finalize import ArrayFinalizeSpec, finalize_array_result
from ..plan import ArrayOperandContext, ArrayPlan, build_array_plan
from ..result_type import rewrap_binary_output_array, resolve_binary_output_array_type


def _build_unary_finalize_spec(
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


def _finalize_source_array(ao: AnalysisObject) -> Array:
    if type(ao) is Array:
        return ao
    return Array._from_unvalidated(analysis_object_dataset(ao))


def coerce_unary_operand(
    value: object,
    *,
    owner: str,
) -> tuple[AnalysisObject, type[Array]]:
    output_cls = resolve_binary_output_array_type(value)
    out = coerce_operand(value, owner=owner)
    if not isinstance(out, AnalysisObject):
        raise TypeError(f"{owner}: unary operand must be AO-like.")
    return out, output_cls


def build_strict_unary_plan(
    ao: AnalysisObject,
    *,
    owner: str,
    allowed_core_arity: tuple[int, ...] = (1, 2),
) -> tuple[ArrayPlan, ArrayOperandContext]:
    plan = build_array_plan(
        [ao],
        owner=owner,
        require_declared_roles=False,
        allowed_core_arity=allowed_core_arity,
        align_mode="exact",
        require_single_numeric_var=True,
    )
    return plan, plan.operands[0]


def finalize_unary_output(
    *,
    plan: ArrayPlan,
    operand: ArrayOperandContext,
    result: xr.DataArray,
    output_var_name: str,
    output_core_dims: tuple[str, ...],
    output_cls: type[Array],
    owner: str,
    validate: bool = True,
) -> Array:
    finalized = finalize_array_result(
        _finalize_source_array(operand.ao),
        result,
        spec=_build_unary_finalize_spec(
            output_var_name=output_var_name,
            sequence_dim=plan.sequence_dim,
            batch_dims=plan.batch_dims,
            output_core_dims=output_core_dims,
            shared_param_coord=plan.shared_param_coord,
            shared_sequence_size_coord=plan.shared_sequence_size_coord,
        ),
        optional_sources=(operand.data,),
        owner=owner,
        validate=validate,
    )
    return rewrap_binary_output_array(finalized, output_cls=output_cls)


__all__ = [
    "build_strict_unary_plan",
    "coerce_unary_operand",
    "finalize_unary_output",
]
