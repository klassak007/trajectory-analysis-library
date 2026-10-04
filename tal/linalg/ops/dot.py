from __future__ import annotations

import xarray as xr

from ...core.analysis_object import AnalysisObject
from ..array import Array
from ..result_type import resolve_binary_output_array_type_by_core_arity
from .binops import (
    binary_output_var_name,
    build_strict_binary_plan,
    coerce_binary_operands,
    finalize_binary_output,
)
from .vector_ops import require_matching_vector_core_dim


def compute_dot(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    core_dim: str,
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    """Kernel: compute vector dot product over a resolved core dimension.

    Parameters
    ----------
    left : xr.DataArray
        Operand value participating in the operation.
    right : xr.DataArray
        Operand value participating in the operation.
    core_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    output_var_name : str | None, optional
        Output naming metadata used during finalization.
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
    _ = owner
    out = xr.dot(left, right, dim=[core_dim])
    if output_var_name:
        out = out.rename(output_var_name)
    return out


def dot(
    left: "Array | AnalysisObject | object",
    right: "Array | AnalysisObject | object",
) -> Array:
    """Compute strict vector dot product.

    Parameters
    ----------
    left, right
        AO-like vector operands with matching core dimension names.

    Returns
    -------
    Array
        Scalar-core array result.

    Notes
    -----
    Dot uses label-aware alignment and strict role/topology checks.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.linalg import Vector, dot
    >>> left = Vector(AnalysisObject.from_data(
    ...     xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... ))
    >>> right = Vector(AnalysisObject.from_data(
    ...     xr.Dataset({"v": (("sample", "axis"), [[4.0, 5.0, 6.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... ))
    >>> dot(left, right).as_dataset()["datavar"].to_numpy().tolist()
    [32.0]
    """
    owner = "linalg.dot"
    left_ao, right_ao, _ = coerce_binary_operands(left, right, owner=owner)
    plan = build_strict_binary_plan(left_ao, right_ao, owner=owner)
    left_op, right_op = plan.operands
    core_dim = require_matching_vector_core_dim(left_op, right_op, owner=owner)
    output_var_name = binary_output_var_name(left_op.var_name, right_op.var_name)
    output_core_dims: tuple[str, ...] = ()
    output_cls = resolve_binary_output_array_type_by_core_arity(left, output_core_dims=output_core_dims)
    out = compute_dot(
        left_op.data,
        right_op.data,
        core_dim=core_dim,
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
    )


__all__ = [
    "compute_dot",
    "dot",
]
