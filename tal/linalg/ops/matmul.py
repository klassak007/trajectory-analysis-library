from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from ...core.analysis_object import AnalysisObject
from ..array import Array
from ..plan import ArrayOperandContext
from ..result_type import resolve_binary_output_array_type_by_core_arity
from .binops import (
    binary_output_var_name,
    build_strict_binary_plan,
    coerce_binary_operands,
    finalize_binary_output,
)


@dataclass(frozen=True)
class MatmulOptions:
    """Options for strict role-driven matrix multiplication.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    strict_core: bool = True
    allow_vector_row_convention: bool = True


def coerce_matmul_options(opts: MatmulOptions | None, *, owner: str) -> MatmulOptions:
    """Validate/normalize matmul options.

    Parameters
    ----------
    opts : MatmulOptions | None
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    MatmulOptions
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if opts is None:
        return MatmulOptions()
    if not isinstance(opts, MatmulOptions):
        raise TypeError(f"{owner}: opts must be MatmulOptions or None.")
    if not isinstance(opts.strict_core, bool):
        raise ValueError(f"{owner}: opts.strict_core must be bool.")
    if not isinstance(opts.allow_vector_row_convention, bool):
        raise ValueError(f"{owner}: opts.allow_vector_row_convention must be bool.")
    if not opts.strict_core:
        raise ValueError(f"{owner}: opts.strict_core=False is not supported.")
    return opts


def _validate_vector_row_policy(
    left_core_dims: tuple[str, ...],
    right_core_dims: tuple[str, ...],
    *,
    allow_vector_row_convention: bool,
    owner: str,
) -> None:
    if allow_vector_row_convention:
        return
    if len(left_core_dims) == 1 and len(right_core_dims) == 2:
        raise ValueError(
            f"{owner}: vector-matrix row convention is disabled by opts.allow_vector_row_convention=False."
        )


def _resolve_matmul_topology(
    left: ArrayOperandContext,
    right: ArrayOperandContext,
    *,
    owner: str,
) -> tuple[str, tuple[str, ...]]:
    contract_dim = left.core_dims[-1]
    right_leading = right.core_dims[0]
    if contract_dim != right_leading:
        raise ValueError(
            f"{owner}: contraction requires left core trailing dim to equal right core leading dim; "
            f"got {contract_dim!r} and {right_leading!r}."
        )
    output_core_dims = left.core_dims[:-1] + right.core_dims[1:]
    if len(set(output_core_dims)) != len(output_core_dims):
        raise ValueError(f"{owner}: output core_dims are ambiguous (duplicate dims) {output_core_dims!r}.")
    if contract_dim not in left.data.dims:
        raise ValueError(f"{owner}: left operand is missing contraction dim {contract_dim!r}.")
    if contract_dim not in right.data.dims:
        raise ValueError(f"{owner}: right operand is missing contraction dim {contract_dim!r}.")
    return contract_dim, output_core_dims


def compute_matmul(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    contract_dim: str,
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    """Kernel: matrix multiply over resolved contraction dimension.

    Parameters
    ----------
    left : xr.DataArray
        Operand value participating in the operation.
    right : xr.DataArray
        Operand value participating in the operation.
    contract_dim : str, optional
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
    out = xr.dot(left, right, dim=[contract_dim])
    if output_var_name:
        out = out.rename(output_var_name)
    return out


def matmul(
    left: "Array | AnalysisObject | object",
    right: "Array | AnalysisObject | object",
    *,
    opts: MatmulOptions | None = None,
) -> "Array":
    """Compute strict role-driven matrix multiplication.

    Parameters
    ----------
    left : Array | AnalysisObject | object
        Left/first operand.
    right : Array | AnalysisObject | object
        Right/second operand.
    opts : MatmulOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``MatmulOptions`` key fields: ``strict_core`` (default True), ``allow_vector_row_convention`` (default True).

    Returns
    -------
    Array
        Operation result preserving TAL semantic/topology guarantees.

    Raises
    ------
    TypeError
        If option payload types are invalid for this API.
    ValueError
        If option values violate fail-closed semantic/layout constraints.

    Notes
    -----
    Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.linalg import Matrix, Vector, matmul
    >>> from tal.linalg.ops.matmul import MatmulOptions
    >>> A = Matrix(AnalysisObject.from_data(
    ...     xr.Dataset({"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 3.0]]])}, coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("row", "col"),
    ...     validate=True,
    ... ))
    >>> v = Vector(AnalysisObject.from_data(
    ...     xr.Dataset({"v": (("sample", "col"), [[4.0, 5.0]])}, coords={"sample": [0], "col": ["c0", "c1"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("col",),
    ...     validate=True,
    ... ))
    >>> matmul(A, v, opts=MatmulOptions()).unsafe_data["datavar"].values.tolist()
    [[8.0, 15.0]]
    """
    owner = "linalg.matmul"
    options = coerce_matmul_options(opts, owner=owner)
    left_ao, right_ao, _ = coerce_binary_operands(left, right, owner=owner)
    plan = build_strict_binary_plan(left_ao, right_ao, owner=owner)
    left_op, right_op = plan.operands
    _validate_vector_row_policy(
        left_op.core_dims,
        right_op.core_dims,
        allow_vector_row_convention=options.allow_vector_row_convention,
        owner=owner,
    )
    contract_dim, output_core_dims = _resolve_matmul_topology(left_op, right_op, owner=owner)
    output_cls = resolve_binary_output_array_type_by_core_arity(left, output_core_dims=output_core_dims)
    output_var_name = binary_output_var_name(left_op.var_name, right_op.var_name)
    out = compute_matmul(
        left_op.data,
        right_op.data,
        contract_dim=contract_dim,
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
    "MatmulOptions",
    "coerce_matmul_options",
    "compute_matmul",
    "matmul",
]
