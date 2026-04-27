from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

from ...core.analysis_object import AnalysisObject
from ..array import Array
from ..plan import ArrayOperandContext, ArrayPlan
from ..result_type import resolve_binary_output_array_type_by_core_arity
from .binops import (
    binary_output_var_name,
    build_strict_binary_plan,
    coerce_binary_operands,
    finalize_binary_output,
)
from .linear_systems import (
    enforce_unchunked_linear_system_inputs,
    require_matrix_core_dims,
    require_square_matrix,
    run_with_linalgerror_normalization,
)
from .solve_backends import LSTSQ_BACKEND_NUMPY_ROW, lstsq_solution_backend


@dataclass(frozen=True)
class SolveOptions:
    """Options for linear solve execution.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    method: Literal["auto", "solve", "lstsq"] = "auto"
    rcond: float | None = None


@dataclass(frozen=True)
class SolveTopology:
    """Resolved solve topology metadata.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    contract_dim: str
    solution_dim: str
    rhs_extra_dim: str | None
    output_core_dims: tuple[str, ...]


@dataclass(frozen=True)
class SolveRuntime:
    """Prepared solve runtime bundle.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    plan: ArrayPlan
    left_op: ArrayOperandContext
    right_op: ArrayOperandContext
    topology: SolveTopology
    method: Literal["solve", "lstsq"]
    rcond: float | None
    output_var_name: str
    output_cls: type[Array]


def coerce_solve_options(opts: SolveOptions | None, *, owner: str) -> SolveOptions:
    """Validate/normalize solve options.

    Parameters
    ----------
    opts : SolveOptions | None
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    SolveOptions
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if opts is None:
        return SolveOptions()
    if not isinstance(opts, SolveOptions):
        raise TypeError(f"{owner}: opts must be SolveOptions or None.")
    if opts.method not in ("auto", "solve", "lstsq"):
        raise ValueError(f"{owner}: opts.method must be one of 'auto', 'solve', 'lstsq'.")
    if opts.rcond is not None:
        if not isinstance(opts.rcond, (float, int)):
            raise TypeError(f"{owner}: opts.rcond must be float | int | None.")
        if not np.isfinite(float(opts.rcond)):
            raise ValueError(f"{owner}: opts.rcond must be finite when provided.")
        if float(opts.rcond) < 0.0:
            raise ValueError(f"{owner}: opts.rcond must be >= 0 when provided.")
    return opts


def _resolve_solve_topology(
    left: ArrayOperandContext,
    right: ArrayOperandContext,
    *,
    owner: str,
) -> SolveTopology:
    contract_dim, solution_dim = require_matrix_core_dims(left, owner=owner, operand_label="left operand")
    if len(right.core_dims) not in (1, 2):
        raise ValueError(
            f"{owner}: rhs operand must have one or two core dims; got {right.core_dims!r}."
        )
    if right.core_dims[0] != contract_dim:
        raise ValueError(
            f"{owner}: solve requires left core leading dim to equal rhs core leading dim; "
            f"got {contract_dim!r} and {right.core_dims[0]!r}."
        )
    rhs_extra_dim = right.core_dims[1] if len(right.core_dims) == 2 else None
    output_core_dims = (solution_dim,) if rhs_extra_dim is None else (solution_dim, rhs_extra_dim)
    if len(set(output_core_dims)) != len(output_core_dims):
        raise ValueError(f"{owner}: output core_dims are ambiguous (duplicate dims) {output_core_dims!r}.")
    return SolveTopology(
        contract_dim=contract_dim,
        solution_dim=solution_dim,
        rhs_extra_dim=rhs_extra_dim,
        output_core_dims=output_core_dims,
    )


def _effective_solve_method(
    opts: SolveOptions,
    *,
    square_left: bool,
    owner: str,
) -> Literal["solve", "lstsq"]:
    _ = owner
    if opts.method == "auto":
        return "solve" if square_left else "lstsq"
    return opts.method


def _rhs_input_core_dims(topology: SolveTopology) -> list[str]:
    if topology.rhs_extra_dim is None:
        return [topology.contract_dim]
    return [topology.contract_dim, topology.rhs_extra_dim]


def _output_core_dims(topology: SolveTopology) -> list[str]:
    if topology.rhs_extra_dim is None:
        return [topology.solution_dim]
    return [topology.solution_dim, topology.rhs_extra_dim]


def _solve_numpy(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rhs_is_vector: bool,
) -> np.ndarray:
    if rhs_is_vector:
        return np.linalg.solve(a, b[..., None])[..., 0]
    return np.linalg.solve(a, b)


def compute_solve_kernel(
    left: xr.DataArray,
    rhs: xr.DataArray,
    *,
    topology: SolveTopology,
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    _ = owner
    # Keep solve dtype inferred from NumPy output to avoid integer truncation.
    out = xr.apply_ufunc(
        _solve_numpy,
        left,
        rhs,
        input_core_dims=[[topology.contract_dim, topology.solution_dim], _rhs_input_core_dims(topology)],
        output_core_dims=[_output_core_dims(topology)],
        kwargs={"rhs_is_vector": topology.rhs_extra_dim is None},
        vectorize=False,
        dask="forbidden",
    )
    if output_var_name:
        out = out.rename(output_var_name)
    return out


def compute_lstsq_kernel(
    left: xr.DataArray,
    rhs: xr.DataArray,
    *,
    topology: SolveTopology,
    rcond: float | None,
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    _ = owner
    # Keep lstsq dtype inferred from NumPy output to avoid integer truncation.
    out = xr.apply_ufunc(
        lstsq_solution_backend,
        left,
        rhs,
        input_core_dims=[[topology.contract_dim, topology.solution_dim], _rhs_input_core_dims(topology)],
        output_core_dims=[_output_core_dims(topology)],
        vectorize=True,
        dask="forbidden",
        kwargs={"rcond": rcond, "backend": LSTSQ_BACKEND_NUMPY_ROW},
    )
    if output_var_name:
        out = out.rename(output_var_name)
    return out


def compute_solve(
    left: xr.DataArray,
    rhs: xr.DataArray,
    *,
    topology: SolveTopology,
    method: Literal["solve", "lstsq"],
    rcond: float | None,
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    if method == "solve":
        return compute_solve_kernel(
            left,
            rhs,
            topology=topology,
            output_var_name=output_var_name,
            owner=owner,
        )
    return compute_lstsq_kernel(
        left,
        rhs,
        topology=topology,
        rcond=rcond,
        output_var_name=output_var_name,
        owner=owner,
    )


def _validate_solve_sequence_and_chunked(
    plan: ArrayPlan,
    left_op: ArrayOperandContext,
    right_op: ArrayOperandContext,
    *,
    owner: str,
) -> None:
    _ = plan
    enforce_unchunked_linear_system_inputs(
        left_op.data,
        right_op.data,
        owner=owner,
        message=(
            "chunked solve inputs are not supported; "
            "rechunk or materialize operands before solve."
        ),
    )


def _enforce_solve_square_policy_if_requested(
    options: SolveOptions,
    left_op: ArrayOperandContext,
    topology: SolveTopology,
    *,
    owner: str,
) -> None:
    if options.method != "solve":
        return
    require_square_matrix(
        left_op,
        row_dim=topology.contract_dim,
        col_dim=topology.solution_dim,
        owner=owner,
        message=(
            f"{owner}: opts.method='solve' requires square left matrix; "
            "use opts.method='lstsq' or opts.method='auto' for non-square systems."
        ),
    )


def _prepare_solve_runtime(
    left: "Array | AnalysisObject | object",
    right: "Array | AnalysisObject | object",
    *,
    opts: SolveOptions | None,
    owner: str,
) -> SolveRuntime:
    options = coerce_solve_options(opts, owner=owner)
    left_ao, right_ao, _ = coerce_binary_operands(left, right, owner=owner)
    plan = build_strict_binary_plan(left_ao, right_ao, owner=owner)
    left_op, right_op = plan.operands
    _validate_solve_sequence_and_chunked(plan, left_op, right_op, owner=owner)
    topology = _resolve_solve_topology(left_op, right_op, owner=owner)
    square_left = left_op.data.sizes[topology.contract_dim] == left_op.data.sizes[topology.solution_dim]
    _enforce_solve_square_policy_if_requested(options, left_op, topology, owner=owner)
    method = _effective_solve_method(options, square_left=square_left, owner=owner)
    output_var_name = binary_output_var_name(left_op.var_name, right_op.var_name)
    output_cls = resolve_binary_output_array_type_by_core_arity(
        left,
        output_core_dims=topology.output_core_dims,
    )
    return SolveRuntime(
        plan=plan,
        left_op=left_op,
        right_op=right_op,
        topology=topology,
        method=method,
        rcond=options.rcond,
        output_var_name=output_var_name,
        output_cls=output_cls,
    )


def _compute_solve_result(runtime: SolveRuntime, *, owner: str) -> xr.DataArray:
    return run_with_linalgerror_normalization(
        lambda: compute_solve(
            runtime.left_op.data,
            runtime.right_op.data,
            topology=runtime.topology,
            method=runtime.method,
            rcond=runtime.rcond,
            output_var_name=runtime.output_var_name,
            owner=owner,
        ),
        owner=owner,
        message="solve failed due to a singular or ill-conditioned linear system.",
    )


def _finalize_solve_result(
    runtime: SolveRuntime,
    result: xr.DataArray,
    *,
    owner: str,
) -> Array:
    return finalize_binary_output(
        plan=runtime.plan,
        left_op=runtime.left_op,
        right_op=runtime.right_op,
        result=result,
        output_var_name=runtime.output_var_name,
        output_core_dims=runtime.topology.output_core_dims,
        output_cls=runtime.output_cls,
        owner=owner,
    )


def solve(
    left: "Array | AnalysisObject | object",
    rhs: "Array | AnalysisObject | object",
    *,
    opts: SolveOptions | None = None,
) -> "Array":
    """Solve ``left @ x = rhs`` with strict role-driven semantics.

    Parameters
    ----------
    left : Array | AnalysisObject | object
        Left/first operand.
    rhs : Array | AnalysisObject | object
        Right-hand-side vector or matrix in ``A @ x = rhs``.
    opts : SolveOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``SolveOptions`` key fields: ``method`` (default 'auto'), ``rcond`` (default None).

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
    >>> from tal.linalg import Matrix, Vector, solve
    >>> from tal.linalg.ops.solve import SolveOptions
    >>> A = Matrix(AnalysisObject.from_data(
    ...     xr.Dataset({"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 3.0]]])}, coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("row", "col"),
    ...     validate=True,
    ... ))
    >>> b = Vector(AnalysisObject.from_data(
    ...     xr.Dataset({"b": (("sample", "row"), [[8.0, 15.0]])}, coords={"sample": [0], "row": ["r0", "r1"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("row",),
    ...     validate=True,
    ... ))
    >>> solve(A, b, opts=SolveOptions(method="solve")).unsafe_data["datavar"].values.tolist()
    [[4.0, 5.0]]
    """
    owner = "linalg.solve"
    runtime = _prepare_solve_runtime(left, rhs, opts=opts, owner=owner)
    result = _compute_solve_result(runtime, owner=owner)
    return _finalize_solve_result(runtime, result, owner=owner)


__all__ = [
    "SolveOptions",
    "coerce_solve_options",
    "compute_lstsq_kernel",
    "compute_solve",
    "compute_solve_kernel",
    "solve",
]
