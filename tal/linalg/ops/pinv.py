from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ...core.analysis_object import AnalysisObject
from ...core.var_naming import default_datavar_name
from ..array import Array
from ..plan import ArrayOperandContext
from ..result_type import resolve_binary_output_array_type_by_core_arity
from .linear_systems import (
    enforce_unchunked_linear_system_inputs,
    require_matrix_core_dims,
    require_square_matrix,
    run_with_linalgerror_normalization,
)
from .unary import build_strict_unary_plan, coerce_unary_operand, finalize_unary_output


@dataclass(frozen=True)
class PInvOptions:
    """Options for Moore-Penrose pseudoinverse.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    rcond: float | None = None
    hermitian: bool = False


@dataclass(frozen=True)
class PInvTopology:
    """Resolved pseudoinverse topology metadata.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    row_dim: str
    col_dim: str
    output_core_dims: tuple[str, str]


def coerce_pinv_options(opts: PInvOptions | None, *, owner: str) -> PInvOptions:
    """Validate/normalize pseudoinverse options.

    Parameters
    ----------
    opts : PInvOptions | None
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    PInvOptions
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if opts is None:
        return PInvOptions()
    if not isinstance(opts, PInvOptions):
        raise TypeError(f"{owner}: opts must be PInvOptions or None.")
    if opts.rcond is not None:
        if not isinstance(opts.rcond, (float, int)):
            raise TypeError(f"{owner}: opts.rcond must be float | int | None.")
        if not np.isfinite(float(opts.rcond)):
            raise ValueError(f"{owner}: opts.rcond must be finite when provided.")
        if float(opts.rcond) < 0.0:
            raise ValueError(f"{owner}: opts.rcond must be >= 0 when provided.")
    if not isinstance(opts.hermitian, bool):
        raise TypeError(f"{owner}: opts.hermitian must be bool.")
    return opts


def _resolve_pinv_topology(left: ArrayOperandContext, *, owner: str) -> PInvTopology:
    row_dim, col_dim = require_matrix_core_dims(left, owner=owner, operand_label="operand")
    return PInvTopology(
        row_dim=row_dim,
        col_dim=col_dim,
        output_core_dims=(col_dim, row_dim),
    )


def _enforce_hermitian_square(
    left: ArrayOperandContext,
    *,
    topology: PInvTopology,
    hermitian: bool,
    owner: str,
) -> None:
    if not hermitian:
        return
    row_size = left.data.sizes[topology.row_dim]
    col_size = left.data.sizes[topology.col_dim]
    require_square_matrix(
        left,
        row_dim=topology.row_dim,
        col_dim=topology.col_dim,
        owner=owner,
        message=(
            f"{owner}: opts.hermitian=True requires square matrix; "
            f"got sizes {row_size} and {col_size}."
        ),
    )


def compute_pinv_kernel(
    left: xr.DataArray,
    *,
    topology: PInvTopology,
    rcond: float | None,
    hermitian: bool,
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    """Kernel: compute Moore-Penrose pseudoinverse.

    Parameters
    ----------
    left : xr.DataArray
        Operand value participating in the operation.
    topology : PInvTopology, optional
        Resolved runtime context/payload used by this orchestration boundary.
    rcond : float | None, optional
        Numerical algorithm parameter forwarded to the kernel implementation.
    hermitian : bool, optional
        Numerical algorithm parameter forwarded to the kernel implementation.
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
    out = xr.apply_ufunc(
        np.linalg.pinv,
        left,
        input_core_dims=[[topology.row_dim, topology.col_dim]],
        output_core_dims=[[topology.col_dim, topology.row_dim]],
        vectorize=False,
        dask="forbidden",
        kwargs={"rcond": rcond, "hermitian": hermitian},
    )
    if output_var_name:
        out = out.rename(output_var_name)
    return out


def pinv(
    left: "Array | AnalysisObject | object",
    *,
    opts: PInvOptions | None = None,
) -> "Array":
    """Compute strict role-driven pseudoinverse.

    Parameters
    ----------
    left : Array | AnalysisObject | object
        Left/first operand.
    opts : PInvOptions | None, optional
        When ``None``, operation-specific defaults are resolved by internal option coercion. ``PInvOptions`` key fields: ``rcond`` (default None), ``hermitian`` (default False).

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
    >>> from tal.linalg import Matrix, pinv
    >>> from tal.linalg.ops.pinv import PInvOptions
    >>> A = Matrix(AnalysisObject.from_data(
    ...     xr.Dataset({"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 4.0]]])}, coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("row", "col"),
    ...     validate=True,
    ... ))
    >>> pinv(A, opts=PInvOptions(rcond=1e-8)).as_dataset()["datavar"].values.tolist()
    [[[0.5, 0.0], [0.0, 0.25]]]
    """
    owner = "linalg.pinv"
    options = coerce_pinv_options(opts, owner=owner)
    left_ao, _ = coerce_unary_operand(left, owner=owner)
    plan, left_op = build_strict_unary_plan(left_ao, owner=owner, allowed_core_arity=(1, 2))
    enforce_unchunked_linear_system_inputs(
        left_op.data,
        owner=owner,
        message=(
            "chunked pseudoinverse inputs are not supported; "
            "rechunk or materialize operands before pseudoinverse."
        ),
    )
    topology = _resolve_pinv_topology(left_op, owner=owner)
    _enforce_hermitian_square(
        left_op,
        topology=topology,
        hermitian=options.hermitian,
        owner=owner,
    )
    output_cls = resolve_binary_output_array_type_by_core_arity(left, output_core_dims=topology.output_core_dims)
    output_var_name = default_datavar_name()
    out = run_with_linalgerror_normalization(
        lambda: compute_pinv_kernel(
            left_op.data,
            topology=topology,
            rcond=options.rcond,
            hermitian=options.hermitian,
            output_var_name=output_var_name,
            owner=owner,
        ),
        owner=owner,
        message="pseudoinverse failed due to a singular or ill-conditioned matrix.",
    )
    return finalize_unary_output(
        plan=plan,
        operand=left_op,
        result=out,
        output_var_name=output_var_name,
        output_core_dims=topology.output_core_dims,
        output_cls=output_cls,
        owner=owner,
    )


__all__ = [
    "PInvOptions",
    "coerce_pinv_options",
    "compute_pinv_kernel",
    "pinv",
]
