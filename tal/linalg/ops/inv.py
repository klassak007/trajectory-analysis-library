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
class InvTopology:
    """Resolved inverse topology metadata.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    row_dim: str
    col_dim: str
    output_core_dims: tuple[str, str]

def _resolve_inv_topology(left: ArrayOperandContext, *, owner: str) -> InvTopology:
    row_dim, col_dim = require_matrix_core_dims(left, owner=owner, operand_label="left operand")
    require_square_matrix(
        left,
        row_dim=row_dim,
        col_dim=col_dim,
        owner=owner,
        message=(
            f"{owner}: inverse requires square left matrix; "
            f"got sizes {left.data.sizes[row_dim]} and {left.data.sizes[col_dim]}."
        ),
    )
    return InvTopology(
        row_dim=row_dim,
        col_dim=col_dim,
        output_core_dims=(col_dim, row_dim),
    )


def compute_inv_kernel(
    left: xr.DataArray,
    *,
    topology: InvTopology,
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    """Kernel: compute matrix inverse for resolved topology.

    Parameters
    ----------
    left : xr.DataArray
        Operand value participating in the operation.
    topology : InvTopology, optional
        Resolved runtime context/payload used by this orchestration boundary.
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
        np.linalg.inv,
        left,
        input_core_dims=[[topology.row_dim, topology.col_dim]],
        output_core_dims=[[topology.col_dim, topology.row_dim]],
        vectorize=False,
        dask="forbidden",
    )
    if output_var_name:
        out = out.rename(output_var_name)
    return out


def inv(
    left: "Array | AnalysisObject | object",
) -> "Array":
    """Compute strict matrix inverse.

    Parameters
    ----------
    left
        AO-like matrix operand.

    Returns
    -------
    Array
        Inverse output with swapped core dimensions.

    Notes
    -----
    Inverse requires square matrix topology and unchunked kernel inputs.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.linalg import Matrix, inv
    >>> A = Matrix(AnalysisObject.from_data(
    ...     xr.Dataset(
    ...         {"A": (("sample", "row", "col"), [[[2.0, 0.0], [0.0, 4.0]]])},
    ...         coords={"sample": [0], "row": ["r0", "r1"], "col": ["c0", "c1"]},
    ...     ),
    ...     sequence_dim="sample",
    ...     core_dims=("row", "col"),
    ...     validate=True,
    ... ))
    >>> inv(A).unsafe_data["datavar"].to_numpy().tolist()
    [[[0.5, 0.0], [0.0, 0.25]]]
    """
    owner = "linalg.inv"
    left_ao, _ = coerce_unary_operand(left, owner=owner)
    plan, left_op = build_strict_unary_plan(left_ao, owner=owner, allowed_core_arity=(1, 2))
    enforce_unchunked_linear_system_inputs(
        left_op.data,
        owner=owner,
        message=(
            "chunked inverse inputs are not supported; "
            "rechunk or materialize operands before inverse."
        ),
    )
    topology = _resolve_inv_topology(left_op, owner=owner)
    output_cls = resolve_binary_output_array_type_by_core_arity(left, output_core_dims=topology.output_core_dims)
    output_var_name = default_datavar_name()
    out = run_with_linalgerror_normalization(
        lambda: compute_inv_kernel(
            left_op.data,
            topology=topology,
            output_var_name=output_var_name,
            owner=owner,
        ),
        owner=owner,
        message="inverse failed due to a singular or ill-conditioned matrix.",
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
    "compute_inv_kernel",
    "inv",
]
