from __future__ import annotations

import numpy as np
import xarray as xr

from ...core.analysis_object import AnalysisObject
from ...core.var_naming import default_datavar_name
from ..array import Array
from ..result_type import resolve_binary_output_array_type_by_core_arity
from .unary import build_strict_unary_plan, coerce_unary_operand, finalize_unary_output
from .vector_ops import coerce_norm_ord, require_vector_core_dim


def compute_norm(
    operand: xr.DataArray,
    *,
    core_dim: str,
    ord: int | float | None,
    output_var_name: str | None,
    owner: str,
) -> xr.DataArray:
    """Kernel: compute vector norm over resolved core dimension.

    Parameters
    ----------
    operand : xr.DataArray
        Operand/component input consumed by this operation.
    core_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    ord : int | float | None, optional
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
        np.linalg.norm,
        operand,
        input_core_dims=[[core_dim]],
        output_core_dims=[[]],
        vectorize=False,
        dask="allowed",
        kwargs={"ord": ord, "axis": -1},
    )
    if output_var_name:
        out = out.rename(output_var_name)
    return out


def norm(
    left: "Array | AnalysisObject | object",
    *,
    ord: int | float | None = 2,
) -> Array:
    """Compute strict vector norm.

    Parameters
    ----------
    left
        AO-like vector operand.
    ord
        Norm order (forwarded to NumPy norm kernel).

    Returns
    -------
    Array
        Scalar-core array result.

    Notes
    -----
    Norm uses strict vector-core checks and label-aware alignment.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.linalg import Vector, norm
    >>> vec = Vector(AnalysisObject.from_data(
    ...     xr.Dataset({"v": (("sample", "axis"), [[3.0, 4.0, 0.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
    ...     sequence_dim="sample",
    ...     core_dims=("axis",),
    ...     validate=True,
    ... ))
    >>> norm(vec).as_dataset()["datavar"].to_numpy().tolist()
    [5.0]
    """
    owner = "linalg.norm"
    norm_ord = coerce_norm_ord(ord, owner=owner)
    left_ao, _ = coerce_unary_operand(left, owner=owner)
    plan, left_op = build_strict_unary_plan(left_ao, owner=owner, allowed_core_arity=(1, 2))
    core_dim = require_vector_core_dim(left_op, owner=owner, operand_label="operand")
    output_var_name = default_datavar_name()
    output_core_dims: tuple[str, ...] = ()
    output_cls = resolve_binary_output_array_type_by_core_arity(left, output_core_dims=output_core_dims)
    out = compute_norm(
        left_op.data,
        core_dim=core_dim,
        ord=norm_ord,
        output_var_name=output_var_name,
        owner=owner,
    )
    return finalize_unary_output(
        plan=plan,
        operand=left_op,
        result=out,
        output_var_name=output_var_name,
        output_core_dims=output_core_dims,
        output_cls=output_cls,
        owner=owner,
    )


__all__ = [
    "compute_norm",
    "norm",
]
