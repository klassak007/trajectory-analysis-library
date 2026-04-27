from __future__ import annotations

from operator import add as _op_add

from ...core.analysis_object import AnalysisObject
from ..array import Array
from .binops import run_elementwise_binary


def add(
    left: "Array | AnalysisObject | object",
    right: "Array | AnalysisObject | object",
) -> Array:
    """Compute strict elementwise addition.

    Parameters
    ----------
    left, right
        AO-like operands with compatible topology/core semantics.

    Returns
    -------
    Array
        Added output with resolved topology.

    Notes
    -----
    Alignment is name/label-aware. Default result variable naming uses the
    centralized ``datavar`` policy.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.linalg import Array, add
    >>> left = Array(AnalysisObject.from_data(
    ...     xr.Dataset({"x": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... ))
    >>> right = Array(AnalysisObject.from_data(
    ...     xr.Dataset({"x": ("sample", [3.0, 4.0])}, coords={"sample": [0, 1]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... ))
    >>> out = add(left, right)
    >>> out.unsafe_data["datavar"].to_numpy().tolist()
    [4.0, 6.0]
    """
    return run_elementwise_binary(
        left,
        right,
        owner="linalg.add",
        op_name="add",
        operator_fn=_op_add,
    )


__all__ = [
    "add",
]
