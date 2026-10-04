from __future__ import annotations

from operator import sub as _op_sub

from ...core.analysis_object import AnalysisObject
from ..array import Array
from .binops import run_elementwise_binary


def sub(
    left: "Array | AnalysisObject | object",
    right: "Array | AnalysisObject | object",
) -> Array:
    """Compute strict elementwise subtraction.

    Parameters
    ----------
    left, right
        AO-like operands with compatible topology/core semantics.

    Returns
    -------
    Array
        Subtracted output with resolved topology.

    Notes
    -----
    Alignment is name/label-aware. Default result variable naming uses the
    centralized ``datavar`` policy.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.linalg import Array, sub
    >>> left = Array(AnalysisObject.from_data(
    ...     xr.Dataset({"x": ("sample", [5.0, 7.0])}, coords={"sample": [0, 1]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... ))
    >>> right = Array(AnalysisObject.from_data(
    ...     xr.Dataset({"x": ("sample", [2.0, 3.0])}, coords={"sample": [0, 1]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... ))
    >>> out = sub(left, right)
    >>> out.as_dataset()["datavar"].to_numpy().tolist()
    [3.0, 4.0]
    """
    return run_elementwise_binary(
        left,
        right,
        owner="linalg.sub",
        op_name="sub",
        operator_fn=_op_sub,
    )


__all__ = [
    "sub",
]
