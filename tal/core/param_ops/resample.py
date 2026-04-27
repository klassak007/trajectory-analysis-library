from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .evaluate import evaluate_param
from .types import ParamEvalOptions, ParamRuntimeContext


def resample_param(
    context: ParamRuntimeContext,
    *,
    grid: xr.DataArray | np.ndarray | Sequence[float] | float,
    opts: ParamEvalOptions,
    validate: bool,
) -> "AnalysisObject":
    """Resample AO data to a parameter grid.

    Parameters
    ----------
    context : ParamRuntimeContext
        Resolved runtime context/payload used by this orchestration boundary.
    grid : xr.DataArray | np.ndarray | Sequence[float] | float, optional
        Parameter-domain input used for temporal evaluation/alignment.
    opts : ParamEvalOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return evaluate_param(context, query=grid, opts=opts, validate=validate)


__all__ = ["resample_param"]
