from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._event_constants import EDGE_ENTER, EDGE_EXIT, EDGE_INVALID, EDGE_TRIGGER, SAMPLE_SENTINEL

if TYPE_CHECKING:
    from .resolve import EventEvalContext


def batch_dims(context: "EventEvalContext") -> tuple[str, ...]:
    """Return canonical batch dims for event extraction context.

    Parameters
    ----------
    context : EventEvalContext
        Resolved runtime context/payload used by this orchestration boundary.

    Returns
    -------
    tuple[str, ...]
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    return context.runtime.batch_dims


def lane_count(lane: xr.DataArray, *, batch_dims: tuple[str, ...]) -> int:
    """Return flattened lane count across batch dims.

    Parameters
    ----------
    lane : xr.DataArray
        Per-row event lane payload used for boundary extraction.
    batch_dims : tuple[str, ...], optional
        Optional override for batch dimensions used by temporal semantics.

    Returns
    -------
    int
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if not batch_dims:
        return 1
    return int(np.prod([lane.sizes[dim] for dim in batch_dims], dtype=np.int64))


def lane_data(
    da: xr.DataArray,
    *,
    context: "EventEvalContext",
    core_dim: str,
) -> np.ndarray:
    """Reshape batch-major data into ``(lanes, core_dim)`` for row kernels.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray value processed by this operation.
    context : EventEvalContext, optional
        Resolved runtime context/payload used by this orchestration boundary.
    core_dim : str, optional
        Dimension name used to resolve labeled array semantics.

    Returns
    -------
    np.ndarray
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    dims = batch_dims(context) + (core_dim,)
    lane = da.transpose(*dims)
    count = lane_count(lane, batch_dims=batch_dims(context))
    return np.asarray(lane.data).reshape(count, lane.sizes[core_dim])


__all__ = [
    "EDGE_ENTER",
    "EDGE_EXIT",
    "EDGE_INVALID",
    "EDGE_TRIGGER",
    "SAMPLE_SENTINEL",
    "batch_dims",
    "lane_count",
    "lane_data",
]
