from __future__ import annotations

"""Concat-sequence overlap preflight ownership."""

import numpy as np
import xarray as xr

from ..orchestration.lazy import require_unchunked_dataarray
from .concat_plan import ConcatSequencePlan
from .types import SequenceConcatOptions


def _check_monotonic(param: np.ndarray, *, owner: str) -> None:
    if np.any(~np.isfinite(param)):
        raise ValueError(f"{owner}: overlap='error' requires finite param_coord values on the valid domain.")
    if param.size >= 2 and np.any(np.diff(param) < 0):
        raise ValueError(f"{owner}: param_coord must be monotonic non-decreasing within each segment.")


def _batched_param_values(
    ds: xr.Dataset,
    *,
    name: str,
    flat_dim: str,
    sequence_dim: str,
    overlap: str,
    owner: str,
) -> np.ndarray:
    coord = ds.coords[name]
    require_unchunked_dataarray(
        coord,
        owner=owner,
        field="overlap checks do not support chunked param_coord values",
        guidance=f"compute or rechunk coordinate {name!r} before overlap={overlap!r}.",
    )
    if flat_dim in coord.dims:
        return np.asarray(coord.transpose(flat_dim, sequence_dim).data, dtype="float64")
    row = np.asarray(coord.transpose(sequence_dim).data, dtype="float64")
    return np.broadcast_to(row[None, :], (int(ds.sizes[flat_dim]), row.shape[0]))


def validate_grouped_sort_payload(
    ds: xr.Dataset,
    *,
    flat_dim: str,
    sequence_dim: str,
    owner: str,
) -> None:
    for name, var in ds.data_vars.items():
        if sequence_dim in var.dims and flat_dim not in var.dims:
            raise ValueError(
                f"{owner}: overlap='sort' requires batch-aware sequence payload for grouped inputs; "
                f"data variable {name!r} is sample-only."
            )
    for name, coord in ds.coords.items():
        if name in {sequence_dim, "valid"}:
            continue
        if sequence_dim in coord.dims and flat_dim not in coord.dims:
            raise ValueError(
                f"{owner}: overlap='sort' requires batch-aware sequence payload for grouped inputs; "
                f"coordinate {name!r} is sample-only."
            )


def _validate_grouped_sort_inputs(
    datasets: list[xr.Dataset],
    *,
    flat_dim: str,
    sequence_dim: str,
    owner: str,
) -> None:
    for ds in datasets:
        validate_grouped_sort_payload(
            ds,
            flat_dim=flat_dim,
            sequence_dim=sequence_dim,
            owner=owner,
        )


def _overlap_checks(
    plan: ConcatSequencePlan,
    *,
    overlap: str,
    owner: str,
) -> None:
    if plan.param_name is None:
        return
    rows = int(plan.lengths.shape[1]) if plan.lengths.ndim == 2 else 0
    has_prev = np.zeros(rows, dtype=bool)
    prev_last = np.zeros(rows, dtype="float64")
    for seg, ds in enumerate(plan.aligned):
        if plan.param_name not in ds.coords:
            raise ValueError(f"{owner}: overlap checks require param_coord on all segments.")
        arr = _batched_param_values(
            ds,
            name=plan.param_name,
            flat_dim=plan.flat_dim,
            sequence_dim=plan.sequence_dim,
            overlap=overlap,
            owner=owner,
        )
        for row in range(rows):
            n = int(plan.lengths[seg, row])
            if n <= 0:
                continue
            row_vals = arr[row, :n]
            _check_monotonic(row_vals, owner=owner)
            if overlap == "error" and has_prev[row] and prev_last[row] > row_vals[0]:
                raise ValueError(f"{owner}: append would break monotonic order; use overlap='sort'.")
            prev_last[row] = float(row_vals[n - 1])
            has_prev[row] = True


def validate_concat_overlap(
    plan: ConcatSequencePlan,
    *,
    opts: SequenceConcatOptions,
    owner: str,
) -> None:
    """Validate overlap preconditions for concat-sequence before packing.

    Parameters
    ----------
    plan : ConcatSequencePlan
        Resolved runtime context/payload used by this orchestration boundary.
    opts : SequenceConcatOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    None
        Returns ``None``; side effects are applied through owned state/metadata updates.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if opts.overlap == "sort" and plan.batch_dims:
        _validate_grouped_sort_inputs(
            plan.aligned,
            flat_dim=plan.flat_dim,
            sequence_dim=plan.sequence_dim,
            owner=owner,
        )
    _overlap_checks(plan, overlap=opts.overlap, owner=owner)


__all__ = [
    "validate_concat_overlap",
    "validate_grouped_sort_payload",
]
