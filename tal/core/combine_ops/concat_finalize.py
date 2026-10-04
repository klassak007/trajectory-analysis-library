from __future__ import annotations

"""Concat-sequence postprocess/finalize ownership."""

from typing import TYPE_CHECKING

import xarray as xr

from .concat_pack import apply_concat_overlap_sort, apply_concat_valid_mask
from .concat_plan import ConcatSequencePlan
from .concat_topology import restore_batch as _restore_batch
from .finalize import CombineFinalizationPlan, finalize_combine_output
from .metadata import canonicalize_optional_names, resolve_core_dims
from .types import CombineContext, SequenceConcatOptions

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


def _assign_total_size(
    out: xr.Dataset,
    *,
    size_name: str | None,
    plan: ConcatSequencePlan,
) -> xr.Dataset:
    if size_name is None:
        return out
    total = xr.DataArray(
        plan.lengths.sum(axis=0),
        dims=(plan.flat_dim,),
        coords={plan.flat_dim: plan.labels},
    )
    return out.assign_coords({size_name: total})


def postprocess_concat_dataset(
    ds: xr.Dataset,
    *,
    valid: xr.DataArray,
    opts: SequenceConcatOptions,
    plan: ConcatSequencePlan,
) -> tuple[xr.Dataset, str | None, str | None]:
    """Apply concat postprocess stages and return canonical optional names.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    valid : xr.DataArray, optional
        Validity/mask payload used by this operation.
    opts : SequenceConcatOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    plan : ConcatSequencePlan, optional
        Resolved runtime context/payload used by this orchestration boundary.

    Returns
    -------
    tuple[xr.Dataset, str | None, str | None]
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    out = apply_concat_valid_mask(ds, valid=valid, plan=plan)
    out = apply_concat_overlap_sort(out, opts=opts, plan=plan)
    out = _assign_total_size(out, size_name=plan.size_name, plan=plan)
    out = _restore_batch(out, batch_dims=plan.batch_dims, flat_dim=plan.flat_dim)
    return canonicalize_optional_names(
        out,
        param_name=plan.param_name,
        size_name=plan.size_name,
        sequence_dim=plan.sequence_dim,
        batch_dims=plan.batch_dims,
    )


def finalize_concat_sequence_contexts(
    contexts: list[CombineContext],
    *,
    ds: xr.Dataset,
    plan: ConcatSequencePlan,
    param_name: str | None,
    size_name: str | None,
    validate: bool,
    finalization: CombineFinalizationPlan,
) -> "AnalysisObject":
    """Finalize concat-sequence output using shared combine finalize boundary.

    Parameters
    ----------
    contexts : list[CombineContext]
        Resolved runtime context/payload used by this orchestration boundary.
    ds : xr.Dataset, optional
        Input dataset/source value processed by this operation.
    plan : ConcatSequencePlan, optional
        Resolved runtime context/payload used by this orchestration boundary.
    param_name : str | None, optional
        Parameter-domain input used for temporal evaluation/alignment.
    size_name : str | None, optional
        Output naming metadata used during finalization.
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
    core_dims = resolve_core_dims(
        contexts,
        ds=ds,
        sequence_dim=plan.sequence_dim,
        batch_dims=plan.batch_dims,
        owner="concat_sequence",
    )
    return finalize_combine_output(
        finalization,
        ds,
        sequence_dim=plan.sequence_dim,
        batch_dims=plan.batch_dims,
        core_dims=core_dims,
        param_coord=param_name,
        sequence_size_coord=size_name,
        validate=validate,
    )


__all__ = [
    "finalize_concat_sequence_contexts",
    "postprocess_concat_dataset",
]
