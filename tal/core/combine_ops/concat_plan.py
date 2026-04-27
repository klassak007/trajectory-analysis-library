from __future__ import annotations

"""Concat-sequence planning ownership."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .concat_topology import aligned_sequence_inputs, flat_dim_name
from .metadata import shared_optional_name
from .normalize import effective_batch_dims, effective_sequence_dim
from .types import CombineContext, SequenceConcatOptions


@dataclass(frozen=True)
class ConcatSequencePlan:
    """Normalized plan for concat-sequence execution.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    sequence_dim: str
    batch_dims: tuple[str, ...]
    flat_dim: str
    aligned: list[xr.Dataset]
    labels: pd.Index
    lengths: np.ndarray
    param_name: str | None
    size_name: str | None


def _shared_optional_names(contexts: list[CombineContext]) -> tuple[str | None, str | None]:
    return (
        shared_optional_name([ctx.param_coord for ctx in contexts]),
        shared_optional_name([ctx.sequence_size_coord for ctx in contexts]),
    )


def build_concat_sequence_plan(
    contexts: list[CombineContext],
    *,
    opts: SequenceConcatOptions,
    owner: str,
) -> ConcatSequencePlan:
    """Build the normalized concat-sequence plan without overlap or packing.

    Parameters
    ----------
    contexts : list[CombineContext]
        Resolved runtime context/payload used by this orchestration boundary.
    opts : SequenceConcatOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    ConcatSequencePlan
        Ordered collection produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    sequence_dim = effective_sequence_dim(contexts, owner=owner, require=True)
    assert sequence_dim is not None
    batch_dims = effective_batch_dims(contexts)
    flat_dim = flat_dim_name(contexts)
    aligned, labels, lengths = aligned_sequence_inputs(
        contexts,
        batch_dims=batch_dims,
        flat_dim=flat_dim,
        sequence_dim=sequence_dim,
        opts=opts,
    )
    param_name, size_name = _shared_optional_names(contexts)
    return ConcatSequencePlan(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        flat_dim=flat_dim,
        aligned=aligned,
        labels=labels,
        lengths=lengths,
        param_name=param_name,
        size_name=size_name,
    )


__all__ = ["ConcatSequencePlan", "build_concat_sequence_plan"]
