from __future__ import annotations

"""Shared orchestration finalization boundary wrappers."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import xarray as xr

from ..ao_internal import finalize_structural, from_unvalidated_like
from .topology import restore_dataset_batch_topology

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject
    from ..param_ops.batch_topology import BatchFlattenPlan


def finalize_like(
    source_ao: "AnalysisObject",
    ds: xr.Dataset,
    *,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    """Finalize a dataset using the source AO structural boundary.

    Parameters
    ----------
    source_ao : AnalysisObject
        Input dataset/source value processed by this operation.
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    _ = owner
    return finalize_structural(source_ao, ds, validate=validate)


def rewrap_unvalidated_like(
    source_ao: "AnalysisObject",
    ds: xr.Dataset,
    *,
    owner: str,
) -> "AnalysisObject":
    """Rewrap a dataset without validation using the source AO class.

    Parameters
    ----------
    source_ao : AnalysisObject
        Input dataset/source value processed by this operation.
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    _ = owner
    return from_unvalidated_like(source_ao, ds)


def restore_and_finalize(
    source_ao: "AnalysisObject",
    ds: xr.Dataset,
    *,
    plan: "BatchFlattenPlan",
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    """Restore flattened batch topology then finalize in one boundary call.

    Parameters
    ----------
    source_ao : AnalysisObject
        Input dataset/source value processed by this operation.
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    plan : BatchFlattenPlan, optional
        Resolved runtime context/payload used by this orchestration boundary.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    restored = restore_dataset_batch_topology(ds, plan=plan, owner=owner)
    return finalize_like(source_ao, restored, validate=validate, owner=owner)


def finalize_many_like(
    sources: Sequence["AnalysisObject"],
    datasets: Sequence[xr.Dataset],
    *,
    validate: bool,
    owner: str,
) -> list["AnalysisObject"]:
    """Finalize many datasets with deterministic source/output pairing.

    Parameters
    ----------
    sources : Sequence['AnalysisObject']
        Input dataset/source collection processed by this operation.
    datasets : Sequence[xr.Dataset]
        Input dataset/source collection processed by this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    list['AnalysisObject']
        Ordered collection produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if len(sources) != len(datasets):
        raise ValueError(
            f"{owner}: expected equal source and dataset counts; "
            f"got {len(sources)} and {len(datasets)}."
        )
    return [
        finalize_like(source, ds, validate=validate, owner=owner)
        for source, ds in zip(sources, datasets, strict=True)
    ]


__all__ = [
    "finalize_like",
    "finalize_many_like",
    "restore_and_finalize",
    "rewrap_unvalidated_like",
]
