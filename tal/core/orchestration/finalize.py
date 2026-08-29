from __future__ import annotations

"""Shared orchestration finalization boundary wrappers."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import xarray as xr

from ..ao_internal import finalize_structural, from_unvalidated_like
from ..schema import _transfer_dataset_attrs_for_finalize
from .topology import restore_dataset_batch_topology

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject
    from ..param_ops.batch_topology import BatchFlattenPlan


def transfer_dataset_attrs(
    source: xr.Dataset,
    target: xr.Dataset,
    *,
    validate: bool = False,
) -> xr.Dataset:
    """Transfer source attrs at an internal dataset-finalization boundary.

    Ordinary attrs replace the target mapping using xarray's shallow value
    semantics. The TAL payload is independently copied by the canonical schema
    implementation, and optional validation occurs only after the candidate is
    complete. Neither input dataset is mutated.
    """
    return _transfer_dataset_attrs_for_finalize(
        source,
        target,
        validate=validate,
    )


def finalize_like(
    source_ao: "AnalysisObject",
    ds: xr.Dataset,
    *,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    """Finalize a dataset with the structural policy of a source AO.

    Parameters
    ----------
    source_ao : AnalysisObject
        Source object whose schema and subclass rewrap behavior define the
        finalization boundary.
    ds : xr.Dataset
        Candidate output dataset produced by an operation kernel.
    validate : bool
        When ``True``, validate output schema/layout invariants before returning.
    owner : str
        Public owner string used to build deterministic diagnostics.

    Returns
    -------
    AnalysisObject
        Output rewrapped like ``source_ao`` with schema repaired after structural
        changes.

    Notes
    -----
    Finalization repairs role references, validity metadata, and component
    registry entries affected by structural changes. It preserves xarray
    payload laziness and delegates subclass rewrap behavior to the source AO.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.finalize import finalize_like
    >>> source = AnalysisObject.from_data(
    ...     xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> out_ds = source.unsafe_data.assign(celsius=source.unsafe_data["celsius"] + 1.0)
    >>> out = finalize_like(source, out_ds, validate=True, owner="thermal.bias_temperature")
    >>> out.unsafe_data.attrs["tal"]["core"]["roles"]["sequence_dim"]
    'sample'
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
    "transfer_dataset_attrs",
]
