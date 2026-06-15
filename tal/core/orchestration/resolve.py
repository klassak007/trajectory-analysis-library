from __future__ import annotations

"""Shared schema/role context assembly for combine and param orchestration."""

from collections.abc import Sequence

import numpy as np
import xarray as xr

from ..combine_ops.types import CombineContext
from .context import DatasetContextOptions, resolve_dataset_contexts
from ..param_engine.schema_resolve import _resolve_schema_context_validated
from ..param_engine.types import ParamCoordSpec
from ..param_engine.validity_mask import _resolve_param_valid_mask_validated
from ..param_ops.axis_coords import batch_coord
from ..param_ops.types import ParamKind, ParamRuntimeContext


def _resolve_param_kind(coord: xr.DataArray, *, name: str) -> ParamKind:
    dtype = np.dtype(coord.dtype)
    if np.issubdtype(dtype, np.number):
        return "numeric"
    if np.issubdtype(dtype, np.datetime64):
        return "datetime64"
    raise ValueError(
        "param operations require numeric or datetime64 param_coord values; "
        f"coord {name!r} has dtype {coord.dtype!r}. "
        "Object datetime coordinates must be converted to xarray-visible datetime64 before calling ao.param.*."
    )


def resolve_param_runtime_context(
    ao: "AnalysisObject",
    *,
    on: str | None = None,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
    sequence_size_coord: str | None = None,
) -> ParamRuntimeContext:
    """Resolve schema, param coordinate, and validity context for param ops.

    Parameters
    ----------
    ao : AnalysisObject
        AnalysisObject-like input value.
    on : str | None, optional
        Coordinate/dimension name used as the operation domain.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : Sequence[str] | None, optional
        Optional override for batch dimensions used by temporal semantics.
    sequence_size_coord : str | None, optional
        Optional sequence-size coordinate used for ragged validity handling.

    Returns
    -------
    ParamRuntimeContext
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    schema_ctx = _resolve_schema_context_validated(
        ao.unsafe_data,
        explicit_sequence_dim=sequence_dim,
        explicit_batch_dims=batch_dims,
        explicit_param_name=on,
        explicit_sequence_size_coord=sequence_size_coord,
    )
    if schema_ctx.param_name is None:
        raise ValueError("param operations require a resolved param_coord; set it in schema or pass explicit on=...")
    spec = ParamCoordSpec(
        name=schema_ctx.param_name,
        coord=schema_ctx.ds.coords[schema_ctx.param_name],
        sequence_dim=schema_ctx.sequence_dim,
        batch_dims=schema_ctx.batch_dims,
    )
    param_kind = _resolve_param_kind(spec.coord, name=spec.name)
    valid = _resolve_param_valid_mask_validated(
        schema_ctx.ds,
        spec=spec,
        sequence_size_coord=sequence_size_coord,
    )
    coords = {dim: batch_coord(schema_ctx.ds, dim=dim) for dim in schema_ctx.batch_dims}
    return ParamRuntimeContext(
        ao=ao,
        ds=schema_ctx.ds,
        spec=spec,
        sequence_dim=schema_ctx.sequence_dim,
        batch_dims=schema_ctx.batch_dims,
        core_dims=schema_ctx.core_dims,
        valid_mask=valid,
        sequence_size_coord=schema_ctx.sequence_size_coord,
        batch_coords=coords,
        param_kind=param_kind,
    )


def _batch_dims_from_contexts(contexts: Sequence[CombineContext]) -> tuple[str, ...]:
    nonempty = [ctx.batch_dims for ctx in contexts if ctx.batch_dims]
    if not nonempty:
        return ()
    base = nonempty[0]
    for dims in nonempty[1:]:
        if dims != base:
            raise ValueError(f"combine operations: inputs have different batch_dims {base!r} vs {dims!r}.")
    return base


def _sequence_dim_from_contexts(
    contexts: Sequence[CombineContext],
    *,
    require: bool,
    owner: str,
) -> str | None:
    dims = [ctx.sequence_dim for ctx in contexts if ctx.sequence_dim is not None]
    if not dims:
        if require:
            raise ValueError(f"{owner}: sequence_dim is required for this operation.")
        return None
    base = dims[0]
    for dim in dims[1:]:
        if dim != base:
            raise ValueError(f"{owner}: inputs have different sequence_dim values {base!r} vs {dim!r}.")
    return base


def resolve_combine_contexts(
    aos: Sequence["AnalysisObject"],
    *,
    require_sequence: bool,
    owner: str,
) -> list[CombineContext]:
    """Resolve schema-backed combine contexts from normalized AO inputs.

    Parameters
    ----------
    aos : Sequence['AnalysisObject']
        AO-like inputs consumed by this orchestration boundary.
    require_sequence : bool, optional
        Behavior flag/policy controlling boundary semantics.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    list[CombineContext]
        Ordered collection produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    dataset_contexts = resolve_dataset_contexts(
        aos,
        owner=owner,
        options=DatasetContextOptions(),
    )
    contexts = [
        CombineContext(
            ao=ctx.ao,
            ds=ctx.ds,
            roles_declared=ctx.roles_declared,
            sequence_dim=ctx.sequence_dim,
            batch_dims=ctx.batch_dims,
            core_dims=ctx.core_dims,
            param_coord=ctx.param_coord,
            sequence_size_coord=ctx.sequence_size_coord,
        )
        for ctx in dataset_contexts
    ]
    _ = _sequence_dim_from_contexts(contexts, require=require_sequence, owner=owner)
    _ = _batch_dims_from_contexts(contexts)
    return contexts


def effective_sequence_dim(contexts: Sequence[CombineContext], *, owner: str, require: bool) -> str | None:
    return _sequence_dim_from_contexts(contexts, require=require, owner=owner)


def effective_batch_dims(contexts: Sequence[CombineContext]) -> tuple[str, ...]:
    return _batch_dims_from_contexts(contexts)


__all__ = [
    "effective_batch_dims",
    "effective_sequence_dim",
    "resolve_combine_contexts",
    "resolve_param_runtime_context",
]
