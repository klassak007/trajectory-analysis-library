from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..analysis_object import AnalysisObject
from ..schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
    validate_schema_if_needed,
)
from .topology import SemanticTopology


@dataclass(frozen=True)
class DatasetContextOptions:
    require_roles: bool = False
    require_sequence_dim: bool = False
    select_numeric_var: bool = False
    require_single_numeric_var: bool = False
    allowed_core_arity: tuple[int, ...] | None = None
    require_semantic_dims_in_var: bool = False


@dataclass(frozen=True)
class DatasetContext:
    ao: AnalysisObject
    ds: xr.Dataset
    roles_declared: bool
    sequence_dim: str | None
    batch_dims: tuple[str, ...]
    core_dims: tuple[str, ...]
    param_coord: str | None
    sequence_size_coord: str | None
    var_name: str | None
    data: xr.DataArray | None


def _source_label(index: int | None) -> str:
    if index is None:
        return "input"
    return f"operand {index}"


def _single_numeric_var(
    ds: xr.Dataset,
    *,
    owner: str,
    index: int | None,
    require_single: bool,
) -> tuple[str, xr.DataArray]:
    source = _source_label(index)
    names = list(ds.data_vars)
    if not names:
        raise ValueError(f"{owner}: {source} must contain at least one data variable.")
    if require_single and len(names) != 1:
        raise ValueError(f"{owner}: {source} must contain exactly one data variable; got {len(names)}.")
    name = str(names[0])
    arr = ds[name]
    if not np.issubdtype(np.dtype(arr.dtype), np.number):
        raise TypeError(f"{owner}: {source} variable {name!r} must be numeric; got {arr.dtype!r}.")
    return name, arr


def _require_core_arity(
    core_dims: tuple[str, ...],
    *,
    owner: str,
    index: int | None,
    allowed: tuple[int, ...],
) -> None:
    if len(core_dims) in allowed:
        return
    source = _source_label(index)
    raise ValueError(
        f"{owner}: {source} core_dims cardinality must be in {allowed!r}; got {core_dims!r}."
    )


def _require_semantic_dims_present(
    data: xr.DataArray,
    *,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, ...],
    owner: str,
    index: int | None,
    var_name: str,
) -> None:
    if sequence_dim is None:
        return
    dims = (sequence_dim,) + batch_dims + core_dims
    missing = tuple(dim for dim in dims if dim not in data.dims)
    if not missing:
        return
    source = _source_label(index)
    raise ValueError(
        f"{owner}: {source} variable {var_name!r} is missing declared semantic dims {missing!r}."
    )


def _resolve_context_roles(
    ds: xr.Dataset,
    *,
    opts: DatasetContextOptions,
    owner: str,
    index: int | None,
) -> tuple[bool, str | None, tuple[str, ...], tuple[str, ...]]:
    roles_declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    source = _source_label(index)
    if opts.require_roles and not roles_declared:
        raise ValueError(f"{owner}: {source} requires declared roles.")
    if opts.require_sequence_dim and sequence_dim is None:
        raise ValueError(f"{owner}: {source} requires declared roles with sequence_dim.")
    if roles_declared and opts.allowed_core_arity:
        _require_core_arity(core_dims, owner=owner, index=index, allowed=opts.allowed_core_arity)
    if roles_declared:
        return roles_declared, sequence_dim, batch_dims, core_dims
    return roles_declared, None, (), ()


def _resolve_selected_numeric_var(
    ds: xr.Dataset,
    *,
    opts: DatasetContextOptions,
    roles_declared: bool,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, ...],
    owner: str,
    index: int | None,
) -> tuple[str | None, xr.DataArray | None, tuple[str, ...]]:
    if not opts.select_numeric_var:
        return None, None, core_dims
    var_name, data = _single_numeric_var(
        ds,
        owner=owner,
        index=index,
        require_single=opts.require_single_numeric_var,
    )
    resolved_core_dims = core_dims
    if not roles_declared:
        resolved_core_dims = tuple(str(dim) for dim in data.dims)
        if opts.allowed_core_arity:
            _require_core_arity(
                resolved_core_dims,
                owner=owner,
                index=index,
                allowed=opts.allowed_core_arity,
            )
    if opts.require_semantic_dims_in_var:
        _require_semantic_dims_present(
            data,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            core_dims=resolved_core_dims,
            owner=owner,
            index=index,
            var_name=var_name,
        )
    return var_name, data, resolved_core_dims


def resolve_dataset_context(
    ao: AnalysisObject,
    *,
    owner: str,
    options: DatasetContextOptions | None = None,
    index: int | None = None,
) -> DatasetContext:
    opts = options if options is not None else DatasetContextOptions()
    ds = validate_schema_if_needed(ao.unsafe_data)
    roles_declared, sequence_dim, batch_dims, core_dims = _resolve_context_roles(
        ds,
        opts=opts,
        owner=owner,
        index=index,
    )
    var_name, data, resolved_core_dims = _resolve_selected_numeric_var(
        ds,
        opts=opts,
        roles_declared=roles_declared,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
        owner=owner,
        index=index,
    )
    return DatasetContext(
        ao=ao,
        ds=ds,
        roles_declared=roles_declared,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=resolved_core_dims,
        param_coord=read_param_coord_name(ds),
        sequence_size_coord=read_sequence_size_coord_name(ds),
        var_name=var_name,
        data=data,
    )


def resolve_dataset_contexts(
    aos: Sequence[AnalysisObject],
    *,
    owner: str,
    options: DatasetContextOptions | None = None,
) -> list[DatasetContext]:
    return [
        resolve_dataset_context(ao, owner=owner, options=options, index=index)
        for index, ao in enumerate(aos)
    ]


def resolve_semantic_topology_from_dataset(
    ds: xr.Dataset,
    *,
    var_name: str,
    core_dims: tuple[str, ...],
    owner: str,
    what: str,
    allow_missing_sequence_dim: bool = False,
    allow_missing_batch_dims: bool = False,
) -> SemanticTopology:
    candidate = validate_schema_if_needed(ds)
    if var_name not in candidate.data_vars:
        raise ValueError(f"{owner}: {what} variable {var_name!r} was not found in dataset.")
    declared, sequence_dim, batch_dims, _ = read_roles(candidate)
    if not declared:
        raise ValueError(f"{owner}: {what} requires declared roles.")
    dims = candidate[var_name].dims
    missing: list[str] = []
    if sequence_dim is not None and sequence_dim not in dims and not allow_missing_sequence_dim:
        missing.append(sequence_dim)
    if not allow_missing_batch_dims:
        missing.extend(dim for dim in batch_dims if dim not in dims)
    missing.extend(dim for dim in core_dims if dim not in dims)
    if missing:
        raise ValueError(
            f"{owner}: {what} var {var_name!r} is missing required semantic dims {tuple(missing)!r}; "
            f"present dims={tuple(dims)!r}."
        )
    return SemanticTopology(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=tuple(core_dims),
    )


__all__ = [
    "DatasetContext",
    "DatasetContextOptions",
    "resolve_semantic_topology_from_dataset",
    "resolve_dataset_context",
    "resolve_dataset_contexts",
]
