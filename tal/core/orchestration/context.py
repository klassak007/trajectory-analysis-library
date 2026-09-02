from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..analysis_object import AnalysisObject
from ..dataset_ownership import analysis_object_dataset
from ..schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
    validate_schema_if_needed,
)
from .topology import SemanticTopology


@dataclass(frozen=True)
class DatasetContextOptions:
    """Options for resolving operation-time dataset context.

    Parameters
    ----------
    require_roles : bool, optional
        Require TAL role metadata to be declared.
    require_sequence_dim : bool, optional
        Require declared roles to include ``sequence_dim``.
    select_numeric_var : bool, optional
        Select a numeric data variable and expose it on the resolved context.
    require_single_numeric_var : bool, optional
        Require the dataset to contain exactly one data variable when selecting
        numeric data.
    allowed_core_arity : tuple[int, ...] | None, optional
        Allowed number of core dimensions after role or payload resolution.
    require_semantic_dims_in_var : bool, optional
        Require the selected variable to contain all declared semantic
        dimensions.

    Notes
    -----
    These options describe orchestration-time requirements only. They do not
    mutate datasets and they do not perform numeric computation on payload
    values.

    Examples
    --------
    >>> from tal.core.orchestration.context import DatasetContextOptions
    >>> opts = DatasetContextOptions(
    ...     require_roles=True,
    ...     select_numeric_var=True,
    ...     allowed_core_arity=(0,),
    ... )
    >>> (opts.require_roles, opts.select_numeric_var, opts.allowed_core_arity)
    (True, True, (0,))
    """

    require_roles: bool = False
    require_sequence_dim: bool = False
    select_numeric_var: bool = False
    require_single_numeric_var: bool = False
    allowed_core_arity: tuple[int, ...] | None = None
    require_semantic_dims_in_var: bool = False


@dataclass(frozen=True)
class DatasetContext:
    """Resolved dataset metadata for an operation boundary.

    Parameters
    ----------
    ao : AnalysisObject
        Source object used for later finalization.
    ds : xr.Dataset
        Schema-checked backing dataset used by the operation.
    roles_declared : bool
        Whether ``tal.core.roles`` was present.
    sequence_dim : str | None
        Declared sequence dimension, when present.
    batch_dims : tuple[str, ...]
        Declared batch dimensions.
    core_dims : tuple[str, ...]
        Declared or inferred core dimensions.
    param_coord : str | None
        Declared parameter coordinate name, when present.
    sequence_size_coord : str | None
        Declared sequence-size validity coordinate name, when present.
    var_name : str | None
        Selected data variable name, when variable selection was requested.
    data : xr.DataArray | None
        Selected data variable, when variable selection was requested.

    Notes
    -----
    The context is immutable and contains only metadata plus references to
    xarray objects. It preserves xarray laziness by avoiding payload value
    materialization.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.context import (
    ...     DatasetContext,
    ...     DatasetContextOptions,
    ...     resolve_dataset_context,
    ... )
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ctx = resolve_dataset_context(
    ...     ao,
    ...     owner="thermal.bias_temperature",
    ...     options=DatasetContextOptions(select_numeric_var=True),
    ... )
    >>> (isinstance(ctx, DatasetContext), ctx.sequence_dim, ctx.var_name)
    (True, 'sample', 'celsius')
    """

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
    """Resolve roles, optional metadata, and selected data for one AO.

    Parameters
    ----------
    ao : AnalysisObject
        Source object to inspect.
    owner : str
        Public owner string used to build deterministic diagnostics.
    options : DatasetContextOptions | None, optional
        Resolution requirements. ``None`` uses default permissive options.
    index : int | None, optional
        Operand index for diagnostics in multi-input operations.

    Returns
    -------
    DatasetContext
        Immutable context containing the schema-checked dataset, role metadata,
        optional coordinate metadata, and optional selected numeric variable.

    Raises
    ------
    TypeError
        If a selected variable is not numeric.
    ValueError
        If required roles, sequence metadata, variable cardinality, core arity,
        or semantic dimensions are missing.

    Notes
    -----
    Resolution is xarray-native and metadata-oriented. It validates schema when
    needed and selects arrays without reading payload values, so Dask-backed
    data remains lazy.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.context import DatasetContextOptions, resolve_dataset_context
    >>> ao = AnalysisObject.from_data(
    ...     xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> ctx = resolve_dataset_context(
    ...     ao,
    ...     owner="thermal.bias_temperature",
    ...     options=DatasetContextOptions(
    ...         require_roles=True,
    ...         select_numeric_var=True,
    ...         require_single_numeric_var=True,
    ...         allowed_core_arity=(0,),
    ...         require_semantic_dims_in_var=True,
    ...     ),
    ... )
    >>> (ctx.roles_declared, ctx.sequence_dim, ctx.core_dims, ctx.var_name, ctx.data.dims)
    (True, 'sample', (), 'celsius', ('sample',))
    """
    opts = options if options is not None else DatasetContextOptions()
    ds = validate_schema_if_needed(analysis_object_dataset(ao))
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
    """Resolve dataset context for a sequence of AO inputs.

    Parameters
    ----------
    aos : Sequence[AnalysisObject]
        Ordered AO inputs to inspect.
    owner : str
        Public owner string used to build deterministic diagnostics.
    options : DatasetContextOptions | None, optional
        Resolution requirements shared by every input.

    Returns
    -------
    list[DatasetContext]
        Contexts in the same order as ``aos``.

    Raises
    ------
    TypeError
        If selected variables violate numeric requirements.
    ValueError
        If any input violates the requested context options.

    Notes
    -----
    Diagnostics include the operand index so variadic operation failures point
    to the source input.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.orchestration.context import DatasetContextOptions, resolve_dataset_contexts
    >>> ds = xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]})
    >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), validate=True)
    >>> contexts = resolve_dataset_contexts(
    ...     [ao, ao],
    ...     owner="thermal.combine_temperatures",
    ...     options=DatasetContextOptions(select_numeric_var=True),
    ... )
    >>> [ctx.var_name for ctx in contexts]
    ['celsius', 'celsius']
    """
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
