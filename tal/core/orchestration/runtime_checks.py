from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from ..schema_read import read_roles

_NUMERIC_KINDS = {"i", "u", "f", "c"}


def select_single_numeric_var(ds: xr.Dataset, *, owner: str, what: str) -> str:
    if len(ds.data_vars) != 1:
        raise ValueError(f"{owner}: {what} requires exactly one data variable.")
    var_name = str(next(iter(ds.data_vars)))
    if ds[var_name].dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{owner}: {what} data variable {var_name!r} must be numeric.")
    return var_name


def require_declared_roles_with_sequence(
    ds: xr.Dataset,
    *,
    owner: str,
    what: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    if not declared or sequence_dim is None:
        raise ValueError(f"{owner}: {what} requires declared roles with sequence_dim.")
    return sequence_dim, batch_dims, core_dims


def require_declared_roles(
    ds: xr.Dataset,
    *,
    owner: str,
    what: str,
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: {what} requires declared roles.")
    return sequence_dim, batch_dims, core_dims


def require_var_contains_dims(
    ds: xr.Dataset,
    *,
    var_name: str,
    required_dims: Sequence[str],
    owner: str,
    what: str,
) -> None:
    dims = ds[var_name].dims
    missing = [dim for dim in required_dims if dim not in dims]
    if not missing:
        return
    raise ValueError(
        f"{owner}: {what} var {var_name!r} is missing required dims {missing!r}; "
        f"present dims={list(dims)!r}."
    )


def require_explicit_unique_dim_labels(ds: xr.Dataset, *, dim: str, owner: str, what: str) -> tuple[object, ...]:
    if dim not in ds.dims:
        raise ValueError(f"{owner}: {what} core dim {dim!r} must be a dataset dimension.")
    if dim not in ds.coords:
        raise ValueError(f"{owner}: {what} core dim {dim!r} must declare explicit coordinate labels.")
    labels = tuple(ds.get_index(dim).tolist())
    if len(labels) != len(set(labels)):
        raise ValueError(f"{owner}: {what} core dim {dim!r} labels must be unique.")
    return labels


def require_exact_labels(
    labels: tuple[object, ...],
    *,
    expected: tuple[object, ...],
    owner: str,
    what: str,
) -> None:
    if tuple(labels) == tuple(expected):
        return
    raise ValueError(f"{owner}: {what} labels must equal {tuple(expected)!r}; got {tuple(labels)!r}.")


def require_single_core_dim_with_length(
    ds: xr.Dataset,
    *,
    expected_length: int,
    owner: str,
    what: str,
) -> str:
    _, _, core_dims = require_declared_roles(ds, owner=owner, what=what)
    if len(core_dims) != 1:
        raise ValueError(f"{owner}: {what} requires exactly one core dim; got {core_dims!r}.")
    core_dim = core_dims[0]
    if int(ds.sizes.get(core_dim, -1)) != expected_length:
        raise ValueError(f"{owner}: {what} core dim {core_dim!r} must have length {expected_length}.")
    return core_dim


def resolve_single_numeric_var_single_core_dim(ds: xr.Dataset, *, owner: str, what: str) -> tuple[str, str]:
    var_name = select_single_numeric_var(ds, owner=owner, what=what)
    _, _, core_dims = require_declared_roles(ds, owner=owner, what=what)
    if len(core_dims) != 1:
        raise ValueError(f"{owner}: {what} requires exactly one core dim; got {core_dims!r}.")
    core_dim = core_dims[0]
    require_var_contains_dims(
        ds,
        var_name=var_name,
        required_dims=(core_dim,),
        owner=owner,
        what=what,
    )
    return var_name, core_dim


__all__ = [
    "require_declared_roles",
    "require_declared_roles_with_sequence",
    "require_exact_labels",
    "require_explicit_unique_dim_labels",
    "require_single_core_dim_with_length",
    "require_var_contains_dims",
    "resolve_single_numeric_var_single_core_dim",
    "select_single_numeric_var",
]
