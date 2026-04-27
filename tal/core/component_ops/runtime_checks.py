from __future__ import annotations

import xarray as xr

from ..orchestration.runtime_checks import require_declared_roles_with_sequence as _require_roles_with_sequence
from .types import ComponentSpec

_NUMERIC_KINDS = {"i", "u", "f", "c"}


def require_declared_roles_with_sequence(
    ds: xr.Dataset,
    *,
    owner: str,
    operand: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    try:
        return _require_roles_with_sequence(ds, owner=owner, what=operand)
    except ValueError as exc:
        raise ValueError(f"{owner}: declared roles with sequence_dim are required for {operand}.") from exc


def select_component_var(
    ds: xr.Dataset,
    *,
    spec: ComponentSpec,
    component_name: str,
    owner: str,
    operand: str | None = None,
) -> str:
    suffix = f" on {operand}" if operand is not None else ""
    prefix = f"{owner}: component {component_name!r}{suffix}"
    if spec.var is not None:
        if spec.var not in ds.data_vars:
            raise ValueError(f"{prefix} expects var {spec.var!r}, but it is not present.")
        var_name = spec.var
    else:
        if len(ds.data_vars) != 1:
            raise ValueError(f"{prefix} requires exactly one data variable when spec.var is not set.")
        var_name = str(next(iter(ds.data_vars)))
    if ds[var_name].dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{prefix} selected var {var_name!r} must be numeric.")
    return var_name


def require_component_numeric_var(
    ds: xr.Dataset,
    *,
    component_name: str,
    var_name: str,
    required_dims: tuple[str, ...],
    owner: str,
    operand: str | None = None,
) -> None:
    suffix = f" on {operand}" if operand is not None else ""
    prefix = f"{owner}: component {component_name!r}{suffix}"
    if var_name not in ds.data_vars:
        raise ValueError(f"{prefix} expects var {var_name!r}, but it is not present.")
    if ds[var_name].dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{prefix} selected var {var_name!r} must be numeric.")
    missing = tuple(dim for dim in required_dims if dim not in ds[var_name].dims)
    if missing:
        raise ValueError(f"{prefix} var {var_name!r} missing required dims {missing!r}.")


def require_core_dim_in_data(
    data: xr.DataArray,
    *,
    dim: str,
    owner: str,
    operand: str,
) -> None:
    if dim not in data.dims:
        raise ValueError(f"{owner}: {operand} is missing core dim {dim!r}.")


def require_explicit_unique_labels(
    data: xr.DataArray,
    *,
    dim: str,
    owner: str,
    operand: str,
) -> tuple[object, ...]:
    require_core_dim_in_data(data, dim=dim, owner=owner, operand=operand)
    if dim not in data.coords:
        raise ValueError(f"{owner}: {operand} must declare explicit coord labels for dim {dim!r}.")
    try:
        index = data.get_index(dim)
    except KeyError as exc:
        raise ValueError(f"{owner}: {operand} must expose dim {dim!r} as an indexable dimension.") from exc
    if index.has_duplicates:
        raise ValueError(f"{owner}: {operand} labels for dim {dim!r} must be unique.")
    return tuple(index.tolist())


def require_exact_label_set(
    labels: tuple[object, ...],
    *,
    expected: tuple[object, ...],
    owner: str,
    component_name: str,
    kind: str = "labels",
) -> None:
    if set(labels) == set(expected):
        return
    raise ValueError(
        f"{owner}: component {component_name!r} {kind} must exactly match declared labels "
        f"{list(expected)!r}; got {list(labels)!r}."
    )


__all__ = [
    "require_component_numeric_var",
    "require_core_dim_in_data",
    "require_declared_roles_with_sequence",
    "require_exact_label_set",
    "require_explicit_unique_labels",
    "select_component_var",
]
