from __future__ import annotations

"""Schema-aware role/param/validity resolution for param-engine entrypoints."""

from collections.abc import Sequence
from typing import Any

import xarray as xr

from ..schema_read import (
    read_param_coord as _read_param_coord,
    read_roles as _read_roles_raw,
    read_validity as _read_validity,
    validate_schema_if_needed as _validate_schema_if_needed_raw,
)
from ..schema_errors import schema_error
from .types import ParamCoordSpec, ParamSchemaContext

# Internal compatibility alias used by tests and fast-path monkeypatching.
_validate_schema = _validate_schema_if_needed_raw


def _schema_fail(*, code: str, path: str, expected: Any, actual: Any, hint: str) -> None:
    raise schema_error(code=code, path=path, expected=expected, actual=actual, hint=hint)


def _validate_schema_if_needed(ds: xr.Dataset, *, assume_validated: bool) -> xr.Dataset:
    if assume_validated:
        return ds
    return _validate_schema(ds)


def _read_roles(ds: xr.Dataset) -> tuple[str | None, tuple[str, ...], tuple[str, ...], bool]:
    declared, sequence_dim, batch_dims, core_dims = _read_roles_raw(ds)
    return sequence_dim, batch_dims, core_dims, declared


def _check_role_conflicts(
    *,
    explicit_sequence_dim: str | None,
    explicit_batch_dims: tuple[str, ...] | None,
    declared_sequence_dim: str | None,
    declared_batch_dims: tuple[str, ...],
) -> None:
    if explicit_sequence_dim and declared_sequence_dim and explicit_sequence_dim != declared_sequence_dim:
        raise ValueError(
            "resolve_role_dims: explicit sequence_dim conflicts with declared roles.sequence_dim "
            f"({explicit_sequence_dim!r} != {declared_sequence_dim!r})."
        )
    if explicit_batch_dims is None or declared_sequence_dim is None:
        return
    if explicit_batch_dims != declared_batch_dims:
        raise ValueError(
            "resolve_role_dims: explicit batch_dims conflicts with declared roles.batch_dims "
            f"({explicit_batch_dims!r} != {declared_batch_dims!r})."
        )


def _check_name_conflict(*, label: str, explicit: str | None, declared: str | None, declared_present: bool) -> None:
    if explicit is None or not declared_present or declared is None:
        return
    if explicit == declared:
        return
    raise ValueError(
        f"{label}: explicit override conflicts with declared schema ({explicit!r} != {declared!r})."
    )


def _resolve_effective_roles(
    ds: xr.Dataset,
    *,
    explicit_sequence_dim: str | None,
    explicit_batch_dims: tuple[str, ...] | None,
    declared_sequence_dim: str | None,
    declared_batch_dims: tuple[str, ...],
    declared_core_dims: tuple[str, ...],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    sequence_dim = explicit_sequence_dim or declared_sequence_dim
    if sequence_dim is None:
        raise ValueError("resolve_role_dims: no roles declared and no explicit sequence_dim provided.")
    batch_dims = explicit_batch_dims if explicit_batch_dims is not None else declared_batch_dims
    if sequence_dim in batch_dims:
        raise ValueError(
            f"resolve_role_dims: sequence_dim {sequence_dim!r} cannot appear in batch_dims {batch_dims!r}."
        )
    missing = [dim for dim in (sequence_dim, *batch_dims) if dim not in ds.dims]
    if missing:
        raise ValueError(f"resolve_role_dims: missing dims in dataset: {missing!r}.")
    core_dims = tuple(dim for dim in declared_core_dims if dim in ds.dims and dim not in (sequence_dim, *batch_dims))
    if declared_sequence_dim is None:
        core_dims = tuple(dim for dim in ds.dims if dim not in (sequence_dim, *batch_dims))
    return str(sequence_dim), batch_dims, core_dims


def _validate_param_target(
    ds: xr.Dataset,
    *,
    name: str | None,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    declared: bool,
) -> None:
    if name is None:
        return
    if name not in ds.coords:
        if declared:
            _schema_fail(
                code="schema.param_coord.not_found",
                path="tal.core.param_coord.name",
                expected="coord in ds.coords",
                actual={"coord": name, "coords": list(ds.coords)},
                hint="add the coordinate or choose an existing coord name",
            )
        raise ValueError(f"resolve_param_coord: explicit coord {name!r} not found in dataset coords.")
    dims = tuple(ds.coords[name].dims)
    allowed = ((sequence_dim,), batch_dims + (sequence_dim,))
    if dims in allowed:
        return
    if declared:
        _schema_fail(
            code="schema.param_coord.dims.invalid",
            path="tal.core.param_coord.name",
            expected=[list(item) for item in allowed],
            actual={"coord": name, "dims": list(dims)},
            hint="use dims (sequence_dim,) or (*batch_dims, sequence_dim)",
        )
    raise ValueError(
        f"resolve_param_coord: explicit coord {name!r} dims {dims!r} must be {allowed!r}."
    )


def _validate_sequence_size_target(
    ds: xr.Dataset,
    *,
    name: str | None,
    batch_dims: tuple[str, ...],
    declared: bool,
) -> None:
    if name is None:
        return
    if name not in ds.coords:
        if declared:
            _schema_fail(
                code="schema.validity.sequence_size_coord.not_found",
                path="tal.core.validity.sequence_size_coord",
                expected="coord in ds.coords",
                actual={"coord": name, "coords": list(ds.coords)},
                hint="add the coordinate or choose an existing coord name",
            )
        raise ValueError(f"resolve_param_valid_mask: explicit sequence_size_coord {name!r} not found.")
    dims = tuple(ds.coords[name].dims)
    expected = batch_dims if batch_dims else ()
    if dims == expected:
        return
    if declared:
        _schema_fail(
            code="schema.validity.sequence_size_coord.dims.invalid",
            path="tal.core.validity.sequence_size_coord",
            expected=list(batch_dims),
            actual={"coord": name, "dims": list(dims)},
            hint="set sequence_size_coord dims to exactly batch_dims",
        )
    raise ValueError(
        f"resolve_param_valid_mask: explicit sequence_size_coord dims {dims!r} must equal {expected!r}."
    )


def _resolve_param_name(
    ds: xr.Dataset,
    *,
    explicit_param_name: str | None,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
) -> tuple[str | None, bool]:
    declared, declared_name = _read_param_coord(ds)
    _check_name_conflict(
        label="resolve_param_coord",
        explicit=explicit_param_name,
        declared=declared_name,
        declared_present=declared,
    )
    param_name = explicit_param_name or declared_name
    _validate_param_target(ds, name=param_name, sequence_dim=sequence_dim, batch_dims=batch_dims, declared=declared)
    return param_name, declared


def _resolve_sequence_size_name(
    ds: xr.Dataset,
    *,
    explicit_sequence_size_coord: str | None,
    batch_dims: tuple[str, ...],
) -> tuple[str | None, bool]:
    declared, declared_name, _ = _read_validity(ds)
    _check_name_conflict(
        label="resolve_param_valid_mask",
        explicit=explicit_sequence_size_coord,
        declared=declared_name,
        declared_present=declared,
    )
    size_name = explicit_sequence_size_coord or declared_name
    _validate_sequence_size_target(ds, name=size_name, batch_dims=batch_dims, declared=declared)
    return size_name, declared


def _normalize_explicit_batch_dims(explicit_batch_dims: Sequence[str] | None) -> tuple[str, ...] | None:
    if explicit_batch_dims is None:
        return None
    normalized = tuple(str(dim) for dim in explicit_batch_dims)
    seen: set[str] = set()
    duplicates: list[str] = []
    for dim in normalized:
        if dim in seen and dim not in duplicates:
            duplicates.append(dim)
        seen.add(dim)
    if duplicates:
        raise ValueError(
            "resolve_role_dims: explicit batch_dims contains duplicate entries "
            f"{duplicates!r}; remove duplicate entries from batch_dims."
        )
    return normalized


def _resolve_schema_context(
    ds: xr.Dataset,
    *,
    explicit_sequence_dim: str | None = None,
    explicit_batch_dims: Sequence[str] | None = None,
    explicit_param_name: str | None = None,
    explicit_sequence_size_coord: str | None = None,
    assume_validated: bool = False,
) -> ParamSchemaContext:
    """Resolve effective schema context from explicit overrides and declared schema."""
    ds_checked = _validate_schema_if_needed(ds, assume_validated=assume_validated)
    dec_seq, dec_batch, dec_core, _ = _read_roles(ds_checked)
    explicit_batch = _normalize_explicit_batch_dims(explicit_batch_dims)
    _check_role_conflicts(
        explicit_sequence_dim=explicit_sequence_dim,
        explicit_batch_dims=explicit_batch,
        declared_sequence_dim=dec_seq,
        declared_batch_dims=dec_batch,
    )
    seq, batch, core = _resolve_effective_roles(
        ds_checked,
        explicit_sequence_dim=explicit_sequence_dim,
        explicit_batch_dims=explicit_batch,
        declared_sequence_dim=dec_seq,
        declared_batch_dims=dec_batch,
        declared_core_dims=dec_core,
    )
    param_name, param_declared = _resolve_param_name(
        ds_checked,
        explicit_param_name=explicit_param_name,
        sequence_dim=seq,
        batch_dims=batch,
    )
    size_name, val_declared = _resolve_sequence_size_name(
        ds_checked,
        explicit_sequence_size_coord=explicit_sequence_size_coord,
        batch_dims=batch,
    )
    return ParamSchemaContext(
        ds=ds_checked,
        sequence_dim=seq,
        batch_dims=batch,
        core_dims=core,
        param_name=param_name,
        param_declared=param_declared,
        sequence_size_coord=size_name,
        validity_declared=val_declared,
    )


def _resolve_schema_context_validated(
    ds: xr.Dataset,
    *,
    explicit_sequence_dim: str | None = None,
    explicit_batch_dims: Sequence[str] | None = None,
    explicit_param_name: str | None = None,
    explicit_sequence_size_coord: str | None = None,
) -> ParamSchemaContext:
    return _resolve_schema_context(
        ds,
        explicit_sequence_dim=explicit_sequence_dim,
        explicit_batch_dims=explicit_batch_dims,
        explicit_param_name=explicit_param_name,
        explicit_sequence_size_coord=explicit_sequence_size_coord,
        assume_validated=True,
    )


def resolve_role_dims(
    ds: xr.Dataset,
    *,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Resolve sequence and batch role dimensions for param operations.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : Sequence[str] | None, optional
        Optional override for batch dimensions used by temporal semantics.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    context = _resolve_schema_context(
        ds,
        explicit_sequence_dim=sequence_dim,
        explicit_batch_dims=batch_dims,
    )
    return context.sequence_dim, context.batch_dims


def declared_param_coord_name(ds: xr.Dataset) -> str | None:
    ds_checked = _validate_schema_if_needed(ds, assume_validated=False)
    declared, name = _read_param_coord(ds_checked)
    return name if declared else None


def resolve_param_coord_name(ds: xr.Dataset, *, explicit_name: str | None = None) -> str | None:
    if explicit_name is None:
        return declared_param_coord_name(ds)
    ds_checked = _validate_schema_if_needed(ds, assume_validated=False)
    if explicit_name not in ds_checked.coords:
        raise ValueError(f"resolve_param_coord_name: explicit coord {explicit_name!r} not found.")
    sequence_dim, batch_dims, _, declared = _read_roles(ds_checked)
    if declared and sequence_dim is not None:
        _validate_param_target(
            ds_checked,
            name=explicit_name,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            declared=False,
        )
    return explicit_name


def resolve_param_coord(
    ds: xr.Dataset,
    *,
    explicit_name: str | None = None,
    sequence_dim: str | None = None,
    batch_dims: Sequence[str] | None = None,
) -> ParamCoordSpec | None:
    """Resolve canonical param coordinate spec or return ``None`` when undeclared/unset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset/source value processed by this operation.
    explicit_name : str | None, optional
        Explicit coordinate name override supplied by the caller.
    sequence_dim : str | None, optional
        Optional override for the sequence dimension used by temporal semantics.
    batch_dims : Sequence[str] | None, optional
        Optional override for batch dimensions used by temporal semantics.

    Returns
    -------
    ParamCoordSpec | None
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    context = _resolve_schema_context(
        ds,
        explicit_sequence_dim=sequence_dim,
        explicit_batch_dims=batch_dims,
        explicit_param_name=explicit_name,
    )
    if context.param_name is None:
        return None
    coord = context.ds.coords[context.param_name]
    return ParamCoordSpec(
        name=context.param_name,
        coord=coord,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
    )


def declared_sequence_size_coord_name(ds: xr.Dataset) -> str | None:
    ds_checked = _validate_schema_if_needed(ds, assume_validated=False)
    declared, name, _ = _read_validity(ds_checked)
    return name if declared else None


__all__ = [
    "declared_param_coord_name",
    "declared_sequence_size_coord_name",
    "resolve_param_coord",
    "resolve_param_coord_name",
    "resolve_role_dims",
]
