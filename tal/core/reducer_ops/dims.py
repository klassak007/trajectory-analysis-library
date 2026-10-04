from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from .types import DimLike


def normalize_reduce_dim_input(dim: DimLike, *, owner: str) -> tuple[str, ...] | None:
    if dim is None:
        return None
    if isinstance(dim, str):
        return (dim,)
    if not isinstance(dim, Sequence):
        raise TypeError(f"{owner}: dim must be str, sequence[str], or None.")
    out: list[str] = []
    for value in dim:
        if not isinstance(value, str):
            raise TypeError(f"{owner}: dim entries must be strings; got {type(value).__name__}.")
        out.append(value)
    return tuple(out)


def _dedupe_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def resolve_reduce_dims(
    ds: xr.Dataset,
    *,
    dim: DimLike,
    component_dims: Sequence[str],
    owner: str,
) -> tuple[str, ...]:
    explicit = normalize_reduce_dim_input(dim, owner=owner)
    blocked = set(component_dims)
    if explicit is None:
        return tuple(name for name in ds.dims if name not in blocked)
    resolved = _dedupe_preserve_order(explicit)
    missing = tuple(name for name in resolved if name not in ds.dims)
    if missing:
        raise ValueError(f"{owner}: dim entries must reference dataset dims; missing={missing!r}.")
    blocked_used = tuple(name for name in resolved if name in blocked)
    if blocked_used:
        raise ValueError(
            f"{owner}: explicit reduction over required component dims is not allowed: {blocked_used!r}."
        )
    return resolved


def resolve_active_reduce_dims(
    ds: xr.Dataset,
    *,
    names: Sequence[str],
    reduce_dims: Sequence[str],
) -> tuple[str, ...]:
    """Return requested dimensions used by at least one eligible payload."""
    return tuple(
        dim
        for dim in reduce_dims
        if any(dim in ds[name].dims for name in names)
    )


__all__ = [
    "normalize_reduce_dim_input",
    "resolve_active_reduce_dims",
    "resolve_reduce_dims",
]
