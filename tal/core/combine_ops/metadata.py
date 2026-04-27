from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from ..metadata_optional import (
    align_param_coord_to_canonical,
    align_size_coord_to_canonical,
    canonical_param_dims,
    canonical_size_dims,
    canonicalize_optional_names,
    shared_optional_name,
)
from .types import CombineContext


def resolve_core_dims(
    contexts: Sequence[CombineContext],
    *,
    ds: xr.Dataset,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    owner: str,
) -> tuple[str, ...]:
    base = _shared_declared_core_dims(contexts, owner=owner)
    excluded = set(batch_dims + ((sequence_dim,) if sequence_dim else ()))
    return tuple(dim for dim in base if dim in ds.dims and dim not in excluded)


def _shared_declared_core_dims(
    contexts: Sequence[CombineContext],
    *,
    owner: str,
) -> tuple[str, ...]:
    declared = [ctx.core_dims for ctx in contexts if ctx.roles_declared and ctx.core_dims]
    if not declared:
        return ()
    base = declared[0]
    for dims in declared[1:]:
        if dims != base:
            raise ValueError(f"{owner}: declared core_dims conflict across inputs {base!r} vs {dims!r}.")
    return base


__all__ = [
    "align_param_coord_to_canonical",
    "align_size_coord_to_canonical",
    "canonicalize_optional_names",
    "canonical_param_dims",
    "canonical_size_dims",
    "resolve_core_dims",
    "shared_optional_name",
]
