from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from ..dataset_ownership import analysis_object_dataset
from ..orchestration.finalize import finalize_like
from ..validity_finalize import set_left_packed_validity_or_prune_from_size_coord

if TYPE_CHECKING:
    from ..analysis_object import AnalysisObject


def finalize_event_output(
    base: "AnalysisObject",
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    batch_dims: tuple[str, ...],
    core_dims: tuple[str, ...],
    param_name: str,
    size_name: str,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    out = finalize_like(base, ds, validate=validate, owner=owner)
    out = out.set_roles(
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        core_dims=core_dims,
        validate=validate,
    )
    if param_name not in analysis_object_dataset(out).coords:
        raise ValueError(f"{owner}: gathered output is missing param coord {param_name!r}.")
    out = out.set_param_coord(name=param_name, validate=validate)
    return set_left_packed_validity_or_prune_from_size_coord(
        out,
        size_name=size_name,
        validate=validate,
        owner=owner,
    )


__all__ = ["finalize_event_output"]
