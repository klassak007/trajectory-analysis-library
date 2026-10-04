"""Structural validity for wholly unavailable spatial temporal results."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core.param_ops.guards import mark_generated_size_coord
from tal.core.schema import set_validity
from tal.core.schema_read import read_roles, read_sequence_size_coord_name
from tal.core.validity_values import require_valid_sequence_size_values
from tal.utils.xarray_namespace import dataset_namespace_names, unique_temp_dim


def has_no_usable_sequence_rows(source: xr.Dataset, *, owner: str) -> bool:
    """Read only the declared source-size coordinate, never the payload."""
    _, sequence_dim, _, _ = read_roles(source)
    if sequence_dim is None:
        return False
    sequence_len = int(source.sizes[sequence_dim])
    if sequence_len == 0:
        return True
    size_name = read_sequence_size_coord_name(source)
    if size_name is None:
        return False
    counts = require_valid_sequence_size_values(
        source.coords[size_name], sequence_size_coord=size_name,
        sequence_len=sequence_len, owner=owner,
    )
    return bool(np.all(counts.data == 0))


def declare_no_usable_sequence_rows(output: xr.Dataset, *, owner: str) -> xr.Dataset:
    """Give all-missing results an explicit zero-length validity domain."""
    _, sequence_dim, batch_dims, _ = read_roles(output)
    if sequence_dim is None:
        raise ValueError(f"{owner}: empty spatial result requires a sequence dimension.")
    size_name = read_sequence_size_coord_name(output)
    if size_name is None:
        size_name = unique_temp_dim(
            f"{sequence_dim}_size", taken_dims=dataset_namespace_names(output),
        )
    zeros = xr.DataArray(
        np.zeros(tuple(int(output.sizes[dim]) for dim in batch_dims), dtype=np.int64),
        dims=batch_dims,
    )
    with_size = output.assign_coords({size_name: zeros})
    with_size = mark_generated_size_coord(with_size, name=size_name)
    return set_validity(with_size, sequence_size_coord=size_name, validate=False)


__all__ = ["declare_no_usable_sequence_rows", "has_no_usable_sequence_rows"]
