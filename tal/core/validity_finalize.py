from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import xarray as xr

from .dataset_ownership import analysis_object_dataset
from .orchestration.lazy import is_chunked_dataarray
from .schema import set_validity
from .schema_errors import SchemaError
from .schema_read import read_roles, read_sequence_size_coord_name
from .validity_layout import is_left_packed_mask, sequence_size_from_mask

if TYPE_CHECKING:
    from .analysis_object import AnalysisObject

TransformKind = Literal["identity", "prefix", "unsafe"]


def assign_sequence_size_from_valid_mask(
    ds: xr.Dataset,
    *,
    valid: xr.DataArray | None,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    sequence_size_coord: str | None,
) -> tuple[xr.Dataset, str | None]:
    if valid is None or sequence_dim is None or sequence_size_coord is None:
        return ds, None
    if not is_left_packed_mask(valid, sequence_dim=sequence_dim):
        return ds.drop_vars(sequence_size_coord, errors="ignore"), None
    size = sequence_size_from_mask(
        valid,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
    )
    return ds.assign_coords({sequence_size_coord: size}), sequence_size_coord


def set_left_packed_validity_or_prune_from_size_coord(
    ao: "AnalysisObject",
    *,
    size_name: str,
    validate: bool,
    owner: str,
) -> "AnalysisObject":
    ds = analysis_object_dataset(ao)
    if size_name not in ds.coords:
        raise ValueError(f"{owner}: missing sequence_size_coord {size_name!r}.")
    if is_chunked_dataarray(ds.coords[size_name]):
        return ao.set_validity(sequence_size_coord=None, validate=validate)
    return ao.set_validity(sequence_size_coord=size_name, layout="left_packed", validate=validate)


def _try_read_roles(ds: xr.Dataset) -> tuple[bool, str | None, tuple[str, ...]] | None:
    try:
        declared, sequence_dim, batch_dims, _ = read_roles(ds)
    except SchemaError:
        return None
    return declared, sequence_dim, batch_dims


def _try_read_validity_name(ds: xr.Dataset) -> str | None | Literal["__invalid__"]:
    try:
        return read_sequence_size_coord_name(ds)
    except SchemaError:
        return "__invalid__"


def _read_sequence_index(ds: xr.Dataset, sequence_dim: str):
    if sequence_dim not in ds.dims:
        return None
    if sequence_dim not in ds.indexes:
        return None
    try:
        return ds.get_index(sequence_dim)
    except Exception:
        return None


def _read_dim_index(ds: xr.Dataset, dim: str):
    if dim not in ds.dims or dim not in ds.indexes:
        return None
    try:
        return ds.get_index(dim)
    except Exception:
        return None


def _batch_topology_matches(
    source_ds: xr.Dataset,
    candidate_ds: xr.Dataset,
    *,
    source_batch_dims: tuple[str, ...],
    candidate_batch_dims: tuple[str, ...],
    rename_map: Mapping[str, str] | None,
) -> bool:
    mapped_source_dims = tuple(
        dim if rename_map is None else rename_map.get(dim, dim)
        for dim in source_batch_dims
    )
    if mapped_source_dims != candidate_batch_dims:
        return False
    for source_dim, candidate_dim in zip(source_batch_dims, candidate_batch_dims, strict=True):
        source_index = _read_dim_index(source_ds, source_dim)
        candidate_index = _read_dim_index(candidate_ds, candidate_dim)
        if source_index is None or candidate_index is None:
            return False
        if not source_index.equals(candidate_index):
            return False
    return True


def _classify_sequence_transform(
    source_ds: xr.Dataset,
    candidate_ds: xr.Dataset,
    *,
    source_sequence_dim: str,
    candidate_sequence_dim: str,
) -> TransformKind:
    source_index = _read_sequence_index(source_ds, source_sequence_dim)
    candidate_index = _read_sequence_index(candidate_ds, candidate_sequence_dim)
    if source_index is None or candidate_index is None:
        return "unsafe"
    if source_index.equals(candidate_index):
        return "identity"
    if len(candidate_index) > len(source_index):
        return "unsafe"
    if source_index[: len(candidate_index)].equals(candidate_index):
        return "prefix"
    return "unsafe"


def _clamp_sequence_size_coord(
    ds: xr.Dataset,
    *,
    coord_name: str,
    max_size: int,
) -> xr.Dataset:
    coord = ds.coords[coord_name]
    clamped = xr.where(coord > max_size, max_size, coord)
    if coord.name is not None:
        clamped = clamped.rename(coord.name)
    return ds.assign_coords({coord_name: clamped})


def _coord_is_chunked(ds: xr.Dataset, coord_name: str) -> bool:
    if coord_name not in ds.coords:
        return False
    return getattr(ds.coords[coord_name].data, "chunks", None) is not None


def reconcile_sequence_validity_after_structure(
    source_ds: xr.Dataset,
    candidate_ds: xr.Dataset,
    *,
    validate: bool,
    owner: str,
    rename_map: Mapping[str, str] | None = None,
) -> xr.Dataset:
    _ = (validate, owner)
    candidate_roles = _try_read_roles(candidate_ds)
    if candidate_roles is None:
        return candidate_ds
    declared, candidate_sequence_dim, candidate_batch_dims = candidate_roles
    if not declared or candidate_sequence_dim is None:
        return candidate_ds
    candidate_size_name = _try_read_validity_name(candidate_ds)
    if candidate_size_name in (None, "__invalid__"):
        return candidate_ds
    if _coord_is_chunked(candidate_ds, candidate_size_name):
        return set_validity(candidate_ds, sequence_size_coord=None, validate=False)
    source_roles = _try_read_roles(source_ds)
    source_size_name = _try_read_validity_name(source_ds)
    if source_roles is None or source_size_name in (None, "__invalid__"):
        return candidate_ds
    source_declared, source_sequence_dim, source_batch_dims = source_roles
    if not source_declared or source_sequence_dim is None:
        return set_validity(candidate_ds, sequence_size_coord=None, validate=False)
    if source_sequence_dim != candidate_sequence_dim:
        mapped_sequence_dim = None if rename_map is None else rename_map.get(source_sequence_dim)
        if mapped_sequence_dim != candidate_sequence_dim:
            return set_validity(candidate_ds, sequence_size_coord=None, validate=False)
    if not _batch_topology_matches(
        source_ds,
        candidate_ds,
        source_batch_dims=source_batch_dims,
        candidate_batch_dims=candidate_batch_dims,
        rename_map=rename_map,
    ):
        return set_validity(candidate_ds, sequence_size_coord=None, validate=False)
    transform = _classify_sequence_transform(
        source_ds,
        candidate_ds,
        source_sequence_dim=source_sequence_dim,
        candidate_sequence_dim=candidate_sequence_dim,
    )
    if transform == "identity":
        return candidate_ds
    if transform == "prefix":
        max_size = int(candidate_ds.sizes.get(candidate_sequence_dim, 0))
        return _clamp_sequence_size_coord(
            candidate_ds,
            coord_name=candidate_size_name,
            max_size=max_size,
        )
    return set_validity(candidate_ds, sequence_size_coord=None, validate=False)


__all__ = [
    "assign_sequence_size_from_valid_mask",
    "reconcile_sequence_validity_after_structure",
    "set_left_packed_validity_or_prune_from_size_coord",
]
