from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr
from tal.utils.xarray_namespace import dataset_namespace_names, unique_temp_dim

from ..schema_read import read_roles

_RESERVED_OWNER_KEY = "tal_reserved_owner"
_RESERVED_NAME_KEY = "tal_reserved_name"
_RESERVED_TOKEN_KEY = "tal_reserved_token"
_RESERVED_OWNER_VALUE = "param_ops"
_RESERVED_TOKEN_VALUE = "tal:param_ops:reserved:v1"
_RESERVED_QUERY_DIM_NAMES = frozenset({"valid", "sample_index"})


def assert_unique_dim_labels(
    obj: xr.Dataset | xr.DataArray,
    *,
    dim: str,
    owner: str,
) -> None:
    if dim not in obj.dims:
        return
    index = obj.get_index(dim)
    if bool(getattr(index, "is_unique", True)):
        return
    raise ValueError(
        f"{owner}: labels along {dim!r} must be unique. "
        "Provide unique coordinate labels for batch/query alignment."
    )


def assert_query_dim_safe(
    ds: xr.Dataset,
    *,
    sequence_dim: str,
    query_dim: str,
    owner: str,
) -> None:
    if query_dim in ds.dims and query_dim != sequence_dim:
        raise ValueError(
            f"{owner}: query_dim {query_dim!r} collides with existing non-sequence dim. "
            "Choose a distinct query_dim for param operations."
        )
    if query_dim in ds.coords and query_dim not in ds.dims:
        raise ValueError(
            f"{owner}: query_dim {query_dim!r} collides with existing coordinate name. "
            "Choose a distinct query_dim for param operations."
        )
    if query_dim in ds.data_vars and query_dim != sequence_dim:
        raise ValueError(
            f"{owner}: query_dim {query_dim!r} collides with existing data variable name. "
            "Choose a distinct query_dim for param operations."
        )


def validate_query_dim_name(
    query_dim: object,
    *,
    owner: str,
) -> str:
    if not isinstance(query_dim, str) or not query_dim:
        raise ValueError(f"{owner}: query_dim must be a non-empty string.")
    if query_dim in _RESERVED_QUERY_DIM_NAMES:
        raise ValueError(
            f"{owner}: query_dim {query_dim!r} is reserved for TAL runtime metadata. "
            "Choose a distinct query_dim for param operations."
        )
    return query_dim


def mark_reserved_coord(da: xr.DataArray, *, name: str) -> xr.DataArray:
    return da.assign_attrs(
        {
            _RESERVED_OWNER_KEY: _RESERVED_OWNER_VALUE,
            _RESERVED_NAME_KEY: str(name),
            _RESERVED_TOKEN_KEY: _RESERVED_TOKEN_VALUE,
        }
    )


def _reserved_coord_is_owned(ds: xr.Dataset, *, name: str) -> bool:
    attrs = ds.coords[name].attrs
    return (
        attrs.get(_RESERVED_OWNER_KEY) == _RESERVED_OWNER_VALUE
        and attrs.get(_RESERVED_NAME_KEY) == str(name)
        and attrs.get(_RESERVED_TOKEN_KEY) == _RESERVED_TOKEN_VALUE
    )


def _indexes_align_with_dataset(
    ds: xr.Dataset,
    *,
    coord: xr.DataArray,
) -> bool:
    for dim in coord.dims:
        if dim not in ds.dims:
            return False
        if not coord.get_index(dim).equals(ds.get_index(dim)):
            return False
    return True


def _is_compatible_valid_coord(
    ds: xr.Dataset,
    *,
    name: str,
) -> bool:
    if name != "valid":
        return False
    coord = ds.coords[name]
    if coord.dtype.kind != "b":
        return False
    if not _indexes_align_with_dataset(ds, coord=coord):
        return False
    declared, sequence_dim, batch_dims, _ = read_roles(ds)
    dims = tuple(coord.dims)
    if not declared or sequence_dim is None or not batch_dims:
        return False
    allowed = {(sequence_dim,), batch_dims + (sequence_dim,)}
    return dims in allowed


def coerce_float_scalar(value: object, *, owner: str, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{owner}: {field} must be a numeric scalar.")
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: {field} must be a numeric scalar.") from exc
    if arr.ndim != 0:
        raise ValueError(f"{owner}: {field} must be a numeric scalar.")
    if arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{owner}: {field} must be a numeric scalar.")
    try:
        return float(arr.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner}: {field} must be a numeric scalar.") from exc


def assert_reserved_metadata_safe(
    ds: xr.Dataset,
    *,
    reserved: Sequence[str],
    param_name: str,
    owner: str,
) -> None:
    names = tuple(str(name) for name in reserved)
    collisions = [name for name in names if name in ds.dims]
    collisions.extend(name for name in names if name in ds.data_vars)
    for name in names:
        if name not in ds.coords:
            continue
        if _reserved_coord_is_owned(ds, name=name):
            continue
        if _is_compatible_valid_coord(ds, name=name):
            continue
        collisions.append(name)
    if param_name in names:
        collisions.append(param_name)
    if not collisions:
        return
    unique = sorted(set(collisions))
    raise ValueError(
        f"{owner}: reserved metadata name collision for {unique!r}. "
        "Rename conflicting coords/data_vars/param_coord before calling param operations."
    )


__all__ = [
    "dataset_namespace_names",
    "mark_reserved_coord",
    "assert_query_dim_safe",
    "validate_query_dim_name",
    "assert_reserved_metadata_safe",
    "assert_unique_dim_labels",
    "coerce_float_scalar",
    "unique_temp_dim",
]
