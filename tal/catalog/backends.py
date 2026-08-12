from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr

from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.schema_read import read_roles

from .types import CatalogBackend

_DATASET_BATCH_DEFAULT = "group"
_DATATREE_BATCH_DEFAULT = "group"


def detect_catalog_backend(value: object, *, backend: str, owner: str) -> CatalogBackend:
    if backend == "dataset":
        return "dataset"
    if backend == "datatree":
        return "datatree"
    if isinstance(value, xr.DataTree):
        return "datatree"
    if isinstance(value, (xr.Dataset, xr.DataArray)) or _is_analysis_object(value):
        return "dataset"
    raise TypeError(
        f"{owner}: unsupported input type {type(value).__name__!r}; expected AnalysisObject, "
        "xr.Dataset, xr.DataArray, or xr.DataTree."
    )


def normalize_catalog_payload(value: object, *, backend: CatalogBackend, owner: str) -> xr.Dataset | xr.DataTree:
    if backend == "dataset":
        ao = coerce_analysis_object_input(value, owner=owner)
        return ao.unsafe_data.copy(deep=True)
    if not isinstance(value, xr.DataTree):
        raise TypeError(f"{owner}: backend='datatree' requires xr.DataTree input.")
    tree = value.copy(deep=True)
    _require_flat_datatree(tree, owner=owner)
    return tree


def resolve_catalog_batch_dim(
    payload: xr.Dataset | xr.DataTree,
    *,
    backend: CatalogBackend,
    batch_dim: str | None,
    owner: str,
) -> tuple[str, xr.Dataset | xr.DataTree]:
    if backend == "dataset":
        ds = payload if isinstance(payload, xr.Dataset) else _invalid_dataset_payload(payload)
        resolved = _resolve_dataset_batch_dim(ds, batch_dim=batch_dim, owner=owner)
        return resolved, _ensure_dataset_batch_coord(ds, batch_dim=resolved)
    tree = payload if isinstance(payload, xr.DataTree) else _invalid_datatree_payload(payload)
    resolved = _resolve_datatree_batch_dim(tree, batch_dim=batch_dim, owner=owner)
    return resolved, tree


def _resolve_dataset_batch_dim(ds: xr.Dataset, *, batch_dim: str | None, owner: str) -> str:
    if batch_dim is not None:
        if batch_dim not in ds.dims:
            raise ValueError(f"{owner}: batch_dim {batch_dim!r} is not a dataset dimension.")
        _validate_explicit_dataset_batch_dim(ds, batch_dim=batch_dim, owner=owner)
        return batch_dim
    roles_declared, _, batch_dims, _ = read_roles(ds)
    if roles_declared and len(batch_dims) == 1:
        dim = batch_dims[0]
        if dim not in ds.dims:
            raise ValueError(f"{owner}: schema batch dim {dim!r} is not present in dataset dims.")
        return dim
    if roles_declared and len(batch_dims) > 1:
        raise ValueError(
            f"{owner}: ambiguous batch-axis resolution from schema batch_dims {batch_dims!r}; provide batch_dim explicitly."
        )
    if _DATASET_BATCH_DEFAULT in ds.dims:
        return _DATASET_BATCH_DEFAULT
    if len(ds.dims) == 1:
        return str(next(iter(ds.dims)))
    raise ValueError(
        f"{owner}: could not resolve batch_dim from dataset dims {tuple(ds.dims)!r}; provide batch_dim explicitly."
    )


def _resolve_datatree_batch_dim(tree: xr.DataTree, *, batch_dim: str | None, owner: str) -> str:
    if batch_dim is not None:
        colliding_child = _first_child_dim_collision(tree, dim=batch_dim)
        if colliding_child is not None:
            raise ValueError(
                f"{owner}: explicit batch_dim {batch_dim!r} collides with child dataset dim in group "
                f"{colliding_child!r}; this override is unsupported for flat-tree extract semantics."
            )
        return batch_dim
    roles_declared, _, batch_dims, _ = read_roles(tree.ds.copy(deep=True))
    if roles_declared and len(batch_dims) == 1:
        return batch_dims[0]
    if roles_declared and len(batch_dims) > 1:
        raise ValueError(
            f"{owner}: ambiguous batch-axis resolution from schema batch_dims {batch_dims!r}; provide batch_dim explicitly."
        )
    return _DATATREE_BATCH_DEFAULT


def _ensure_dataset_batch_coord(ds: xr.Dataset, *, batch_dim: str) -> xr.Dataset:
    if batch_dim in ds.coords:
        return ds
    return ds.assign_coords({batch_dim: xr.DataArray(np.arange(ds.sizes[batch_dim]), dims=(batch_dim,))})


def _require_flat_datatree(tree: xr.DataTree, *, owner: str) -> None:
    for name, child in tree.children.items():
        if child.children:
            raise ValueError(
                f"{owner}: nested DataTree group {name!r} is not supported."
            )


def _validate_explicit_dataset_batch_dim(ds: xr.Dataset, *, batch_dim: str, owner: str) -> None:
    declared, sequence_dim, _, core_dims = read_roles(ds)
    if not declared:
        return
    validate_batch_dim_role_compatibility(
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
        core_dims=core_dims,
        owner=owner,
    )


def validate_batch_dim_role_compatibility(
    *,
    batch_dim: str,
    sequence_dim: str | None,
    core_dims: tuple[str, ...],
    owner: str,
) -> None:
    if sequence_dim is not None and batch_dim == sequence_dim:
        raise ValueError(
            f"{owner}: explicit batch_dim {batch_dim!r} conflicts with declared sequence_dim."
        )
    if batch_dim in core_dims:
        raise ValueError(
            f"{owner}: explicit batch_dim {batch_dim!r} conflicts with declared core_dims {core_dims!r}."
        )


def _first_child_dim_collision(tree: xr.DataTree, *, dim: str) -> str | None:
    for name, child in tree.children.items():
        if dim in child.to_dataset(inherit=False).dims:
            return str(name)
    return None


def _is_analysis_object(value: object) -> bool:
    try:
        from tal.core import AnalysisObject
    except Exception:
        return False
    return isinstance(value, AnalysisObject)


def _invalid_dataset_payload(payload: xr.Dataset | xr.DataTree) -> xr.Dataset:
    raise TypeError(f"Catalog backend internal error: expected xr.Dataset payload; got {type(payload).__name__}.")


def _invalid_datatree_payload(payload: xr.Dataset | xr.DataTree) -> xr.DataTree:
    raise TypeError(f"Catalog backend internal error: expected xr.DataTree payload; got {type(payload).__name__}.")


def datatree_children(tree: xr.DataTree) -> Mapping[str, xr.DataTree]:
    return tree.children


def validate_datatree_root_batch_alignment(
    tree: xr.DataTree,
    *,
    batch_dim: str,
    owner: str,
) -> None:
    root = tree.to_dataset(inherit=False)
    if batch_dim not in root.dims:
        return
    size = int(root.sizes[batch_dim])
    group_count = len(datatree_children(tree))
    if size != group_count:
        raise ValueError(
            f"{owner}: root batch dimension {batch_dim!r} length {size} "
            f"does not match DataTree group count {group_count}."
        )


def datatree_child_payload_dataset(
    child: xr.DataTree,
    *,
    batch_dim: str,
    batch_position: int,
) -> xr.Dataset:
    local = child.to_dataset(inherit=False)
    parent = child.parent
    if parent is None:
        return local
    parent_ds = parent.to_dataset(inherit=False)
    payload_dims = set(local.dims)
    inherited_payload_coords: dict[str, xr.Variable] = {}
    for name, coord in parent_ds.coords.items():
        if name in local.variables:
            continue
        resolved = _root_payload_coord(
            coord,
            payload_dims=payload_dims,
            batch_dim=batch_dim,
            batch_position=batch_position,
        )
        if resolved is not None:
            inherited_payload_coords[name] = resolved.variable
    if not inherited_payload_coords:
        return local
    indexes = _inherited_payload_indexes(
        parent_ds,
        coord_names=frozenset(inherited_payload_coords),
        batch_dim=batch_dim,
    )
    coords = xr.Coordinates(inherited_payload_coords, indexes=indexes)
    return local.assign_coords(coords)


def _inherited_payload_indexes(
    parent: xr.Dataset,
    *,
    coord_names: frozenset[str],
    batch_dim: str,
) -> dict[object, xr.Index]:
    indexes: dict[object, xr.Index] = {}
    for index, variables in parent.xindexes.group_by_index():
        names = frozenset(variables)
        if not names or not names.issubset(coord_names):
            continue
        if any(batch_dim in variable.dims for variable in variables.values()):
            continue
        indexes.update(dict.fromkeys(names, index))
    return indexes


def _root_payload_coord(
    coord: xr.DataArray,
    *,
    payload_dims: set[str],
    batch_dim: str,
    batch_position: int,
) -> xr.DataArray | None:
    if coord.dims and set(coord.dims).issubset(payload_dims):
        return coord
    if batch_dim not in coord.dims:
        return None
    row_dims = tuple(dim for dim in coord.dims if dim != batch_dim)
    if not row_dims or not set(row_dims).issubset(payload_dims):
        return None
    return coord.isel({batch_dim: batch_position}, drop=True)


def datatree_extract_template(
    tree: xr.DataTree,
    *,
    batch_dim: str,
    owner: str,
) -> xr.Dataset | None:
    validate_datatree_root_batch_alignment(tree, batch_dim=batch_dim, owner=owner)
    children = datatree_children(tree)
    if not children:
        return None
    first = next(iter(children.values()))
    return datatree_child_payload_dataset(
        first,
        batch_dim=batch_dim,
        batch_position=0,
    ).copy(deep=False)


__all__ = [
    "datatree_child_payload_dataset",
    "datatree_children",
    "datatree_extract_template",
    "detect_catalog_backend",
    "normalize_catalog_payload",
    "resolve_catalog_batch_dim",
    "validate_datatree_root_batch_alignment",
]
