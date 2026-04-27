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
        if dim in child.ds.dims:
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


def datatree_extract_template(tree: xr.DataTree) -> xr.Dataset | None:
    children = datatree_children(tree)
    if not children:
        return None
    first = next(iter(children.values()))
    return first.ds.copy(deep=True)


__all__ = [
    "datatree_children",
    "datatree_extract_template",
    "detect_catalog_backend",
    "normalize_catalog_payload",
    "resolve_catalog_batch_dim",
]
