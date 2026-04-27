from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from tal.core.orchestration.lazy import is_chunked_dataarray

from .backends import datatree_children
from .options import CatalogMetadataPromotionOptions, CatalogQueryOptions
from .query_plan import QueryFieldIndex
from .query_types import FieldRef
from .types import CatalogState

_MISSING = object()


@dataclass(frozen=True)
class MetadataProjection:
    labels: tuple[object, ...]
    columns: Mapping[tuple[str, str], np.ndarray]


def discover_metadata_fields(state: CatalogState, *, owner: str) -> QueryFieldIndex:
    if state.backend == "dataset":
        ds = _require_dataset_payload(state.data)
        return QueryFieldIndex(
            attr=frozenset(_dataset_attr_names(ds)),
            coord=frozenset(_dataset_scalar_coord_names(ds)),
            batch=frozenset(_dataset_batch_coord_names(ds, batch_dim=state.batch_dim)),
        )
    tree = _require_datatree_payload(state.data)
    return QueryFieldIndex(
        attr=frozenset(_datatree_attr_names(tree)),
        coord=frozenset(_datatree_scalar_coord_names(tree)),
        batch=frozenset(_datatree_batch_names(tree, batch_dim=state.batch_dim, owner=owner)),
    )


def project_metadata(
    state: CatalogState,
    *,
    fields: Sequence[FieldRef],
    options: CatalogQueryOptions,
    owner: str,
) -> MetadataProjection:
    labels = _catalog_labels(state)
    if not fields:
        return MetadataProjection(labels=labels, columns={})
    columns: dict[tuple[str, str], np.ndarray] = {}
    for field in fields:
        if field.namespace is None:
            raise TypeError(f"{owner}: projected field {field.name!r} was not namespace-resolved.")
        key = (field.namespace, field.name)
        if key in columns:
            continue
        columns[key] = _project_one_field(state, field=field, options=options, owner=owner)
    return MetadataProjection(labels=labels, columns=columns)


def collect_extract_metadata_columns(
    state: CatalogState,
    *,
    owner: str,
    options: CatalogQueryOptions,
) -> dict[str, np.ndarray]:
    if state.backend != "datatree":
        return {}
    index = discover_metadata_fields(state, owner=owner)
    namespaced: dict[str, np.ndarray] = {}
    for name in sorted(index.attr):
        arr = _project_one_field(
            state,
            field=FieldRef(namespace="attr", name=name),
            options=options,
            owner=owner,
        )
        _add_named_metadata(namespaced, name=name, values=arr, owner=owner)
    for name in sorted(index.coord):
        arr = _project_one_field(
            state,
            field=FieldRef(namespace="coord", name=name),
            options=options,
            owner=owner,
        )
        _add_named_metadata(namespaced, name=name, values=arr, owner=owner)
    return namespaced


def promote_extract_metadata(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    metadata: Mapping[str, np.ndarray],
    options: CatalogMetadataPromotionOptions,
    owner: str,
) -> xr.Dataset:
    out = ds
    for name, values in metadata.items():
        is_scalar = _is_scalar_column(values)
        if is_scalar:
            out = _promote_scalar_column(
                out,
                batch_dim=batch_dim,
                name=name,
                scalar=_column_scalar(values),
                target=options.scalar_target,
                owner=owner,
            )
            continue
        if options.nonscalar_target == "none":
            continue
        _require_promotion_name_available(out, name=name, target="attrs", owner=owner)
        attrs = dict(out.attrs)
        attrs[name] = values.tolist()
        out = out.copy(deep=False)
        out.attrs = attrs
    return out


def _catalog_labels(state: CatalogState) -> tuple[object, ...]:
    if state.backend == "dataset":
        ds = _require_dataset_payload(state.data)
        return tuple(ds.coords[state.batch_dim].to_numpy().tolist())
    tree = _require_datatree_payload(state.data)
    return tuple(datatree_children(tree).keys())


def _project_one_field(
    state: CatalogState,
    *,
    field: FieldRef,
    options: CatalogQueryOptions,
    owner: str,
) -> np.ndarray:
    if field.namespace is None:
        raise TypeError(f"{owner}: unresolved metadata namespace for field {field.name!r}.")
    if state.backend == "dataset":
        ds = _require_dataset_payload(state.data)
        return _dataset_field_values(ds, batch_dim=state.batch_dim, field=field, options=options, owner=owner)
    tree = _require_datatree_payload(state.data)
    return _datatree_field_values(
        tree,
        batch_dim=state.batch_dim,
        field=field,
        options=options,
        owner=owner,
    )


def _dataset_attr_names(ds: xr.Dataset) -> set[str]:
    return {name for name in ds.attrs if isinstance(name, str) and name != "tal"}


def _dataset_scalar_coord_names(ds: xr.Dataset) -> set[str]:
    return {name for name, coord in ds.coords.items() if coord.dims == ()}


def _dataset_batch_coord_names(ds: xr.Dataset, *, batch_dim: str) -> set[str]:
    return {name for name, coord in ds.coords.items() if coord.dims == (batch_dim,)}


def _datatree_attr_names(tree: xr.DataTree) -> set[str]:
    names = _dataset_attr_names(tree.ds)
    for child in datatree_children(tree).values():
        names.update(_dataset_attr_names(child.ds))
    return names


def _datatree_scalar_coord_names(tree: xr.DataTree) -> set[str]:
    names = _dataset_scalar_coord_names(tree.ds)
    for child in datatree_children(tree).values():
        names.update(_dataset_scalar_coord_names(child.ds))
    return names


def _datatree_batch_names(tree: xr.DataTree, *, batch_dim: str, owner: str) -> set[str]:
    _ = owner
    names = {batch_dim}
    names.update(_dataset_batch_coord_names(tree.ds, batch_dim=batch_dim))
    return names


def _dataset_field_values(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    field: FieldRef,
    options: CatalogQueryOptions,
    owner: str,
) -> np.ndarray:
    size = int(ds.sizes[batch_dim])
    if field.namespace == "attr":
        return _dataset_attr_values(ds, field=field.name, size=size, owner=owner)
    if field.namespace == "coord":
        coord = _require_coord(ds, name=field.name, dims=(), owner=owner)
        scalar = _materialize_scalar_coord(coord, options=options, owner=owner, field=f"coord.{field.name}")
        return np.full((size,), scalar, dtype=object)
    coord = _require_coord(ds, name=field.name, dims=(batch_dim,), owner=owner)
    return _materialize_vector_coord(coord, options=options, owner=owner, field=f"batch.{field.name}")


def _dataset_attr_values(ds: xr.Dataset, *, field: str, size: int, owner: str) -> np.ndarray:
    if field not in ds.attrs:
        raise ValueError(f"{owner}: unknown metadata field attr.{field!r}.")
    value = ds.attrs[field]
    if _is_scalar_like(value):
        return np.full((size,), _python_scalar(value), dtype=object)
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.shape[0] != size:
            raise ValueError(
                f"{owner}: attr.{field!r} must be scalar or 1-D with length {size}; got shape {value.shape!r}."
            )
        return np.asarray(value, dtype=object)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != size:
            raise ValueError(
                f"{owner}: attr.{field!r} must be scalar or length {size}; got length {len(value)}."
            )
        return np.asarray(tuple(_python_scalar(item) for item in value), dtype=object)
    raise ValueError(
        f"{owner}: attr.{field!r} is not queryable; expected scalar or length-{size} sequence metadata."
    )


def _datatree_field_values(
    tree: xr.DataTree,
    *,
    batch_dim: str,
    field: FieldRef,
    options: CatalogQueryOptions,
    owner: str,
) -> np.ndarray:
    labels = tuple(datatree_children(tree).keys())
    if field.namespace == "batch":
        return _datatree_batch_values(tree, labels=labels, batch_dim=batch_dim, name=field.name, options=options, owner=owner)
    if field.namespace == "attr":
        return _datatree_attrs_values(tree, labels=labels, name=field.name)
    return _datatree_scalar_coord_values(tree, labels=labels, name=field.name, options=options, owner=owner)


def _datatree_batch_values(
    tree: xr.DataTree,
    *,
    labels: tuple[str, ...],
    batch_dim: str,
    name: str,
    options: CatalogQueryOptions,
    owner: str,
) -> np.ndarray:
    if name == batch_dim:
        return np.asarray(labels, dtype=object)
    coord = _require_coord(tree.ds, name=name, dims=(batch_dim,), owner=owner)
    values = _materialize_vector_coord(coord, options=options, owner=owner, field=f"batch.{name}")
    if values.shape[0] != len(labels):
        raise ValueError(
            f"{owner}: batch.{name!r} length {values.shape[0]} does not match group count {len(labels)}."
        )
    return values


def _datatree_attrs_values(tree: xr.DataTree, *, labels: tuple[str, ...], name: str) -> np.ndarray:
    root = tree.ds.attrs.get(name, _MISSING)
    out: list[object] = []
    for label in labels:
        child = datatree_children(tree)[label]
        if name in child.ds.attrs:
            out.append(_python_scalar(child.ds.attrs[name]))
        elif root is not _MISSING:
            out.append(_python_scalar(root))
        else:
            out.append(None)
    return np.asarray(out, dtype=object)


def _datatree_scalar_coord_values(
    tree: xr.DataTree,
    *,
    labels: tuple[str, ...],
    name: str,
    options: CatalogQueryOptions,
    owner: str,
) -> np.ndarray:
    root = tree.ds.coords[name] if name in tree.ds.coords and tree.ds.coords[name].dims == () else None
    root_value = (
        _materialize_scalar_coord(root, options=options, owner=owner, field=f"coord.{name}")
        if root is not None
        else _MISSING
    )
    out: list[object] = []
    for label in labels:
        child = datatree_children(tree)[label]
        coord = child.ds.coords[name] if name in child.ds.coords and child.ds.coords[name].dims == () else None
        if coord is not None:
            out.append(_materialize_scalar_coord(coord, options=options, owner=owner, field=f"coord.{name}"))
        elif root_value is not _MISSING:
            out.append(root_value)
        else:
            out.append(None)
    return np.asarray(out, dtype=object)


def _require_coord(ds: xr.Dataset, *, name: str, dims: tuple[str, ...], owner: str) -> xr.DataArray:
    if name not in ds.coords:
        raise ValueError(f"{owner}: unknown metadata coordinate {name!r}.")
    coord = ds.coords[name]
    if coord.dims != dims:
        raise ValueError(
            f"{owner}: coordinate {name!r} must have dims {dims!r} for metadata query; got {coord.dims!r}."
        )
    return coord


def _materialize_vector_coord(
    coord: xr.DataArray,
    *,
    options: CatalogQueryOptions,
    owner: str,
    field: str,
) -> np.ndarray:
    arr = _materialize_coord_data(coord, options=options, owner=owner, field=field)
    if arr.ndim != 1:
        raise ValueError(f"{owner}: metadata field {field} must be 1-D; got {arr.ndim}-D.")
    return np.asarray(arr, dtype=object)


def _materialize_scalar_coord(
    coord: xr.DataArray,
    *,
    options: CatalogQueryOptions,
    owner: str,
    field: str,
) -> object:
    arr = _materialize_coord_data(coord, options=options, owner=owner, field=field)
    if arr.ndim != 0:
        raise ValueError(f"{owner}: metadata field {field} must be scalar; got shape {arr.shape!r}.")
    return _python_scalar(arr.item())


def _materialize_coord_data(
    coord: xr.DataArray,
    *,
    options: CatalogQueryOptions,
    owner: str,
    field: str,
) -> np.ndarray:
    if is_chunked_dataarray(coord):
        if options.metadata_eager_policy != "allow":
            raise ValueError(
                f"{owner}: metadata field {field!r} is chunked; set CatalogQueryOptions(metadata_eager_policy='allow') to realize it explicitly."
            )
        coord = coord.compute()
    return coord.to_numpy()


def _is_scalar_like(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return not isinstance(value, (Sequence, Mapping)) or isinstance(value, (str, bytes))


def _python_scalar(value: object) -> object:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _add_named_metadata(
    out: dict[str, np.ndarray],
    *,
    name: str,
    values: np.ndarray,
    owner: str,
) -> None:
    if name in out:
        raise ValueError(
            f"{owner}: metadata promotion field name collision for {name!r}; qualify promotion explicitly."
        )
    out[name] = values


def _is_scalar_column(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    first = values[0]
    for value in values[1:]:
        if _equal_or_nan(first, value):
            continue
        return False
    return True


def _column_scalar(values: np.ndarray) -> object:
    if values.size == 0:
        return None
    return _python_scalar(values[0])


def _equal_or_nan(left: object, right: object) -> bool:
    try:
        if bool(np.isnan(left)) and bool(np.isnan(right)):  # type: ignore[arg-type]
            return True
    except Exception:
        pass
    return left == right


def _promote_scalar_column(
    ds: xr.Dataset,
    *,
    batch_dim: str,
    name: str,
    scalar: object,
    target: str,
    owner: str,
) -> xr.Dataset:
    if target == "none":
        return ds
    if target == "attr":
        _require_promotion_name_available(ds, name=name, target="attrs", owner=owner)
        attrs = dict(ds.attrs)
        attrs[name] = _python_scalar(scalar)
        out = ds.copy(deep=False)
        out.attrs = attrs
        return out
    _require_promotion_name_available(ds, name=name, target="batch_coord", owner=owner)
    values = np.full((ds.sizes[batch_dim],), _python_scalar(scalar), dtype=object)
    return ds.assign_coords({name: xr.DataArray(values, dims=(batch_dim,))})


def _require_promotion_name_available(ds: xr.Dataset, *, name: str, target: str, owner: str) -> None:
    if target == "attrs" and name == "tal":
        raise ValueError(f"{owner}: metadata name 'tal' is reserved and cannot be promoted to attrs.")
    if name in ds.dims or name in ds.data_vars or name in ds.coords or name in ds.attrs:
        raise ValueError(f"{owner}: metadata promotion target name {name!r} collides with existing dataset names.")


def _require_dataset_payload(payload: xr.Dataset | xr.DataTree) -> xr.Dataset:
    if isinstance(payload, xr.Dataset):
        return payload
    raise TypeError(f"catalog metadata internal error: expected xr.Dataset; got {type(payload).__name__}.")


def _require_datatree_payload(payload: xr.Dataset | xr.DataTree) -> xr.DataTree:
    if isinstance(payload, xr.DataTree):
        return payload
    raise TypeError(f"catalog metadata internal error: expected xr.DataTree; got {type(payload).__name__}.")


__all__ = [
    "MetadataProjection",
    "collect_extract_metadata_columns",
    "discover_metadata_fields",
    "project_metadata",
    "promote_extract_metadata",
]
