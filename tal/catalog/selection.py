from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from tal.core.group_ops.label_keys import canonical_group_label_key

from .backends import (
    datatree_children,
    datatree_extract_template,
    validate_datatree_root_batch_alignment,
)
from .types import CatalogBackend, CatalogState, make_catalog_state


def catalog_group_labels(state: CatalogState) -> tuple[object, ...]:
    if state.backend == "dataset":
        ds = state.data if isinstance(state.data, xr.Dataset) else _invalid_dataset_payload(state.data)
        labels = ds.coords[state.batch_dim].to_numpy()
        return tuple(labels.tolist())
    tree = state.data if isinstance(state.data, xr.DataTree) else _invalid_datatree_payload(state.data)
    return tuple(datatree_children(tree).keys())


def normalize_label_selector(selector: object, *, batch_dim: str, owner: str) -> tuple[object, ...]:
    raw = _unwrap_mapping_selector(selector, batch_dim=batch_dim, owner=owner)
    labels = (raw,) if _is_scalar_selector(raw) else tuple(raw)
    _require_hashable_labels(labels, owner=owner)
    _require_unique_labels(labels, owner=owner)
    return labels


def normalize_position_selector(selector: object, *, batch_dim: str, owner: str) -> slice | tuple[int, ...]:
    raw = _unwrap_mapping_selector(selector, batch_dim=batch_dim, owner=owner)
    if isinstance(raw, slice):
        return raw
    if _is_scalar_selector(raw):
        return (_coerce_int_index(raw, owner=owner),)
    return tuple(_coerce_int_index(item, owner=owner) for item in raw)


def select_by_labels(state: CatalogState, labels: tuple[object, ...], *, owner: str) -> CatalogState:
    if state.backend == "dataset":
        ds = state.data if isinstance(state.data, xr.Dataset) else _invalid_dataset_payload(state.data)
        selected = _dataset_select_labels(ds, labels=labels, batch_dim=state.batch_dim, owner=owner)
        return make_catalog_state(
            backend="dataset",
            batch_dim=state.batch_dim,
            data=selected,
            template=state.template,
        )
    tree = state.data if isinstance(state.data, xr.DataTree) else _invalid_datatree_payload(state.data)
    selected_tree = _datatree_select_labels(
        tree,
        labels=labels,
        batch_dim=state.batch_dim,
        owner=owner,
    )
    return make_catalog_state(
        backend="datatree",
        batch_dim=state.batch_dim,
        data=selected_tree,
        template=_selected_datatree_template(
            selected_tree,
            batch_dim=state.batch_dim,
            fallback=state.template,
            owner=owner,
        ),
    )


def select_by_positions(
    state: CatalogState,
    position_selector: slice | tuple[int, ...],
    *,
    owner: str,
) -> CatalogState:
    normalized = _normalize_effective_position_selector(
        position_selector,
        size=_group_axis_size(state),
        owner=owner,
    )
    if state.backend == "dataset":
        ds = state.data if isinstance(state.data, xr.Dataset) else _invalid_dataset_payload(state.data)
        selected = _dataset_select_positions(
            ds,
            selector=normalized,
            batch_dim=state.batch_dim,
            owner=owner,
        )
        return make_catalog_state(
            backend="dataset",
            batch_dim=state.batch_dim,
            data=selected,
            template=state.template,
        )
    tree = state.data if isinstance(state.data, xr.DataTree) else _invalid_datatree_payload(state.data)
    labels, positions = _datatree_position_plan(tree, selector=normalized)
    selected_tree = _datatree_select_labels(
        tree,
        labels=labels,
        batch_dim=state.batch_dim,
        owner=owner,
        positions=positions,
    )
    return make_catalog_state(
        backend="datatree",
        batch_dim=state.batch_dim,
        data=selected_tree,
        template=_selected_datatree_template(
            selected_tree,
            batch_dim=state.batch_dim,
            fallback=state.template,
            owner=owner,
        ),
    )


def coerce_head_tail_n(n: object, *, owner: str) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"{owner}: n must be a non-negative integer; got {type(n).__name__}.")
    if int(n) < 0:
        raise ValueError(f"{owner}: n must be a non-negative integer; got {n!r}.")
    return int(n)


def _unwrap_mapping_selector(selector: object, *, batch_dim: str, owner: str) -> object:
    if not isinstance(selector, Mapping):
        return selector
    if len(selector) != 1:
        raise ValueError(f"{owner}: selector mapping must contain exactly one key {batch_dim!r}.")
    key = next(iter(selector))
    if key != batch_dim:
        raise ValueError(f"{owner}: selector key must be {batch_dim!r}; got {key!r}.")
    return selector[key]


def _is_scalar_selector(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return not isinstance(value, Sequence)


def _require_hashable_labels(labels: tuple[object, ...], *, owner: str) -> None:
    for label in labels:
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError(f"{owner}: label selectors must be hashable; got {label!r}.") from exc


def _require_unique_labels(labels: tuple[object, ...], *, owner: str) -> None:
    seen: set[object] = set()
    for label in labels:
        key = canonical_group_label_key(label)
        if key in seen:
            raise ValueError(f"{owner}: duplicate labels are not supported in selector {labels!r}.")
        seen.add(key)


def _coerce_int_index(value: object, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{owner}: positional selectors must be integers; got {value!r}.")
    return int(value)


def _require_unique_indices(values: tuple[int, ...], *, owner: str) -> None:
    if len(values) == len(set(values)):
        return
    raise ValueError(f"{owner}: duplicate positional selectors are not supported; got {values!r}.")


def _group_axis_size(state: CatalogState) -> int:
    if state.backend == "dataset":
        ds = state.data if isinstance(state.data, xr.Dataset) else _invalid_dataset_payload(state.data)
        return int(ds.sizes[state.batch_dim])
    tree = state.data if isinstance(state.data, xr.DataTree) else _invalid_datatree_payload(state.data)
    return len(datatree_children(tree))


def _normalize_effective_position_selector(
    selector: slice | tuple[int, ...],
    *,
    size: int,
    owner: str,
) -> slice | tuple[int, ...]:
    if isinstance(selector, slice):
        return selector
    normalized: list[int] = []
    for idx in selector:
        norm = idx + size if idx < 0 else idx
        if norm < 0 or norm >= size:
            raise IndexError(f"{owner}: positional selector {idx} is out of range for {size} groups.")
        normalized.append(norm)
    values = tuple(normalized)
    _require_unique_indices(values, owner=owner)
    return values


def _dataset_select_labels(ds: xr.Dataset, *, labels: tuple[object, ...], batch_dim: str, owner: str) -> xr.Dataset:
    if not labels:
        return ds.isel({batch_dim: slice(0, 0)})
    try:
        return ds.sel({batch_dim: list(labels)})
    except KeyError as exc:
        raise KeyError(f"{owner}: unknown group label in selector {labels!r}.") from exc


def _dataset_select_positions(
    ds: xr.Dataset,
    *,
    selector: slice | tuple[int, ...],
    batch_dim: str,
    owner: str,
) -> xr.Dataset:
    try:
        return ds.isel({batch_dim: selector if isinstance(selector, slice) else list(selector)})
    except IndexError as exc:
        raise IndexError(f"{owner}: positional selector {selector!r} is out of range.") from exc


def _datatree_position_plan(
    tree: xr.DataTree,
    *,
    selector: slice | tuple[int, ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    children = datatree_children(tree)
    positions = tuple(range(len(children))[selector]) if isinstance(selector, slice) else selector
    return _datatree_labels_at_positions(children, positions=positions), positions


def _datatree_select_labels(
    tree: xr.DataTree,
    *,
    labels: tuple[object, ...],
    batch_dim: str,
    owner: str,
    positions: tuple[int, ...] | None = None,
) -> xr.DataTree:
    children = datatree_children(tree)
    selected_labels = _require_datatree_labels(children, labels=labels, owner=owner)
    validate_datatree_root_batch_alignment(tree, batch_dim=batch_dim, owner=owner)
    root = tree.to_dataset(inherit=False)
    if batch_dim in root.dims and positions is None:
        positions = _datatree_label_positions(children, labels=selected_labels)
    root = _project_datatree_root_dataset(
        root,
        positions=positions or (),
        batch_dim=batch_dim,
    )
    selected: dict[str, xr.DataTree] = {
        label: xr.DataTree(dataset=children[label].to_dataset(inherit=False).copy(deep=True))
        for label in selected_labels
    }
    return xr.DataTree(dataset=root, children=selected, name=tree.name)


def _require_datatree_labels(
    children: Mapping[str, xr.DataTree],
    *,
    labels: tuple[object, ...],
    owner: str,
) -> tuple[str, ...]:
    selected: list[str] = []
    for label in labels:
        if not isinstance(label, str) or label not in children:
            raise KeyError(f"{owner}: unknown group label {label!r}.")
        selected.append(label)
    return tuple(selected)


def _datatree_label_positions(
    children: Mapping[str, xr.DataTree],
    *,
    labels: tuple[str, ...],
) -> tuple[int, ...]:
    if not labels:
        return ()
    wanted = set(labels)
    positions: dict[str, int] = {}
    for index, label in enumerate(children):
        if label in wanted:
            positions[label] = index
        if len(positions) == len(wanted):
            break
    return tuple(positions[label] for label in labels)


def _datatree_labels_at_positions(
    children: Mapping[str, xr.DataTree],
    *,
    positions: tuple[int, ...],
) -> tuple[str, ...]:
    if not positions:
        return ()
    wanted = set(positions)
    labels: dict[int, str] = {}
    for index, label in enumerate(children):
        if index in wanted:
            labels[index] = label
        if len(labels) == len(wanted):
            break
    return tuple(labels[index] for index in positions)


def _project_datatree_root_dataset(
    root: xr.Dataset,
    *,
    positions: tuple[int, ...],
    batch_dim: str,
) -> xr.Dataset:
    if batch_dim not in root.dims:
        return root.copy(deep=True)
    return root.isel({batch_dim: list(positions)}).copy(deep=True)


def _selected_datatree_template(
    tree: xr.DataTree,
    *,
    batch_dim: str,
    fallback: xr.Dataset | None,
    owner: str,
) -> xr.Dataset | None:
    selected = datatree_extract_template(tree, batch_dim=batch_dim, owner=owner)
    return fallback if selected is None else selected


def _invalid_dataset_payload(payload: xr.Dataset | xr.DataTree) -> xr.Dataset:
    raise TypeError(f"catalog selection internal error: expected xr.Dataset; got {type(payload).__name__}.")


def _invalid_datatree_payload(payload: xr.Dataset | xr.DataTree) -> xr.DataTree:
    raise TypeError(f"catalog selection internal error: expected xr.DataTree; got {type(payload).__name__}.")


__all__ = [
    "catalog_group_labels",
    "coerce_head_tail_n",
    "normalize_label_selector",
    "normalize_position_selector",
    "select_by_labels",
    "select_by_positions",
]
