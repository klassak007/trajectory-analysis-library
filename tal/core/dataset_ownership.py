from __future__ import annotations

from collections.abc import Hashable, Mapping, MutableMapping
from copy import deepcopy
from typing import Any, Literal, cast

import xarray as xr

from .dataset_utils import dataset_to_dataarray, ensure_dataset
from .orchestration.finalize import transfer_dataset_attrs


DatasetCopyMode = Literal["deep", "shallow", "none"]

_COPY_MODES: tuple[DatasetCopyMode, ...] = ("deep", "shallow", "none")


def coerce_dataset_copy_mode(value: object, *, owner: str) -> DatasetCopyMode:
    """Resolve one internal Dataset copy policy."""
    if isinstance(value, str) and value in _COPY_MODES:
        return cast(DatasetCopyMode, value)
    raise ValueError(
        f"{owner}: copy must be one of {_COPY_MODES!r}; got {value!r}."
    )


def _is_builtin_index(index: xr.Index) -> bool:
    return type(index).__module__.startswith("xarray.")


def _detached_index(index: xr.Index, *, deep: bool) -> xr.Index:
    if deep or not _is_builtin_index(index):
        return index
    return index.copy(deep=False)


def _rebuilt_index_variables(
    index: xr.Index,
    detached: xr.Index,
    coords: Mapping[Hashable, xr.Variable],
    *,
    deep: bool,
) -> dict[Hashable, xr.Variable]:
    if not deep and isinstance(index, xr.indexes.NDPointIndex):
        return dict(coords)
    rebuilt = detached.create_variables(coords)
    return {
        name: var.copy(deep=deep) if var is coords.get(name) else var
        for name, var in rebuilt.items()
    }


def _copy_index_coordinates(target: xr.Dataset, *, deep: bool) -> xr.Dataset:
    variables: dict[Hashable, xr.Variable] = {}
    indexes: dict[Hashable, xr.Index] = {}
    for index, coords in target.xindexes.group_by_index():
        if isinstance(index, xr.indexes.PandasMultiIndex):
            continue
        detached = _detached_index(index, deep=deep)
        rebuilt = _rebuilt_index_variables(index, detached, coords, deep=deep)
        for name, var in rebuilt.items():
            variables[name] = var
            indexes[name] = detached
    if not variables:
        return target
    return target.assign_coords(xr.Coordinates(variables, indexes=indexes))


def _deepcopy_shared_mapping_values(
    source: Mapping[str, Any],
    target: MutableMapping[str, Any],
    *,
    excluded_key: str | None = None,
) -> None:
    for key, value in source.items():
        if key != excluded_key and (key not in target or target[key] is value):
            target[key] = deepcopy(value)


def _isolate_metadata(source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    source_tal = source.attrs.get("tal")
    if source_tal is not None and target.attrs.get("tal") is source_tal:
        target = transfer_dataset_attrs(source, target, validate=False)
    _deepcopy_shared_mapping_values(source.attrs, target.attrs, excluded_key="tal")
    _deepcopy_shared_mapping_values(source.encoding, target.encoding)
    for name, source_var in source.variables.items():
        target_var = target.variables[name]
        _deepcopy_shared_mapping_values(source_var.attrs, target_var.attrs)
        _deepcopy_shared_mapping_values(source_var.encoding, target_var.encoding)
    return target


def _detached_dataset_copy(source: xr.Dataset, *, deep: bool) -> xr.Dataset:
    target = source.copy(deep=deep)
    target = _copy_index_coordinates(target, deep=deep)
    target = _isolate_metadata(source, target)
    target.set_close(None)
    return target


def dataset_view(
    source: xr.Dataset,
    *,
    copy: DatasetCopyMode,
    owner: str,
) -> xr.Dataset:
    """Return a Dataset under the centralized internal copy policy."""
    mode = coerce_dataset_copy_mode(copy, owner=owner)
    if mode == "none":
        return source
    return _detached_dataset_copy(source, deep=mode == "deep")


def isolate_external_dataset(data: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Deep-isolate caller-owned xarray input without realizing lazy payloads."""
    source = ensure_dataset(data)
    return dataset_view(
        source,
        copy="deep",
        owner="dataset_ownership.isolate_external_dataset",
    )


def deep_public_dataset(source: xr.Dataset, *, owner: str) -> xr.Dataset:
    """Return the current mutation-safe public Dataset snapshot."""
    return dataset_view(source, copy="deep", owner=owner)


def metadata_isolated_dataset(source: xr.Dataset, *, owner: str) -> xr.Dataset:
    """Return a structural/metadata copy that shares payload buffers or graphs."""
    return dataset_view(source, copy="shallow", owner=owner)


def raw_dataset_reference(source: xr.Dataset) -> xr.Dataset:
    """Return the exact backing Dataset for an explicit internal/raw boundary."""
    return source


def dataset_to_dataarray_view(
    source: xr.Dataset,
    *,
    name: str | None,
    copy: DatasetCopyMode,
    owner: str,
) -> xr.DataArray:
    """Convert one owned Dataset view through the canonical single-var helper."""
    selected = dataset_view(source, copy=copy, owner=owner)
    return dataset_to_dataarray(selected, name=name)


__all__ = [
    "DatasetCopyMode",
    "coerce_dataset_copy_mode",
    "dataset_to_dataarray_view",
    "dataset_view",
    "deep_public_dataset",
    "isolate_external_dataset",
    "metadata_isolated_dataset",
    "raw_dataset_reference",
]
