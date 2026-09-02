from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, MutableMapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

import xarray as xr

from .dataset_utils import ensure_dataset, require_single_data_var
from .orchestration.finalize import transfer_dataset_attrs

if TYPE_CHECKING:
    from .analysis_object import AnalysisObject


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


def _capture_close_failure(
    callback: Callable[[], None],
    failure: BaseException | None,
) -> BaseException | None:
    try:
        callback()
    except BaseException as exc:  # noqa: BLE001 -- retain first failure
        return exc if failure is None else failure
    return failure


class _CloseOnce:
    def __init__(
        self,
        primary: Callable[[], None],
        cleanup: _CloseOnce | None = None,
    ) -> None:
        self._primary = primary
        self._cleanup = cleanup
        self._closed = False

    def __call__(self) -> None:
        pending: list[Callable[[], None] | _CloseOnce] = [self]
        failure: BaseException | None = None
        while pending:
            callback = pending.pop()
            if not isinstance(callback, _CloseOnce):
                failure = _capture_close_failure(callback, failure)
                continue
            if callback._closed:
                continue
            callback._closed = True
            if callback._cleanup is not None:
                pending.append(callback._cleanup)
            pending.append(callback._primary)
        if failure is not None:
            raise failure


def _idempotent_close(callback: Callable[[], None]) -> _CloseOnce:
    if isinstance(callback, _CloseOnce):
        return callback
    return _CloseOnce(callback)


def _compose_close_callbacks(
    primary: Callable[[], None],
    cleanup: Callable[[], None],
) -> _CloseOnce:
    return _CloseOnce(primary, cleanup=_idempotent_close(cleanup))


def couple_dataset_resource(source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Give a promoted Dataset the source Dataset's coupled close lifetime."""
    source_close = getattr(source, "_close", None)
    if source is target or source_close is None:
        return target
    source_close = _idempotent_close(source_close)
    source.set_close(source_close)
    target_close = getattr(target, "_close", None)
    close = source_close
    if target_close is not None and target_close is not source_close:
        close = _compose_close_callbacks(target_close, source_close)
    target.set_close(close)
    return target


def analysis_object_dataset(source: AnalysisObject) -> xr.Dataset:
    """Return the exact Dataset backing one internal AnalysisObject boundary."""
    return cast(xr.Dataset, source._data)


def dataset_to_dataarray_view(
    source: xr.Dataset,
    *,
    name: str | None,
    copy: DatasetCopyMode,
    owner: str,
) -> xr.DataArray:
    """Convert one owned Dataset view through the canonical single-var helper."""
    mode = coerce_dataset_copy_mode(copy, owner=owner)
    var_name = require_single_data_var(source)
    if mode == "none":
        selected = source[var_name]
    else:
        projection = source[var_name].to_dataset(name=var_name)
        selected = dataset_view(projection, copy=mode, owner=owner)[var_name]
    return selected if name is None else selected.rename(name)


__all__ = [
    "DatasetCopyMode",
    "analysis_object_dataset",
    "coerce_dataset_copy_mode",
    "couple_dataset_resource",
    "dataset_to_dataarray_view",
    "dataset_view",
    "deep_public_dataset",
    "isolate_external_dataset",
    "metadata_isolated_dataset",
]
