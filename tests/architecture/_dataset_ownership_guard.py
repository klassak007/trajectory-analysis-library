"""Runtime probes for the Dataset-ownership architecture boundary."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr
from tal.core import dataset_ownership as dataset_owner


OwnerResultKind = Literal["dataset", "dataarray"]
OwnerCopyMode = Literal["deep", "shallow", "none"]
_OWNER_LINEAGE_ATTR = "_tal_architecture_owner_lineage"


class _OwnerLineageMarker:
    def __deepcopy__(self, memo: dict[int, object]) -> _OwnerLineageMarker:
        del memo
        return self


@dataclass(frozen=True)
class OwnerResult:
    operation: str
    result_kind: OwnerResultKind
    copy_mode: OwnerCopyMode
    value: xr.Dataset | xr.DataArray
    marker: _OwnerLineageMarker | None


def _is_frozen_dataclass(value: object) -> bool:
    params = getattr(type(value), "__dataclass_params__", None)
    return bool(is_dataclass(value) and params is not None and params.frozen)


def _nested_xarray_values(
    value: object,
    *,
    seen: set[int],
) -> tuple[xr.Dataset | xr.DataArray, ...]:
    if isinstance(value, (xr.Dataset, xr.DataArray)):
        return (value,)
    if id(value) in seen:
        return ()
    seen.add(id(value))
    if isinstance(value, (tuple, list)):
        return tuple(
            item
            for nested in value
            for item in _nested_xarray_values(nested, seen=seen)
        )
    if isinstance(value, Mapping):
        return tuple(
            item
            for pair in value.items()
            for nested in pair
            for item in _nested_xarray_values(nested, seen=seen)
        )
    if _is_frozen_dataclass(value):
        return tuple(
            item
            for field in fields(value)
            for item in _nested_xarray_values(getattr(value, field.name), seen=seen)
        )
    return ()


def _unique(
    values: tuple[xr.Dataset | xr.DataArray, ...],
) -> tuple[xr.Dataset | xr.DataArray, ...]:
    out: list[xr.Dataset | xr.DataArray] = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            out.append(value)
    return tuple(out)


def _frame_xarray_arguments(frame: Any) -> tuple[xr.Dataset | xr.DataArray, ...]:
    seen: set[int] = set()
    values = tuple(
        item
        for value in frame.f_locals.values()
        for item in _nested_xarray_values(value, seen=seen)
    )
    return _unique(values)


def _array_has_lineage(value: object, origin: object) -> bool:
    if value is origin:
        return True
    if isinstance(value, np.ndarray) and isinstance(origin, np.ndarray):
        return bool(np.shares_memory(value, origin))
    return False


def _payload_pairs(
    value: xr.Dataset | xr.DataArray,
    source: xr.Dataset | xr.DataArray,
) -> tuple[tuple[object, object], ...] | None:
    if isinstance(value, xr.Dataset) and isinstance(source, xr.Dataset):
        if value.variables.keys() != source.variables.keys():
            return None
        return tuple(
            (value.variables[name].data, source.variables[name].data)
            for name in value.variables
        )
    if isinstance(value, xr.DataArray) and isinstance(source, xr.DataArray):
        if value.coords.keys() != source.coords.keys():
            return None
        pairs = [(value.data, source.data)]
        pairs.extend(
            (value.coords[name].data, source.coords[name].data)
            for name in value.coords
        )
        return tuple(pairs)
    return _cross_container_payload_pairs(value, source)


def _cross_container_payload_pairs(
    value: xr.Dataset | xr.DataArray,
    source: xr.Dataset | xr.DataArray,
) -> tuple[tuple[object, object], ...] | None:
    dataset = value if isinstance(value, xr.Dataset) else source
    array = source if isinstance(source, xr.DataArray) else value
    if not isinstance(dataset, xr.Dataset) or not isinstance(array, xr.DataArray):
        return None
    name = array.name
    if name not in dataset.data_vars:
        if len(dataset.data_vars) != 1:
            return None
        name = next(iter(dataset.data_vars))
    pairs = [(dataset[name].data, array.data)]
    for coord_name, coord in array.coords.items():
        if coord_name not in dataset.coords:
            return None
        pairs.append((dataset.coords[coord_name].data, coord.data))
    return tuple(pairs)


def _shares_metadata(
    value: xr.Dataset | xr.DataArray,
    source: xr.Dataset | xr.DataArray,
) -> bool:
    if isinstance(value, xr.DataArray) and isinstance(source, xr.Dataset):
        name = value.name
        if name not in source.data_vars and len(source.data_vars) == 1:
            name = next(iter(source.data_vars))
        return name in source.data_vars and value.attrs is source[name].attrs
    if isinstance(value, xr.Dataset) and isinstance(source, xr.DataArray):
        name = source.name
        if name not in value.data_vars and len(value.data_vars) == 1:
            name = next(iter(value.data_vars))
        return name in value.data_vars and value[name].attrs is source.attrs
    return value.attrs is source.attrs


def _observed_copy_mode(
    value: xr.Dataset | xr.DataArray,
    sources: tuple[xr.Dataset | xr.DataArray, ...],
) -> OwnerCopyMode:
    if any(value is source for source in sources):
        return "none"
    matched = False
    for source in sources:
        pairs = _payload_pairs(value, source)
        if not pairs:
            continue
        matched = True
        shared = tuple(_array_has_lineage(item, origin) for item, origin in pairs)
        if all(shared):
            return "none" if _shares_metadata(value, source) else "shallow"
        if any(shared):
            raise AssertionError("ownership operation returned partial payload lineage")
    if matched:
        return "deep"
    raise AssertionError("ownership operation returned an unrelated xarray container")


def _owner_code_names() -> dict[object, str]:
    out: dict[object, str] = {}
    for name in dataset_owner.__all__:
        operation = getattr(dataset_owner, name, None)
        if inspect.isfunction(operation):
            out.setdefault(operation.__code__, name)
    return out


def observe_owner_results(
    invoke: Callable[[], Any],
) -> tuple[Any, tuple[OwnerResult, ...]]:
    """Invoke one public root and record xarray results from exported owners."""
    by_code = _owner_code_names()
    active: dict[int, tuple[str, tuple[xr.Dataset | xr.DataArray, ...]]] = {}
    observed: list[OwnerResult] = []
    previous = sys.getprofile()

    def profile(frame: Any, event: str, arg: object) -> None:
        if previous is not None:
            previous(frame, event, arg)
        name = by_code.get(frame.f_code)
        if name is None:
            return
        if event == "call":
            active[id(frame)] = (name, _frame_xarray_arguments(frame))
            return
        if event != "return" or not isinstance(arg, (xr.Dataset, xr.DataArray)):
            return
        operation, sources = active.pop(id(frame), (name, ()))
        if not sources:
            return
        copy_mode = _observed_copy_mode(arg, sources)
        marker = None if copy_mode == "none" else _OwnerLineageMarker()
        if marker is not None:
            arg.attrs[_OWNER_LINEAGE_ATTR] = marker
        kind: OwnerResultKind = (
            "dataset" if isinstance(arg, xr.Dataset) else "dataarray"
        )
        observed.append(OwnerResult(operation, kind, copy_mode, arg, marker))

    sys.setprofile(profile)
    try:
        result = invoke()
    finally:
        sys.setprofile(previous)
    return result, tuple(observed)


def _container_payloads(
    value: xr.Dataset | xr.DataArray,
) -> dict[object, object]:
    if isinstance(value, xr.Dataset):
        return {name: variable.data for name, variable in value.variables.items()}
    payloads: dict[object, object] = {None: value.variable.data}
    payloads.update({name: coord.data for name, coord in value.coords.items()})
    return payloads


def _container_has_lineage(value: object, result: OwnerResult) -> bool:
    origin = result.value
    if value is origin:
        return True
    if not isinstance(value, type(origin)):
        return False
    if result.marker is not None:
        if value.attrs.get(_OWNER_LINEAGE_ATTR) is not result.marker:
            return False
    target_payloads = _container_payloads(value)
    origin_payloads = _container_payloads(origin)
    if not origin_payloads or target_payloads.keys() != origin_payloads.keys():
        return False
    return all(
        _array_has_lineage(target_payloads[name], payload)
        for name, payload in origin_payloads.items()
    )


def has_owner_lineage(
    value: object,
    observed: tuple[OwnerResult, ...],
    *,
    result_kind: OwnerResultKind,
    copy_mode: OwnerCopyMode,
) -> bool:
    """Return whether *value* retains payload lineage from the owner result."""
    return any(
        item.result_kind == result_kind
        and item.copy_mode == copy_mode
        and _container_has_lineage(value, item)
        for item in observed
    )
