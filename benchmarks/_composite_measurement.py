"""Validation and measurement owners for the composite benchmark."""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from collections.abc import Callable

import dask
import xarray as xr


def _datasets(value: object) -> tuple[xr.Dataset, ...]:
    values = value if isinstance(value, tuple) else (value,)
    return tuple(
        item if isinstance(item, xr.Dataset) else item.as_dataset(copy="none")
        for item in values
    )


def _materialize(value: object) -> tuple[xr.Dataset, ...]:
    return tuple(dask.compute(*_datasets(value), scheduler="synchronous"))


def _index_groups(dataset: xr.Dataset) -> dict[tuple[str, ...], xr.Index]:
    return {
        tuple(coordinates): index
        for index, coordinates in dataset.xindexes.group_by_index()
    }


def _assert_index_parity(observed: xr.Dataset, reference: xr.Dataset) -> None:
    observed_groups = _index_groups(observed)
    reference_groups = _index_groups(reference)
    if observed_groups.keys() != reference_groups.keys():
        raise AssertionError("composite benchmark index groups changed")
    for names, observed_index in observed_groups.items():
        reference_index = reference_groups[names]
        if type(observed_index) is not type(reference_index):
            raise AssertionError(
                f"composite benchmark index type changed for {names!r}"
            )
        if not observed_index.equals(reference_index):
            raise AssertionError(
                f"composite benchmark index values changed for {names!r}"
            )


def _assert_metadata_parity(observed: xr.Dataset, reference: xr.Dataset) -> None:
    if tuple(observed.sizes.items()) != tuple(reference.sizes.items()):
        raise AssertionError("composite benchmark dimension topology changed")
    if tuple(observed.data_vars) != tuple(reference.data_vars):
        raise AssertionError("composite benchmark data variables changed")
    if tuple(observed.coords) != tuple(reference.coords):
        raise AssertionError("composite benchmark coordinates changed")
    if observed.attrs != reference.attrs or observed.encoding != reference.encoding:
        raise AssertionError("composite benchmark Dataset metadata changed")
    for name in observed.variables:
        actual = observed[name]
        expected = reference[name]
        if actual.dims != expected.dims:
            raise AssertionError(
                f"composite benchmark dimensions changed for {name!r}"
            )
        if actual.attrs != expected.attrs or actual.encoding != expected.encoding:
            raise AssertionError(
                f"composite benchmark metadata changed for {name!r}"
            )
    _assert_index_parity(observed, reference)


def _assert_result_parity(observed: xr.Dataset, reference: xr.Dataset) -> None:
    xr.testing.assert_allclose(observed, reference)
    _assert_metadata_parity(observed, reference)


def _validate(value: object, expected: tuple[xr.Dataset, ...]) -> None:
    actual = _materialize(value)
    if len(actual) != len(expected):
        raise AssertionError("composite benchmark result count changed")
    for observed, reference in zip(actual, expected, strict=True):
        _assert_result_parity(observed, reference)


def _identical_results(
    left: tuple[xr.Dataset, ...],
    right: tuple[xr.Dataset, ...],
) -> bool:
    if len(left) != len(right):
        return False
    try:
        for left_ds, right_ds in zip(left, right, strict=True):
            _assert_result_parity(left_ds, right_ds)
    except AssertionError:
        return False
    return True


def _dataset_task_keys(dataset: xr.Dataset) -> set[object]:
    keys: set[object] = set()
    for variable in dataset.variables.values():
        graph = getattr(variable.data, "dask", None)
        if graph is not None:
            keys.update(graph.keys())
    return keys


def _task_count(value: object) -> int:
    keys: set[object] = set()
    for dataset in _datasets(value):
        keys.update(_dataset_task_keys(dataset))
    return len(keys)


def _warm_routes(
    routes: dict[str, Callable[[], object]],
    expected: dict[str, tuple[xr.Dataset, ...]],
    *,
    warmups: int,
) -> None:
    for _ in range(warmups):
        for name, route in routes.items():
            value = route()
            _validate(value, expected[name])
            del value


def _timing_sample(
    route: Callable[[], object],
    expected: tuple[xr.Dataset, ...],
) -> float:
    gc.collect()
    started = time.perf_counter()
    value = route()
    elapsed = time.perf_counter() - started
    _validate(value, expected)
    del value
    return elapsed


def _timing_routes(
    routes: dict[str, Callable[[], object]],
    expected: dict[str, tuple[xr.Dataset, ...]],
    *,
    warmups: int,
    repeats: int,
) -> dict[str, list[float]]:
    _warm_routes(routes, expected, warmups=warmups)
    samples: dict[str, list[float]] = {name: [] for name in routes}
    names = tuple(routes)
    for repetition in range(repeats):
        order = names if repetition % 2 == 0 else tuple(reversed(names))
        for name in order:
            samples[name].append(_timing_sample(routes[name], expected[name]))
    return samples


def _allocation_sample(
    route: Callable[[], object],
    expected: tuple[xr.Dataset, ...],
) -> int:
    gc.collect()
    tracemalloc.start()
    value = route()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _validate(value, expected)
    del value
    return peak


def _measure_routes(
    routes: dict[str, Callable[[], object]],
    expected: dict[str, tuple[xr.Dataset, ...]],
    *,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    samples = _timing_routes(
        routes,
        expected,
        warmups=warmups,
        repeats=repeats,
    )
    report: dict[str, object] = {}
    for name, route in routes.items():
        report[name] = {
            "seconds": samples[name],
            "median_seconds": statistics.median(samples[name]),
            "peak_bytes": _allocation_sample(route, expected[name]),
        }
    return report


__all__: list[str] = []
