from __future__ import annotations

import gc
import tracemalloc

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import AnalysisObject
from tal.spatial import Rotation


def _source(*, typed: bool, lazy: bool, calls: list[str]) -> AnalysisObject:
    values = np.tile([0., 0., 0., 1.], (2, 2, 1)) if typed else np.array([[1., 3.], [5., 7.]])

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return values

    data = da.from_delayed(payload(), shape=values.shape, dtype=float) if lazy else values
    core = ("q",) if typed else ()
    coords = {"trial": [0, 1], "lane": [0, 1], "key": ("trial", ["a", "a"])}
    if typed:
        coords["q"] = ["x", "y", "z", "w"]
    source = AnalysisObject.from_data(
        xr.Dataset({"value": (("trial", "lane", *core), data)}, coords=coords),
        batch_dims=("trial", "lane"), core_dims=core,
    )
    return Rotation(source) if typed else source


def _weights(first: list[float], second: list[float], *, indexed: bool) -> dict:
    factors = {"trial": np.array(first), "lane": np.array(second)}
    if indexed:
        return {dim: xr.DataArray(value, dims=dim, coords={dim: [0, 1]}) for dim, value in factors.items()}
    return factors


@pytest.mark.parametrize("typed,op", [(False, "mean"), (False, "sum"), (True, "mean")])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("first,second", [([-1., -1.], [-1., -1.]), ([-1., -1.], [0., 0.]), ([0., 0.], [-1., -1.])])
@pytest.mark.parametrize("skipna", [False, True])
def test_mapping_factors_cannot_hide_negative_weights(
    typed: bool, op: str, grouped: bool, indexed: bool,
    first: list[float], second: list[float], skipna: bool,
) -> None:
    """ID: REDUCE_FACTOR_076_negative_factors_fail_before_product."""
    calls: list[str] = []
    source = _source(typed=typed, lazy=True, calls=calls)
    target = source.group.groupby("key") if grouped else source
    weights = _weights(first, second, indexed=indexed)
    with pytest.raises(ValueError, match="negative weights"):
        getattr(target, op)(dim=("trial", "lane"), weights=weights, skipna=skipna)
    assert calls == []


@pytest.mark.parametrize("typed,op,expected", [(False, "mean", 4.), (False, "sum", 16.), (True, "mean", [0., 0., 0., 1.])])
@pytest.mark.parametrize("grouped", [False, True])
def test_nonnegative_mapping_factors_preserve_lazy_payloads(
    typed: bool, op: str, expected: object, grouped: bool,
) -> None:
    """ID: REDUCE_FACTOR_077_positive_factors_preserve_lazy_reduction."""
    calls: list[str] = []
    source = _source(typed=typed, lazy=True, calls=calls)
    target = source.group.groupby("key") if grouped else source
    result = getattr(target, op)(dim=("trial", "lane"), weights=_weights([1., 1.], [1., 1.], indexed=True))
    values = result.as_dataset(copy="none")["value"]
    assert calls == []
    assert isinstance(values.data, da.Array)
    if grouped:
        values = values.isel(group_key=0, drop=True)
    np.testing.assert_allclose(values.compute(), expected)


@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("op", ["mean", "sum"])
def test_mapping_sign_checks_are_scoped_to_structural_validity(skipna: bool, op: str) -> None:
    """ID: REDUCE_FACTOR_078_mapping_sign_validation_excludes_invalid_tails."""
    source = AnalysisObject.from_data(
        xr.Dataset({"value": (("trial", "sample"), [[1., 2., 99.], [3., 99., 99.]])},
                   coords={"trial": [0, 1], "sample": [0, 1, 2], "size": ("trial", [2, 1])}),
        sequence_dim="sample", batch_dims=("trial",), sequence_size_coord="size",
    )
    weights = {"trial": np.ones(2), "sample": np.array([1., 1., -9.])}
    result = getattr(source, op)(dim=("trial", "sample"), weights=weights, skipna=skipna)
    np.testing.assert_allclose(result.as_dataset()["value"], 2. if op == "mean" else 6.)
    weights["sample"] = np.array([1., -9., 1.])
    with pytest.raises(ValueError, match="negative weights"):
        getattr(source, op)(dim=("trial", "sample"), weights=weights, skipna=skipna)


def test_mapping_sign_validation_follows_deferred_key_planning() -> None:
    """ID: REDUCE_FACTOR_079_sign_validation_runs_after_group_key_planning."""
    calls: list[str] = []
    source = _source(typed=False, lazy=True, calls=calls)

    @delayed
    def key_values() -> np.ndarray:
        calls.append("key")
        return np.array(["a", "a"])

    key = xr.DataArray(da.from_delayed(key_values(), shape=(2,), dtype="U1"),
                       dims="trial", coords={"trial": [0, 1]})
    grouped = source.group.groupby(key)
    assert calls == []
    with pytest.raises(ValueError, match="negative weights"):
        grouped.mean(dim=("trial", "lane"), weights=_weights([-1., -1.], [-1., -1.], indexed=False))
    assert calls == ["key"]


@pytest.mark.parametrize("typed,op", [(False, "mean"), (False, "sum"), (True, "mean")])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("empty_dim", ["trial", "lane"])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
def test_mapping_factors_on_empty_domains_have_no_participating_entries(
    typed: bool, op: str, grouped: bool, empty_dim: str, indexed: bool, skipna: bool,
) -> None:
    """ID: REDUCE_FACTOR_080_empty_domains_exclude_nonparticipating_factors."""
    calls: list[str] = []
    source = _source(typed=typed, lazy=True, calls=calls).isel({empty_dim: slice(0, 0)})
    ds = source.as_dataset(copy="none")
    weights = {dim: np.full(ds.sizes[dim], -1.) for dim in ("trial", "lane")}
    if indexed:
        weights = {dim: xr.DataArray(value, dims=dim, coords={dim: ds.coords[dim]})
                   for dim, value in weights.items()}
    target = source.group.groupby("key") if grouped else source
    result = getattr(target, op)(dim=("trial", "lane"), weights=weights, skipna=skipna)
    values = result.as_dataset(copy="none")["value"]
    assert calls == []
    assert isinstance(values.data, da.Array)
    assert isinstance(result, Rotation) == typed
    expected_dims = (("group_key",) if grouped else ()) + (("q",) if typed else ())
    assert values.dims == expected_dims
    if grouped:
        assert values.sizes["group_key"] == (0 if empty_dim == "trial" else 1)
    np.testing.assert_allclose(values.compute(), 0. if op == "sum" else np.nan)


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("invalid", ["missing", "length", "index"])
def test_empty_domain_factors_still_require_valid_structure_and_alignment(
    typed: bool, grouped: bool, invalid: str,
) -> None:
    """ID: REDUCE_FACTOR_081_empty_domains_preserve_weight_validation."""
    calls: list[str] = []
    source = _source(typed=typed, lazy=True, calls=calls).isel(trial=slice(0, 0))
    weights = {"trial": np.empty(0), "lane": np.ones(2)}
    if invalid == "missing":
        del weights["lane"]
    elif invalid == "length":
        weights["lane"] = np.ones(3)
    else:
        weights["lane"] = xr.DataArray([1., 1.], dims="lane", coords={"lane": [1, 0]})
    target = source.group.groupby("key") if grouped else source
    with pytest.raises(ValueError):
        target.mean(dim=("trial", "lane"), weights=weights)
    assert calls == []


def _empty_unreduced_source(
    *,
    typed: bool,
    lazy: bool,
    calls: list[str],
    rows: int = 2,
    lanes: int = 2,
) -> AnalysisObject:
    core_shape = (4,) if typed else ()
    values = np.empty((rows, lanes, 0, *core_shape))

    @delayed
    def payload() -> np.ndarray:
        calls.append("payload")
        return values

    data = da.from_delayed(payload(), shape=values.shape, dtype=float) if lazy else values
    dims = ("trial", "lane", "empty", *(("q",) if typed else ()))
    coords: dict[str, object] = {
        "trial": np.arange(rows),
        "lane": np.arange(lanes),
        "empty": np.empty(0, dtype=np.int64),
        "key": ("trial", np.asarray([f"g{row}" for row in range(rows)])),
    }
    if typed:
        coords["q"] = ["x", "y", "z", "w"]
    source = AnalysisObject.from_data(
        xr.Dataset({"value": (dims, data)}, coords=coords),
        batch_dims=("trial", "lane", "empty"),
        core_dims=("q",) if typed else (),
    )
    return Rotation(source) if typed else source


def _empty_unreduced_weight(
    source: AnalysisObject,
    *,
    kind: str,
    invalid: float,
) -> tuple[object, object]:
    ds = source.as_dataset(copy="none")
    trial = np.ones(ds.sizes["trial"])
    trial[0] = invalid
    if kind == "ndarray":
        return "trial", trial
    if kind == "dataarray":
        return "trial", xr.DataArray(trial, dims="trial", coords={"trial": ds.coords["trial"]})
    return ("trial", "lane"), {"trial": trial, "lane": np.ones(ds.sizes["lane"])}


@pytest.mark.parametrize("typed,op", [(False, "mean"), (False, "sum"), (True, "mean")])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("kind", ["ndarray", "dataarray", "mapping"])
@pytest.mark.parametrize("invalid", [-1.0, np.inf])
def test_empty_unreduced_axis_skips_weight_value_validation(
    typed: bool,
    op: str,
    grouped: bool,
    kind: str,
    invalid: float,
) -> None:
    """ID: REDUCE_FACTOR_082_empty_unreduced_axis_has_no_weight_entries."""
    calls: list[str] = []
    source = _empty_unreduced_source(typed=typed, lazy=True, calls=calls)
    dim, weights = _empty_unreduced_weight(source, kind=kind, invalid=invalid)
    target = source.group.groupby("key") if grouped else source
    result = getattr(target, op)(dim=dim, weights=weights, skipna=False)
    values = result.as_dataset(copy="none")["value"]
    assert calls == []
    assert isinstance(values.data, da.Array)
    assert values.size == 0
    assert isinstance(result, Rotation) == typed


@pytest.mark.parametrize("typed,op", [(False, "mean"), (False, "sum"), (True, "mean")])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("kind", ["dataarray", "mapping"])
def test_empty_unreduced_axis_computation_ignores_lazy_weight_values(
    typed: bool,
    op: str,
    grouped: bool,
    kind: str,
) -> None:
    """ID: REDUCE_FACTOR_083_empty_unreduced_axis_keeps_weights_lazy."""
    calls: list[str] = []
    source = _empty_unreduced_source(typed=typed, lazy=True, calls=calls)

    @delayed
    def weight_values() -> np.ndarray:
        calls.append("weight")
        raise RuntimeError("empty-domain weight values must not execute")

    weight = xr.DataArray(
        da.from_delayed(weight_values(), shape=(2,), dtype=float),
        dims="trial",
        coords={"trial": source.as_dataset(copy="none").coords["trial"]},
    )
    if kind == "mapping":
        dims: object = ("trial", "lane")
        weights: object = {"trial": weight, "lane": np.ones(2)}
    else:
        dims = "trial"
        weights = weight
    target = source.group.groupby("key") if grouped else source
    result = getattr(target, op)(dim=dims, weights=weights, skipna=False)
    values = result.as_dataset(copy="none")["value"]
    assert calls == []
    assert isinstance(values.data, da.Array)
    assert values.size == 0
    assert values.compute().size == 0
    assert "weight" not in calls


@pytest.mark.parametrize("invalid", ["ndarray_length", "dataarray_index", "mapping_missing"])
def test_empty_unreduced_axis_preserves_weight_structure_validation(invalid: str) -> None:
    """ID: REDUCE_FACTOR_084_empty_unreduced_axis_preserves_weight_structure."""
    source = _empty_unreduced_source(typed=False, lazy=False, calls=[])
    if invalid == "ndarray_length":
        dim, weights = "trial", np.ones(3)
    elif invalid == "dataarray_index":
        dim = "trial"
        weights = xr.DataArray([1.0, 1.0], dims="trial", coords={"trial": [1, 0]})
    else:
        dim, weights = ("trial", "lane"), {"trial": np.ones(2)}
    with pytest.raises(ValueError):
        source.mean(dim=dim, weights=weights)


def _empty_weight_reduction_peak(source: AnalysisObject, weights: object) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        result = source.mean(dim=("trial", "lane"), weights=weights, skipna=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.as_dataset(copy="none")["value"].size == 0
    return peak


def test_empty_payload_allocation_does_not_scale_as_factor_product() -> None:
    """ID: REDUCE_ALLOC_057_empty_payload_avoids_cartesian_weight_product."""
    small = _empty_unreduced_source(typed=False, lazy=False, calls=[], rows=32, lanes=2048)
    large = _empty_unreduced_source(typed=False, lazy=False, calls=[], rows=2048, lanes=2048)
    small_weights = {"trial": np.ones(32), "lane": np.ones(2048)}
    large_weights = {"trial": np.ones(2048), "lane": np.ones(2048)}
    small.mean(dim=("trial", "lane"), weights=small_weights)
    small_peak = _empty_weight_reduction_peak(small, small_weights)
    large_peak = _empty_weight_reduction_peak(large, large_weights)
    assert large_peak - small_peak < 4 * 1024**2, (small_peak, large_peak)
