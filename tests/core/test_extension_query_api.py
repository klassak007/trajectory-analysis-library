"""Contract 135A: observable extension-author query and ownership semantics."""

import doctest
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisObject
from tal.core.orchestration.resolve import resolve_param_runtime_context
from tal.core.param_engine.map_apply import apply_param_map_with_batch_dims
from tal.core.param_ops import query as query_api
from tal.core.param_ops.query import (
    ParamQueryOptions,
    finalize_param_query,
    prepare_param_query,
)
from tal.core.schema_read import read_roles


def _source(*, lazy=False, batched=False, empty=False):
    count = 0 if empty else 3
    values = np.arange(count, dtype=float) * 10
    dims = ("sample",)
    coords = {"time": ("sample", np.arange(count, dtype=float)), "site": "A"}
    if batched:
        values = np.stack((values, values + 100))
        dims = ("trial", "sample")
        coords["trial"] = ["a", "b"]
    ds = xr.Dataset({"value": (dims, values)}, coords=coords)
    if lazy:
        ds = ds.chunk({"sample": max(1, count)})
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",) if batched else (),
        param_coord="time",
    )


def _evaluate(source, query, *, validate=True, opts=None):
    context = resolve_param_runtime_context(source)
    plan = prepare_param_query(context, query, opts=opts, owner="author.query")
    # An independent domain kernel uses the published map application owner.
    values = apply_param_map_with_batch_dims(
        context.ds["value"],
        param_map=plan.param_map,
        sequence_dim=context.sequence_dim,
        batch_dims=context.batch_dims,
    )
    coords = {
        name: c
        for name, c in context.ds.coords.items()
        if context.sequence_dim not in c.dims
    }
    return plan, finalize_param_query(
        plan, xr.Dataset({"value": values}, coords=coords), validate=validate
    )


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("validate", (False, True))
@pytest.mark.parametrize(
    "query",
    (
        0.5,
        [0.25, 1.75],
        [],
        xr.DataArray(
            [[0.25, 0.75], [1.25, 1.75]],
            dims=("row", "col"),
            coords={"row": ["b", "a"], "col": [9, 8], "note": ("row", [4, 5])},
        ),
        xr.DataArray(np.empty((2, 0)), dims=("row", "col")),
    ),
)
def test_query_shapes_values_topology_and_immutability(lazy, validate, query):
    source = _source(lazy=lazy)
    before = source.as_dataset(copy="deep")
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        plan, result = _evaluate(source, query, validate=validate)
    assert tasks == []
    actual = result.as_dataset()
    grid = isinstance(query, xr.DataArray) and query.ndim > 1
    expected_dims = query.dims if grid else ("sample",)
    assert actual.value.dims == expected_dims
    assert set(actual.data_vars) == {"value"}
    assert plan.trajectory is not grid
    assert read_roles(actual)[1] == (None if grid else "sample")
    computed = actual.compute(scheduler="synchronous")
    expected = np.asarray(query) if grid else np.atleast_1d(query)
    np.testing.assert_allclose(computed.value, expected * 10)
    np.testing.assert_array_equal(computed.valid, np.ones(expected.shape, dtype=bool))
    np.testing.assert_array_equal(computed.time, expected)
    if grid and "row" in query.xindexes:
        for dim in query.dims:
            assert actual.xindexes[dim].equals(query.xindexes[dim])
        xr.testing.assert_equal(
            actual.note.reset_coords(drop=True), query.note.reset_coords(drop=True)
        )
    if lazy:
        assert actual.value.chunks is not None
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("empty", (False, True))
def test_batched_query_reindexes_by_label_and_retains_validity(lazy, empty):
    source = _source(lazy=lazy, batched=True, empty=empty)
    before = source.as_dataset(copy="deep")
    query = xr.DataArray(
        [[1.5], [0.5]],
        dims=("trial", "when"),
        coords={"trial": ["b", "a"], "label": ("when", ["mid"])},
    )
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        _, result = _evaluate(
            source, query, opts=ParamQueryOptions(output_intent="trajectory")
        )
    assert not tasks
    actual = result.as_dataset().compute(scheduler="synchronous")
    assert actual.value.dims == ("trial", "sample")
    np.testing.assert_array_equal(actual.trial, ["a", "b"])
    np.testing.assert_array_equal(actual.time, [[0.5], [1.5]])
    np.testing.assert_array_equal(actual.valid, np.full((2, 1), not empty))
    np.testing.assert_allclose(
        actual.value, [[np.nan], [np.nan]] if empty else [[5.0], [115.0]]
    )
    np.testing.assert_array_equal(actual.label, ["mid"])
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("shape", ((1, 2), (2, 0)))
@pytest.mark.parametrize("name", ("value", "axis", "site"))
def test_protected_names_fail_before_payload_work(lazy, shape, name):
    source = _source(lazy=lazy)
    ds = source.as_dataset().expand_dims({"axis": ["x", "y"]})
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("axis",), param_coord="time"
    )
    before = source.as_dataset(copy="deep")
    query = xr.DataArray(
        np.full(shape, 0.5),
        dims=("row", "col"),
        coords={name: (("row", "col"), np.full(shape, 99.0))},
    )
    tasks = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(ValueError, match="^author.query:"),
    ):
        prepare_param_query(
            resolve_param_runtime_context(source), query, owner="author.query"
        )
    assert not tasks
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("shape", ((1, 2), (2, 0)))
@pytest.mark.parametrize("validate", (False, True))
def test_generated_shared_and_caller_coordinates(lazy, shape, validate):
    source = _source(lazy=lazy)
    query = xr.DataArray(
        np.full(shape, 0.5),
        dims=("row", "col"),
        coords={
            "site": "A",
            "note": ("row", np.arange(shape[0])),
            "time": (("row", "col"), np.full(shape, 99.0)),
            "valid": (("row", "col"), np.zeros(shape, dtype=bool)),
        },
    )
    query_before = query.copy(deep=True)
    _, result = _evaluate(source, query, validate=validate)
    actual = result.as_dataset().compute(scheduler="synchronous")
    assert set(actual.data_vars) == {"value"}
    assert dict(actual.sizes) == dict(zip(query.dims, shape, strict=True))
    np.testing.assert_array_equal(actual.time, np.full(shape, 0.5))
    np.testing.assert_array_equal(actual.valid, np.ones(shape, dtype=bool))
    xr.testing.assert_equal(actual.site, query.site)
    xr.testing.assert_equal(actual.note, query.note)
    xr.testing.assert_identical(query, query_before)


def test_surviving_auxiliary_index_is_protected():
    ds = _source().as_dataset().expand_dims(sensor=["s"])
    ds = ds.assign_coords(serial=("sensor", ["one"])).set_xindex("serial")
    source = AnalysisObject.from_data(
        ds, sequence_dim="sample", core_dims=("sensor",), param_coord="time"
    )
    query = xr.DataArray([0.5], dims="when", coords={"serial": ("when", ["wrong"])})
    with pytest.raises(ValueError, match="^author.query:"):
        prepare_param_query(
            resolve_param_runtime_context(source), query, owner="author.query"
        )


@pytest.mark.parametrize("shape", ((1,), (0,), (1, 2), (2, 0)))
@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("validate", (False, True))
@pytest.mark.parametrize("kind", ("payload", "core"))
def test_finalization_protects_new_kernel_names(shape, lazy, validate, kind):
    source = _source(lazy=lazy)
    before = source.as_dataset(copy="deep")
    dims = ("row",) if len(shape) == 1 else ("row", "col")
    query = xr.DataArray(
        np.full(shape, 0.5), dims=dims, coords={"note": (dims, np.full(shape, 99.0))}
    )
    plan = prepare_param_query(
        resolve_param_runtime_context(source), query, owner="author.query"
    )
    values = xr.full_like(plan.query, 5.0).reset_coords(drop=True)
    kernel = xr.Dataset({"value": values, "note": values}, coords={"site": "A"})
    if kind == "core":
        kernel = kernel.drop_vars("note").expand_dims({"note": ["x", "y"]})
    if lazy:
        kernel = kernel.chunk({plan.query_dim: 1})
    kernel_before = kernel.copy(deep=True)
    tasks = []
    with (
        Callback(pretask=lambda key, *_: tasks.append(key)),
        pytest.raises(ValueError, match="^author.query:"),
    ):
        finalize_param_query(plan, kernel, validate=validate)
    assert not tasks
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(kernel, kernel_before)


@pytest.mark.parametrize("mapped", (False, True))
def test_mapping_mode_declares_selected_payload_validity(mapped):
    _, result = _evaluate(
        _source(), [0.5], opts=ParamQueryOptions(mapped_dataset=mapped)
    )
    assert ("valid" in result.as_dataset().coords) is mapped


def test_options_plan_freezing_and_invalid_boundaries():
    options = ParamQueryOptions()
    with pytest.raises(FrozenInstanceError):
        options.mapped_dataset = False
    plan, _ = _evaluate(_source(), [0.5])
    with pytest.raises(FrozenInstanceError):
        plan.query_dim = "other"
    context = resolve_param_runtime_context(_source())
    for opts, error in (
        (object(), TypeError),
        (ParamQueryOptions(output_intent="other"), ValueError),
        (ParamQueryOptions(mapped_dataset=1), TypeError),
    ):
        with pytest.raises(error, match="^author.query:"):
            prepare_param_query(context, [0.5], opts=opts, owner="author.query")
    with pytest.raises(TypeError, match="^author.query:"):
        prepare_param_query(None, [0.5], owner="author.query")
    with pytest.raises(TypeError, match="^finalize_param_query:"):
        finalize_param_query(None, xr.Dataset())


def test_author_query_documented_examples_execute():
    result = doctest.testmod(query_api, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0 and result.attempted > 0
