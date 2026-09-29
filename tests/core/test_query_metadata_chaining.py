"""Contracts 009/027: operation-owned metadata is consumed on query reuse."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisLayoutSpec
from tal.core.event_ops import AtBoundariesOptions


def _source(*, lazy=False, count_name="count"):
    ds = xr.Dataset({"value": (("trial", "sample"), [[3., 2., 1., 0.], [3., 2., np.nan, np.nan]])},
                    coords={"trial": ["a", "b"], "sample": [0, 1, 2, 3], "time": ("sample", [0., 1., 2., 3.]), count_name: ("trial", [4, 2])})
    if lazy:
        ds["value"] = ds.value.chunk({"trial": 1, "sample": 2})
    return AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",), param_coord="time", sequence_size_coord=count_name).wrap(ds)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("operation", ["at", "sel", "resample_to", "index"])
@pytest.mark.parametrize("shape", [(2,), (1, 2), (2, 0)])
def test_prior_result_query_consumes_metadata(lazy, operation, shape):
    """ID: PARAM_QUERY_CHAIN_001; prior sizes never define current validity."""
    source = _source(lazy=lazy)
    prior = _source(count_name="old_count").param.at([0.5, 1.5]).isel(sample=slice(0, 1))
    metadata = prior.as_dataset().coords["old_count"]
    assert metadata.attrs.get("tal_reserved_owner") == "param_ops"
    dims = tuple(f"q{i}" for i in range(len(shape)))
    query = xr.DataArray(np.full((2, *shape), .5), dims=("trial", *dims), coords={"trial": ["a", "b"]}).assign_coords(old_count=metadata, note="caller")
    if lazy:
        query = query.chunk({dim: max(size, 1) for dim, size in zip(dims, shape, strict=True)})
    before = query.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = getattr(source.param, operation)(query)
    assert not tasks
    ds = result if operation == "index" else result.as_dataset(copy="shallow")
    assert "old_count" not in ds.coords
    assert ds.coords["note"].item() == "caller"
    actual = ds.compute()
    data = actual if operation == "index" else actual["value"]
    np.testing.assert_allclose(data, 0 if operation == "index" else 3 if operation == "sel" else 2.5)
    xr.testing.assert_identical(query, before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("layout", ["segments", "stacked"])
def test_boundary_results_reuse_as_window_anchors(lazy, layout):
    """ID: EVENT_QUERY_CHAIN_001; missing event rows and independent window values."""
    source = _source()
    boundaries = source.events.at_boundaries(source < 1.5, opts=AtBoundariesOptions(edges="enter", mode="first"))
    anchors = boundaries.as_dataset()["time"]
    if lazy:
        anchors = anchors.chunk({"trial": 1})
        source = _source(lazy=True)
    before = anchors.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = source.events.around(anchors, pre=.25, post=.25, dt=.25, layout=layout)
    assert not tasks
    ds = result.as_dataset().compute()
    values = ds.value.sel(trial="a").data.reshape(-1)
    np.testing.assert_allclose(values, [1.25, 1., .75])
    assert np.isnan(ds.value.sel(trial="b")).all()
    xr.testing.assert_identical(anchors, before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("operation", ["at", "sel", "resample_to"])
@pytest.mark.parametrize("collision", ["payload", "core", "source_coord"])
def test_consumed_query_metadata_does_not_claim_output_names(lazy, empty, operation, collision):
    """ID: PARAM_QUERY_CHAIN_002; only inherited query claims disappear."""
    source = _source(lazy=lazy)
    if collision == "payload":
        source = source.rename({"value": "old_count"})
    elif collision == "core":
        ds = source.as_dataset().expand_dims(old_count=["x", "y"])
        ds["count"] = ds["count"].isel(old_count=0, drop=True) if "old_count" in ds["count"].dims else ds["count"]
        source = AnalysisLayoutSpec(sequence_dim="sample", batch_dims=("trial",), core_dims=("old_count",), param_coord="time", sequence_size_coord="count").wrap(ds.drop_attrs())
    else:
        source = type(source)(source.as_dataset().assign_coords(old_count=("trial", [41, 42])))
    metadata = _source(count_name="old_count").param.at([.5]).as_dataset().coords["old_count"]
    query = xr.DataArray(np.full((2, 0 if empty else 1), .5), dims=("trial", "q"), coords={"trial": ["a", "b"], "note": "caller", "old_count": metadata})
    if lazy:
        query = query.chunk({"q": 1})
    before = source.as_dataset()
    query_before = query.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        out = getattr(source.param, operation)(query)
    assert not tasks
    ds = out.as_dataset().compute()
    name = "old_count" if collision == "payload" else "value"
    assert name in ds.data_vars
    np.testing.assert_allclose(ds[name], 3. if operation == "sel" else 2.5)
    assert ds.note.item() == "caller"
    if collision == "core":
        np.testing.assert_array_equal(ds.old_count, ["x", "y"])
    if collision == "source_coord":
        np.testing.assert_array_equal(ds.old_count, [41, 42])
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(query, query_before)
    ordinary = query.copy(deep=True)
    ordinary.coords["old_count"].attrs = {}
    with pytest.raises(ValueError, match="collides|conflicts"):
        getattr(source.param, operation)(ordinary)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("family", ["position", "rotation", "pose"])
@pytest.mark.parametrize("operation", ["at", "resample_to"])
def test_typed_queries_consume_inherited_payload_name(lazy, family, operation):
    """ID: PARAM_QUERY_CHAIN_003; typed families use the same preflight consumption."""
    from tal.spatial import Pose
    from tests.core.test_spatial_numerical_validity import _components

    position, rotation = _components(lazy=lazy)
    source = {"position": position, "rotation": rotation, "pose": Pose.from_components(rotation, position)}[family]
    name = next(iter(source.as_dataset().data_vars))
    source = source.rename({name: "old_count"})
    metadata = _source(count_name="old_count").param.at([.5]).as_dataset().coords["old_count"]
    query = xr.DataArray([[.5], [.5]], dims=("trial", "sample"), coords={"trial": ["a", "b"], "old_count": metadata})
    if lazy:
        query = query.chunk({"sample": 1})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = getattr(source.param, operation)(query)
        expected = getattr(source.param, operation)(query.drop_vars("old_count"))
    assert not tasks
    xr.testing.assert_identical(result.as_dataset().compute(), expected.as_dataset().compute())
    assert "old_count" in result.as_dataset().data_vars


def test_generated_marker_does_not_consume_a_native_index():
    """ID: PARAM_QUERY_CHAIN_004; indexes retain namespace protection despite marker attrs."""
    metadata = _source(count_name="old_count").param.at([.5]).as_dataset().coords["old_count"]
    query = xr.DataArray([.5, 1.], dims="q", coords={"old_count": ("q", [0, 1], metadata.attrs)}).set_xindex("old_count")
    source = _source().rename({"value": "old_count"})
    with pytest.raises(ValueError, match="collides with an output data variable"):
        source.param.at(query)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("name", ["count", "old_count"])
@pytest.mark.parametrize("operation", ["index", "at", "sel", "resample_to"])
def test_indexed_prior_size_is_not_consumed(lazy, empty, name, operation):
    """ID: PARAM_QUERY_CHAIN_005; every size-consumption branch protects indexes."""
    source = _source(lazy=lazy)
    prior = _source(count_name=name).param.at(xr.DataArray(
        [[.5, 8.], [.5, 1.]], dims=("trial", "q"), coords={"trial": ["a", "b"]},
    ))
    metadata = prior.as_dataset().coords[name]
    query = xr.DataArray(
        np.full((2, 0 if empty else 1), .5), dims=("trial", "q"),
        coords={"trial": ["a", "b"], name: metadata, "note": "caller"},
    ).set_xindex(name)
    if lazy:
        query = query.chunk({"q": 1})
    before, query_before = source.as_dataset(), query.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])), pytest.raises(
        ValueError, match=r"^param .*incompatible xarray index topology",
    ):
        getattr(source.param, operation)(query)
    assert not tasks
    xr.testing.assert_identical(source.as_dataset(), before)
    xr.testing.assert_identical(query, query_before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("plain", [False, True])
def test_index_preserves_source_size_index_when_shared(lazy, empty, plain):
    """ID: PARAM_QUERY_CHAIN_006; preflight compares surviving source index groups."""
    source = _source(lazy=lazy)
    source = type(source)(source.as_dataset().set_xindex("count"))
    query = xr.DataArray(
        np.full((2, 0 if empty else 1), .5), dims=("trial", "q"),
        coords={"trial": ["a", "b"], "count": ("trial", [4, 2]), "note": "caller"},
    ).set_xindex("count")
    if lazy:
        query = query.chunk({"q": 1})
    if plain:
        query = [] if empty else [.5]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = source.param.index(query)
    assert not tasks
    assert result.dims == ("trial", "query")
    assert result.xindexes["count"].equals(source.as_dataset().xindexes["count"])
    np.testing.assert_array_equal(result.coords["count"], [4, 2])
    np.testing.assert_array_equal(result.compute(), np.zeros((2, 0 if empty else 1), dtype=int))
    if not plain:
        assert result.note.item() == "caller"


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("name", ["valid", "sample_index"])
@pytest.mark.parametrize("axis", [False, True])
def test_index_cannot_consume_reserved_axis_or_index(lazy, name, axis):
    """ID: PARAM_QUERY_CHAIN_007; provenance does not authorize destroying axes."""
    source = _source(lazy=lazy)
    metadata = _source().param.sel([.5, 1.]).as_dataset().coords[name]
    dim = name if axis else "q"
    query = xr.DataArray([.5, 1.], dims=dim, coords={name: (dim, [0, 1], metadata.attrs)})
    if not axis:
        query = query.set_xindex(name)
    if lazy:
        query = query.chunk({dim: 1})
    before = query.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])), pytest.raises(
        ValueError, match=rf"^param index: query axis or index '{name}' conflicts",
    ):
        source.param.index(query)
    assert not tasks
    xr.testing.assert_identical(query, before)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("name", ["valid", "sample_index"])
@pytest.mark.parametrize("operation", ["index", "at", "sel", "resample_to"])
def test_source_reserved_indexes_fail_before_metadata_consumption(lazy, name, operation):
    """ID: PARAM_QUERY_CHAIN_008; source metadata cannot dismantle native groups."""
    prior = _source().param.sel([.5])
    ds = prior.as_dataset().isel(sample=0, drop=True).expand_dims(sample=[0])
    ds = ds.assign_coords(time=("sample", [0.])).set_xindex(name)
    if lazy:
        ds["value"] = ds.value.chunk({"trial": 1}).variable
    source = type(prior)(ds)
    before = source.as_dataset()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])), pytest.raises(
        ValueError, match=rf"^param .*reserved metadata name collision.*{name}",
    ):
        getattr(source.param, operation)([0.])
    assert not tasks
    xr.testing.assert_identical(source.as_dataset(), before)
