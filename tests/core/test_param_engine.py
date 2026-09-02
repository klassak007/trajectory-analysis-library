import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, set_param_coord, set_roles, set_validity
from tal.core.param_engine import (
    ParamMap,
    ParamMapOptions,
    apply_param_map,
    build_param_bounds_map,
    build_param_map,
    declared_param_coord_name,
    normalize_query_grid,
    resolve_param_coord,
    resolve_param_coord_name,
    resolve_param_valid_mask,
    resolve_role_dims,
)


def _ds_unbatched() -> xr.Dataset:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 10.0, 20.0, 30.0])},
        coords={"sample": [0, 1, 2, 3], "tau": ("sample", [0.0, 1.0, 1.0, 2.0])},
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=())
    return set_param_coord(ds, name="tau")


def _ds_batched() -> xr.Dataset:
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 10.0, 20.0], [100.0, 110.0, 120.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (("trial", "sample"), [[0.0, 0.5, 1.0], [10.0, 11.0, np.nan]]),
            "group_size": ("trial", [3, 2]),
        },
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=())
    ds = set_param_coord(ds, name="phase")
    return set_validity(ds, sequence_size_coord="group_size")


def test_param_engine_001_resolve_role_dims_from_schema() -> None:
    ds = _ds_batched()
    sequence_dim, batch_dims = resolve_role_dims(ds)
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)


def test_param_engine_002_resolve_param_name_explicit_overrides_schema() -> None:
    ds = _ds_unbatched().assign_coords(explicit_tau=("sample", [0.0, 2.0, 4.0, 6.0]))
    assert declared_param_coord_name(ds) == "tau"
    assert resolve_param_coord_name(ds, explicit_name="explicit_tau") == "explicit_tau"


def test_param_engine_003_resolve_param_spec_uses_schema() -> None:
    spec = resolve_param_coord(_ds_batched())
    assert spec is not None
    assert spec.name == "phase"
    assert spec.sequence_dim == "sample"
    assert spec.batch_dims == ("trial",)
    assert tuple(spec.coord.dims) == ("trial", "sample")


def test_param_engine_004_resolve_param_valid_mask_prefers_sequence_size() -> None:
    ds = _ds_batched()
    spec = resolve_param_coord(ds)
    assert spec is not None
    mask = resolve_param_valid_mask(ds, spec=spec)
    assert tuple(mask.dims) == ("trial", "sample")
    np.testing.assert_array_equal(mask.sel(trial="a").values, [True, True, True])
    np.testing.assert_array_equal(mask.sel(trial="b").values, [True, True, False])


def test_param_engine_005_valid_mask_fallbacks_to_param_finite() -> None:
    ds = set_validity(_ds_batched(), sequence_size_coord=None)
    spec = resolve_param_coord(ds)
    assert spec is not None
    mask = resolve_param_valid_mask(ds, spec=spec)
    assert tuple(mask.dims) == ("trial", "sample")
    np.testing.assert_array_equal(mask.sel(trial="a").values, [True, True, True])
    np.testing.assert_array_equal(mask.sel(trial="b").values, [True, True, False])


def test_param_engine_006_normalize_query_grid_scalar() -> None:
    grid = normalize_query_grid(0.5, query_dim="q")
    assert grid.query_dim == "q"
    assert tuple(grid.values.dims) == ("q",)
    np.testing.assert_allclose(grid.values.values, [0.5])


def test_param_engine_007_normalize_query_grid_batched_reindex() -> None:
    ds = _ds_batched()
    query = xr.DataArray(
        [[10.5, 11.5], [0.25, 0.75]],
        dims=("trial", "k"),
        coords={"trial": ["b", "a"], "k": [0, 1]},
    )
    grid = normalize_query_grid(
        query,
        query_dim="k",
        batch_dims=("trial",),
        batch_coords={"trial": ds.coords["trial"]},
    )
    assert tuple(grid.values.dims) == ("trial", "k")
    assert list(grid.values.coords["trial"].values) == ["a", "b"]
    np.testing.assert_allclose(grid.values.sel(trial="a").values, [0.25, 0.75])
    np.testing.assert_allclose(grid.values.sel(trial="b").values, [10.5, 11.5])


def test_param_engine_008_build_map_linear_default_duplicate_invalid() -> None:
    ds = _ds_unbatched()
    spec = resolve_param_coord(ds)
    assert spec is not None
    grid = normalize_query_grid([1.0], query_dim="q")
    pmap = build_param_map(
        param=spec.coord,
        query=grid.values,
        sequence_dim=spec.sequence_dim,
        query_dim=grid.query_dim,
    )
    assert bool(pmap.valid.values[0]) is False


def test_param_engine_009_build_map_linear_duplicate_right_policy() -> None:
    ds = _ds_unbatched()
    spec = resolve_param_coord(ds)
    assert spec is not None
    grid = normalize_query_grid([1.0], query_dim="q")
    pmap = build_param_map(
        param=spec.coord,
        query=grid.values,
        sequence_dim=spec.sequence_dim,
        query_dim=grid.query_dim,
        options=ParamMapOptions(method="linear", duplicate_policy="right"),
    )
    assert bool(pmap.valid.values[0]) is True
    assert int(pmap.i0.values[0]) == 2
    assert int(pmap.i1.values[0]) == 2


def test_param_engine_010_build_map_nearest() -> None:
    ds = _ds_batched()
    spec = resolve_param_coord(ds)
    assert spec is not None
    grid = normalize_query_grid(
        xr.DataArray([[0.2, 0.8], [10.6, 10.9]], dims=("trial", "q"), coords={"trial": ["a", "b"]}),
        query_dim="q",
        batch_dims=("trial",),
        batch_coords={"trial": ds.coords["trial"]},
    )
    pmap = build_param_map(
        param=spec.coord,
        query=grid.values,
        sequence_dim=spec.sequence_dim,
        query_dim=grid.query_dim,
        valid_mask=resolve_param_valid_mask(ds, spec=spec),
        options=ParamMapOptions(method="nearest"),
    )
    np.testing.assert_array_equal(pmap.valid.sel(trial="a").values, [True, True])
    np.testing.assert_array_equal(pmap.valid.sel(trial="b").values, [True, True])
    np.testing.assert_array_equal(pmap.i0.sel(trial="a").values, [0, 2])
    np.testing.assert_array_equal(pmap.i0.sel(trial="b").values, [1, 1])


def test_param_engine_011_apply_map_linear() -> None:
    ds = _ds_unbatched()
    spec = resolve_param_coord(ds)
    assert spec is not None
    grid = normalize_query_grid([0.5, 1.5], query_dim="q")
    pmap = build_param_map(
        param=spec.coord,
        query=grid.values,
        sequence_dim=spec.sequence_dim,
        query_dim=grid.query_dim,
        options=ParamMapOptions(method="linear", duplicate_policy="left"),
    )
    out = apply_param_map(ds["value"], param_map=pmap, sequence_dim="sample")
    np.testing.assert_allclose(out.values, [5.0, 25.0])


def test_param_engine_012_monotonic_check_raises() -> None:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 10.0, 20.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 2.0, 1.0])},
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=())
    ds = set_param_coord(ds, name="tau")
    spec = resolve_param_coord(ds)
    assert spec is not None
    grid = normalize_query_grid([0.5], query_dim="q")
    with pytest.raises(ValueError) as err:
        build_param_map(
            param=spec.coord,
            query=grid.values,
            sequence_dim=spec.sequence_dim,
            query_dim=grid.query_dim,
        )
    assert "monotonic non-decreasing" in str(err.value)


def test_param_engine_013_normalize_query_grid_batch_scalar_axis_preserved() -> None:
    """ID: PARAM_ENGINE_013_normalize_query_grid_batch_scalar_axis_preserved."""
    ds = _ds_batched()
    query = xr.DataArray([0.25, 10.5], dims=("trial",), coords={"trial": ["a", "b"]})
    grid = normalize_query_grid(
        query,
        query_dim="q",
        batch_dims=("trial",),
        batch_coords={"trial": ds.coords["trial"]},
    )
    assert tuple(grid.values.dims) == ("trial", "q")
    assert grid.values.sizes["q"] == 1
    np.testing.assert_allclose(grid.values.sel(trial="a").values, [0.25])
    np.testing.assert_allclose(grid.values.sel(trial="b").values, [10.5])


def test_param_engine_014_normalize_query_grid_existing_query_dim_with_extra_axis() -> None:
    """ID: PARAM_ENGINE_014_normalize_query_grid_existing_query_dim_with_extra_axis."""
    ds = _ds_batched()
    query = xr.DataArray(
        np.arange(8, dtype="float64").reshape(2, 2, 2),
        dims=("trial", "q", "extra"),
        coords={"trial": ["a", "b"], "q": [0, 1], "extra": [0, 1]},
    )
    grid = normalize_query_grid(
        query,
        query_dim="q",
        batch_dims=("trial",),
        batch_coords={"trial": ds.coords["trial"]},
    )
    assert tuple(grid.values.dims) == ("trial", "q")
    assert grid.values.sizes["q"] == 4
    assert grid.stacked_dims == ("q", "extra")


def test_param_engine_015_normalize_query_grid_partial_batch_dims_fail_fast() -> None:
    """ID: PARAM_ENGINE_015_normalize_query_grid_partial_batch_dims_fail_fast."""
    query = xr.DataArray([1.0, 2.0], dims=("trial",), coords={"trial": ["a", "b"]})
    with pytest.raises(ValueError) as err:
        normalize_query_grid(query, query_dim="q", batch_dims=("trial", "sensor"))
    assert "partial batch topology" in str(err.value)


def test_param_engine_016_apply_map_empty_sequence_returns_nan() -> None:
    """ID: PARAM_ENGINE_016_apply_map_empty_sequence_returns_nan."""
    values = xr.DataArray(
        np.empty((2, 0), dtype="float64"),
        dims=("channel", "sample"),
        coords={"channel": ["x", "y"], "sample": []},
    )
    q = xr.DataArray([0, 1], dims=("q",))
    pmap = ParamMap(
        i0=xr.zeros_like(q, dtype="int64"),
        i1=xr.zeros_like(q, dtype="int64"),
        alpha=xr.zeros_like(q, dtype="float64"),
        valid=xr.zeros_like(q, dtype=bool),
        query_dim="q",
    )
    out = apply_param_map(values, param_map=pmap, sequence_dim="sample")
    assert tuple(out.dims) == ("channel", "q")
    assert np.isnan(out.values).all()


def test_param_engine_017_normalize_query_grid_query_dim_batch_collision_fails() -> None:
    """ID: PARAM_ENGINE_017_normalize_query_grid_query_dim_batch_collision_fails."""
    query = xr.DataArray([0.1, 0.2], dims=("trial",), coords={"trial": ["a", "b"]})
    with pytest.raises(ValueError) as err:
        normalize_query_grid(query, query_dim="trial", batch_dims=("trial",))
    assert "collides with batch_dims" in str(err.value)


def test_param_engine_018_normalize_query_grid_unlabeled_nd_numpy_rejected() -> None:
    """ID: PARAM_ENGINE_018_normalize_query_grid_unlabeled_nd_numpy_rejected."""
    with pytest.raises(ValueError) as err:
        normalize_query_grid(np.asarray([[0.1, 0.2], [10.0, 11.0]]), query_dim="q")
    assert "unlabeled numpy query" in str(err.value)


def test_param_engine_019_build_param_map_query_dim_equals_sequence_dim_fails_fast() -> None:
    """ID: PARAM_ENGINE_019_build_param_map_query_dim_equals_sequence_dim_fails_fast."""
    ds = _ds_unbatched()
    spec = resolve_param_coord(ds)
    assert spec is not None
    query = xr.DataArray([0.25, 1.75], dims=("sample",), coords={"sample": [0, 1]})
    with pytest.raises(ValueError) as err:
        build_param_map(
            param=spec.coord,
            query=query,
            sequence_dim=spec.sequence_dim,
            query_dim=spec.sequence_dim,
        )
    msg = str(err.value)
    assert "query_dim must differ from sequence_dim" in msg


def test_param_engine_033_integer_query_dtype_preserved() -> None:
    """ID: PARAM_ENGINE_033_integer_query_dtype_preserved."""
    integer = normalize_query_grid(np.asarray([1, 2], dtype="int32"), query_dim="query")
    floating = normalize_query_grid(np.asarray([0.25, 0.75], dtype="float32"), query_dim="query")
    assert integer.values.dtype == np.dtype("int32")
    assert floating.values.dtype == np.dtype("float32")


def test_param_engine_034_large_integer_exact_lookup_preserved() -> None:
    """ID: PARAM_ENGINE_034_large_integer_exact_lookup_preserved."""
    base = 2**53
    ds = xr.Dataset(
        {"value": ("sample", [10.0, 20.0])},
        coords={"sample": [0, 1], "tau": ("sample", np.asarray([base, base + 1], dtype="int64"))},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), param_coord="tau")
    assert int(ao.param.index(np.int64(base + 1)).item()) == 1
    assert float(ao.param.sel(np.int64(base + 1)).as_dataset(copy="none")["value"].item()) == 20.0

    maximum = np.iinfo(np.uint64).max
    param = xr.DataArray(np.asarray([maximum - 1, maximum], dtype="uint64"), dims=("sample",))
    query = xr.DataArray(np.asarray([maximum], dtype="uint64"), dims=("query",))
    pmap = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="nearest"),
    )
    assert int(pmap.i0.item()) == 1

    tie_param = xr.DataArray(np.asarray([maximum - 2, maximum], dtype="uint64"), dims=("sample",))
    tie_query = xr.DataArray(np.asarray([maximum - 1], dtype="uint64"), dims=("query",))
    tie = build_param_map(
        param=tie_param,
        query=tie_query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="nearest"),
    )
    assert int(tie.i0.item()) == 0


def test_param_engine_035_large_integer_linear_and_bounds_preserved() -> None:
    """ID: PARAM_ENGINE_035_large_integer_linear_and_bounds_preserved."""
    base = 2**53
    param = xr.DataArray(np.asarray([base, base + 2], dtype="int64"), dims=("sample",))
    query = xr.DataArray(np.asarray([base + 1], dtype="int64"), dims=("query",))
    pmap = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear"),
    )
    assert (int(pmap.i0.item()), int(pmap.i1.item()), float(pmap.alpha.item())) == (0, 1, 0.5)

    bound_param = xr.DataArray(np.asarray([base, base + 1, base + 2], dtype="int64"), dims=("sample",))
    bounds = build_param_bounds_map(
        param=bound_param,
        start=np.int64(base + 1),
        stop=np.int64(base + 2),
        sequence_dim="sample",
    )
    assert (int(bounds.i0.item()), int(bounds.i1.item())) == (1, 3)
    open_bounds = build_param_bounds_map(
        param=bound_param,
        start=None,
        stop=np.int64(base + 1),
        sequence_dim="sample",
    )
    assert (int(open_bounds.i0.item()), int(open_bounds.i1.item())) == (0, 2)

    duplicates = xr.DataArray(
        np.asarray([base, base + 1, base + 1, base + 2], dtype="int64"),
        dims=("sample",),
    )
    duplicate_map = build_param_map(
        param=duplicates,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear", duplicate_policy="right"),
    )
    assert (int(duplicate_map.i0.item()), int(duplicate_map.i1.item())) == (2, 2)
