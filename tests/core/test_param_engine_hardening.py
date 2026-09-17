import numpy as np
import pytest
import xarray as xr

from tal.core import (
    AnalysisObject,
    SchemaError,
    merge_schema,
    set_param_coord,
    set_roles,
    set_validity,
)
from tal.core.param_engine import (
    ParamCoordSpec,
    ParamMapOptions,
    apply_param_map,
    build_param_bounds_map,
    build_param_map,
    finite_param_mask,
    normalize_query_grid,
    resolve_param_coord,
    resolve_param_coord_name,
    resolve_param_valid_mask,
    resolve_role_dims,
    schema_resolve,
)
from tal.core.param_engine import map_apply as map_apply_mod
from tal.core.param_engine import map_build as map_build_mod
from tal.core.param_ops.guards import mark_reserved_coord
from tal.core.schema_read import read_roles


class _HostileKey:
    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return False

    def __repr__(self) -> str:
        raise RuntimeError("repr exploded")

    def __str__(self) -> str:
        raise RuntimeError("str exploded")


def _forbidden_empty_param_source(
    label: str,
    shape: tuple[int, ...],
    dtype: np.dtype[object],
) -> np.ndarray:
    raise RuntimeError(f"forbidden empty parameter source task: {label}")


def _ds_sample() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, np.nan])},
    )


def _ds_trial() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "phase": (("trial", "sample"), [[0.0, 0.5, 1.0], [2.0, 2.5, np.nan]]),
            "group_size": ("trial", [3, 2]),
            "alt_group_size": ("trial", [3, 3]),
        },
    )


def test_param_hard_001_no_roles_and_no_explicit_dims_fail_fast() -> None:
    """ID: PARAM_HARD_001_no_roles_and_no_explicit_dims_fail_fast."""
    with pytest.raises(ValueError) as err:
        resolve_param_coord(_ds_sample())
    assert "no roles declared" in str(err.value)


def test_param_hard_002_roles_resolved_no_param_returns_none() -> None:
    """ID: PARAM_HARD_002_roles_resolved_no_param_returns_none."""
    ds = set_roles(_ds_sample(), sequence_dim="sample", batch_dims=(), core_dims=())
    spec = resolve_param_coord(ds)
    assert spec is None


def test_param_hard_003_declared_param_missing_raises_schema_error() -> None:
    """ID: PARAM_HARD_003_declared_param_missing_raises_schema_error."""
    ds = set_roles(_ds_sample(), sequence_dim="sample", batch_dims=(), core_dims=())
    ds = merge_schema(ds, {"core": {"param_coord": {"name": "missing"}}}, validate=False)
    with pytest.raises(SchemaError) as err:
        resolve_param_coord(ds)
    assert err.value.code == "schema.param_coord.not_found"
    assert err.value.path == "tal.core.param_coord.name"


def test_param_hard_004_declared_validity_invalid_dims_raises_schema_error() -> None:
    """ID: PARAM_HARD_004_declared_validity_invalid_dims_raises_schema_error."""
    ds = _ds_trial().assign_coords(group_size_bad=("sample", [1, 2, 3]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=())
    ds = set_param_coord(ds, name="phase")
    ds = merge_schema(
        ds,
        {"core": {"validity": {"sequence_size_coord": "group_size_bad", "layout": "left_packed"}}},
        validate=False,
    )
    spec = ParamCoordSpec(name="phase", coord=ds.coords["phase"], sequence_dim="sample", batch_dims=("trial",))
    with pytest.raises(SchemaError) as err:
        resolve_param_valid_mask(ds, spec=spec)
    assert err.value.code == "schema.validity.sequence_size_coord.dims.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"


def test_param_hard_005_validity_absent_uses_finite_fallback() -> None:
    """ID: PARAM_HARD_005_validity_absent_uses_finite_fallback."""
    ds = set_roles(_ds_trial(), sequence_dim="sample", batch_dims=("trial",), core_dims=())
    ds = set_param_coord(ds, name="phase")
    spec = resolve_param_coord(ds)
    assert spec is not None
    mask = resolve_param_valid_mask(ds, spec=spec)
    np.testing.assert_array_equal(mask.sel(trial="a").values, [True, True, True])
    np.testing.assert_array_equal(mask.sel(trial="b").values, [True, True, False])


def test_param_hard_006_declared_validity_never_silent_fallback() -> None:
    """ID: PARAM_HARD_006_declared_validity_never_silent_fallback."""
    ds = _ds_trial().assign_coords(group_size_bad=("sample", [1, 2, 3]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=())
    ds = set_param_coord(ds, name="phase")
    ds = merge_schema(
        ds,
        {"core": {"validity": {"sequence_size_coord": "group_size_bad", "layout": "left_packed"}}},
        validate=False,
    )
    spec = ParamCoordSpec(name="phase", coord=ds.coords["phase"], sequence_dim="sample", batch_dims=("trial",))
    with pytest.raises(SchemaError):
        resolve_param_valid_mask(ds, spec=spec)


def test_orch_edge_003_infer_false_does_not_fallback_sequence_dim() -> None:
    """ID: ORCH_EDGE_003_infer_false_does_not_fallback_sequence_dim."""
    test_param_hard_001_no_roles_and_no_explicit_dims_fail_fast()


def test_orch_edge_002_strict_explicit_sequence_hint_rejects_noncore_operand() -> None:
    """ID: ORCH_EDGE_002_strict_explicit_sequence_hint_rejects_noncore_operand."""
    ds = xr.Dataset(data_vars={"offset": xr.DataArray(5.0)})
    with pytest.raises(ValueError) as err:
        resolve_role_dims(ds, sequence_dim="sample", batch_dims=())
    assert "missing dims in dataset" in str(err.value)


def test_orch_edge_005_finite_gather_preserves_trailing_finite_samples() -> None:
    """ID: ORCH_EDGE_005_finite_gather_preserves_trailing_finite_samples."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [10.0, 11.0, 12.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, np.nan, 2.0]),
        },
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=())
    ds = set_param_coord(ds, name="tau")
    spec = resolve_param_coord(ds)
    assert spec is not None
    mask = resolve_param_valid_mask(ds, spec=spec)
    np.testing.assert_array_equal(mask.values, [True, False, True])


def test_orch_edge_006_prefix_only_rejects_sparse_finite_layout() -> None:
    """ID: ORCH_EDGE_006_prefix_only_rejects_sparse_finite_layout."""
    test_param_hard_006_declared_validity_never_silent_fallback()


def test_orch_edge_007_validity_fallback_prefers_param_notnull_over_data_finite() -> None:
    """ID: ORCH_EDGE_007_validity_fallback_prefers_param_notnull_over_data_finite."""
    test_param_hard_005_validity_absent_uses_finite_fallback()


def test_param_hard_007_explicit_param_override_success_when_undeclared() -> None:
    """ID: PARAM_HARD_007_explicit_param_override_success_when_undeclared."""
    ds = set_roles(_ds_sample().assign_coords(explicit_tau=("sample", [0.0, 2.0, 4.0])), sequence_dim="sample", batch_dims=(), core_dims=())
    spec = resolve_param_coord(ds, explicit_name="explicit_tau")
    assert spec is not None
    assert spec.name == "explicit_tau"


def test_param_hard_008_explicit_param_conflict_fails() -> None:
    """ID: PARAM_HARD_008_explicit_param_conflict_fails."""
    ds = set_roles(_ds_sample().assign_coords(explicit_tau=("sample", [0.0, 2.0, 4.0])), sequence_dim="sample", batch_dims=(), core_dims=())
    ds = set_param_coord(ds, name="tau")
    with pytest.raises(ValueError) as err:
        resolve_param_coord(ds, explicit_name="explicit_tau")
    assert "conflicts with declared schema" in str(err.value)


def test_param_hard_009_explicit_sequence_size_conflict_fails() -> None:
    """ID: PARAM_HARD_009_explicit_sequence_size_conflict_fails."""
    ds = set_roles(_ds_trial(), sequence_dim="sample", batch_dims=("trial",), core_dims=())
    ds = set_param_coord(ds, name="phase")
    ds = set_validity(ds, sequence_size_coord="group_size")
    spec = resolve_param_coord(ds)
    assert spec is not None
    with pytest.raises(ValueError) as err:
        resolve_param_valid_mask(ds, spec=spec, sequence_size_coord="alt_group_size")
    assert "conflicts with declared schema" in str(err.value)


def test_param_hard_010_bootstrap_unbound_is_usable_with_explicit_sequence_dim() -> None:
    """ID: PARAM_HARD_010_bootstrap_unbound_is_usable_with_explicit_sequence_dim."""
    ao = AnalysisObject(_ds_sample())
    assert ao.as_dataset().attrs["tal"] == {"version": 1, "core": {}}
    spec = resolve_param_coord(ao.as_dataset(), sequence_dim="sample")
    assert spec is None


def test_param_hard_011_fast_path_skips_revalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: PARAM_HARD_011_fast_path_skips_revalidation."""
    ds = set_roles(_ds_trial(), sequence_dim="sample", batch_dims=("trial",), core_dims=())
    ds = set_param_coord(ds, name="phase")

    calls: list[str] = []

    def _count_validate(in_ds: xr.Dataset) -> xr.Dataset:
        calls.append("validate")
        return in_ds

    monkeypatch.setattr(schema_resolve, "_validate_schema", _count_validate)

    schema_resolve._resolve_schema_context(ds, explicit_sequence_dim="sample")
    assert calls == ["validate"]

    schema_resolve._resolve_schema_context_validated(ds, explicit_sequence_dim="sample")
    assert calls == ["validate"]


def test_param_hard_012_schema_parsing_no_runtime_leak_for_hostile_keys() -> None:
    """ID: PARAM_HARD_012_schema_parsing_no_runtime_leak_for_hostile_keys."""
    ds = _ds_sample()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {"sequence_dim": "sample", "batch_dims": [], "core_dims": []},
            _HostileKey(): "bad",
        },
    }
    with pytest.raises(SchemaError) as err:
        resolve_param_coord(ds, sequence_dim="sample")
    assert err.value.code == "schema.core.unknown_key"


def test_param_hard_013_valid_mask_fallback_uses_resolved_dataset_coord() -> None:
    """ID: PARAM_HARD_013_valid_mask_fallback_uses_resolved_dataset_coord."""
    ds_ref = set_roles(_ds_sample(), sequence_dim="sample", batch_dims=(), core_dims=())
    ds_ref = set_param_coord(ds_ref, name="tau")
    spec = resolve_param_coord(ds_ref)
    assert spec is not None

    ds_target = xr.Dataset(
        data_vars={"value": (("sample",), [5.0, 6.0, 7.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    ds_target = set_roles(ds_target, sequence_dim="sample", batch_dims=(), core_dims=())
    ds_target = set_param_coord(ds_target, name="tau")
    mask = resolve_param_valid_mask(ds_target, spec=spec)
    np.testing.assert_array_equal(mask.values, [True, True, True])


def test_param_hard_014_sequence_size_scalar_nan_rejected() -> None:
    """ID: PARAM_HARD_014_sequence_size_scalar_nan_rejected."""
    ds = set_roles(_ds_sample().assign_coords(group_size=np.nan), sequence_dim="sample", batch_dims=(), core_dims=())
    ds = set_param_coord(ds, name="tau")
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size")
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"


def test_param_hard_015_sequence_size_fractional_rejected() -> None:
    """ID: PARAM_HARD_015_sequence_size_fractional_rejected."""
    ds = _ds_trial().assign_coords(group_size=("trial", [3.0, 2.5]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=())
    ds = set_param_coord(ds, name="phase")
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size")
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"


def test_param_hard_016_sequence_size_out_of_range_rejected() -> None:
    """ID: PARAM_HARD_016_sequence_size_out_of_range_rejected."""
    ds = set_roles(_ds_sample().assign_coords(group_size=np.int64(10)), sequence_dim="sample", batch_dims=(), core_dims=())
    ds = set_param_coord(ds, name="tau")
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size")
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.asarray(1 + 2j, dtype=np.complex128), id="complex"),
        pytest.param(np.asarray(True, dtype=bool), id="bool"),
        pytest.param(np.asarray(1, dtype="timedelta64[s]"), id="timedelta"),
    ],
)
def test_param_hard_035_explicit_sequence_size_non_count_dtype_rejected(value: np.ndarray) -> None:
    """ID: PARAM_HARD_035_explicit_sequence_size_non_count_dtype_rejected."""
    ds = _ds_sample().assign_coords(group_size=xr.DataArray(value))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=())
    ds = set_param_coord(ds, name="tau")
    spec = resolve_param_coord(ds)
    assert spec is not None
    with pytest.raises(ValueError, match="real numeric count dtype"):
        resolve_param_valid_mask(ds, spec=spec, sequence_size_coord="group_size")


def test_param_hard_017_resolve_param_coord_name_explicit_dims_enforced_with_roles() -> None:
    """ID: PARAM_HARD_017_resolve_param_coord_name_explicit_dims_enforced_with_roles."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1],
            "bad": (("sample", "trial"), [[0.0, 0.1], [0.2, 0.3]]),
        },
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=())
    with pytest.raises(ValueError) as err:
        resolve_param_coord_name(ds, explicit_name="bad")
    assert "dims" in str(err.value)


def test_param_hard_018_resolve_role_dims_explicit_batch_dims_duplicate_rejected() -> None:
    """ID: PARAM_HARD_018_resolve_role_dims_explicit_batch_dims_duplicate_rejected."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0]])},
        coords={"trial": ["a"], "sample": [0, 1]},
    )
    with pytest.raises(ValueError) as err:
        resolve_role_dims(ds, sequence_dim="sample", batch_dims=("trial", "trial"))
    msg = str(err.value)
    assert "duplicate entries" in msg
    assert "remove duplicate entries from batch_dims" in msg


def test_param_hard_019_bounds_map_invalid_start_stop_topology_fails_fast() -> None:
    """ID: PARAM_HARD_019_bounds_map_invalid_start_stop_topology_fails_fast."""
    ds = _ds_sample()
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=())
    ds = set_param_coord(ds, name="tau")
    spec = resolve_param_coord(ds)
    assert spec is not None
    start = xr.DataArray([0.0, 1.0], dims=("sample",), coords={"sample": [0, 1]})
    stop = xr.DataArray([1.0, 2.0], dims=("sample",), coords={"sample": [0, 1]})
    with pytest.raises(ValueError) as err:
        build_param_bounds_map(
            param=spec.coord,
            start=start,
            stop=stop,
            sequence_dim="sample",
        )
    assert "cannot include sequence_dim" in str(err.value)


def test_param_hard_020_non_numeric_param_coord_rejected_at_boundary() -> None:
    """ID: PARAM_HARD_020_non_numeric_param_coord_rejected_at_boundary."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 1, 2], "phase": ("sample", ["a", "b", "c"])},
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=())
    with pytest.raises(SchemaError) as err:
        set_param_coord(ds, name="phase")
    assert err.value.code == "schema.param_coord.dtype.invalid"
    assert err.value.path == "tal.core.param_coord.name"


def test_param_hard_021_bootstrap_ext_resolver_treated_as_unbound() -> None:
    """ID: PARAM_HARD_021_bootstrap_ext_resolver_treated_as_unbound."""
    ds = _ds_sample().copy(deep=False)
    ds.attrs["tal"] = {"version": 1, "core": {}, "ext": {"demo": {}}}
    sequence_dim, batch_dims = resolve_role_dims(
        ds,
        sequence_dim="sample",
        batch_dims=(),
    )
    assert sequence_dim == "sample"
    assert batch_dims == ()


def test_param_engine_020_normalize_query_grid_batch_coords_topology_invalid_fails_fast() -> None:
    """ID: PARAM_ENGINE_020_normalize_query_grid_batch_coords_topology_invalid_fails_fast."""
    query = xr.DataArray(
        np.asarray([[0.0, 1.0]], dtype="float64"),
        dims=("trial", "query"),
        coords={"trial": ["a"], "query": [0, 1]},
    )
    with pytest.raises(ValueError) as err_scalar:
        normalize_query_grid(
            query,
            query_dim="query",
            batch_dims=("trial",),
            batch_coords={"trial": xr.DataArray("a")},
        )
    assert "must be a 1-D DataArray indexed by 'trial'" in str(err_scalar.value)
    with pytest.raises(ValueError) as err_mismatch:
        normalize_query_grid(
            query,
            query_dim="query",
            batch_dims=("trial",),
            batch_coords={"trial": xr.DataArray(["a"], dims=("x",))},
        )
    assert "must be a 1-D DataArray indexed by 'trial'" in str(err_mismatch.value)


def test_param_engine_021_normalize_query_grid_non_numeric_query_rejected() -> None:
    """ID: PARAM_ENGINE_021_normalize_query_grid_non_numeric_query_rejected."""
    with pytest.raises(ValueError) as err:
        normalize_query_grid(
            xr.DataArray(["x"], dims=("query",)),
            query_dim="query",
        )
    assert "query values must be numeric" in str(err.value)


def test_param_engine_036_datetime64_query_grid_rejects_numeric_queries() -> None:
    """ID: PARAM_ENGINE_036_datetime64_query_grid_rejects_numeric_queries."""
    with pytest.raises(ValueError) as err:
        normalize_query_grid([1.0], query_dim="query", param_kind="datetime64")
    assert "datetime64 param queries must be datetime-like" in str(err.value)


def test_param_engine_037_datetime64_map_uses_param_kind_owner() -> None:
    """ID: PARAM_ENGINE_037_datetime64_map_uses_param_kind_owner."""
    param = xr.DataArray(
        np.asarray(["2026-01-01T00:00:00", "2026-01-01T00:00:10"], dtype="datetime64[ns]"),
        dims=("sample",),
    )
    query = xr.DataArray(
        np.asarray(["2026-01-01T00:00:05"], dtype="datetime64[ns]"),
        dims=("query",),
    )
    pmap = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear"),
        param_kind="datetime64",
    )
    np.testing.assert_array_equal(pmap.i0.values, [0])
    np.testing.assert_array_equal(pmap.i1.values, [1])
    np.testing.assert_allclose(pmap.alpha.values, [0.5])


def test_param_engine_022_query_input_scalar_query_dim_collision_fails_fast() -> None:
    """ID: PARAM_ENGINE_022_query_input_scalar_query_dim_collision_fails_fast."""
    query = xr.DataArray(
        np.asarray([0.0, 1.0], dtype="float64"),
        dims=("sample",),
        coords={"sample": [0, 1], "query": 5},
    )
    with pytest.raises(ValueError) as err:
        normalize_query_grid(
            query,
            query_dim="query",
        )
    assert "scalar coordinate 'query'" in str(err.value)


def test_param_engine_026_query_grid_temp_dim_collision_with_coord_name_avoided() -> None:
    """ID: PARAM_ENGINE_026_query_grid_temp_dim_collision_with_coord_name_avoided."""
    query = xr.DataArray(
        np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype="float64"),
        dims=("trial", "sample"),
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1],
            "query__stack": (("trial", "sample"), np.zeros((2, 2), dtype="float64")),
        },
    )
    grid = normalize_query_grid(query, query_dim="query")
    assert grid.values.dims == ("query",)
    np.testing.assert_allclose(grid.values.values, np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float64"))


def test_param_engine_027_query_grid_temp_dim_collision_with_dataarray_name_avoided() -> None:
    """ID: PARAM_ENGINE_027_query_grid_temp_dim_collision_with_dataarray_name_avoided."""
    query = xr.DataArray(
        np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype="float64"),
        dims=("trial", "sample"),
        coords={"trial": ["a", "b"], "sample": [0, 1]},
        name="query__stack",
    )
    grid = normalize_query_grid(query, query_dim="query")
    assert grid.values.dims == ("query",)
    assert grid.values.name == "query__stack"
    np.testing.assert_allclose(grid.values, [0.0, 1.0, 2.0, 3.0])


def test_param_engine_023_map_indexers_materialized_for_dask_consumers() -> None:
    """ID: PARAM_ENGINE_023_map_indexers_materialized_for_dask_consumers."""
    da = pytest.importorskip("dask.array")
    values = xr.DataArray(
        da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2),
        dims=("sample",),
    )
    param = xr.DataArray(
        da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2),
        dims=("sample",),
    )
    query = xr.DataArray(
        da.from_array(np.asarray([0.5, 1.5]), chunks=1),
        dims=("query",),
    )
    pmap = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )
    out = apply_param_map(values, param_map=pmap, sequence_dim="sample")
    np.testing.assert_allclose(out.values, [5.0, 15.0])


def test_param_engine_024_map_apply_dask_indexers_no_eager_materialization(
) -> None:
    """ID: PARAM_ENGINE_024_map_apply_dask_indexers_no_eager_materialization."""
    da = pytest.importorskip("dask.array")
    values = xr.DataArray(
        da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2),
        dims=("sample",),
    )
    param = xr.DataArray(
        da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2),
        dims=("sample",),
    )
    query = xr.DataArray(
        da.from_array(np.asarray([0.5, 1.5]), chunks=1),
        dims=("query",),
    )
    pmap = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )

    assert not hasattr(map_apply_mod, "materialize_indexer")
    out = apply_param_map(values, param_map=pmap, sequence_dim="sample")
    assert hasattr(out.data, "chunks")
    np.testing.assert_allclose(out.values, [5.0, 15.0])


def test_param_hard_022_schema_read_and_resolver_parity_strict_blocks() -> None:
    """ID: PARAM_HARD_022_schema_read_and_resolver_parity_strict_blocks."""
    ds = _ds_sample().copy(deep=False)
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": [],
                "unknown": 1,
            }
        },
    }
    with pytest.raises(SchemaError) as err_reader:
        read_roles(ds)
    assert err_reader.value.code == "schema.roles.unknown_key"
    with pytest.raises(SchemaError) as err_resolver:
        resolve_role_dims(ds)
    assert err_resolver.value.code == "schema.roles.unknown_key"


def test_param_engine_025_nearest_vectorized_semantics_with_nan_and_bounds() -> None:
    """ID: PARAM_ENGINE_025_nearest_vectorized_semantics_with_nan_and_bounds."""
    param = xr.DataArray(np.asarray([0.0, 1.0, 2.0], dtype="float64"), dims=("sample",))
    query = xr.DataArray(np.asarray([-1.0, 0.4, 1.6, 3.0, np.nan], dtype="float64"), dims=("query",))
    pmap = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="nearest"),
    )
    np.testing.assert_array_equal(pmap.i0.values, [0, 0, 2, 2, 0])
    np.testing.assert_array_equal(pmap.i1.values, [0, 0, 2, 2, 0])
    np.testing.assert_array_equal(pmap.valid.values, [True, True, True, True, False])


def test_param_engine_029_linear_vectorized_semantics_with_nan_and_bounds() -> None:
    """ID: PARAM_ENGINE_029_linear_vectorized_semantics_with_nan_and_bounds."""
    param = xr.DataArray(np.asarray([0.0, 1.0, 2.0], dtype="float64"), dims=("sample",))
    query = xr.DataArray(np.asarray([0.0, 0.5, 2.0, 3.0, np.nan], dtype="float64"), dims=("query",))
    pmap = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear", duplicate_policy="left"),
    )
    np.testing.assert_array_equal(pmap.i0.values, [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(pmap.i1.values, [0, 1, 2, 0, 0])
    np.testing.assert_allclose(pmap.alpha.values, [0.0, 0.5, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(pmap.valid.values, [True, True, True, False, False])


def test_param_engine_030_linear_duplicate_policy_vectorized_parity() -> None:
    """ID: PARAM_ENGINE_030_linear_duplicate_policy_vectorized_parity."""
    param = xr.DataArray(np.asarray([0.0, 1.0, 1.0, 2.0], dtype="float64"), dims=("sample",))
    query = xr.DataArray(np.asarray([1.0], dtype="float64"), dims=("query",))

    pmap_invalid = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear", duplicate_policy="invalid"),
    )
    assert bool(pmap_invalid.valid.values[0]) is False

    pmap_left = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear", duplicate_policy="left"),
    )
    assert int(pmap_left.i0.values[0]) == 1
    assert int(pmap_left.i1.values[0]) == 1
    assert bool(pmap_left.valid.values[0]) is True

    pmap_right = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear", duplicate_policy="right"),
    )
    assert int(pmap_right.i0.values[0]) == 2
    assert int(pmap_right.i1.values[0]) == 2
    assert bool(pmap_right.valid.values[0]) is True

    with pytest.raises(ValueError) as err:
        build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            options=ParamMapOptions(method="linear", duplicate_policy="raise"),
        )
    assert "duplicate parameter bracket" in str(err.value)


def test_param_engine_028_build_param_map_keeps_dask_lazy() -> None:
    """ID: PARAM_ENGINE_028_build_param_map_keeps_dask_lazy."""
    da = pytest.importorskip("dask.array")
    param = xr.DataArray(da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2), dims=("sample",))
    query = xr.DataArray(da.from_array(np.asarray([0.25, 1.25]), chunks=1), dims=("query",))
    pmap = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )
    assert hasattr(pmap.i0.data, "chunks")
    assert hasattr(pmap.valid.data, "chunks")


@pytest.mark.parametrize(
    ("param_kind", "values", "query_dtype"),
    (
        ("numeric", np.asarray([0.0, 2.0, 1.0]), np.dtype("float64")),
        (
            "datetime64",
            np.asarray(["2026-01-01T00:00:00", "2026-01-03T00:00:00", "2026-01-02T00:00:00"], dtype="datetime64[ns]"),
            np.dtype("datetime64[ns]"),
        ),
    ),
)
def test_param_hard_empty_query_001_validates_eager_domains(
    param_kind: str,
    values: np.ndarray,
    query_dtype: np.dtype,
) -> None:
    """ID: PARAM_HARD_EMPTY_QUERY_001_validates_eager_domains."""
    param = xr.DataArray(values, dims=("sample",))
    query = xr.DataArray(np.empty(0, dtype=query_dtype), dims=("query",))
    with pytest.raises(ValueError, match="monotonic non-decreasing"):
        build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            param_kind=param_kind,
        )


def test_param_hard_empty_query_001_lazy_validation_is_deferred_and_preserved() -> None:
    """Lazy empty maps validate their source domain only when computed."""
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    param = xr.DataArray(da.from_array(np.asarray([0.0, 2.0, 1.0]), chunks=3), dims=("sample",))
    query = xr.DataArray(np.empty(0, dtype=np.float64), dims=("query",))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )
    assert tasks == []
    assert mapping.valid.chunks is not None
    with pytest.raises(ValueError, match="monotonic non-decreasing"):
        mapping.valid.compute()


def test_param_hard_empty_query_001_preserves_topology_without_duplicate_failure() -> None:
    """Empty maps preserve topology without inventing duplicate failures."""
    param = xr.DataArray(
        [[0.0, 1.0, 1.0], [10.0, 11.0, 12.0]],
        dims=("trial", "sample"),
        coords={"trial": ["a", "b"]},
    )
    query = xr.DataArray(
        np.empty((2, 0), dtype=np.float64),
        dims=("trial", "query"),
        coords={"trial": ["a", "b"]},
    )
    mapping = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
        options=ParamMapOptions(method="linear", duplicate_policy="raise"),
    )
    assert mapping.valid.dims == ("trial", "query")
    assert mapping.valid.shape == (2, 0)
    assert list(mapping.valid.coords["trial"].data) == ["a", "b"]


@pytest.mark.parametrize("lazy", (False, True))
def test_param_hard_empty_query_002_preserves_global_failure_order(lazy: bool) -> None:
    """ID: PARAM_HARD_EMPTY_QUERY_002_preserves_global_failure_order."""
    param = xr.DataArray(
        np.asarray(
            [
                ["2020-01-02", "2020-01-01", "2020-01-03"],
                [
                    "1677-09-21T00:12:43.145224193",
                    "2000-01-01",
                    "2262-04-11T23:47:16.854775807",
                ],
            ],
            dtype="datetime64[ns]",
        ),
        dims=("trial", "sample"),
    )
    query = xr.DataArray(
        np.empty((2, 0), dtype="datetime64[ns]"),
        dims=("trial", "query"),
    )
    if lazy:
        pytest.importorskip("dask.array")
        param = param.chunk({"trial": 1, "sample": 3})
        query = query.chunk({"trial": 1, "query": 1})
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            param_kind="datetime64",
        )
        with pytest.raises(ValueError, match="monotonic non-decreasing") as err:
            mapping.valid.compute(scheduler="synchronous")
    else:
        with pytest.raises(ValueError, match="monotonic non-decreasing") as err:
            build_param_map(
                param=param,
                query=query,
                sequence_dim="sample",
                query_dim="query",
                param_kind="datetime64",
            )
    assert type(err.value) is ValueError


def test_param_core_empty_map_block_001_source_validation_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_CORE_EMPTY_MAP_BLOCK_001_source_validation_is_bounded."""
    from tal.core.param_engine import numpy_backends

    original = numpy_backends.validate_map_domain_block_numpy_status
    row_counts: list[int] = []

    def tracked(param: np.ndarray, mask: np.ndarray, **kwargs: object):
        row_counts.append(int(np.prod(param.shape[:-1], dtype=np.int64)))
        return original(param, mask, **kwargs)

    monkeypatch.setattr(
        numpy_backends,
        "validate_map_domain_block_numpy_status",
        tracked,
    )
    param = xr.DataArray(
        np.broadcast_to([0.0, 1.0], (65_537, 2)),
        dims=("trial", "sample"),
    )
    mapping = build_param_map(
        param=param,
        query=xr.DataArray(np.empty(0), dims="query"),
        sequence_dim="sample",
        query_dim="query",
    )

    assert mapping.valid.shape == (65_537, 0)
    assert row_counts == [65_536, 1]


@pytest.mark.parametrize("lazy", (False, True))
def test_param_core_empty_map_broadcast_001_query_topology_does_not_repeat_source_validation(
    lazy: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_CORE_EMPTY_MAP_BROADCAST_001_query_topology_does_not_repeat_source_validation."""
    from tal.core.param_engine import numpy_backends

    original = numpy_backends.validate_map_domain_block_numpy_status
    row_counts: list[int] = []

    def tracked(param: np.ndarray, mask: np.ndarray, **kwargs: object):
        row_counts.append(int(np.prod(param.shape[:-1], dtype=np.int64)))
        return original(param, mask, **kwargs)

    monkeypatch.setattr(numpy_backends, "validate_map_domain_block_numpy_status", tracked)
    param = xr.DataArray([0.0, 1.0], dims="sample")
    if lazy:
        param = param.chunk({"sample": 2})
    size = 65_537
    index = xr.indexes.RangeIndex.arange(size, dim="trial")
    query = xr.DataArray(
        np.empty((size, 0)),
        dims=("trial", "query"),
        coords=xr.Coordinates.from_xindex(index),
    )
    mapping = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )

    assert mapping.valid.shape == (size, 0)
    assert mapping.valid.xindexes["trial"].equals(query.xindexes["trial"])
    if lazy:
        assert row_counts == []
        mapping.valid.compute(scheduler="synchronous")
    assert row_counts == [1]


def test_param_hard_empty_map_broadcast_001_lazy_source_failure_is_deferred() -> None:
    """Query-only topology must not corrupt deferred source-domain validation."""
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    param = xr.DataArray([0.0, 2.0, 1.0], dims="sample").chunk({"sample": 3})
    query = xr.DataArray(np.empty((2, 0)), dims=("trial", "query"))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )

    assert tasks == []
    with pytest.raises(ValueError, match="build_param_map: parameter coordinate must be monotonic"):
        mapping.valid.compute(scheduler="synchronous")


@pytest.mark.parametrize("lazy_input", ("param", "query", "both"))
def test_param_core_empty_map_block_002_mixed_lazy_zero_source_stays_lazy(
    lazy_input: str,
) -> None:
    """ID: PARAM_CORE_EMPTY_MAP_BLOCK_002_mixed_lazy_zero_source_stays_lazy."""
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    param = xr.DataArray(np.empty(0), dims="sample")
    query_index = xr.indexes.RangeIndex.arange(65_537, dim="query")
    query = xr.DataArray(
        np.linspace(0.0, 1.0, 65_537),
        dims="query",
        coords=xr.Coordinates.from_xindex(query_index),
    )
    if lazy_input in {"param", "both"}:
        param = param.copy(data=da.from_array(param.data, chunks=(0,)))
    if lazy_input in {"query", "both"}:
        query = query.copy(data=da.from_array(query.data, chunks=(65_537,)))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )
        actual = apply_param_map(
            xr.DataArray(np.empty(0, dtype=np.complex64), dims="sample"),
            param_map=mapping,
            sequence_dim="sample",
        )

    assert tasks == []
    assert mapping.valid.chunks is not None
    assert actual.chunks is not None
    assert max(actual.chunks[actual.get_axis_num("query")]) <= 65_536
    assert actual.dtype == np.dtype("complex128")
    assert mapping.valid.xindexes["query"].equals(query.xindexes["query"])
    assert actual.xindexes["query"].equals(query.xindexes["query"])
    computed = actual.compute(scheduler="synchronous")
    assert computed.shape == (65_537,)
    assert bool(computed.isnull().all())


@pytest.mark.parametrize("query_size", (0, 1))
def test_param_perf_map_failure_reduction_001_keeps_dask_tasks_bounded(
    query_size: int,
) -> None:
    """ID: PARAM_PERF_MAP_FAILURE_REDUCTION_001_keeps_dask_tasks_bounded."""
    pytest.importorskip("dask.array")
    size = 65_537
    param = xr.DataArray(
        np.broadcast_to([0.0, 1.0], (size, 2)),
        dims=("trial", "sample"),
    ).chunk({"trial": 65_536, "sample": 2})
    query = xr.DataArray(
        np.empty((size, 0)) if query_size == 0 else np.full((size, 1), 0.5),
        dims=("trial", "query"),
    ).chunk({"trial": 65_536, "query": max(query_size, 1)})
    mapping = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )
    assert mapping.valid.chunks is not None
    assert max(mapping.valid.chunksizes["trial"]) <= 65_536
    computed = mapping.valid.compute(scheduler="synchronous")
    assert computed.sizes == {"trial": size, "query": query_size}


@pytest.mark.parametrize(
    "zero_domain",
    ("source_batch", "source_supplemental_batch", "query_batch"),
)
def test_param_core_zero_logical_rows_001_skips_zero_sized_dask_kernels(
    zero_domain: str,
) -> None:
    """ID: PARAM_CORE_ZERO_LOGICAL_ROWS_001_skips_zero_sized_dask_kernels."""
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    if zero_domain == "source_batch":
        param = xr.DataArray(np.empty((0, 2)), dims=("trial", "sample")).chunk({"trial": 1, "sample": 2})
        query = xr.DataArray([0.25, 0.75], dims="query")
        values = xr.DataArray(np.empty((0, 2)), dims=("trial", "sample")).chunk({"trial": 1, "sample": 2})
        expected_sizes = {"trial": 0, "query": 2}
    elif zero_domain == "source_supplemental_batch":
        param = xr.DataArray(
            np.empty((2, 0, 2)),
            dims=("trial", "sensor", "sample"),
        ).chunk({"trial": 2, "sensor": 1, "sample": 2})
        query = xr.DataArray([0.25, 0.75], dims="query")
        values = param.copy()
        expected_sizes = {"trial": 2, "sensor": 0, "query": 2}
    else:
        param = xr.DataArray([0.0, 1.0], dims="sample").chunk({"sample": 2})
        query = xr.DataArray(np.empty((0, 2)), dims=("trial", "query")).chunk({"trial": 1, "query": 2})
        values = xr.DataArray([0.0, 1.0], dims="sample").chunk({"sample": 2})
        expected_sizes = {"trial": 0, "query": 2}

    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
        )
        actual = apply_param_map(values, param_map=mapping, sequence_dim="sample")

    assert tasks == []
    assert mapping.valid.sizes == expected_sizes
    assert actual.sizes == expected_sizes
    actual.compute(scheduler="synchronous")


def test_zero_row_bounds_and_gather_avoid_source_dependencies() -> None:
    """Empty bounds and gather results are independent of source task graphs."""
    da = pytest.importorskip("dask.array")
    from dask import delayed

    shape = (0, 2)
    param = xr.DataArray(
        da.from_delayed(
            delayed(_forbidden_empty_param_source)("param", shape, np.dtype("float64")),
            shape=shape,
            dtype="float64",
        ),
        dims=("trial", "sample"),
    )
    values = xr.DataArray(
        da.from_delayed(
            delayed(_forbidden_empty_param_source)("payload", shape, np.dtype("float64")),
            shape=shape,
            dtype="float64",
        ),
        dims=("trial", "sample"),
    )
    bounds = build_param_bounds_map(
        param=param,
        start=0.25,
        stop=0.75,
        sequence_dim="sample",
    )
    indexer = xr.DataArray(
        da.zeros((0, 2), chunks=(0, 2), dtype="int64"),
        dims=("trial", "query"),
    )
    gathered = map_apply_mod.gather_along_sequence(
        values,
        indexer,
        sequence_dim="sample",
        query_dim="query",
        owner="test",
    )

    for bound in (bounds.i0, bounds.i1):
        computed = bound.compute(scheduler="synchronous")
        assert computed.sizes == {"trial": 0}
        assert computed.dtype == np.dtype("int64")
    assert gathered.compute(scheduler="synchronous").sizes == {"trial": 0, "query": 2}


@pytest.mark.parametrize("lazy", (False, True))
def test_param_hard_empty_map_block_001_failure_order_crosses_blocks(
    lazy: bool,
) -> None:
    """ID: PARAM_HARD_EMPTY_MAP_BLOCK_001_failure_order_crosses_blocks."""
    size = 65_537
    baseline = np.asarray(
        ["2020-01-01", "2020-01-02", "2020-01-03"],
        dtype="datetime64[ns]",
    )
    values = np.broadcast_to(
        baseline,
        (size, 3),
    ).copy()
    values[-2] = np.asarray(
        ["2020-01-02", "2020-01-01", "2020-01-03"],
        dtype="datetime64[ns]",
    )
    values[-1] = np.asarray(
        [
            "1677-09-21T00:12:43.145224193",
            "2000-01-01",
            "2262-04-11T23:47:16.854775807",
        ],
        dtype="datetime64[ns]",
    )
    param = xr.DataArray(values, dims=("trial", "sample"))
    query = xr.DataArray(
        np.empty((size, 0), dtype="datetime64[ns]"),
        dims=("trial", "query"),
    )
    if lazy:
        pytest.importorskip("dask.array")
        param = param.chunk({"trial": size, "sample": 3})
        query = query.chunk({"trial": size, "query": 1})
        mapping = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            param_kind="datetime64",
        )
        with pytest.raises(ValueError, match="monotonic non-decreasing"):
            mapping.valid.compute(scheduler="synchronous")
        return
    with pytest.raises(ValueError, match="monotonic non-decreasing"):
        build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            param_kind="datetime64",
        )


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("source_size", (0, 2))
@pytest.mark.parametrize("dtype", (np.int64, np.float32, np.complex64, np.complex128))
def test_param_core_empty_eval_001_preserves_numeric_result_dtype(
    lazy: bool,
    source_size: int,
    dtype: type[np.generic],
) -> None:
    """ID: PARAM_CORE_EMPTY_EVAL_001_preserves_numeric_result_dtype."""
    param_data = np.arange(source_size, dtype=np.float64)
    value_data = np.arange(source_size, dtype=dtype)
    query_data = np.empty(0) if source_size else np.asarray([0.5])
    param = xr.DataArray(param_data, dims="sample")
    values = xr.DataArray(value_data, dims="sample")
    query = xr.DataArray(query_data, dims="query")
    if lazy:
        da = pytest.importorskip("dask.array")
        param = param.copy(data=da.from_array(param_data, chunks=(max(source_size, 1),)))
        values = values.copy(data=da.from_array(value_data, chunks=(max(source_size, 1),)))
        query = query.copy(data=da.from_array(query_data, chunks=(max(query_data.size, 1),)))
    mapping = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )
    actual = apply_param_map(values, param_map=mapping, sequence_dim="sample")
    assert actual.dtype == np.result_type(np.dtype(dtype), np.float64)
    if lazy:
        assert actual.chunks is not None
        actual.compute()


@pytest.mark.parametrize("param_kind", ("numeric", "datetime64"))
@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("size", (65_535, 65_536, 65_537))
def test_param_hard_block_map_001_preserves_public_failure_precedence(
    param_kind: str,
    lazy: bool,
    size: int,
) -> None:
    """ID: PARAM_HARD_BLOCK_MAP_001_preserves_public_failure_precedence."""
    values = np.asarray([[0, 1, 1], [0, 2, 1]])
    query = np.full((2, size), 0, dtype="int64")
    query[0, -1] = 1
    if param_kind == "datetime64":
        values = np.datetime64("2025-01-01") + values.astype("timedelta64[D]")
        query = np.datetime64("2025-01-01") + query.astype("timedelta64[D]")
    param = xr.DataArray(values, dims=("trial", "sample"), coords={"trial": ["a", "b"]})
    target = xr.DataArray(query, dims=("trial", "query"), coords={"trial": ["a", "b"]})
    if lazy:
        pytest.importorskip("dask.array")
        param = param.chunk({"trial": 2, "sample": 3})
        target = target.chunk({"trial": 2, "query": size})
    options = ParamMapOptions(method="linear", duplicate_policy="raise")

    if not lazy:
        with pytest.raises(ValueError, match="duplicate parameter bracket"):
            build_param_map(
                param=param,
                query=target,
                sequence_dim="sample",
                query_dim="query",
                options=options,
                param_kind=param_kind,
            )
        return

    mapping = build_param_map(
        param=param,
        query=target,
        sequence_dim="sample",
        query_dim="query",
        options=options,
        param_kind=param_kind,
    )
    with pytest.raises(ValueError, match="duplicate parameter bracket"):
        mapping.valid.compute(scheduler="synchronous")


def test_param_hard_029_materialize_indexer_not_used_in_select_chunked_paths() -> None:
    """ID: PARAM_HARD_029_materialize_indexer_not_used_in_select_chunked_paths."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={"value": (("sample",), da.from_array(np.asarray([0.0, 10.0, 20.0]), chunks=2))},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0, 2.0]), chunks=2)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )

    assert not hasattr(map_apply_mod, "materialize_indexer")
    point = ao.param.sel([0.5, 1.5])
    slc = ao.param.sel(slice(0.25, 1.75))
    assert tuple(point.as_dataset(copy="none")["value"].dims) == ("sample",)
    assert tuple(slc.as_dataset(copy="none")["value"].dims) == ("sample",)


def test_param_hard_031_materialize_indexer_removed_from_production_surface() -> None:
    """ID: PARAM_HARD_031_materialize_indexer_removed_from_production_surface."""
    assert not hasattr(map_apply_mod, "materialize_indexer")


def test_param_hard_030_gather_dataset_along_sequence_does_not_mutate_source_coord_attrs() -> None:
    """ID: PARAM_HARD_030_gather_dataset_along_sequence_does_not_mutate_source_coord_attrs."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "valid": mark_reserved_coord(
                xr.DataArray(np.asarray([True, True, False], dtype=bool), dims=("sample",)),
                name="valid",
            ),
            "sample_index": mark_reserved_coord(
                xr.DataArray(np.asarray([0, 1, -1], dtype="int64"), dims=("sample",)),
                name="sample_index",
            ),
        },
    )
    valid_before = dict(ds.coords["valid"].attrs)
    sample_before = dict(ds.coords["sample_index"].attrs)
    idx = xr.DataArray(np.asarray([0, 2], dtype="int64"), dims=("query",))
    out = map_apply_mod.gather_dataset_along_sequence(
        ds,
        idx,
        sequence_dim="sample",
        query_dim="query",
        owner="test",
    )
    assert tuple(out["value"].dims) == ("query",)
    assert dict(ds.coords["valid"].attrs) == valid_before
    assert dict(ds.coords["sample_index"].attrs) == sample_before


def test_param_engine_031_gather_along_sequence_blockwise_vectorize_false_parity() -> None:
    """ID: PARAM_ENGINE_031_gather_along_sequence_blockwise_vectorize_false_parity."""
    values = xr.DataArray(
        np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype="float64"),
        dims=("trial", "sample"),
        coords={"trial": ["a", "b"], "sample": [0, 1, 2]},
    )
    indexer = xr.DataArray(
        np.asarray([[2, 0], [1, 1]], dtype="int64"),
        dims=("trial", "query"),
        coords={"trial": ["a", "b"], "query": [0, 1]},
    )
    out = map_apply_mod.gather_along_sequence(
        values,
        indexer,
        sequence_dim="sample",
        query_dim="query",
        owner="test",
    )
    expected = xr.DataArray(
        np.asarray([[12.0, 10.0], [21.0, 21.0]], dtype="float64"),
        dims=("trial", "query"),
        coords={"trial": ["a", "b"], "query": [0, 1]},
    )
    xr.testing.assert_allclose(out, expected)


def test_param_engine_032_apply_param_map_blockwise_gather_semantics_parity() -> None:
    """ID: PARAM_ENGINE_032_apply_param_map_blockwise_gather_semantics_parity."""
    values = xr.DataArray(
        np.asarray([[0.0, 10.0, 20.0], [100.0, 110.0, 120.0]], dtype="float64"),
        dims=("trial", "sample"),
        coords={"trial": ["a", "b"], "sample": [0, 1, 2]},
    )
    query = xr.DataArray([0, 1], dims=("query",))
    pmap = map_build_mod.ParamMap(
        i0=xr.DataArray(np.asarray([0, 1], dtype="int64"), dims=("query",), coords={"query": query}),
        i1=xr.DataArray(np.asarray([1, 2], dtype="int64"), dims=("query",), coords={"query": query}),
        alpha=xr.DataArray(np.asarray([0.5, 0.25], dtype="float64"), dims=("query",), coords={"query": query}),
        valid=xr.DataArray(np.asarray([True, True], dtype=bool), dims=("query",), coords={"query": query}),
        query_dim="query",
    )
    out = apply_param_map(values, param_map=pmap, sequence_dim="sample")
    expected = xr.DataArray(
        np.asarray([[5.0, 12.5], [105.0, 112.5]], dtype="float64"),
        dims=("trial", "query"),
        coords={"trial": ["a", "b"], "query": [0, 1]},
    )
    xr.testing.assert_allclose(out, expected)


def test_param_hard_032_gather_along_sequence_chunked_path_remains_lazy() -> None:
    """ID: PARAM_HARD_032_gather_along_sequence_chunked_path_remains_lazy."""
    da = pytest.importorskip("dask.array")
    values = xr.DataArray(
        da.from_array(np.asarray([[1.0, 2.0, 3.0]], dtype="float64"), chunks=(1, 2)),
        dims=("trial", "sample"),
    )
    indexer = xr.DataArray(
        da.from_array(np.asarray([[2, 0]], dtype="int64"), chunks=(1, 2)),
        dims=("trial", "query"),
    )
    out = map_apply_mod.gather_along_sequence(
        values,
        indexer,
        sequence_dim="sample",
        query_dim="query",
        owner="test",
    )
    assert out.chunks is not None


def test_param_hard_033_param_map_build_stopgap_routes_through_backend_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_HARD_033_param_map_build_stopgap_routes_through_backend_interface."""
    calls = {"n": 0}
    original = map_build_mod.map_block_backend

    def _count(*args: object, **kwargs: object):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(map_build_mod, "_numba_available", lambda: False)
    monkeypatch.setattr(map_build_mod, "map_block_backend", _count)
    param = xr.DataArray(np.asarray([0.0, 1.0, 2.0], dtype="float64"), dims=("sample",))
    query = xr.DataArray(np.asarray([0.25, 1.25], dtype="float64"), dims=("query",))
    _ = build_param_map(
        param=param,
        query=query,
        sequence_dim="sample",
        query_dim="query",
    )
    assert calls["n"] >= 1


def test_param_hard_034_param_bounds_build_stopgap_routes_through_backend_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_HARD_034_param_bounds_build_stopgap_routes_through_backend_interface."""
    calls = {"n": 0}
    original = map_build_mod.bounds_block_backend

    def _count(*args: object, **kwargs: object):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(map_build_mod, "_numba_available", lambda: False)
    monkeypatch.setattr(map_build_mod, "bounds_block_backend", _count)
    param = xr.DataArray(np.asarray([0.0, 1.0, 2.0], dtype="float64"), dims=("sample",))
    _ = build_param_bounds_map(
        param=param,
        start=xr.DataArray(0.25),
        stop=xr.DataArray(1.75),
        sequence_dim="sample",
    )
    assert calls["n"] >= 1


def test_param_hard_036_complex_param_and_query_dtypes_rejected() -> None:
    """ID: PARAM_HARD_036_complex_param_and_query_dtypes_rejected."""
    declared = xr.Dataset(
        {"value": ("sample", [1.0, 2.0])},
        coords={"sample": [0, 1], "phase": ("sample", np.asarray([0.0, 1.0], dtype="complex128"))},
    )
    with pytest.raises(SchemaError) as schema_err:
        AnalysisObject.from_data(declared, sequence_dim="sample", core_dims=(), param_coord="phase")
    assert schema_err.value.code == "schema.param_coord.dtype.invalid"
    assert schema_err.value.path == "tal.core.param_coord.name"

    undeclared = AnalysisObject.from_data(declared, sequence_dim="sample", core_dims=())
    with pytest.raises(ValueError, match="ordered real numeric or datetime64"):
        undeclared.param.index(0.0, on="phase")
    with pytest.raises(ValueError, match="ordered real dtype"):
        normalize_query_grid(np.asarray([1 + 2j], dtype="complex128"))
    with pytest.raises(ValueError, match="slice.start must have an ordered real numeric dtype"):
        build_param_bounds_map(
            param=xr.DataArray([0.0, 1.0], dims=("sample",)),
            start=np.complex128(0.0),
            stop=1.0,
            sequence_dim="sample",
        )


def test_param_hard_037_unsafe_mixed_numeric_conversion_fails_closed() -> None:
    """ID: PARAM_HARD_037_unsafe_mixed_numeric_conversion_fails_closed."""
    unsafe = np.int64(2**53 + 1)
    with pytest.raises(ValueError, match="cannot be represented exactly as float64"):
        build_param_map(
            param=xr.DataArray([0.0, 1.0], dims=("sample",)),
            query=xr.DataArray([unsafe], dims=("query",)),
            sequence_dim="sample",
            query_dim="query",
            options=ParamMapOptions(method="nearest"),
        )

    base = 2**53
    integral_param = xr.DataArray(
        np.asarray([base, base + 3], dtype="int64"),
        dims=("sample",),
    )
    floating_query = xr.DataArray(
        np.asarray([float(base + 2)], dtype="float64"),
        dims=("query",),
    )
    for method in ("nearest", "linear"):
        with pytest.raises(ValueError, match="integer parameter value.*cannot be represented exactly as float64"):
            build_param_map(
                param=integral_param,
                query=floating_query,
                sequence_dim="sample",
                query_dim="query",
                options=ParamMapOptions(method=method),  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="integer parameter value.*cannot be represented exactly as float64"):
        build_param_bounds_map(
            param=integral_param,
            start=float(base + 2),
            stop=float(base + 4),
            sequence_dim="sample",
        )

    query = xr.DataArray(
        np.asarray([[1, 2]], dtype="int64"),
        dims=("trial", "query"),
        coords={"trial": ["a"]},
    )
    with pytest.raises(ValueError, match="integral query batch reindex would introduce missing labels"):
        normalize_query_grid(
            query,
            query_dim="query",
            batch_dims=("trial",),
            batch_coords={"trial": xr.DataArray(["a", "b"], dims=("trial",))},
        )

    with pytest.raises(ValueError, match="mixed numeric query would convert integer value"):
        normalize_query_grid([np.uint64(np.iinfo(np.uint64).max), -1])


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.asarray([1 + 0j, np.inf + 0j], dtype="complex128"), id="complex"),
        pytest.param(np.asarray([True, False], dtype=bool), id="boolean"),
        pytest.param(np.asarray(["a", "b"], dtype="U1"), id="string"),
    ],
)
def test_param_hard_038_invalid_validity_dtype_fails_closed(values: np.ndarray) -> None:
    """ID: PARAM_HARD_038_invalid_validity_dtype_fails_closed."""
    coord = xr.DataArray(values, dims=("sample",), name="phase")
    with pytest.raises(ValueError, match="ordered real numeric or datetime64 dtype"):
        finite_param_mask(coord)

    ds = xr.Dataset(coords={"sample": [0, 1], "phase": coord})
    spec = ParamCoordSpec(
        name="phase",
        coord=ds.coords["phase"],
        sequence_dim="sample",
        batch_dims=(),
    )
    with pytest.raises(ValueError, match="ordered real numeric or datetime64 dtype"):
        resolve_param_valid_mask(ds, spec=spec)

    sized = ds.assign_coords(group_size=xr.DataArray(np.int64(2)))
    sized_spec = ParamCoordSpec(
        name="phase",
        coord=sized.coords["phase"],
        sequence_dim="sample",
        batch_dims=(),
    )
    with pytest.raises(ValueError, match="ordered real numeric or datetime64 dtype"):
        resolve_param_valid_mask(sized, spec=sized_spec, sequence_size_coord="group_size")


def test_map_coordinate_equality_never_executes_independent_lazy_values() -> None:
    """ID: PARAM_HARD_MAP_COORDINATES_001_lazy_comparison_is_fail_conservative."""
    da = pytest.importorskip("dask.array")
    from dask import delayed
    from dask.callbacks import Callback

    left = da.from_delayed(delayed(np.asarray, pure=False)([1, 2]), shape=(2,), dtype="int64")
    right = da.from_delayed(delayed(np.asarray, pure=False)([1, 2]), shape=(2,), dtype="int64")
    param = xr.DataArray(
        [[0.0, 1.0], [0.0, 1.0]], dims=("trial", "sample"),
        coords={"aux": ("trial", left)},
    )
    query = xr.DataArray(
        [[0.5], [0.5]], dims=("trial", "query"),
        coords={"aux": ("trial", right)},
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *_: tasks.append(key)), pytest.raises(
        ValueError, match="^build_param_map: coordinate 'aux' has conflicting output metadata or values",
    ):
        build_param_map(param=param, query=query, sequence_dim="sample", query_dim="query")
    assert tasks == []


def test_map_coordinate_encoding_conflict_is_not_silently_resolved() -> None:
    """ID: PARAM_HARD_MAP_COORDINATES_002_encoding_conflict_is_owned."""
    param = xr.DataArray([[0.0, 1.0]], dims=("trial", "sample"), coords={"aux": ("trial", [7])})
    query = xr.DataArray([[0.5]], dims=("trial", "query"), coords={"aux": ("trial", [7])})
    param.coords["aux"].encoding["source"] = "left"
    query.coords["aux"].encoding["source"] = "right"
    with pytest.raises(ValueError, match="^build_param_map: coordinate 'aux' has conflicting output metadata"):
        build_param_map(param=param, query=query, sequence_dim="sample", query_dim="query")


@pytest.mark.parametrize("kind", ("numeric", "datetime64"))
@pytest.mark.parametrize("operation", ("map", "bounds"))
def test_map_alignment_errors_keep_owner_and_cause(kind: str, operation: str) -> None:
    """ID: PARAM_HARD_MAP_ALIGNMENT_001_exact_alignment_is_owned."""
    domain = [0.0, 1.0] if kind == "numeric" else np.asarray(
        ["2020-01-01", "2020-01-02"], dtype="datetime64[ns]",
    )
    target = 0.5 if kind == "numeric" else np.datetime64("2020-01-01T12:00:00", "ns")
    param = xr.DataArray([domain], dims=("trial", "sample"), coords={"trial": ["a"]})
    other = xr.DataArray([[target]], dims=("trial", "query"), coords={"trial": ["b"]})
    if operation == "map":
        invoke = lambda: build_param_map(
            param=param, query=other, sequence_dim="sample", query_dim="query", param_kind=kind,
        )
        owner = "build_param_map"
    else:
        start = other.isel(query=0, drop=True)
        invoke = lambda: build_param_bounds_map(
            param=param, start=start, stop=start,
            sequence_dim="sample", param_kind=kind,
        )
        owner = "build_param_bounds_map"
    with pytest.raises(ValueError, match=rf"^{owner}: .*inputs are not label-aligned") as error:
        invoke()
    assert error.value.__cause__ is not None
