from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core.analysis_object import AnalysisObject
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.linalg import Vector3
from tal.spatial import Position


def _ao_numeric_bool_object() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "signal": (("trial", "sample"), np.array([[1.0, np.nan, 5.0], [2.0, 3.0, np.nan]], dtype=float)),
            "energy": (("trial", "sample"), np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=float)),
            "flag": (("trial", "sample"), np.array([[True, False, True], [False, False, True]], dtype=bool)),
            "label": (("trial", "sample"), np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=object)),
        },
        coords={
            "trial": np.array(["t0", "t1"], dtype=object),
            "sample": np.array([0, 1, 2], dtype=int),
            "time_s": (("trial", "sample"), np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=float)),
            "group_size": (("trial",), np.array([2, 1], dtype=np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
        sequence_size_coord="group_size",
    )


def test_reduce_core_p9c_001_ao_reducer_surface_restored_with_deterministic_dim_semantics() -> None:
    """ID: REDUCE_CORE_P9C_001_ao_reducer_surface_restored_with_deterministic_dim_semantics."""
    ao = _ao_numeric_bool_object()
    out = ao.mean(validate=True)
    assert isinstance(out, AnalysisObject)
    assert set(out.as_dataset(copy="none").data_vars) == {"signal", "energy"}
    assert out.as_dataset(copy="none")["signal"].dims == ()
    assert out.as_dataset(copy="none")["energy"].dims == ()


def test_reduce_core_p9c_001b_reducer_preserves_declared_roles_when_sequence_dim_removed() -> None:
    """ID: REDUCE_CORE_P9C_001b_reducer_preserves_declared_roles_when_sequence_dim_removed."""
    ao = _ao_numeric_bool_object()
    out = ao.min(dim="sample", validate=True)
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.as_dataset(copy="none"))
    assert declared is True
    assert sequence_dim is None
    assert batch_dims == ("trial",)
    assert core_dims == ()
    assert read_param_coord_name(out.as_dataset(copy="none")) is None
    assert read_sequence_size_coord_name(out.as_dataset(copy="none")) is None


@pytest.mark.parametrize("name", ["sum", "std", "var", "median", "min", "max", "count", "any", "all"])
def test_reduce_core_p9c_001a_all_reducer_methods_exist(name: str) -> None:
    """ID: REDUCE_CORE_P9C_008_all_reducer_methods_surface_invocable."""
    ao = _ao_numeric_bool_object()
    fn = getattr(ao, name)
    out = fn(dim="sample", validate=True)
    assert isinstance(out, AnalysisObject)


def test_reduce_core_p9c_006_weight_forms_and_exact_alignment_rules_are_enforced() -> None:
    """ID: REDUCE_CORE_P9C_006_weight_forms_and_exact_alignment_rules_are_enforced."""
    ao = _ao_numeric_bool_object()
    w_nd = np.array([1.0, 2.0, 3.0], dtype=float)
    out = ao.mean(dim="sample", weights=w_nd, validate=True).as_dataset(copy="none")
    np.testing.assert_allclose(out["energy"].to_numpy(), np.array([50.0 / 3.0, 40.0], dtype=float))

    w_map = {
        "sample": np.array([1.0, 2.0, 3.0], dtype=float),
        "trial": np.array([1.0, 2.0], dtype=float),
    }
    out_map = ao.sum(dim=("trial", "sample"), weights=w_map, validate=True).as_dataset(copy="none")
    assert out_map["energy"].ndim == 0

    bad = xr.DataArray(np.array([1.0, 1.0, 1.0], dtype=float), dims=("sample",), coords={"sample": [0, 1, 99]})
    with pytest.raises(ValueError, match="aligned|exact|labels"):
        _ = ao.mean(dim="sample", weights=bad)


def test_reduce_core_p9c_003_weighted_mean_is_explicit_and_not_silently_ignored() -> None:
    """ID: REDUCE_CORE_P9C_003_weighted_mean_is_explicit_and_not_silently_ignored."""
    ao = _ao_numeric_bool_object()
    unweighted = ao.mean(dim="sample", validate=True).as_dataset(copy="none")["energy"].to_numpy()
    weighted = ao.mean(dim="sample", weights=np.array([1.0, 2.0, 1.0], dtype=float), validate=True).as_dataset(copy="none")[
        "energy"
    ].to_numpy()
    assert not np.allclose(unweighted, weighted)
    np.testing.assert_allclose(weighted, np.array([50.0 / 3.0, 40.0], dtype=float))


def test_reduce_core_p9c_005_dim_none_reduces_over_reducible_non_component_dims() -> None:
    """ID: REDUCE_CORE_P9C_005_dim_none_reduces_over_reducible_non_component_dims."""
    ao = _ao_numeric_bool_object()
    out_default = ao.sum(validate=True).as_dataset(copy="none")
    assert out_default["signal"].dims == ()
    assert out_default["energy"].dims == ()

    out_explicit = ao.sum(dim="sample", validate=True).as_dataset(copy="none")
    assert out_explicit["signal"].dims == ("trial",)
    assert out_explicit["energy"].dims == ("trial",)


def test_reduce_hard_p9c_004_negative_weights_are_rejected_by_default() -> None:
    """ID: REDUCE_HARD_P9C_004_negative_weights_are_rejected_by_default."""
    ao = _ao_numeric_bool_object()
    with pytest.raises(ValueError, match="negative weights"):
        _ = ao.mean(dim="sample", weights=np.array([1.0, -1.0, 1.0], dtype=float))


def test_reduce_hard_p9c_005_skipna_false_fails_closed_on_nonfinite_valid_prefix_weights() -> None:
    """ID: REDUCE_HARD_P9C_005_skipna_false_fails_closed_on_nonfinite_valid_prefix_weights."""
    ao = _ao_numeric_bool_object()
    with pytest.raises(ValueError, match="finite weights"):
        _ = ao.mean(dim="sample", weights=np.array([1.0, np.nan, 1.0], dtype=float), skipna=False)


def test_reduce_core_p9c_007_any_all_include_numeric_and_bool_var_domain() -> None:
    """ID: REDUCE_CORE_P9C_007_any_all_include_numeric_and_bool_var_domain."""
    ao = _ao_numeric_bool_object()
    any_out = ao.any(dim="sample", validate=True).as_dataset(copy="none")
    all_out = ao.all(dim="sample", validate=True).as_dataset(copy="none")
    assert set(any_out.data_vars) == {"signal", "energy", "flag"}
    assert set(all_out.data_vars) == {"signal", "energy", "flag"}


def test_reduce_hard_p9c_006_unsupported_weighted_reducers_fail_closed() -> None:
    """ID: REDUCE_HARD_P9C_006_unsupported_weighted_reducers_fail_closed."""
    ao = _ao_numeric_bool_object()
    for name in ("var", "std", "median", "min", "max", "count", "any", "all"):
        fn = getattr(ao, name)
        with pytest.raises(ValueError, match="weights are supported only"):
            _ = fn(dim="sample", weights=np.array([1.0, 1.0, 1.0], dtype=float))


def test_reduce_hard_p9c_007_fail_closed_when_no_eligible_vars() -> None:
    """ID: REDUCE_HARD_P9C_007_fail_closed_when_no_eligible_vars."""
    ds = xr.Dataset(
        data_vars={"label": (("sample",), np.array(["a", "b", "c"], dtype=object))},
        coords={"sample": np.array([0, 1, 2], dtype=int)},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=())
    with pytest.raises(ValueError, match="numeric"):
        _ = ao.mean(dim="sample")
    with pytest.raises(ValueError, match="numeric or boolean"):
        _ = ao.any(dim="sample")


def test_reduce_core_p9c_009_inherited_typed_reducers_demote_to_analysisobject() -> None:
    """ID: REDUCE_CORE_P9C_009_inherited_typed_reducers_demote_to_analysisobject."""
    pos_arr = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        dims=("sample", "axis"),
        coords={"sample": np.array([0, 1], dtype=int), "axis": np.array(["x", "y", "z"], dtype=object)},
        name="position",
    )
    pos = Position(
        AnalysisObject.from_data(
            pos_arr.to_dataset(name="position"),
            sequence_dim="sample",
            core_dims=("axis",),
            validate=True,
        ).as_dataset(copy="none")
    )
    pos_out = pos.mean(dim="sample", validate=True)
    assert type(pos_out) is AnalysisObject
    assert not isinstance(pos_out, Position)

    vec_arr = xr.DataArray(
        np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=float),
        dims=("sample", "trial", "axis"),
        coords={
            "sample": np.array([0, 1], dtype=int),
            "trial": np.array(["t0"], dtype=object),
            "axis": np.array(["x", "y", "z"], dtype=object),
        },
        name="x",
    )
    vec = Vector3(
        AnalysisObject.from_data(
            vec_arr.to_dataset(name="x"),
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("axis",),
            validate=True,
        )
    )
    vec_out = vec.mean(dim="sample", validate=True)
    assert type(vec_out) is AnalysisObject
    assert not isinstance(vec_out, Vector3)
