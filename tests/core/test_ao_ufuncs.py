from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal import ufuncs as tal_ufuncs
from tal.core import AnalysisObject
from tal.core.event_ops.types import Condition
from tal.core.ufunc_ops.api import apply_unary_ufunc
from tal.core.ufunc_ops.finalize import finalize_unary_ao_result


def _ao(values: np.ndarray, *, name: str = "x") -> AnalysisObject:
    ds = xr.Dataset(
        {name: (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )


def _ao_missing_sequence_declared(values: np.ndarray, *, name: str = "x") -> AnalysisObject:
    ds = xr.Dataset(
        {name: (("trial", "axis"), values)},
        coords={
            "sample": ("sample", np.arange(2, dtype=np.int64)),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.arange(values.shape[1], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=False,
    )


def _undeclared_ao(values: np.ndarray, *, name: str = "x") -> AnalysisObject:
    ds = xr.Dataset(
        {name: (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject(ds)


def _multivar_ao(values: np.ndarray) -> AnalysisObject:
    ds = xr.Dataset(
        {
            "x": (("sample", "trial", "axis"), values),
            "y": (("sample", "trial", "axis"), values + 1.0),
        },
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )


def _param_ao(
    values: np.ndarray,
    *,
    sample_labels: np.ndarray,
    param_values: np.ndarray,
    name: str = "x",
) -> AnalysisObject:
    ds = xr.Dataset(
        {name: (("sample", "axis"), values)},
        coords={
            "sample": sample_labels,
            "axis": np.arange(values.shape[1], dtype=np.int64),
            "time_s": ("sample", param_values),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
        param_coord="time_s",
        validate=True,
    )


def _xarray_ufunc_wrapper_names() -> set[str]:
    out: set[str] = set()
    for name in dir(xr.ufuncs):
        if name.startswith("_"):
            continue
        if type(getattr(xr.ufuncs, name)).__name__ in {"_unary_ufunc", "_binary_ufunc"}:
            out.add(name)
    return out


def _roles(ds: xr.Dataset) -> tuple[bool, str | None, tuple[str, ...], tuple[str, ...]]:
    from tal.core.schema_read import read_roles

    return read_roles(ds)


def test_ao_ufunc_core_001_registry_matches_xarray_wrappers() -> None:
    """ID: AO_UFUNC_CORE_001_registry_matches_xarray_wrappers."""
    expected = _xarray_ufunc_wrapper_names()
    exported = set(tal_ufuncs.__all__)
    assert exported == expected
    for name in expected:
        assert callable(getattr(tal_ufuncs, name))


def test_ao_ufunc_core_002_unary_ao_value_parity_and_schema_truthful() -> None:
    """ID: AO_UFUNC_CORE_002_unary_ao_value_parity_and_schema_truthful."""
    ao = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    out = tal_ufuncs.sin(ao)
    expected = xr.ufuncs.sin(ao.unsafe_data)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected["x"])
    assert _roles(out.unsafe_data)[3] == ("axis",)


def test_ao_ufunc_core_003_binary_ao_value_parity_and_schema_truthful() -> None:
    """ID: AO_UFUNC_CORE_003_binary_ao_value_parity_and_schema_truthful."""
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 5.0)
    out = tal_ufuncs.multiply(left, right)
    expected = xr.ufuncs.multiply(left.unsafe_data, right.unsafe_data)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected["x"])
    assert _roles(out.unsafe_data)[3] == ("axis",)


def test_ao_ufunc_core_004_non_ao_passthrough_remains_xarray() -> None:
    """ID: AO_UFUNC_CORE_004_non_ao_passthrough_remains_xarray."""
    arr = xr.DataArray(np.arange(5, dtype=float), dims=("sample",))
    out = tal_ufuncs.exp(arr)
    expected = xr.ufuncs.exp(arr)
    assert isinstance(out, xr.DataArray)
    xr.testing.assert_identical(out, expected)


def test_ao_ufunc_core_005_comparison_non_ao_passthrough_returns_native_boolean_xarray() -> None:
    """ID: AO_UFUNC_CORE_005_comparison_non_ao_passthrough_returns_native_boolean_xarray."""
    arr = xr.DataArray(np.arange(5, dtype=float), dims=("sample",))
    out = tal_ufuncs.less(arr, 2.0)
    expected = xr.ufuncs.less(arr, 2.0)
    assert isinstance(out, xr.DataArray)
    assert out.dtype == np.dtype(bool)
    xr.testing.assert_identical(out, expected)


def test_ao_ufunc_core_006_comparison_ao_inputs_return_condition() -> None:
    """ID: AO_UFUNC_CORE_006_comparison_ao_inputs_return_condition."""
    ao = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    out = tal_ufuncs.less(ao, 2.0)
    assert isinstance(out, Condition)


def test_ao_ufunc_core_007_finalize_dataarray_preserves_single_source_carrier_name() -> None:
    """ID: AO_UFUNC_CORE_007_finalize_dataarray_preserves_single_source_carrier_name."""
    source = _ao(np.arange(12, dtype=float).reshape(2, 2, 3), name="carrier")
    carrier = source.unsafe_data["carrier"]
    result = xr.DataArray(np.sin(carrier.values), dims=carrier.dims, coords=carrier.coords)
    out = finalize_unary_ao_result(source, result, owner="test", validate=True)
    assert list(out.unsafe_data.data_vars) == ["carrier"]


def test_ao_ufunc_core_008_finalize_dataarray_ambiguous_source_falls_back_to_datavar() -> None:
    """ID: AO_UFUNC_CORE_008_finalize_dataarray_ambiguous_source_falls_back_to_datavar."""
    source = _multivar_ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    left = source.unsafe_data["x"]
    result = xr.DataArray(np.cos(left.values), dims=left.dims, coords=left.coords)
    out = finalize_unary_ao_result(source, result, owner="test", validate=True)
    assert list(out.unsafe_data.data_vars) == ["datavar"]


def test_ao_ufunc_core_009_finalize_dataarray_preserves_explicit_kernel_name() -> None:
    """ID: AO_UFUNC_CORE_009_finalize_dataarray_preserves_explicit_kernel_name."""
    source = _multivar_ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    left = source.unsafe_data["x"]
    result = xr.DataArray(
        np.cos(left.values),
        dims=left.dims,
        coords=left.coords,
        name="kernel_name",
    )
    out = finalize_unary_ao_result(source, result, owner="test", validate=True)
    assert list(out.unsafe_data.data_vars) == ["kernel_name"]


def test_ao_ufunc_hard_001_unknown_name_fail_closed() -> None:
    """ID: AO_UFUNC_HARD_001_unknown_name_fail_closed."""
    ao = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    with pytest.raises(ValueError, match="unknown xarray ufunc"):
        _ = apply_unary_ufunc("not_a_real_ufunc", ao, owner="test")


def test_ao_ufunc_hard_002_binary_ao_subclass_left_wins() -> None:
    """ID: AO_UFUNC_HARD_002_binary_ao_subclass_left_wins."""

    class MyAO(AnalysisObject):
        pass

    left = MyAO(_ao(np.arange(12, dtype=float).reshape(2, 2, 3)).unsafe_data)
    right = _ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 2.0) / 7.0)
    out = tal_ufuncs.add(left, right)
    expected = xr.ufuncs.add(left.unsafe_data, right.unsafe_data)
    assert isinstance(out, MyAO)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected["x"])


def test_ao_ufunc_hard_003_chunked_ao_preserves_laziness() -> None:
    """ID: AO_UFUNC_HARD_003_chunked_ao_preserves_laziness."""
    pytest.importorskip("dask.array")
    ao = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    chunked = AnalysisObject.from_data(
        ao.unsafe_data.chunk({"sample": 1}),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    out = tal_ufuncs.cos(chunked)
    assert out.unsafe_data["x"].chunks is not None


def test_bcast_core_025_ufunc_arithmetic_accepts_broadcast_intent_semantic_path() -> None:
    """ID: BCAST_CORE_025_ufunc_arithmetic_accepts_broadcast_intent_semantic_path."""
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao_missing_sequence_declared((np.arange(6, dtype=float).reshape(2, 3) + 1.0) / 5.0)
    out = tal_ufuncs.add(left.b(), right)
    expected_right = right.unsafe_data["x"].expand_dims(sample=left.unsafe_data.coords["sample"]).transpose(
        "sample",
        "trial",
        "axis",
    )
    expected = xr.ufuncs.add(left.unsafe_data["x"], expected_right)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)


def test_bcast_core_029_semantic_default_enabled_for_ao_ufunc_arithmetic_paths() -> None:
    """ID: BCAST_CORE_029_semantic_default_enabled_for_ao_ufunc_arithmetic_paths."""
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao_missing_sequence_declared((np.arange(6, dtype=float).reshape(2, 3) + 2.0) / 9.0)
    out = tal_ufuncs.add(left, right)
    expected_right = right.unsafe_data["x"].expand_dims(sample=left.unsafe_data.coords["sample"]).transpose(
        "sample",
        "trial",
        "axis",
    )
    expected = xr.ufuncs.add(left.unsafe_data["x"], expected_right)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)


def test_bcast_core_027_comparison_and_logical_families_accept_broadcast_intent() -> None:
    """ID: BCAST_CORE_027_comparison_and_logical_families_accept_broadcast_intent."""
    ao = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    cond_left = tal_ufuncs.less(ao.b(), 4.0)
    cond_right = tal_ufuncs.greater(ao, 1.0)
    assert isinstance(cond_left, Condition)
    assert isinstance(tal_ufuncs.logical_and(cond_left, cond_right), Condition)


def test_bcast_hard_018_ufunc_broadcast_intent_undeclared_roles_fail_closed() -> None:
    """ID: BCAST_HARD_018_ufunc_broadcast_intent_undeclared_roles_fail_closed."""
    left = _undeclared_ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 2.0) / 9.0)
    with pytest.raises(ValueError, match=r"^tal\.ufuncs\.add: .*matching core dims"):
        _ = tal_ufuncs.add(left.b(), right)


def test_ao_ufunc_core_011_undeclared_binary_defaults_to_core_only_semantics() -> None:
    """ID: AO_UFUNC_CORE_011_undeclared_binary_defaults_to_core_only_semantics."""
    left = _undeclared_ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _undeclared_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 7.0)
    out = tal_ufuncs.add(left, right)
    expected = xr.ufuncs.add(left.unsafe_data["x"], right.unsafe_data["x"])
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)
    roles_declared, sequence_dim, batch_dims, core_dims = _roles(out.unsafe_data)
    assert roles_declared is True
    assert sequence_dim is None
    assert batch_dims == ()
    assert core_dims == ("sample", "trial", "axis")


def test_bcast_hard_019_ufunc_broadcast_intent_multivar_layout_fail_closed() -> None:
    """ID: BCAST_HARD_019_ufunc_broadcast_intent_multivar_layout_fail_closed."""
    left = _multivar_ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 3.0) / 11.0)
    with pytest.raises(ValueError, match=r"^tal\.ufuncs\.add: operand 0 must contain exactly one data variable"):
        _ = tal_ufuncs.add(left.b(), right)


def test_bcast_hard_045_ao_ufunc_binary_paths_fail_closed_on_strict_core_dim_mismatch_before_finalize() -> None:
    """ID: BCAST_HARD_045_ao_ufunc_binary_paths_fail_closed_on_strict_core_dim_mismatch_before_finalize."""
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right_ds = xr.Dataset(
        {"x": (("sample", "trial", "beta"), np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "beta": np.arange(3, dtype=np.int64),
        },
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("beta",),
        validate=True,
    )
    with pytest.raises(ValueError, match=r"^tal\.ufuncs\.add: .*core_policy='strict'"):
        _ = tal_ufuncs.add(left, right)


def test_ufunc_alignment_intent_param_primary_path_supported() -> None:
    left = _param_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    right = _param_ao(
        np.arange(6, dtype=float).reshape(3, 2) + 1.0,
        sample_labels=np.asarray(["a", "b", "c"], dtype=object),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    out = tal_ufuncs.add(left.a(on="param", sequence_join=None), right)
    np.testing.assert_array_equal(out.unsafe_data.coords["sample"].values, np.asarray([0, 1, 2], dtype=np.int64))


def test_ufunc_alignment_intent_conflict_fail_closed() -> None:
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 2.0)
    with pytest.raises(ValueError, match=r"^tal\.ufuncs\.add: conflicting alignment intents"):
        _ = tal_ufuncs.add(left.a(on="sequence"), right.a(on="param", sequence_join=None))
