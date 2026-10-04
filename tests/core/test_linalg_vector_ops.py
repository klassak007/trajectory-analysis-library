from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.linalg import Array, Matrix, Vector, dot, norm


def _vector_ao(values: np.ndarray, *, axis: str) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", "trial", axis), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            axis: np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(axis,),
        validate=True,
    )


def _matrix_ao(values: np.ndarray, *, row: str, col: str) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", "trial", row, col), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row: np.arange(values.shape[2], dtype=np.int64),
            col: np.arange(values.shape[3], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=True,
    )


def _core_dims(ds: xr.Dataset) -> tuple[str, ...]:
    from tal.core.schema_read import read_roles

    return read_roles(ds)[3]


def _expected_norm(values: xr.DataArray, *, axis: str, ord: int | float | None) -> xr.DataArray:
    return xr.apply_ufunc(
        np.linalg.norm,
        values,
        input_core_dims=[[axis]],
        output_core_dims=[[]],
        vectorize=True,
        dask="allowed",
        kwargs={"ord": ord},
    ).rename("datavar")


def test_linalg_vector_001_dot_norm_core_dim_semantics() -> None:
    """ID: LINALG_VECTOR_001_dot_norm_core_dim_semantics."""
    left = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    right = Vector(_vector_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 4.0, axis="axis"))
    out_dot = dot(left, right)
    out_norm = norm(left)
    assert _core_dims(out_dot.as_dataset(copy="none")) == ()
    assert _core_dims(out_norm.as_dataset(copy="none")) == ()
    assert "datavar" in out_dot.as_dataset(copy="none").data_vars
    assert "datavar" in out_norm.as_dataset(copy="none").data_vars
    assert not any("_dot_" in name for name in out_dot.as_dataset(copy="none").data_vars)
    assert not any("_norm" in name for name in out_norm.as_dataset(copy="none").data_vars)


def test_linalg_core_017_vector_dot_semantics_values_parity() -> None:
    """ID: LINALG_CORE_017_vector_dot_semantics_values_parity."""
    left = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    right = Vector(_vector_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 3.0, axis="axis"))
    out = dot(left, right)
    out_method = left.dot(right)
    expected = xr.dot(left.as_dataset(copy="none")["x"], right.as_dataset(copy="none")["x"], dim=["axis"]).rename("datavar")
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)
    xr.testing.assert_identical(out.as_dataset(copy="none"), out_method.as_dataset(copy="none"))


def test_linalg_core_018_vector_norm_semantics_values_parity() -> None:
    """ID: LINALG_CORE_018_vector_norm_semantics_values_parity."""
    left = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    out = norm(left, ord=2)
    out_method = left.norm(ord=2)
    expected = _expected_norm(left.as_dataset(copy="none")["x"], axis="axis", ord=2)
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)
    xr.testing.assert_identical(out.as_dataset(copy="none"), out_method.as_dataset(copy="none"))


def test_linalg_hard_048_dot_requires_vector_operands() -> None:
    """ID: LINALG_HARD_048_dot_requires_vector_operands."""
    left = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="row", col="col"))
    right = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="row"))
    with pytest.raises(ValueError, match="left operand must have exactly one core dim"):
        _ = dot(left, right)


def test_linalg_hard_049_dot_requires_matching_vector_core_dim_name() -> None:
    """ID: LINALG_HARD_049_dot_requires_matching_vector_core_dim_name."""
    left = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="left_axis"))
    right = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="right_axis"))
    with pytest.raises(ValueError, match="matching vector core dim names"):
        _ = dot(left, right)


def test_linalg_hard_050_norm_requires_vector_operand() -> None:
    """ID: LINALG_HARD_050_norm_requires_vector_operand."""
    left = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="row", col="col"))
    with pytest.raises(ValueError, match="operand must have exactly one core dim"):
        _ = norm(left)


def test_linalg_hard_051_dot_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_051_dot_plain_ao_inputs_fallback_to_array."""
    left_ao = _vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis")
    right_ao = _vector_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 2.0, axis="axis")
    out = dot(left_ao, right_ao)
    out_ds = dot(left_ao.as_dataset(copy="none"), right_ao.as_dataset(copy="none"))
    assert type(out) is Array
    assert type(out_ds) is Array


def test_linalg_hard_052_norm_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_052_norm_plain_ao_inputs_fallback_to_array."""
    left_ao = _vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis")
    out = norm(left_ao)
    out_ds = norm(left_ao.as_dataset(copy="none"))
    assert type(out) is Array
    assert type(out_ds) is Array


def test_linalg_hard_053_dot_builtin_wrapper_type_routing_preserves_values() -> None:
    """ID: LINALG_HARD_053_dot_builtin_wrapper_type_routing_preserves_values."""
    left = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    right = Vector(_vector_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 2.0) / 7.0, axis="axis"))
    out = dot(left, right)
    expected = xr.dot(left.as_dataset(copy="none")["x"], right.as_dataset(copy="none")["x"], dim=["axis"]).rename("datavar")
    assert type(out) is Array
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)


def test_linalg_hard_054_norm_builtin_wrapper_type_routing_preserves_values() -> None:
    """ID: LINALG_HARD_054_norm_builtin_wrapper_type_routing_preserves_values."""
    left = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    out = norm(left, ord=1)
    expected = _expected_norm(left.as_dataset(copy="none")["x"], axis="axis", ord=1)
    assert type(out) is Array
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)


def test_linalg_hard_057_dot_custom_array_subclass_left_wins_policy_preserved() -> None:
    """ID: LINALG_HARD_057_dot_custom_array_subclass_left_wins_policy_preserved."""

    class MyArray(Array):
        pass

    left = MyArray(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    right = Vector(_vector_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 2.0) / 5.0, axis="axis"))
    out = dot(left, right)
    expected = xr.dot(left.as_dataset(copy="none")["x"], right.as_dataset(copy="none")["x"], dim=["axis"]).rename("datavar")
    assert isinstance(out, MyArray)
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)


def test_linalg_hard_058_norm_custom_array_subclass_left_wins_policy_preserved() -> None:
    """ID: LINALG_HARD_058_norm_custom_array_subclass_left_wins_policy_preserved."""

    class MyArray(Array):
        pass

    left = MyArray(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    out = norm(left, ord=2)
    expected = _expected_norm(left.as_dataset(copy="none")["x"], axis="axis", ord=2)
    assert isinstance(out, MyArray)
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)


def test_linalg_hard_070_dot_custom_vector_subclass_arity_change_falls_back_to_array() -> None:
    """ID: LINALG_HARD_070_dot_custom_vector_subclass_arity_change_falls_back_to_array."""

    class MyVector(Vector):
        pass

    left = MyVector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    right = MyVector(_vector_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 2.0) / 5.0, axis="axis"))
    out = dot(left, right)
    expected = xr.dot(left.as_dataset(copy="none")["x"], right.as_dataset(copy="none")["x"], dim=["axis"]).rename("datavar")
    assert type(out) is Array
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)


def test_linalg_hard_071_norm_custom_vector_subclass_arity_change_falls_back_to_array() -> None:
    """ID: LINALG_HARD_071_norm_custom_vector_subclass_arity_change_falls_back_to_array."""

    class MyVector(Vector):
        pass

    left = MyVector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    out = norm(left, ord=2)
    expected = _expected_norm(left.as_dataset(copy="none")["x"], axis="axis", ord=2)
    assert type(out) is Array
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)


def test_linalg_hard_055_norm_ord_validation_fail_closed() -> None:
    """ID: LINALG_HARD_055_norm_ord_validation_fail_closed."""
    left = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    with pytest.raises(TypeError, match="ord must be int | float | None"):
        _ = norm(left, ord="fro")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ord must not be NaN"):
        _ = norm(left, ord=float("nan"))


def test_linalg_hard_056_dot_norm_chunked_inputs_preserve_laziness() -> None:
    """ID: LINALG_HARD_056_dot_norm_chunked_inputs_preserve_laziness."""
    pytest.importorskip("dask.array")
    left = _vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis")
    right = _vector_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 3.0, axis="axis")
    left_chunked = AnalysisObject.from_data(
        left.as_dataset(copy="none").chunk({"sample": 1}),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    right_chunked = AnalysisObject.from_data(
        right.as_dataset(copy="none").chunk({"sample": 1}),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    out_dot = dot(left_chunked, right_chunked)
    out_norm = norm(left_chunked, ord=2)
    assert out_dot.as_dataset(copy="none")["datavar"].chunks is not None
    assert out_norm.as_dataset(copy="none")["datavar"].chunks is not None


def test_linalg_hard_090_norm_kernel_vectorize_false_semantics_parity() -> None:
    """ID: LINALG_HARD_090_norm_kernel_vectorize_false_semantics_parity."""
    text = Path("tal/linalg/ops/norm.py").read_text(encoding="utf-8")
    assert "def compute_norm(" in text
    assert "vectorize=False" in text
    assert "vectorize=True" not in text
