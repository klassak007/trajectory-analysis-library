from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.linalg import Array, Matrix, SolveOptions, Vector, solve


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


def _expected_solve(left: xr.DataArray, rhs: xr.DataArray, *, row: str, col: str) -> xr.DataArray:
    return xr.apply_ufunc(
        np.linalg.solve,
        left,
        rhs,
        input_core_dims=[[row, col], [row]],
        output_core_dims=[[col]],
        vectorize=True,
        dask="forbidden",
    ).rename("datavar")


def _expected_solve_matrix(
    left: xr.DataArray,
    rhs: xr.DataArray,
    *,
    row: str,
    col: str,
    rhs_col: str,
) -> xr.DataArray:
    return xr.apply_ufunc(
        np.linalg.solve,
        left,
        rhs,
        input_core_dims=[[row, col], [row, rhs_col]],
        output_core_dims=[[col, rhs_col]],
        vectorize=True,
        dask="forbidden",
    ).rename("datavar")


def _expected_lstsq(left: xr.DataArray, rhs: xr.DataArray, *, row: str, col: str) -> xr.DataArray:
    def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(a, b, rcond=None)[0]

    return xr.apply_ufunc(
        _kernel,
        left,
        rhs,
        input_core_dims=[[row, col], [row]],
        output_core_dims=[[col]],
        vectorize=True,
        dask="forbidden",
    ).rename("datavar")


def _core_dims(ds: xr.Dataset) -> tuple[str, ...]:
    from tal.core.schema_read import read_roles

    return read_roles(ds)[3]


def test_linalg_matrix_001_matrix_vector_solve_semantics() -> None:
    """ID: LINALG_MATRIX_001_matrix_vector_solve_semantics."""
    left = Matrix(_matrix_ao(np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2), row="eq", col="sol"))
    rhs = Vector(_vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="eq"))
    out = solve(left, rhs)
    out_method = left.solve(rhs)
    expected = _expected_solve(left.unsafe_data["x"], rhs.unsafe_data["x"], row="eq", col="sol")
    assert isinstance(out, Vector)
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)
    assert list(out.unsafe_data.data_vars) == ["datavar"]
    assert not any("_solve_" in name for name in out.unsafe_data.data_vars)
    xr.testing.assert_identical(out.unsafe_data, out_method.unsafe_data)


def test_linalg_matrix_002_matrix_matrix_solve_semantics() -> None:
    """ID: LINALG_MATRIX_002_matrix_matrix_solve_semantics."""
    left_vals = np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2)
    rhs_vals = (np.arange(24, dtype=float).reshape(2, 2, 2, 3) + 1.0) / 4.0
    left = Matrix(_matrix_ao(left_vals, row="eq", col="sol"))
    rhs = Matrix(_matrix_ao(rhs_vals, row="eq", col="rhs"))
    out = solve(left, rhs)
    expected = _expected_solve_matrix(
        left.unsafe_data["x"],
        rhs.unsafe_data["x"],
        row="eq",
        col="sol",
        rhs_col="rhs",
    )
    assert isinstance(out, Matrix)
    assert _core_dims(out.unsafe_data) == ("sol", "rhs")
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)


def test_linalg_matrix_004_solve_auto_policy_deterministic() -> None:
    """ID: LINALG_MATRIX_004_solve_auto_policy_deterministic."""
    square_left = Matrix(_matrix_ao(np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2), row="eq", col="sol"))
    square_rhs = Vector(_vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="eq"))
    auto_square = solve(square_left, square_rhs, opts=SolveOptions(method="auto"))
    explicit_solve = solve(square_left, square_rhs, opts=SolveOptions(method="solve"))
    xr.testing.assert_allclose(auto_square.unsafe_data["datavar"], explicit_solve.unsafe_data["datavar"])

    tall_left = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0, row="eq", col="sol"))
    tall_rhs = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="eq"))
    auto_tall = solve(tall_left, tall_rhs, opts=SolveOptions(method="auto"))
    explicit_lstsq = solve(tall_left, tall_rhs, opts=SolveOptions(method="lstsq"))
    xr.testing.assert_allclose(auto_tall.unsafe_data["datavar"], explicit_lstsq.unsafe_data["datavar"])


def test_linalg_hard_022_solve_requires_matrix_left_operand() -> None:
    """ID: LINALG_HARD_022_solve_requires_matrix_left_operand."""
    left = Vector(_vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="eq"))
    rhs = Vector(_vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="eq"))
    with pytest.raises(ValueError, match="left operand must have exactly two core dims"):
        _ = solve(left, rhs)


def test_linalg_hard_023_solve_rhs_core_arity_restricted_to_vector_or_matrix() -> None:
    """ID: LINALG_HARD_023_solve_rhs_core_arity_restricted_to_vector_or_matrix."""
    left = Matrix(_matrix_ao(np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2), row="eq", col="sol"))
    rhs_ds = xr.Dataset(
        {"x": (("sample", "trial", "eq", "rhs", "extra"), np.ones((2, 2, 2, 2, 2), dtype=float))},
        coords={
            "sample": [0, 1],
            "trial": ["t0", "t1"],
            "eq": [0, 1],
            "rhs": [0, 1],
            "extra": [0, 1],
        },
    )
    rhs = AnalysisObject.from_data(
        rhs_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("eq", "rhs", "extra"),
        validate=True,
    )
    with pytest.raises(ValueError, match="core_dims cardinality must be in"):
        _ = solve(left, rhs)


def test_linalg_hard_024_solve_contract_dim_mismatch_fail_closed() -> None:
    """ID: LINALG_HARD_024_solve_contract_dim_mismatch_fail_closed."""
    left = Matrix(_matrix_ao(np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2), row="eq", col="sol"))
    rhs = Vector(_vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="other"))
    with pytest.raises(ValueError, match="left core leading dim to equal rhs core leading dim"):
        _ = solve(left, rhs)


def test_linalg_hard_025_solve_method_solve_requires_square_left() -> None:
    """ID: LINALG_HARD_025_solve_method_solve_requires_square_left."""
    left = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0, row="eq", col="sol"))
    rhs = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="eq"))
    with pytest.raises(ValueError, match="requires square left matrix"):
        _ = solve(left, rhs, opts=SolveOptions(method="solve"))


def test_linalg_hard_026_solve_chunked_inputs_fail_fast_no_eager() -> None:
    """ID: LINALG_HARD_026_solve_chunked_inputs_fail_fast_no_eager."""
    pytest.importorskip("dask.array")
    left = _matrix_ao(np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2), row="eq", col="sol")
    rhs = _vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="eq")
    left_chunked = AnalysisObject.from_data(
        left.unsafe_data.chunk({"sample": 1}),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("eq", "sol"),
        validate=True,
    )
    with pytest.raises(ValueError, match="chunked solve inputs are not supported"):
        _ = solve(left_chunked, rhs)


def test_linalg_hard_027_solve_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_027_solve_plain_ao_inputs_fallback_to_array."""
    left_ao = _matrix_ao(np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2), row="eq", col="sol")
    rhs_ao = _vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="eq")
    out = solve(left_ao, rhs_ao)
    out_ds = solve(left_ao.unsafe_data, rhs_ao.unsafe_data)
    assert type(out) is Array
    assert type(out_ds) is Array


def test_linalg_hard_028_solve_builtin_wrapper_type_routing_preserves_values() -> None:
    """ID: LINALG_HARD_028_solve_builtin_wrapper_type_routing_preserves_values."""
    left_vals = np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2)
    rhs_vals = (np.arange(24, dtype=float).reshape(2, 2, 2, 3) + 1.0) / 3.0
    left = Matrix(_matrix_ao(left_vals, row="eq", col="sol"))
    rhs = Matrix(_matrix_ao(rhs_vals, row="eq", col="rhs"))
    out = solve(left, rhs)
    expected = _expected_solve_matrix(
        left.unsafe_data["x"],
        rhs.unsafe_data["x"],
        row="eq",
        col="sol",
        rhs_col="rhs",
    )
    assert isinstance(out, Matrix)
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)


def test_linalg_hard_072_solve_custom_matrix_subclass_vector_rhs_arity_change_falls_back_to_vector() -> None:
    """ID: LINALG_HARD_072_solve_custom_matrix_subclass_vector_rhs_arity_change_falls_back_to_vector."""

    class MyMatrix(Matrix):
        pass

    left_vals = np.arange(16, dtype=float).reshape(2, 2, 2, 2) + np.eye(2)
    rhs_vals = np.arange(8, dtype=float).reshape(2, 2, 2)
    left = MyMatrix(_matrix_ao(left_vals, row="eq", col="sol"))
    rhs = Vector(_vector_ao(rhs_vals, axis="eq"))
    out = solve(left, rhs)
    expected = _expected_solve(left.unsafe_data["x"], rhs.unsafe_data["x"], row="eq", col="sol")
    assert isinstance(out, Vector)
    assert type(out) is Vector
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)


def test_linalg_hard_029_solve_integer_inputs_preserve_fractional_solution() -> None:
    """ID: LINALG_HARD_029_solve_integer_inputs_preserve_fractional_solution."""
    left_vals = np.tile(np.asarray([[2, 0], [0, 2]], dtype=np.int64), (2, 2, 1, 1))
    rhs_vals = np.ones((2, 2, 2), dtype=np.int64)
    left = Matrix(_matrix_ao(left_vals, row="eq", col="sol"))
    rhs = Vector(_vector_ao(rhs_vals, axis="eq"))
    out = solve(left, rhs)
    expected = _expected_solve(left.unsafe_data["x"], rhs.unsafe_data["x"], row="eq", col="sol")
    data = out.unsafe_data["datavar"]
    assert data.dtype.kind in {"f", "c"}
    xr.testing.assert_allclose(data, expected)
    np.testing.assert_allclose(data.values, 0.5)


def test_linalg_hard_030_lstsq_integer_inputs_preserve_fractional_solution() -> None:
    """ID: LINALG_HARD_030_lstsq_integer_inputs_preserve_fractional_solution."""
    left_vals = np.tile(np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.int64), (2, 2, 1, 1))
    rhs_vals = np.ones((2, 2, 3), dtype=np.int64)
    left = Matrix(_matrix_ao(left_vals, row="eq", col="sol"))
    rhs = Vector(_vector_ao(rhs_vals, axis="eq"))
    out = solve(left, rhs, opts=SolveOptions(method="lstsq"))
    expected = _expected_lstsq(left.unsafe_data["x"], rhs.unsafe_data["x"], row="eq", col="sol")
    data = out.unsafe_data["datavar"]
    assert data.dtype.kind in {"f", "c"}
    xr.testing.assert_allclose(data, expected)
    np.testing.assert_allclose(data.values, 2.0 / 3.0)


def test_linalg_hard_031_solve_integer_matrix_rhs_preserve_fractional_solution() -> None:
    """ID: LINALG_HARD_031_solve_integer_matrix_rhs_preserve_fractional_solution."""
    left_vals = np.tile(np.asarray([[2, 0], [0, 2]], dtype=np.int64), (2, 2, 1, 1))
    rhs_vals = np.tile(np.asarray([[1, 3], [1, 3]], dtype=np.int64), (2, 2, 1, 1))
    left = Matrix(_matrix_ao(left_vals, row="eq", col="sol"))
    rhs = Matrix(_matrix_ao(rhs_vals, row="eq", col="rhs"))
    out = solve(left, rhs)
    expected = _expected_solve_matrix(
        left.unsafe_data["x"],
        rhs.unsafe_data["x"],
        row="eq",
        col="sol",
        rhs_col="rhs",
    )
    data = out.unsafe_data["datavar"]
    assert data.dtype.kind in {"f", "c"}
    xr.testing.assert_allclose(data, expected)
    np.testing.assert_allclose(data.values, np.tile(np.asarray([[0.5, 1.5], [0.5, 1.5]]), (2, 2, 1, 1)))


def test_linalg_hard_043_linalgerror_solve_normalized_owner_prefixed() -> None:
    """ID: LINALG_HARD_043_linalgerror_solve_normalized_owner_prefixed."""
    singular = np.tile(np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=float), (2, 2, 1, 1))
    left = Matrix(_matrix_ao(singular, row="eq", col="sol"))
    rhs = Vector(_vector_ao(np.ones((2, 2, 2), dtype=float), axis="eq"))
    with pytest.raises(ValueError, match="linalg.solve: solve failed due to a singular or ill-conditioned linear system"):
        _ = solve(left, rhs, opts=SolveOptions(method="solve"))


def test_linalg_hard_045_solve_mixed_dtype_float32_int_numpy_dtype_parity() -> None:
    """ID: LINALG_HARD_045_solve_mixed_dtype_float32_int_numpy_dtype_parity."""
    left_vals = np.tile(np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32), (2, 2, 1, 1))
    rhs_vals = np.ones((2, 2, 2), dtype=np.int32)
    left = Matrix(_matrix_ao(left_vals, row="eq", col="sol"))
    rhs = Vector(_vector_ao(rhs_vals, axis="eq"))
    out = solve(left, rhs, opts=SolveOptions(method="solve"))
    expected = _expected_solve(left.unsafe_data["x"], rhs.unsafe_data["x"], row="eq", col="sol")
    data = out.unsafe_data["datavar"]
    assert data.dtype == expected.dtype
    xr.testing.assert_allclose(data, expected)


def test_linalg_hard_093_solve_kernel_vectorize_false_semantics_parity() -> None:
    """ID: LINALG_HARD_093_solve_kernel_vectorize_false_semantics_parity."""
    text = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    assert "def compute_solve_kernel(" in text
    assert "vectorize=False" in text


def test_linalg_hard_094_lstsq_kernel_vectorize_true_stopgap_is_explicit() -> None:
    """ID: LINALG_HARD_094_lstsq_kernel_vectorize_true_stopgap_is_explicit."""
    solve_text = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    backend_text = Path("tal/linalg/ops/solve_backends.py").read_text(encoding="utf-8")
    assert "def compute_lstsq_kernel(" in solve_text
    assert "vectorize=True" in solve_text
    assert "lstsq_solution_backend" in solve_text
    assert "LSTSQ_BACKEND_NUMPY_ROW" in solve_text
    assert "def lstsq_solution_backend(" in backend_text
