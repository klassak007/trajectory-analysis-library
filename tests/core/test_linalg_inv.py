from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.linalg import Array, Matrix, Vector, inv


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


def _expected_inv(left: xr.DataArray, *, row: str, col: str) -> xr.DataArray:
    return xr.apply_ufunc(
        np.linalg.inv,
        left,
        input_core_dims=[[row, col]],
        output_core_dims=[[col, row]],
        vectorize=True,
        dask="forbidden",
    ).rename("datavar")


def _matrix_batch_core_ao(values: np.ndarray, *, batch: str, row: str, col: str) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": ((batch, row, col), values)},
        coords={
            batch: np.arange(values.shape[0], dtype=np.int64),
            row: np.arange(values.shape[1], dtype=np.int64),
            col: np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        batch_dims=(batch,),
        core_dims=(row, col),
        validate=True,
    )


def _core_dims(ds: xr.Dataset) -> tuple[str, ...]:
    from tal.core.schema_read import read_roles

    return read_roles(ds)[3]


def test_linalg_matrix_005_matrix_inv_square_semantics() -> None:
    """ID: LINALG_MATRIX_005_matrix_inv_square_semantics."""
    values = np.tile(np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=float), (2, 2, 1, 1))
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    out = inv(left)
    out_method = left.inv()
    expected = _expected_inv(left.unsafe_data["x"], row="eq", col="sol")
    assert isinstance(out, Matrix)
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)
    assert list(out.unsafe_data.data_vars) == ["datavar"]
    assert not any("_inv" in name for name in out.unsafe_data.data_vars)
    xr.testing.assert_identical(out.unsafe_data, out_method.unsafe_data)


def test_linalg_core_015_matrix_inv_swaps_core_axes_truthfully() -> None:
    """ID: LINALG_CORE_015_matrix_inv_swaps_core_axes_truthfully."""
    values = np.tile(np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=float), (2, 2, 1, 1))
    left = Matrix(_matrix_ao(values, row="row", col="col"))
    out = inv(left)
    assert _core_dims(out.unsafe_data) == ("col", "row")
    assert out.unsafe_data["datavar"].dims[-2:] == ("col", "row")


def test_linalg_core_016_matrix_inv_batch_core_without_sequence_supported() -> None:
    """ID: LINALG_INV_101_matrix_inv_batch_core_without_sequence_supported."""
    values = np.tile(np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=float), (3, 1, 1))
    left = _matrix_batch_core_ao(values, batch="trial", row="row", col="col")
    out = inv(left)
    expected = _expected_inv(left.unsafe_data["x"], row="row", col="col")
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)
    from tal.core.schema_read import read_roles

    declared, sequence_dim, batch_dims, core_dims = read_roles(out.unsafe_data)
    assert declared is True
    assert sequence_dim is None
    assert batch_dims == ("trial",)
    assert core_dims == ("col", "row")


def test_linalg_hard_032_inv_requires_matrix_operand() -> None:
    """ID: LINALG_HARD_032_inv_requires_matrix_operand."""
    vec = Vector(_vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="eq"))
    with pytest.raises(ValueError, match="left operand must have exactly two core dims"):
        _ = inv(vec)


def test_linalg_hard_033_inv_requires_square_matrix() -> None:
    """ID: LINALG_HARD_033_inv_requires_square_matrix."""
    values = np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    with pytest.raises(ValueError, match="inverse requires square left matrix"):
        _ = inv(left)


def test_linalg_hard_034_inv_chunked_inputs_fail_fast_no_eager() -> None:
    """ID: LINALG_HARD_034_inv_chunked_inputs_fail_fast_no_eager."""
    pytest.importorskip("dask.array")
    values = np.tile(np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=float), (2, 2, 1, 1))
    left = _matrix_ao(values, row="eq", col="sol")
    left_chunked = AnalysisObject.from_data(
        left.unsafe_data.chunk({"sample": 1}),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("eq", "sol"),
        validate=True,
    )
    with pytest.raises(ValueError, match="chunked inverse inputs are not supported"):
        _ = inv(left_chunked)


def test_linalg_hard_035_inv_integer_inputs_preserve_float_or_complex_output() -> None:
    """ID: LINALG_HARD_035_inv_integer_inputs_preserve_float_or_complex_output."""
    values = np.tile(np.asarray([[2, 0], [0, 2]], dtype=np.int64), (2, 2, 1, 1))
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    out = inv(left)
    expected = _expected_inv(left.unsafe_data["x"], row="eq", col="sol")
    data = out.unsafe_data["datavar"]
    assert data.dtype.kind in {"f", "c"}
    xr.testing.assert_allclose(data, expected)
    expected_values = np.tile(np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype=float), (2, 2, 1, 1))
    np.testing.assert_allclose(data.values, expected_values)


def test_linalg_hard_036_inv_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_036_inv_plain_ao_inputs_fallback_to_array."""
    values = np.tile(np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=float), (2, 2, 1, 1))
    left_ao = _matrix_ao(values, row="eq", col="sol")
    out = inv(left_ao)
    out_ds = inv(left_ao.unsafe_data)
    assert type(out) is Array
    assert type(out_ds) is Array


def test_linalg_hard_037_inv_builtin_wrapper_type_routing_preserves_values() -> None:
    """ID: LINALG_HARD_037_inv_builtin_wrapper_type_routing_preserves_values."""
    values = np.tile(np.asarray([[5.0, 2.0], [1.0, 3.0]], dtype=float), (2, 2, 1, 1))
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    out = inv(left)
    expected = _expected_inv(left.unsafe_data["x"], row="eq", col="sol")
    assert isinstance(out, Matrix)
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)


def test_linalg_hard_042_linalgerror_inv_normalized_owner_prefixed() -> None:
    """ID: LINALG_HARD_042_linalgerror_inv_normalized_owner_prefixed."""
    singular = np.tile(np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=float), (2, 2, 1, 1))
    left = Matrix(_matrix_ao(singular, row="eq", col="sol"))
    with pytest.raises(ValueError, match="linalg.inv: inverse failed due to a singular or ill-conditioned matrix"):
        _ = inv(left)


def test_linalg_hard_046_inv_mixed_dtype_complex64_numpy_dtype_parity() -> None:
    """ID: LINALG_HARD_046_inv_mixed_dtype_complex64_numpy_dtype_parity."""
    values = np.tile(np.asarray([[1.0 + 2.0j, 0.0], [0.0, 2.0 - 1.0j]], dtype=np.complex64), (2, 2, 1, 1))
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    out = inv(left)
    expected = _expected_inv(left.unsafe_data["x"], row="eq", col="sol")
    data = out.unsafe_data["datavar"]
    assert data.dtype == expected.dtype
    xr.testing.assert_allclose(data, expected)


def test_linalg_hard_091_inv_kernel_vectorize_false_semantics_parity() -> None:
    """ID: LINALG_HARD_091_inv_kernel_vectorize_false_semantics_parity."""
    text = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    assert "def compute_inv_kernel(" in text
    assert "vectorize=False" in text
    assert "vectorize=True" not in text
