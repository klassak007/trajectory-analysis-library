from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.linalg import Array, Matrix, Vector, matmul


def _matrix_ao(values: np.ndarray, *, row: str = "row", col: str = "col") -> AnalysisObject:
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


def _vector_ao(values: np.ndarray, *, axis: str = "axis") -> AnalysisObject:
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


def test_linalg_core_011_vector_wrapper_enforces_single_core_dim() -> None:
    """ID: LINALG_CORE_011_vector_wrapper_enforces_single_core_dim."""
    vec = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    assert isinstance(vec.set_core_dims("axis"), Vector)
    assert _core_dims(vec.unsafe_data) == ("axis",)


def test_linalg_core_012_matrix_wrapper_enforces_two_core_dims() -> None:
    """ID: LINALG_CORE_012_matrix_wrapper_enforces_two_core_dims."""
    mat = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="r", col="c"))
    assert isinstance(mat.set_core_dims("r", "c"), Matrix)
    assert _core_dims(mat.unsafe_data) == ("r", "c")


def test_linalg_hard_097_vector_typed_lifecycle_parity() -> None:
    """ID: LINALG_HARD_097_vector_typed_lifecycle_parity."""
    vec = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    from_ds = Vector(vec.unsafe_data)

    assert isinstance(vec, Vector)
    assert isinstance(from_ds, Vector)
    assert _core_dims(from_ds.unsafe_data) == ("axis",)
    for validate in (True, False):
        with pytest.raises(ValueError, match="Vector requires exactly one core dim"):
            _ = vec.set_roles(
                sequence_dim="sample",
                batch_dims=(),
                core_dims=("trial", "axis"),
                validate=validate,
            )


def test_linalg_hard_098_matrix_typed_lifecycle_parity() -> None:
    """ID: LINALG_HARD_098_matrix_typed_lifecycle_parity."""
    mat = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="row", col="col"))
    from_ds = Matrix(mat.unsafe_data)

    assert isinstance(mat, Matrix)
    assert isinstance(from_ds, Matrix)
    assert _core_dims(from_ds.unsafe_data) == ("row", "col")
    for validate in (True, False):
        with pytest.raises(ValueError, match="Matrix requires exactly two core dims"):
            _ = mat.set_roles(
                sequence_dim="sample",
                batch_dims=("trial",),
                core_dims=("row",),
                validate=validate,
            )
    with pytest.raises(ValueError, match="duplicate|distinct"):
        _ = Matrix._from_unvalidated(mat.set_roles(core_dims=("row", "row"), validate=False).unsafe_data)


def test_linalg_core_013_matrix_transpose_swaps_core_axes_truthfully() -> None:
    """ID: LINALG_CORE_013_matrix_transpose_swaps_core_axes_truthfully."""
    mat = Matrix(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    transposed = mat.T
    assert isinstance(transposed, Matrix)
    assert _core_dims(transposed.unsafe_data) == ("col", "row")
    expected = mat.unsafe_data.transpose("sample", "trial", "col", "row")
    xr.testing.assert_allclose(transposed.unsafe_data["x"], expected["x"])


def test_linalg_core_014_matmul_builtins_route_output_type_by_core_arity() -> None:
    """ID: LINALG_CORE_014_matmul_builtins_route_output_type_by_core_arity."""
    mm_left = Matrix(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid"))
    mm_right = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out"))
    mv_right = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="mid"))
    vm_right = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out"))
    vv_right = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="mid"))

    out_mm = matmul(mm_left, mm_right)
    out_mv = matmul(mm_left, mv_right)
    out_vm = matmul(Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="mid")), vm_right)
    out_vv = matmul(Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="mid")), vv_right)

    assert isinstance(out_mm, Matrix)
    assert isinstance(out_mv, Vector)
    assert isinstance(out_vm, Vector)
    assert type(out_vv) is Array


def test_linalg_hard_016_vector_matrix_wrapper_invalid_arity_fail_closed() -> None:
    """ID: LINALG_HARD_016_vector_matrix_wrapper_invalid_arity_fail_closed."""
    with pytest.raises(ValueError, match="Vector requires exactly one core dim"):
        _ = Vector(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3)))
    with pytest.raises(ValueError, match="Matrix requires exactly two core dims"):
        _ = Matrix(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3)))

    vec = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    mat = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="r", col="c"))
    with pytest.raises(ValueError, match="Vector requires exactly one core dim"):
        _ = vec.set_core_dims("a", "b")
    with pytest.raises(ValueError, match="Matrix requires exactly two core dims"):
        _ = mat.set_core_dims("r")


def test_linalg_hard_019_vector_set_roles_rewrap_enforces_invariant() -> None:
    """ID: LINALG_HARD_019_vector_set_roles_rewrap_enforces_invariant."""
    vec = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    with pytest.raises(ValueError, match="Vector requires exactly one core dim"):
        _ = vec.set_roles(
            sequence_dim="sample",
            batch_dims=(),
            core_dims=("trial", "axis"),
            validate=True,
        )


def test_linalg_hard_020_matrix_set_roles_rewrap_enforces_invariant() -> None:
    """ID: LINALG_HARD_020_matrix_set_roles_rewrap_enforces_invariant."""
    mat = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="row", col="col"))
    with pytest.raises(ValueError, match="Matrix requires exactly two core dims"):
        _ = mat.set_roles(
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("row",),
            validate=True,
        )


def test_linalg_hard_021_wrapper_invariants_enforced_on_validate_false_rewrap() -> None:
    """ID: LINALG_HARD_021_wrapper_invariants_enforced_on_validate_false_rewrap."""
    vec = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    with pytest.raises(ValueError, match="Vector requires exactly one core dim"):
        _ = vec.set_roles(
            sequence_dim="sample",
            batch_dims=(),
            core_dims=("trial", "axis"),
            validate=False,
        )

    mat = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="row", col="col"))
    with pytest.raises(ValueError, match="Matrix requires exactly two core dims"):
        _ = mat.set_roles(
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("row",),
            validate=False,
        )


def test_linalg_hard_017_matmul_custom_array_subclass_left_wins_policy_preserved() -> None:
    """ID: LINALG_HARD_017_matmul_custom_array_subclass_left_wins_policy_preserved."""

    class MyArray(Array):
        pass

    left = MyArray(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid"))
    right = Matrix(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out"))
    out = matmul(left, right)
    assert isinstance(out, MyArray)


def test_linalg_hard_018_matmul_builtin_wrapper_type_routing_no_regression_in_values() -> None:
    """ID: LINALG_HARD_018_matmul_builtin_wrapper_type_routing_no_regression_in_values."""
    left = Matrix(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid"))
    right = Vector(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="mid"))
    out = matmul(left, right)
    expected = xr.dot(left.unsafe_data["x"], right.unsafe_data["x"], dim=["mid"]).rename("datavar")
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)
    assert isinstance(out, Vector)


def _core_dims(ds: xr.Dataset) -> tuple[str, ...]:
    from tal.core.schema_read import read_roles

    return read_roles(ds)[3]
