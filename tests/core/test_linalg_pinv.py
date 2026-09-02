from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.linalg import Array, Matrix, PInvOptions, Vector, pinv


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


def _expected_pinv(
    left: xr.DataArray,
    *,
    row: str,
    col: str,
    rcond: float | None = None,
    hermitian: bool = False,
) -> xr.DataArray:
    return xr.apply_ufunc(
        np.linalg.pinv,
        left,
        input_core_dims=[[row, col]],
        output_core_dims=[[col, row]],
        vectorize=True,
        dask="forbidden",
        kwargs={"rcond": rcond, "hermitian": hermitian},
    ).rename("datavar")


def _core_dims(ds: xr.Dataset) -> tuple[str, ...]:
    from tal.core.schema_read import read_roles

    return read_roles(ds)[3]


def test_linalg_matrix_006_matrix_pinv_rectangular_semantics() -> None:
    """ID: LINALG_MATRIX_006_matrix_pinv_rectangular_semantics."""
    values = (np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0) / 4.0
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    out = pinv(left)
    out_method = left.pinv()
    expected = _expected_pinv(left.as_dataset(copy="none")["x"], row="eq", col="sol")
    assert isinstance(out, Matrix)
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)
    assert list(out.as_dataset(copy="none").data_vars) == ["datavar"]
    assert not any("_pinv" in name for name in out.as_dataset(copy="none").data_vars)
    xr.testing.assert_identical(out.as_dataset(copy="none"), out_method.as_dataset(copy="none"))


def test_linalg_core_016_matrix_pinv_swaps_core_axes_truthfully() -> None:
    """ID: LINALG_CORE_016_matrix_pinv_swaps_core_axes_truthfully."""
    values = (np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0) / 5.0
    left = Matrix(_matrix_ao(values, row="row", col="col"))
    out = pinv(left)
    assert _core_dims(out.as_dataset(copy="none")) == ("col", "row")
    assert out.as_dataset(copy="none")["datavar"].dims[-2:] == ("col", "row")


def test_linalg_hard_038_pinv_requires_matrix_operand() -> None:
    """ID: LINALG_HARD_038_pinv_requires_matrix_operand."""
    vec = Vector(_vector_ao(np.arange(8, dtype=float).reshape(2, 2, 2), axis="eq"))
    with pytest.raises(ValueError, match="operand must have exactly two core dims"):
        _ = pinv(vec)


def test_linalg_hard_039_pinv_chunked_inputs_fail_fast_no_eager() -> None:
    """ID: LINALG_HARD_039_pinv_chunked_inputs_fail_fast_no_eager."""
    pytest.importorskip("dask.array")
    values = (np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0) / 4.0
    left = _matrix_ao(values, row="eq", col="sol")
    left_chunked = AnalysisObject.from_data(
        left.as_dataset(copy="none").chunk({"sample": 1}),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("eq", "sol"),
        validate=True,
    )
    with pytest.raises(ValueError, match="chunked pseudoinverse inputs are not supported"):
        _ = pinv(left_chunked)


def test_linalg_hard_040_pinv_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_040_pinv_plain_ao_inputs_fallback_to_array."""
    values = (np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0) / 4.0
    left_ao = _matrix_ao(values, row="eq", col="sol")
    out = pinv(left_ao)
    out_ds = pinv(left_ao.as_dataset(copy="none"))
    assert type(out) is Array
    assert type(out_ds) is Array


def test_linalg_hard_041_pinv_builtin_wrapper_type_routing_preserves_values() -> None:
    """ID: LINALG_HARD_041_pinv_builtin_wrapper_type_routing_preserves_values."""
    values = (np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 2.0) / 7.0
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    out = pinv(left, opts=PInvOptions(rcond=1e-8))
    expected = _expected_pinv(left.as_dataset(copy="none")["x"], row="eq", col="sol", rcond=1e-8)
    assert isinstance(out, Matrix)
    xr.testing.assert_allclose(out.as_dataset(copy="none")["datavar"], expected)


def test_linalg_hard_047_pinv_hermitian_non_square_policy_error_specific() -> None:
    """ID: LINALG_HARD_047_pinv_hermitian_non_square_policy_error_specific."""
    values = (np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0) / 4.0
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    with pytest.raises(ValueError) as excinfo:
        _ = pinv(left, opts=PInvOptions(hermitian=True))
    text = str(excinfo.value)
    assert "opts.hermitian=True requires square matrix" in text
    assert "singular or ill-conditioned matrix" not in text


def test_linalg_hard_044_linalgerror_pinv_normalized_owner_prefixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: LINALG_HARD_044_linalgerror_pinv_normalized_owner_prefixed."""

    def _raise(*_args, **_kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    values = (np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0) / 4.0
    left = Matrix(_matrix_ao(values, row="eq", col="sol"))
    monkeypatch.setattr(np.linalg, "pinv", _raise)
    with pytest.raises(ValueError, match="linalg.pinv: pseudoinverse failed due to a singular or ill-conditioned matrix"):
        _ = pinv(left)


def test_linalg_hard_092_pinv_kernel_vectorize_false_semantics_parity() -> None:
    """ID: LINALG_HARD_092_pinv_kernel_vectorize_false_semantics_parity."""
    text = Path("tal/linalg/ops/pinv.py").read_text(encoding="utf-8")
    assert "def compute_pinv_kernel(" in text
    assert "vectorize=False" in text
    assert "vectorize=True" not in text
