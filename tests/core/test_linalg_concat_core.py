from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.linalg import Array, CoreConcatOptions, Matrix, Vector, concat_core


def _vector_leaf(*, offset: float = 0.0, axis_start: int = 0) -> AnalysisObject:
    values = (np.arange(12, dtype=float).reshape(2, 2, 3) + offset) / 10.0
    ds = xr.Dataset(
        {"x": (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.arange(axis_start, axis_start + 3, dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )


def _matrix_leaf(*, offset: float = 0.0, row_start: int = 0) -> AnalysisObject:
    values = np.arange(16, dtype=float).reshape(2, 2, 2, 2) + offset
    ds = xr.Dataset(
        {"x": (("sample", "trial", "row", "col"), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "row": np.arange(row_start, row_start + 2, dtype=np.int64),
            "col": np.arange(2, dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        validate=True,
    )


def _core_dims(ao: AnalysisObject) -> tuple[str, ...]:
    return read_roles(validate_schema_if_needed(ao.unsafe_data))[3]


def test_linalg_core_021_concat_core_functional_and_typed_constructor_parity() -> None:
    """ID: LINALG_CORE_021_concat_core_functional_and_typed_constructor_parity."""
    leaves = [_vector_leaf(offset=0.0, axis_start=0), _vector_leaf(offset=2.0, axis_start=3)]
    opts = CoreConcatOptions(core_dim="axis")
    out_fn = concat_core(leaves, opts=opts, validate=True)
    out_cls = Array.concat_core(leaves, opts=opts, validate=True)
    assert type(out_fn) is Array
    assert type(out_cls) is Array
    xr.testing.assert_identical(out_fn.unsafe_data, out_cls.unsafe_data)


def test_linalg_hard_063_concat_core_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_063_concat_core_plain_ao_inputs_fallback_to_array."""
    ao0 = _vector_leaf(offset=0.0, axis_start=0)
    ao1 = _vector_leaf(offset=1.0, axis_start=3)
    out = concat_core([ao0, ao1], opts=CoreConcatOptions(core_dim="axis"), validate=True)
    assert type(out) is Array


def test_linalg_hard_064_concat_core_custom_array_subclass_first_operand_wins() -> None:
    """ID: LINALG_HARD_064_concat_core_custom_array_subclass_first_operand_wins."""

    class MyArray(Array):
        pass

    left = MyArray(_vector_leaf(offset=0.0, axis_start=0))
    right = Array(_vector_leaf(offset=1.0, axis_start=3))
    out = concat_core([left, right], opts=CoreConcatOptions(core_dim="axis"), validate=True)
    assert type(out) is MyArray


def test_linalg_hard_065_concat_core_vector_matrix_inputs_preserve_valid_typed_output() -> None:
    """ID: LINALG_HARD_065_concat_core_vector_matrix_inputs_preserve_valid_typed_output."""
    v0 = Vector(_vector_leaf(offset=0.0, axis_start=0))
    v1 = Vector(_vector_leaf(offset=1.0, axis_start=3))
    out_v = concat_core([v0, v1], opts=CoreConcatOptions(core_dim="axis"), validate=True)
    assert type(out_v) is Vector
    assert _core_dims(out_v) == ("axis",)

    m0 = Matrix(_matrix_leaf(offset=0.0, row_start=0))
    m1 = Matrix(_matrix_leaf(offset=1.0, row_start=2))
    out_m = concat_core([m0, m1], opts=CoreConcatOptions(core_dim="row"), validate=True)
    assert type(out_m) is Matrix
    assert _core_dims(out_m) == ("row", "col")
