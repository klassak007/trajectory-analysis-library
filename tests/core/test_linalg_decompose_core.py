from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, CoreDecomposeOptions, assemble_core
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.linalg import Array, Matrix, Vector, Vector3, decompose_core


def _vector_leaf(*, offset: float = 0.0) -> AnalysisObject:
    values = (np.arange(12, dtype=float).reshape(2, 2, 3) + offset) / 10.0
    ds = xr.Dataset(
        {"x": (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.arange(3, dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )


def _matrix_leaf(*, offset: float = 0.0) -> AnalysisObject:
    values = np.arange(16, dtype=float).reshape(2, 2, 2, 2) + offset
    ds = xr.Dataset(
        {"x": (("sample", "trial", "row", "col"), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "row": np.arange(2, dtype=np.int64),
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


def _vector3_leaf() -> AnalysisObject:
    values = np.arange(12, dtype=float).reshape(2, 2, 3)
    ds = xr.Dataset(
        {"x": (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.asarray(["x", "y", "z"], dtype=object),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )


def _core_dims(ao: AnalysisObject) -> tuple[str, ...]:
    return read_roles(validate_schema_if_needed(ao.as_dataset(copy="none")))[3]


def _assembled_grid() -> AnalysisObject:
    leaves = [_vector_leaf(offset=0.0), _vector_leaf(offset=1.0), _vector_leaf(offset=2.0), _vector_leaf(offset=3.0)]
    return assemble_core(
        [[leaves[0], leaves[1]], [leaves[2], leaves[3]]],
        core_dims=("row", "col"),
        core_labels=(("r0", "r1"), ("c0", "c1")),
        validate=True,
    )


def _assert_mapping_identical(
    left: dict[tuple[object, ...], Array],
    right: dict[tuple[object, ...], Array],
) -> None:
    assert list(left.keys()) == list(right.keys())
    for key in left:
        assert type(left[key]) is Array
        assert type(right[key]) is Array
        xr.testing.assert_identical(left[key].as_dataset(copy="none"), right[key].as_dataset(copy="none"))


def test_linalg_core_026_decompose_core_functional_and_method_parity() -> None:
    """ID: LINALG_CORE_026_decompose_core_functional_and_method_parity."""
    assembled = _assembled_grid()
    opts = CoreDecomposeOptions(core_dims=("row", "col"), key_mode="index")
    out_fn = decompose_core(assembled, opts=opts, validate=True)
    out_method = Array(assembled).decompose_core(opts=opts, validate=True)
    _assert_mapping_identical(out_fn, out_method)


def test_linalg_hard_074_decompose_core_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_074_decompose_core_plain_ao_inputs_fallback_to_array."""
    assembled = _assembled_grid()
    out = decompose_core(
        assembled,
        opts=CoreDecomposeOptions(core_dims=("row", "col"), key_mode="index"),
        validate=True,
    )
    assert out
    assert all(type(value) is Array for value in out.values())


def test_linalg_hard_075_decompose_core_custom_subclass_outputs_use_array_safe_fallback() -> None:
    """ID: LINALG_HARD_075_decompose_core_custom_subclass_outputs_use_array_safe_fallback."""

    class MyArray(Array):
        pass

    wrapped = MyArray(_assembled_grid())
    out = wrapped.decompose_core(
        opts=CoreDecomposeOptions(core_dims=("row", "col"), key_mode="index"),
        validate=True,
    )
    assert out
    assert all(type(value) is Array for value in out.values())


def test_linalg_hard_076_decompose_core_vector_matrix_vector3_inputs_do_not_invalid_rewrap() -> None:
    """ID: LINALG_HARD_076_decompose_core_vector_matrix_vector3_inputs_do_not_invalid_rewrap."""
    vector_out = decompose_core(
        Vector(_vector_leaf()),
        opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="index"),
        validate=True,
    )
    assert list(vector_out.keys()) == [(0,), (1,), (2,)]
    assert all(type(value) is Array for value in vector_out.values())

    matrix_out = decompose_core(
        Matrix(_matrix_leaf()),
        opts=CoreDecomposeOptions(core_dims=("row",), key_mode="index"),
        validate=True,
    )
    assert list(matrix_out.keys()) == [(0,), (1,)]
    assert all(type(value) is Array for value in matrix_out.values())
    assert _core_dims(matrix_out[(0,)]) == ("col",)

    vector3_out = decompose_core(
        Vector3(_vector3_leaf()),
        opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label"),
        validate=True,
    )
    assert list(vector3_out.keys()) == [("x",), ("y",), ("z",)]
    assert all(type(value) is Array for value in vector3_out.values())


def test_linalg_hard_077_decompose_core_opts_required_signature_alignment() -> None:
    """ID: LINALG_HARD_077_decompose_core_opts_required_signature_alignment."""
    assembled = _assembled_grid()
    with pytest.raises(TypeError, match="required keyword-only argument: 'opts'"):
        _ = decompose_core(assembled, validate=True)
    with pytest.raises(TypeError, match="required keyword-only argument: 'opts'"):
        _ = Array(assembled).decompose_core(validate=True)
