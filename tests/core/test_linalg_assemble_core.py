from __future__ import annotations

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.linalg import Array, Matrix, Vector, assemble_core, block_core, stack_core


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


def _core_dims(ao: AnalysisObject) -> tuple[str, ...]:
    return read_roles(validate_schema_if_needed(ao.unsafe_data))[3]


def test_linalg_core_019_assemble_core_functional_and_typed_constructor_parity() -> None:
    """ID: LINALG_CORE_019_assemble_core_functional_and_typed_constructor_parity."""
    leaves = [_vector_leaf(offset=0.0), _vector_leaf(offset=2.0)]
    out_fn = assemble_core(leaves, core_dims=("row",), core_labels=(("r0", "r1"),))
    out_cls = Array.assemble_core(leaves, core_dims=("row",), core_labels=(("r0", "r1"),))
    assert type(out_fn) is Array
    assert type(out_cls) is Array
    xr.testing.assert_identical(out_fn.unsafe_data, out_cls.unsafe_data)


def test_linalg_core_020_stack_block_wrappers_delegate_to_nd_assemble_owner() -> None:
    """ID: LINALG_CORE_020_stack_block_wrappers_delegate_to_nd_assemble_owner."""
    a = _vector_leaf(offset=0.0)
    b = _vector_leaf(offset=1.0)
    c = _vector_leaf(offset=2.0)
    d = _vector_leaf(offset=3.0)
    out_stack = stack_core([a, b], core_dim="row")
    out_nd_stack = assemble_core([a, b], core_dims=("row",))
    xr.testing.assert_identical(out_stack.unsafe_data, out_nd_stack.unsafe_data)
    out_block = block_core([[a, b], [c, d]], row_dim="row", col_dim="col")
    out_nd_block = assemble_core([[a, b], [c, d]], core_dims=("row", "col"))
    xr.testing.assert_identical(out_block.unsafe_data, out_nd_block.unsafe_data)


def test_linalg_hard_059_assemble_core_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_059_assemble_core_plain_ao_inputs_fallback_to_array."""
    leaves = [_vector_leaf(offset=0.0), _vector_leaf(offset=2.0)]
    out = assemble_core(leaves, core_dims=("row",))
    assert type(out) is Array


def test_linalg_hard_060_assemble_core_custom_array_subclass_constructor_preserved() -> None:
    """ID: LINALG_HARD_060_assemble_core_custom_array_subclass_constructor_preserved."""

    class MyArray(Array):
        pass

    leaves = [_vector_leaf(offset=0.0), _vector_leaf(offset=2.0)]
    out = MyArray.assemble_core(leaves, core_dims=("row",))
    assert type(out) is MyArray
    expected = assemble_core(leaves, core_dims=("row",))
    xr.testing.assert_identical(out.unsafe_data, expected.unsafe_data)


def test_linalg_hard_061_stack_core_vector_leaves_finalize_rewrap_safe() -> None:
    """ID: LINALG_HARD_061_stack_core_vector_leaves_finalize_rewrap_safe."""
    top = Vector(_vector_leaf(offset=0.0))
    bottom = Vector(_vector_leaf(offset=1.0))
    out = stack_core([top, bottom], core_dim="row")
    assert type(out) is Array
    assert _core_dims(out) == ("row", "axis")
    expected = stack_core([_vector_leaf(offset=0.0), _vector_leaf(offset=1.0)], core_dim="row")
    xr.testing.assert_allclose(out.unsafe_data["x"], expected.unsafe_data["x"])


def test_linalg_hard_062_block_core_matrix_leaves_finalize_rewrap_safe() -> None:
    """ID: LINALG_HARD_062_block_core_matrix_leaves_finalize_rewrap_safe."""
    a = Matrix(_matrix_leaf(offset=0.0))
    b = Matrix(_matrix_leaf(offset=1.0))
    c = Matrix(_matrix_leaf(offset=2.0))
    d = Matrix(_matrix_leaf(offset=3.0))
    out = block_core([[a, b], [c, d]], row_dim="blk_r", col_dim="blk_c")
    assert type(out) is Array
    assert _core_dims(out) == ("blk_r", "blk_c", "row", "col")
    expected = block_core(
        [
            [_matrix_leaf(offset=0.0), _matrix_leaf(offset=1.0)],
            [_matrix_leaf(offset=2.0), _matrix_leaf(offset=3.0)],
        ],
        row_dim="blk_r",
        col_dim="blk_c",
    )
    xr.testing.assert_allclose(out.unsafe_data["x"], expected.unsafe_data["x"])
