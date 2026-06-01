from __future__ import annotations

import numpy as np
import pytest

from tal.utils.block_rows import BlockInputSpec, BlockRows, prepare_block_rows, row_count


def test_numba_shared_001_block_rows_matches_existing_broadcast_shapes() -> None:
    """ID: NUMBA_SHARED_001_block_rows_matches_existing_broadcast_shapes."""
    assert row_count((2, 3, 4)) == 24

    param_map = prepare_block_rows(
        (
            np.zeros((2, 1, 4)),
            np.ones((1, 3, 4), dtype=bool),
            np.zeros((3, 5)),
        ),
        (
            BlockInputSpec("param", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("query", 1, np.float64),
        ),
        output_core_shape=(5,),
        owner="numba.shared",
    )
    assert isinstance(param_map, BlockRows)
    assert param_map.outer_shape == (2, 3)
    assert param_map.output_shape == (2, 3, 5)
    assert tuple(array.shape for array in param_map.row_arrays) == ((6, 4), (6, 4), (6, 5))
    assert all(array.flags.c_contiguous for array in param_map.row_arrays)

    param_bounds = prepare_block_rows(
        (
            np.zeros((2, 1, 4)),
            np.ones((1, 3, 4), dtype=bool),
            np.zeros((2, 1)),
            np.ones((3,)),
        ),
        (
            BlockInputSpec("param", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("start", 0, np.float64),
            BlockInputSpec("stop", 0, np.float64),
        ),
        output_core_shape=(),
        owner="numba.shared",
    )
    assert param_bounds.outer_shape == (2, 3)
    assert tuple(array.shape for array in param_bounds.row_arrays) == ((6, 4), (6, 4), (6,), (6,))

    event_boundary = prepare_block_rows(
        (np.zeros((4, 1, 6), dtype=bool), np.ones((1, 5, 6), dtype=bool), np.arange(6.0)),
        (
            BlockInputSpec("mask", 1, bool),
            BlockInputSpec("valid", 1, bool),
            BlockInputSpec("clock", 1, np.float64),
        ),
        output_core_shape=(),
        owner="numba.shared",
    )
    assert event_boundary.outer_shape == (4, 5)
    assert tuple(array.shape for array in event_boundary.row_arrays) == ((20, 6), (20, 6), (20, 6))

    event_intervals = prepare_block_rows(
        (
            np.zeros((2, 1, 7)),
            np.zeros((1, 3, 7), dtype=np.int8),
            np.zeros((7,), dtype=np.int64),
            np.ones((7,), dtype=np.int64),
        ),
        (
            BlockInputSpec("time", 1, np.float64),
            BlockInputSpec("edge", 1, np.int8),
            BlockInputSpec("before", 1, np.int64),
            BlockInputSpec("after", 1, np.int64),
        ),
        output_core_shape=(4, 2),
        owner="numba.shared",
    )
    assert event_intervals.outer_shape == (2, 3)
    assert event_intervals.output_shape == (2, 3, 4, 2)
    assert tuple(array.shape for array in event_intervals.row_arrays) == ((6, 7), (6, 7), (6, 7), (6, 7))

    linalg_vector = prepare_block_rows(
        (np.zeros((8, 1, 4, 2)), np.zeros((1, 5, 4))),
        (BlockInputSpec("a", 2), BlockInputSpec("b", 1)),
        output_core_shape=(2,),
        owner="numba.shared",
    )
    assert linalg_vector.output_shape == (8, 5, 2)
    assert tuple(array.shape for array in linalg_vector.row_arrays) == ((40, 4, 2), (40, 4))

    linalg_matrix = prepare_block_rows(
        (np.zeros((8, 1, 4, 2)), np.zeros((1, 5, 4, 3))),
        (BlockInputSpec("a", 2), BlockInputSpec("b", 2)),
        output_core_shape=(2, 3),
        owner="numba.shared",
    )
    assert linalg_matrix.output_shape == (8, 5, 2, 3)
    assert tuple(array.shape for array in linalg_matrix.row_arrays) == ((40, 4, 2), (40, 4, 3))

    with pytest.raises(ValueError, match=r"numba\.shared: block 'bad' could not be coerced to dtype"):
        prepare_block_rows(
            (np.array(["bad"], dtype=object),),
            (BlockInputSpec("bad", 1, np.float64),),
            output_core_shape=(),
            owner="numba.shared",
        )
    with pytest.raises(ValueError, match=r"numba\.shared: block 'overflow' could not be coerced to dtype"):
        prepare_block_rows(
            (np.array([10**100], dtype=object),),
            (BlockInputSpec("overflow", 1, np.int8),),
            output_core_shape=(),
            owner="numba.shared",
        )
    with pytest.raises(ValueError, match=r"numba\.shared: block/spec count mismatch"):
        prepare_block_rows(
            (np.zeros((1,)),),
            (),
            output_core_shape=(),
            owner="numba.shared",
        )
