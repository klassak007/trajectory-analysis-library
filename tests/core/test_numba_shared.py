from __future__ import annotations

import numpy as np
import pytest

from tal.utils.block_rows import BlockInputSpec, BlockRows, prepare_block_rows, row_count
from tal.utils.numba_scan import ScanAxisSpec, ScanInputSpec, ScanRows, prepare_scan_rows


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

    spatial_slerp = prepare_block_rows(
        (
            np.zeros((2, 1, 5, 4)),
            np.zeros((1, 3, 5, 4)),
            np.zeros((5,)),
            np.ones((1, 3, 5), dtype=bool),
        ),
        (
            BlockInputSpec("q0", 2, np.float64),
            BlockInputSpec("q1", 2, np.float64),
            BlockInputSpec("alpha", 1, np.float64),
            BlockInputSpec("valid", 1, bool),
        ),
        output_core_shape=(5, 4),
        owner="numba.shared",
    )
    assert spatial_slerp.output_shape == (2, 3, 5, 4)
    assert tuple(array.shape for array in spatial_slerp.row_arrays) == ((6, 5, 4), (6, 5, 4), (6, 5), (6, 5))

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


def test_numba_scan_001_scan_rows_preserve_primary_ordered_axis() -> None:
    """ID: NUMBA_SCAN_001_scan_rows_preserve_primary_ordered_axis."""
    prepared = prepare_scan_rows(
        (
            np.zeros((2, 1, 5, 3)),
            np.zeros((1, 4, 5)),
            np.ones((2, 4, 5), dtype=bool),
        ),
        (
            ScanInputSpec("values", 1, 1, np.float64),
            ScanInputSpec("param", 1, 0, np.float64),
            ScanInputSpec("valid", 1, 0, bool),
        ),
        ordered_axes=(ScanAxisSpec("sequence", "scan"),),
        output_core_shapes=((), (3,)),
        owner="numba.scan",
    )
    assert isinstance(prepared, ScanRows)
    assert prepared.outer_shape == (2, 4)
    assert prepared.ordered_shape == (5,)
    assert prepared.core_shapes == ((3,), (), ())
    assert prepared.output_shapes == ((2, 4, 5), (2, 4, 5, 3))
    assert tuple(array.shape for array in prepared.row_arrays) == ((8, 5, 3), (8, 5), (8, 5))
    assert all(array.flags.c_contiguous for array in prepared.row_arrays)

    with pytest.raises(ValueError, match=r"numba\.scan: block/spec count mismatch"):
        prepare_scan_rows(
            (np.zeros((1, 2)),),
            (),
            ordered_axes=(ScanAxisSpec("sequence", "scan"),),
            output_core_shapes=((),),
            owner="numba.scan",
        )
    with pytest.raises(ValueError, match=r"numba\.scan: scan inputs must not be empty"):
        prepare_scan_rows(
            (),
            (),
            ordered_axes=(ScanAxisSpec("sequence", "scan"),),
            output_core_shapes=((),),
            owner="numba.scan",
        )
    with pytest.raises(ValueError, match=r"numba\.scan: ordered_axes must not be empty"):
        prepare_scan_rows(
            (np.zeros((1, 2)),),
            (ScanInputSpec("values", 1, 0),),
            ordered_axes=(),
            output_core_shapes=((),),
            owner="numba.scan",
        )
    with pytest.raises(ValueError, match=r"numba\.scan: ordered axes must have matching lengths"):
        prepare_scan_rows(
            (np.zeros((1, 2, 3)), np.zeros((1, 4))),
            (ScanInputSpec("values", 1, 1), ScanInputSpec("param", 1, 0)),
            ordered_axes=(ScanAxisSpec("sequence", "scan"),),
            output_core_shapes=((),),
            owner="numba.scan",
        )
    with pytest.raises(ValueError, match=r"numba\.scan: block 'bad' must include"):
        prepare_scan_rows(
            (np.zeros((2,)),),
            (ScanInputSpec("bad", 1, 1),),
            ordered_axes=(ScanAxisSpec("sequence", "scan"),),
            output_core_shapes=((),),
            owner="numba.scan",
        )
    with pytest.raises(ValueError, match=r"numba\.scan: block 'bad' could not be coerced"):
        prepare_scan_rows(
            (np.array(["bad"], dtype=object),),
            (ScanInputSpec("bad", 1, 0, np.float64),),
            ordered_axes=(ScanAxisSpec("sequence", "scan"),),
            output_core_shapes=((),),
            owner="numba.scan",
        )


def test_numba_scan_002_core_scan_axis_is_owner_declared() -> None:
    """ID: NUMBA_SCAN_002_core_scan_axis_is_owner_declared."""
    prepared = prepare_scan_rows(
        (np.zeros((3, 4, 2)),),
        (ScanInputSpec("system", 1, 1, np.float64),),
        ordered_axes=(ScanAxisSpec("state", "core_scan"),),
        output_core_shapes=((2,),),
        owner="numba.scan",
    )
    assert prepared.outer_shape == (3,)
    assert prepared.ordered_shape == (4,)
    assert prepared.core_shapes == ((2,),)
    assert prepared.output_shapes == ((3, 4, 2),)
    assert prepared.row_arrays[0].shape == (3, 4, 2)

    with pytest.raises(ValueError, match=r"numba\.scan: block 'system' ordered_ndim must match ordered_axes"):
        prepare_scan_rows(
            (np.zeros((3, 4, 2)),),
            (ScanInputSpec("system", 0, 1),),
            ordered_axes=(ScanAxisSpec("state", "core_scan"),),
            output_core_shapes=((2,),),
            owner="numba.scan",
        )


def test_numba_scan_003_nested_scan_metadata_is_shape_only() -> None:
    """ID: NUMBA_SCAN_003_nested_scan_metadata_is_shape_only."""
    prepared = prepare_scan_rows(
        (np.zeros((2, 3, 4, 5)),),
        (ScanInputSpec("values", 2, 1, np.float64),),
        ordered_axes=(ScanAxisSpec("time", "scan"), ScanAxisSpec("chain", "topology")),
        output_core_shapes=((), (5,)),
        owner="numba.scan",
    )
    assert prepared.outer_shape == (2,)
    assert prepared.ordered_shape == (3, 4)
    assert prepared.core_shapes == ((5,),)
    assert prepared.output_shapes == ((2, 3, 4), (2, 3, 4, 5))
    assert prepared.row_arrays[0].shape == (2, 3, 4, 5)
