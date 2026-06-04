from __future__ import annotations

import numpy as np
import pytest

import tal.utils.numba as tal_numba
import tal.utils.numba_support as numba_support


def test_numba_public_001_public_numba_namespace_is_optional_dependency_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: NUMBA_PUBLIC_001_public_numba_namespace_is_optional_dependency_safe."""
    bounds = tal_numba.centered_window_bounds(3, radius=1, owner="numba.public")
    assert bounds.start.tolist() == [0, 0, 1]
    assert bounds.stop.tolist() == [2, 3, 3]

    prepared = tal_numba.prepare_block_rows(
        (np.arange(6.0).reshape(2, 3),),
        (tal_numba.BlockInputSpec("values", 1, np.float64),),
        output_core_shape=(3,),
        owner="numba.public",
    )
    assert isinstance(prepared, tal_numba.BlockRows)
    assert prepared.row_arrays[0].shape == (2, 3)

    def _missing_numba():
        raise ImportError("missing")

    monkeypatch.setattr(numba_support, "_import_numba", _missing_numba)
    with pytest.raises(ImportError, match=r"numba\.public: numba is required"):
        tal_numba.require_numba("numba.public")


def test_numba_public_002_public_numba_block_and_scan_examples_are_schema_free() -> None:
    """ID: NUMBA_PUBLIC_002_public_numba_block_and_scan_examples_are_schema_free."""
    block_rows = tal_numba.prepare_block_rows(
        (
            np.zeros((2, 1, 4, 3)),
            np.ones((1, 5, 4), dtype=bool),
        ),
        (
            tal_numba.BlockInputSpec("values", 2, np.float64),
            tal_numba.BlockInputSpec("mask", 1, bool),
        ),
        output_core_shape=(4, 3),
        owner="numba.public",
    )
    assert block_rows.outer_shape == (2, 5)
    assert block_rows.output_shape == (2, 5, 4, 3)
    assert tuple(array.shape for array in block_rows.row_arrays) == ((10, 4, 3), (10, 4))

    scan_rows = tal_numba.prepare_scan_rows(
        (np.zeros((2, 3, 4)),),
        (tal_numba.ScanInputSpec("values", 1, 1, np.float64),),
        ordered_axes=(tal_numba.ScanAxisSpec("sequence", "scan"),),
        output_core_shapes=((4,),),
        owner="numba.public",
    )
    assert isinstance(scan_rows, tal_numba.ScanRows)
    assert scan_rows.outer_shape == (2,)
    assert scan_rows.ordered_shape == (3,)
    assert scan_rows.core_shapes == ((4,),)


def test_numba_public_003_public_numba_stencil_substrate_is_policy_free() -> None:
    """ID: NUMBA_PUBLIC_003_public_numba_stencil_substrate_is_policy_free."""
    centered = tal_numba.centered_window_bounds(5, radius=1, owner="numba.public")
    assert centered.start.dtype == np.dtype(np.int64)
    assert centered.stop.dtype == np.dtype(np.int64)
    assert centered.start.tolist() == [0, 0, 1, 2, 3]
    assert centered.stop.tolist() == [2, 3, 4, 5, 5]

    forward = tal_numba.forward_window_bounds(5, width=3, owner="numba.public")
    assert forward.start.tolist() == [0, 1, 2, 3, 4]
    assert forward.stop.tolist() == [3, 4, 5, 5, 5]

    backward = tal_numba.backward_window_bounds(5, width=3, owner="numba.public")
    assert backward.start.tolist() == [0, 0, 0, 1, 2]
    assert backward.stop.tolist() == [1, 2, 3, 4, 5]

    clipped = tal_numba.clipped_window_bounds(4, before=2, after=0, owner="numba.public")
    assert isinstance(clipped, tal_numba.WindowBounds)
    assert clipped.start.tolist() == [0, 0, 0, 1]
    assert clipped.stop.tolist() == [1, 2, 3, 4]

    huge_before = tal_numba.clipped_window_bounds(2, before=10**100, after=0, owner="numba.public")
    assert huge_before.start.tolist() == [0, 0]
    assert huge_before.stop.tolist() == [1, 2]
    huge_after = tal_numba.clipped_window_bounds(2, before=0, after=10**100, owner="numba.public")
    assert huge_after.start.tolist() == [0, 1]
    assert huge_after.stop.tolist() == [2, 2]
    huge_radius = tal_numba.centered_window_bounds(2, radius=10**100, owner="numba.public")
    assert huge_radius.start.tolist() == [0, 0]
    assert huge_radius.stop.tolist() == [2, 2]
    huge_width = tal_numba.forward_window_bounds(2, width=10**100, owner="numba.public")
    assert huge_width.start.tolist() == [0, 1]
    assert huge_width.stop.tolist() == [2, 2]

    invalid_cases = [
        lambda: tal_numba.centered_window_bounds(3.0, radius=1, owner="numba.public"),
        lambda: tal_numba.centered_window_bounds(3, radius=1.2, owner="numba.public"),
        lambda: tal_numba.clipped_window_bounds(3, before=True, after=0, owner="numba.public"),
        lambda: tal_numba.forward_window_bounds(3, width=False, owner="numba.public"),
        lambda: tal_numba.forward_window_bounds(3, width=0, owner="numba.public"),
        lambda: tal_numba.centered_window_bounds(-1, radius=1, owner="numba.public"),
        lambda: tal_numba.centered_window_bounds(10**100, radius=1, owner="numba.public"),
    ]
    for call in invalid_cases:
        with pytest.raises(ValueError, match=r"numba\.public:"):
            call()


def test_numba_public_004_public_numba_topology_substrate_requires_explicit_axes() -> None:
    """ID: NUMBA_PUBLIC_004_public_numba_topology_substrate_requires_explicit_axes."""
    prepared = tal_numba.prepare_scan_rows(
        (np.zeros((2, 3, 4)),),
        (tal_numba.ScanInputSpec("links", 1, 1, np.float64),),
        ordered_axes=(tal_numba.ScanAxisSpec("chain", "topology"),),
        output_core_shapes=((4,),),
        owner="numba.public",
    )
    assert prepared.outer_shape == (2,)
    assert prepared.ordered_shape == (3,)

    with pytest.raises(ValueError, match=r"numba\.public: ordered_axes must not be empty"):
        tal_numba.prepare_scan_rows(
            (np.zeros((2, 3, 4)),),
            (tal_numba.ScanInputSpec("links", 1, 1, np.float64),),
            ordered_axes=(),
            output_core_shapes=((4,),),
            owner="numba.public",
        )
