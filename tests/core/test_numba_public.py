from __future__ import annotations

import subprocess
import warnings

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


def test_numba_public_005_public_numba_window_iteration_metadata_is_policy_free() -> None:
    """ID: NUMBA_PUBLIC_005_public_numba_window_iteration_metadata_is_policy_free."""
    bounds = tal_numba.WindowBounds(
        start=np.asarray([0.0, 1, 1.0], dtype=object),
        stop=np.asarray([1, 2.0, 3], dtype=object),
    )
    rows = tal_numba.prepare_window_rows(bounds, owner="numba.public")
    assert isinstance(rows, tal_numba.WindowRows)
    assert rows.length == 3
    assert rows.bounds.start.dtype == np.dtype(np.int64)
    assert rows.bounds.stop.dtype == np.dtype(np.int64)
    assert rows.bounds.start.tolist() == [0, 1, 1]
    assert rows.bounds.stop.tolist() == [1, 2, 3]
    assert rows.widths.tolist() == [1, 1, 2]
    assert rows.max_width == 2

    safe_float = tal_numba.prepare_window_rows(
        tal_numba.WindowBounds(np.asarray([0.0]), np.asarray([1.0])),
        owner="numba.public",
    )
    assert safe_float.bounds.start.tolist() == [0]
    assert safe_float.bounds.stop.tolist() == [1]

    empty = tal_numba.prepare_window_rows(
        tal_numba.WindowBounds(np.asarray([], dtype=float), np.asarray([], dtype=object)),
        owner="numba.public",
    )
    assert empty.length == 0
    assert empty.widths.dtype == np.dtype(np.int64)
    assert empty.widths.tolist() == []
    assert empty.max_width == 0

    invalid_bounds = [
        object(),
        tal_numba.WindowBounds(np.asarray([[0]], dtype=np.int64), np.asarray([[1]], dtype=np.int64)),
        tal_numba.WindowBounds(np.asarray([False]), np.asarray([1])),
        tal_numba.WindowBounds(np.asarray([0.5]), np.asarray([1])),
        tal_numba.WindowBounds(np.asarray([np.inf]), np.asarray([1])),
        tal_numba.WindowBounds(np.asarray(["0"]), np.asarray([1])),
        tal_numba.WindowBounds(np.asarray([10**100], dtype=object), np.asarray([1])),
        tal_numba.WindowBounds(np.asarray([0, 1]), np.asarray([1])),
        tal_numba.WindowBounds(np.asarray([-1]), np.asarray([1])),
        tal_numba.WindowBounds(np.asarray([1]), np.asarray([0])),
        tal_numba.WindowBounds(np.asarray([0]), np.asarray([2])),
    ]
    for invalid in invalid_bounds:
        with pytest.raises(ValueError, match=r"numba\.public:"):
            tal_numba.prepare_window_rows(invalid, owner="numba.public")  # type: ignore[arg-type]

    unsafe_float_stop = tal_numba.WindowBounds(
        np.asarray([0.0]),
        np.asarray([float(np.iinfo(np.int64).max)]),
    )
    unsafe_object_stop = tal_numba.WindowBounds(
        np.asarray([0.0], dtype=object),
        np.asarray([float(np.iinfo(np.int64).max)], dtype=object),
    )
    for invalid in (unsafe_float_stop, unsafe_object_stop):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", RuntimeWarning)
            with pytest.raises(ValueError, match=r"numba\.public: window stop bounds must fit int64"):
                tal_numba.prepare_window_rows(invalid, owner="numba.public")
        assert recorded == []


def test_numba_public_006_public_numba_topology_rows_require_explicit_axis() -> None:
    """ID: NUMBA_PUBLIC_006_public_numba_topology_rows_require_explicit_axis."""
    values = np.zeros((2, 4, 3), dtype=np.float64)
    rows = tal_numba.prepare_topology_rows(
        (values,),
        (tal_numba.ScanInputSpec("links", 1, 1, np.float64),),
        topology_axis="chain",
        output_core_shapes=((3,),),
        owner="numba.public",
    )
    assert rows.outer_shape == (2,)
    assert rows.ordered_shape == (4,)
    assert rows.output_shapes == ((2, 4, 3),)

    nested = tal_numba.prepare_scan_rows(
        (np.zeros((2, 5, 4, 3), dtype=np.float64),),
        (tal_numba.ScanInputSpec("links", 2, 1, np.float64),),
        ordered_axes=(
            tal_numba.ScanAxisSpec("time", "scan"),
            tal_numba.ScanAxisSpec("chain", "topology"),
        ),
        output_core_shapes=((3,),),
        owner="numba.public",
    )
    assert nested.outer_shape == (2,)
    assert nested.ordered_shape == (5, 4)

    for invalid_axis in ("", "   ", 3, None):
        with pytest.raises(ValueError, match=r"numba\.public: topology_axis must be a non-empty string"):
            tal_numba.prepare_topology_rows(
                (values,),
                (tal_numba.ScanInputSpec("links", 1, 1, np.float64),),
                topology_axis=invalid_axis,  # type: ignore[arg-type]
                output_core_shapes=((3,),),
                owner="numba.public",
            )

    with pytest.raises(ValueError, match=r"numba\.public: block 'links' ordered_ndim must match ordered_axes"):
        tal_numba.prepare_topology_rows(
            (np.zeros((2, 5, 4, 3), dtype=np.float64),),
            (tal_numba.ScanInputSpec("links", 2, 1, np.float64),),
            topology_axis="chain",
            output_core_shapes=((3,),),
            owner="numba.public",
        )


def test_numba_public_007_public_numba_benchmark_helpers_are_optional_dependency_safe(
    tmp_path,
) -> None:
    """ID: NUMBA_PUBLIC_007_public_numba_benchmark_helpers_are_optional_dependency_safe."""
    assert tal_numba.time_once(lambda value: value + 1, 1) >= 0.0
    assert tal_numba.warm_median(lambda value: value + 1, 1, repeats=1) >= 0.0
    assert tal_numba.break_even_calls(10.0, 25.0, 5.0) == 4.0
    assert np.isinf(tal_numba.break_even_calls(10.0, 25.0, 10.0))

    for repeats in (0, True, 1.5):
        with pytest.raises(ValueError, match=r"repeats must be a positive integer"):
            tal_numba.warm_median(lambda: None, repeats=repeats)  # type: ignore[arg-type]

    success = tmp_path / "success.py"
    success.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "cache = os.environ.get('NUMBA_CACHE_DIR', '')\n"
        "print('0.125' if cache and Path(cache).is_dir() else 'bad')\n",
        encoding="utf-8",
    )
    assert tal_numba.cold_subprocess(str(success), (), cache_prefix="tal-test-numba-") == 0.125

    bad_stdout = tmp_path / "bad_stdout.py"
    bad_stdout.write_text("print('not-a-float')\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cold subprocess stdout must contain a float"):
        tal_numba.cold_subprocess(str(bad_stdout), (), cache_prefix="tal-test-numba-")

    failure = tmp_path / "failure.py"
    failure.write_text(
        "import sys\n"
        "sys.stderr.write('boom')\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        tal_numba.cold_subprocess(str(failure), (), cache_prefix="tal-test-numba-")
    assert "boom" in exc_info.value.stderr
