from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tal.core.param_engine.backends import (
    PARAM_BOUNDS_BACKEND_NUMBA,
    PARAM_BOUNDS_BACKEND_NUMPY_BLOCK,
    PARAM_MAP_BACKEND_NUMBA,
    PARAM_MAP_BACKEND_NUMPY_BLOCK,
    bounds_block_backend,
    map_block_backend,
)
from tal.core.event_ops.backends import (
    EVENT_BOUNDARY_BACKEND_NUMBA,
    EVENT_INTERVALS_BACKEND_NUMBA,
    boundary_bounded_block_backend,
    intervals_bounded_block_backend,
)
from tal.linalg.ops.solve_backends import LSTSQ_BACKEND_NUMBA, lstsq_block_backend
from tal.core.param_engine.map_build import (
    _DUPLICATE_CODES,
    _bounds_row,
    _map_row,
    build_param_bounds_map,
    build_param_map,
)
import tal.core.param_engine.map_build as map_build_mod


def _require_numba() -> None:
    pytest.importorskip("numba")


def _baseline_map_block(
    param: np.ndarray,
    valid: np.ndarray,
    query: np.ndarray,
    *,
    method: str = "linear",
    dup_code: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outer = np.broadcast_shapes(param.shape[:-1], valid.shape[:-1], query.shape[:-1])
    rows = int(np.prod(outer, dtype=np.int64)) if outer else 1
    param_rows = np.broadcast_to(param, outer + (param.shape[-1],)).reshape(rows, param.shape[-1])
    valid_rows = np.broadcast_to(valid, outer + (valid.shape[-1],)).reshape(rows, valid.shape[-1])
    query_rows = np.broadcast_to(query, outer + (query.shape[-1],)).reshape(rows, query.shape[-1])
    outputs = [
        _map_row(param_rows[row], valid_rows[row], query_rows[row], method=method, dup_code=dup_code)
        for row in range(rows)
    ]
    return tuple(np.stack([out[idx] for out in outputs]).reshape(outer + (query.shape[-1],)) for idx in range(4))


def _baseline_bounds_block(
    param: np.ndarray,
    valid: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outer = np.broadcast_shapes(param.shape[:-1], valid.shape[:-1], start.shape, stop.shape)
    rows = int(np.prod(outer, dtype=np.int64)) if outer else 1
    param_rows = np.broadcast_to(param, outer + (param.shape[-1],)).reshape(rows, param.shape[-1])
    valid_rows = np.broadcast_to(valid, outer + (valid.shape[-1],)).reshape(rows, valid.shape[-1])
    start_rows = np.broadcast_to(start, outer).reshape(rows)
    stop_rows = np.broadcast_to(stop, outer).reshape(rows)
    outputs = [_bounds_row(param_rows[row], valid_rows[row], start_rows[row], stop_rows[row]) for row in range(rows)]
    return tuple(np.stack([out[idx] for out in outputs]).reshape(outer) for idx in range(2))


def _assert_map_equal(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2])
    np.testing.assert_array_equal(actual[3], expected[3])


def _decision_section(target: str) -> str:
    text = Path("contracts/114-numba-default-baseline-migration-slice-f2c.md").read_text(encoding="utf-8")
    section = text.split(f"### {target}", 1)[1]
    return section.split("\n### ", 1)[0]


def _assert_migrated_decision_record(target: str, benchmark: str, cases: tuple[str, ...]) -> None:
    section = _decision_section(target)
    for required in (
        "Decision: migrated",
        "Gate result: PASS",
        "Benchmark evidence:",
        "Selected normal path: numba if available, numpy_block otherwise",
        "Public routing status: blockwise vectorize=False",
        "No-Numba behavior: numpy_block fallback",
        "Explicit Numba behavior: ImportError, no silent fallback",
        "Contract 083 status: all primary F2 targets closed",
    ):
        assert required in section
    assert benchmark in section
    for case in cases:
        assert case in section
    contract_083 = Path("contracts/083-compiled-kernel-backend-followon-phase-f2.md").read_text(encoding="utf-8")
    assert "Status: Implemented" in contract_083
    assert "all primary F2 targets closed" in contract_083


def test_param_f2c_001_map_default_or_baseline_migration_decision() -> None:
    """ID: PARAM_F2C_001_map_default_or_baseline_migration_decision."""
    _assert_migrated_decision_record("param_map", "benchmarks/bench_param_numba_backends.py", ("many-short", "fewer-long"))
    text = Path("tal/core/param_engine/map_build.py").read_text(encoding="utf-8")
    section = text.split("def _apply_param_map_block(", 1)[1].split(
        "def build_param_map(",
        1,
    )[0]
    assert "map_block_backend" in section
    assert "vectorize=False" in section
    assert "vectorize=True" not in section


def test_param_f2c_002_bounds_default_or_baseline_migration_decision() -> None:
    """ID: PARAM_F2C_002_bounds_default_or_baseline_migration_decision."""
    _assert_migrated_decision_record("param_bounds", "benchmarks/bench_param_numba_backends.py", ("many-short", "fewer-long"))
    text = Path("tal/core/param_engine/map_build.py").read_text(encoding="utf-8")
    section = text.split("def _apply_param_bounds_block(", 1)[1].split(
        "def build_param_bounds_map(",
        1,
    )[0]
    assert "bounds_block_backend" in section
    assert "vectorize=False" in section
    assert "vectorize=True" not in section


def test_param_f2c_003_map_normal_path_uses_block_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: PARAM_F2C_003_map_normal_path_uses_block_backend."""
    seen = []
    original = map_build_mod.map_block_backend

    def _capture(*args: object, **kwargs: object):
        seen.append(kwargs["backend"])
        kwargs = dict(kwargs)
        kwargs["backend"] = PARAM_MAP_BACKEND_NUMPY_BLOCK
        return original(*args, **kwargs)

    monkeypatch.setattr(map_build_mod, "_numba_available", lambda: True)
    monkeypatch.setattr(map_build_mod, "map_block_backend", _capture)
    pmap = build_param_map(
        param=xr.DataArray(np.asarray([0.0, 1.0, 2.0]), dims=("sample",)),
        query=xr.DataArray(np.asarray([0.5, np.nan]), dims=("query",)),
        sequence_dim="sample",
        query_dim="query",
    )
    assert seen == [PARAM_MAP_BACKEND_NUMBA]
    np.testing.assert_array_equal(pmap.valid.values, [True, False])


def test_param_f2c_004_bounds_normal_path_uses_block_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: PARAM_F2C_004_bounds_normal_path_uses_block_backend."""
    seen = []
    original = map_build_mod.bounds_block_backend

    def _capture(*args: object, **kwargs: object):
        seen.append(kwargs["backend"])
        kwargs = dict(kwargs)
        kwargs["backend"] = PARAM_BOUNDS_BACKEND_NUMPY_BLOCK
        return original(*args, **kwargs)

    monkeypatch.setattr(map_build_mod, "_numba_available", lambda: True)
    monkeypatch.setattr(map_build_mod, "bounds_block_backend", _capture)
    bounds = build_param_bounds_map(
        param=xr.DataArray(np.asarray([0.0, 1.0, 2.0]), dims=("sample",)),
        start=xr.DataArray(0.25),
        stop=xr.DataArray(1.75),
        sequence_dim="sample",
    )
    assert seen == [PARAM_BOUNDS_BACKEND_NUMBA]
    assert int(bounds.i0.values) == 1
    assert int(bounds.i1.values) == 2


def test_param_f2c_005_param_no_numba_block_fallback_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: PARAM_F2C_005_param_no_numba_block_fallback_parity."""
    seen = []
    original_map = map_build_mod.map_block_backend
    original_bounds = map_build_mod.bounds_block_backend

    def _capture_map(*args: object, **kwargs: object):
        seen.append(kwargs["backend"])
        return original_map(*args, **kwargs)

    def _capture_bounds(*args: object, **kwargs: object):
        seen.append(kwargs["backend"])
        return original_bounds(*args, **kwargs)

    monkeypatch.setattr(map_build_mod, "_numba_available", lambda: False)
    monkeypatch.setattr(map_build_mod, "map_block_backend", _capture_map)
    monkeypatch.setattr(map_build_mod, "bounds_block_backend", _capture_bounds)
    param = xr.DataArray(np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, np.nan]]), dims=("trial", "sample"))
    valid = xr.DataArray(np.asarray([[True, True, True], [True, True, True]]), dims=("trial", "sample"))
    query = xr.DataArray(np.asarray([[0.5, 2.0], [10.5, 12.0]]), dims=("trial", "query"))
    pmap = build_param_map(param=param, query=query, sequence_dim="sample", query_dim="query", valid_mask=valid)
    bounds = build_param_bounds_map(
        param=param,
        start=xr.DataArray(np.asarray([0.25, 10.25]), dims=("trial",)),
        stop=xr.DataArray(np.asarray([1.75, 12.0]), dims=("trial",)),
        sequence_dim="sample",
        valid_mask=valid,
    )
    assert seen == [PARAM_MAP_BACKEND_NUMPY_BLOCK, PARAM_BOUNDS_BACKEND_NUMPY_BLOCK]
    np.testing.assert_array_equal(pmap.valid.values, [[True, True], [True, False]])
    np.testing.assert_array_equal(bounds.i0.values, [1, 1])
    np.testing.assert_array_equal(bounds.i1.values, [2, 2])


def test_numba_opt_010_param_default_migration_falls_back_without_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: NUMBA_OPT_010_param_default_migration_falls_back_without_numba."""
    monkeypatch.setattr(map_build_mod, "_numba_available", lambda: False)
    pmap = build_param_map(
        param=xr.DataArray(np.asarray([0.0, 1.0, 2.0]), dims=("sample",)),
        query=xr.DataArray(np.asarray([0.5]), dims=("query",)),
        sequence_dim="sample",
        query_dim="query",
    )
    bounds = build_param_bounds_map(
        param=xr.DataArray(np.asarray([0.0, 1.0, 2.0]), dims=("sample",)),
        start=xr.DataArray(0.5),
        stop=xr.DataArray(1.5),
        sequence_dim="sample",
    )
    np.testing.assert_array_equal(pmap.valid.values, [True])
    assert int(bounds.i0.values) == 1
    assert int(bounds.i1.values) == 2


def test_numba_opt_011_param_explicit_numba_still_fails_closed_after_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: NUMBA_OPT_011_param_explicit_numba_still_fails_closed_after_migration."""
    import tal.utils.numba_support as numba_support

    def _raise_import_error():
        raise ImportError("missing numba")

    monkeypatch.setattr(numba_support, "_import_numba", _raise_import_error)
    with pytest.raises(ImportError, match=r"build_param_map: numba is required for backend='numba'"):
        map_block_backend(
            np.asarray([[0.0, 1.0]]),
            np.asarray([[True, True]]),
            np.asarray([[0.5]]),
            method="linear",
            dup_code=0,
            backend=PARAM_MAP_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"build_param_bounds_map: numba is required for backend='numba'"):
        bounds_block_backend(
            np.asarray([[0.0, 1.0]]),
            np.asarray([[True, True]]),
            np.asarray([0.0]),
            np.asarray([1.0]),
            backend=PARAM_BOUNDS_BACKEND_NUMBA,
        )


def test_numba_opt_005_backends_fail_closed_when_numba_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: NUMBA_OPT_005_primary_backends_fail_closed_when_numba_requested_without_numba."""
    import tal.utils.numba_support as numba_support

    def _raise_import_error():
        raise ImportError("missing numba")

    monkeypatch.setattr(numba_support, "_import_numba", _raise_import_error)
    with pytest.raises(ImportError, match=r"build_param_map: numba is required for backend='numba'"):
        map_block_backend(
            np.asarray([[0.0, 1.0]]),
            np.asarray([[True, True]]),
            np.asarray([[0.5]]),
            method="linear",
            dup_code=0,
            backend=PARAM_MAP_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"build_param_bounds_map: numba is required for backend='numba'"):
        bounds_block_backend(
            np.asarray([[0.0, 1.0]]),
            np.asarray([[True, True]]),
            np.asarray([0.0]),
            np.asarray([1.0]),
            backend=PARAM_BOUNDS_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"events.boundaries: numba is required for backend='numba'"):
        boundary_bounded_block_backend(
            np.asarray([[False, True]]),
            np.asarray([[True, True]]),
            np.asarray([[0.0, 1.0]]),
            include_initial=False,
            emit_triggers=False,
            dedupe_atol=0.0,
            max_events=2,
            owner="events.boundaries",
            backend=EVENT_BOUNDARY_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"events.intervals: numba is required for backend='numba'"):
        intervals_bounded_block_backend(
            np.asarray([[1.0, 2.0]]),
            np.asarray([[1, 2]], dtype="int8"),
            np.asarray([[-1, 1]]),
            np.asarray([[1, -1]]),
            max_segments=1,
            owner="events.intervals",
            backend=EVENT_INTERVALS_BACKEND_NUMBA,
        )
    with pytest.raises(ImportError, match=r"linalg.solve: numba is required for backend='numba'"):
        lstsq_block_backend(
            np.asarray([[[1.0], [2.0]]]),
            np.asarray([[1.0, 2.0]]),
            rcond=None,
            rhs_is_vector=True,
            backend=LSTSQ_BACKEND_NUMBA,
        )


def test_param_numba_001_map_linear_backend_parity() -> None:
    """ID: PARAM_NUMBA_001_map_linear_backend_parity."""
    _require_numba()
    param = np.asarray([[0.0, 1.0, 2.0, 4.0], [10.0, 11.0, np.nan, 13.0]])
    valid = np.asarray([[True, True, True, True], [True, True, True, True]])
    query = np.asarray([[0.5, 3.0, np.nan], [10.5, 12.0, 14.0]])
    actual = map_block_backend(
        param,
        valid,
        query,
        method="linear",
        dup_code=_DUPLICATE_CODES["invalid"],
        backend=PARAM_MAP_BACKEND_NUMBA,
    )
    expected = _baseline_map_block(param, valid, query)
    _assert_map_equal(actual, expected)


def test_param_numba_002_map_nearest_backend_parity() -> None:
    """ID: PARAM_NUMBA_002_map_nearest_backend_parity."""
    _require_numba()
    param = np.asarray([[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]])
    valid = np.ones_like(param, dtype=bool)
    query = np.asarray([[1.0, 3.1, np.nan], [11.0, 13.5, 20.0]])
    actual = map_block_backend(
        param,
        valid,
        query,
        method="nearest",
        dup_code=_DUPLICATE_CODES["invalid"],
        backend=PARAM_MAP_BACKEND_NUMBA,
    )
    expected = _baseline_map_block(param, valid, query, method="nearest")
    _assert_map_equal(actual, expected)


def test_param_numba_003_map_duplicate_policy_parity() -> None:
    """ID: PARAM_NUMBA_003_map_duplicate_policy_parity."""
    _require_numba()
    param = np.asarray([[0.0, 1.0, 1.0, 2.0]])
    valid = np.asarray([[True, True, True, True]])
    query = np.asarray([[1.0]])
    for policy in ("invalid", "left", "right"):
        dup_code = _DUPLICATE_CODES[policy]
        actual = map_block_backend(
            param,
            valid,
            query,
            method="linear",
            dup_code=dup_code,
            backend=PARAM_MAP_BACKEND_NUMBA,
        )
        expected = _baseline_map_block(param, valid, query, dup_code=dup_code)
        _assert_map_equal(actual, expected)
    with pytest.raises(ValueError, match="duplicate parameter bracket"):
        map_block_backend(
            param,
            valid,
            query,
            method="linear",
            dup_code=_DUPLICATE_CODES["raise"],
            backend=PARAM_MAP_BACKEND_NUMBA,
        )


def test_param_numba_004_map_monotonic_fail_closed() -> None:
    """ID: PARAM_NUMBA_004_map_monotonic_fail_closed."""
    _require_numba()
    with pytest.raises(ValueError, match="monotonic non-decreasing"):
        map_block_backend(
            np.asarray([[0.0, 2.0, 1.0]]),
            np.asarray([[True, True, True]]),
            np.asarray([[1.0]]),
            method="linear",
            dup_code=0,
            backend=PARAM_MAP_BACKEND_NUMBA,
        )


def test_param_numba_007_map_broadcast_and_dtype_parity() -> None:
    """ID: PARAM_NUMBA_007_map_broadcast_and_dtype_parity."""
    _require_numba()
    param = np.asarray([[[0.0, 1.0, 2.0, 3.0]], [[10.0, 11.0, 12.0, 13.0]]])
    valid = np.ones_like(param, dtype=bool)
    query = np.asarray([0.5, 2.5, np.nan])
    actual = map_block_backend(param, valid, query, method="linear", dup_code=0, backend=PARAM_MAP_BACKEND_NUMBA)
    expected = _baseline_map_block(param, valid, query)
    _assert_map_equal(actual, expected)
    assert actual[0].dtype == np.int64
    assert actual[1].dtype == np.int64
    assert actual[2].dtype == np.float64
    assert actual[3].dtype == bool


def test_param_numba_005_bounds_backend_parity() -> None:
    """ID: PARAM_NUMBA_005_bounds_backend_parity."""
    _require_numba()
    param = np.asarray([[0.0, 1.0, 2.0, 4.0], [10.0, 11.0, np.nan, 13.0]])
    valid = np.asarray([[True, True, True, True], [True, True, True, True]])
    start = np.asarray([0.5, 10.5])
    stop = np.asarray([3.0, 13.0])
    actual = bounds_block_backend(param, valid, start, stop, backend=PARAM_BOUNDS_BACKEND_NUMBA)
    expected = _baseline_bounds_block(param, valid, start, stop)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[0].dtype == np.int64
    assert actual[1].dtype == np.int64


def test_param_numba_006_bounds_empty_nan_no_overlap_parity() -> None:
    """ID: PARAM_NUMBA_006_bounds_empty_nan_no_overlap_parity."""
    _require_numba()
    param = np.asarray([[np.nan, np.nan], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    valid = np.asarray([[False, False], [True, True], [True, True], [True, True]])
    start = np.asarray([0.0, np.nan, 2.0, -2.0])
    stop = np.asarray([1.0, 1.0, 3.0, -1.0])
    actual = bounds_block_backend(param, valid, start, stop, backend=PARAM_BOUNDS_BACKEND_NUMBA)
    expected = _baseline_bounds_block(param, valid, start, stop)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_param_numba_008_bounds_monotonic_fail_closed() -> None:
    """ID: PARAM_NUMBA_008_bounds_monotonic_fail_closed."""
    _require_numba()
    with pytest.raises(ValueError, match="monotonic non-decreasing"):
        bounds_block_backend(
            np.asarray([[0.0, 2.0, 1.0]]),
            np.asarray([[True, True, True]]),
            np.asarray([0.0]),
            np.asarray([2.0]),
            backend=PARAM_BOUNDS_BACKEND_NUMBA,
        )
