import numpy as np
import pytest

from tal.core.param_engine.backends import (
    PARAM_BOUNDS_BACKEND_NUMBA,
    PARAM_MAP_BACKEND_NUMBA,
    bounds_block_backend,
    map_block_backend,
)
from tal.core.param_engine.map_build import (
    _DUPLICATE_CODES,
    _bounds_row,
    _map_row,
)


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


def test_numba_opt_005_backends_fail_closed_when_numba_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: NUMBA_OPT_005_primary_backends_fail_closed_when_numba_requested_without_numba."""
    import tal.core.param_engine.numba_backends as numba_backends

    def _raise_import_error():
        raise ImportError("missing numba")

    monkeypatch.setattr(numba_backends, "_import_numba", _raise_import_error)
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
