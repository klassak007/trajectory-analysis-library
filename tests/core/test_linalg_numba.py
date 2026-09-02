from pathlib import Path
import importlib

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.linalg import Matrix, SolveOptions, Vector, solve
from tal.linalg.ops.solve_backends import (
    LSTSQ_BACKEND_NUMBA,
    LSTSQ_BACKEND_NUMPY_BLOCK,
    _select_lstsq_backend,
    lstsq_block_backend,
)

solve_mod = importlib.import_module("tal.linalg.ops.solve")


def _row_count(shape: tuple[int, ...]) -> int:
    rows = 1
    for size in shape:
        rows *= int(size)
    return rows


def _matrix(values: np.ndarray, *, row: str = "eq", col: str = "sol") -> Matrix:
    ds = xr.Dataset(
        {"x": (("batch", row, col), values)},
        coords={
            "batch": np.arange(values.shape[0], dtype=np.int64),
            row: np.arange(values.shape[1], dtype=np.int64),
            col: np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return Matrix(
        AnalysisObject.from_data(
            ds,
            batch_dims=("batch",),
            core_dims=(row, col),
            validate=True,
        )
    )


def _vector(values: np.ndarray, *, axis: str = "eq") -> Vector:
    ds = xr.Dataset(
        {"x": (("batch", axis), values)},
        coords={
            "batch": np.arange(values.shape[0], dtype=np.int64),
            axis: np.arange(values.shape[1], dtype=np.int64),
        },
    )
    return Vector(
        AnalysisObject.from_data(
            ds,
            batch_dims=("batch",),
            core_dims=(axis,),
            validate=True,
        )
    )


def _baseline_lstsq_block(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rcond: float | None,
    rhs_is_vector: bool,
) -> np.ndarray:
    if rhs_is_vector:
        return _baseline_lstsq_vector_block(a, b, rcond=rcond)
    return _baseline_lstsq_matrix_block(a, b, rcond=rcond)


def _baseline_lstsq_vector_block(a: np.ndarray, b: np.ndarray, *, rcond: float | None) -> np.ndarray:
    equations = int(a.shape[-2])
    solutions = int(a.shape[-1])
    outer = np.broadcast_shapes(a.shape[:-2], b.shape[:-1])
    rows = _row_count(outer)
    a_rows = np.broadcast_to(a, outer + (equations, solutions)).reshape(rows, equations, solutions)
    b_rows = np.broadcast_to(b, outer + (equations,)).reshape(rows, equations)
    outputs = [np.linalg.lstsq(a_rows[row], b_rows[row], rcond=rcond)[0] for row in range(rows)]
    return np.stack(outputs).reshape(outer + (solutions,))


def _baseline_lstsq_matrix_block(a: np.ndarray, b: np.ndarray, *, rcond: float | None) -> np.ndarray:
    equations = int(a.shape[-2])
    solutions = int(a.shape[-1])
    rhs_cols = int(b.shape[-1])
    outer = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    rows = _row_count(outer)
    a_rows = np.broadcast_to(a, outer + (equations, solutions)).reshape(rows, equations, solutions)
    b_rows = np.broadcast_to(b, outer + (equations, rhs_cols)).reshape(rows, equations, rhs_cols)
    outputs = [np.linalg.lstsq(a_rows[row], b_rows[row], rcond=rcond)[0] for row in range(rows)]
    return np.stack(outputs).reshape(outer + (solutions, rhs_cols))


def _assert_lstsq_parity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rcond: float | None,
    rhs_is_vector: bool,
) -> None:
    actual = lstsq_block_backend(
        a,
        b,
        rcond=rcond,
        rhs_is_vector=rhs_is_vector,
        backend=LSTSQ_BACKEND_NUMBA,
    )
    expected = _baseline_lstsq_block(a, b, rcond=rcond, rhs_is_vector=rhs_is_vector)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def _decision_section(target: str) -> str:
    text = Path("contracts/114-numba-default-baseline-migration-slice-f2c.md").read_text(encoding="utf-8")
    section = text.split(f"### {target}", 1)[1]
    return section.split("\n### ", 1)[0]


def test_linalg_f2c_001_lstsq_default_or_baseline_migration_decision() -> None:
    """ID: LINALG_F2C_001_lstsq_default_or_baseline_migration_decision."""
    section = _decision_section("linalg_lstsq")
    for required in (
        "Decision: migrated",
        "Gate result: PASS",
        "Benchmark evidence:",
        "Selected normal path: shape-aware numba if available, numpy_block otherwise",
        "Public routing status: blockwise vectorize=False",
        "No-Numba behavior: numpy_block fallback",
        "Explicit Numba behavior: ImportError, no silent fallback",
        "Contract 083 status: all primary F2 targets closed",
    ):
        assert required in section
    assert "benchmarks/bench_linalg_lstsq_numba_backends.py" in section
    assert "small-4x2-vector-4096" in section
    assert "fewer-large" in section
    solve_text = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    lstsq_section = solve_text.split("def compute_lstsq_kernel(", 1)[1].split("def compute_solve(", 1)[0]
    assert "lstsq_block_backend" in lstsq_section
    assert "vectorize=False" in lstsq_section
    assert "vectorize=True" not in lstsq_section
    contract_083 = Path("contracts/083-compiled-kernel-backend-followon-phase-f2.md").read_text(encoding="utf-8")
    assert "Status: Implemented" in contract_083
    assert "all primary F2 targets closed" in contract_083


def test_linalg_f2c_002_lstsq_normal_path_uses_block_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: LINALG_F2C_002_lstsq_normal_path_uses_block_backend."""
    seen = []
    original = solve_mod.lstsq_block_backend

    def _capture(*args: object, **kwargs: object):
        seen.append(kwargs["backend"])
        kwargs = dict(kwargs)
        kwargs["backend"] = LSTSQ_BACKEND_NUMPY_BLOCK
        return original(*args, **kwargs)

    monkeypatch.setattr(solve_mod, "_numba_available", lambda: True)
    monkeypatch.setattr(solve_mod, "lstsq_block_backend", _capture)
    left = _matrix(np.tile(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]), (512, 1, 1)))
    rhs = _vector(np.tile(np.asarray([1.0, 2.0, 3.0, 4.0]), (512, 1)))
    out = solve(left, rhs, opts=SolveOptions(method="lstsq"))
    assert seen == [LSTSQ_BACKEND_NUMBA]
    assert out.as_dataset(copy="none")["datavar"].sizes["batch"] == 512


def test_linalg_f2c_003_lstsq_no_numba_block_fallback_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: LINALG_F2C_003_lstsq_no_numba_block_fallback_parity."""
    seen = []
    original = solve_mod.lstsq_block_backend

    def _capture(*args: object, **kwargs: object):
        seen.append(kwargs["backend"])
        return original(*args, **kwargs)

    monkeypatch.setattr(solve_mod, "_numba_available", lambda: False)
    monkeypatch.setattr(solve_mod, "lstsq_block_backend", _capture)
    left_values = np.tile(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]), (512, 1, 1))
    rhs_values = np.tile(np.asarray([1.0, 2.0, 3.0, 4.0]), (512, 1))
    out = solve(_matrix(left_values), _vector(rhs_values), opts=SolveOptions(method="lstsq"))
    expected = _baseline_lstsq_block(left_values, rhs_values, rcond=None, rhs_is_vector=True)
    assert seen == [LSTSQ_BACKEND_NUMPY_BLOCK]
    np.testing.assert_allclose(out.as_dataset(copy="none")["datavar"].values, expected)
    with pytest.raises(ValueError, match=r"linalg\.solve: block 'a' must include 2 trailing core dimensions"):
        lstsq_block_backend(
            np.asarray([1.0]),
            np.asarray([1.0]),
            rcond=None,
            rhs_is_vector=True,
            backend=LSTSQ_BACKEND_NUMPY_BLOCK,
        )
    with pytest.raises(ValueError, match=r"linalg\.solve: block 'b' must include 1 trailing core dimensions"):
        lstsq_block_backend(
            np.asarray([[[1.0], [2.0]]]),
            np.asarray(1.0),
            rcond=None,
            rhs_is_vector=True,
            backend=LSTSQ_BACKEND_NUMPY_BLOCK,
        )
    with pytest.raises(ValueError, match=r"linalg\.solve: block 'b' must include 2 trailing core dimensions"):
        lstsq_block_backend(
            np.asarray([[[1.0], [2.0]]]),
            np.asarray([1.0, 2.0]),
            rcond=None,
            rhs_is_vector=False,
            backend=LSTSQ_BACKEND_NUMPY_BLOCK,
        )


def test_numba_opt_014_linalg_lstsq_default_migration_falls_back_without_numba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: NUMBA_OPT_014_linalg_lstsq_default_migration_falls_back_without_numba."""
    monkeypatch.setattr(solve_mod, "_numba_available", lambda: False)
    left = _matrix(np.asarray([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]))
    rhs = _vector(np.asarray([[1.0, 2.0, 4.0]]))
    out = solve(left, rhs, opts=SolveOptions(method="lstsq"))
    expected = _baseline_lstsq_block(left.as_dataset(copy="none")["x"].values, rhs.as_dataset(copy="none")["x"].values, rcond=None, rhs_is_vector=True)
    np.testing.assert_allclose(out.as_dataset(copy="none")["datavar"].values, expected)


def test_numba_opt_015_linalg_lstsq_explicit_numba_still_fails_closed_after_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: NUMBA_OPT_015_linalg_lstsq_explicit_numba_still_fails_closed_after_migration."""
    import tal.utils.numba_support as numba_support

    def _raise_import_error():
        raise ImportError("missing numba")

    monkeypatch.setattr(numba_support, "_import_numba", _raise_import_error)
    with pytest.raises(ImportError, match=r"linalg.solve: numba is required for backend='numba'"):
        lstsq_block_backend(
            np.asarray([[[1.0], [2.0]]]),
            np.asarray([[1.0, 2.0]]),
            rcond=None,
            rhs_is_vector=True,
            backend=LSTSQ_BACKEND_NUMBA,
        )


def test_linalg_numba_001_lstsq_backend_decision_is_explicit() -> None:
    """ID: LINALG_NUMBA_001_lstsq_backend_decision_is_explicit."""
    assert LSTSQ_BACKEND_NUMPY_BLOCK == "numpy_block"
    assert LSTSQ_BACKEND_NUMBA == "numba"
    assert _select_lstsq_backend(np.zeros((512, 4, 2)), np.zeros((512, 4)), rhs_is_vector=True) == "numba"
    assert _select_lstsq_backend(np.zeros((511, 4, 2)), np.zeros((511, 4)), rhs_is_vector=True) == "numpy_block"
    assert _select_lstsq_backend(np.zeros((512, 17, 2)), np.zeros((512, 17)), rhs_is_vector=True) == "numpy_block"
    assert _select_lstsq_backend(np.zeros((512, 4, 9)), np.zeros((512, 4)), rhs_is_vector=True) == "numpy_block"
    assert _select_lstsq_backend(np.zeros((512, 4, 2)), np.zeros((512, 4, 9)), rhs_is_vector=False) == "numpy_block"
    assert _select_lstsq_backend(np.zeros((512, 4, 2), dtype=object), np.zeros((512, 4)), rhs_is_vector=True) == "numpy_block"
    with pytest.raises(ValueError, match=r"linalg.solve: unsupported lstsq backend 'unknown'"):
        lstsq_block_backend(
            np.asarray([[[1.0], [2.0]]]),
            np.asarray([[1.0, 2.0]]),
            rcond=None,
            rhs_is_vector=True,
            backend="unknown",
        )


def test_linalg_numba_002_lstsq_numba_backend_parity_if_implemented() -> None:
    """ID: LINALG_NUMBA_002_lstsq_numba_backend_parity_if_implemented."""
    pytest.importorskip("numba")
    vector_a = np.asarray([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    vector_b = np.asarray([[1.0, 2.0, 4.0]])
    _assert_lstsq_parity(vector_a, vector_b, rcond=None, rhs_is_vector=True)

    matrix_b = np.asarray([[[1.0, 2.0], [2.0, 0.0], [4.0, 1.0]]])
    _assert_lstsq_parity(vector_a, matrix_b, rcond=None, rhs_is_vector=False)
    _assert_lstsq_parity(vector_a, vector_b, rcond=1e-3, rhs_is_vector=True)

    int_a = np.asarray([[[1, 0], [0, 1], [1, 1]]], dtype=np.int64)
    int_b = np.asarray([[1, 2, 4]], dtype=np.int64)
    _assert_lstsq_parity(int_a, int_b, rcond=None, rhs_is_vector=True)

    f32_a = vector_a.astype(np.float32)
    f32_b = vector_b.astype(np.float32)
    _assert_lstsq_parity(f32_a, f32_b, rcond=None, rhs_is_vector=True)
    _assert_lstsq_parity(f32_a, int_b.astype(np.int32), rcond=None, rhs_is_vector=True)

    complex_a = (vector_a + 0.25j * vector_a).astype(np.complex64)
    complex_b = (vector_b - 0.5j * vector_b).astype(np.complex64)
    _assert_lstsq_parity(complex_a, complex_b, rcond=None, rhs_is_vector=True)

    broadcast_a = np.broadcast_to(vector_a, (2, 1, 3, 2)).copy()
    broadcast_b = np.broadcast_to(vector_b[0], (3, 3)).copy()
    _assert_lstsq_parity(broadcast_a, broadcast_b, rcond=None, rhs_is_vector=True)
