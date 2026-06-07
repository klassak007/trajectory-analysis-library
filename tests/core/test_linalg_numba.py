from pathlib import Path

import numpy as np
import pytest

from tal.linalg.ops.solve_backends import (
    LSTSQ_BACKEND_NUMBA,
    LSTSQ_BACKEND_NUMPY_ROW,
    _select_lstsq_backend,
    lstsq_block_backend,
)


def _row_count(shape: tuple[int, ...]) -> int:
    rows = 1
    for size in shape:
        rows *= int(size)
    return rows


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
        "Decision: promote",
        "Gate result: PASS",
        "Benchmark evidence:",
        "Reason:",
        "Public routing status:",
        "No-Numba behavior:",
        "Next F2C-B action:",
    ):
        assert required in section
    assert "benchmarks/bench_linalg_lstsq_numba_backends.py" in section
    assert "small-4x2-vector-4096" in section
    assert "fewer-large" in section
    solve_text = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    lstsq_section = solve_text.split("def compute_lstsq_kernel(", 1)[1].split("def compute_solve(", 1)[0]
    assert "LSTSQ_BACKEND_NUMPY_ROW" in lstsq_section
    assert "vectorize=True" in lstsq_section
    assert "LSTSQ_BACKEND_NUMBA" not in lstsq_section
    contract_083 = Path("contracts/083-compiled-kernel-backend-followon-phase-f2.md").read_text(encoding="utf-8")
    assert "Status: Draft" in contract_083
    assert "event/linalg closeout open" in contract_083


def test_linalg_numba_001_lstsq_backend_decision_is_explicit() -> None:
    """ID: LINALG_NUMBA_001_lstsq_backend_decision_is_explicit."""
    assert LSTSQ_BACKEND_NUMPY_ROW == "numpy_row"
    assert LSTSQ_BACKEND_NUMBA == "numba"
    assert _select_lstsq_backend(np.zeros((512, 4, 2)), np.zeros((512, 4)), rhs_is_vector=True) == "numba"
    assert _select_lstsq_backend(np.zeros((511, 4, 2)), np.zeros((511, 4)), rhs_is_vector=True) == "numpy_row"
    assert _select_lstsq_backend(np.zeros((512, 17, 2)), np.zeros((512, 17)), rhs_is_vector=True) == "numpy_row"
    assert _select_lstsq_backend(np.zeros((512, 4, 9)), np.zeros((512, 4)), rhs_is_vector=True) == "numpy_row"
    assert _select_lstsq_backend(np.zeros((512, 4, 2)), np.zeros((512, 4, 9)), rhs_is_vector=False) == "numpy_row"
    assert _select_lstsq_backend(np.zeros((512, 4, 2), dtype=object), np.zeros((512, 4)), rhs_is_vector=True) == "numpy_row"
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
