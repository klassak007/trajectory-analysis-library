from __future__ import annotations

import numpy as np
import xarray as xr

from tal import ufuncs as tal_ufuncs
from tal.core import AnalysisObject
from tal.core.event_ops.types import CompareNode, Condition


def _ao(values: np.ndarray) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )


def _ao_missing_sequence_declared(values: np.ndarray) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("trial", "axis"), values)},
        coords={
            "sample": ("sample", np.arange(2, dtype=np.int64)),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "axis": np.arange(values.shape[1], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=False,
    )


def test_ao_op_001_arithmetic_operator_parity() -> None:
    """ID: AO_OP_001_arithmetic_operator_parity."""
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 2.0) / 7.0)

    xr.testing.assert_allclose((left + right).as_dataset(copy="none")["x"], tal_ufuncs.add(left, right).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((left - right).as_dataset(copy="none")["x"], tal_ufuncs.subtract(left, right).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((left * right).as_dataset(copy="none")["x"], tal_ufuncs.multiply(left, right).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((left / right).as_dataset(copy="none")["x"], tal_ufuncs.true_divide(left, right).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((left % 3).as_dataset(copy="none")["x"], tal_ufuncs.mod(left, 3).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((left ** 2).as_dataset(copy="none")["x"], tal_ufuncs.power(left, 2).as_dataset(copy="none")["x"])


def test_ao_op_002_unary_operator_parity() -> None:
    """ID: AO_OP_002_unary_operator_parity."""
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3) - 6.0)

    xr.testing.assert_allclose((-left).as_dataset(copy="none")["x"], tal_ufuncs.negative(left).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((+left).as_dataset(copy="none")["x"], tal_ufuncs.positive(left).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose(abs(left).as_dataset(copy="none")["x"], tal_ufuncs.absolute(left).as_dataset(copy="none")["x"])


def test_ao_op_003_rbinary_operator_paths() -> None:
    """ID: AO_OP_003_rbinary_operator_paths."""
    ao = _ao(np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0)
    xr.testing.assert_allclose((2 + ao).as_dataset(copy="none")["x"], tal_ufuncs.add(2, ao).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((2 - ao).as_dataset(copy="none")["x"], tal_ufuncs.subtract(2, ao).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((2 * ao).as_dataset(copy="none")["x"], tal_ufuncs.multiply(2, ao).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((2 / ao).as_dataset(copy="none")["x"], tal_ufuncs.true_divide(2, ao).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((2 % ao).as_dataset(copy="none")["x"], tal_ufuncs.mod(2, ao).as_dataset(copy="none")["x"])
    xr.testing.assert_allclose((2**ao).as_dataset(copy="none")["x"], tal_ufuncs.power(2, ao).as_dataset(copy="none")["x"])


def test_ao_op_004_comparison_operators_return_condition() -> None:
    """ID: AO_OP_004_comparison_operators_return_condition."""
    ao = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    ops = [ao < 2.0, ao <= 2.0, ao > 2.0, ao >= 2.0]
    for cond in ops:
        assert isinstance(cond, Condition)
        assert isinstance(cond.node, CompareNode)


def test_ao_op_hard_001_analysis_object_hashability_identity_policy_restored() -> None:
    """ID: AO_OP_HARD_001_analysis_object_hashability_identity_policy_restored."""
    ao = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    cache = {ao: "ok"}
    assert cache[ao] == "ok"
    assert isinstance(hash(ao), int)


def test_ao_op_hard_002_analysis_object_eq_ne_identity_semantics() -> None:
    """ID: AO_OP_HARD_002_analysis_object_eq_ne_identity_semantics."""
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    assert (left == left) is True
    assert (left != left) is False
    assert (left == right) is False
    assert (left != right) is True


def test_bcast_core_026_dunder_arithmetic_accepts_broadcast_intent() -> None:
    """ID: BCAST_CORE_026_dunder_arithmetic_accepts_broadcast_intent."""
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 3.0)
    out = left.b() + right
    expected = tal_ufuncs.add(left.b(), right)
    xr.testing.assert_allclose(out.as_dataset(copy="none")["x"], expected.as_dataset(copy="none")["x"])


def test_dunder_arithmetic_semantic_default_without_b_helper() -> None:
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao_missing_sequence_declared((np.arange(6, dtype=float).reshape(2, 3) + 1.0) / 4.0)
    out = left + right
    expected = tal_ufuncs.add(left, right)
    xr.testing.assert_allclose(out.as_dataset(copy="none")["x"], expected.as_dataset(copy="none")["x"])


def test_dunder_arithmetic_accepts_alignment_intent() -> None:
    left = _ao(np.arange(12, dtype=float).reshape(2, 2, 3))
    right = _ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 2.0) / 9.0)
    out = left.a(on="sequence", sequence_join="exact", batch_join="exact") + right
    expected = tal_ufuncs.add(left.a(on="sequence", sequence_join="exact", batch_join="exact"), right)
    xr.testing.assert_allclose(out.as_dataset(copy="none")["x"], expected.as_dataset(copy="none")["x"])
