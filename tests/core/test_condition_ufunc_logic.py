from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal import ufuncs as tal_ufuncs
from tal.core import AnalysisObject
from tal.core.event_ops.types import CompareNode, Condition


def _ao() -> AnalysisObject:
    values = np.asarray([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]], dtype=float)
    time = np.tile(np.asarray([0.0, 1.0, 2.0], dtype=float), (2, 1))
    ds = xr.Dataset(
        {"value": (("trial", "sample"), values)},
        coords={
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "sample": np.arange(values.shape[1], dtype=np.int64),
            "time": (("trial", "sample"), time),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        validate=True,
    )


def test_cond_ufunc_001_comparison_ufuncs_return_condition() -> None:
    """ID: COND_UFUNC_001_comparison_ufuncs_return_condition."""
    cond = tal_ufuncs.greater(Condition.var("value"), 1.0)
    assert isinstance(cond, Condition)
    assert isinstance(cond.node, CompareNode)
    assert cond.node.op == "gt"


def test_cond_ufunc_002_logical_ufuncs_match_condition_operators() -> None:
    """ID: COND_UFUNC_002_logical_ufuncs_match_condition_operators."""
    ao = _ao()
    low = tal_ufuncs.greater(Condition.var("value"), 0.5)
    high = tal_ufuncs.less(Condition.var("value"), 2.0)
    via_ops = (low & high) | ~low
    via_ufunc = tal_ufuncs.logical_or(tal_ufuncs.logical_and(low, high), tal_ufuncs.logical_not(low))
    xr.testing.assert_identical(ao.events.mask(via_ops), ao.events.mask(via_ufunc))


def test_cond_ufunc_003_logical_xor_parity_with_condition_xor() -> None:
    """ID: COND_UFUNC_003_logical_xor_parity_with_condition_xor."""
    ao = _ao()
    low = tal_ufuncs.greater(Condition.var("value"), 0.5)
    high = tal_ufuncs.less(Condition.var("value"), 2.0)
    via_xor_op = low ^ high
    via_xor_ufunc = tal_ufuncs.logical_xor(low, high)
    xr.testing.assert_identical(ao.events.mask(via_xor_op), ao.events.mask(via_xor_ufunc))


def test_cond_ufunc_hard_001_invalid_logical_operands_fail_closed() -> None:
    """ID: COND_UFUNC_HARD_001_invalid_logical_operands_fail_closed."""
    low = tal_ufuncs.greater(Condition.var("value"), 0.5)
    with pytest.raises(TypeError, match="operand must be Condition"):
        _ = tal_ufuncs.logical_not(1.0)
    with pytest.raises(TypeError, match="operand must be Condition"):
        _ = tal_ufuncs.logical_and(low, 1.0)


def test_cond_ufunc_hard_002_condition_bool_raises_fail_fast() -> None:
    """ID: COND_UFUNC_HARD_002_condition_bool_raises_fail_fast."""
    cond = tal_ufuncs.greater(Condition.var("value"), 0.5)
    with pytest.raises(TypeError, match="does not support truth-value testing"):
        bool(cond)
