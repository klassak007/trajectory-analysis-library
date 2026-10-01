from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

import tal.core.event_ops.evaluate as event_eval_mod
from tal.core import AnalysisObject
from tal.core.event_ops import (
    Condition,
    ConditionEvalOptions,
    EventExtractOptions,
    WhenOptions,
)
from tal.core.orchestration.axis_map import (
    resolve_role_axis_map as resolve_role_axis_map_owner,
)
from tal.utils.xarray_namespace import (
    rename_dims_collision_safe as rename_dims_collision_safe_owner,
)


def _ao_series(
    *,
    values: list[float],
    time: list[float],
    sequence_size: int | None = None,
    name: str = "value",
) -> AnalysisObject:
    coords: dict[str, object] = {
        "sample": np.arange(len(values), dtype="int64"),
        "time": ("sample", np.asarray(time, dtype="float64")),
    }
    kwargs: dict[str, object] = {
        "sequence_dim": "sample",
        "batch_dims": (),
        "core_dims": (),
        "param_coord": "time",
    }
    if sequence_size is not None:
        coords["group_size"] = xr.DataArray(np.asarray(sequence_size, dtype="int64"), dims=())
        kwargs["sequence_size_coord"] = "group_size"
    ds = xr.Dataset(
        data_vars={name: (("sample",), np.asarray(values, dtype="float64"))},
        coords=coords,
    )
    return AnalysisObject.from_data(ds, **kwargs)


def _ao_with_clock(clock: np.ndarray) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray([0.25, 0.75], dtype="float64"))},
        coords={
            "sample": np.arange(2, dtype="int64"),
            "time": (("sample",), clock),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )


def test_event_cond_001_three_valued_not_preserves_unknown() -> None:
    """ID: EVENT_COND_001_three_valued_not_preserves_unknown."""
    ao = _ao_series(values=[0.0, 1.0, 2.0], time=[0.0, 1.0, np.nan])
    cond = Condition.compare(Condition.coord("time"), "eq", 1.0)
    mask = ao.events.mask(cond)
    inv = ao.events.mask(~cond)
    np.testing.assert_array_equal(mask.values, np.asarray([False, True, False], dtype=bool))
    np.testing.assert_array_equal(inv.values, np.asarray([True, False, False], dtype=bool))


def test_event_cond_002_and_or_short_circuit_determinate_semantics() -> None:
    """ID: EVENT_COND_002_and_or_short_circuit_determinate_semantics."""
    ao = _ao_series(values=[0.0, np.nan, 20.0], time=[0.0, 1.0, 2.0])
    low = Condition.compare(Condition.var("value"), "lt", 5.0)
    high = Condition.compare(Condition.var("value"), "gt", 10.0)
    and_mask = ao.events.mask(low & high)
    or_mask = ao.events.mask(low | high)
    np.testing.assert_array_equal(and_mask.values, np.asarray([False, False, False], dtype=bool))
    np.testing.assert_array_equal(or_mask.values, np.asarray([True, False, True], dtype=bool))


def test_event_cond_003_mask_on_context_clock_aligns_operands() -> None:
    """ID: EVENT_COND_003_mask_on_context_clock_aligns_operands."""
    ao = _ao_series(values=[0.0, 10.0, 20.0], time=[0.0, 1.0, 2.0])
    other = _ao_series(values=[0.0, 20.0], time=[0.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "eq", other)
    linear_mask = ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="linear"))
    nearest_mask = ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="nearest"))
    np.testing.assert_array_equal(linear_mask.values, np.asarray([True, True, True], dtype=bool))
    np.testing.assert_array_equal(nearest_mask.values, np.asarray([True, False, True], dtype=bool))


def test_event_cond_004_mask_intersects_context_validity() -> None:
    """ID: EVENT_COND_004_mask_intersects_context_validity."""
    ao = _ao_series(values=[1.0, 1.0, 1.0], time=[0.0, 1.0, 2.0], sequence_size=2)
    mask = ao.events.mask(Condition.compare(Condition.var("value"), "ge", 0.0))
    np.testing.assert_array_equal(mask.values, np.asarray([True, True, False], dtype=bool))


def test_event_cond_005_truth_eval_before_after_exact_nearest_contract() -> None:
    """ID: EVENT_COND_005_truth_eval_before_after_exact_nearest_contract."""
    ao = _ao_series(values=[0.0, 10.0, 20.0], time=[0.0, 1.0, 2.0])
    other = _ao_series(values=[0.0, 20.0], time=[0.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "eq", other)
    exact_like = ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="linear"))
    nearest_like = ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="nearest"))
    assert bool(exact_like.sel(sample=1).item()) is True
    assert bool(nearest_like.sel(sample=1).item()) is False


def test_event_parity_001_mask_matches_existing_param_eval_for_simple_compare() -> None:
    """ID: EVENT_PARITY_001_mask_matches_existing_param_eval_for_simple_compare."""
    ao = _ao_series(values=[0.0, 10.0, 20.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "ge", 10.0)
    mask = ao.events.mask(cond)
    np.testing.assert_array_equal(mask.values, np.asarray([False, True, True], dtype=bool))


def test_event_cond_006_dataarray_operand_label_alignment_is_not_positional() -> None:
    ao = _ao_series(values=[0.0, 1.0, 2.0], time=[0.0, 1.0, 2.0])
    wrong = xr.DataArray(
        np.asarray([0.0, 1.0, 2.0], dtype="float64"),
        dims=("sample",),
        coords={"sample": [2, 1, 0]},
    )
    with pytest.raises(ValueError) as err:
        ao.events.mask(Condition.compare(Condition.var("value"), "eq", wrong))
    assert "labels for dim 'sample' must match context clock labels" in str(err.value)


def test_event_cond_007_ao_operand_alignment_decoupled_from_schema_names() -> None:
    """ID: EVENT_COND_007_ao_operand_alignment_decoupled_from_schema_names."""
    ao = _ao_series(values=[0.0, 10.0, 20.0], time=[0.0, 1.0, 2.0])
    other_ds = xr.Dataset(
        data_vars={"value": (("step",), np.asarray([0.0, 10.0, 20.0], dtype="float64"))},
        coords={
            "step": np.arange(3, dtype="int64"),
            "clock": ("step", np.asarray([0.0, 1.0, 2.0], dtype="float64")),
        },
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=(),
        core_dims=(),
        param_coord="clock",
    )
    cond = Condition.compare(Condition.var("value"), "eq", other)
    mask = ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="linear"))
    np.testing.assert_array_equal(mask.values, np.asarray([True, True, True], dtype=bool))


def test_event_cond_008_ao_operand_alignment_multi_batch_role_remap() -> None:
    """ID: EVENT_COND_008_ao_operand_alignment_multi_batch_role_remap."""
    values = np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype="float64")
    times = np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype="float64")
    labels = np.asarray([101, 202], dtype="int64")
    ctx_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), values)},
        coords={
            "trial": labels,
            "sample": np.arange(values.shape[1], dtype="int64"),
            "time": (("trial", "sample"), times),
        },
    )
    other_ds = xr.Dataset(
        data_vars={"value": (("run", "step"), values)},
        coords={
            "run": labels,
            "step": np.arange(values.shape[1], dtype="int64"),
            "clock": (("run", "step"), times),
        },
    )
    ao = AnalysisObject.from_data(
        ctx_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=("run",),
        core_dims=(),
        param_coord="clock",
    )
    cond = Condition.compare(Condition.var("value"), "eq", other)
    mask = ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="linear"))
    np.testing.assert_array_equal(mask.values, np.ones((2, 3), dtype=bool))


def test_event_cond_011_ao_operand_alignment_multi_batch_permutation() -> None:
    """ID: EVENT_COND_011_ao_operand_alignment_multi_batch_permutation."""
    vals = np.arange(2 * 3 * 4, dtype="float64").reshape(2, 3, 4)
    time_ctx = np.broadcast_to(np.arange(4, dtype="float64"), (2, 3, 4))
    time_other = np.broadcast_to(np.arange(4, dtype="float64"), (3, 2, 4))
    ctx_ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), vals)},
        coords={
            "trial": np.asarray([101, 202], dtype="int64"),
            "sensor": np.asarray(["s0", "s1", "s2"], dtype="U2"),
            "sample": np.arange(4, dtype="int64"),
            "time": (("trial", "sensor", "sample"), time_ctx),
        },
    )
    other_ds = xr.Dataset(
        data_vars={"value": (("sensor_id", "run", "step"), vals.transpose(1, 0, 2))},
        coords={
            "sensor_id": np.asarray(["s0", "s1", "s2"], dtype="U2"),
            "run": np.asarray([101, 202], dtype="int64"),
            "step": np.arange(4, dtype="int64"),
            "clock": (("sensor_id", "run", "step"), time_other),
        },
    )
    ao = AnalysisObject.from_data(
        ctx_ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="time",
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=("sensor_id", "run"),
        core_dims=(),
        param_coord="clock",
    )
    cond = Condition.compare(Condition.var("value"), "eq", other)
    mask = ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="linear"))
    np.testing.assert_array_equal(mask.values, np.ones((2, 3, 4), dtype=bool))


def test_event_cond_012_ao_operand_alignment_name_priority_breaks_label_ties() -> None:
    """ID: EVENT_COND_012_ao_operand_alignment_name_priority_breaks_label_ties."""
    vals = np.arange(2 * 2 * 3, dtype="float64").reshape(2, 2, 3)
    time = np.broadcast_to(np.arange(3, dtype="float64"), (2, 2, 3))
    ctx_ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), vals)},
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sensor": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sensor", "sample"), time),
        },
    )
    other_ds = xr.Dataset(
        data_vars={"value": (("sensor", "trial", "step"), vals.transpose(1, 0, 2))},
        coords={
            "sensor": np.asarray([0, 1], dtype="int64"),
            "trial": np.asarray([0, 1], dtype="int64"),
            "step": np.arange(3, dtype="int64"),
            "clock": (("sensor", "trial", "step"), time.transpose(1, 0, 2)),
        },
    )
    ao = AnalysisObject.from_data(
        ctx_ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="time",
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=("sensor", "trial"),
        core_dims=(),
        param_coord="clock",
    )
    cond = Condition.compare(Condition.var("value"), "eq", other)
    mask = ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="linear"))
    np.testing.assert_array_equal(mask.values, np.ones((2, 2, 3), dtype=bool))


def test_event_cond_013_ao_operand_alignment_ambiguous_without_name_anchor_rejected() -> None:
    """ID: EVENT_COND_013_ao_operand_alignment_ambiguous_without_name_anchor_rejected."""
    vals = np.arange(2 * 2 * 3, dtype="float64").reshape(2, 2, 3)
    time = np.broadcast_to(np.arange(3, dtype="float64"), (2, 2, 3))
    ctx_ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), vals)},
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sensor": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sensor", "sample"), time),
        },
    )
    other_ds = xr.Dataset(
        data_vars={"value": (("run", "device", "step"), vals)},
        coords={
            "run": np.asarray([0, 1], dtype="int64"),
            "device": np.asarray([0, 1], dtype="int64"),
            "step": np.arange(3, dtype="int64"),
            "clock": (("run", "device", "step"), time),
        },
    )
    ao = AnalysisObject.from_data(
        ctx_ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="time",
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=("run", "device"),
        core_dims=(),
        param_coord="clock",
    )
    cond = Condition.compare(Condition.var("value"), "eq", other)
    with pytest.raises(ValueError) as err:
        ao.events.mask(cond, opts=ConditionEvalOptions(ao_interp="linear"))
    assert "batch mapping is ambiguous under labels" in str(err.value)


def test_event_hard_008_temp_dim_allocation_avoids_coord_name_collision() -> None:
    """ID: EVENT_HARD_008_temp_dim_allocation_avoids_coord_name_collision."""
    vals = np.arange(2 * 2 * 3, dtype="float64").reshape(2, 2, 3)
    time = np.broadcast_to(np.arange(3, dtype="float64"), (2, 2, 3))
    ctx_ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), vals)},
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sensor": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sensor", "sample"), time),
            "__tal_dim_tmp__": (("trial", "sensor", "sample"), np.zeros_like(time)),
        },
    )
    other_ds = xr.Dataset(
        data_vars={"value": (("sensor", "trial", "step"), vals.transpose(1, 0, 2))},
        coords={
            "sensor": np.asarray([0, 1], dtype="int64"),
            "trial": np.asarray([0, 1], dtype="int64"),
            "step": np.arange(3, dtype="int64"),
            "clock": (("sensor", "trial", "step"), time.transpose(1, 0, 2)),
        },
    )
    ao = AnalysisObject.from_data(
        ctx_ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="time",
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=("sensor", "trial"),
        core_dims=(),
        param_coord="clock",
    )
    mask = ao.events.mask(Condition.compare(Condition.var("value"), "eq", other))
    np.testing.assert_array_equal(mask.values, np.ones((2, 2, 3), dtype=bool))


def test_event_hard_009_temp_dim_allocation_avoids_existing_temp_suffix_collisions() -> None:
    """ID: EVENT_HARD_009_temp_dim_allocation_avoids_existing_temp_suffix_collisions."""
    vals = np.arange(2 * 2 * 3, dtype="float64").reshape(2, 2, 3)
    time = np.broadcast_to(np.arange(3, dtype="float64"), (2, 2, 3))
    ctx_ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), vals)},
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sensor": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "time": (("trial", "sensor", "sample"), time),
            "__tal_dim_tmp__": (("trial", "sensor", "sample"), np.zeros_like(time)),
            "__tal_dim_tmp___": (("trial", "sensor", "sample"), np.ones_like(time)),
        },
    )
    other_ds = xr.Dataset(
        data_vars={"value": (("sensor", "trial", "step"), vals.transpose(1, 0, 2))},
        coords={
            "sensor": np.asarray([0, 1], dtype="int64"),
            "trial": np.asarray([0, 1], dtype="int64"),
            "step": np.arange(3, dtype="int64"),
            "clock": (("sensor", "trial", "step"), time.transpose(1, 0, 2)),
        },
    )
    ao = AnalysisObject.from_data(
        ctx_ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="time",
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=("sensor", "trial"),
        core_dims=(),
        param_coord="clock",
    )
    mask = ao.events.mask(Condition.compare(Condition.var("value"), "eq", other))
    np.testing.assert_array_equal(mask.values, np.ones((2, 2, 3), dtype=bool))


def test_event_hard_010_event_ao_remap_uses_orchestration_axis_map_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: EVENT_HARD_010_event_ao_remap_uses_orchestration_axis_map_owner."""
    ao = _ao_series(values=[0.0, 10.0, 20.0], time=[0.0, 1.0, 2.0])
    other_ds = xr.Dataset(
        data_vars={"value": (("step",), np.asarray([0.0, 10.0, 20.0], dtype="float64"))},
        coords={
            "step": np.arange(3, dtype="int64"),
            "clock": ("step", np.asarray([0.0, 1.0, 2.0], dtype="float64")),
        },
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=(),
        core_dims=(),
        param_coord="clock",
    )
    calls = {"n": 0}

    def _spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return resolve_role_axis_map_owner(*args, **kwargs)

    monkeypatch.setattr(event_eval_mod, "resolve_role_axis_map", _spy)
    mask = ao.events.mask(Condition.compare(Condition.var("value"), "eq", other))
    np.testing.assert_array_equal(mask.values, np.asarray([True, True, True], dtype=bool))
    assert calls["n"] >= 1


def test_event_hard_011_event_dim_rename_collision_safe_via_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: EVENT_HARD_011_event_dim_rename_collision_safe_via_shared_helper."""
    vals = np.arange(2 * 2 * 3, dtype="float64").reshape(2, 2, 3)
    time = np.broadcast_to(np.arange(3, dtype="float64"), (2, 2, 3))
    ctx_ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), vals)},
        coords={
            "trial": np.asarray([0, 1], dtype="int64"),
            "sensor": np.asarray([0, 1], dtype="int64"),
            "sample": np.arange(3, dtype="int64"),
            "__tal_dim_tmp__": (("trial", "sensor", "sample"), time),
            "__tal_dim_tmp___": (("trial", "sensor", "sample"), np.ones_like(time)),
        },
    )
    other_ds = xr.Dataset(
        data_vars={"value": (("sensor", "trial", "step"), vals.transpose(1, 0, 2))},
        coords={
            "sensor": np.asarray([0, 1], dtype="int64"),
            "trial": np.asarray([0, 1], dtype="int64"),
            "step": np.arange(3, dtype="int64"),
            "clock": (("sensor", "trial", "step"), time.transpose(1, 0, 2)),
        },
    )
    ao = AnalysisObject.from_data(
        ctx_ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="__tal_dim_tmp__",
    )
    other = AnalysisObject.from_data(
        other_ds,
        sequence_dim="step",
        batch_dims=("sensor", "trial"),
        core_dims=(),
        param_coord="clock",
    )
    calls = {"n": 0}

    def _spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return rename_dims_collision_safe_owner(*args, **kwargs)

    monkeypatch.setattr(event_eval_mod, "rename_dims_collision_safe", _spy)
    mask = ao.events.mask(
        Condition.compare(Condition.var("value"), "eq", other),
        opts=ConditionEvalOptions(coord_name="__tal_dim_tmp__"),
    )
    np.testing.assert_array_equal(mask.values, np.ones((2, 2, 3), dtype=bool))
    assert calls["n"] >= 2


def test_event_cond_009_unlabeled_context_dim_broadcast_alignment_stable() -> None:
    """ID: EVENT_COND_009_unlabeled_context_dim_broadcast_alignment_stable."""
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray([0.0, 1.0, 2.0], dtype="float64"))},
        coords={"time": ("sample", np.asarray([0.0, 1.0, 2.0], dtype="float64"))},
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )
    cond = Condition.compare(Condition.var("value"), "gt", xr.DataArray(np.asarray(1.0, dtype="float64")))
    mask = ao.events.mask(cond)
    np.testing.assert_array_equal(mask.values, np.asarray([False, False, True], dtype=bool))


def test_event_hard_006_sparse_finite_prefix_only_rejected() -> None:
    """ID: EVENT_HARD_006_sparse_finite_prefix_only_rejected."""
    ao = _ao_series(values=[1.0, 1.0, 1.0], time=[0.0, np.nan, 2.0])
    cond = Condition.compare(Condition.var("value"), "ge", 0.0)
    with pytest.raises(ValueError) as err:
        ao.events.mask(cond, opts=ConditionEvalOptions(validity_mode="prefix_only"))
    assert "opts.validity_mode='prefix_only'" in str(err.value)


def test_event_hard_007_finite_gather_preserves_trailing_finite_samples() -> None:
    """ID: EVENT_HARD_007_finite_gather_preserves_trailing_finite_samples."""
    ao = _ao_series(values=[1.0, 1.0, 1.0], time=[0.0, np.nan, 2.0], sequence_size=3)
    cond = Condition.compare(Condition.var("value"), "ge", 0.0)
    auto = ao.events.mask(cond, opts=ConditionEvalOptions(validity_mode="auto"))
    finite = ao.events.mask(cond, opts=ConditionEvalOptions(validity_mode="finite_gather"))
    np.testing.assert_array_equal(auto.values, np.asarray([True, True, True], dtype=bool))
    np.testing.assert_array_equal(finite.values, np.asarray([True, False, True], dtype=bool))


def test_event_cond_010_validity_mode_auto_parity_with_resolved_runtime_mask() -> None:
    """ID: EVENT_COND_010_validity_mode_auto_parity_with_resolved_runtime_mask."""
    ao = _ao_series(values=[1.0, 1.0, 1.0], time=[0.0, 1.0, 2.0], sequence_size=2)
    cond = Condition.compare(Condition.var("value"), "ge", 0.0)
    default_mask = ao.events.mask(cond)
    auto_mask = ao.events.mask(cond, opts=ConditionEvalOptions(validity_mode="auto"))
    np.testing.assert_array_equal(default_mask.values, np.asarray([True, True, False], dtype=bool))
    np.testing.assert_array_equal(auto_mask.values, default_mask.values)


@pytest.mark.parametrize(
    "clock",
    [
        pytest.param(np.asarray([0, 1], dtype="int64"), id="integer"),
        pytest.param(np.asarray([0.0, 1.0], dtype="float64"), id="floating"),
        pytest.param(
            np.asarray(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"),
            id="datetime",
        ),
    ],
)
def test_event_cond_014_scalar_operand_dtype_independent_of_context_clock(clock: np.ndarray) -> None:
    """ID: EVENT_COND_014_scalar_operand_dtype_independent_of_context_clock."""
    ao = _ao_with_clock(clock)
    scalar = np.float32(0.5)
    context_clock = ao.as_dataset(copy="none").coords["time"]
    mask = ao.events.mask(Condition.compare(Condition.var("value"), "gt", scalar))
    np.testing.assert_array_equal(mask.values, np.asarray([False, True], dtype=bool))
    assert mask.dims == context_clock.dims
    for name in context_clock.coords:
        xr.testing.assert_identical(mask.coords[name], context_clock.coords[name])

    with pytest.raises(ValueError, match="events.mask: right operand must be numeric"):
        ao.events.mask(Condition.compare(Condition.var("value"), "gt", "threshold"))


def test_event_cond_015_scalar_operand_broadcast_is_side_symmetric() -> None:
    """ID: EVENT_COND_015_scalar_operand_broadcast_is_side_symmetric."""
    ao = _ao_with_clock(np.asarray([0, 1], dtype="int64"))
    scalar = np.float32(0.5)
    right_scalar = ao.events.mask(Condition.compare(Condition.var("value"), "gt", scalar))
    left_scalar = ao.events.mask(Condition.compare(scalar, "lt", Condition.var("value")))

    expected = np.asarray([False, True], dtype=bool)
    np.testing.assert_array_equal(right_scalar.values, expected)
    np.testing.assert_array_equal(left_scalar.values, expected)


def test_event_cond_017_scalar_metadata_does_not_leak_into_mask() -> None:
    """ID: EVENT_COND_017_scalar_metadata_does_not_leak_into_mask."""
    ds = xr.Dataset(
        data_vars={
            "value": xr.DataArray(
                np.asarray([0.25, 0.75], dtype="float64"),
                dims=("sample",),
                attrs={"units": "m", "source": "value"},
            ),
        },
        coords={
            "sample": xr.DataArray(
                np.arange(2, dtype="int64"),
                dims=("sample",),
                attrs={"axis": "sample"},
            ),
            "time": xr.DataArray(
                np.asarray([0, 1], dtype="int64"),
                dims=("sample",),
                attrs={"units": "s", "source": "clock"},
            ),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )
    clock = ao.as_dataset(copy="none").coords["time"]
    with xr.set_options(keep_attrs=True):
        left = ao.events.mask(Condition.compare(np.float32(0.5), "lt", Condition.var("value")))
        right = ao.events.mask(Condition.compare(Condition.var("value"), "gt", np.float32(0.5)))

    expected = np.asarray([False, True], dtype=bool)
    np.testing.assert_array_equal(left.values, expected)
    np.testing.assert_array_equal(right.values, expected)
    assert left.name == right.name
    assert left.attrs == right.attrs == {}
    xr.testing.assert_identical(left.coords["sample"], clock.coords["sample"])


@pytest.mark.parametrize(
    "clock",
    [
        pytest.param(np.asarray([0, 1], dtype="int64"), id="integer"),
        pytest.param(
            np.asarray(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"),
            id="datetime",
        ),
    ],
)
def test_event_hard_024_temporal_scalar_operands_fail_owned_validation(clock: np.ndarray) -> None:
    """ID: EVENT_HARD_024_temporal_scalar_operands_fail_owned_validation."""
    ao = _ao_with_clock(clock)
    duration = np.timedelta64(1, "s")

    with pytest.raises(ValueError) as right_err:
        ao.events.mask(Condition.compare(Condition.var("value"), "gt", duration))
    assert "events.mask: right operand must be numeric" in str(right_err.value)

    with pytest.raises(ValueError) as left_err:
        ao.events.mask(Condition.compare(duration, "lt", Condition.var("value")))
    assert "events.mask: left operand must be numeric" in str(left_err.value)


def test_event_cond_016_timedelta_array_ordering_preserved() -> None:
    """ID: EVENT_COND_016_timedelta_array_ordering_preserved."""
    ds = xr.Dataset(
        data_vars={
            "duration": (
                ("sample",),
                np.asarray([1, 3], dtype="timedelta64[s]"),
            ),
        },
        coords={
            "sample": np.arange(2, dtype="int64"),
            "time": (("sample",), np.asarray([0.0, 1.0], dtype="float64")),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )
    threshold = xr.DataArray(
        np.asarray([2, 2], dtype="timedelta64[s]"),
        dims=("sample",),
        coords={"sample": np.arange(2, dtype="int64")},
    )

    named_left = ao.events.mask(Condition.compare(Condition.var("duration"), "gt", threshold))
    array_left = ao.events.mask(Condition.compare(threshold, "lt", Condition.var("duration")))
    expected = np.asarray([False, True], dtype=bool)
    np.testing.assert_array_equal(named_left.values, expected)
    np.testing.assert_array_equal(array_left.values, expected)


def test_event_perf_001_scalar_operand_broadcast_preserves_dask_laziness() -> None:
    """ID: EVENT_PERF_001_scalar_operand_broadcast_preserves_dask_laziness."""
    da = pytest.importorskip("dask.array")
    ds = xr.Dataset(
        data_vars={
            "value": (("sample",), da.from_array(np.asarray([0.25, 0.75]), chunks=1)),
        },
        coords={
            "sample": np.arange(2, dtype="int64"),
            "time": (
                ("sample",),
                da.from_array(
                    np.asarray(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"),
                    chunks=1,
                ),
            ),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="time",
    )

    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        mask = ao.events.mask(Condition.compare(Condition.var("value"), "gt", np.float32(0.5)))
    assert not tasks
    assert isinstance(mask.data, da.Array)
    assert mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(mask.compute(scheduler="synchronous"), [False, True])



@pytest.mark.parametrize("invalid", ["x", None])
def test_event_hard_014_condition_tolerance_invalid_raises_tal_valueerror(invalid: object) -> None:
    """ID: EVENT_HARD_014_condition_tolerance_invalid_raises_tal_valueerror."""
    ao = _ao_series(values=[0.0, 1.0, 2.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "ge", 0.0)
    with pytest.raises(ValueError) as err:
        ao.events.mask(cond, opts=ConditionEvalOptions(eq_atol=invalid))  # type: ignore[arg-type]
    assert "events.mask: opts.eq_atol must be a numeric scalar" in str(err.value)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("operation", ("mask", "events"))
@pytest.mark.parametrize("sliced", (False, True))
@pytest.mark.parametrize("native", (False, True))
def test_tut_001_events_preserve_the_contexts_own_sample_labels(lazy, operation, sliced, native):
    """TUT-001 / Contract 021: parameter evaluation preserves context correspondence."""
    from dask.callbacks import Callback

    ds = _ao_series(values=[0., 2., 0., 0.], time=[0., 1., 2., 3.]).as_dataset()
    axis = (xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(10, 50, 10, dim="sample"))
            if native else {"sample": [10, 20, 30, 40]})
    ds = ds.assign_coords(axis)
    source = AnalysisObject.from_data(ds.chunk({"sample": 2}) if lazy else ds)
    if sliced:
        source = source.isel(sample=slice(1, None))
    snapshot = source.as_dataset(copy="deep")
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = (source.events.events(source > 1., opts=EventExtractOptions(include_initial=True, max_events=2))
                  if operation == "events" else source.events.mask(source > 1.))
    assert tasks == []
    actual = result.compute(scheduler="synchronous")
    if operation == "mask":
        np.testing.assert_array_equal(actual, [True, False, False] if sliced else [False, True, False, False])
        xr.testing.assert_identical(actual["sample"].variable, snapshot["sample"].variable)
        assert actual.dims == ("sample",)
        assert actual.xindexes["sample"].equals(snapshot.xindexes["sample"])
        if native is True:
            assert isinstance(actual.xindexes["sample"], xr.indexes.RangeIndex)
    else:
        np.testing.assert_allclose(actual["time"], [1., 1.])
        np.testing.assert_array_equal(actual["sample_index_after"], [0, 1] if sliced else [1, 2])
        np.testing.assert_array_equal(actual["sample_index_before"], [-1, 0] if sliced else [0, 1])
    xr.testing.assert_identical(source.as_dataset(), snapshot)


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("method", ("linear", "nearest"))
def test_tut_001_independent_operand_restores_context_roles_and_validity(lazy, method):
    """TUT-001: independently sampled, renamed operands remain label-safe and lazy."""
    from dask.callbacks import Callback

    ds = xr.Dataset({"value": (("trial", "sample"), [[0., 10., 99.], [10., 20., 30.]])},
        coords={"trial": ["a", "b"], "sample": [10, 20, 30], "time": ("sample", [0., 1., 2.]),
                "length": ("trial", [2, 3])})
    other_ds = xr.Dataset({"value": (("run", "step"), [[0., 20.], [10., 30.]])},
        coords={"run": ["a", "b"], "step": [100, 200], "clock": ("step", [0., 2.])})
    if lazy:
        ds, other_ds = ds.chunk({"sample": 2}).assign_coords(length=ds["length"]), other_ds.chunk({"step": 1})
    source = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=("trial",),
                                      param_coord="time", sequence_size_coord="length")
    other = AnalysisObject.from_data(other_ds, sequence_dim="step", batch_dims=("run",), param_coord="clock")
    snapshots = source.as_dataset(copy="deep"), other.as_dataset(copy="deep")
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.events.mask(Condition.compare(Condition.var("value"), "eq", other),
                                    opts=ConditionEvalOptions(ao_interp=method))
    assert tasks == []
    np.testing.assert_array_equal(result.compute(scheduler="synchronous"),
        [[True, True, False], [True, True, True]] if method == "linear" else
        [[True, False, False], [True, False, True]])
    assert result.dims == ("trial", "sample")
    np.testing.assert_array_equal(result["trial"], ["a", "b"])
    np.testing.assert_array_equal(result["sample"], [10, 20, 30])
    xr.testing.assert_identical(source.as_dataset(), snapshots[0])
    xr.testing.assert_identical(other.as_dataset(), snapshots[1])


def test_event_clock_name_is_an_explicit_supported_option():
    """Contract 021: a non-default parameter name needs an explicit clock option."""
    ds = xr.Dataset({"value": ("sample", [0., 2., 0.])},
                    coords={"sample": [0, 1, 2], "clock": ("sample", [0., 1., 2.])})
    source = AnalysisObject.from_data(ds, sequence_dim="sample", param_coord="clock")
    result = source.events.mask(source > 1., opts=ConditionEvalOptions(coord_name="clock"))
    np.testing.assert_array_equal(result, [False, True, False])


def _tut_audit_ao(ds,*,batch=(),sequence='sample',param='time',lazy=False):
    if lazy: ds=ds.chunk({sequence:2})
    return AnalysisObject.from_data(ds,sequence_dim=sequence,batch_dims=batch,param_coord=param)


def _tut_audit_grouped(*,shared=False,n=4,lazy=False,native=True):
    times=np.arange(n,dtype=float)
    ds=xr.Dataset({'value':(('trial','sample'),np.tile(times,(2,1)))},coords={'trial':['a','b'],'sample':np.arange(n)*10+10,'time':('sample',times) if shared else (('trial','sample'),np.tile(times,(2,1)))})
    if native: ds=ds.assign_coords(xr.Coordinates.from_xindex(xr.indexes.RangeIndex.arange(2,dim='trial')))
    return _tut_audit_ao(ds,batch=('trial',),lazy=lazy)

@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('lazy',[False,True])
@pytest.mark.parametrize('n',[0,4])
@pytest.mark.parametrize('operation',['mask','segments','stream'])
def test_tut_007_range_batch_topology(shared,lazy,n,operation):
    ao=_tut_audit_grouped(shared=shared,n=n,lazy=lazy)
    cond=Condition.compare(Condition.var('value'),'gt',2.)
    tasks=[]
    with Callback(pretask=lambda key,*_:tasks.append(key)):
        result=ao.events.mask(cond) if operation=='mask' else ao.events.when(cond,opts=WhenOptions(layout=operation,max_segments=2))
    assert tasks==[]
    ds=result if isinstance(result,xr.DataArray) else result.as_dataset()
    assert isinstance(ds.xindexes['trial'],xr.indexes.RangeIndex)

@pytest.mark.parametrize('method',['linear','nearest'])
@pytest.mark.parametrize('case',['indexed_self','station_data','control'])
@pytest.mark.parametrize('lazy',[False,True])
def test_tut_008_context_query_carriers(case,lazy,method):
    ds=xr.Dataset({'value':('sample',[0.,1.,2.])},coords={'sample':[10,20,30],'time':('sample',[0.,1.,2.]),'station':'lab','tag':('sample',[100.,102.,104.])})
    if case=='indexed_self':ds=ds.set_xindex('tag')
    ao=_tut_audit_ao(ds,lazy=lazy)
    other=ao if case=='indexed_self' else _tut_audit_ao(xr.Dataset({'station' if case=='station_data' else 'value':('sample',[0.,1.,2.])},coords={'sample':[1,2,3],'time':('sample',[0.,1.,2.]),**({'station':'lab'} if case!='station_data' else {})}),lazy=lazy)
    tasks=[]
    with Callback(pretask=lambda key,*_:tasks.append(key)):mask=ao.events.mask(Condition.compare(Condition.var('value'),'eq',other), opts=ConditionEvalOptions(ao_interp=method))
    assert tasks==[]
    np.testing.assert_equal(mask.compute(),np.ones(3,dtype=bool))
    assert mask.dims == ("sample",)
    for name, index in ao.as_dataset().xindexes.items():
        assert type(mask.xindexes[name]) is type(index) and mask.xindexes[name].equals(index)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("axis,shared,batch,operand,ragged", [
    (True, False, (2, 3), "self", False), (True, False, (2,), "independent", True),
    (True, False, (2, 3), "renamed", False), (False, True, (2, 3), "self", False),
    (False, True, (2, 3), "independent", True), (False, True, (2, 3), "renamed", False),
    (False, False, (2,), "self", True), (True, True, (), "self", False)])
def test_tut_011_013_ao_conditions_preserve_lazy_context(tutorial_audit_source, lazy, method, axis, shared, batch, operand, ragged):
    """TUT-011/013: interpolation preserves correspondence without evaluating metadata."""
    source = tutorial_audit_source(lazy=lazy, axis=axis, shared=shared, batch=batch, ragged=ragged)
    if operand == "independent":
        original = source.as_dataset()
        other_ds = original.copy(deep=True).drop_vars("value")
        other_ds["other"] = xr.Variable(original.value.dims, np.broadcast_to(np.arange(6.), original.value.shape))
        other = AnalysisObject.from_data(other_ds, sequence_dim="sample", batch_dims=tuple(f"b{i}" for i in range(len(batch))),
                                        core_dims=(), param_coord="sample" if axis else "clock",
                                        sequence_size_coord="count" if ragged else None)
    elif operand == "renamed":
        other = source.rename({"sample": "record", "b0": "trial", "b1": "subject"})
    else:
        other = source
    snapshot, other_snapshot = source.as_dataset(copy="deep"), other.as_dataset(copy="deep")
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.events.mask(Condition.compare(Condition.var("value"), "eq", other),
            opts=ConditionEvalOptions(coord_name="sample" if axis else "clock", ao_interp=method))
    assert not tasks
    expected = np.ones(snapshot.value.shape, dtype=bool) if operand != "independent" else np.broadcast_to(
        np.arange(6.) == np.array([0., 1., 1., 0., 1., 0.]), snapshot.value.shape).copy()
    if ragged:
        expected = expected & (np.arange(6.) < snapshot["count"].data[..., None])
    np.testing.assert_array_equal(result.compute(scheduler="synchronous"), expected)
    assert result.dims == snapshot.value.dims
    for name, index in snapshot.xindexes.items():
        assert type(result.xindexes[name]) is type(index) and result.xindexes[name].equals(index)
    xr.testing.assert_identical(source.as_dataset(), snapshot)
    xr.testing.assert_identical(other.as_dataset(), other_snapshot)


@pytest.mark.parametrize("predecessor", ["sel", "at"])
@pytest.mark.parametrize("axis,batch,lazy_part", [(False, (), "all"), (False, (2,), "all"),
    (False, (2, 3), "all"), (False, (2, 3), "clock"), (False, (2,), "payload"), (True, (2, 3), "all")])
def test_tut_013_chained_conditions_do_not_compute_generated_metadata(tutorial_audit_source, predecessor, axis, batch, lazy_part):
    """TUT-013: returned Dask backing alone is insufficient; construction executes no tasks."""
    source = tutorial_audit_source(lazy=True, axis=axis, batch=batch, lazy_part=lazy_part)
    query = xr.DataArray([1., 2. ** 1.5, 4. ** 1.5], dims="request").chunk(request=1)
    selected = getattr(source.param, predecessor)(query, validate=False)
    snapshot, original = selected.as_dataset(copy="deep"), query.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = selected.events.mask(Condition.compare(Condition.var("value"), "eq", selected),
                                     opts=ConditionEvalOptions(coord_name="sample" if axis else "clock"))
    assert not tasks and result.chunks is not None
    np.testing.assert_array_equal(result.compute(scheduler="synchronous"), np.ones(result.shape, dtype=bool))
    for name, index in snapshot.xindexes.items():
        assert type(result.xindexes[name]) is type(index) and result.xindexes[name].equals(index)
    for name in ("valid", "sample_index", "sample" if axis else "clock"):
        if name in snapshot.coords:
            xr.testing.assert_identical(result.coords[name].compute(), snapshot.coords[name].compute())
    xr.testing.assert_identical(selected.as_dataset(), snapshot)
    xr.testing.assert_identical(query, original)


LABEL_CASES = [
    (labels, batch, operand)
    for labels in ["indexed", "unindexed", "lazy"]
    for batch in [(), (2,), (2, 3)]
    for operand in ["scalar", "self", "dataarray"]
]


@pytest.mark.parametrize("labels,batch,operand", LABEL_CASES)
def test_tut_017_lazy_sequence_topology(tutorial_audit_source, labels, batch, operand):
    """TUT-017: public ownership regression and controls."""
    a = tutorial_audit_source(
        clock_values=[0.0, 1.0, 3.0, 5.0, 8.0, 10.0],
        lazy=True,
        batch=batch,
        labels=labels,
    )
    ds = a.as_dataset()
    before = a.as_dataset(copy="deep")
    tasks = []
    right = 0.5 if operand == "scalar" else a if operand == "self" else ds.value
    c = Condition.compare(
        Condition.var("value"), "gt" if operand == "scalar" else "eq", right
    )
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = a.events.mask(c, opts=ConditionEvalOptions(coord_name="clock"))
    assert not tasks
    assert ("sample" in result.xindexes) == (labels == "indexed")
    expected = (
        np.broadcast_to(np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0]) > 0.5, ds.value.shape)
        if operand == "scalar"
        else np.ones(ds.value.shape, bool)
    )
    np.testing.assert_array_equal(result.compute(scheduler="synchronous"), expected)
    for name, index in ds.xindexes.items():
        assert type(result.xindexes[name]) is type(index) and result.xindexes[
            name
        ].equals(index)
    xr.testing.assert_identical(a.as_dataset(), before)


@pytest.mark.parametrize('case,lazy_part,batch,method', [
    ('named', 'labels', (), 'linear'), ('reduced', 'all', (2,), 'linear'),
    ('independent', 'payload', (2,3), 'linear'), ('renamed', 'clock', (2,), 'nearest'),
    ('named', 'all', (2,3), 'nearest'), ('self', 'labels', (2,), 'linear'),
])
def test_tut_017_separate_lazy_labels_and_operand_data(tutorial_audit_source, case, lazy_part, batch, method):
    """TUT-017: numerical expansion preserves ordinary and native context topology."""
    source = tutorial_audit_source(lazy=lazy_part != 'labels', lazy_part=lazy_part, labels='lazy', batch=batch, ragged=bool(batch))
    ds = source.as_dataset()
    if case == 'reduced':
        right = ds.value.isel(sample=1, drop=True) / 2
    elif case in ('independent', 'renamed'):
        target = ds.copy(deep=False)
        # f(t)=t is an independent interpolation oracle at the source clock.
        target['value'] = ds.clock.broadcast_like(ds.value).variable
        target = target.assign_coords(xr.Coordinates({'sample': xr.Variable('sample', np.arange(6)+300)}, indexes={}))
        other = AnalysisObject.from_data(target, sequence_dim='sample', batch_dims=tuple(f'b{i}' for i in range(len(batch))), param_coord='clock', sequence_size_coord='count' if batch else None)
        if case == 'renamed':
            other = other.rename({'sample':'tick'}, validate=False)
        right = other
    else:
        right = source if case == 'self' else .5
    condition = Condition.compare(Condition.var('value'), 'eq' if case == 'self' else 'gt', right)
    before = source.as_dataset(copy='deep')
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = source.events.mask(condition, opts=ConditionEvalOptions(coord_name='clock', ao_interp=method))
    assert not tasks
    actual = result.compute(scheduler='synchronous')
    expected_right = ds.clock if case in ('independent','renamed') else 1. if case == 'self' else .5
    expected = xr.ones_like(ds.value, dtype=bool) if case == 'self' else ds.value > expected_right
    if batch:
        positions = xr.DataArray(np.arange(6), dims='sample')
        expected = expected & (positions < ds['count'])
    np.testing.assert_array_equal(actual, expected.compute(scheduler='synchronous'))
    assert 'sample' not in result.xindexes
    xr.testing.assert_identical(actual['sample'], ds['sample'].compute(scheduler='synchronous'))
    for name, index in ds.xindexes.items():
        assert type(result.xindexes[name]) is type(index) and result.xindexes[name].equals(index)
    xr.testing.assert_identical(source.as_dataset(), before)


@pytest.mark.parametrize("lane", ["sample", "trial"])
@pytest.mark.parametrize("claim", ["matching", "reversed", "missing", "extra"])
@pytest.mark.parametrize("lazy", [False, True])
def test_tut_021_condition_complete_native_correspondence(
    window_review_source, lane, claim, lazy
):
    ao = window_review_source(lazy, 2 if lane == "trial" else None, extra_index=True)
    context = ao.as_dataset()
    operand = context.value.drop_vars("time")
    if claim == "reversed":
        operand = (
            operand.drop_indexes("alias")
            .assign_coords(alias=(lane, context.alias.data[::-1]))
            .set_xindex("alias")
        )
    if claim == "missing":
        operand = operand.drop_vars("alias")
    if claim == "extra":
        operand = operand.assign_coords(
            extra=(lane, np.arange(context.sizes[lane]) + 500)
        ).set_xindex("extra")
    before = operand.copy(deep=True)
    expression = Condition.compare(operand, "eq", Condition.var("value"))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        if claim == "matching":
            result = ao.events.mask(
                expression, opts=ConditionEvalOptions(ao_interp="linear")
            )
        else:
            with pytest.raises(ValueError, match="index|topology|label"):
                ao.events.mask(expression, opts=ConditionEvalOptions(ao_interp="linear"))
            assert not tasks
            xr.testing.assert_identical(operand, before)
            xr.testing.assert_identical(ao.as_dataset(), context)
            return
    assert not tasks
    assert result.xindexes["alias"].equals(context.xindexes["alias"])
    np.testing.assert_array_equal(result.compute(), True)
    xr.testing.assert_identical(operand, before)
    xr.testing.assert_identical(ao.as_dataset(), context)
