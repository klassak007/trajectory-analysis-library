from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
import tal.core.event_ops.evaluate as event_eval_mod
from tal.core.event_ops import Condition, ConditionEvalOptions
from tal.core.orchestration.axis_map import resolve_role_axis_map as resolve_role_axis_map_owner
from tal.utils.xarray_namespace import rename_dims_collision_safe as rename_dims_collision_safe_owner


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
    resolved = event_eval_mod._broadcast_scalar_operand(
        scalar,
        clock=context_clock,
        owner="events.mask",
        field="right operand",
    )

    assert np.dtype(resolved.dtype) == np.dtype("float32")
    assert resolved.dims == context_clock.dims
    assert set(resolved.coords) == set(context_clock.coords)
    for name in context_clock.coords:
        xr.testing.assert_identical(resolved.coords[name], context_clock.coords[name])

    mask = ao.events.mask(Condition.compare(Condition.var("value"), "gt", scalar))
    np.testing.assert_array_equal(mask.values, np.asarray([False, True], dtype=bool))

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
    scalar = event_eval_mod._broadcast_scalar_operand(
        np.float32(0.5),
        clock=clock,
        owner="events.mask",
        field="left operand",
    )

    assert scalar.name is None
    assert scalar.attrs == {}
    xr.testing.assert_identical(scalar.coords["sample"], clock.coords["sample"])

    with xr.set_options(keep_attrs=True):
        left = ao.events.mask(Condition.compare(np.float32(0.5), "lt", Condition.var("value")))
        right = ao.events.mask(Condition.compare(Condition.var("value"), "gt", np.float32(0.5)))

    expected = np.asarray([False, True], dtype=bool)
    np.testing.assert_array_equal(left.values, expected)
    np.testing.assert_array_equal(right.values, expected)
    assert left.name == right.name
    assert left.attrs == right.attrs == {}


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


def test_event_perf_002_invalid_scalar_fails_before_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: EVENT_PERF_002_invalid_scalar_fails_before_broadcast."""
    clock = _ao_with_clock(np.asarray([0, 1], dtype="int64")).as_dataset(copy="none").coords["time"]

    def reject_broadcast(*args: object, **kwargs: object) -> xr.DataArray:
        raise AssertionError("invalid scalar must fail before xr.full_like")

    monkeypatch.setattr(event_eval_mod.xr, "full_like", reject_broadcast)
    invalid_scalars = (
        "threshold",
        np.bool_(True),
        np.datetime64("2025-01-01", "ns"),
        np.timedelta64(1, "s"),
    )
    for invalid in invalid_scalars:
        assert np.isscalar(invalid)
        with pytest.raises(ValueError, match="events.mask: right operand must be numeric"):
            event_eval_mod._broadcast_scalar_operand(
                invalid,
                clock=clock,
                owner="events.mask",
                field="right operand",
            )


def test_event_perf_001_scalar_operand_broadcast_preserves_dask_laziness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def _fail_compute(*args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected eager compute")

    monkeypatch.setattr(da.Array, "compute", _fail_compute, raising=True)
    context_clock = ao.as_dataset(copy="none").coords["time"]
    scalar_operand = event_eval_mod._broadcast_scalar_operand(
        np.float32(0.5),
        clock=context_clock,
        owner="events.mask",
        field="right operand",
    )
    assert isinstance(scalar_operand.data, da.Array)
    assert scalar_operand.chunks == context_clock.chunks
    assert scalar_operand.dtype == np.dtype("float32")

    mask = ao.events.mask(Condition.compare(Condition.var("value"), "gt", np.float32(0.5)))
    assert isinstance(mask.data, da.Array)
    assert mask.dtype == np.dtype(bool)

    monkeypatch.undo()
    np.testing.assert_array_equal(mask.compute().values, np.asarray([False, True], dtype=bool))


@pytest.mark.parametrize("invalid", ["x", None])
def test_event_hard_014_condition_tolerance_invalid_raises_tal_valueerror(invalid: object) -> None:
    """ID: EVENT_HARD_014_condition_tolerance_invalid_raises_tal_valueerror."""
    ao = _ao_series(values=[0.0, 1.0, 2.0], time=[0.0, 1.0, 2.0])
    cond = Condition.compare(Condition.var("value"), "ge", 0.0)
    with pytest.raises(ValueError) as err:
        ao.events.mask(cond, opts=ConditionEvalOptions(eq_atol=invalid))  # type: ignore[arg-type]
    assert "events.mask: opts.eq_atol must be a numeric scalar" in str(err.value)
