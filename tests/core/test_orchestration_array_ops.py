from __future__ import annotations

import inspect

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.param_ops import batch_topology as batch_topology_ops
from tal.core.orchestration.alignment import align_exact_for_plan
from tal.core.orchestration.context import (
    DatasetContextOptions,
    resolve_dataset_context,
    resolve_semantic_topology_from_dataset,
)
from tal.core.orchestration.alignment_intent import (
    read_alignment_intent,
)
from tal.core.orchestration.broadcast_intent import (
    read_broadcast_intent,
)
from tal.core.orchestration.inputs import coerce_operand
from tal.core.orchestration.topology import (
    ResolvedTopologyPlan,
    SEMANTIC_EXACT_POLICY,
    SEMANTIC_NON_CORE_POLICY,
    STRICT_EXACT_POLICY,
    STRICT_NON_CORE_POLICY,
    SemanticTopology,
    TopologyOperand,
    TopologyPolicy,
    restore_combine_batch_axis,
    resolve_binary_topology,
    resolve_nary_topology,
    stack_combine_batch_axis,
    resolve_unary_topology,
    realize_operands_for_plan,
)
from tal.core.param_ops.batch_topology import (
    BatchFlattenPlan,
    restore_dataset_batch_dims,
    stacked_batch_coords_with_labels,
)
from tal.linalg.finalize import ArrayFinalizeSpec, finalize_array_result
from tal.linalg.plan import build_array_binary_plan, build_array_plan
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.utils.topology_operation_families import (
    alignment_intent_supported_for_operation_family,
    default_topology_mode_for_operation_family,
)


def _matrix_ao(
    values: np.ndarray,
    *,
    row: str,
    col: str,
    trial_labels: tuple[str, ...] = ("t0", "t1"),
    tau_offset: float = 0.0,
) -> AnalysisObject:
    sample = np.arange(values.shape[0], dtype=np.int64)
    trial = np.asarray(trial_labels, dtype=object)
    tau = np.arange(values.shape[0] * values.shape[1], dtype=float).reshape(values.shape[1], values.shape[0])
    ds = xr.Dataset(
        {"x": (("sample", "trial", row, col), values)},
        coords={
            "sample": sample,
            "trial": trial,
            row: np.arange(values.shape[2], dtype=np.int64),
            col: np.arange(values.shape[3], dtype=np.int64),
            "tau": (("trial", "sample"), tau + tau_offset),
            "sample_size": ("trial", np.full(values.shape[1], values.shape[0], dtype=np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )


def _topology_operand(
    ao: AnalysisObject,
    *,
    index: int,
    var_name: str = "x",
) -> TopologyOperand:
    _, _, _, core_dims = read_roles(ao.unsafe_data)
    return TopologyOperand(
        index=index,
        data=ao.unsafe_data[var_name],
        semantic=resolve_semantic_topology_from_dataset(
            ao.unsafe_data,
            var_name=var_name,
            core_dims=core_dims,
            owner="test.topology",
            what=f"operand {index}",
        ),
        param_coord=read_param_coord_name(ao.unsafe_data),
    )


def _matrix_ao_with_extra_non_core(
    values: np.ndarray,
    *,
    row: str,
    col: str,
    extra_dim: str,
) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", "trial", extra_dim, row, col), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            extra_dim: np.arange(values.shape[2], dtype=np.int64),
            row: np.arange(values.shape[3], dtype=np.int64),
            col: np.arange(values.shape[4], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=True,
    )


def _matrix_ao_missing_sequence(values: np.ndarray, *, row: str, col: str) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("trial", row, col), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row: np.arange(values.shape[1], dtype=np.int64),
            col: np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=False,
    )


def _matrix_ao_missing_batch(values: np.ndarray, *, row: str, col: str) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", row, col), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row: np.arange(values.shape[1], dtype=np.int64),
            col: np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=False,
    )


def _param_vector_ao(
    values: np.ndarray,
    *,
    sample_labels: np.ndarray,
    param_values: np.ndarray,
    axis: str = "axis",
) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", axis), values)},
        coords={
            "sample": sample_labels,
            axis: np.arange(values.shape[1], dtype=np.int64),
            "time_s": ("sample", param_values),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(axis,),
        param_coord="time_s",
        validate=True,
    )


def test_orch_array_001_build_array_plan_declared_roles_and_semantic_dims_guard() -> None:
    """ID: ORCH_ARRAY_001_build_array_plan_declared_roles_and_semantic_dims_guard."""
    left_ds = xr.Dataset(
        {"x": (("sample", "trial", "row"), np.arange(12, dtype=float).reshape(2, 2, 3))},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "row": np.arange(3, dtype=np.int64),
            "mid": np.arange(3, dtype=np.int64),
        },
    )
    left = AnalysisObject.from_data(
        left_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "mid"),
        validate=True,
    )
    right = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")
    with pytest.raises(ValueError, match="missing required semantic dims"):
        _ = build_array_plan([left, right], owner="orch.array")


def test_orch_array_002_build_array_plan_exact_alignment_fail_closed() -> None:
    """ID: ORCH_ARRAY_002_build_array_plan_exact_alignment_fail_closed."""
    left = _matrix_ao(
        np.arange(36, dtype=float).reshape(2, 2, 3, 3),
        row="row",
        col="mid",
        trial_labels=("t0", "t1"),
    )
    right = _matrix_ao(
        np.arange(24, dtype=float).reshape(2, 2, 3, 2),
        row="mid",
        col="out",
        trial_labels=("t0", "t2"),
    )
    with pytest.raises(ValueError, match="requires exact 'trial' labels under sequence/batch policy"):
        _ = build_array_plan([left, right], owner="orch.array")


def test_orch_array_003_build_array_binary_plan_equivalent_to_general_plan_arity2() -> None:
    """ID: ORCH_ARRAY_003_build_array_binary_plan_equivalent_to_general_plan_arity2."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")

    general = build_array_plan([left, right], owner="orch.array")
    binary = build_array_binary_plan(left, right, owner="orch.array")

    assert binary.sequence_dim == general.sequence_dim
    assert binary.batch_dims == general.batch_dims
    assert binary.shared_param_coord == general.shared_param_coord
    assert binary.shared_sequence_size_coord == general.shared_sequence_size_coord
    xr.testing.assert_identical(binary.operands[0].data, general.operands[0].data)
    xr.testing.assert_identical(binary.operands[1].data, general.operands[1].data)


def test_orch_array_004_finalize_array_result_optional_metadata_restore_and_canonicalize() -> None:
    """ID: ORCH_ARRAY_004_finalize_array_result_optional_metadata_restore_and_canonicalize."""
    left = _matrix_ao(
        np.arange(36, dtype=float).reshape(2, 2, 3, 3),
        row="row",
        col="mid",
        tau_offset=0.0,
    )
    right = _matrix_ao(
        np.arange(24, dtype=float).reshape(2, 2, 3, 2),
        row="mid",
        col="out",
        tau_offset=100.0,
    )
    plan = build_array_binary_plan(left, right, owner="orch.array")
    result = xr.dot(plan.operands[0].data, plan.operands[1].data, dim=["mid"]).rename("ignored")

    spec = ArrayFinalizeSpec(
        output_var_name="x_out",
        sequence_dim=plan.sequence_dim,
        batch_dims=plan.batch_dims,
        core_dims=("row", "out"),
        param_name=plan.shared_param_coord,
        size_name=plan.shared_sequence_size_coord,
    )
    out = finalize_array_result(
        plan.operands[0].ao,
        result,
        spec=spec,
        optional_sources=(plan.operands[0].data, plan.operands[1].data),
        owner="orch.array",
        validate=True,
    )

    declared, sequence_dim, batch_dims, core_dims = read_roles(out.unsafe_data)
    assert declared
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)
    assert core_dims == ("row", "out")
    assert read_param_coord_name(out.unsafe_data) == "tau"
    assert read_sequence_size_coord_name(out.unsafe_data) == "sample_size"
    assert out.unsafe_data.coords["tau"].dims == ("trial", "sample")
    assert out.unsafe_data.coords["sample_size"].dims == ("trial",)


def test_orch_array_005_resolve_dataset_context_matches_array_plan_operand_context() -> None:
    """ID: ORCH_ARRAY_005_resolve_dataset_context_matches_array_plan_operand_context."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")
    plan = build_array_binary_plan(left, right, owner="orch.array")
    left_ctx = resolve_dataset_context(
        left,
        owner="orch.array",
        options=DatasetContextOptions(
            require_roles=True,
            require_sequence_dim=True,
            select_numeric_var=True,
            require_single_numeric_var=True,
            allowed_core_arity=(1, 2),
            require_semantic_dims_in_var=True,
        ),
    )
    assert left_ctx.sequence_dim == plan.operands[0].sequence_dim
    assert left_ctx.batch_dims == plan.operands[0].batch_dims
    assert left_ctx.core_dims == plan.operands[0].core_dims
    assert left_ctx.param_coord == plan.operands[0].param_coord
    assert left_ctx.sequence_size_coord == plan.operands[0].sequence_size_coord


def test_orch_array_006_finalize_array_result_without_sequence_preserves_core_only_roles() -> None:
    """ID: ORCH_ARRAY_006_finalize_array_result_without_sequence_preserves_core_only_roles."""
    ds = xr.Dataset({"x": ("sample", np.asarray([1.0, 2.0], dtype=float))}, coords={"sample": [0, 1]})
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), validate=True)
    result = xr.DataArray(np.asarray([2.0, 4.0], dtype=float), dims=("sample",), name="x")
    out = finalize_array_result(
        ao,
        result,
        spec=ArrayFinalizeSpec(
            output_var_name="x2",
            sequence_dim=None,
            batch_dims=(),
            core_dims=(),
            param_name=None,
            size_name=None,
        ),
        optional_sources=(ao.unsafe_data["x"],),
        owner="orch.array",
        validate=True,
    )
    declared, sequence_dim, batch_dims, core_dims = read_roles(out.unsafe_data)
    assert declared
    assert sequence_dim is None
    assert batch_dims == ()
    assert core_dims == ()


def test_orch_array_007_coerce_operand_scalar_policy_is_deterministic() -> None:
    """ID: ORCH_ARRAY_007_coerce_operand_scalar_policy_is_deterministic."""
    ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    out_ao = coerce_operand(ao, owner="orch.array")
    out_scalar = coerce_operand(3.0, owner="orch.array", allow_scalar=True, return_scalar_none=True)
    assert isinstance(out_ao, AnalysisObject)
    assert out_scalar is None


def test_topo_core_001_shared_topology_model_roundtrip_extraction() -> None:
    """ID: TOPO_CORE_001_shared_topology_model_roundtrip_extraction."""
    ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    semantic = resolve_semantic_topology_from_dataset(
        ao.unsafe_data,
        var_name="x",
        core_dims=("row", "mid"),
        owner="test.topology",
        what="matrix operand",
    )
    operand = TopologyOperand(index=0, data=ao.unsafe_data["x"], semantic=semantic)
    plan = resolve_unary_topology(operand, owner="test.topology", what="unary")
    assert plan.sequence_dim == "sample"
    assert plan.batch_dims == ("trial",)
    assert plan.align_exclude_dims == frozenset()


def test_topo_core_002_strict_binary_topology_resolution_parity() -> None:
    """ID: TOPO_CORE_002_strict_binary_topology_resolution_parity."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")
    plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="binary",
    )
    assert plan.sequence_dim == "sample"
    assert plan.batch_dims == ("trial",)
    assert plan.align_exclude_dims == frozenset()


def test_topo_core_003_realize_operand_topology_preserves_existing_coords() -> None:
    """ID: TOPO_CORE_003_realize_operand_topology_preserves_existing_coords."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")
    plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="binary",
    )
    realized_left, realized_right = realize_operands_for_plan(plan, owner="test.topology", what="binary")
    xr.testing.assert_identical(realized_left, left.unsafe_data["x"])
    xr.testing.assert_identical(realized_right, right.unsafe_data["x"])


def test_topo_core_010_nary_topology_path_uses_shared_touchpoint() -> None:
    """ID: TOPO_CORE_010_nary_topology_path_uses_shared_touchpoint."""
    a = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="r0", col="r1")
    b = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="r1", col="r2")
    c = _matrix_ao(np.arange(96, dtype=float).reshape(2, 2, 2, 12), row="r2", col="r3")
    plan = resolve_nary_topology(
        (
            _topology_operand(a, index=0),
            _topology_operand(b, index=1),
            _topology_operand(c, index=2),
        ),
        owner="test.topology",
        what="nary",
    )
    aligned = align_exact_for_plan(plan, owner="test.topology", what="nary")
    assert len(aligned) == 3
    assert plan.sequence_dim == "sample"
    assert plan.batch_dims == ("trial",)


def test_topo_core_011_nary_resolution_order_stability_strict_mode() -> None:
    """ID: TOPO_CORE_011_nary_resolution_order_stability_strict_mode."""
    a = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="r0", col="r1")
    b = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="r1", col="r2")
    c = _matrix_ao(np.arange(96, dtype=float).reshape(2, 2, 2, 12), row="r2", col="r3")
    plan_abc = resolve_nary_topology(
        (_topology_operand(a, index=0), _topology_operand(b, index=1), _topology_operand(c, index=2)),
        owner="test.topology",
        what="nary",
    )
    plan_cab = resolve_nary_topology(
        (_topology_operand(c, index=0), _topology_operand(a, index=1), _topology_operand(b, index=2)),
        owner="test.topology",
        what="nary",
    )
    assert plan_abc.sequence_dim == plan_cab.sequence_dim
    assert plan_abc.batch_dims == plan_cab.batch_dims
    assert plan_abc.align_exclude_dims == plan_cab.align_exclude_dims


def test_topo_core_015_policy_split_linalg_exact_vs_spatial_non_core_parity() -> None:
    """ID: TOPO_CORE_015_policy_split_linalg_exact_vs_spatial_non_core_parity."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")
    exact_plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="binary",
        policy=STRICT_EXACT_POLICY,
    )
    non_core_plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="binary",
        policy=STRICT_NON_CORE_POLICY,
    )
    assert exact_plan.align_exclude_dims == frozenset()
    assert non_core_plan.align_exclude_dims == frozenset({"row", "mid", "out"})


def test_topo_core_016_batch_stack_label_realization_parity_after_eager_cleanup() -> None:
    """ID: TOPO_CORE_016_batch_stack_label_realization_parity_after_eager_cleanup."""
    source = xr.Dataset(
        {
            "x": (
                ("sample", "trial", "sensor"),
                np.arange(24, dtype=float).reshape(2, 3, 4),
            )
        },
        coords={
            "sample": np.asarray([0, 1], dtype=np.int64),
            "trial": np.asarray(["t0", "t1", "t2"], dtype=object),
            "sensor": np.asarray([10, 20, 30, 40], dtype=np.int64),
        },
    )
    stacked = stack_combine_batch_axis(
        source,
        batch_dims=("trial", "sensor"),
        flat_dim="flat",
        owner="test.topology",
        singleton_unbatched=False,
    )
    labels = stacked.coords["flat"].to_numpy()
    assert tuple(labels[0]) == ("t0", 10)
    assert tuple(labels[-1]) == ("t2", 40)
    restored = restore_combine_batch_axis(
        stacked,
        batch_dims=("trial", "sensor"),
        flat_dim="flat",
        owner="test.topology",
        drop_unbatched_flat=True,
    )
    xr.testing.assert_allclose(restored["x"], source["x"])
    assert list(restored.coords["trial"].values) == ["t0", "t1", "t2"]
    assert list(restored.coords["sensor"].values) == [10, 20, 30, 40]


def test_topo_core_017_batch_restore_label_realization_parity_after_restore_eager_cleanup() -> None:
    """ID: TOPO_CORE_017_batch_restore_label_realization_parity_after_restore_eager_cleanup."""
    plan = BatchFlattenPlan(enabled=True, flat_dim="flat", batch_dims=("trial", "sensor"))
    flat_labels = np.empty(2, dtype=object)
    flat_labels[:] = [("sensor_0", "trial_0"), ("sensor_1", "trial_1")]
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("flat", "sample"),
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float64"),
            )
        },
        coords={
            "flat": ("flat", flat_labels),
            "trial": ("flat", ["trial_0", "trial_1"]),
            "sensor": ("flat", ["sensor_0", "sensor_1"]),
            "sample": [0, 1],
        },
    )
    restored = restore_dataset_batch_dims(ds, plan=plan, owner="test.topology")
    assert list(restored.coords["trial"].values) == ["trial_0", "trial_1"]
    assert list(restored.coords["sensor"].values) == ["sensor_0", "sensor_1"]
    np.testing.assert_allclose(
        restored["value"].sel(trial="trial_0", sensor="sensor_0").values,
        np.asarray([1.0, 2.0], dtype="float64"),
    )


def test_bcast_core_004_input_coercion_preserves_broadcast_intent() -> None:
    """ID: BCAST_CORE_004_input_coercion_preserves_broadcast_intent."""
    source = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    coerced = coerce_operand(source.a().b(), owner="test.broadcast", label="left")
    assert isinstance(coerced, AnalysisObject)
    intent = read_broadcast_intent(coerced, owner="test.broadcast")
    assert intent is not None
    assert intent.mode == "semantic_broadcast"
    alignment_intent = read_alignment_intent(coerced, owner="test.broadcast")
    assert alignment_intent is not None
    assert alignment_intent.on == "sequence"


def test_bcast_core_005_semantic_mode_accepts_missing_sequence_dim_operand() -> None:
    """ID: BCAST_CORE_005_semantic_mode_accepts_missing_sequence_dim_operand."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao_missing_sequence(
        np.arange(18, dtype=float).reshape(2, 3, 3),
        row="row",
        col="mid",
    )
    plan = resolve_binary_topology(
        TopologyOperand(
            index=0,
            data=left.unsafe_data["x"],
            semantic=resolve_semantic_topology_from_dataset(
                left.unsafe_data,
                var_name="x",
                core_dims=("row", "mid"),
                owner="test.broadcast",
                what="left",
                allow_missing_sequence_dim=True,
                allow_missing_batch_dims=True,
            ),
        ),
        TopologyOperand(
            index=1,
            data=right.unsafe_data["x"],
            semantic=resolve_semantic_topology_from_dataset(
                right.unsafe_data,
                var_name="x",
                core_dims=("row", "mid"),
                owner="test.broadcast",
                what="right",
                allow_missing_sequence_dim=True,
                allow_missing_batch_dims=True,
            ),
        ),
        owner="test.broadcast",
        what="semantic sequence broadcast",
        policy=SEMANTIC_EXACT_POLICY,
    )
    _, realized_right = align_exact_for_plan(plan, owner="test.broadcast", what="semantic sequence broadcast")
    assert "sample" in realized_right.dims
    assert realized_right.sizes["sample"] == left.unsafe_data.sizes["sample"]


def test_bcast_core_006_semantic_mode_accepts_missing_batch_dims_operand() -> None:
    """ID: BCAST_CORE_006_semantic_mode_accepts_missing_batch_dims_operand."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao_missing_batch(
        np.arange(18, dtype=float).reshape(2, 3, 3),
        row="row",
        col="mid",
    )
    plan = resolve_binary_topology(
        TopologyOperand(
            index=0,
            data=left.unsafe_data["x"],
            semantic=resolve_semantic_topology_from_dataset(
                left.unsafe_data,
                var_name="x",
                core_dims=("row", "mid"),
                owner="test.broadcast",
                what="left",
                allow_missing_sequence_dim=True,
                allow_missing_batch_dims=True,
            ),
        ),
        TopologyOperand(
            index=1,
            data=right.unsafe_data["x"],
            semantic=resolve_semantic_topology_from_dataset(
                right.unsafe_data,
                var_name="x",
                core_dims=("row", "mid"),
                owner="test.broadcast",
                what="right",
                allow_missing_sequence_dim=True,
                allow_missing_batch_dims=True,
            ),
        ),
        owner="test.broadcast",
        what="semantic batch broadcast",
        policy=SEMANTIC_EXACT_POLICY,
    )
    _, realized_right = align_exact_for_plan(plan, owner="test.broadcast", what="semantic batch broadcast")
    assert "trial" in realized_right.dims
    assert realized_right.sizes["trial"] == left.unsafe_data.sizes["trial"]


def test_bcast_core_012_nary_optin_path_uses_shared_topology_touchpoint() -> None:
    """ID: BCAST_CORE_012_nary_optin_path_uses_shared_topology_touchpoint."""
    full = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    missing_seq = _matrix_ao_missing_sequence(
        (np.arange(18, dtype=float).reshape(2, 3, 3) + 1.0) / 3.0,
        row="row",
        col="mid",
    )
    missing_batch = _matrix_ao_missing_batch(
        (np.arange(18, dtype=float).reshape(2, 3, 3) + 2.0) / 5.0,
        row="row",
        col="mid",
    )
    plan = resolve_nary_topology(
        (
            TopologyOperand(
                index=0,
                data=full.unsafe_data["x"],
                semantic=resolve_semantic_topology_from_dataset(
                    full.unsafe_data,
                    var_name="x",
                    core_dims=("row", "mid"),
                    owner="test.broadcast",
                    what="full",
                    allow_missing_sequence_dim=True,
                    allow_missing_batch_dims=True,
                ),
            ),
            TopologyOperand(
                index=1,
                data=missing_seq.unsafe_data["x"],
                semantic=resolve_semantic_topology_from_dataset(
                    missing_seq.unsafe_data,
                    var_name="x",
                    core_dims=("row", "mid"),
                    owner="test.broadcast",
                    what="missing sequence",
                    allow_missing_sequence_dim=True,
                    allow_missing_batch_dims=True,
                ),
            ),
            TopologyOperand(
                index=2,
                data=missing_batch.unsafe_data["x"],
                semantic=resolve_semantic_topology_from_dataset(
                    missing_batch.unsafe_data,
                    var_name="x",
                    core_dims=("row", "mid"),
                    owner="test.broadcast",
                    what="missing batch",
                    allow_missing_sequence_dim=True,
                    allow_missing_batch_dims=True,
                ),
            ),
        ),
        owner="test.broadcast",
        what="semantic nary",
        policy=SEMANTIC_EXACT_POLICY,
    )
    realized = align_exact_for_plan(plan, owner="test.broadcast", what="semantic nary")
    assert len(realized) == 3
    assert "sample" in realized[1].dims
    assert "trial" in realized[2].dims


def test_bcast_hard_003_semantic_mode_rejects_core_dim_broadcast_attempts() -> None:
    """ID: BCAST_HARD_003_semantic_mode_rejects_core_dim_broadcast_attempts."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right_ds = xr.Dataset(
        {"x": (("sample", "trial", "row"), np.arange(12, dtype=float).reshape(2, 2, 3))},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "row": np.arange(3, dtype=np.int64),
        },
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "mid"),
        validate=False,
    )
    with pytest.raises(ValueError, match="requires core dims"):
        _ = resolve_binary_topology(
            TopologyOperand(
                index=0,
                data=left.unsafe_data["x"],
                semantic=SemanticTopology("sample", ("trial",), ("row", "mid")),
            ),
            TopologyOperand(
                index=1,
                data=right.unsafe_data["x"],
                semantic=SemanticTopology("sample", ("trial",), ("row", "mid")),
            ),
            owner="test.broadcast",
            what="semantic core",
            policy=SEMANTIC_EXACT_POLICY,
        )


def test_bcast_hard_008_nary_mixed_intent_conflict_fail_closed_with_owner_context() -> None:
    """ID: BCAST_HARD_008_nary_mixed_intent_conflict_fail_closed_with_owner_context."""
    full = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    bad = _matrix_ao_with_extra_non_core(
        np.arange(24, dtype=float).reshape(2, 2, 1, 3, 2),
        row="row",
        col="mid",
        extra_dim="camera",
    )
    with pytest.raises(
        ValueError,
        match=r"^owner\.broadcast: .*semantic broadcast only permits missing sequence/batch dims",
    ):
        _ = resolve_nary_topology(
            (
                TopologyOperand(
                    index=0,
                    data=full.unsafe_data["x"],
                    semantic=resolve_semantic_topology_from_dataset(
                        full.unsafe_data,
                        var_name="x",
                        core_dims=("row", "mid"),
                        owner="owner.broadcast",
                        what="full",
                        allow_missing_sequence_dim=True,
                        allow_missing_batch_dims=True,
                    ),
                ),
                TopologyOperand(
                    index=1,
                    data=bad.unsafe_data["x"],
                    semantic=resolve_semantic_topology_from_dataset(
                        bad.unsafe_data,
                        var_name="x",
                        core_dims=("row", "mid"),
                        owner="owner.broadcast",
                        what="bad",
                        allow_missing_sequence_dim=True,
                        allow_missing_batch_dims=True,
                    ),
                ),
            ),
            owner="owner.broadcast",
            what="semantic nary",
            policy=SEMANTIC_NON_CORE_POLICY,
        )


def test_bcast_hard_024_no_interpolation_side_effects_in_e3a_default_paths() -> None:
    """ID: BCAST_HARD_024_no_interpolation_side_effects_in_e3a_default_paths."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao(
        (np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 1.0) / 9.0,
        row="row",
        col="mid",
    )
    right_da = right.unsafe_data["x"].assign_coords(sample=np.asarray([10, 11], dtype=np.int64))
    plan = resolve_binary_topology(
        TopologyOperand(
            index=0,
            data=left.unsafe_data["x"],
            semantic=resolve_semantic_topology_from_dataset(
                left.unsafe_data,
                var_name="x",
                core_dims=("row", "mid"),
                owner="owner.broadcast",
                what="left",
                allow_missing_sequence_dim=True,
                allow_missing_batch_dims=True,
            ),
        ),
        TopologyOperand(
            index=1,
            data=right_da,
            semantic=resolve_semantic_topology_from_dataset(
                right.unsafe_data,
                var_name="x",
                core_dims=("row", "mid"),
                owner="owner.broadcast",
                what="right",
                allow_missing_sequence_dim=True,
                allow_missing_batch_dims=True,
            ),
        ),
        owner="owner.broadcast",
        what="semantic exact no interpolation",
        policy=SEMANTIC_EXACT_POLICY,
    )
    with pytest.raises(ValueError, match="requires exact 'sample' labels under sequence/batch policy"):
        _ = align_exact_for_plan(plan, owner="owner.broadcast", what="semantic exact no interpolation")


def test_topo_hard_012_batch_stack_chunked_coordinate_labels_fail_closed_no_hidden_compute() -> None:
    """ID: TOPO_HARD_012_batch_stack_chunked_coordinate_labels_fail_closed_no_hidden_compute."""
    stacked = xr.Dataset(
        {
            "trial": (
                ("flat",),
                da.from_array(np.asarray(["t0", "t1"], dtype=object), chunks=(1,)),
            ),
            "sensor": (
                ("flat",),
                da.from_array(np.asarray([0, 1], dtype=np.int64), chunks=(1,)),
            ),
        }
    )
    with pytest.raises(
        ValueError,
        match=r"^owner\.topology: chunked batch coordinate labels are not supported",
    ):
        _ = stacked_batch_coords_with_labels(
            stacked=stacked,
            batch_dims=("trial", "sensor"),
            flat_dim="flat",
            owner="owner.topology",
        )


def test_topo_hard_013_orchestration_stack_path_avoids_values_and_np_asarray_patterns() -> None:
    """ID: TOPO_HARD_013_orchestration_stack_path_avoids_values_and_np_asarray_patterns."""
    source = inspect.getsource(stack_combine_batch_axis)
    assert ".values" not in source
    assert "np.asarray(" not in source


def test_topo_hard_014_batch_restore_chunked_named_flat_coordinate_labels_fail_closed_no_hidden_compute() -> None:
    """ID: TOPO_HARD_014_batch_restore_chunked_named_flat_coordinate_labels_fail_closed_no_hidden_compute."""
    labels = np.empty(2, dtype=object)
    labels[:] = [("t0", "s0"), ("t1", "s1")]
    ds = xr.Dataset(
        data_vars={"x": (("flat", "sample"), np.asarray([[1.0], [2.0]], dtype=float))},
        coords={
            "flat": ("flat", labels),
            "trial": (
                ("flat",),
                da.from_array(np.asarray(["t0", "t1"], dtype=object), chunks=(1,)),
            ),
            "sensor": (
                ("flat",),
                da.from_array(np.asarray(["s0", "s1"], dtype=object), chunks=(1,)),
            ),
            "sample": [0],
        },
    )
    with pytest.raises(
        ValueError,
        match=r"^owner\.topology: chunked batch coordinate labels are not supported for dim",
    ):
        _ = restore_combine_batch_axis(
            ds,
            batch_dims=("trial", "sensor"),
            flat_dim="flat",
            owner="owner.topology",
            drop_unbatched_flat=True,
        )


def test_topo_hard_015_batch_restore_chunked_composite_flat_labels_fail_closed_no_hidden_compute() -> None:
    """ID: TOPO_HARD_015_batch_restore_chunked_composite_flat_labels_fail_closed_no_hidden_compute."""
    labels = np.empty(2, dtype=object)
    labels[:] = [("t0", "s0"), ("t1", "s1")]
    coord = xr.DataArray(
        da.from_array(labels, chunks=(1,)),
        dims=("flat",),
        name="flat",
    )
    with pytest.raises(
        ValueError,
        match=r"^owner\.topology: chunked batch coordinate labels are not supported for dim 'flat'",
    ):
        _ = batch_topology_ops._coord_values_for_labels(  # type: ignore[attr-defined]
            coord,
            owner="owner.topology",
            dim="flat",
        )


def test_topo_hard_001_strict_mode_rejects_non_core_dim_name_mismatch() -> None:
    """ID: TOPO_HARD_001_strict_mode_rejects_non_core_dim_name_mismatch."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao_with_extra_non_core(
        np.arange(24, dtype=float).reshape(2, 2, 1, 3, 2),
        row="mid",
        col="out",
        extra_dim="camera",
    )
    with pytest.raises(ValueError, match="matching non-core dim names"):
        _ = resolve_binary_topology(
            _topology_operand(left, index=0),
            _topology_operand(right, index=1),
            owner="test.topology",
            what="binary",
        )


def test_topo_hard_002_strict_mode_rejects_extra_unmatched_non_core_dims() -> None:
    """ID: TOPO_HARD_002_strict_mode_rejects_extra_unmatched_non_core_dims."""
    left = _matrix_ao_with_extra_non_core(
        np.arange(24, dtype=float).reshape(2, 2, 1, 3, 2),
        row="row",
        col="mid",
        extra_dim="camera",
    )
    right = _matrix_ao_with_extra_non_core(
        np.arange(24, dtype=float).reshape(2, 2, 1, 3, 2),
        row="mid",
        col="out",
        extra_dim="sensor",
    )
    with pytest.raises(ValueError, match="left_only|right_only"):
        _ = resolve_binary_topology(
            _topology_operand(left, index=0),
            _topology_operand(right, index=1),
            owner="test.topology",
            what="binary",
        )


def test_topo_hard_003_topology_resolution_failures_owner_prefixed() -> None:
    """ID: TOPO_HARD_003_topology_resolution_failures_owner_prefixed."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right_ds = left.unsafe_data.rename({"sample": "step"})
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="step",
        batch_dims=("trial",),
        core_dims=("row", "mid"),
        validate=True,
    )
    with pytest.raises(ValueError, match="^owner\\.topology:"):
        _ = resolve_binary_topology(
            _topology_operand(left, index=0),
            _topology_operand(right, index=1),
            owner="owner.topology",
            what="binary",
        )


def test_topo_hard_004_realization_failures_owner_prefixed() -> None:
    """ID: TOPO_HARD_004_realization_failures_owner_prefixed."""
    plan = ResolvedTopologyPlan(
        sequence_dim=None,
        batch_dims=(),
        operands=(),
        align_exclude_dims=frozenset(),
    )
    with pytest.raises(ValueError, match="^owner\\.topology:"):
        _ = realize_operands_for_plan(plan, owner="owner.topology", what="empty plan")


def test_topo_hard_006_exact_alignment_still_required_after_realization() -> None:
    """ID: TOPO_HARD_006_exact_alignment_still_required_after_realization."""
    left = _matrix_ao(
        np.arange(36, dtype=float).reshape(2, 2, 3, 3),
        row="row",
        col="mid",
        trial_labels=("t0", "t1"),
    )
    right = _matrix_ao(
        np.arange(24, dtype=float).reshape(2, 2, 3, 2),
        row="mid",
        col="out",
        trial_labels=("t0", "t2"),
    )
    plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="binary",
    )
    with pytest.raises(ValueError, match="requires exact 'trial' labels under sequence/batch policy"):
        _ = align_exact_for_plan(plan, owner="test.topology", what="binary")


def test_bcast_core_039_operation_uses_exactly_one_primary_alignment_key() -> None:
    """ID: BCAST_CORE_039_operation_uses_exactly_one_primary_alignment_key."""
    left = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    right = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2) + 1.0,
        sample_labels=np.asarray(["a", "b", "c"], dtype=object),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    seq_plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="primary sequence",
        policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="sequence"),
    )
    param_plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="primary param",
        policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="param", sequence_join=None),
    )
    assert seq_plan.primary_key == "sequence"
    assert param_plan.primary_key == "param"


def test_bcast_core_040_sequence_and_batch_join_policies_are_applied_independently() -> None:
    """ID: BCAST_CORE_040_sequence_and_batch_join_policies_are_applied_independently."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    right_ds = _matrix_ao(
        (np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 1.0) / 5.0,
        row="row",
        col="col",
    ).unsafe_data.assign_coords(sample=np.asarray([1, 2], dtype=np.int64))
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        validate=True,
    )
    plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="sequence inner batch exact",
        policy=TopologyPolicy(
            mode="semantic_broadcast",
            alignment_on="sequence",
            sequence_join="inner",
            batch_join="exact",
        ),
    )
    aligned_left, aligned_right = align_exact_for_plan(plan, owner="test.topology", what="sequence inner batch exact")
    np.testing.assert_array_equal(aligned_left.coords["sample"].values, np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(aligned_right.coords["sample"].values, np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(aligned_left.coords["trial"].values, np.asarray(["t0", "t1"], dtype=object))


def test_bcast_core_041_param_primary_key_alignment_exact_mode_supported() -> None:
    """ID: BCAST_CORE_041_param_primary_key_alignment_exact_mode_supported."""
    left = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    right = _param_vector_ao(
        (np.arange(6, dtype=float).reshape(3, 2) + 2.0) / 3.0,
        sample_labels=np.asarray(["a", "b", "c"], dtype=object),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="param exact",
        policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="param", sequence_join=None),
    )
    aligned_left, aligned_right = align_exact_for_plan(plan, owner="test.topology", what="param exact")
    np.testing.assert_array_equal(aligned_left.coords["sample"].values, np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(aligned_right.coords["sample"].values, np.asarray([0, 1, 2], dtype=np.int64))


def test_bcast_core_043_param_primary_key_disables_sequence_label_matching() -> None:
    """ID: BCAST_CORE_043_param_primary_key_disables_sequence_label_matching."""
    left = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    right = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2) + 5.0,
        sample_labels=np.asarray(["a", "b", "c"], dtype=object),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    sequence_plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="sequence exact",
        policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="sequence", sequence_join="exact"),
    )
    with pytest.raises(ValueError, match="requires exact 'sample' labels"):
        _ = align_exact_for_plan(sequence_plan, owner="test.topology", what="sequence exact")
    param_plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="param exact",
        policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="param", sequence_join=None),
    )
    _ = align_exact_for_plan(param_plan, owner="test.topology", what="param exact")


def test_bcast_core_052_spatial_nonapproved_families_remain_on_existing_strict_behavior() -> None:
    """ID: BCAST_CORE_052_spatial_nonapproved_families_remain_on_existing_strict_behavior."""
    family = "spatial.path_solve"
    mode = default_topology_mode_for_operation_family(family, owner="test.topology")
    assert mode == "strict"
    assert alignment_intent_supported_for_operation_family(family, owner="test.topology") is False


def test_bcast_hard_026_invalid_mixed_key_requests_fail_closed() -> None:
    """ID: BCAST_HARD_026_invalid_mixed_key_requests_fail_closed."""
    left = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    right = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray(["a", "b", "c"], dtype=object),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="param invalid sequence join",
        policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="param", sequence_join="inner"),
    )
    with pytest.raises(ValueError, match="would re-key correspondence under on='param'"):
        _ = align_exact_for_plan(plan, owner="test.topology", what="param invalid sequence join")


def test_bcast_hard_027_auto_key_ambiguity_fails_closed() -> None:
    """ID: BCAST_HARD_027_auto_key_ambiguity_fails_closed."""
    left = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    right = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2) + 1.0,
        sample_labels=np.asarray(["a", "b", "c"], dtype=object),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    with pytest.raises(ValueError, match="alignment_on='auto' is ambiguous"):
        _ = resolve_binary_topology(
            _topology_operand(left, index=0),
            _topology_operand(right, index=1),
            owner="test.topology",
            what="auto key",
            policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="auto"),
        )


def test_bcast_hard_028_param_alignment_requires_shared_usable_param_coord() -> None:
    """ID: BCAST_HARD_028_param_alignment_requires_shared_usable_param_coord."""
    left = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    right = AnalysisObject.from_data(
        xr.Dataset(
            {"x": (("sample", "axis"), np.arange(6, dtype=float).reshape(3, 2))},
            coords={"sample": np.asarray([0, 1, 2], dtype=np.int64), "axis": np.asarray([0, 1], dtype=np.int64)},
        ),
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
        validate=True,
    )
    with pytest.raises(ValueError, match="requires shared param_coord"):
        _ = resolve_binary_topology(
            _topology_operand(left, index=0),
            _topology_operand(right, index=1),
            owner="test.topology",
            what="param required",
            policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="param", sequence_join=None),
        )


def test_bcast_hard_029_param_alignment_rejects_ambiguous_or_nondeterministic_key_mapping() -> None:
    """ID: BCAST_HARD_029_param_alignment_rejects_ambiguous_or_nondeterministic_key_mapping."""
    left = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2),
        sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
        param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    right = _param_vector_ao(
        np.arange(6, dtype=float).reshape(3, 2) + 1.0,
        sample_labels=np.asarray(["a", "b", "c"], dtype=object),
        param_values=np.asarray([0.0, 0.5, 0.5], dtype=float),
    )
    plan = resolve_binary_topology(
        _topology_operand(left, index=0),
        _topology_operand(right, index=1),
        owner="test.topology",
        what="param duplicate labels",
        policy=TopologyPolicy(mode="semantic_broadcast", alignment_on="param", sequence_join=None),
    )
    with pytest.raises(ValueError, match="requires unique param labels"):
        _ = align_exact_for_plan(plan, owner="test.topology", what="param duplicate labels")


def test_bcast_hard_030_numpy_named_core_policy_rejects_ambiguous_core_name_mismatch() -> None:
    """ID: BCAST_HARD_030_numpy_named_core_policy_rejects_ambiguous_core_name_mismatch."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    right = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="u", col="v")
    with pytest.raises(ValueError, match="core_policy='numpy_named'"):
        _ = resolve_binary_topology(
            _topology_operand(left, index=0),
            _topology_operand(right, index=1),
            owner="test.topology",
            what="numpy named mismatch",
            policy=TopologyPolicy(mode="semantic_broadcast", core_policy="numpy_named"),
        )


def test_bcast_hard_044_strict_core_policy_rejects_mismatched_core_dims_in_shared_topology() -> None:
    """ID: BCAST_HARD_044_strict_core_policy_rejects_mismatched_core_dims_in_shared_topology."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    right = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="u", col="v")
    with pytest.raises(ValueError, match="matching core dims under core_policy='strict'"):
        _ = resolve_binary_topology(
            _topology_operand(left, index=0),
            _topology_operand(right, index=1),
            owner="test.topology",
            what="strict mismatch",
            policy=TopologyPolicy(
                mode="semantic_broadcast",
                core_policy="strict",
                strict_core_match_required=True,
            ),
        )


def test_topo_hard_007_nary_strict_mode_rejects_incompatible_operand_set_fail_closed() -> None:
    """ID: TOPO_HARD_007_nary_strict_mode_rejects_incompatible_operand_set_fail_closed."""
    a = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="r0", col="r1")
    b = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="r1", col="r2")
    c_ds = xr.Dataset(
        {"x": (("sample", "sensor", "r2", "r3"), np.arange(24, dtype=float).reshape(2, 2, 3, 2))},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "sensor": np.asarray(["s0", "s1"], dtype=object),
            "r2": np.arange(3, dtype=np.int64),
            "r3": np.arange(2, dtype=np.int64),
        },
    )
    c = AnalysisObject.from_data(
        c_ds,
        sequence_dim="sample",
        batch_dims=("sensor",),
        core_dims=("r2", "r3"),
        validate=True,
    )
    with pytest.raises(ValueError, match="matching batch_dims"):
        _ = resolve_nary_topology(
            (_topology_operand(a, index=0), _topology_operand(b, index=1), _topology_operand(c, index=2)),
            owner="test.topology",
            what="nary",
        )


def test_topo_hard_008_unary_topology_misclassification_fail_closed_with_owner_context() -> None:
    """ID: TOPO_HARD_008_unary_topology_misclassification_fail_closed_with_owner_context."""
    ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    with pytest.raises(ValueError, match="^owner\\.topology:"):
        _ = resolve_unary_topology(
            _topology_operand(ao, index=0),
            owner="owner.topology",
            what="unary",
            policy=TopologyPolicy(mode="bad_mode"),  # type: ignore[arg-type]
        )
