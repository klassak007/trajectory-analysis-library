from __future__ import annotations

from pathlib import Path

from ._budget import file_loc, function_lengths


def test_orch_concat_arch_001_single_owner_concat_plan_module() -> None:
    """ID: ORCH_CONCAT_ARCH_001_single_owner_concat_plan_module."""
    text = Path("tal/core/combine_ops/concat_plan.py").read_text(encoding="utf-8")
    assert "class ConcatSequencePlan" in text
    assert "def build_concat_sequence_plan(" in text


def test_orch_concat_arch_002_single_owner_concat_overlap_module() -> None:
    """ID: ORCH_CONCAT_ARCH_002_single_owner_concat_overlap_module."""
    text = Path("tal/core/combine_ops/concat_overlap.py").read_text(encoding="utf-8")
    assert "def validate_concat_overlap(" in text
    assert "def validate_grouped_sort_payload(" in text


def test_orch_concat_arch_003_concat_sequence_no_local_plan_owner_helpers() -> None:
    """ID: ORCH_CONCAT_ARCH_003_concat_sequence_no_local_plan_owner_helpers."""
    text = Path("tal/core/combine_ops/concat_sequence.py").read_text(encoding="utf-8")
    for needle in [
        "class ConcatSequencePlan",
        "def _prepare_concat_sequence(",
        "def _shared_name(",
    ]:
        assert needle not in text, f"local plan owner helper {needle!r} remains in concat_sequence.py"


def test_orch_concat_arch_004_concat_sequence_no_local_overlap_owner_helpers() -> None:
    """ID: ORCH_CONCAT_ARCH_004_concat_sequence_no_local_overlap_owner_helpers."""
    text = Path("tal/core/combine_ops/concat_sequence.py").read_text(encoding="utf-8")
    for needle in [
        "def _validate_concat_sequence_inputs(",
        "def _overlap_checks(",
        "def _check_monotonic(",
        "def _batched_param_values(",
        "def _reject_grouped_sample_only_payload(",
        "def _reject_grouped_sample_only_inputs(",
    ]:
        assert needle not in text, f"local overlap owner helper {needle!r} remains in concat_sequence.py"


def test_orch_concat_arch_005_concat_plan_reuses_topology_and_lazy_helpers() -> None:
    """ID: ORCH_CONCAT_ARCH_005_concat_plan_reuses_topology_and_lazy_helpers."""
    plan_text = Path("tal/core/combine_ops/concat_plan.py").read_text(encoding="utf-8")
    overlap_text = Path("tal/core/combine_ops/concat_overlap.py").read_text(encoding="utf-8")
    assert "from .concat_topology import" in plan_text
    assert "from ..orchestration.lazy import require_unchunked_dataarray" in overlap_text


def test_orch_concat_arch_006_single_owner_concat_pack_module() -> None:
    """ID: ORCH_CONCAT_ARCH_006_single_owner_concat_pack_module."""
    text = Path("tal/core/combine_ops/concat_pack.py").read_text(encoding="utf-8")
    assert "def build_packed_concat_dataset(" in text
    assert "def apply_concat_valid_mask(" in text
    assert "def apply_concat_overlap_sort(" in text


def test_orch_concat_arch_007_single_owner_concat_finalize_module() -> None:
    """ID: ORCH_CONCAT_ARCH_007_single_owner_concat_finalize_module."""
    text = Path("tal/core/combine_ops/concat_finalize.py").read_text(encoding="utf-8")
    assert "def postprocess_concat_dataset(" in text
    assert "def finalize_concat_sequence_contexts(" in text


def test_orch_concat_arch_008_concat_sequence_no_local_pack_owner_helpers() -> None:
    """ID: ORCH_CONCAT_ARCH_008_concat_sequence_no_local_pack_owner_helpers."""
    text = Path("tal/core/combine_ops/concat_sequence.py").read_text(encoding="utf-8")
    for needle in [
        "def _build_packing_index(",
        "def _mask_invalid_slots(",
        "def _normalize_runtime_reserved_coords(",
        "def _broadcast_sequence_only_payload(",
        "def _sequence_coord_names(",
        "def _concat_segment_inputs(",
        "def _packed_concat_dataset(",
        "def _apply_valid_mask(",
        "def _apply_overlap_sort(",
    ]:
        assert needle not in text, f"local pack owner helper {needle!r} remains in concat_sequence.py"


def test_orch_concat_arch_009_concat_sequence_no_local_finalize_owner_helpers() -> None:
    """ID: ORCH_CONCAT_ARCH_009_concat_sequence_no_local_finalize_owner_helpers."""
    text = Path("tal/core/combine_ops/concat_sequence.py").read_text(encoding="utf-8")
    for needle in [
        "def _canonical_optional_names(",
        "def _assign_total_size(",
        "def _postprocess_concat_output(",
    ]:
        assert needle not in text, f"local finalize owner helper {needle!r} remains in concat_sequence.py"


def test_orch_concat_arch_010_concat_finalize_reuses_orchestration_finalize_boundary() -> None:
    """ID: ORCH_CONCAT_ARCH_010_concat_finalize_reuses_orchestration_finalize_boundary."""
    text = Path("tal/core/combine_ops/concat_finalize.py").read_text(encoding="utf-8")
    assert "from .finalize import finalize_combine_output" in text
    assert "from ..ao_internal import" not in text
    assert "finalize_structural(" not in text
    assert "from_unvalidated_like(" not in text


def test_orch_concat_metric_001_concat_sequence_driver_size_reduced() -> None:
    """ID: ORCH_CONCAT_METRIC_001_concat_sequence_driver_size_reduced."""
    path = Path("tal/core/combine_ops/concat_sequence.py")
    count = file_loc(path=path)
    assert count <= 240, f"concat_sequence.py exceeds driver target: {count} > 240"


def test_orch_concat_metric_002_concat_sequence_driver_nesting_budget() -> None:
    """ID: ORCH_CONCAT_METRIC_002_concat_sequence_driver_nesting_budget."""
    path = Path("tal/core/combine_ops/concat_sequence.py")
    funcs = function_lengths(path)
    assert funcs["concat_sequence_contexts"] <= 50


def test_orch_concat_metric_003_phase_owner_modules_within_budget() -> None:
    """ID: ORCH_CONCAT_METRIC_003_phase_owner_modules_within_budget."""
    for rel in [
        "tal/core/combine_ops/concat_pack.py",
        "tal/core/combine_ops/concat_finalize.py",
    ]:
        path = Path(rel)
        assert file_loc(path=path) <= 600, f"{rel} exceeds file budget"
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{rel}:{name} exceeds function budget: {length} > 50"
