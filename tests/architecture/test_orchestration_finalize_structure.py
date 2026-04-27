from __future__ import annotations

from pathlib import Path


def test_orch_finalize_arch_001_single_owner_finalize_wrappers() -> None:
    """ID: ORCH_FINALIZE_ARCH_001_single_owner_finalize_wrappers."""
    text = Path("tal/core/orchestration/finalize.py").read_text(encoding="utf-8")
    for needle in [
        "def finalize_like(",
        "def rewrap_unvalidated_like(",
        "def restore_and_finalize(",
        "def finalize_many_like(",
    ]:
        assert needle in text


def test_orch_finalize_arch_002_no_paramops_direct_ao_internal_finalize_calls() -> None:
    """ID: ORCH_FINALIZE_ARCH_002_no_paramops_direct_ao_internal_finalize_calls."""
    for rel in [
        "tal/core/param_ops/sync.py",
        "tal/core/param_ops/interp_like.py",
    ]:
        text = Path(rel).read_text(encoding="utf-8")
        assert "from ..ao_internal import" not in text, rel
        assert "finalize_structural(" not in text, rel
        assert "from_unvalidated_like(" not in text, rel


def test_orch_finalize_arch_003_no_combine_direct_ao_internal_finalize_calls() -> None:
    """ID: ORCH_FINALIZE_ARCH_003_no_combine_direct_ao_internal_finalize_calls."""
    for rel in [
        "tal/core/combine_ops/finalize.py",
        "tal/core/combine_ops/align.py",
    ]:
        text = Path(rel).read_text(encoding="utf-8")
        assert "from ..ao_internal import" not in text, rel
        assert "finalize_structural(" not in text, rel
        assert "from_unvalidated_like(" not in text, rel


def test_orch_finalize_arch_004_orchestration_finalize_reuses_ao_internal_only() -> None:
    """ID: ORCH_FINALIZE_ARCH_004_orchestration_finalize_reuses_ao_internal_only."""
    text = Path("tal/core/orchestration/finalize.py").read_text(encoding="utf-8")
    assert "from ..ao_internal import finalize_structural, from_unvalidated_like" in text
    assert "._finalize_structural(" not in text
    assert ".__class__._from_unvalidated(" not in text


def test_orch_finalize_arch_005_schema_finalize_owner_module_single_owner() -> None:
    """ID: ORCH_FINALIZE_ARCH_005_schema_finalize_owner_module_single_owner."""
    text = Path("tal/core/orchestration/schema_finalize.py").read_text(encoding="utf-8")
    assert "class CoreSchemaFinalizeSpec" in text
    assert "def clear_core_schema_blocks(" in text
    assert "def stamp_core_schema(" in text
    assert "def finalize_with_schema(" in text


def test_orch_finalize_arch_006_linalg_combine_reuse_schema_finalize_owner() -> None:
    """ID: ORCH_FINALIZE_ARCH_006_linalg_combine_reuse_schema_finalize_owner."""
    linalg = Path("tal/linalg/finalize.py").read_text(encoding="utf-8")
    combine = Path("tal/core/combine_ops/finalize.py").read_text(encoding="utf-8")
    assert "from ..core.orchestration.schema_finalize import" in linalg
    assert "from ..orchestration.schema_finalize import" in combine
    assert "set_roles(" not in linalg
    assert "set_param_coord(" not in linalg
    assert "set_validity(" not in linalg
    assert "set_roles(" not in combine
    assert "set_param_coord(" not in combine
    assert "set_validity(" not in combine


def test_orch_finalize_arch_007_event_layouts_reuse_event_finalize_owner() -> None:
    """ID: ORCH_FINALIZE_ARCH_007_event_layouts_reuse_event_finalize_owner."""
    event_finalize = Path("tal/core/event_ops/finalize.py").read_text(encoding="utf-8")
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    stacked = Path("tal/core/event_ops/around_stacked.py").read_text(encoding="utf-8")
    stream = Path("tal/core/event_ops/when_stream.py").read_text(encoding="utf-8")
    segments = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    assert "def finalize_event_output(" in event_finalize
    for text in (around, stacked, stream, segments):
        assert "finalize_event_output(" in text
    assert "def _finalize_around_output(" not in around
    assert "def _finalize_stacked_output(" not in stacked
    assert "def _finalize_stream_output(" not in stream
    assert "def _finalize_output_schema(" not in segments
