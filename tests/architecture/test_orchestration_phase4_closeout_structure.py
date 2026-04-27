from __future__ import annotations

import ast
from pathlib import Path


def _function_doc_count(path: Path) -> int:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in module.body
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node)
    )


def test_orch_guard_001_no_direct_xr_align_outside_orchestration_layer() -> None:
    """ID: ORCH_GUARD_001_no_direct_xr_align_outside_orchestration_layer."""
    for rel in [
        "tal/core/param_ops",
        "tal/core/combine_ops",
    ]:
        for path in sorted(Path(rel).glob("*.py")):
            text = path.read_text(encoding="utf-8")
            assert "xr.align(" not in text, f"direct xr.align usage must route through orchestration policies: {path}"


def test_orch_guard_002_no_direct_ao_private_finalize_calls_outside_ao_internal_or_orchestration_finalize() -> None:
    """ID: ORCH_GUARD_002_no_direct_ao_private_finalize_calls_outside_ao_internal_or_orchestration_finalize."""
    allowed = {
        "tal/core/analysis_object.py",
        "tal/core/ao_internal.py",
        "tal/core/orchestration/finalize.py",
    }
    for path in sorted(Path("tal/core").rglob("*.py")):
        rel = path.as_posix()
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        assert "._finalize_structural(" not in text, rel
        assert ".__class__._from_unvalidated(" not in text, rel


def test_orch_guard_003_no_local_flatten_plan_allocators_in_param_combine_callsites() -> None:
    """ID: ORCH_GUARD_003_no_local_flatten_plan_allocators_in_param_combine_callsites."""
    checks = {
        "tal/core/param_ops/sync.py": (
            "def _batch_flatten_plan(",
            "flatten_batch_contexts(",
            "flatten_query_for_plan(",
            "restore_dataset_batch_dims(",
        ),
        "tal/core/param_ops/interp_like.py": (
            "def _batch_flatten_plan(",
            "flatten_batch_contexts(",
            "flatten_query_for_plan(",
            "restore_dataset_batch_dims(",
        ),
        "tal/core/combine_ops/align.py": (
            "flatten_batch_contexts(",
            "flatten_query_for_plan(",
            "restore_dataset_batch_dims(",
        ),
        "tal/core/combine_ops/concat_topology.py": (
            "flatten_batch_contexts(",
            "restore_dataset_batch_dims(",
        ),
    }
    for rel, needles in checks.items():
        text = Path(rel).read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text, f"local flatten/restore allocator path remains in {rel}: {needle!r}"


def test_orch_doc_001_hotspot_docstrings_present() -> None:
    """ID: ORCH_DOC_001_hotspot_docstrings_present."""
    for rel in [
        "tal/core/combine_ops/concat_sequence.py",
        "tal/core/param_engine/schema_resolve.py",
        "tal/core/param_ops/sync_runtime.py",
    ]:
        path = Path(rel)
        assert _function_doc_count(path) >= 1, f"hotspot module lacks function docstrings: {rel}"
