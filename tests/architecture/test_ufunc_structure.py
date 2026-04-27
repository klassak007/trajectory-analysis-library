from __future__ import annotations

from pathlib import Path


def test_arch_ufunc_001_owner_split_modules_present() -> None:
    """ID: ARCH_UFUNC_001_owner_split_modules_present."""
    pkg = Path("tal/core/ufunc_ops")
    assert (pkg / "__init__.py").exists()
    assert (pkg / "registry.py").exists()
    assert (pkg / "orchestrate.py").exists()
    assert (pkg / "kernel.py").exists()
    assert (pkg / "finalize.py").exists()
    assert (pkg / "api.py").exists()


def test_arch_ufunc_002_no_eager_antipatterns_in_orchestrate_finalize() -> None:
    """ID: ARCH_UFUNC_002_no_eager_antipatterns_in_orchestrate_finalize."""
    orchestrate = Path("tal/core/ufunc_ops/orchestrate.py").read_text(encoding="utf-8")
    finalize = Path("tal/core/ufunc_ops/finalize.py").read_text(encoding="utf-8")
    for text in (orchestrate, finalize):
        assert ".values" not in text
        assert ".item(" not in text
        assert ".compute(" not in text
        assert "np.asarray(" not in text


def test_arch_ufunc_003_no_core_to_linalg_imports() -> None:
    """ID: ARCH_UFUNC_003_no_core_to_linalg_imports."""
    for path in Path("tal/core/ufunc_ops").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "tal.linalg" not in text
        assert ".linalg" not in text


def test_arch_ufunc_004_public_facade_uses_registry_driven_export() -> None:
    """ID: ARCH_UFUNC_004_public_facade_uses_registry_driven_export."""
    text = Path("tal/ufuncs.py").read_text(encoding="utf-8")
    assert "UNARY_UFUNC_NAMES" in text
    assert "BINARY_UFUNC_NAMES" in text
    assert "apply_unary_ufunc" in text
    assert "apply_binary_ufunc" in text


def test_arch_ufunc_005_docs_surface_present() -> None:
    """ID: AO_UFUNC_DOC_001_docs_surface_present."""
    api_index = Path("docs/api/index.md").read_text(encoding="utf-8")
    api_ufuncs = Path("docs/api/ufuncs.md").read_text(encoding="utf-8")
    user_core = Path("docs/user-guide/core_concepts.md").read_text(encoding="utf-8")
    assert "ufuncs" in api_index
    assert "ufuncs" in api_ufuncs.lower()
    assert "tal.ufuncs" in api_ufuncs
    assert "tal.ufuncs" in user_core


def test_arch_ufunc_006_condition_truthiness_guard_present() -> None:
    """ID: ARCH_UFUNC_005_condition_truthiness_guard_present."""
    types_text = Path("tal/core/event_ops/types.py").read_text(encoding="utf-8")
    assert "def __bool__(self)" in types_text
    assert "ao.events.mask(condition)" in types_text


def test_arch_ufunc_007_tal_ufuncs_root_facade_path_stable() -> None:
    """ID: ARCH_UFUNC_007_tal_ufuncs_root_facade_path_stable."""
    assert Path("tal/ufuncs.py").exists()
    assert not Path("tal/core/ufuncs.py").exists()


def test_arch_ufunc_008_tal_ufuncs_root_facade_delegates_to_core_ufunc_ops() -> None:
    """ID: ARCH_UFUNC_008_tal_ufuncs_root_facade_delegates_to_core_ufunc_ops."""
    text = Path("tal/ufuncs.py").read_text(encoding="utf-8")
    assert "from tal.core.ufunc_ops import" in text
    assert "apply_unary_ufunc(" in text
    assert "apply_binary_ufunc(" in text
    assert "tal.spatial" not in text


def test_arch_bcast_016_ufunc_api_contains_no_broadcast_intent_rejection_gate() -> None:
    """ID: ARCH_BCAST_016_ufunc_api_contains_no_broadcast_intent_rejection_gate."""
    api_text = Path("tal/core/ufunc_ops/api.py").read_text(encoding="utf-8")
    assert "_reject_broadcast_intent" not in api_text


def test_arch_bcast_017_ufunc_orchestration_uses_shared_topology_policy_for_intent() -> None:
    """ID: ARCH_BCAST_017_ufunc_orchestration_uses_shared_topology_policy_for_intent."""
    orchestrate_text = Path("tal/core/ufunc_ops/orchestrate.py").read_text(encoding="utf-8")
    assert "select_topology_policy_with_intents(" in orchestrate_text
    assert 'operation_family="ufunc.arithmetic"' in orchestrate_text
    assert "resolve_unary_topology(" in orchestrate_text
    assert "resolve_binary_topology(" in orchestrate_text
    assert "align_exact_for_plan(" in orchestrate_text


def test_arch_bcast_039_prepare_binary_ao_context_delegates_to_helper_paths() -> None:
    """ID: ARCH_BCAST_039_prepare_binary_ao_context_delegates_to_helper_paths."""
    orchestrate_text = Path("tal/core/ufunc_ops/orchestrate.py").read_text(encoding="utf-8")
    assert "def _prepare_strict_binary_ao_context(" in orchestrate_text
    assert "def _prepare_semantic_binary_ao_context(" in orchestrate_text
    body = orchestrate_text.split("def prepare_binary_ao_context(", 1)[1].split("\ndef ", 1)[0]
    assert "_prepare_strict_binary_ao_context(" in body
    assert "_prepare_semantic_binary_ao_context(" in body
