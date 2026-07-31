from __future__ import annotations

import ast
from pathlib import Path


def _imports_orchestration_lazy(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("orchestration.lazy"):
                return True
            parent = node.module in {"orchestration", "tal.core.orchestration"}
            if parent and any(alias.name == "lazy" for alias in node.names):
                return True
        if isinstance(node, ast.Import):
            if any(alias.name.endswith("orchestration.lazy") for alias in node.names):
                return True
    return False


def test_orch_lazy_arch_001_single_owner_lazy_wrappers() -> None:
    """ID: ORCH_LAZY_ARCH_001_single_owner_lazy_wrappers."""
    text = Path("tal/core/orchestration/lazy.py").read_text(encoding="utf-8")
    for needle in [
        "def is_chunked_dataarray(",
        "def is_chunked_variable(",
        "def require_unchunked_dataarray(",
        "def require_unchunked_auto_grid_sources(",
        "def fail_if_chunked_boundary(",
    ]:
        assert needle in text


def test_orch_lazy_arch_002_no_sync_local_chunked_guard_helpers_in_migrated_files() -> None:
    """ID: ORCH_LAZY_ARCH_002_no_sync_local_chunked_guard_helpers_in_migrated_files."""
    checks = {
        "tal/core/param_ops/sync.py": (
            "def _is_chunked_dataarray(",
            "def _assert_auto_grid_chunked_supported(",
        ),
        "tal/core/param_ops/sync_runtime.py": ("def _require_unchunked_auto_grid_source(",),
    }
    for rel, needles in checks.items():
        text = Path(rel).read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text, f"local chunked guard helper {needle!r} remains in {rel}"


def test_orch_lazy_arch_003_no_combine_local_chunked_guard_helpers_in_migrated_files() -> None:
    """ID: ORCH_LAZY_ARCH_003_no_combine_local_chunked_guard_helpers_in_migrated_files."""
    concat_text = Path("tal/core/combine_ops/concat_sequence.py").read_text(encoding="utf-8")
    overlap_text = Path("tal/core/combine_ops/concat_overlap.py").read_text(encoding="utf-8")
    align_path = Path("tal/core/combine_ops/align.py")
    align_text = align_path.read_text(encoding="utf-8")
    assert "getattr(coord.data, \"chunks\", None)" not in concat_text
    assert "getattr(coord.data, \"chunks\", None)" not in align_text
    assert "from .concat_overlap import" in concat_text
    assert "require_unchunked_dataarray(" in overlap_text
    assert not _imports_orchestration_lazy(ast.parse(align_text, filename=str(align_path)))
    forbidden_sources = (
        "from ..orchestration import lazy",
        "from tal.core.orchestration import lazy as lazy_owner",
    )
    for source in forbidden_sources:
        assert _imports_orchestration_lazy(ast.parse(source))
    assert not _imports_orchestration_lazy(ast.parse("from ..orchestration import topology"))


def test_orch_lazy_arch_004_orchestration_lazy_no_hidden_eager_calls() -> None:
    """ID: ORCH_LAZY_ARCH_004_orchestration_lazy_no_hidden_eager_calls."""
    text = Path("tal/core/orchestration/lazy.py").read_text(encoding="utf-8")
    for forbidden in [".values", ".item(", "np.asarray", ".compute("]:
        assert forbidden not in text, f"unexpected eager pattern {forbidden!r} in orchestration lazy owner"


def test_orch_lazy_arch_005_sync_autogrid_materialization_reuses_shared_lazy_guard_owner() -> None:
    """ID: ORCH_LAZY_ARCH_005_sync_autogrid_materialization_reuses_shared_lazy_guard_owner."""
    sync_text = Path("tal/core/param_ops/sync.py").read_text(encoding="utf-8")
    autogrid_text = Path("tal/core/param_ops/sync_autogrid.py").read_text(encoding="utf-8")
    assert "require_unchunked_auto_grid_sources(" in sync_text
    assert "from ..orchestration.lazy import require_unchunked_dataarray" not in autogrid_text
    assert "require_unchunked_dataarray(" not in autogrid_text
    assert "def _is_chunked_dataarray(" not in autogrid_text
    assert "getattr(da.data, \"chunks\", None)" not in autogrid_text
