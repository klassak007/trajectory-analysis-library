from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.architecture._budget import file_loc, function_lengths


def _has_direct_tal_schema_write(text: str) -> bool:
    return bool(re.search(r"attrs\[['\"]tal['\"]\]\s*=", text))


def test_arch_viz_001_viz_owner_modules_present_and_budgeted() -> None:
    """ID: ARCH_VIZ_001_viz_owner_modules_present_and_budgeted."""
    files = (
        Path("tal/viz/options.py"),
        Path("tal/viz/plan.py"),
        Path("tal/viz/prepare.py"),
        Path("tal/viz/holoviews_backend.py"),
        Path("tal/viz/accessor.py"),
        Path("tal/viz/surface.py"),
    )
    for path in files:
        assert path.exists(), f"missing viz owner module: {path}"
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget: {length} > 50"


def test_arch_viz_002_optional_imports_guarded_for_holoviews_backend() -> None:
    """ID: ARCH_VIZ_002_optional_imports_guarded_for_holoviews_backend."""
    text = Path("tal/viz/holoviews_backend.py").read_text(encoding="utf-8")
    assert 'importlib.import_module("holoviews")' in text
    assert 'importlib.import_module("hvplot.xarray")' in text
    assert "import holoviews" not in text
    assert "import hvplot.xarray" not in text


def test_arch_viz_003_reuses_core_owner_paths_no_local_duplication() -> None:
    """ID: ARCH_VIZ_003_reuses_core_owner_paths_no_local_duplication."""
    plan_text = Path("tal/viz/plan.py").read_text(encoding="utf-8")
    prep_text = Path("tal/viz/prepare.py").read_text(encoding="utf-8")
    surface_text = Path("tal/viz/surface.py").read_text(encoding="utf-8")
    assert "coerce_analysis_object_input" in plan_text
    assert "resolve_dataset_context" in plan_text
    assert "DatasetContextOptions(require_roles=True, require_sequence_dim=True)" in plan_text
    assert "resolve_structural_valid_mask" in prep_text
    assert "apply_structural_mask" in prep_text
    assert "resolve_grouping_foundation_context" in prep_text
    assert "extract_components" in surface_text
    assert "read_components" in surface_text


def test_arch_viz_004_no_schema_bypass_writes_and_no_core_to_viz_import() -> None:
    """ID: ARCH_VIZ_004_no_schema_bypass_writes_and_no_core_to_viz_import."""
    for path in sorted(Path("tal/viz").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not _has_direct_tal_schema_write(text)
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.viz" not in text
        assert "from ..viz" not in text
        assert "from .viz" not in text


def test_arch_viz_005_top_level_installation_wired() -> None:
    """ID: ARCH_VIZ_005_top_level_installation_wired."""
    text = Path("tal/__init__.py").read_text(encoding="utf-8")
    assert "from .viz import install_analysis_object_viz_surface" in text
    assert "install_analysis_object_viz_surface()" in text


def test_viz_doc_001_api_and_user_docs_include_viz_surface() -> None:
    """ID: VIZ_DOC_001_api_and_user_docs_include_viz_surface."""
    api_index = Path("docs/api/index.md").read_text(encoding="utf-8").lower()
    api_ao = Path("docs/api/analysis-object.md").read_text(encoding="utf-8").lower()
    api_viz = Path("docs/api/viz.md").read_text(encoding="utf-8").lower()
    viewing = Path("docs/user-guide/viewing.md").read_text(encoding="utf-8").lower()
    assert "viz" in api_index
    assert "ao.viz" in api_ao
    assert "ao.viz.line" in api_viz
    assert "tal.viz.line" in api_viz
    assert "ao.viz.component" in api_viz
    assert "group_key" in viewing
    assert "component(" in viewing
