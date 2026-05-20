from __future__ import annotations

from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVELOPER_GUIDE_DIR = REPO_ROOT / "docs" / "developer-guide"
DOMAIN_EXTENSIONS_PATH = DEVELOPER_GUIDE_DIR / "domain_extensions.md"
DEVELOPER_INDEX_PATH = DEVELOPER_GUIDE_DIR / "index.md"
DOCS_INDEX_PATH = REPO_ROOT / "docs" / "index.md"
PYTHON_BLOCK_RE = re.compile(r"```python\b")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
EXAMPLE_BLOCK_RE = re.compile(
    r"<!--\s*example-id:\s*(?P<example_id>[A-Za-z0-9_]+)\s*-->\s*"
    r"```python\n(?P<code>.*?)\n```",
    re.DOTALL,
)


def _domain_extension_text() -> str:
    return DOMAIN_EXTENSIONS_PATH.read_text(encoding="utf-8")


def _visible_domain_extension_text() -> str:
    return HTML_COMMENT_RE.sub("", _domain_extension_text())


def _visible_developer_guide_text() -> str:
    parts = [
        HTML_COMMENT_RE.sub("", DEVELOPER_INDEX_PATH.read_text(encoding="utf-8")),
        _visible_domain_extension_text(),
    ]
    return "\n".join(parts)


def _example_blocks() -> dict[str, str]:
    return {
        match.group("example_id"): match.group("code")
        for match in EXAMPLE_BLOCK_RE.finditer(_domain_extension_text())
    }


def test_docs_dev_001_domain_extension_guide_exists() -> None:
    """ID: DOCS_DEV_001_domain_extension_guide_exists."""
    assert DEVELOPER_INDEX_PATH.exists()
    assert DOMAIN_EXTENSIONS_PATH.exists()
    assert "domain_extensions" in DEVELOPER_INDEX_PATH.read_text(encoding="utf-8")
    assert "developer-guide/index" in DOCS_INDEX_PATH.read_text(encoding="utf-8")


def test_docs_dev_002_domain_extension_schema_recipe() -> None:
    """ID: DOCS_DEV_002_domain_extension_schema_recipe."""
    text = _visible_domain_extension_text()
    required = [
        'ds.attrs["tal"]',
        "tal.ext.<namespace>",
        "tal.ext.thermal",
        "Unknown extension namespaces must be preserved",
        "AnalysisObject.from_data",
        "AnalysisObject.merge_schema",
        "Typed AO Anatomy",
        "TypedAnalysisObject",
        "TypedLifecycleSpec.normalize",
        "TypedLifecycleSpec.enforce",
        "class Temperature(TypedAnalysisObject)",
        "typing.Self",
        "from typing import Self",
        "from_celsius(...) -> Self",
        "default_coerce_source",
        "identity_init_options",
        "identity_normalize",
        "no_op_enforce",
    ]
    missing = [snippet for snippet in required if snippet not in text]
    assert not missing, f"Missing schema recipe snippets: {missing!r}"


def test_docs_dev_003_domain_extension_orchestrate_kernel_finalize_recipe() -> None:
    """ID: DOCS_DEV_003_domain_extension_orchestrate_kernel_finalize_recipe."""
    text = _visible_domain_extension_text()
    required = [
        "coerce_operand",
        "resolve_dataset_context",
        "pure array math",
        "finalize_like",
        "_from_validated",
        "_from_unvalidated",
        "owner string",
        "fully qualified public import path",
        "thermal.bias_temperature",
        "thermal.Temperature.from_celsius",
        "thermal.Temperature.celsius_values",
    ]
    missing = [snippet for snippet in required if snippet not in text]
    assert not missing, f"Missing operation recipe snippets: {missing!r}"


def test_docs_dev_004_domain_extension_dask_discipline_recipe() -> None:
    """ID: DOCS_DEV_004_domain_extension_dask_discipline_recipe."""
    text = _visible_domain_extension_text()
    required = [
        "Dask",
        ".values",
        ".item()",
        "np.asarray(...)",
        ".compute()",
        "eager materialization",
        "Subclass invariant hooks run on every typed object instantiation path",
        "cheap metadata checks",
        "Do not check actual data values or numeric ranges",
        "validate=False",
        "private rewrap helper",
        "performance regression coverage",
    ]
    missing = [snippet for snippet in required if snippet not in text]
    assert not missing, f"Missing Dask discipline snippets: {missing!r}"


def test_docs_dev_005_domain_extension_testing_recipe() -> None:
    """ID: DOCS_DEV_005_domain_extension_testing_recipe."""
    text = _visible_domain_extension_text()
    required = [
        "For each documented example, add executable coverage",
        "roles, metadata, values, and failure behavior",
        "deterministic",
        "independent from each other",
    ]
    missing = [snippet for snippet in required if snippet not in text]
    assert not missing, f"Missing coverage recipe snippets: {missing!r}"


@pytest.mark.parametrize(
    "example_id",
    (
        "DOC_EXAMPLE_001_domain_extension_minimal_type_recipe",
        "DOC_EXAMPLE_002_domain_extension_operation_recipe",
    ),
)
def test_domain_extension_documented_examples_execute(example_id: str) -> None:
    """ID: DOC_EXAMPLE_001_domain_extension_minimal_type_recipe; DOC_EXAMPLE_002_domain_extension_operation_recipe."""
    blocks = _example_blocks()
    assert example_id in blocks
    namespace = {"__name__": f"docs_domain_extension_{example_id}"}
    exec(compile(blocks[example_id], f"{DOMAIN_EXTENSIONS_PATH}:{example_id}", "exec"), namespace)


def test_arch_docs_001_domain_extension_docs_do_not_reference_tal_v2_as_spec() -> None:
    """ID: ARCH_DOCS_001_domain_extension_docs_do_not_reference_tal_v2_as_spec."""
    assert "tal_v2" not in _visible_developer_guide_text()


def test_arch_docs_002_domain_extension_docs_do_not_advertise_missing_apis() -> None:
    """ID: ARCH_DOCS_002_domain_extension_docs_do_not_advertise_missing_apis."""
    text = _visible_developer_guide_text().lower()
    forbidden = ["coming soon", "future api", "not yet available", "todo", "placeholder", "pseudo-code"]
    found = [snippet for snippet in forbidden if snippet in text]
    assert not found, f"Guide advertises missing or placeholder APIs: {found!r}"


def test_arch_docs_003_domain_extension_docs_respect_core_import_boundaries() -> None:
    """ID: ARCH_DOCS_003_domain_extension_docs_respect_core_import_boundaries."""
    text = _visible_domain_extension_text()
    assert "`tal.core` must not import domain packages" in text
    assert "domain packages may\nimport `tal.core`" in text


def test_domain_extension_visible_docs_avoid_internal_references() -> None:
    text = _visible_developer_guide_text()
    forbidden = [
        "AGENTS.md",
        "contracts/",
        "tal_v2",
        "tests/",
        "DOCS_DEV_",
        "DOC_EXAMPLE_",
        "ARCH_DOCS_",
    ]
    found = [snippet for snippet in forbidden if snippet in text]
    assert not found, f"Visible developer docs expose internal references: {found!r}"


def test_domain_extension_python_blocks_are_annotated_and_self_contained() -> None:
    text = _domain_extension_text()
    blocks = _example_blocks()
    assert len(PYTHON_BLOCK_RE.findall(text)) == len(blocks)
    assert set(blocks) == {
        "DOC_EXAMPLE_001_domain_extension_minimal_type_recipe",
        "DOC_EXAMPLE_002_domain_extension_operation_recipe",
    }


def test_domain_extension_python_comments_do_not_reference_internal_docs() -> None:
    forbidden = ("AGENTS", "contract", "contracts/", "tests/", "tal_v2", "DOC_EXAMPLE", "ARCH_DOCS")
    failures: list[str] = []
    for example_id, code in _example_blocks().items():
        for line_number, line in enumerate(code.splitlines(), start=1):
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                continue
            if any(snippet in stripped for snippet in forbidden):
                failures.append(f"{example_id}:{line_number}: {stripped}")
    assert not failures, "\n".join(failures)
