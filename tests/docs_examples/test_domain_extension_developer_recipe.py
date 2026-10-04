from __future__ import annotations

import doctest
import io
import inspect
from pathlib import Path
import re

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVELOPER_GUIDE_DIR = REPO_ROOT / "docs" / "developer-guide"
DOMAIN_EXTENSIONS_PATH = DEVELOPER_GUIDE_DIR / "domain_extensions.md"
DEVELOPER_INDEX_PATH = DEVELOPER_GUIDE_DIR / "index.md"
DOCS_INDEX_PATH = REPO_ROOT / "docs" / "index.md"
API_INDEX_PATH = REPO_ROOT / "docs" / "api" / "index.md"
API_ANALYSIS_OBJECT_PATH = REPO_ROOT / "docs" / "api" / "analysis-object.md"
API_SCHEMA_PATH = REPO_ROOT / "docs" / "api" / "schema.md"
API_DOMAIN_EXTENSIONS_PATH = REPO_ROOT / "docs" / "api" / "domain-extensions.md"
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
        HTML_COMMENT_RE.sub("", API_DOMAIN_EXTENSIONS_PATH.read_text(encoding="utf-8")),
    ]
    return "\n".join(parts)


def _example_blocks() -> dict[str, str]:
    return {
        match.group("example_id"): match.group("code")
        for match in EXAMPLE_BLOCK_RE.finditer(_domain_extension_text())
    }


def _referenced_domain_extension_api_objects() -> dict[str, object]:
    from tal.core import AnalysisObject
    from tal.core.orchestration.context import (
        DatasetContext,
        DatasetContextOptions,
        resolve_dataset_context,
        resolve_dataset_contexts,
    )
    from tal.core.orchestration.finalize import finalize_like
    from tal.core.orchestration.inputs import coerce_analysis_object_input, coerce_operand
    from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec, finalize_with_schema
    from tal.core.schema import merge_schema
    from tal.core.schema_read import read_roles
    from tal.core.typed_lifecycle import (
        TypedAnalysisObject,
        TypedLifecycleContext,
        TypedLifecycleSpec,
        default_coerce_source,
        identity_init_options,
        identity_normalize,
        no_op_enforce,
    )

    return {
        "AnalysisObject": AnalysisObject,
        "AnalysisObject.from_data": AnalysisObject.from_data,
        "AnalysisObject.merge_schema": AnalysisObject.merge_schema,
        "merge_schema": merge_schema,
        "read_roles": read_roles,
        "TypedLifecycleContext": TypedLifecycleContext,
        "TypedLifecycleSpec": TypedLifecycleSpec,
        "TypedAnalysisObject": TypedAnalysisObject,
        "TypedAnalysisObject.__init__": TypedAnalysisObject.__init__,
        "TypedAnalysisObject._from_validated": TypedAnalysisObject._from_validated,
        "TypedAnalysisObject._from_unvalidated": TypedAnalysisObject._from_unvalidated,
        "default_coerce_source": default_coerce_source,
        "identity_init_options": identity_init_options,
        "identity_normalize": identity_normalize,
        "no_op_enforce": no_op_enforce,
        "coerce_analysis_object_input": coerce_analysis_object_input,
        "coerce_operand": coerce_operand,
        "DatasetContextOptions": DatasetContextOptions,
        "DatasetContext": DatasetContext,
        "resolve_dataset_context": resolve_dataset_context,
        "resolve_dataset_contexts": resolve_dataset_contexts,
        "finalize_like": finalize_like,
        "CoreSchemaFinalizeSpec": CoreSchemaFinalizeSpec,
        "finalize_with_schema": finalize_with_schema,
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


def test_docs_dev_006_domain_extension_referenced_apis_are_documented() -> None:
    """ID: DOCS_DEV_006_domain_extension_referenced_apis_are_documented."""
    assert API_DOMAIN_EXTENSIONS_PATH.exists()
    assert "domain-extensions" in API_INDEX_PATH.read_text(encoding="utf-8")
    assert "{doc}`../api/domain-extensions`" in _visible_domain_extension_text()

    api_text = API_DOMAIN_EXTENSIONS_PATH.read_text(encoding="utf-8")
    analysis_object_text = API_ANALYSIS_OBJECT_PATH.read_text(encoding="utf-8")
    schema_text = API_SCHEMA_PATH.read_text(encoding="utf-8")
    required_by_file = {
        "domain-extensions.md": [
            "tal.core.typed_lifecycle.TypedAnalysisObject",
            "tal.core.typed_lifecycle.TypedLifecycleContext",
            "tal.core.typed_lifecycle.TypedLifecycleSpec",
            "tal.core.typed_lifecycle.default_coerce_source",
            "tal.core.typed_lifecycle.identity_init_options",
            "tal.core.typed_lifecycle.identity_normalize",
            "tal.core.typed_lifecycle.no_op_enforce",
            "tal.core.typed_lifecycle.TypedAnalysisObject._from_validated",
            "tal.core.typed_lifecycle.TypedAnalysisObject._from_unvalidated",
            "tal.core.orchestration.inputs.coerce_operand",
            "tal.core.orchestration.inputs.coerce_analysis_object_input",
            "tal.core.orchestration.context.DatasetContextOptions",
            "tal.core.orchestration.context.DatasetContext",
            "tal.core.orchestration.context.resolve_dataset_context",
            "tal.core.orchestration.finalize.finalize_like",
            "tal.core.orchestration.schema_finalize.CoreSchemaFinalizeSpec",
            "tal.core.orchestration.schema_finalize.finalize_with_schema",
            "tal.core.schema_read.read_roles",
        ],
        "analysis-object.md": [
            "tal.AnalysisObject",
            "tal.AnalysisObject.from_data",
            "tal.AnalysisObject.merge_schema",
        ],
        "schema.md": [
            "tal.core.merge_schema",
        ],
    }
    texts_by_file = {
        "domain-extensions.md": api_text,
        "analysis-object.md": analysis_object_text,
        "schema.md": schema_text,
    }
    missing = {
        name: [snippet for snippet in snippets if snippet not in texts_by_file[name]]
        for name, snippets in required_by_file.items()
    }
    missing = {name: snippets for name, snippets in missing.items() if snippets}
    assert not missing, f"Domain extension guide references APIs without docs entries: {missing!r}"


def test_docs_dev_007_domain_extension_referenced_apis_have_docstrings() -> None:
    """ID: DOCS_DEV_007_domain_extension_referenced_apis_have_docstrings."""
    required = {
        "AnalysisObject": ("Notes", "See Also", "Examples"),
        "AnalysisObject.from_data": ("Parameters", "Returns", "Notes", "Examples"),
        "AnalysisObject.merge_schema": ("Parameters", "Returns", "Examples", "See Also"),
        "merge_schema": ("Parameters", "Returns", "Notes", "Examples"),
        "read_roles": ("Parameters", "Returns", "Raises", "Notes", "Examples"),
        "TypedLifecycleContext": ("Parameters", "Notes", "Examples"),
        "TypedLifecycleSpec": ("Parameters", "Raises", "Notes", "Examples"),
        "TypedAnalysisObject": ("Parameters", "Notes", "Examples"),
        "TypedAnalysisObject.__init__": ("Parameters", "Raises", "Examples"),
        "TypedAnalysisObject._from_validated": ("Parameters", "Returns", "Notes", "Examples"),
        "TypedAnalysisObject._from_unvalidated": ("Parameters", "Returns", "Notes", "Examples"),
        "default_coerce_source": ("Parameters", "Returns", "Raises", "See Also", "Examples"),
        "identity_init_options": ("Parameters", "Returns", "Notes", "Examples"),
        "identity_normalize": ("Parameters", "Returns", "Notes", "Examples"),
        "no_op_enforce": ("Parameters", "Returns", "Notes", "Examples"),
        "coerce_analysis_object_input": ("Parameters", "Returns", "Raises", "Notes", "Examples"),
        "coerce_operand": ("Parameters", "Returns", "Raises", "Notes", "Examples"),
        "DatasetContextOptions": ("Parameters", "Notes", "Examples"),
        "DatasetContext": ("Parameters", "Notes", "Examples"),
        "resolve_dataset_context": ("Parameters", "Returns", "Raises", "Notes", "Examples"),
        "resolve_dataset_contexts": ("Parameters", "Returns", "Raises", "Notes", "Examples"),
        "finalize_like": ("Parameters", "Returns", "Notes", "Examples"),
        "CoreSchemaFinalizeSpec": ("Parameters", "Notes", "Examples"),
        "finalize_with_schema": ("Parameters", "Returns", "Notes", "Examples"),
    }
    objects = _referenced_domain_extension_api_objects()
    failures: dict[str, list[str]] = {}
    for name, snippets in required.items():
        obj = objects[name]
        doc = inspect.getdoc(obj) or ""
        missing = [snippet for snippet in snippets if snippet not in doc]
        if missing or not doc:
            failures[name] = missing or ["docstring"]
    assert not failures, f"Referenced domain-extension APIs lack complete docstrings: {failures!r}"


def test_docs_dev_008_domain_extension_referenced_api_docstring_examples_execute() -> None:
    """ID: DOCS_DEV_008_domain_extension_referenced_api_docstring_examples_execute."""
    parser = doctest.DocTestParser()
    failures: dict[str, str] = {}
    for name, obj in _referenced_domain_extension_api_objects().items():
        doc = inspect.getdoc(obj) or ""
        source = inspect.getsourcefile(obj) or name
        test = parser.get_doctest(doc, {}, name, source, 0)
        if not test.examples:
            failures[name] = "docstring has no executable doctest examples"
            continue
        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
        output = io.StringIO()
        result = runner.run(test, out=output.write)
        if result.failed:
            failures[name] = output.getvalue()
    assert not failures, f"Domain-extension API docstring examples failed: {failures!r}"


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
    forbidden = ("AGENTS", "contract", "contracts/", "tests/", "DOC_EXAMPLE", "ARCH_DOCS")
    failures: list[str] = []
    for example_id, code in _example_blocks().items():
        for line_number, line in enumerate(code.splitlines(), start=1):
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                continue
            if any(snippet in stripped for snippet in forbidden):
                failures.append(f"{example_id}:{line_number}: {stripped}")
    assert not failures, "\n".join(failures)
