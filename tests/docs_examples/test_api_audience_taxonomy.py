from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "docs" / "api"
API_INDEX_PATH = API_DIR / "index.md"
API_EXTENSION_AUTHOR_PATH = API_DIR / "domain-extensions.md"
API_INTERNAL_PATH = API_DIR / "internal-api.md"

USER_API_PAGES = (
    "analysis-object.md",
    "schema.md",
    "components.md",
    "concat.md",
    "timebase.md",
    "events.md",
    "reducers.md",
    "frames.md",
    "io.md",
    "viz.md",
    "ufuncs.md",
    "numba.md",
    "types/index.md",
    "types/array.md",
    "types/vector.md",
    "types/vector3.md",
    "types/matrix.md",
    "types/position.md",
    "types/rotation.md",
    "types/pose.md",
    "types/velocity.md",
    "types/acceleration.md",
    "types/path_solve.md",
    "types/covariance.md",
)

USER_FORBIDDEN_AUTOSUMMARY_SYMBOLS = (
    "tal.core.typed_lifecycle.TypedAnalysisObject",
    "tal.core.typed_lifecycle.TypedLifecycleSpec",
    "tal.core.orchestration.context.resolve_dataset_context",
    "tal.core.orchestration.finalize.finalize_like",
    "tal.core.orchestration.schema_finalize.finalize_with_schema",
)

EXTENSION_AUTHOR_SYMBOLS = (
    "tal.core.typed_lifecycle.TypedAnalysisObject",
    "tal.core.typed_lifecycle.TypedLifecycleSpec",
    "tal.core.typed_lifecycle.TypedLifecycleContext",
    "tal.core.typed_lifecycle.SourceCoercer",
    "tal.core.typed_lifecycle.DatasetHook",
    "tal.core.typed_lifecycle.EnforceHook",
    "tal.core.typed_lifecycle.default_coerce_source",
    "tal.core.typed_lifecycle.identity_init_options",
    "tal.core.typed_lifecycle.identity_normalize",
    "tal.core.typed_lifecycle.no_op_enforce",
    "tal.core.typed_lifecycle.TypedAnalysisObject._from_validated",
    "tal.core.typed_lifecycle.TypedAnalysisObject._from_unvalidated",
    "tal.core.orchestration.inputs.coerce_operand",
    "tal.core.orchestration.inputs.coerce_analysis_object_input",
    "tal.core.orchestration.inputs.normalize_analysis_object_inputs",
    "tal.core.orchestration.context.DatasetContextOptions",
    "tal.core.orchestration.context.DatasetContext",
    "tal.core.orchestration.context.resolve_dataset_context",
    "tal.core.orchestration.context.resolve_dataset_contexts",
    "tal.core.orchestration.resolve.resolve_combine_contexts",
    "tal.core.orchestration.resolve.resolve_param_runtime_context",
    "tal.core.orchestration.topology.SemanticTopology",
    "tal.core.orchestration.topology.TopologyPolicy",
    "tal.core.orchestration.topology.TopologyOperand",
    "tal.core.orchestration.topology.ResolvedTopologyPlan",
    "tal.core.orchestration.topology.resolve_unary_topology",
    "tal.core.orchestration.topology.resolve_binary_topology",
    "tal.core.orchestration.topology.resolve_nary_topology",
    "tal.core.orchestration.topology.realize_operands_for_plan",
    "tal.core.orchestration.topology.flatten_param_contexts",
    "tal.core.orchestration.topology.flatten_query_for_batch_plan",
    "tal.core.orchestration.topology.restore_dataset_batch_topology",
    "tal.core.orchestration.topology.restore_dataset_multi_batch",
    "tal.core.orchestration.finalize.finalize_like",
    "tal.core.orchestration.finalize.finalize_many_like",
    "tal.core.orchestration.finalize.restore_and_finalize",
    "tal.core.orchestration.finalize.rewrap_unvalidated_like",
    "tal.core.orchestration.schema_finalize.CoreSchemaFinalizeSpec",
    "tal.core.orchestration.schema_finalize.finalize_with_schema",
    "tal.core.orchestration.schema_finalize.clear_core_schema_blocks",
    "tal.core.schema_read.read_roles",
    "tal.core.schema_read.read_param_coord_name",
    "tal.core.schema_read.read_sequence_size_coord_name",
    "tal.core.schema_read.validate_schema_if_needed",
    "tal.core.validity_finalize.assign_sequence_size_from_valid_mask",
    "tal.core.validity_finalize.set_left_packed_validity_or_prune_from_size_coord",
    "tal.core.validity_finalize.reconcile_sequence_validity_after_structure",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _toctree_entries_for_caption(text: str, caption: str) -> tuple[str, ...]:
    for match in re.finditer(r"```\{toctree\}\n(?P<body>.*?)\n```", text, re.DOTALL):
        body = match.group("body")
        if f":caption: {caption}" not in body:
            continue
        entries: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            entries.append(stripped)
        return tuple(entries)
    raise AssertionError(f"Missing toctree caption {caption!r}")


def test_api_taxonomy_001_index_has_three_audience_sections() -> None:
    """ID: API_TAXONOMY_001_index_has_three_audience_sections."""
    text = _read(API_INDEX_PATH)
    assert ":caption: User API" in text
    assert ":caption: Extension Author API" in text
    assert ":caption: Internal API Policy" in text
    assert "domain-extensions" not in _toctree_entries_for_caption(text, "User API")
    assert _toctree_entries_for_caption(text, "Extension Author API") == ("domain-extensions",)
    assert _toctree_entries_for_caption(text, "Internal API Policy") == ("internal-api",)


def test_api_taxonomy_002_user_api_pages_have_user_audience_marker() -> None:
    """ID: API_TAXONOMY_002_user_api_pages_have_user_audience_marker."""
    missing = [
        page
        for page in USER_API_PAGES
        if "**Audience:** User API" not in _read(API_DIR / page)
    ]
    assert not missing, f"User API pages missing audience marker: {missing!r}"


def test_api_taxonomy_003_user_api_pages_do_not_autosummarize_extension_helpers() -> None:
    """ID: API_TAXONOMY_003_user_api_pages_do_not_autosummarize_extension_helpers."""
    failures: dict[str, list[str]] = {}
    for page in USER_API_PAGES:
        text = _read(API_DIR / page)
        found = [symbol for symbol in USER_FORBIDDEN_AUTOSUMMARY_SYMBOLS if symbol in text]
        if found:
            failures[page] = found
    assert not failures, f"User API pages advertise extension helper APIs: {failures!r}"


def test_api_taxonomy_004_extension_author_api_lists_supported_helper_surface() -> None:
    """ID: API_TAXONOMY_004_extension_author_api_lists_supported_helper_surface."""
    text = _read(API_EXTENSION_AUTHOR_PATH)
    assert "# Extension Author API" in text
    assert "For package authors building TAL domain extensions" in text
    missing = [symbol for symbol in EXTENSION_AUTHOR_SYMBOLS if symbol not in text]
    assert not missing, f"Extension Author API missing supported symbols: {missing!r}"


def test_api_taxonomy_005_internal_api_policy_documents_private_surface() -> None:
    """ID: API_TAXONOMY_005_internal_api_policy_documents_private_surface."""
    text = _read(API_INTERNAL_PATH)
    required = [
        "**Audience:** Internal API Policy",
        "Undocumented helpers whose names begin with `_` are unsupported",
        "tal.core.typed_lifecycle.TypedAnalysisObject._from_validated",
        "tal.core.typed_lifecycle.TypedAnalysisObject._from_unvalidated",
        "Spatial metadata helpers that are not exported from `tal.spatial` are internal",
        "Internal APIs may change without deprecation",
    ]
    missing = [snippet for snippet in required if snippet not in text]
    assert not missing, f"Internal API policy missing snippets: {missing!r}"
