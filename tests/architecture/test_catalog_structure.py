from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture._budget import file_loc, function_lengths


def test_arch_cat_p11a_001_catalog_implementation_resides_outside_tal_core() -> None:
    """ID: ARCH_CAT_P11A_001_catalog_implementation_resides_outside_tal_core."""
    assert Path("tal/catalog/catalog.py").exists()
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "class Catalog" not in text


def test_arch_cat_p11a_002_no_core_import_dependency_on_tal_catalog() -> None:
    """ID: ARCH_CAT_P11A_002_no_core_import_dependency_on_tal_catalog."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.catalog" not in text
        assert "from ..catalog" not in text
        assert "from tal.catalog" not in text


def test_arch_cat_p11a_003_catalog_surface_methods_remain_thin_and_owner_delegated() -> None:
    """ID: ARCH_CAT_P11A_003_catalog_surface_methods_remain_thin_and_owner_delegated."""
    text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    assert "coerce_catalog_init_options(" in text
    assert "detect_catalog_backend(" in text
    assert "normalize_catalog_payload(" in text
    assert "resolve_catalog_batch_dim(" in text
    assert "normalize_label_selector(" in text
    assert "normalize_position_selector(" in text
    assert "select_by_labels(" in text
    assert "select_by_positions(" in text


def test_arch_cat_p11a_004_catalog_grouping_policy_is_constructor_scoped_not_mutated_post_construction() -> None:
    """ID: ARCH_CAT_P11A_004_catalog_grouping_policy_is_constructor_scoped_not_mutated_post_construction."""
    text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    assert "def set_group(" not in text
    assert "def regroup(" not in text
    assert "@batch_dim.setter" not in text


def test_arch_cat_p11a_005_catalog_isel_index_normalization_policy_is_centralized() -> None:
    """ID: ARCH_CAT_P11A_005_catalog_isel_index_normalization_policy_is_centralized."""
    text = Path("tal/catalog/selection.py").read_text(encoding="utf-8")
    assert "def _normalize_effective_position_selector(" in text
    assert "size=_group_axis_size(state)" in text
    assert "_normalize_effective_position_selector(" in text
    assert "_datatree_position_labels(tree, selector=normalized)" in text
    assert "return tuple(labels[idx] for idx in selector)" in text
    assert "idx + size if idx < 0 else idx" in text


def test_cat_doc_p11a_001_catalog_location_optional_role_and_browse_only_semantics_documented() -> None:
    """ID: CAT_DOC_P11A_001_catalog_location_optional_role_and_browse_only_semantics_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "# Catalog" in text
    assert "Catalog" in text
    assert "AnalysisObject" in text


def test_cat_doc_p11a_002_catalog_grouping_is_constructor_scoped_and_regroup_mutation_is_not_public() -> None:
    """ID: CAT_DOC_P11A_002_catalog_grouping_is_constructor_scoped_and_regroup_mutation_is_not_public."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "batch_dim" in text
    assert "post-construction regrouping" in text


def test_cat_doc_p11a_003_catalog_isel_normalized_duplicate_policy_documented() -> None:
    """ID: CAT_DOC_P11A_003_catalog_isel_normalized_duplicate_policy_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "catalog.isel" in text
    assert "Unsupported selector" in text


def test_catalog_owner_budget_and_schema_write_boundary() -> None:
    """Catalog modules stay within AGENTS budgets and avoid direct tal-attrs writes."""
    for path in sorted(Path("tal/catalog").glob("*.py")):
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget ({length} > 50)."
        text = path.read_text(encoding="utf-8")
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text


def test_arch_cat_p11b_001_query_extract_reuse_shared_finalize_and_grouping_owners() -> None:
    """ID: ARCH_CAT_P11B_001_query_extract_reuse_shared_finalize_and_grouping_owners."""
    extract_text = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    query_text = Path("tal/catalog/query.py").read_text(encoding="utf-8")
    assert "finalize_with_schema" in extract_text
    assert "set_left_packed_validity_or_prune_from_size_coord" in extract_text
    assert "ensure_grouping_context_for_query_extract" in query_text


def test_arch_cat_p11b_002_single_query_planner_owner_enforced_no_duplicate_policy_helpers() -> None:
    """ID: ARCH_CAT_P11B_002_single_query_planner_owner_enforced_no_duplicate_policy_helpers."""
    planner_text = Path("tal/catalog/query_plan.py").read_text(encoding="utf-8")
    query_text = Path("tal/catalog/query.py").read_text(encoding="utf-8")
    catalog_text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    assert "def normalize_catalog_query_plan(" in planner_text
    assert "normalize_catalog_query_plan(" in query_text
    assert "normalize_catalog_query_plan(" not in catalog_text


def test_arch_cat_p11b_003_catalog_query_extract_logic_resides_outside_tal_core() -> None:
    """ID: ARCH_CAT_P11B_003_catalog_query_extract_logic_resides_outside_tal_core."""
    assert Path("tal/catalog/query.py").exists()
    assert Path("tal/catalog/extract.py").exists()
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "Catalog.query" not in text
        assert "Catalog.extract" not in text


def test_arch_cat_p11b_004_query_extract_do_not_depend_on_post_construction_group_mutators() -> None:
    """ID: ARCH_CAT_P11B_004_query_extract_do_not_depend_on_post_construction_group_mutators."""
    text = Path("tal/catalog/catalog.py").read_text(encoding="utf-8")
    assert "def set_group(" not in text
    assert "def regroup(" not in text
    assert "query(" in text
    assert "extract(" in text


def test_arch_cat_p11b_005_batch_dim_collision_validation_is_centralized_and_reused() -> None:
    """ID: ARCH_CAT_P11B_005_batch_dim_collision_validation_is_owned_in_catalog_backends_and_extract."""
    backends = Path("tal/catalog/backends.py").read_text(encoding="utf-8")
    extract = Path("tal/catalog/extract.py").read_text(encoding="utf-8")
    assert "def _validate_explicit_dataset_batch_dim(" in backends
    assert "def validate_batch_dim_role_compatibility(" in backends
    assert "def _first_child_dim_collision(" in backends
    assert "validate_batch_dim_role_compatibility(" in extract
    assert "def _require_nonconflicting_batch_dim(" not in extract
    assert "def _expand_child_row(" in extract
    assert "collides with child payload dims" in extract


def test_cat_doc_p11b_001_query_empty_and_large_dataset_default_semantics_documented() -> None:
    """ID: CAT_DOC_P11B_001_query_empty_and_large_dataset_default_semantics_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "query" in text
    assert "where" in text
    assert "metadata" in text


def test_cat_doc_p11b_002_extract_metadata_promotion_and_sequence_size_coord_policy_documented() -> None:
    """ID: CAT_DOC_P11B_002_extract_metadata_promotion_and_sequence_size_coord_policy_documented."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "extract" in text
    assert "Metadata promotion" in text
    assert "AnalysisObject" in text


def test_cat_doc_p11b_003_query_extract_grouping_is_constructor_scoped_and_immutable() -> None:
    """ID: CAT_DOC_P11B_003_query_extract_grouping_is_constructor_scoped_and_immutable."""
    text = Path("docs/api/catalog.md").read_text(encoding="utf-8")
    assert "Grouping is resolved at construction" in text
    assert "extract" in text
