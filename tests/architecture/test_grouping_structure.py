from __future__ import annotations

import ast
from pathlib import Path

from ._budget import executable_source, file_loc, function_lengths


def _method_has_named_call(path: Path, *, class_name: str, method_name: str, callee: str) -> bool:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != method_name:
                continue
            for call in (n for n in ast.walk(item) if isinstance(n, ast.Call)):
                if isinstance(call.func, ast.Name) and call.func.id == callee:
                    return True
                if isinstance(call.func, ast.Attribute) and call.func.attr == callee:
                    return True
            return False
    return False


def test_arch_group_p9a_001_grouping_foundation_reuses_core_orchestration_owners() -> None:
    """ID: ARCH_GROUP_P9A_001_grouping_foundation_reuses_core_orchestration_owners."""
    foundation_text = Path("tal/core/group_ops/foundation.py").read_text(encoding="utf-8")
    key_text = Path("tal/core/group_ops/key_resolve.py").read_text(encoding="utf-8")
    assert "coerce_analysis_object_input" in foundation_text
    assert "resolve_dataset_contexts" in foundation_text
    assert "select_topology_policy_with_intents" in key_text
    assert "resolve_binary_topology" in key_text
    assert "align_exact_for_plan" in key_text
    assert "operation_intent_support_for_operation_family" in key_text
    assert "semantic_policy=STRICT_EXACT_POLICY" in key_text
    assert "xr.align(" not in key_text


def test_arch_group_p9a_002_no_duplicate_grouping_policy_helpers_outside_foundation_owner() -> None:
    """ID: ARCH_GROUP_P9A_002_no_duplicate_grouping_policy_helpers_outside_foundation_owner."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        rel = path.as_posix()
        if rel.startswith("tal/core/group_ops/"):
            continue
        text = path.read_text(encoding="utf-8")
        assert "def resolve_grouping_foundation_context(" not in text
        assert "def normalize_grouping_key_input(" not in text
        assert "GroupingFoundationOptions" not in text


def test_arch_group_p9a_003_grouping_na_policy_scalar_checks_use_shared_lazy_guard_owner() -> None:
    """ID: ARCH_GROUP_P9A_003_grouping_na_policy_scalar_checks_use_shared_lazy_guard_owner."""
    foundation_text = Path("tal/core/group_ops/foundation.py").read_text(encoding="utf-8")
    assert "require_unchunked_dataarray" in foundation_text
    assert "bool(combined_mask.any())" not in foundation_text
    assert "bool(collision.any())" not in foundation_text


def test_arch_group_p9a_004_bin_spec_labels_uniqueness_enforced_in_validation_owner() -> None:
    """ID: ARCH_GROUP_P9A_004_bin_spec_labels_uniqueness_enforced_in_validation_owner."""
    options_text = Path("tal/core/group_ops/options.py").read_text(encoding="utf-8")
    assert "def _first_duplicate_label(" in options_text
    assert "canonical_group_label_key(" in options_text
    assert "bin spec labels must be unique" in options_text


def test_arch_group_p9a_007_unhashable_bin_labels_fail_closed_in_validation_owner() -> None:
    """ID: ARCH_GROUP_P9A_007_unhashable_bin_labels_fail_closed_in_validation_owner."""
    options_text = Path("tal/core/group_ops/options.py").read_text(encoding="utf-8")
    assert "def _first_unhashable_label(" in options_text
    assert "bin spec labels must be hashable" in options_text


def test_arch_group_p9a_005_grouping_key_alignment_probe_is_metadata_only_not_dense() -> None:
    """ID: ARCH_GROUP_P9A_005_grouping_key_alignment_probe_is_metadata_only_not_dense."""
    key_text = Path("tal/core/group_ops/key_resolve.py").read_text(encoding="utf-8")
    assert "def _reference_alignment_array(" in key_text
    assert "np.broadcast_to(" in key_text
    assert "np.zeros(" not in key_text


def test_arch_group_p9a_006_row_dim_compatibility_policy_has_single_shared_owner() -> None:
    """ID: ARCH_GROUP_P9A_006_row_dim_compatibility_policy_has_single_shared_owner."""
    helper_text = Path("tal/core/group_ops/row_dim_compat.py").read_text(encoding="utf-8")
    key_text = Path("tal/core/group_ops/key_resolve.py").read_text(encoding="utf-8")
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    assert "def require_row_dim_compatibility(" in helper_text
    assert "from .row_dim_compat import require_row_dim_compatibility" in key_text
    assert "from .row_dim_compat import require_row_dim_compatibility" in runtime_text
    assert "def _require_row_dim_compatibility(" not in key_text
    assert "def _require_row_dim_compatibility(" not in runtime_text


def test_group_hard_p9a_003_grouping_foundation_remains_datatree_agnostic_in_core() -> None:
    """ID: GROUP_HARD_P9A_003_grouping_foundation_remains_datatree_agnostic_in_core."""
    for path in sorted(Path("tal/core/group_ops").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "DataTree" not in text
        assert "tal.catalog" not in text
        assert "catalog" not in text


def test_grouping_owner_budget_and_schema_write_boundary() -> None:
    """Grouping owners stay within AGENTS budgets and avoid direct tal-attrs writes."""
    for path in sorted(Path("tal/core/group_ops").glob("*.py")):
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget ({length} > 50)."
        text = path.read_text(encoding="utf-8")
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text


def test_arch_group_p9b_001_grouped_surface_wrappers_are_thin_and_delegate_to_foundation_owners() -> None:
    """ID: ARCH_GROUP_P9B_001_grouped_surface_wrappers_are_thin_and_delegate_to_foundation_owners."""
    accessor_text = executable_source(path="tal/core/group_ops/accessor.py")
    assert "class GroupAccessor" in accessor_text
    assert "resolve_grouping_foundation_context(" in accessor_text
    assert "resolve_grouped_runtime_plan(" in accessor_text
    assert "materialize_grouped_view(" in accessor_text
    assert ".mean(" not in accessor_text
    assert ".sum(" not in accessor_text


def test_arch_group_p9b_002_no_inline_grouping_policy_duplication_in_surface_modules() -> None:
    """ID: ARCH_GROUP_P9B_002_no_inline_grouping_policy_duplication_in_surface_modules."""
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    materialize_text = Path("tal/core/group_ops/materialize.py").read_text(encoding="utf-8")
    assert "resolve_grouping_foundation_context(" not in runtime_text
    assert "resolve_binary_topology(" not in materialize_text
    assert "align_exact_for_plan(" not in materialize_text
    assert "xr.align(" not in runtime_text
    assert "xr.align(" not in materialize_text


def test_arch_group_p9b_003_preserve_batch_padded_materialization_uses_global_group_axis() -> None:
    """ID: ARCH_GROUP_P9B_003_preserve_batch_padded_materialization_uses_global_group_axis."""
    text = Path("tal/core/group_ops/materialize.py").read_text(encoding="utf-8")
    assert "batch_dims=plan.foundation.batch_dims + (group_dim,)" in text
    assert "_dense_preserve_padded_rows(" in text
    assert "coords[group_dim] = _group_labels_coord(labels, group_dim=group_dim)" in text


def test_arch_group_p9b_004_group_accessor_is_canonical_analysisobject_surface() -> None:
    """ID: ARCH_GROUP_P9B_004_group_accessor_is_canonical_analysisobject_surface."""
    ao_text = Path("tal/core/analysis_object.py").read_text(encoding="utf-8")
    assert "def group(self)" in ao_text
    assert "def groupby(" not in ao_text
    assert "def groupby_bins(" not in ao_text


def test_arch_group_p9b_005_padded_sequence_coord_is_rank_derived_not_source_coord_reuse() -> None:
    """ID: ARCH_GROUP_P9B_005_padded_sequence_coord_is_rank_derived_not_source_coord_reuse."""
    text = Path("tal/core/group_ops/materialize.py").read_text(encoding="utf-8")
    assert "def _output_sequence_coord(*, sequence_dim: str, size: int)" in text
    assert "np.arange(size" in text
    assert "sequence_dim in ds.coords" not in text


def test_arch_group_p9b_006_bin_ordering_uses_bin_domain_metadata_not_float_sort_fallback() -> None:
    """ID: ARCH_GROUP_P9B_006_bin_ordering_uses_bin_domain_metadata_not_float_sort_fallback."""
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    key_text = Path("tal/core/group_ops/key_resolve.py").read_text(encoding="utf-8")
    assert "domain_order = keys[0].domain_order" in runtime_text
    assert "key=lambda value: float(value)" not in runtime_text
    assert "domain_order=domain_order" in key_text


def test_arch_group_p9b_007_include_empty_groups_is_consumed_by_materialization_owner() -> None:
    """ID: ARCH_GROUP_P9B_007_include_empty_groups_is_consumed_by_materialization_owner."""
    materialize_text = Path("tal/core/group_ops/materialize.py").read_text(encoding="utf-8")
    assert "include_empty_groups" in materialize_text
    assert "_effective_global_padded_groups(" in materialize_text
    assert "_effective_batch_padded_groups(" in materialize_text


def test_arch_group_p9b_008_bin_ordering_appends_observed_non_domain_labels_after_domain_order() -> None:
    """ID: ARCH_GROUP_P9B_008_bin_ordering_appends_observed_non_domain_labels_after_domain_order."""
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    assert "ordered_domain = tuple(" in runtime_text
    assert "for label in domain_order" in runtime_text
    assert "extras = tuple(" in runtime_text
    assert "for label in labels" in runtime_text
    assert "return ordered_domain + extras" in runtime_text


def test_arch_group_p9b_009_groupby_accessor_defers_runtime_plan_resolution() -> None:
    """ID: ARCH_GROUP_P9B_009_groupby_accessor_defers_runtime_plan_resolution."""
    path = Path("tal/core/group_ops/accessor.py")
    assert not _method_has_named_call(
        path,
        class_name="GroupAccessor",
        method_name="groupby",
        callee="resolve_grouped_runtime_plan",
    )
    assert _method_has_named_call(
        path,
        class_name="GroupedView",
        method_name="_resolve_plan",
        callee="resolve_grouped_runtime_plan",
    )


def test_arch_group_p9b_010_runtime_plan_materialization_isolated_to_single_boundary_helper() -> None:
    """ID: ARCH_GROUP_P9B_010_runtime_plan_materialization_isolated_to_single_boundary_helper."""
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    assert "def _realize_row_values(" in runtime_text
    assert "np.asarray(data.transpose(*row_dims).data)" in runtime_text
    assert "return np.asarray(data.transpose(*row_dims).data)" in runtime_text


def test_arch_group_p9b_011_runtime_plan_rejects_duplicate_domain_order_labels() -> None:
    """ID: ARCH_GROUP_P9B_011_runtime_plan_rejects_duplicate_domain_order_labels."""
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    assert "def _first_duplicate_domain_label(" in runtime_text
    assert "canonical_group_label_key(" in runtime_text
    assert "domain_order contains duplicate label" in runtime_text


def test_arch_group_p9b_012_materialize_uses_centralized_layout_name_collision_validator() -> None:
    """ID: ARCH_GROUP_P9B_012_materialize_uses_centralized_layout_name_collision_validator."""
    options_text = Path("tal/core/group_ops/grouped_options.py").read_text(encoding="utf-8")
    materialize_text = Path("tal/core/group_ops/materialize.py").read_text(encoding="utf-8")
    assert "def validate_layout_name_collisions(" in options_text
    assert "validate_layout_name_collisions(" in materialize_text
    assert "plan.foundation.sequence_dim in {group_dim, member_dim}" not in materialize_text


def test_arch_group_p9b_013_runtime_owner_reuse_probe_avoids_dense_reference_allocation() -> None:
    """ID: ARCH_GROUP_P9B_013_runtime_owner_reuse_probe_avoids_dense_reference_allocation."""
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    assert "def _reference_alignment_array(context: GroupingFoundationContext, *, owner: str)" in runtime_text
    assert ".isel(indexers).rename(\"__group_plan_ref__\")" in runtime_text
    assert "np.zeros(" not in runtime_text


def test_arch_group_p9b_014_runtime_plan_row_dim_validation_is_name_based_not_order_strict() -> None:
    """ID: ARCH_GROUP_P9B_014_runtime_plan_row_dim_validation_is_name_based_not_order_strict."""
    helper_text = Path("tal/core/group_ops/row_dim_compat.py").read_text(encoding="utf-8")
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    assert "set(dims) != set(row_dims)" in helper_text
    assert "tuple(data.dims) != row_dims" not in helper_text
    assert "require_row_dim_compatibility(" in runtime_text


def test_arch_group_p9b_015_runtime_group_label_hashability_guard_prevents_raw_typeerror_leakage() -> None:
    """ID: ARCH_GROUP_P9B_015_runtime_group_label_hashability_guard_prevents_raw_typeerror_leakage."""
    runtime_text = Path("tal/core/group_ops/runtime_plan.py").read_text(encoding="utf-8")
    assert "def _require_hashable_group_label(" in runtime_text
    assert "produced unhashable group label" in runtime_text


def test_arch_group_p9d_001_grouped_reducer_surface_is_installed_via_single_owner() -> None:
    """ID: ARCH_GROUP_P9D_001_grouped_reducer_surface_is_installed_via_single_owner."""
    accessor_text = Path("tal/core/group_ops/accessor.py").read_text(encoding="utf-8")
    surface_text = Path("tal/core/group_ops/reducer_surface.py").read_text(encoding="utf-8")
    assert "install_grouped_view_reducers(GroupedView)" in accessor_text
    assert "def install_grouped_view_reducers(" in surface_text
    assert "def _dispatch(" in surface_text
    assert "grouped_reduce(" in surface_text
    assert "xr.apply_ufunc(" not in surface_text


def test_arch_group_p9d_002_grouped_reducer_dispatcher_reuses_finalize_policy_and_materialize_owner() -> None:
    """ID: ARCH_GROUP_P9D_002_grouped_reducer_dispatcher_reuses_finalize_policy_and_materialize_owner."""
    text = Path("tal/core/group_ops/reducer_dispatch.py").read_text(encoding="utf-8")
    assert "resolve_reducer_finalize_source" in text
    assert "materialize_grouped_view(" in text
    assert "source_ao=source_for_finalize" in text
    assert "require_supported_op(" in text
    assert "xr.apply_ufunc(" not in text
    assert "np.linalg" not in text


def test_arch_group_p9d_003_grouped_reducer_dispatcher_uses_canonical_padded_compute_path() -> None:
    """ID: ARCH_GROUP_P9D_003_grouped_reducer_dispatcher_uses_canonical_padded_compute_path."""
    text = Path("tal/core/group_ops/reducer_dispatch.py").read_text(encoding="utf-8")
    assert "replace(materialize_opts, layout=\"padded\")" in text
    assert "def _map_dim_for_canonical_padded(" in text
    assert "requested_layout=materialize_opts.layout" in text
