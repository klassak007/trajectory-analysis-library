from __future__ import annotations

from pathlib import Path

from ._budget import file_loc, function_lengths


def test_orch_arch_001_single_owner_input_coercion() -> None:
    """ID: ORCH_ARCH_001_single_owner_input_coercion."""
    inputs_text = Path("tal/core/orchestration/inputs.py").read_text(encoding="utf-8")
    assert "def coerce_analysis_object_input(" in inputs_text
    assert "def normalize_analysis_object_inputs(" in inputs_text


def test_orch_arch_002_schema_read_only_boundary() -> None:
    """ID: ORCH_ARCH_002_schema_read_only_boundary."""
    resolve_text = Path("tal/core/orchestration/resolve.py").read_text(encoding="utf-8")
    assert "from .context import" in resolve_text
    for rel in [
        "tal/core/param_ops/accessor.py",
        "tal/core/combine_ops/normalize.py",
    ]:
        text = Path(rel).read_text(encoding="utf-8")
        assert "schema_read" not in text, f"schema_read usage should be centralized; found in {rel}"


def test_orch_arch_003_no_param_combine_local_input_coercion_helpers() -> None:
    """ID: ORCH_ARCH_003_no_param_combine_local_input_coercion_helpers."""
    checks = {
        "tal/core/param_ops/sync.py": ("def _coerce_sync_input(", "def _normalize_sync_inputs("),
        "tal/core/param_ops/interp_like.py": ("def _other_dataset(", "def _grid_from_other("),
        "tal/core/combine_ops/normalize.py": ("def _coerce_input(",),
    }
    for rel, needles in checks.items():
        text = Path(rel).read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text, f"local coercion helper {needle!r} remains in {rel}"


def test_orch_arch_004_no_param_combine_direct_schema_attrs_tal_parsing() -> None:
    """ID: ORCH_ARCH_004_no_param_combine_direct_schema_attrs_tal_parsing."""
    for rel in [
        "tal/core/param_ops/accessor.py",
        "tal/core/param_ops/sync.py",
        "tal/core/param_ops/interp_like.py",
        "tal/core/combine_ops/normalize.py",
        "tal/core/combine_ops/accessor.py",
    ]:
        text = Path(rel).read_text(encoding="utf-8")
        assert 'attrs["tal"]' not in text, f"direct tal attrs parsing found in {rel}"


def test_orch_arch_005_dataset_context_owner_single_source_of_truth() -> None:
    """ID: ORCH_ARCH_005_dataset_context_owner_single_source_of_truth."""
    context_text = Path("tal/core/orchestration/context.py").read_text(encoding="utf-8")
    linalg_plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    vector3_text = Path("tal/linalg/vector3.py").read_text(encoding="utf-8")
    assert "class DatasetContext:" in context_text
    assert "def resolve_dataset_context(" in context_text
    assert "def _build_operand_context(" not in linalg_plan_text
    assert "def _build_reference_spec(" not in vector3_text


def test_orch_arch_006_context_resolution_reused_by_linalg_and_combine() -> None:
    """ID: ORCH_ARCH_006_context_resolution_reused_by_linalg_and_combine."""
    plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    resolve_text = Path("tal/core/orchestration/resolve.py").read_text(encoding="utf-8")
    component_text = Path("tal/linalg/component_context.py").read_text(encoding="utf-8")
    assert "resolve_dataset_context(" in plan_text
    assert "resolve_dataset_contexts(" in resolve_text
    assert "resolve_dataset_context(" in component_text


def test_orch_arch_007_operand_coercion_single_owner_reused() -> None:
    """ID: ORCH_ARCH_007_operand_coercion_single_owner_reused."""
    inputs_text = Path("tal/core/orchestration/inputs.py").read_text(encoding="utf-8")
    binops_text = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    unary_text = Path("tal/linalg/ops/unary.py").read_text(encoding="utf-8")
    vector3_text = Path("tal/linalg/vector3.py").read_text(encoding="utf-8")
    assert "def coerce_operand(" in inputs_text
    assert "coerce_operand(" in binops_text
    assert "coerce_operand(" in unary_text
    assert "def _coerce_operand(" not in binops_text
    assert "def _coerce_component(" not in vector3_text


def test_orch_arch_008_validity_finalize_owner_single_source() -> None:
    """ID: ORCH_ARCH_008_validity_finalize_owner_single_source."""
    validity_text = Path("tal/core/validity_finalize.py").read_text(encoding="utf-8")
    combine_finalize = Path("tal/core/combine_ops/finalize.py").read_text(encoding="utf-8")
    param_finalize = Path("tal/core/param_ops/finalize.py").read_text(encoding="utf-8")
    assert "def assign_sequence_size_from_valid_mask(" in validity_text
    assert "def assign_validity_from_mask(" not in combine_finalize
    assert "def assign_sequence_size_if_left_packed(" not in param_finalize


def test_orch_arch_010_typed_lifecycle_core_owner_exists() -> None:
    """ID: ORCH_ARCH_010_typed_lifecycle_core_owner_exists."""
    path = Path("tal/core/typed_lifecycle.py")
    text = path.read_text(encoding="utf-8")
    assert path.exists()
    assert "LifecyclePhase = Literal[" in text
    assert "class TypedLifecycleContext" in text
    assert "class TypedLifecycleSpec" in text
    assert "class TypedAnalysisObject(AnalysisObject):" in text
    assert "def default_coerce_source(" in text
    assert "__all__ = [" in text
    assert file_loc(path=path) <= 600
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{path}:{name} exceeds function budget: {length} > 50"


def test_orch_arch_011_typed_lifecycle_core_owner_has_no_domain_imports() -> None:
    """ID: ORCH_ARCH_011_typed_lifecycle_core_owner_has_no_domain_imports."""
    text = Path("tal/core/typed_lifecycle.py").read_text(encoding="utf-8")
    forbidden = (
        "tal.linalg",
        "tal.spatial",
        "tal.frames",
        "from ..linalg",
        "from ..spatial",
        "from ..frames",
    )
    for needle in forbidden:
        assert needle not in text


def test_orch_arch_012_typed_lifecycle_no_tal_v2_imports() -> None:
    """ID: ORCH_ARCH_012_typed_lifecycle_no_tal_v2_imports."""
    text = Path("tal/core/typed_lifecycle.py").read_text(encoding="utf-8")
    assert "tal_v2" not in text


def test_arch_topo_001_core_topology_owner_module_present_and_budgeted() -> None:
    """ID: ARCH_TOPO_001_core_topology_owner_module_present_and_budgeted."""
    path = Path("tal/core/orchestration/topology.py")
    assert path.exists()
    assert file_loc(path=path) <= 600
    lengths = function_lengths(path)
    assert "resolve_unary_topology" in lengths
    assert "resolve_binary_topology" in lengths
    assert "resolve_nary_topology" in lengths
    for name, length in lengths.items():
        assert length <= 50, f"{path}:{name} exceeds function budget: {length} > 50"


def test_arch_topo_005_alignment_owner_remains_final_exact_step() -> None:
    """ID: ARCH_TOPO_005_alignment_owner_remains_final_exact_step."""
    text = Path("tal/core/orchestration/alignment.py").read_text(encoding="utf-8")
    assert "def align_exact(" in text
    assert "def align_exact_for_plan(" in text
    assert 'join="exact"' in text
    assert "realize_operands_for_plan(" in text


def test_arch_topo_006_core_topology_owner_contains_no_spatial_semantics() -> None:
    """ID: ARCH_TOPO_006_core_topology_owner_contains_no_spatial_semantics."""
    text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    assert "resolve_apply_output_frames" not in text
    assert "resolve_compose_output_frames" not in text
    assert "set_pose_rep" not in text
    assert "set_rotation_rep" not in text


def test_arch_topo_007_unary_binary_nary_resolvers_route_through_single_internal_planner() -> None:
    """ID: ARCH_TOPO_007_unary_binary_nary_resolvers_route_through_single_internal_planner."""
    text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    assert "def _resolve_topology_plan(" in text
    assert "return _resolve_topology_plan((operand,)" in text
    assert "return _resolve_topology_plan((left, right)" in text
    assert "return _resolve_topology_plan(tuple(operands)" in text


def test_arch_topo_008_nary_planning_avoids_pairwise_fold_chaining() -> None:
    """ID: ARCH_TOPO_008_nary_planning_avoids_pairwise_fold_chaining."""
    text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    assert "def resolve_nary_topology(" in text
    assert "resolve_binary_topology(" not in text.split("def resolve_nary_topology(", 1)[1].split("def ", 1)[0]
    assert "reduce(" not in text


def test_arch_topo_009_topology_policy_exposes_core_dim_alignment_modes() -> None:
    """ID: ARCH_TOPO_009_topology_policy_exposes_core_dim_alignment_modes."""
    text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    assert "class TopologyPolicy:" in text
    assert "core_dim_alignment" in text
    assert '"exact"' in text
    assert '"exclude_all_core"' in text
    assert "STRICT_EXACT_POLICY" in text
    assert "STRICT_NON_CORE_POLICY" in text


def test_arch_topo_013_orchestration_stack_batch_axis_delegates_label_realization_to_batch_topology_owner() -> None:
    """ID: ARCH_TOPO_013_orchestration_stack_batch_axis_delegates_label_realization_to_batch_topology_owner."""
    text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    assert "from .topology_batch import (" in text
    assert "stack_combine_batch_axis," in text
    batch_text = Path("tal/core/orchestration/topology_batch.py").read_text(encoding="utf-8")
    assert "from ..param_ops.batch_topology import" in batch_text
    body = batch_text.split("def stack_combine_batch_axis(", 1)[1].split("\ndef ", 1)[0]
    assert "stacked_batch_coords_with_labels(" in body


def test_arch_topo_014_orchestration_stack_batch_axis_contains_no_values_or_np_asarray_materialization() -> None:
    """ID: ARCH_TOPO_014_orchestration_stack_batch_axis_contains_no_values_or_np_asarray_materialization."""
    text = Path("tal/core/orchestration/topology_batch.py").read_text(encoding="utf-8")
    body = text.split("def stack_combine_batch_axis(", 1)[1].split("\ndef ", 1)[0]
    assert ".values" not in body
    assert "np.asarray(" not in body


def test_arch_topo_015_batch_restore_path_delegates_coord_label_reads_to_guarded_batch_topology_helper() -> None:
    """ID: ARCH_TOPO_015_batch_restore_path_delegates_coord_label_reads_to_guarded_batch_topology_helper."""
    text = Path("tal/core/param_ops/batch_topology.py").read_text(encoding="utf-8")
    assert "def _coord_values_for_labels(" in text
    named_body = text.split("def _batch_cols_from_named_flat_coords(", 1)[1].split("\ndef ", 1)[0]
    restore_body = text.split("def _batch_cols_for_restore(", 1)[1].split("\ndef ", 1)[0]
    finalize_body = text.split("def restore_dataset_batch_dims(", 1)[1].split("\ndef ", 1)[0]
    assert "_coord_values_for_labels(" in named_body
    assert "_coord_values_for_labels(" in restore_body
    assert "_coord_values_for_labels(" in finalize_body


def test_arch_bcast_001_b_helper_defined_once_on_analysis_object() -> None:
    """ID: ARCH_BCAST_001_b_helper_defined_once_on_analysis_object."""
    ao_text = Path("tal/core/analysis_object.py").read_text(encoding="utf-8")
    assert 'def b(self, *, mode: Literal["semantic_broadcast"] = "semantic_broadcast")' in ao_text


def test_arch_bcast_003_inputs_owner_preserves_broadcast_intent() -> None:
    """ID: ARCH_BCAST_003_inputs_owner_preserves_broadcast_intent."""
    inputs_text = Path("tal/core/orchestration/inputs.py").read_text(encoding="utf-8")
    assert "read_broadcast_intent(" in inputs_text
    assert "read_alignment_intent(" in inputs_text
    assert "_broadcast_intent = None" not in inputs_text


def test_arch_bcast_004_topology_owner_contains_semantic_mode_logic() -> None:
    """ID: ARCH_BCAST_004_topology_owner_contains_semantic_mode_logic."""
    topology_text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    assert '"semantic_broadcast"' in topology_text
    assert "SEMANTIC_EXACT_POLICY" in topology_text
    assert "SEMANTIC_NON_CORE_POLICY" in topology_text
    assert "_realize_semantic_broadcast_operands(" in topology_text


def test_arch_bcast_007_no_global_broadcast_state_is_introduced() -> None:
    """ID: ARCH_BCAST_007_no_global_broadcast_state_is_introduced."""
    intent_text = Path("tal/core/orchestration/broadcast_intent.py").read_text(encoding="utf-8")
    assert "global " not in intent_text
    assert "_BROADCAST_STATE" not in intent_text
    assert "setdefault(" not in intent_text


def test_arch_bcast_018_default_policy_selection_matrix_is_explicit_and_localized() -> None:
    """ID: ARCH_BCAST_018_default_policy_selection_matrix_is_explicit_and_localized."""
    matrix_text = Path("tal/utils/topology_operation_families.py").read_text(encoding="utf-8")
    broadcast_text = Path("tal/core/orchestration/broadcast_intent.py").read_text(encoding="utf-8")
    alignment_text = Path("tal/core/orchestration/alignment_intent.py").read_text(encoding="utf-8")
    assert "_SEMANTIC_DEFAULT_OPERATION_FAMILIES" in matrix_text
    assert "_ALIGNMENT_SUPPORTED_OPERATION_FAMILIES" in matrix_text
    assert '"linalg.elementwise"' in matrix_text
    assert '"ufunc.arithmetic"' in matrix_text
    assert '"spatial.rotation.apply"' in matrix_text
    assert '"linalg.elementwise"' not in broadcast_text
    assert '"spatial.rotation.apply"' not in broadcast_text
    assert '"linalg.elementwise"' not in alignment_text
    assert '"spatial.rotation.apply"' not in alignment_text


def test_arch_bcast_019_no_global_default_toggle_side_channel_introduced() -> None:
    """ID: ARCH_BCAST_019_no_global_default_toggle_side_channel_introduced."""
    text = Path("tal/core/orchestration/broadcast_intent.py").read_text(encoding="utf-8")
    assert "os.environ" not in text
    assert "set_global" not in text
    assert "DEFAULT_BROADCAST_MODE" not in text
    assert "global " not in text


def test_arch_bcast_021_approved_family_list_remains_centralized_and_guarded() -> None:
    """ID: ARCH_BCAST_021_approved_family_list_remains_centralized_and_guarded."""
    matrix_text = Path("tal/utils/topology_operation_families.py").read_text(encoding="utf-8")
    linalg_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    ufunc_text = Path("tal/core/ufunc_ops/orchestrate.py").read_text(encoding="utf-8")
    assert "_SEMANTIC_DEFAULT_OPERATION_FAMILIES" in matrix_text
    assert "_ALIGNMENT_SUPPORTED_OPERATION_FAMILIES" in matrix_text
    assert "_SEMANTIC_DEFAULT_OPERATION_FAMILIES" not in linalg_text
    assert "_SEMANTIC_DEFAULT_OPERATION_FAMILIES" not in ufunc_text
    assert "operation_intent_support_for_operation_family(" in linalg_text
    assert "operation_intent_support_for_operation_family(" in ufunc_text
    assert "select_topology_policy_with_intents(" in linalg_text
    assert "select_topology_policy_with_intents(" in ufunc_text


def test_arch_bcast_022_a_helper_defined_once_on_analysis_object() -> None:
    """ID: ARCH_BCAST_022_a_helper_defined_once_on_analysis_object."""
    ao_text = Path("tal/core/analysis_object.py").read_text(encoding="utf-8")
    assert "def a(" in ao_text
    assert "AnalysisObject.a" in ao_text


def test_arch_bcast_024_alignment_intent_merge_owner_is_centralized() -> None:
    """ID: ARCH_BCAST_024_alignment_intent_merge_owner_is_centralized."""
    intent_text = Path("tal/core/orchestration/alignment_intent.py").read_text(encoding="utf-8")
    assert "class AlignmentIntent" in intent_text
    assert "class OperationIntentSupport" in intent_text
    assert "def merge_alignment_intent(" in intent_text
    assert "def select_topology_policy_with_intents(" in intent_text
    assert "operation_family" in intent_text


def test_arch_bcast_025_sequence_vs_param_key_selection_is_localized_and_guarded() -> None:
    """ID: ARCH_BCAST_025_sequence_vs_param_key_selection_is_localized_and_guarded."""
    topology_text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    assert "alignment_on" in topology_text
    assert "def _resolve_primary_key(" in topology_text
    assert "alignment_on='auto' is ambiguous" in topology_text


def test_arch_bcast_026_core_policy_matrix_is_centralized_no_local_forks() -> None:
    """ID: ARCH_BCAST_026_core_policy_matrix_is_centralized_no_local_forks."""
    topology_text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    linalg_text = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    assert "core_policy" in topology_text
    assert "def _resolve_numpy_named_core_dims(" in topology_text
    assert "core_policy='numpy_named'" in topology_text
    assert "resolve_elementwise_output_core_dims(" in linalg_text


def test_arch_bcast_038_shared_topology_strict_core_output_resolution_invokes_matching_core_validator() -> None:
    """ID: ARCH_BCAST_038_shared_topology_strict_core_output_resolution_invokes_matching_core_validator."""
    topology_text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    body = topology_text.split("def _resolve_output_core_dims(", 1)[1].split("\ndef ", 1)[0]
    assert "strict_core_match_required" in body
    assert "_require_matching_core_dims(" in body
    assert "return operands[0].semantic.core_dims" in body


def test_arch_bcast_027_no_global_alignment_state_introduced() -> None:
    """ID: ARCH_BCAST_027_no_global_alignment_state_introduced."""
    text = Path("tal/core/orchestration/alignment_intent.py").read_text(encoding="utf-8")
    assert "global " not in text
    assert "os.environ" not in text
    assert "_ALIGNMENT_STATE" not in text


def test_arch_bcast_028_nonconflicting_a_b_chain_order_independence_is_guarded() -> None:
    """ID: ARCH_BCAST_028_nonconflicting_a_b_chain_order_independence_is_guarded."""
    ao_text = Path("tal/core/analysis_object.py").read_text(encoding="utf-8")
    assert "def a(" in ao_text
    assert "def b(" in ao_text
    assert "read_alignment_intent(" in ao_text
    assert "read_broadcast_intent(" in ao_text


def test_arch_bcast_029_alignment_intent_option_objects_remain_budget_compliant() -> None:
    """ID: ARCH_BCAST_029_alignment_intent_option_objects_remain_budget_compliant."""
    path = Path("tal/core/orchestration/alignment_intent.py")
    text = path.read_text(encoding="utf-8")
    assert "class AlignmentIntent" in text
    lengths = function_lengths(path)
    for name, length in lengths.items():
        assert length <= 50, f"{path}:{name} exceeds function budget: {length} > 50"


def test_arch_bcast_030_combine_align_api_semantics_remain_distinct_from_operand_intent() -> None:
    """ID: ARCH_BCAST_030_combine_align_api_semantics_remain_distinct_from_operand_intent."""
    combine_text = Path("tal/core/combine_ops/accessor.py").read_text(encoding="utf-8")
    assert ".a(" not in combine_text
    assert "AlignmentIntent" not in combine_text


def test_bcast_doc_002_e3a_default_vs_strict_family_behavior_documented() -> None:
    """ID: BCAST_DOC_002_e3a_default_vs_strict_family_behavior_documented."""
    numpy_doc = Path("docs/user-guide/numpy.md").read_text(encoding="utf-8")
    ao_doc = Path("docs/api/analysis-object.md").read_text(encoding="utf-8")
    numpy_lower = numpy_doc.lower()
    assert "broadcast intent" in numpy_lower
    assert ".b()" in numpy_doc
    assert "tal.analysisobject.b" in ao_doc.lower()


def test_bcast_doc_003_alignment_intent_api_and_broadcast_intent_interop_documented() -> None:
    """ID: BCAST_DOC_003_alignment_intent_api_and_broadcast_intent_interop_documented."""
    numpy_doc = Path("docs/user-guide/numpy.md").read_text(encoding="utf-8")
    ao_doc = Path("docs/api/analysis-object.md").read_text(encoding="utf-8")
    assert "## Alignment Intent" in numpy_doc
    assert "## Broadcast Intent" in numpy_doc
    assert ".a(" in numpy_doc
    assert ".b()" in numpy_doc
    assert "tal.AnalysisObject.a" in ao_doc


def test_arch_bcast_032_spatial_consumers_do_not_reimplement_key_selection_or_join_policy() -> None:
    """ID: ARCH_BCAST_032_spatial_consumers_do_not_reimplement_key_selection_or_join_policy."""
    files = [
        Path("tal/spatial/rotation.py"),
        Path("tal/spatial/ops/pose_ops.py"),
        Path("tal/spatial/ops/rotation_apply_ops.py"),
        Path("tal/spatial/ops/pose_apply_ops.py"),
        Path("tal/spatial/kinematics/family.py"),
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "select_topology_policy_with_intents(" in text
        assert "default_topology_mode_for_operation_family(" not in text
        assert "select_topology_policy_for_operation_family(" not in text
        assert "alignment_on=" not in text


def test_arch_bcast_035_spatial_alignment_intent_does_not_introduce_global_frame_override_state() -> None:
    """ID: ARCH_BCAST_035_spatial_alignment_intent_does_not_introduce_global_frame_override_state."""
    files = [
        Path("tal/core/orchestration/alignment_intent.py"),
        Path("tal/core/orchestration/broadcast_intent.py"),
        Path("tal/spatial/policies/frame.py"),
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "global " not in text
        assert "FRAME_OVERRIDE" not in text
        assert "setdefault(" not in text


def test_arch_bcast_036_spatial_approved_family_matrix_is_explicit_and_guarded() -> None:
    """ID: ARCH_BCAST_036_spatial_approved_family_matrix_is_explicit_and_guarded."""
    expected = (
        "spatial.rotation.apply",
        "spatial.rotation.compose",
        "spatial.pose.apply",
        "spatial.pose.compose",
        "spatial.pose.components",
        "spatial.velocity.components",
        "spatial.acceleration.components",
        "spatial.velocity.vector6",
        "spatial.acceleration.vector6",
        "spatial.kinematics.path_coupling",
    )
    matrix_text = Path("tal/utils/topology_operation_families.py").read_text(encoding="utf-8")
    for family in expected:
        token = f'"{family}"'
        assert token in matrix_text


def test_bcast_doc_004_spatial_frame_aware_alignment_and_broadcast_boundaries_documented() -> None:
    """ID: BCAST_DOC_004_spatial_frame_aware_alignment_and_broadcast_boundaries_documented."""
    numpy_doc = Path("docs/user-guide/numpy.md").read_text(encoding="utf-8").lower()
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    assert "frame-sensitive spatial" in numpy_doc


def test_arch_spatial_131_temporal_vector_like_param_key_resolution_reuses_shared_param_runtime_owner() -> None:
    """ID: ARCH_SPATIAL_131_temporal_vector_like_param_key_resolution_reuses_shared_param_runtime_owner."""
    accessor_text = Path("tal/core/param_ops/accessor.py").read_text(encoding="utf-8")
    resolve_text = Path("tal/core/orchestration/resolve.py").read_text(encoding="utf-8")
    assert "resolve_param_runtime_context(" in accessor_text
    assert "def resolve_param_runtime_context(" in resolve_text
    assert "_resolve_schema_context_validated(" in resolve_text


def test_arch_spatial_135_temporal_vector_like_execution_owners_contain_no_vectorize_true() -> None:
    """ID: ARCH_SPATIAL_135_temporal_vector_like_execution_owners_contain_no_vectorize_true."""
    files = [
        Path("tal/core/param_ops/accessor.py"),
        Path("tal/core/param_ops/evaluate.py"),
        Path("tal/core/param_ops/resample.py"),
        Path("tal/core/param_engine/map_apply.py"),
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "vectorize=True" not in text


def test_arch_spatial_136_temporal_vector_like_stopgaps_are_backend_routed_and_explicit() -> None:
    """ID: ARCH_SPATIAL_136_temporal_vector_like_stopgaps_are_backend_routed_and_explicit."""
    map_build = Path("tal/core/param_engine/map_build.py").read_text(encoding="utf-8")
    backends = Path("tal/core/param_engine/backends.py").read_text(encoding="utf-8")
    map_section = map_build.split("def _apply_param_map_block(", 1)[1].split("def build_param_map(", 1)[0]
    bounds_section = map_build.split("def _apply_param_bounds_block(", 1)[1].split(
        "def build_param_bounds_map(",
        1,
    )[0]
    assert "_select_map_normal_backend" in map_build
    assert "_select_bounds_normal_backend" in map_build
    assert "PARAM_MAP_BACKEND_NUMPY_BLOCK" in map_build
    assert "PARAM_BOUNDS_BACKEND_NUMPY_BLOCK" in map_build
    assert "map_block_backend" in map_section
    assert "bounds_block_backend" in bounds_section
    assert "vectorize=False" in map_section
    assert "vectorize=False" in bounds_section
    assert "vectorize=True" not in map_section
    assert "vectorize=True" not in bounds_section
    assert "PARAM_MAP_BACKEND_NUMPY_BLOCK" in backends
    assert "PARAM_BOUNDS_BACKEND_NUMPY_BLOCK" in backends
    assert "PARAM_MAP_BACKEND_NUMPY_ROW" not in backends
    assert "PARAM_BOUNDS_BACKEND_NUMPY_ROW" not in backends
    assert "def map_row_backend(" not in backends
    assert "def bounds_row_backend(" not in backends


def test_arch_spatial_151_d3_derivative_integral_paths_do_not_reuse_interpolation_execution_owners() -> None:
    """ID: ARCH_SPATIAL_151_d3_derivative_integral_paths_do_not_reuse_interpolation_execution_owners."""
    text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    assert "resolve_param_runtime_context(" in text
    assert "build_param_map(" not in text
    assert "apply_param_map(" not in text
    assert ".param.at(" not in text
    assert ".param.resample_to(" not in text


def test_arch_spatial_156_d4_param_key_resolution_reuses_shared_param_runtime_owner() -> None:
    """ID: ARCH_SPATIAL_156_d4_param_key_resolution_reuses_shared_param_runtime_owner."""
    temporal_text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    smoothing_text = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    resolve_text = Path("tal/core/orchestration/resolve.py").read_text(encoding="utf-8")
    assert "def _resolve_runtime(" in temporal_text
    assert "resolve_param_runtime_context(" in temporal_text
    assert "_resolve_runtime(" in smoothing_text
    assert "def resolve_param_runtime_context(" in resolve_text


def test_arch_spatial_157_d4_core_helper_promotions_meet_multi_consumer_policy() -> None:
    """ID: ARCH_SPATIAL_157_d4_core_helper_promotions_meet_multi_consumer_policy."""
    smoothing_ops = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    kernel_text = Path("tal/spatial/kernels/kinematics_temporal_kernels.py").read_text(encoding="utf-8")
    helper_text = Path("tal/spatial/kernels/local_poly_weights.py").read_text(encoding="utf-8")
    assert "from .local_poly_weights import build_local_poly_weight_stack" in kernel_text
    assert "build_local_poly_weight_stack(" in kernel_text
    assert "core.param_engine.local_stencil" not in smoothing_ops
    assert "core.param_engine.local_stencil" not in kernel_text
    assert "def build_local_poly_weight_stack(" in helper_text


def test_arch_spatial_160_d5_temporal_numeric_paths_do_not_invoke_interpolation_map_owners() -> None:
    """ID: ARCH_SPATIAL_160_d5_temporal_numeric_paths_do_not_invoke_interpolation_map_owners."""
    temporal_text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    smoothing_text = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    for text in (temporal_text, smoothing_text):
        assert "build_param_map(" not in text
        assert "apply_param_map(" not in text
        assert ".param.at(" not in text
        assert ".param.resample_to(" not in text


def test_arch_spatial_161_d5_runtime_owner_reuse_and_core_promotion_policy_stable() -> None:
    """ID: ARCH_SPATIAL_161_d5_runtime_owner_reuse_and_core_promotion_policy_stable."""
    temporal_text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    smoothing_text = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    kernel_text = Path("tal/spatial/kernels/kinematics_temporal_kernels.py").read_text(encoding="utf-8")
    assert "resolve_param_runtime_context(" in temporal_text
    assert "_resolve_runtime(" in smoothing_text
    assert "core.param_engine.local_stencil" not in kernel_text
    assert "core.param_engine.blockwise_sequence" not in temporal_text


def test_spatial_hard_175_d6_ao_and_family_surfaces_do_not_invoke_hidden_interpolation_owners() -> None:
    """ID: SPATIAL_HARD_175_d6_ao_and_family_surfaces_do_not_invoke_hidden_interpolation_owners."""
    surface_text = Path("tal/spatial/temporal/surface.py").read_text(encoding="utf-8")
    family_text = Path("tal/spatial/ops/kinematics_family_temporal_ops.py").read_text(encoding="utf-8")
    for text in (surface_text, family_text):
        assert "build_param_map(" not in text
        assert "apply_param_map(" not in text
        assert ".param.at(" not in text
        assert ".param.resample_to(" not in text


def test_arch_spatial_c3_c5_001_frame_wrapper_owners_do_not_invoke_hidden_interpolation_owners() -> None:
    """ID: ARCH_SPATIAL_C3_C5_001_frame_wrapper_owners_do_not_invoke_hidden_interpolation_owners."""
    expression_text = Path("tal/spatial/ops/frame_expression_ops.py").read_text(encoding="utf-8")
    kinematics_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    for text in (expression_text, kinematics_text):
        assert "build_param_map(" not in text
        assert "apply_param_map(" not in text
        assert ".param.at(" not in text
        assert ".param.resample_to(" not in text



def test_arch_spatial_c3_c5_002_frame_wrapper_owners_remain_delegation_only_no_vectorize_or_row_loops() -> None:
    """ID: ARCH_SPATIAL_C3_C5_002_frame_wrapper_owners_remain_delegation_only_no_vectorize_or_row_loops."""
    expression_text = Path("tal/spatial/ops/frame_expression_ops.py").read_text(encoding="utf-8")
    kinematics_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    for text in (expression_text, kinematics_text):
        assert "vectorize=True" not in text
        assert "xr.apply_ufunc(" not in text
        assert "for row in" not in text
        assert "for idx in" not in text
