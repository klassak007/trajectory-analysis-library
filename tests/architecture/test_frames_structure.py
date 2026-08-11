from __future__ import annotations

import ast
from pathlib import Path

from ._budget import executable_source, file_loc, function_lengths


def test_arch_frames_001_slice_a_owner_split_and_budget() -> None:
    """ID: ARCH_FRAMES_001_slice_a_owner_split_and_budget."""
    files = [
        Path("tal/frames/__init__.py"),
        Path("tal/frames/registry.py"),
        Path("tal/utils/frame_schema.py"),
    ]
    for path in files:
        assert path.exists(), f"missing frames owner file: {path}"
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget: {length} > 50"


def test_arch_frames_002_no_core_frames_cross_imports() -> None:
    """ID: ARCH_FRAMES_002_no_core_frames_cross_imports."""
    for path in sorted(Path("tal/frames").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.core" not in text
        assert "from ..core" not in text
    for path in sorted(Path("tal/core").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.frames" not in text
        assert "from ..frames" not in text
        assert "from .frames" not in text


def test_arch_frames_003_frame_metadata_bridge_uses_schema_writer_no_direct_attrs_mutation() -> None:
    """ID: ARCH_FRAMES_003_frame_metadata_bridge_uses_schema_writer_no_direct_attrs_mutation."""
    bridge = Path("tal/utils/frame_schema.py").read_text(encoding="utf-8")
    assert "def set_frames(" in bridge
    assert "merge_schema(" in bridge
    set_frames_block = bridge.split("def set_frames(", 1)[1]
    assert 'attrs["tal"]' not in set_frames_block
    assert "attrs['tal']" not in set_frames_block


def test_arch_frames_004_core_schema_has_no_frame_metadata_owner_symbols() -> None:
    """ID: ARCH_FRAMES_004_core_schema_has_no_frame_metadata_owner_symbols."""
    schema = Path("tal/core/schema.py").read_text(encoding="utf-8")
    assert "def get_frames(" not in schema
    assert "def set_frames(" not in schema


def test_arch_frames_005_frame_schema_key_validation_avoids_raw_mixed_key_sort() -> None:
    """ID: ARCH_FRAMES_005_frame_schema_key_validation_avoids_raw_mixed_key_sort."""
    bridge = Path("tal/utils/frame_schema.py").read_text(encoding="utf-8")
    assert "sorted(block.keys())" not in bridge
    assert "def _iter_deterministic_keys(" in bridge
    assert "schema.frames.key.invalid" in bridge


def test_arch_frames_006_framegraph_replace_conflict_ancestry_guard_present() -> None:
    """ID: ARCH_FRAMES_006_framegraph_replace_conflict_ancestry_guard_present."""
    registry = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    assert "def _is_replace_conflict_unsafe(" in registry
    assert "ancestor/descendant" in registry


def test_arch_frames_007_framegraph_public_type_guard_present() -> None:
    """ID: ARCH_FRAMES_007_framegraph_public_type_guard_present."""
    registry = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    assert "def _assert_owned(self, frame: object" in registry
    assert "must be Frame" in registry
    assert "self._assert_owned(parent, context=\"FrameGraph.get_or_create_frame\")" in registry
    assert "self._assert_owned(target, context=owner)" in registry


def test_arch_frames_008_slice_b_owner_split_and_budget() -> None:
    """ID: ARCH_FRAMES_008_slice_b_owner_split_and_budget."""
    topology = Path("tal/frames/topology.py")
    assert topology.exists(), f"missing Slice B topology owner: {topology}"
    assert file_loc(path=topology) <= 600, f"{topology} exceeds file budget."
    for name, length in function_lengths(topology).items():
        assert length <= 50, f"{topology}:{name} exceeds function budget: {length} > 50"


def test_arch_frames_009_find_fold_topology_owner_is_single_source() -> None:
    """ID: ARCH_FRAMES_009_find_fold_topology_owner_is_single_source."""
    topology = Path("tal/frames/topology.py").read_text(encoding="utf-8")
    registry = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    assert "def find_path(" in topology
    assert "def fold_path(" in topology
    assert "def find_path(" not in registry
    assert "def fold_path(" not in registry


def test_arch_frames_010_mutation_conflict_policy_single_owner_in_registry() -> None:
    """ID: ARCH_FRAMES_010_mutation_conflict_policy_single_owner_in_registry."""
    registry = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    topology = Path("tal/frames/topology.py").read_text(encoding="utf-8")
    assert "def reparent_frame(" in registry
    assert "on_conflict" in registry
    assert "def reparent_frame(" not in topology


def test_arch_frames_011_no_frames_core_spatial_import_coupling_slice_b() -> None:
    """ID: ARCH_FRAMES_011_no_frames_core_spatial_import_coupling_slice_b."""
    for path in sorted(Path("tal/frames").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.core" not in text
        assert "from ..core" not in text
        assert "tal.spatial" not in text
        assert "from ..spatial" not in text
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text


def test_arch_frames_012_framegraph_ownership_guard_requires_registration_membership() -> None:
    """ID: ARCH_FRAMES_012_framegraph_ownership_guard_requires_registration_membership."""
    registry = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    assert "def _assert_owned(self, frame: object" in registry
    assert "self._frames.get(frame.id) is not frame" in registry
    assert "not registered in this FrameGraph" in registry


def test_arch_frames_013_topology_registration_check_requires_graph_binding_guard() -> None:
    """ID: ARCH_FRAMES_013_topology_registration_check_requires_graph_binding_guard."""
    topology = Path("tal/frames/topology.py").read_text(encoding="utf-8")
    assert "def _require_frame_graph_binding(" in topology
    assert "is not bound to a valid FrameGraph" in topology
    assert "graph = _require_frame_graph_binding(frame, owner=owner, arg=arg)" in topology


def test_arch_frames_014_snapshot_owner_split_and_budget() -> None:
    """ID: ARCH_FRAMES_014_snapshot_owner_split_and_budget."""
    snapshot = Path("tal/frames/snapshot.py")
    assert snapshot.exists(), f"missing Slice C snapshot owner: {snapshot}"
    assert file_loc(path=snapshot) <= 600, f"{snapshot} exceeds file budget."
    for name, length in function_lengths(snapshot).items():
        assert length <= 50, f"{snapshot}:{name} exceeds function budget: {length} > 50"


def test_arch_frames_015_snapshot_issue_codes_single_owner_constant() -> None:
    """ID: ARCH_FRAMES_015_snapshot_issue_codes_single_owner_constant."""
    snapshot = Path("tal/frames/snapshot.py").read_text(encoding="utf-8")
    registry = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    topology = Path("tal/frames/topology.py").read_text(encoding="utf-8")
    assert "_SNAPSHOT_ISSUE_CODES" in snapshot
    for code in (
        "unknown_child_ref",
        "parent_child_mismatch",
        "duplicate_parent_claim",
        "cycle_detected",
        "unreachable_registered_frame",
    ):
        assert code in snapshot
    assert "_SNAPSHOT_ISSUE_CODES" not in registry
    assert "_SNAPSHOT_ISSUE_CODES" not in topology


def test_arch_frames_016_snapshot_networkx_bridge_guarded_optional_import() -> None:
    """ID: ARCH_FRAMES_016_snapshot_networkx_bridge_guarded_optional_import."""
    snapshot = Path("tal/frames/snapshot.py").read_text(encoding="utf-8")
    assert 'importlib.import_module("networkx")' in snapshot
    assert "import networkx" not in snapshot
    assert "snapshot_to_networkx: networkx is required" in snapshot


def test_arch_frames_017_no_frames_core_spatial_import_coupling_slice_c() -> None:
    """ID: ARCH_FRAMES_017_no_frames_core_spatial_import_coupling_slice_c."""
    snapshot = Path("tal/frames/snapshot.py").read_text(encoding="utf-8")
    assert "tal.core" not in snapshot
    assert "from ..core" not in snapshot
    assert "tal.spatial" not in snapshot
    assert "from ..spatial" not in snapshot
    assert 'attrs["tal"]' not in snapshot
    assert "attrs['tal']" not in snapshot


def test_arch_frames_018_snapshot_seed_resolution_requires_registered_frame_object_check() -> None:
    """ID: ARCH_FRAMES_018_snapshot_seed_resolution_requires_registered_frame_object_check."""
    snapshot = Path("tal/frames/snapshot.py").read_text(encoding="utf-8")
    assert "def _require_registered_seed_name(" in snapshot
    assert "is not a registered Frame object" in snapshot
    assert "_require_registered_seed_name(seed_id, graph=graph, owner=owner)" in snapshot


def test_arch_frames_019_snapshot_default_seed_path_avoids_raw_registry_key_sort() -> None:
    """ID: ARCH_FRAMES_019_snapshot_default_seed_path_avoids_raw_registry_key_sort."""
    snapshot = Path("tal/frames/snapshot.py").read_text(encoding="utf-8")
    assert "sorted(graph._frames.keys())" not in snapshot
    assert "def _resolve_default_seed_ids(" in snapshot
    assert "_resolve_default_seed_ids(graph, owner=owner)" in snapshot


def test_arch_frames_020_slice_d_owner_split_and_budget() -> None:
    """ID: ARCH_FRAMES_020_slice_d_owner_split_and_budget."""
    frame_ops = Path("tal/utils/frame_ops.py")
    assert frame_ops.exists(), f"missing Slice D owner module: {frame_ops}"
    assert file_loc(path=frame_ops) <= 600, f"{frame_ops} exceeds file budget."
    for name, length in function_lengths(frame_ops).items():
        assert length <= 50, f"{frame_ops}:{name} exceeds function budget: {length} > 50"


def test_arch_frames_021_ao_frames_integration_owned_in_utils_not_core_or_frames() -> None:
    """ID: ARCH_FRAMES_021_ao_frames_integration_owned_in_utils_not_core_or_frames."""
    frame_ops = Path("tal/utils/frame_ops.py").read_text(encoding="utf-8")
    analysis_object = Path("tal/core/analysis_object.py").read_text(encoding="utf-8")
    init_text = Path("tal/__init__.py").read_text(encoding="utf-8")
    assert "class FramesAccessor" in frame_ops
    assert "def install_analysis_object_frames_accessor(" in frame_ops
    assert "def frames(" not in analysis_object
    assert "install_analysis_object_frames_accessor()" in init_text


def test_arch_frames_022_slice_d_reuses_frame_schema_and_framegraph_owners_no_local_duplication() -> None:
    """ID: ARCH_FRAMES_022_slice_d_reuses_frame_schema_and_framegraph_owners_no_local_duplication."""
    frame_ops = Path("tal/utils/frame_ops.py").read_text(encoding="utf-8")
    assert "from .frame_schema import get_frames, set_frames" in frame_ops
    assert "from tal.frames import Frame, FrameGraph, get_active_frame_graph" in frame_ops
    assert "def get_frames(" not in frame_ops
    assert "def set_frames(" not in frame_ops


def test_arch_frames_023_slice_d_no_direct_schema_attrs_mutation_or_forbidden_imports() -> None:
    """ID: ARCH_FRAMES_023_slice_d_no_direct_schema_attrs_mutation_or_forbidden_imports."""
    frame_ops = executable_source(path="tal/utils/frame_ops.py")
    assert 'attrs["tal"]' not in frame_ops
    assert "attrs['tal']" not in frame_ops
    assert "tal.spatial" not in frame_ops
    assert "from ..spatial" not in frame_ops
    assert "from tal.core import" not in frame_ops


def test_arch_frames_024_slice_d_frame_rename_preflights_metadata_before_graph_mutation() -> None:
    """ID: ARCH_FRAMES_024_slice_d_frame_rename_preflights_metadata_before_graph_mutation."""
    frame_ops = Path("tal/utils/frame_ops.py").read_text(encoding="utf-8")
    remap_pos = frame_ops.index("remapped = frame_remap_ids(")
    rename_pos = frame_ops.index("resolved_graph.rename_frame(")
    assert remap_pos < rename_pos


def test_arch_frames_025_slice_d_frame_bind_requires_registered_frame_object_guard() -> None:
    """ID: ARCH_FRAMES_025_slice_d_frame_bind_requires_registered_frame_object_guard."""
    frame_ops = Path("tal/utils/frame_ops.py").read_text(encoding="utf-8")
    assert "def _require_registered_frame_object(" in frame_ops
    assert "is not a registered Frame object in graph" in frame_ops
    assert "return _require_registered_frame_object(" in frame_ops


def test_arch_frames_026_slice_d_validate_true_rewrap_uses_schema_validation_owner_path() -> None:
    """ID: ARCH_FRAMES_026_slice_d_validate_true_rewrap_uses_schema_validation_owner_path."""
    frame_ops = Path("tal/utils/frame_ops.py").read_text(encoding="utf-8")
    assert "from tal.core.schema import UNSET, UnsetType, validate_schema" in frame_ops
    assert "return source.__class__._from_validated(validate_schema(ds))" in frame_ops


def test_arch_frames_027_visualization_owner_split_and_budget() -> None:
    """ID: ARCH_FRAMES_027_visualization_owner_split_and_budget."""
    visualization = Path("tal/frames/visualization.py")
    assert visualization.exists(), f"missing visualization owner: {visualization}"
    assert file_loc(path=visualization) <= 600, f"{visualization} exceeds file budget."
    for name, length in function_lengths(visualization).items():
        assert length <= 50, f"{visualization}:{name} exceeds function budget: {length} > 50"


def test_arch_frames_028_visualization_reuses_snapshot_and_path_owners_no_local_duplication() -> None:
    """ID: ARCH_FRAMES_028_visualization_reuses_snapshot_and_path_owners_no_local_duplication."""
    visualization = Path("tal/frames/visualization.py").read_text(encoding="utf-8")
    assert "from .snapshot import FrameSnapshot, snapshot_from_seeds, snapshot_to_networkx" in visualization
    assert "from .topology import FramePath, find_path" in visualization
    assert "snapshot_from_seeds(" in visualization
    assert "snapshot_to_networkx(" in visualization
    assert "find_path(" in visualization
    assert "def snapshot_from_seeds(" not in visualization
    assert "def snapshot_to_networkx(" not in visualization
    assert "def find_path(" not in visualization


def test_arch_frames_029_visualization_optional_imports_are_guarded() -> None:
    """ID: ARCH_FRAMES_029_visualization_optional_imports_are_guarded."""
    visualization = Path("tal/frames/visualization.py").read_text(encoding="utf-8")
    assert 'importlib.import_module("networkx")' in visualization
    assert 'importlib.import_module("matplotlib.pyplot")' in visualization
    assert "import networkx" not in visualization
    assert "import matplotlib.pyplot" not in visualization
    assert "networkx is required" in visualization
    assert "matplotlib is required" in visualization


def test_arch_frames_030_visualization_no_core_spatial_import_boundary_regression() -> None:
    """ID: ARCH_FRAMES_030_visualization_no_core_spatial_import_boundary_regression."""
    visualization = Path("tal/frames/visualization.py").read_text(encoding="utf-8")
    assert "tal.core" not in visualization
    assert "from ..core" not in visualization
    assert "tal.spatial" not in visualization
    assert "from ..spatial" not in visualization
    assert 'attrs["tal"]' not in visualization
    assert "attrs['tal']" not in visualization


def test_arch_frames_031_framegraph_context_tokens_have_context_local_owner() -> None:
    """ID: ARCH_FRAMES_031_framegraph_context_tokens_have_context_local_owner."""
    registry = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    module = ast.parse(registry)
    graph_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "FrameGraph"
    )
    instance_token_attrs = {
        node.attr
        for node in ast.walk(graph_class)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and "token" in node.attr.lower()
    }
    assert not instance_token_attrs
    assert "_FRAME_GRAPH_CONTEXT_STACK: contextvars.ContextVar[" in registry
    assert 'contextvars.ContextVar("tal_frame_graph_context_stack", default=())' in registry
    assert "_FRAME_GRAPH_CONTEXT_STACK.set((*stack, (self, token)))" in registry
    assert "_ACTIVE_FRAME_GRAPH.reset(token)" in registry
    assert "_FRAME_GRAPH_CONTEXT_STACK.set(stack[:-1])" in registry


def test_arch_frames_c6_001_motion_and_inertial_metadata_owners_stay_frames_or_utils() -> None:
    """ID: ARCH_FRAMES_C6_001_motion_and_inertial_metadata_owners_stay_frames_or_utils."""
    spatial_motion_path = Path("tal/spatial/metadata/frame_motion.py")
    assert spatial_motion_path.exists()
    spatial_motion_text = spatial_motion_path.read_text(encoding="utf-8")
    support_text = Path("tal/spatial/ops/kinematics_path_support_ops.py").read_text(encoding="utf-8")
    assert not Path("tal/frames/motion.py").exists()
    assert not Path("tal/utils/frame_motion_ops.py").exists()
    assert "def get_edge_motion_class(" in spatial_motion_text
    assert "def set_edge_motion_class(" in spatial_motion_text
    assert "def get_frame_inertial_status(" in spatial_motion_text
    assert "def set_frame_inertial_status(" in spatial_motion_text
    assert "def propagate_inertial_status(" in spatial_motion_text
    assert "from ..metadata import (" in support_text
    assert "get_edge_motion_class" in support_text
    assert "get_frame_inertial_status" in support_text


def test_arch_frames_c6_002_no_spatial_kernel_ownership_in_c6_motion_tracking() -> None:
    """ID: ARCH_FRAMES_C6_002_no_spatial_kernel_ownership_in_c6_motion_tracking."""
    for path in ("tal/spatial/metadata/frame_motion.py", "tal/spatial/ops/kinematics_path_support_ops.py"):
        text = Path(path).read_text(encoding="utf-8")
        assert "tal.spatial.kernels" not in text
        assert "from ..spatial.kernels" not in text
        assert "xr.apply_ufunc(" not in text


def test_arch_frames_c6_003_no_core_frames_import_boundary_regression() -> None:
    """ID: ARCH_FRAMES_C6_003_no_core_frames_import_boundary_regression."""
    registry_text = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    spatial_motion_text = Path("tal/spatial/metadata/frame_motion.py").read_text(encoding="utf-8")
    assert "tal.core" not in registry_text
    assert "from ..core" not in registry_text
    assert "tal.spatial" not in registry_text
    assert "from ..spatial" not in registry_text
    assert "tal.core" not in spatial_motion_text
    assert "from ..core" not in spatial_motion_text
    assert "from tal.frames import Frame, FrameGraph" in spatial_motion_text


def test_arch_frames_c6_004_non_subtree_remove_uses_detach_to_clear_orphan_edge_extension_state() -> None:
    """ID: ARCH_FRAMES_C6_004_non_subtree_remove_uses_detach_to_clear_orphan_edge_extension_state."""
    registry_text = Path("tal/frames/registry.py").read_text(encoding="utf-8")
    assert (
        "for child in list(target._children.values()):\n"
        "            self._detach_from_parent(child)"
    ) in registry_text


def test_frame_doc_001_phase7_slice_a_frames_docs_and_api_entries_present() -> None:
    """ID: FRAME_DOC_001_phase7_slice_a_frames_docs_and_api_entries_present."""
    schema_doc = Path("docs/api/schema.md").read_text(encoding="utf-8")
    api_frames = Path("docs/api/frames.md").read_text(encoding="utf-8")
    user_frames = Path("docs/user-guide/frames.md").read_text(encoding="utf-8")
    assert "tal.ext.frames" in schema_doc
    assert "tal.utils.frame_schema.get_frames" in schema_doc
    assert "tal.utils.frame_schema.set_frames" in schema_doc
    assert "FrameGraph" in api_frames
    assert "get_active_frame_graph" in api_frames
    assert "task-local" in api_frames.lower()
    assert "task-local" in user_frames.lower()
    assert "metadata" in user_frames.lower()


def test_frame_doc_002_phase7_slice_b_topology_path_docs_and_api_entries_present() -> None:
    """ID: FRAME_DOC_002_phase7_slice_b_topology_path_docs_and_api_entries_present."""
    api_frames = Path("docs/api/frames.md").read_text(encoding="utf-8")
    user_frames = Path("docs/user-guide/frames.md").read_text(encoding="utf-8")
    assert "find_path" in api_frames
    assert "fold_path" in api_frames
    assert "reparent" in user_frames
    assert "conflict" in user_frames.lower()


def test_frame_doc_003_phase7_slice_c_snapshot_docs_and_api_entries_present() -> None:
    """ID: FRAME_DOC_003_phase7_slice_c_snapshot_docs_and_api_entries_present."""
    api_frames = Path("docs/api/frames.md").read_text(encoding="utf-8")
    user_frames = Path("docs/user-guide/frames.md").read_text(encoding="utf-8")
    assert "FrameSnapshot" in api_frames
    assert "snapshot_from_seeds" in api_frames
    assert "render_snapshot_ascii" in api_frames
    assert "snapshot_to_networkx" in api_frames
    assert "snapshot_from_seeds" in user_frames
    assert "render_snapshot_ascii" in user_frames


def test_frame_doc_004_phase7_slice_d_ao_frames_accessor_docs_and_api_entries_present() -> None:
    """ID: FRAME_DOC_004_phase7_slice_d_ao_frames_accessor_docs_and_api_entries_present."""
    api_analysis_object = Path("docs/api/analysis-object.md").read_text(encoding="utf-8")
    api_frames = Path("docs/api/frames.md").read_text(encoding="utf-8")
    user_frames = Path("docs/user-guide/frames.md").read_text(encoding="utf-8")
    assert "ao.frames" in api_analysis_object
    assert "rename_frame" in api_frames
    assert "frame_retag" in api_frames
    assert "frame_bind" in api_frames
    assert "ao.frames" in user_frames
    assert "rename_frame" in user_frames


def test_frame_doc_005_framegraph_visualization_docs_and_api_entries_present() -> None:
    """ID: FRAME_DOC_005_framegraph_visualization_docs_and_api_entries_present."""
    api_frames = Path("docs/api/frames.md").read_text(encoding="utf-8")
    user_frames = Path("docs/user-guide/frames.md").read_text(encoding="utf-8")
    assert "draw_frame_graph" in api_frames
    assert "FrameGraphDrawOptions" in api_frames
    assert "fold_path" in api_frames
    assert "draw_frame_graph" in user_frames
    assert "diagnostics" in user_frames.lower()


def test_frame_doc_c6_001_edge_motion_classes_and_inertial_propagation_documented() -> None:
    """ID: FRAME_DOC_C6_001_edge_motion_classes_and_inertial_propagation_documented."""
    api_frames = Path("docs/api/frames.md").read_text(encoding="utf-8").lower()
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    user_frames = Path("docs/user-guide/frames.md").read_text(encoding="utf-8").lower()
    assert "motion" in spatial_doc
    assert "inertial" in spatial_doc
    assert "path" in spatial_doc
    assert "get_edge_motion_class" not in api_frames
    assert "set_frame_inertial_status" not in api_frames
    assert "motion" in user_frames
    assert "inertial" in user_frames
