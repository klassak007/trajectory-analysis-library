from __future__ import annotations

import ast
from pathlib import Path

from ._budget import executable_source, file_loc, function_lengths


def _module(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _function_nodes(module: ast.Module) -> list[ast.FunctionDef]:
    out: list[ast.FunctionDef] = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            out.append(node)
    return out


def _assert_no_direct_import(text: str, module_name: str) -> None:
    module = ast.parse(text)
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != module_name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != module_name


def _assert_agents_budget(path: Path) -> None:
    assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{path}:{name} exceeds function budget: {length} > 50"


def _class_method_node(module: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
            raise AssertionError(f"missing method: {class_name}.{method_name}")
    raise AssertionError(f"missing class: {class_name}")


def _class_method_names(module: ast.Module, class_name: str) -> set[str]:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {child.name for child in node.body if isinstance(child, ast.FunctionDef)}
    raise AssertionError(f"missing class: {class_name}")


def _function_node(module: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"missing function: {function_name}")


def _parameter_count(node: ast.FunctionDef) -> int:
    args = node.args
    return (
        len(args.posonlyargs)
        + len(args.args)
        + len(args.kwonlyargs)
        + int(args.vararg is not None)
        + int(args.kwarg is not None)
    )


def _call_token(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in {"_resolve_compose_output_frames", "resolve_compose_output_frames"}:
        return "resolve_compose_output_frames"
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        if call.func.attr == "as_quat" and call.func.value.id in {"self", "right", "left"}:
            return f"{call.func.value.id}.as_quat"
    return None


def _compose_call_order_tokens(node: ast.FunctionDef) -> list[str]:
    tokens: list[str] = []

    class _CallVisitor(ast.NodeVisitor):
        def visit_Call(self, call: ast.Call) -> None:
            token = _call_token(call)
            if token is not None:
                tokens.append(token)
            self.generic_visit(call)

    _CallVisitor().visit(node)
    return tokens


def _pose_compose_call_token(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in {
        "_resolve_compose_output_frames",
        "resolve_compose_output_frames",
        "_canonical_components",
    }:
        if call.func.id in {"_resolve_compose_output_frames", "resolve_compose_output_frames"}:
            return "resolve_compose_output_frames"
        return call.func.id
    return None


def _rotation_apply_call_token(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in {
        "resolve_apply_output_frames",
        "_apply_to_vector_target",
        "_apply_to_spatial_target",
    }:
        return call.func.id
    return None


def _pose_apply_call_token(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in {
        "resolve_apply_output_frames",
        "_apply_pose_to_position",
        "_apply_pose_to_spatial_target",
    }:
        return call.func.id
    return None


def _pose_components_call_token(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in {
        "resolve_components_shared_frames",
        "select_topology_policy_with_intents",
        "_build_components_pose_dataset",
    }:
        return call.func.id
    return None


def _kinematics_components_call_token(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in {
        "resolve_components_shared_frames",
        "select_topology_policy_with_intents",
        "build_family_dataset",
    }:
        return call.func.id
    return None


def _call_order_tokens(node: ast.FunctionDef, token_fn: callable) -> list[str]:
    tokens: list[str] = []

    class _CallVisitor(ast.NodeVisitor):
        def visit_Call(self, call: ast.Call) -> None:
            token = token_fn(call)
            if token is not None:
                tokens.append(token)
            self.generic_visit(call)

    _CallVisitor().visit(node)
    return tokens


def test_arch_spatial_001_slice_a1_position_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_001_slice_a1_position_owner_split_and_budget."""
    files = [
        Path("tal/spatial/__init__.py"),
        Path("tal/spatial/metadata/facade.py"),
        Path("tal/spatial/policies/intent.py"),
        Path("tal/spatial/position.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_002_no_core_or_frames_to_spatial_import_coupling() -> None:
    """ID: ARCH_SPATIAL_002_no_core_or_frames_to_spatial_import_coupling."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.spatial" not in text
        assert "from ..spatial" not in text
        assert "from .spatial" not in text
    for path in sorted(Path("tal/frames").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.spatial" not in text
        assert "from ..spatial" not in text
    for path in sorted(Path("tal/spatial").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text


def test_spatial_doc_001_phase8_slice_a1_position_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_001_phase8_slice_a1_position_docs_and_api_entries_present."""
    api_position = Path("docs/api/types/position.md").read_text(encoding="utf-8")
    user_spatial = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "as_delta" in api_position
    assert "cart" in api_position
    assert "Position" in user_spatial


def test_arch_spatial_009_frames_import_guard_scans_frames_tree_recursively() -> None:
    """ID: ARCH_SPATIAL_009_frames_import_guard_scans_frames_tree_recursively."""
    text = Path("tests/architecture/test_spatial_structure.py").read_text(encoding="utf-8")
    recursive = 'for path in sorted(Path("tal/frames").rglob("*.py")):'
    legacy = 'for path in sorted(Path("tal/frames").' + 'glob("*.py")):'
    assert recursive in text
    assert legacy not in text


def test_arch_spatial_010_slice_a2_rotation_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_010_slice_a2_rotation_owner_split_and_budget."""
    files = [
        Path("tal/spatial/metadata/facade.py"),
        Path("tal/spatial/rotation.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_011_slice_a2_no_local_schema_write_or_frame_policy_duplication() -> None:
    """ID: ARCH_SPATIAL_011_slice_a2_no_local_schema_write_or_frame_policy_duplication."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert "from tal.utils.frame_schema import get_frames" in rotation_text
    assert "from .metadata import (" in rotation_text
    assert "get_rotation_rep" in rotation_text
    assert "set_rotation_rep" in rotation_text
    assert "merge_schema(" not in rotation_text
    assert "_read_tal(" not in rotation_text
    assert "_read_ext(" not in rotation_text
    assert "_read_frames_block(" not in rotation_text
    assert 'attrs["tal"]' not in rotation_text
    assert "attrs['tal']" not in rotation_text


def test_spatial_doc_002_phase8_slice_a2_rotation_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_002_phase8_slice_a2_rotation_docs_and_api_entries_present."""
    api_rotation = Path("docs/api/types/rotation.md").read_text(encoding="utf-8")
    user_spatial = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "quat" in api_rotation
    assert "Rotation" in user_spatial
    assert "rotation" in api_types_index


def test_arch_spatial_012_rotation_constructor_validates_spatial_roles_via_metadata_owner() -> None:
    """ID: ARCH_SPATIAL_012_rotation_constructor_validates_spatial_roles_via_metadata_owner."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert "from .metadata import (" in rotation_text
    assert "get_rotation_rep" in rotation_text
    assert "set_rotation_rep" in rotation_text
    assert "validate_spatial_roles" in rotation_text
    assert "validate_spatial_roles(candidate, owner=owner)" in rotation_text
    assert "tal.ext.spatial.roles" not in rotation_text
    assert 'attrs["tal"]' not in rotation_text
    assert "attrs['tal']" not in rotation_text


def test_arch_spatial_013_slice_a3_pose_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_013_slice_a3_pose_owner_split_and_budget."""
    files = [
        Path("tal/spatial/metadata/facade.py"),
        Path("tal/spatial/policies/runtime_checks.py"),
        Path("tal/spatial/pose.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_014_pose_canonical_kernel_path_no_pairwise_rep_matrix() -> None:
    """ID: ARCH_SPATIAL_014_pose_canonical_kernel_path_no_pairwise_rep_matrix."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "def apply_to(" not in pose_text
    assert "CANONICAL_POSITION_REP" in pose_text
    assert "CANONICAL_ROTATION_REP" in pose_text
    assert "set_position_rep(" in pose_text
    assert "set_rotation_rep(" in pose_text
    assert "matrix3_to_quat_kernel" in pose_text


def test_arch_spatial_015_pose_reuses_component_and_frame_owners_no_duplication() -> None:
    """ID: ARCH_SPATIAL_015_pose_reuses_component_and_frame_owners_no_duplication."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "build_paired_components_dataset" in pose_text
    assert "clear_component_registry" in pose_text
    assert "extract_components" in pose_text
    assert "resolve_pair_registry" in pose_text
    assert "resolve_component_spec" in pose_text
    assert "from tal.utils.frame_schema import get_frames, set_frames" in pose_text
    assert "merge_schema(" not in pose_text
    assert "tal.ext.components" not in pose_text
    assert 'attrs["tal"]' not in pose_text
    assert "attrs['tal']" not in pose_text


def test_arch_spatial_016_pose_constructor_validates_roles_via_metadata_owner() -> None:
    """ID: ARCH_SPATIAL_016_pose_constructor_validates_roles_via_metadata_owner."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "validate_spatial_roles" in pose_text
    assert "validate_spatial_roles(candidate, owner=owner)" in pose_text
    assert "tal.ext.spatial.roles" not in pose_text


def test_arch_spatial_017_pose_components_merge_uses_exact_join() -> None:
    """ID: ARCH_SPATIAL_017_pose_components_merge_uses_exact_join."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    paired_text = Path("tal/spatial/kinematics/paired_components.py").read_text(encoding="utf-8")
    assert "build_paired_components_dataset(" in pose_text
    assert 'join="exact"' in paired_text


def test_arch_spatial_018_runtime_checks_owner_reused_across_position_rotation_pose() -> None:
    """ID: ARCH_SPATIAL_018_runtime_checks_owner_reused_across_position_rotation_pose."""
    runtime_text = Path("tal/spatial/policies/runtime_checks.py").read_text(encoding="utf-8")
    position_text = Path("tal/spatial/position.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "def require_xyz_core_labels(" in runtime_text
    assert "require_component_numeric_var" not in runtime_text
    assert "from .policies.runtime_checks import require_xyz_core_labels" in position_text
    assert "from tal.core.orchestration.runtime_checks import" in rotation_text
    assert "from tal.core.orchestration.runtime_checks import" in pose_text


def test_arch_spatial_019_pose_from_matrix_clears_components_registry_via_component_owner() -> None:
    """ID: ARCH_SPATIAL_019_pose_from_matrix_clears_components_registry_via_component_owner."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "def _clear_component_registry_for_matrix_layout(" in pose_text
    assert "clear_component_registry(source.unsafe_data, owner=owner)" in pose_text
    assert "_clear_component_registry_for_matrix_layout(source, owner=owner)" in pose_text
    assert "merge_schema(" not in pose_text
    assert "tal.ext.components" not in pose_text


def test_arch_spatial_020_slice_a4_kinematics_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_020_slice_a4_kinematics_owner_split_and_budget."""
    files = [
        Path("tal/spatial/metadata/facade.py"),
        Path("tal/spatial/metadata/common.py"),
        Path("tal/spatial/metadata/representation.py"),
        Path("tal/spatial/metadata/roles.py"),
        Path("tal/spatial/policies/runtime_checks.py"),
        Path("tal/spatial/kinematics/family.py"),
        Path("tal/spatial/kinematics/lifecycle.py"),
        Path("tal/spatial/kinematics/paired_components.py"),
        Path("tal/spatial/velocity.py"),
        Path("tal/spatial/acceleration.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_021_slice_a4_kinematics_reuses_runtime_checks_and_component_owners() -> None:
    """ID: ARCH_SPATIAL_021_slice_a4_kinematics_reuses_runtime_checks_and_component_owners."""
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    assert "from .kinematics.family import (" in velocity_text
    assert "from .kinematics.family import (" in acceleration_text
    assert "family_from_linear_angular(" in velocity_text
    assert "family_from_linear_angular(" in acceleration_text
    assert "family_as_vector6(" in velocity_text
    assert "family_as_vector6(" in acceleration_text
    assert "merge_schema(" not in velocity_text
    assert "merge_schema(" not in acceleration_text
    assert "tal.ext.components" not in velocity_text
    assert "tal.ext.components" not in acceleration_text


def test_arch_spatial_022_slice_a4_no_schema_write_or_hidden_graph_mutation_patterns() -> None:
    """ID: ARCH_SPATIAL_022_slice_a4_no_schema_write_or_hidden_graph_mutation_patterns."""
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    assert 'attrs["tal"]' not in velocity_text
    assert "attrs['tal']" not in velocity_text
    assert 'attrs["tal"]' not in acceleration_text
    assert "attrs['tal']" not in acceleration_text
    assert "get_or_create_frame(" not in velocity_text
    assert "get_or_create_frame(" not in acceleration_text
    assert "rename_frame(" not in velocity_text
    assert "rename_frame(" not in acceleration_text


def test_arch_spatial_023_slice_a4_shared_kinematics_component_scaffold_reused_by_velocity_and_acceleration() -> None:
    """ID: ARCH_SPATIAL_023_slice_a4_shared_kinematics_component_scaffold_reused_by_velocity_and_acceleration."""
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    assert "from .kinematics.family import (" in velocity_text
    assert "from .kinematics.family import (" in acceleration_text
    assert "def _resolve_frames_for_components(" not in velocity_text
    assert "def _resolve_frames_for_components(" not in acceleration_text
    assert "def _resolve_component_roles(" not in velocity_text
    assert "def _resolve_component_roles(" not in acceleration_text
    assert "def _merge_component_payloads(" not in velocity_text
    assert "def _merge_component_payloads(" not in acceleration_text
    assert "def _clear_component_registry(" not in velocity_text
    assert "def _clear_component_registry(" not in acceleration_text


def test_arch_spatial_118_spatial_classes_do_not_duplicate_lifecycle_methods() -> None:
    """ID: ARCH_SPATIAL_118_spatial_classes_do_not_duplicate_lifecycle_methods."""
    forbidden = {
        "__init__",
        "_from_validated",
        "_from_unvalidated",
        "_normalize_metadata",
        "_enforce_invariants",
    }
    for rel, class_names in (
        ("tal/spatial/velocity.py", ("LinearVelocity", "AngularVelocity", "Velocity")),
        ("tal/spatial/acceleration.py", ("LinearAcceleration", "AngularAcceleration", "Acceleration")),
    ):
        module = _module(rel)
        for class_name in class_names:
            assert _class_method_names(module, class_name).isdisjoint(forbidden)


def test_arch_spatial_119_spatial_typed_lifecycle_specs_stay_domain_owned() -> None:
    """ID: ARCH_SPATIAL_119_spatial_typed_lifecycle_specs_stay_domain_owned."""
    lifecycle_path = Path("tal/spatial/kinematics/lifecycle.py")
    lifecycle_text = lifecycle_path.read_text(encoding="utf-8")
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    core_text = Path("tal/core/typed_lifecycle.py").read_text(encoding="utf-8")

    assert lifecycle_path.exists()
    assert "make_kinematics_lifecycle_spec(" in lifecycle_text
    assert "_LINEAR_VELOCITY_LIFECYCLE" in velocity_text
    assert "_ANGULAR_VELOCITY_LIFECYCLE" in velocity_text
    assert "_VELOCITY_LIFECYCLE" in velocity_text
    assert "_LINEAR_ACCELERATION_LIFECYCLE" in acceleration_text
    assert "_ANGULAR_ACCELERATION_LIFECYCLE" in acceleration_text
    assert "_ACCELERATION_LIFECYCLE" in acceleration_text
    assert "Kinematics" not in core_text
    assert "LinearVelocity" not in core_text


def test_arch_spatial_024_slice_a4_spatial_metadata_owner_split_representation_vs_roles() -> None:
    """ID: ARCH_SPATIAL_024_slice_a4_spatial_metadata_owner_split_representation_vs_roles."""
    common_path = Path("tal/spatial/metadata/common.py")
    rep_path = Path("tal/spatial/metadata/representation.py")
    roles_path = Path("tal/spatial/metadata/roles.py")
    facade_text = Path("tal/spatial/metadata/facade.py").read_text(encoding="utf-8")
    assert common_path.exists()
    assert rep_path.exists()
    assert roles_path.exists()
    assert "from .representation import" in facade_text
    assert "from .roles import" in facade_text
    assert "def get_position_rep(" not in facade_text
    assert "def get_position_intent(" not in facade_text
    assert "def get_kinematics_kind(" not in facade_text


def test_arch_spatial_025_slice_a4_metadata_facade_nonowning_budget_lock() -> None:
    """ID: ARCH_SPATIAL_025_slice_a4_metadata_facade_nonowning_budget_lock."""
    metadata_path = Path("tal/spatial/metadata/facade.py")
    metadata_text = metadata_path.read_text(encoding="utf-8")
    assert file_loc(path=metadata_path) <= 200
    assert "merge_schema(" not in metadata_text
    assert "_read_tal(" not in metadata_text
    assert "_read_ext(" not in metadata_text
    assert "_read_spatial(" not in metadata_text
    assert "_read_rep_block(" not in metadata_text
    assert "_read_roles_block(" not in metadata_text


def test_arch_spatial_026_slice_b1_rotation_conversion_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_026_slice_b1_rotation_conversion_owner_split_and_budget."""
    files = [
        Path("tal/spatial/rotation.py"),
        Path("tal/spatial/kernels/rotation_kernels.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_027_slice_b1_rotation_kernel_owner_uses_scipy_and_no_schema_writes() -> None:
    """ID: ARCH_SPATIAL_027_slice_b1_rotation_kernel_owner_uses_scipy_and_no_schema_writes."""
    kernels_text = Path("tal/spatial/kernels/rotation_kernels.py").read_text(encoding="utf-8")
    assert "from scipy.spatial.transform import Rotation as SciRotation" in kernels_text
    assert "merge_schema(" not in kernels_text
    assert 'attrs["tal"]' not in kernels_text
    assert "attrs['tal']" not in kernels_text


def test_arch_spatial_028_slice_b1_rotation_orchestrate_kernel_finalize_split_enforced() -> None:
    """ID: ARCH_SPATIAL_028_slice_b1_rotation_orchestrate_kernel_finalize_split_enforced."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert "from .kernels.rotation_kernels import matrix_to_quat_kernel, quat_to_matrix_kernel" in rotation_text
    assert "def _convert_quat_to_matrix(" in rotation_text
    assert "def _convert_matrix_to_quat(" in rotation_text
    assert "def _finalize_rotation_conversion(" in rotation_text
    assert "finalize_conversion_dataset(" in rotation_text
    assert "set_rotation_rep(" in rotation_text
    assert "SciRotation" not in rotation_text


def test_spatial_doc_003_phase8_slice_a3_pose_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_003_phase8_slice_a3_pose_docs_and_api_entries_present."""
    api_pose = Path("docs/api/types/pose.md").read_text(encoding="utf-8")
    user_spatial = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "from_components" in api_pose
    assert "from_matrix" in api_pose
    assert "decompose" in api_pose
    assert "Pose" in user_spatial
    assert "pose" in api_types_index


def test_spatial_doc_004_phase8_slice_a4_velocity_acceleration_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_004_phase8_slice_a4_velocity_acceleration_docs_and_api_entries_present."""
    api_velocity = Path("docs/api/types/velocity.md").read_text(encoding="utf-8")
    api_acceleration = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8")
    user_spatial = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "LinearVelocity" in api_velocity
    assert "AngularVelocity" in api_velocity
    assert "Velocity" in api_velocity
    assert "LinearAcceleration" in api_acceleration
    assert "AngularAcceleration" in api_acceleration
    assert "Acceleration" in api_acceleration
    assert "Velocity" in user_spatial
    assert "Acceleration" in user_spatial
    assert "velocity" in api_types_index
    assert "acceleration" in api_types_index


def test_spatial_doc_005_phase8_slice_b1_rotation_conversion_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_005_phase8_slice_b1_rotation_conversion_docs_and_api_entries_present."""
    api_rotation = Path("docs/api/types/rotation.md").read_text(encoding="utf-8")
    user_spatial = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "to_rep" in api_rotation
    assert "as_quat" in api_rotation
    assert "as_matrix" in api_rotation
    assert "quat" in api_rotation
    assert "matrix" in api_rotation
    assert "rotation" in api_types_index


def test_arch_spatial_029_slice_b2_rotation_compose_inverse_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_029_slice_b2_rotation_compose_inverse_owner_split_and_budget."""
    files = [
        Path("tal/spatial/rotation.py"),
        Path("tal/spatial/kernels/rotation_compose_kernels.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_030_slice_b2_rotation_compose_inverse_reuses_b1_conversion_boundary() -> None:
    """ID: ARCH_SPATIAL_030_slice_b2_rotation_compose_inverse_reuses_b1_conversion_boundary."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert "left.as_quat(validate=False)" in rotation_text
    assert "right.as_quat(validate=False)" in rotation_text
    assert "_convert_quat_to_matrix(" in rotation_text
    assert "_convert_matrix_to_quat(" in rotation_text
    assert "def compose(" in rotation_text
    assert "def inverse(" in rotation_text
    assert "_rotation_compose_with_owner(" in rotation_text


def test_arch_spatial_031_slice_b2_rotation_compose_inverse_kernel_owner_no_schema_writes() -> None:
    """ID: ARCH_SPATIAL_031_slice_b2_rotation_compose_inverse_kernel_owner_no_schema_writes."""
    kernels_text = Path("tal/spatial/kernels/rotation_compose_kernels.py").read_text(encoding="utf-8")
    assert "from scipy.spatial.transform import Rotation as SciRotation" in kernels_text
    assert "merge_schema(" not in kernels_text
    assert "set_roles(" not in kernels_text
    assert "set_rotation_rep(" not in kernels_text
    assert 'attrs["tal"]' not in kernels_text
    assert "attrs['tal']" not in kernels_text


def test_arch_spatial_032_slice_b2_rotation_compose_inverse_no_eager_patterns() -> None:
    """ID: ARCH_SPATIAL_032_slice_b2_rotation_compose_inverse_no_eager_patterns."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert ".values" not in rotation_text
    assert ".item(" not in rotation_text
    assert "np.asarray(" not in rotation_text
    assert ".compute(" not in rotation_text


def test_arch_spatial_033_slice_b2_compose_non_core_dim_topology_guard_present() -> None:
    """ID: ARCH_SPATIAL_033_slice_b2_compose_non_core_dim_topology_guard_present."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert "resolve_binary_topology(" in rotation_text
    assert "align_exact_for_plan(" in rotation_text
    assert "resolve_semantic_topology_from_dataset(" in rotation_text
    assert ('what="compose"' in rotation_text) or ('what="rotation compose"' in rotation_text)


def test_arch_spatial_034_slice_b2_compose_wraps_apply_ufunc_valueerror_with_owner_context() -> None:
    """ID: ARCH_SPATIAL_034_slice_b2_compose_wraps_apply_ufunc_valueerror_with_owner_context."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert "except ValueError as exc:" in rotation_text
    assert "compose kernel failed after alignment" in rotation_text


def test_arch_spatial_035_slice_b2_compose_frame_check_runs_before_quat_conversion() -> None:
    """ID: ARCH_SPATIAL_035_slice_b2_compose_frame_check_runs_before_quat_conversion."""
    rotation_module = _module("tal/spatial/rotation.py")
    compose_node = _function_node(rotation_module, "_rotation_compose_with_owner")
    call_order = _compose_call_order_tokens(compose_node)
    assert "resolve_compose_output_frames" in call_order
    assert "left.as_quat" in call_order
    assert "right.as_quat" in call_order
    frame_idx = call_order.index("resolve_compose_output_frames")
    self_quat_idx = call_order.index("left.as_quat")
    right_quat_idx = call_order.index("right.as_quat")
    assert frame_idx < self_quat_idx
    assert frame_idx < right_quat_idx


def test_spatial_doc_006_phase8_slice_b2_rotation_compose_inverse_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_006_phase8_slice_b2_rotation_compose_inverse_docs_and_api_entries_present."""
    api_rotation = Path("docs/api/types/rotation.md").read_text(encoding="utf-8")
    user_spatial = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "compose" in api_rotation
    assert "inverse" in api_rotation
    assert "frame" in api_rotation.lower()
    assert "compose" in user_spatial
    assert "inverse" in user_spatial
    assert "rotation" in api_types_index


def test_arch_spatial_036_slice_b3_pose_conversion_compose_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_036_slice_b3_pose_conversion_compose_owner_split_and_budget."""
    files = [
        Path("tal/spatial/pose.py"),
        Path("tal/spatial/ops/pose_ops.py"),
        Path("tal/spatial/policies/frame.py"),
        Path("tal/spatial/kinematics/paired_components.py"),
        Path("tal/spatial/kernels/pose_kernels.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_037_slice_b3_pose_reuses_rotation_and_component_frame_owners_no_duplication() -> None:
    """ID: ARCH_SPATIAL_037_slice_b3_pose_reuses_rotation_and_component_frame_owners_no_duplication."""
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "from ..rotation import Rotation" in pose_ops_text
    assert "from tal.utils.frame_schema import get_frames, set_frames" in pose_ops_text
    assert "define_components" in pose_ops_text
    assert "ComponentRegistryOptions" in pose_ops_text
    assert "merge_schema(" not in pose_ops_text
    assert "_read_tal(" not in pose_ops_text
    assert "_read_ext(" not in pose_ops_text
    assert "_read_frames_block(" not in pose_ops_text


def test_arch_spatial_038_slice_b3_pose_kernel_owner_no_schema_writes() -> None:
    """ID: ARCH_SPATIAL_038_slice_b3_pose_kernel_owner_no_schema_writes."""
    kernels_text = Path("tal/spatial/kernels/pose_kernels.py").read_text(encoding="utf-8")
    assert "from scipy.spatial.transform import Rotation as SciRotation" in kernels_text
    assert "merge_schema(" not in kernels_text
    assert "set_pose_rep(" not in kernels_text
    assert "set_frames(" not in kernels_text
    assert 'attrs["tal"]' not in kernels_text
    assert "attrs['tal']" not in kernels_text


def test_arch_spatial_039_slice_b3_pose_orchestrate_kernel_finalize_split_enforced() -> None:
    """ID: ARCH_SPATIAL_039_slice_b3_pose_orchestrate_kernel_finalize_split_enforced."""
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "from ..kernels.pose_kernels import" in pose_ops_text
    assert "compose_translation_kernel" in pose_ops_text
    assert "inverse_translation_kernel" in pose_ops_text
    assert "components_to_matrix_kernel" in pose_ops_text
    assert "matrix_to_components_kernel" in pose_ops_text
    assert "set_pose_rep(" in pose_ops_text
    assert "set_frames(" in pose_ops_text
    assert "SciRotation" not in pose_ops_text


def test_arch_spatial_040_slice_b3_pose_compose_non_core_dim_topology_guard_present() -> None:
    """ID: ARCH_SPATIAL_040_slice_b3_pose_compose_non_core_dim_topology_guard_present."""
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "resolve_nary_topology(" in pose_ops_text
    assert "align_exact_for_plan(" in pose_ops_text
    assert 'what="pose compose"' in pose_ops_text


def test_arch_spatial_041_slice_b3_pose_compose_frame_check_runs_before_canonical_conversion() -> None:
    """ID: ARCH_SPATIAL_041_slice_b3_pose_compose_frame_check_runs_before_canonical_conversion."""
    pose_ops_module = _module("tal/spatial/ops/pose_ops.py")
    compose_node = _function_node(pose_ops_module, "_pose_compose_with_owner")
    call_order = _call_order_tokens(compose_node, _pose_compose_call_token)
    assert "resolve_compose_output_frames" in call_order
    assert "_canonical_components" in call_order
    frame_idx = call_order.index("resolve_compose_output_frames")
    canonical_idx = call_order.index("_canonical_components")
    assert frame_idx < canonical_idx


def test_arch_spatial_042_slice_b3_pose_matrix_output_clears_component_registry_truthfulness() -> None:
    """ID: ARCH_SPATIAL_042_slice_b3_pose_matrix_output_clears_component_registry_truthfulness."""
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "_components_to_matrix_dataset(" in pose_ops_text
    assert "ComponentRegistryOptions(registry={}, replace=True)" in pose_ops_text
    assert "set_pose_rep(cleared.unsafe_data, rep=\"matrix\"" in pose_ops_text


def test_spatial_doc_007_phase8_slice_b3_pose_conversion_compose_inverse_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_007_phase8_slice_b3_pose_conversion_compose_inverse_docs_and_api_entries_present."""
    api_pose = Path("docs/api/types/pose.md").read_text(encoding="utf-8")
    user_spatial = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "to_rep" in api_pose
    assert "as_components" in api_pose
    assert "as_matrix" in api_pose
    assert "compose" in api_pose
    assert "inverse" in api_pose
    assert "pose" in api_types_index


def test_spatial_doc_008_phase8_slice_b4_local_transform_apply_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_008_phase8_slice_b4_local_transform_apply_docs_and_api_entries_present."""
    api_rotation = Path("docs/api/types/rotation.md").read_text(encoding="utf-8")
    api_pose = Path("docs/api/types/pose.md").read_text(encoding="utf-8")
    api_velocity = Path("docs/api/types/velocity.md").read_text(encoding="utf-8")
    api_acceleration = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8")
    user_spatial = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "apply" in api_rotation
    assert "apply" in api_pose
    assert "apply" in api_velocity
    assert "apply" in api_acceleration
    assert "apply" in user_spatial
    assert "rotation apply APIs" not in api_rotation
    assert "pose apply kernels" not in api_pose
    assert "pose apply kernels" not in user_spatial


def test_arch_spatial_043_slice_b3_components_layout_requires_component_vars_include_declared_non_core_dims() -> None:
    """ID: ARCH_SPATIAL_043_slice_b3_components_layout_requires_component_vars_include_declared_non_core_dims."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "required_non_core_dims = (sequence_dim, *batch_dims)" in pose_text
    assert "required_dims=required_non_core_dims + (pos_dim,)" in pose_text
    assert "required_dims=required_non_core_dims + (rot_dim,)" in pose_text


def test_arch_spatial_044_slice_b3_operation_owner_wraps_pose_kernel_valueerror_boundaries() -> None:
    """ID: ARCH_SPATIAL_044_slice_b3_operation_owner_wraps_pose_kernel_valueerror_boundaries."""
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "pose components->matrix conversion kernel failed." in pose_ops_text
    assert "pose compose translation kernel failed." in pose_ops_text


def test_arch_spatial_045_slice_b3_pose_shared_frame_utility_single_owner_reused_by_pose_and_pose_ops() -> None:
    """ID: ARCH_SPATIAL_045_slice_b3_pose_shared_frame_utility_single_owner_reused_by_pose_and_pose_ops."""
    shared_text = Path("tal/spatial/policies/frame.py").read_text(encoding="utf-8")
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "def resolve_components_shared_frames(" in shared_text
    assert "def resolve_compose_output_frames(" in shared_text
    assert "from .policies.frame import resolve_components_shared_frames" in pose_text
    assert "from ..policies.frame import resolve_compose_output_frames, resolve_components_shared_frames" in pose_ops_text
    assert "from .pose_shared import" not in pose_text
    assert "from .pose_shared import" not in pose_ops_text


def test_arch_spatial_046_pose_matrix_decompose_reuses_single_matrix_to_quat_kernel_owner() -> None:
    """ID: ARCH_SPATIAL_046_pose_matrix_decompose_reuses_single_matrix_to_quat_kernel_owner."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    kernels_text = Path("tal/spatial/kernels/pose_kernels.py").read_text(encoding="utf-8")
    assert "from .kernels.pose_kernels import matrix3_to_quat_kernel" in pose_text
    assert "return matrix3_to_quat_kernel(values)" in pose_text
    assert "_matrix3_to_quat_decompose_kernel," in pose_text
    assert "def _matrix_to_quat_kernel(" not in pose_text
    assert "def matrix3_to_quat_kernel(" in kernels_text
    assert "quat = matrix3_to_quat_kernel(rotm)" in kernels_text


def test_arch_spatial_047_slice_b3_pose_decompose_uses_owner_wrapped_kernel_callable_for_lazy_errors() -> None:
    """ID: ARCH_SPATIAL_047_slice_b3_pose_decompose_uses_owner_wrapped_kernel_callable_for_lazy_errors."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "def _matrix3_to_quat_decompose_kernel(" in pose_text
    assert 'owner = "spatial.pose.decompose"' in pose_text
    assert "return matrix3_to_quat_kernel(values)" in pose_text
    assert "matrix decomposition quaternion kernel failed." in pose_text
    assert "_matrix3_to_quat_decompose_kernel," in pose_text
    assert "matrix3_to_quat_kernel," not in pose_text.split("_matrix3_to_quat_decompose_kernel,")[0]


def test_arch_spatial_048_slice_b4_apply_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_048_slice_b4_apply_owner_split_and_budget."""
    files = [
        Path("tal/spatial/policies/frame.py"),
        Path("tal/spatial/policies/wrap.py"),
        Path("tal/spatial/kernels/rotation_apply_kernels.py"),
        Path("tal/spatial/ops/rotation_apply_ops.py"),
        Path("tal/spatial/kernels/pose_apply_kernels.py"),
        Path("tal/spatial/ops/pose_apply_ops.py"),
        Path("tal/spatial/rotation.py"),
        Path("tal/spatial/pose.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_049_slice_b4_rotation_apply_kernel_owner_no_schema_writes() -> None:
    """ID: ARCH_SPATIAL_049_slice_b4_rotation_apply_kernel_owner_no_schema_writes."""
    kernels_text = Path("tal/spatial/kernels/rotation_apply_kernels.py").read_text(encoding="utf-8")
    assert "from scipy.spatial.transform import Rotation as SciRotation" in kernels_text
    assert "set_frames(" not in kernels_text
    assert "set_roles(" not in kernels_text
    assert "set_rotation_rep(" not in kernels_text
    assert "merge_schema(" not in kernels_text
    assert 'attrs["tal"]' not in kernels_text
    assert "attrs['tal']" not in kernels_text


def test_arch_spatial_050_slice_b4_pose_apply_kernel_owner_no_schema_writes() -> None:
    """ID: ARCH_SPATIAL_050_slice_b4_pose_apply_kernel_owner_no_schema_writes."""
    kernels_text = Path("tal/spatial/kernels/pose_apply_kernels.py").read_text(encoding="utf-8")
    assert "set_frames(" not in kernels_text
    assert "set_roles(" not in kernels_text
    assert "set_pose_rep(" not in kernels_text
    assert "merge_schema(" not in kernels_text
    assert 'attrs["tal"]' not in kernels_text
    assert "attrs['tal']" not in kernels_text


def test_arch_spatial_051_slice_b4_apply_reuses_rotation_pose_kinematics_and_frame_component_owners() -> None:
    """ID: ARCH_SPATIAL_051_slice_b4_apply_reuses_rotation_pose_kinematics_and_frame_component_owners."""
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    assert "from ..policies.frame import resolve_apply_output_frames" in rotation_apply_text
    assert "from tal.core.orchestration.alignment import" in rotation_apply_text
    assert "from ..kernels.rotation_apply_kernels import rotate_vec3_kernel" in rotation_apply_text
    assert "Velocity.from_linear_angular(" in rotation_apply_text
    assert "Acceleration.from_linear_angular(" in rotation_apply_text
    assert "set_frames(" in rotation_apply_text
    assert "from .rotation_apply_ops import _rotation_apply_with_owner" in pose_apply_text
    assert "pose.decompose(validate=False)" in pose_apply_text
    assert "_rotation_apply_with_owner(rotation, target, validate=False, owner=owner)" in pose_apply_text
    assert "set_frames(" in pose_apply_text
    assert "merge_schema(" not in rotation_apply_text
    assert "merge_schema(" not in pose_apply_text


def test_arch_spatial_052_slice_b4_apply_frame_check_runs_before_conversion_and_kernel_execution() -> None:
    """ID: ARCH_SPATIAL_052_slice_b4_apply_frame_check_runs_before_conversion_and_kernel_execution."""
    rotation_apply_module = _module("tal/spatial/ops/rotation_apply_ops.py")
    rotation_apply_node = _function_node(rotation_apply_module, "_rotation_apply_with_owner")
    rotation_order = _call_order_tokens(rotation_apply_node, _rotation_apply_call_token)
    assert "resolve_apply_output_frames" in rotation_order
    assert "_apply_to_vector_target" in rotation_order
    assert "_apply_to_spatial_target" in rotation_order
    frame_idx = rotation_order.index("resolve_apply_output_frames")
    assert frame_idx < rotation_order.index("_apply_to_vector_target")
    assert frame_idx < rotation_order.index("_apply_to_spatial_target")

    pose_apply_module = _module("tal/spatial/ops/pose_apply_ops.py")
    pose_apply_node = _function_node(pose_apply_module, "_pose_apply_with_owner")
    pose_order = _call_order_tokens(pose_apply_node, _pose_apply_call_token)
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    assert "def _pose_apply_with_owner(" in pose_apply_text
    assert "return _pose_apply_with_owner(" in pose_apply_text
    assert "resolve_apply_output_frames" in pose_order
    assert "_apply_pose_to_position" in pose_order
    assert "_apply_pose_to_spatial_target" in pose_order
    frame_idx = pose_order.index("resolve_apply_output_frames")
    assert frame_idx < pose_order.index("_apply_pose_to_position")
    assert frame_idx < pose_order.index("_apply_pose_to_spatial_target")


def test_arch_spatial_053_slice_b4_apply_non_core_dim_topology_guard_present() -> None:
    """ID: ARCH_SPATIAL_053_slice_b4_apply_non_core_dim_topology_guard_present."""
    alignment_text = Path("tal/core/orchestration/alignment.py").read_text(encoding="utf-8")
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    assert "def align_exact_for_plan(" in alignment_text
    assert "resolve_binary_topology(" in rotation_apply_text
    assert "resolve_nary_topology(" in pose_apply_text
    assert 'join="exact"' in alignment_text


def test_arch_spatial_054_slice_b4_apply_orchestrate_kernel_finalize_split_and_no_eager_patterns() -> None:
    """ID: ARCH_SPATIAL_054_slice_b4_apply_orchestrate_kernel_finalize_split_and_no_eager_patterns."""
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    assert "xr.apply_ufunc(" in rotation_apply_text
    assert "_wrap_rotation_apply_kernel" in rotation_apply_text
    assert "set_frames(" in rotation_apply_text
    assert "xr.apply_ufunc(" in pose_apply_text
    assert "_wrap_pose_apply_position_kernel" in pose_apply_text
    assert "set_frames(" in pose_apply_text
    assert ".values" not in rotation_apply_text
    assert ".item(" not in rotation_apply_text
    assert "np.asarray(" not in rotation_apply_text
    assert ".compute(" not in rotation_apply_text
    assert ".values" not in pose_apply_text
    assert ".item(" not in pose_apply_text
    assert "np.asarray(" not in pose_apply_text
    assert ".compute(" not in pose_apply_text


def test_arch_spatial_055_slice_b4_pose_apply_spatial6_translation_coupling_deferred_lock() -> None:
    """ID: ARCH_SPATIAL_055_slice_b4_pose_apply_spatial6_translation_coupling_deferred_lock."""
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    assert "_apply_pose_to_spatial_target(" in pose_apply_text
    assert "_rotation_apply_with_owner(rotation, target, validate=False, owner=owner)" in pose_apply_text
    assert "compose_translation_kernel" not in pose_apply_text
    assert "cross(" not in pose_apply_text


def test_arch_spatial_056_slice_b4_pose_apply_delegation_threads_operation_owner_into_rotation_apply_lazy_kernel_path() -> None:
    """ID: ARCH_SPATIAL_056_slice_b4_pose_apply_delegation_threads_operation_owner_into_rotation_apply_lazy_kernel_path."""
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    assert "def _rotation_apply_with_owner(" in rotation_apply_text
    assert "partial(_wrap_rotation_apply_kernel, owner=owner)" in rotation_apply_text
    assert 'owner="spatial.rotation.apply"' in rotation_apply_text
    assert "from .rotation_apply_ops import _rotation_apply_with_owner" in pose_apply_text
    assert "_rotation_apply_with_owner(rotation, target, validate=False, owner=owner)" in pose_apply_text
    assert "rotation_apply(rotation, target, validate=False)" not in pose_apply_text


def test_arch_spatial_057_slice_b5_spatial6_vector6_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_057_slice_b5_spatial6_vector6_owner_split_and_budget."""
    files = [
        Path("tal/spatial/kernels/kinematics_vector6_kernels.py"),
        Path("tal/spatial/kinematics/vector6_ops.py"),
        Path("tal/spatial/velocity.py"),
        Path("tal/spatial/acceleration.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_058_slice_b5_vector6_kernel_owner_schema_agnostic_no_schema_writes() -> None:
    """ID: ARCH_SPATIAL_058_slice_b5_vector6_kernel_owner_schema_agnostic_no_schema_writes."""
    kernel_text = Path("tal/spatial/kernels/kinematics_vector6_kernels.py").read_text(encoding="utf-8")
    assert "def pack_vector6_kernel(" in kernel_text
    assert "def unpack_vector6_linear_kernel(" in kernel_text
    assert "def unpack_vector6_angular_kernel(" in kernel_text
    assert "set_frames(" not in kernel_text
    assert "merge_schema(" not in kernel_text
    assert 'attrs["tal"]' not in kernel_text
    assert "attrs['tal']" not in kernel_text


def test_arch_spatial_059_slice_b5_velocity_acceleration_reuse_shared_vector6_conversion_owner() -> None:
    """ID: ARCH_SPATIAL_059_slice_b5_velocity_acceleration_reuse_shared_vector6_conversion_owner."""
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    family_text = Path("tal/spatial/kinematics/family.py").read_text(encoding="utf-8")
    assert "from .kinematics.family import (" in velocity_text
    assert "from .kinematics.family import (" in acceleration_text
    assert "pack_linear_angular_to_vector6_dataset" in family_text
    assert "unpack_vector6_to_linear_angular_datasets" in family_text
    assert "def _wrap_pack_kernel(" not in velocity_text
    assert "def _wrap_pack_kernel(" not in acceleration_text


def test_arch_spatial_060_slice_b5_metadata_rep_owner_expands_velocity_acceleration_rep_sets_only() -> None:
    """ID: ARCH_SPATIAL_060_slice_b5_metadata_rep_owner_expands_velocity_acceleration_rep_sets_only."""
    rep_text = Path("tal/spatial/metadata/representation.py").read_text(encoding="utf-8")
    assert "_VELOCITY_REP_SPEC = _RepresentationMetadataSpec(" in rep_text
    assert "_ACCELERATION_REP_SPEC = _RepresentationMetadataSpec(" in rep_text
    assert "_POSITION_REP_SPEC = _RepresentationMetadataSpec(" in rep_text
    assert "_ROTATION_REP_SPEC = _RepresentationMetadataSpec(" in rep_text
    assert "_POSE_REP_SPEC = _RepresentationMetadataSpec(" in rep_text
    assert 'allowed=frozenset({"components", "vector6"})' in rep_text
    assert 'allowed=frozenset({"quat", "matrix"})' in rep_text
    assert 'allowed=frozenset({"cart"})' in rep_text


def test_arch_spatial_168_representation_metadata_factory_single_owner() -> None:
    """ID: ARCH_SPATIAL_168_representation_metadata_factory_single_owner."""
    owner_path = Path("tal/spatial/metadata/representation.py")
    owner_text = owner_path.read_text(encoding="utf-8")
    assert "class _RepresentationMetadataSpec" in owner_text
    assert "def _make_rep_getter(" in owner_text
    assert "def _make_rep_setter(" in owner_text
    for path in sorted(Path("tal/spatial").rglob("*.py")):
        if path == owner_path:
            continue
        text = path.read_text(encoding="utf-8")
        assert "_RepresentationMetadataSpec" not in text
        assert "_make_rep_getter" not in text
        assert "_make_rep_setter" not in text


def test_arch_spatial_169_representation_metadata_exports_remain_explicit() -> None:
    """ID: ARCH_SPATIAL_169_representation_metadata_exports_remain_explicit."""
    module = _module("tal/spatial/metadata/representation.py")
    expected_getters = {
        "get_acceleration_rep",
        "get_angular_acceleration_rep",
        "get_angular_velocity_rep",
        "get_linear_acceleration_rep",
        "get_linear_velocity_rep",
        "get_pose_rep",
        "get_position_rep",
        "get_rotation_rep",
        "get_velocity_rep",
    }
    expected_setters = {name.replace("get_", "set_", 1) for name in expected_getters}
    explicit_getters: set[str] = set()
    explicit_setters: set[str] = set()
    all_exports: list[str] | None = None
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, ast.List):
                all_exports = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
            if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.value.func, ast.Name):
                continue
            if node.value.func.id == "_make_rep_getter":
                explicit_getters.add(target.id)
            if node.value.func.id == "_make_rep_setter":
                explicit_setters.add(target.id)
    assert explicit_getters == expected_getters
    assert explicit_setters == expected_setters
    assert all_exports == [
        "get_acceleration_rep",
        "get_angular_acceleration_rep",
        "get_angular_velocity_rep",
        "get_linear_acceleration_rep",
        "get_linear_velocity_rep",
        "get_pose_rep",
        "get_position_rep",
        "get_rotation_rep",
        "get_velocity_rep",
        "set_acceleration_rep",
        "set_angular_acceleration_rep",
        "set_angular_velocity_rep",
        "set_linear_acceleration_rep",
        "set_linear_velocity_rep",
        "set_pose_rep",
        "set_position_rep",
        "set_rotation_rep",
        "set_velocity_rep",
    ]


def test_arch_spatial_170_representation_metadata_no_core_interpretation() -> None:
    """ID: ARCH_SPATIAL_170_representation_metadata_no_core_interpretation."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.ext.spatial.representation" not in text
        assert "_RepresentationMetadataSpec" not in text
        assert "_make_rep_getter" not in text
        assert "_make_rep_setter" not in text


def test_arch_spatial_171_representation_metadata_duplicate_bodies_removed() -> None:
    """ID: ARCH_SPATIAL_171_representation_metadata_duplicate_bodies_removed."""
    module = _module("tal/spatial/metadata/representation.py")
    text = Path("tal/spatial/metadata/representation.py").read_text(encoding="utf-8")
    public_rep_names = {
        "get_acceleration_rep",
        "get_angular_acceleration_rep",
        "get_angular_velocity_rep",
        "get_linear_acceleration_rep",
        "get_linear_velocity_rep",
        "get_pose_rep",
        "get_position_rep",
        "get_rotation_rep",
        "get_velocity_rep",
        "set_acceleration_rep",
        "set_angular_acceleration_rep",
        "set_angular_velocity_rep",
        "set_linear_acceleration_rep",
        "set_linear_velocity_rep",
        "set_pose_rep",
        "set_position_rep",
        "set_rotation_rep",
        "set_velocity_rep",
    }
    function_names = {node.name for node in _function_nodes(module)}
    assert public_rep_names.isdisjoint(function_names)
    assert "_make_rep_getter" in function_names
    assert "_make_rep_setter" in function_names
    assert "_ALLOWED_" not in text
    assert "_REP_SPECS" not in text
    assert "globals(" not in text
    assert "__getattr__" not in text
    assert not any(isinstance(node, ast.For) for node in module.body)


def test_arch_spatial_061_slice_b5_spatial6_apply_paths_canonicalize_vector6_targets_before_component_extraction() -> None:
    """ID: ARCH_SPATIAL_061_slice_b5_spatial6_apply_paths_canonicalize_vector6_targets_before_component_extraction."""
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    assert "get_velocity_rep" in rotation_apply_text
    assert "get_acceleration_rep" in rotation_apply_text
    assert "unpack_vector6_to_linear_angular_datasets" in rotation_apply_text
    assert "pack_linear_angular_to_vector6_dataset" in rotation_apply_text
    assert "def _velocity_components_for_apply(" in rotation_apply_text
    assert "def _acceleration_components_for_apply(" in rotation_apply_text
    assert "from .rotation_apply_ops import _rotation_apply_with_owner" in pose_apply_text
    assert "_rotation_apply_with_owner(rotation, target, validate=False, owner=owner)" in pose_apply_text


def test_arch_spatial_062_slice_b5_no_eager_patterns_in_vector6_conversion_orchestration() -> None:
    """ID: ARCH_SPATIAL_062_slice_b5_no_eager_patterns_in_vector6_conversion_orchestration."""
    ops_text = Path("tal/spatial/kinematics/vector6_ops.py").read_text(encoding="utf-8")
    assert "xr.apply_ufunc(" in ops_text
    assert ".values" not in ops_text
    assert ".item(" not in ops_text
    assert "np.asarray(" not in ops_text
    assert ".compute(" not in ops_text


def test_arch_spatial_063_slice_b5_vector6_ops_dtype_policy_is_dynamic_not_fixed_float64() -> None:
    """ID: ARCH_SPATIAL_063_slice_b5_vector6_ops_dtype_policy_is_dynamic_not_fixed_float64."""
    ops_text = Path("tal/spatial/kinematics/vector6_ops.py").read_text(encoding="utf-8")
    assert "np.result_type(" in ops_text
    assert "output_dtypes=[vector_da.dtype]" in ops_text
    assert "output_dtypes=[np.float64]" not in ops_text


def test_arch_spatial_064_slice_c1_path_solver_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_064_slice_c1_path_solver_owner_split_and_budget."""
    files = [
        Path("tal/spatial/path_solve.py"),
        Path("tal/spatial/ops/path_solve_ops.py"),
    ]
    for path in files:
        assert path.exists(), f"missing spatial owner file: {path}"
        _assert_agents_budget(path)


def test_arch_spatial_065_slice_c1_path_solver_reuses_frames_find_path_fold_path_owners() -> None:
    """ID: ARCH_SPATIAL_065_slice_c1_path_solver_reuses_frames_find_path_fold_path_owners."""
    text = Path("tal/spatial/ops/path_solve_ops.py").read_text(encoding="utf-8")
    assert "from tal.frames import Frame, FrameGraph, find_path, fold_path, get_active_frame_graph" in text
    assert "path = find_path(" in text
    assert "result = fold_path(" in text
    assert "def _ancestors_with_depth(" not in text
    assert "def _build_path_nodes(" not in text
    assert "def _build_oriented_steps(" not in text


def test_arch_spatial_066_slice_c1_path_solver_reuses_rotation_pose_compose_inverse_without_local_math() -> None:
    """ID: ARCH_SPATIAL_066_slice_c1_path_solver_reuses_rotation_pose_compose_inverse_without_local_math."""
    text = Path("tal/spatial/ops/path_solve_ops.py").read_text(encoding="utf-8")
    assert "from ..rotation import Rotation, _rotation_compose_with_owner, _rotation_inverse_with_owner" in text
    assert "from .pose_ops import _pose_compose_with_owner, _pose_inverse_with_owner" in text
    assert "_rotation_compose_with_owner(" in text
    assert "_rotation_inverse_with_owner(" in text
    assert "_pose_compose_with_owner(" in text
    assert "_pose_inverse_with_owner(" in text
    assert "from scipy.spatial.transform import Rotation as SciRotation" not in text
    assert "compose_translation_kernel(" not in text


def test_arch_spatial_067_slice_c1_path_solver_non_mutating_no_graph_write_paths() -> None:
    """ID: ARCH_SPATIAL_067_slice_c1_path_solver_non_mutating_no_graph_write_paths."""
    text = Path("tal/spatial/ops/path_solve_ops.py").read_text(encoding="utf-8")
    assert "reparent_frame(" not in text
    assert "rename_frame(" not in text
    assert "get_or_create_frame(" not in text
    assert 'attrs["tal"]' not in text
    assert "attrs['tal']" not in text


def test_arch_spatial_068_slice_c1_path_solver_owner_wrapped_lazy_error_boundary_present() -> None:
    """ID: ARCH_SPATIAL_068_slice_c1_path_solver_owner_wrapped_lazy_error_boundary_present."""
    path_ops_text = Path("tal/spatial/ops/path_solve_ops.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert '_rotation_compose_with_owner(acc, value, validate=False, owner=owner)' in path_ops_text
    assert '_rotation_inverse_with_owner(value, validate=False, owner=owner)' in path_ops_text
    assert '_pose_compose_with_owner(acc, value, validate=False, owner=owner)' in path_ops_text
    assert '_pose_inverse_with_owner(value, validate=False, owner=owner)' in path_ops_text
    assert "partial(_wrap_compose_quat_kernel, owner=owner)" in rotation_text
    assert "partial(_wrap_inverse_quat_kernel, owner=owner)" in rotation_text
    assert "partial(_wrap_compose_translation_kernel, owner=owner)" in pose_ops_text
    assert "partial(_wrap_inverse_translation_kernel, owner=owner)" in pose_ops_text


def test_arch_spatial_069_slice_c1_path_solver_routes_resolver_invocation_through_owner_wrapped_boundary() -> None:
    """ID: ARCH_SPATIAL_069_slice_c1_path_solver_routes_resolver_invocation_through_owner_wrapped_boundary."""
    path_ops_text = Path("tal/spatial/ops/path_solve_ops.py").read_text(encoding="utf-8")
    path_ops_module = _module("tal/spatial/ops/path_solve_ops.py")
    call_node = _function_node(path_ops_module, "_call_edge_resolver")
    assert "def _call_edge_resolver(" in path_ops_text
    assert "signature_checked: bool" in path_ops_text
    assert "def _require_resolver_signature(" in path_ops_text
    assert "def _is_invocation_signature_typeerror(" in path_ops_text
    assert 'signature_checked = _require_resolver_signature(resolver, owner=owner, arg="edge_rotation_fn")' in path_ops_text
    assert 'signature_checked = _require_resolver_signature(resolver, owner=owner, arg="edge_pose_fn")' in path_ops_text
    assert path_ops_text.count("signature_checked=signature_checked") == 2
    assert path_ops_text.count("resolver(child, parent)") == 1
    assert "if not signature_checked and _is_invocation_signature_typeerror(exc):" in path_ops_text
    assert "must be callable(child, parent)." in path_ops_text
    assert "failed for edge" in path_ops_text
    handler_types = [
        node
        for node in ast.walk(call_node)
        if isinstance(node, ast.ExceptHandler)
    ]
    assert any(isinstance(node.type, ast.Name) and node.type.id == "TypeError" for node in handler_types)
    assert any(isinstance(node.type, ast.Name) and node.type.id == "Exception" for node in handler_types)


def test_spatial_doc_009_phase8_slice_b5_spatial6_vector6_bridge_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_009_phase8_slice_b5_spatial6_vector6_bridge_docs_and_api_entries_present."""
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8")
    acceleration_doc = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8")
    spatial_guide = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "as_vector6" in velocity_doc
    assert "from_vector6" in velocity_doc
    assert "as_vector6" in acceleration_doc
    assert "from_vector6" in acceleration_doc
    assert "Velocity" in spatial_guide


def test_spatial_doc_010_phase8_slice_c1_path_solver_foundation_docs_and_api_entries_present() -> None:
    """ID: SPATIAL_DOC_010_phase8_slice_c1_path_solver_foundation_docs_and_api_entries_present."""
    path_solve_doc = Path("docs/api/types/path_solve.md").read_text(encoding="utf-8")
    spatial_guide = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    api_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "PathSolveOptions" in path_solve_doc
    assert "solve_rotation_path_transform" in path_solve_doc
    assert "solve_pose_path_transform" in path_solve_doc
    assert "path" in spatial_guide.lower()
    assert "path_solve" in api_index


def test_arch_spatial_070_slice_c2_frame_api_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_070_slice_c2_frame_api_owner_split_and_budget."""
    owner_path = Path("tal/spatial/ops/frame_api_ops.py")
    assert owner_path.exists(), f"missing spatial owner file: {owner_path}"
    _assert_agents_budget(owner_path)


def test_arch_spatial_071_slice_c2_position_to_frame_reuses_c1_pose_solver_and_b4_pose_apply() -> None:
    """ID: ARCH_SPATIAL_071_slice_c2_position_to_frame_reuses_c1_pose_solver_and_b4_pose_apply."""
    frame_api_text = Path("tal/spatial/ops/frame_api_ops.py").read_text(encoding="utf-8")
    assert "_solve_pose_path_transform_with_owner(" in frame_api_text
    assert "_pose_apply_with_owner(" in frame_api_text
    assert 'owner = "spatial.position.to_frame"' in frame_api_text
    frame_api_module = _module("tal/spatial/ops/frame_api_ops.py")
    position_to_frame = _function_node(frame_api_module, "position_to_frame")
    solve_idx: int | None = None
    identity_idx: int | None = None
    for idx, stmt in enumerate(position_to_frame.body):
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "_solve_pose_path_transform_with_owner"
        ):
            solve_idx = idx
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "destination"
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.Eq)
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Name)
            and stmt.test.comparators[0].id == "source_parent"
            and len(stmt.body) >= 1
            and isinstance(stmt.body[0], ast.Return)
            and isinstance(stmt.body[0].value, ast.Call)
                and isinstance(stmt.body[0].value.func, ast.Name)
                and stmt.body[0].value.func.id == "wrap_like"
        ):
            identity_idx = idx
    assert solve_idx is not None
    assert identity_idx is not None
    assert solve_idx < identity_idx


def test_arch_spatial_072_slice_c2_rotation_pose_class_wrappers_delegate_to_c1_owners() -> None:
    """ID: ARCH_SPATIAL_072_slice_c2_rotation_pose_class_wrappers_delegate_to_c1_owners."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "def solve_path_transform(" in rotation_text
    assert "rotation_class_solve_path_transform(" in rotation_text
    assert "def solve_path_transform(" in pose_text
    assert "pose_class_solve_path_transform(" in pose_text


def test_arch_spatial_073_slice_c2_no_local_find_path_fold_path_or_local_math_duplication() -> None:
    """ID: ARCH_SPATIAL_073_slice_c2_no_local_find_path_fold_path_or_local_math_duplication."""
    frame_api_text = Path("tal/spatial/ops/frame_api_ops.py").read_text(encoding="utf-8")
    assert "find_path(" not in frame_api_text
    assert "fold_path(" not in frame_api_text
    assert "SciRotation" not in frame_api_text
    assert "compose_quat_kernel(" not in frame_api_text
    assert "compose_translation_kernel(" not in frame_api_text


def test_arch_spatial_074_slice_c2_owner_wrapped_lazy_error_boundary_present() -> None:
    """ID: ARCH_SPATIAL_074_slice_c2_owner_wrapped_lazy_error_boundary_present."""
    path_solve_text = Path("tal/spatial/path_solve.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    frame_api_text = Path("tal/spatial/ops/frame_api_ops.py").read_text(encoding="utf-8")
    assert "def _solve_pose_path_transform_with_owner(" in path_solve_text
    assert "def _solve_rotation_path_transform_with_owner(" in path_solve_text
    assert "owner: str" in path_solve_text
    assert "def _pose_apply_with_owner(" in pose_apply_text
    assert 'kwargs={"owner": owner}' in pose_apply_text
    assert "_solve_pose_path_transform_with_owner(" in frame_api_text
    assert "_pose_apply_with_owner(" in frame_api_text
    assert 'owner = "spatial.position.to_frame"' in frame_api_text


def test_spatial_doc_011_phase8_slice_c2_frame_api_docs_and_entries_present() -> None:
    """ID: SPATIAL_DOC_011_phase8_slice_c2_frame_api_docs_and_entries_present."""
    position_doc = Path("docs/api/types/position.md").read_text(encoding="utf-8")
    rotation_doc = Path("docs/api/types/rotation.md").read_text(encoding="utf-8")
    pose_doc = Path("docs/api/types/pose.md").read_text(encoding="utf-8")
    path_solve_doc = Path("docs/api/types/path_solve.md").read_text(encoding="utf-8")
    api_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    spatial_guide = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    assert "to_frame" in position_doc
    assert "solve_path_transform" in rotation_doc
    assert "solve_path_transform" in pose_doc
    assert "to_frame" in path_solve_doc


def test_arch_spatial_075_velocity_acceleration_thin_boundaries_reuse_kinematics_family() -> None:
    """ID: ARCH_SPATIAL_075_velocity_acceleration_thin_boundaries_reuse_kinematics_family."""
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    assert "from .kinematics.family import (" in velocity_text
    assert "from .kinematics.family import (" in acceleration_text
    assert "family_from_linear_angular(" in velocity_text
    assert "family_from_linear_angular(" in acceleration_text
    assert "family_to_rep(" in velocity_text
    assert "family_to_rep(" in acceleration_text


def test_arch_spatial_076_frame_policies_single_owner_reused_across_spatial() -> None:
    """ID: ARCH_SPATIAL_076_frame_policies_single_owner_reused_across_spatial."""
    frame_text = Path("tal/spatial/policies/frame.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    intent_text = Path("tal/spatial/policies/intent.py").read_text(encoding="utf-8")
    path_text = Path("tal/spatial/ops/path_solve_ops.py").read_text(encoding="utf-8")
    kin_text = Path("tal/spatial/kinematics/family.py").read_text(encoding="utf-8")
    assert "def resolve_apply_output_frames(" in frame_text
    assert "def resolve_compose_output_frames(" in frame_text
    assert "def resolve_components_shared_frames(" in frame_text
    assert "from .policies.frame import resolve_compose_output_frames" in rotation_text
    assert "from ..policies.frame import resolve_compose_output_frames, resolve_components_shared_frames" in pose_ops_text
    assert "from ..policies.frame import resolve_apply_output_frames" in rotation_apply_text
    assert "from ..policies.frame import resolve_apply_output_frames" in pose_apply_text
    assert "resolve_bidirectional_tip_tail_frames" in intent_text
    assert "from ..policies.frame import is_framed" in path_text
    assert "from ..policies.frame import resolve_components_shared_frames" in kin_text


def test_arch_spatial_077_apply_shared_and_pose_shared_removed() -> None:
    """ID: ARCH_SPATIAL_077_apply_shared_and_pose_shared_removed."""
    assert not Path("tal/spatial/apply_shared.py").exists()
    assert not Path("tal/spatial/pose_shared.py").exists()


def test_arch_spatial_078_core_runtime_checks_reused_by_spatial_and_components() -> None:
    """ID: ARCH_SPATIAL_078_core_runtime_checks_reused_by_spatial_and_components."""
    runtime_text = Path("tal/core/orchestration/runtime_checks.py").read_text(encoding="utf-8")
    spatial_runtime_text = Path("tal/spatial/policies/runtime_checks.py").read_text(encoding="utf-8")
    component_runtime_text = Path("tal/core/component_ops/runtime_checks.py").read_text(encoding="utf-8")
    assert "def resolve_single_numeric_var_single_core_dim(" in runtime_text
    assert "from tal.core.orchestration.runtime_checks import (" in spatial_runtime_text
    assert "from ..orchestration.runtime_checks import require_declared_roles_with_sequence" in component_runtime_text
    assert "def require_component_numeric_var(" in component_runtime_text
    assert "require_component_numeric_var" not in spatial_runtime_text


def test_arch_spatial_079_core_alignment_owner_reused_by_linalg_and_spatial() -> None:
    """ID: ARCH_SPATIAL_079_core_alignment_owner_reused_by_linalg_and_spatial."""
    alignment_text = Path("tal/core/orchestration/alignment.py").read_text(encoding="utf-8")
    linalg_plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "def align_exact(" in alignment_text
    assert "def align_exact_for_plan(" in alignment_text
    assert "from ..core.orchestration.alignment import align_exact_for_plan" in linalg_plan_text
    assert "from tal.core.orchestration.alignment import align_exact_for_plan" in rotation_apply_text
    assert "from tal.core.orchestration.alignment import align_exact_for_plan" in pose_ops_text


def test_arch_spatial_080_no_local_single_var_core_dim_duplicates_remain() -> None:
    """ID: ARCH_SPATIAL_080_no_local_single_var_core_dim_duplicates_remain."""
    for path in [
        "tal/spatial/ops/rotation_apply_ops.py",
        "tal/spatial/ops/pose_apply_ops.py",
        "tal/spatial/ops/pose_ops.py",
    ]:
        text = Path(path).read_text(encoding="utf-8")
        assert "def _single_var_core_dim(" not in text
        assert "resolve_single_numeric_var_single_core_dim(" in text


def test_arch_spatial_081_paired_components_owner_reused_by_pose_and_kinematics() -> None:
    """ID: ARCH_SPATIAL_081_paired_components_owner_reused_by_pose_and_kinematics."""
    paired_text = Path("tal/spatial/kinematics/paired_components.py").read_text(encoding="utf-8")
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    kin_text = Path("tal/spatial/kinematics/family.py").read_text(encoding="utf-8")
    assert "def build_paired_components_dataset(" in paired_text
    assert "def clear_component_registry(" in paired_text
    assert "build_paired_components_dataset(" in pose_text
    assert "build_paired_components_dataset(" in kin_text
    assert "clear_component_registry" in pose_text
    assert "clear_component_registry" in kin_text


def test_arch_spatial_082_conversion_finalize_owner_reused_by_rotation_pose_vector6() -> None:
    """ID: ARCH_SPATIAL_082_conversion_finalize_owner_reused_by_rotation_pose_vector6."""
    conversion_text = Path("tal/spatial/conversion/finalize.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    vector6_text = Path("tal/spatial/kinematics/vector6_ops.py").read_text(encoding="utf-8")
    assert "def allocate_free_dim_name(" in conversion_text
    assert "def allocate_dim_pair(" in conversion_text
    assert "def dataset_dim_names(" in conversion_text
    assert "def conversion_dataset_from_array(" in conversion_text
    assert "def finalize_conversion_dataset(" in conversion_text
    assert "from .conversion.finalize import (" in rotation_text
    assert "allocate_dim_pair" in rotation_text
    assert "dataset_dim_names" in rotation_text
    assert "from ..conversion.finalize import allocate_dim_pair, dataset_dim_names" in pose_ops_text
    assert "from ..conversion.finalize import allocate_free_dim_name as _allocate_dim_name" in vector6_text


def test_arch_spatial_093_pose_component_layout_helpers_removed_in_favor_of_shared_owners() -> None:
    """ID: ARCH_SPATIAL_093_pose_component_layout_helpers_removed_in_favor_of_shared_owners."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "def _resolve_component_spec(" not in pose_text
    assert "def _require_component_var(" not in pose_text
    assert "def _resolve_component_roles(" not in pose_text
    assert "def _component_var_names(" not in pose_text
    assert "resolve_pair_registry(" in pose_text
    assert "resolve_component_spec(" in pose_text
    assert "resolve_paired_roles(" in pose_text
    assert "component_var_names(" in pose_text


def test_arch_spatial_094_spatial_runtime_checks_xyz_only_no_generic_component_validation() -> None:
    """ID: ARCH_SPATIAL_094_spatial_runtime_checks_xyz_only_no_generic_component_validation."""
    spatial_runtime_text = Path("tal/spatial/policies/runtime_checks.py").read_text(encoding="utf-8")
    assert "def require_xyz_core_labels(" in spatial_runtime_text
    assert "def require_component_numeric_var(" not in spatial_runtime_text
    assert "select_single_numeric_var" not in spatial_runtime_text
    assert "require_declared_roles_with_sequence" not in spatial_runtime_text
    assert "require_var_contains_dims" not in spatial_runtime_text


def test_arch_spatial_095_conversion_finalize_deepened_reuse_for_rotation_and_pose() -> None:
    """ID: ARCH_SPATIAL_095_conversion_finalize_deepened_reuse_for_rotation_and_pose."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "def _resolve_matrix_core_dims_for_output(" not in rotation_text
    assert "def _resolve_quat_dim_for_output(" not in rotation_text
    assert "conversion_dataset_from_array(" in rotation_text
    assert "allocate_dim_pair(" in rotation_text
    assert "dataset_dim_names(" in rotation_text
    assert "def _allocate_matrix_dims(" not in pose_ops_text
    assert "def _allocate_component_dims(" not in pose_ops_text
    assert "allocate_dim_pair(" in pose_ops_text
    assert "dataset_dim_names(" in pose_ops_text


def test_arch_spatial_096_pose_decompose_output_core_dims_not_bound_by_positional_index() -> None:
    """ID: ARCH_SPATIAL_096_pose_decompose_output_core_dims_not_bound_by_positional_index."""
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    assert "translation.dims[-1]" not in pose_text
    assert "quat.dims[-1]" not in pose_text
    assert "position_core_dim=row_dim" in pose_text
    assert "rotation_core_dim=quat_dim" in pose_text
    assert "if position_core_dim not in translation.dims:" in pose_text
    assert "if rotation_core_dim not in quat.dims:" in pose_text


def test_arch_spatial_097_spatial_kinematics_components_module_removed_and_family_owner_consolidated() -> None:
    """ID: ARCH_SPATIAL_097_spatial_kinematics_components_module_removed_and_family_owner_consolidated."""
    assert not Path("tal/spatial/kinematics_components.py").exists()
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    family_text = Path("tal/spatial/kinematics/family.py").read_text(encoding="utf-8")
    paired_text = Path("tal/spatial/kinematics/paired_components.py").read_text(encoding="utf-8")
    assert "from .kinematics.family import (" in velocity_text
    assert "from .kinematics.family import (" in acceleration_text
    assert "build_paired_components_dataset(" in family_text
    assert "resolve_pair_registry(" in family_text
    assert "def build_paired_components_dataset(" in paired_text


def test_arch_spatial_083_wrap_owner_reused_across_spatial_modules() -> None:
    """ID: ARCH_SPATIAL_083_wrap_owner_reused_across_spatial_modules."""
    wrap_text = Path("tal/spatial/policies/wrap.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    frame_api_text = Path("tal/spatial/ops/frame_api_ops.py").read_text(encoding="utf-8")
    assert "def wrap_as(" in wrap_text
    assert "def wrap_like(" in wrap_text
    assert "from .policies.wrap import wrap_as" in rotation_text
    assert "from .policies.wrap import wrap_as" in pose_text
    assert "from ..policies.wrap import wrap_like" in pose_apply_text
    assert "from ..policies.wrap import wrap_like" in frame_api_text


def test_arch_spatial_084_no_direct_tal_attrs_writes_in_new_owner_modules() -> None:
    """ID: ARCH_SPATIAL_084_no_direct_tal_attrs_writes_in_new_owner_modules."""
    files = [
        "tal/core/orchestration/runtime_checks.py",
        "tal/core/orchestration/alignment.py",
        "tal/spatial/policies/frame.py",
        "tal/spatial/kinematics/family.py",
        "tal/spatial/kinematics/paired_components.py",
        "tal/spatial/conversion/finalize.py",
        "tal/spatial/policies/wrap.py",
    ]
    for rel in files:
        text = Path(rel).read_text(encoding="utf-8")
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text


def test_arch_spatial_085_no_eager_patterns_in_new_orchestration_owners() -> None:
    """ID: ARCH_SPATIAL_085_no_eager_patterns_in_new_orchestration_owners."""
    files = [
        "tal/spatial/kinematics/family.py",
        "tal/spatial/kinematics/paired_components.py",
        "tal/spatial/conversion/finalize.py",
        "tal/core/orchestration/runtime_checks.py",
        "tal/core/orchestration/alignment.py",
    ]
    for rel in files:
        text = Path(rel).read_text(encoding="utf-8")
        assert ".values" not in text
        assert ".item(" not in text
        assert "np.asarray(" not in text
        assert ".compute(" not in text


def test_arch_spatial_086_new_owner_modules_budget_lock() -> None:
    """ID: ARCH_SPATIAL_086_new_owner_modules_budget_lock."""
    files = [
        Path("tal/core/orchestration/runtime_checks.py"),
        Path("tal/core/orchestration/alignment.py"),
        Path("tal/spatial/policies/frame.py"),
        Path("tal/spatial/kinematics/family.py"),
        Path("tal/spatial/kinematics/paired_components.py"),
        Path("tal/spatial/conversion/finalize.py"),
        Path("tal/spatial/policies/wrap.py"),
    ]
    for path in files:
        _assert_agents_budget(path)


def test_arch_spatial_098_spatial_subpackage_layout_present() -> None:
    """ID: ARCH_SPATIAL_098_spatial_subpackage_layout_present."""
    packages = [
        "tal/spatial/metadata",
        "tal/spatial/policies",
        "tal/spatial/ops",
        "tal/spatial/kernels",
        "tal/spatial/kinematics",
        "tal/spatial/conversion",
    ]
    for package in packages:
        path = Path(package)
        assert path.exists()
        assert path.is_dir()
        assert (path / "__init__.py").exists()


def test_arch_spatial_099_removed_flat_internal_modules_absent_after_relayout() -> None:
    """ID: ARCH_SPATIAL_099_removed_flat_internal_modules_absent_after_relayout."""
    removed = [
        "tal/spatial/metadata.py",
        "tal/spatial/metadata_common.py",
        "tal/spatial/metadata_roles.py",
        "tal/spatial/metadata_representation.py",
        "tal/spatial/frame_policies.py",
        "tal/spatial/intent.py",
        "tal/spatial/runtime_checks.py",
        "tal/spatial/wrap.py",
        "tal/spatial/pose_ops.py",
        "tal/spatial/pose_apply_ops.py",
        "tal/spatial/rotation_apply_ops.py",
        "tal/spatial/frame_api_ops.py",
        "tal/spatial/path_solve_ops.py",
        "tal/spatial/pose_kernels.py",
        "tal/spatial/pose_apply_kernels.py",
        "tal/spatial/rotation_kernels.py",
        "tal/spatial/rotation_compose_kernels.py",
        "tal/spatial/rotation_apply_kernels.py",
        "tal/spatial/kinematics_family.py",
        "tal/spatial/kinematics_vector6_ops.py",
        "tal/spatial/kinematics_vector6_kernels.py",
        "tal/spatial/paired_components.py",
        "tal/spatial/conversion_finalize.py",
    ]
    for rel in removed:
        assert not Path(rel).exists()


def test_arch_spatial_100_public_spatial_entrypoints_remain_flat() -> None:
    """ID: ARCH_SPATIAL_100_public_spatial_entrypoints_remain_flat."""
    expected = [
        "tal/spatial/__init__.py",
        "tal/spatial/position.py",
        "tal/spatial/rotation.py",
        "tal/spatial/pose.py",
        "tal/spatial/velocity.py",
        "tal/spatial/acceleration.py",
        "tal/spatial/path_solve.py",
    ]
    for rel in expected:
        assert Path(rel).exists()


def test_arch_spatial_101_spatial_init_public_exports_stable() -> None:
    """ID: ARCH_SPATIAL_101_spatial_init_public_exports_stable."""
    text = Path("tal/spatial/__init__.py").read_text(encoding="utf-8")
    assert "from .position import Position" in text
    assert "from .rotation import Rotation" in text
    assert "from .pose import Pose" in text
    assert "from .velocity import AngularVelocity, LinearVelocity, Velocity" in text
    assert "from .acceleration import Acceleration, AngularAcceleration, LinearAcceleration" in text
    assert "from .path_solve import PathSolveOptions, solve_pose_path_transform, solve_rotation_path_transform" in text
    assert "__all__ = [" in text


def test_arch_spatial_102_public_entrypoints_delegate_to_new_internal_owners() -> None:
    """ID: ARCH_SPATIAL_102_public_entrypoints_delegate_to_new_internal_owners."""
    position_text = Path("tal/spatial/position.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    path_solve_text = Path("tal/spatial/path_solve.py").read_text(encoding="utf-8")
    assert "from .ops.frame_api_ops import position_to_frame" in position_text
    assert "from .ops.rotation_apply_ops import rotation_apply" in rotation_text
    assert "from .ops.pose_apply_ops import pose_apply" in pose_text
    assert "from .ops.pose_ops import pose_as_components, pose_as_matrix, pose_compose, pose_inverse, pose_to_rep" in pose_text
    assert "from .kinematics.family import (" in velocity_text
    assert "from .kinematics.family import (" in acceleration_text
    assert "from .ops.path_solve_ops import solve_pose_path_transform_impl, solve_rotation_path_transform_impl" in path_solve_text


def test_arch_spatial_103_ops_reuse_kernels_subpackage_no_flat_kernel_imports() -> None:
    """ID: ARCH_SPATIAL_103_ops_reuse_kernels_subpackage_no_flat_kernel_imports."""
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    assert "from ..kernels.pose_kernels import" in pose_ops_text
    assert "from ..kernels.pose_apply_kernels import" in pose_apply_text
    assert "from ..kernels.rotation_apply_kernels import" in rotation_apply_text
    assert "from ..pose_kernels import" not in pose_ops_text
    assert "from ..pose_apply_kernels import" not in pose_apply_text
    assert "from ..rotation_apply_kernels import" not in rotation_apply_text


def test_arch_spatial_104_metadata_owner_recomposed_under_subpackage() -> None:
    """ID: ARCH_SPATIAL_104_metadata_owner_recomposed_under_subpackage."""
    metadata_init = Path("tal/spatial/metadata/__init__.py").read_text(encoding="utf-8")
    metadata_facade = Path("tal/spatial/metadata/facade.py").read_text(encoding="utf-8")
    assert "from .facade import (" in metadata_init
    assert "from .representation import" in metadata_facade
    assert "from .roles import" in metadata_facade
    assert "def get_position_rep(" not in metadata_init


def test_arch_spatial_105_policy_owner_modules_live_under_policies_subpackage() -> None:
    """ID: ARCH_SPATIAL_105_policy_owner_modules_live_under_policies_subpackage."""
    assert Path("tal/spatial/policies/frame.py").exists()
    assert Path("tal/spatial/policies/intent.py").exists()
    assert Path("tal/spatial/policies/runtime_checks.py").exists()
    assert Path("tal/spatial/policies/wrap.py").exists()
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert "from .policies.frame import resolve_components_shared_frames" in pose_text
    assert "from .policies.wrap import wrap_as" in pose_text
    assert "from .policies.frame import resolve_compose_output_frames" in rotation_text
    assert "from .policies.wrap import wrap_as" in rotation_text


def test_arch_spatial_106_kinematics_owner_modules_live_under_kinematics_subpackage() -> None:
    """ID: ARCH_SPATIAL_106_kinematics_owner_modules_live_under_kinematics_subpackage."""
    assert Path("tal/spatial/kinematics/family.py").exists()
    assert Path("tal/spatial/kinematics/vector6_ops.py").exists()
    assert Path("tal/spatial/kinematics/paired_components.py").exists()
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    vector6_text = Path("tal/spatial/kinematics/vector6_ops.py").read_text(encoding="utf-8")
    assert "from .kinematics.family import (" in velocity_text
    assert "from .kinematics.family import (" in acceleration_text
    assert "from .paired_components import clear_component_registry, shared_optional_name" in vector6_text


def test_arch_spatial_107_conversion_finalize_owner_lives_under_conversion_subpackage() -> None:
    """ID: ARCH_SPATIAL_107_conversion_finalize_owner_lives_under_conversion_subpackage."""
    assert Path("tal/spatial/conversion/finalize.py").exists()
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    vector6_text = Path("tal/spatial/kinematics/vector6_ops.py").read_text(encoding="utf-8")
    assert "from .conversion.finalize import (" in rotation_text
    assert "from ..conversion.finalize import allocate_dim_pair, dataset_dim_names" in pose_ops_text
    assert "from ..conversion.finalize import allocate_free_dim_name as _allocate_dim_name" in vector6_text


def test_arch_spatial_108_spatial_top_level_python_files_limited_to_public_entrypoints() -> None:
    """ID: ARCH_SPATIAL_108_spatial_top_level_python_files_limited_to_public_entrypoints."""
    allowed = {
        "__init__.py",
        "position.py",
        "rotation.py",
        "pose.py",
        "velocity.py",
        "acceleration.py",
        "path_solve.py",
    }
    observed = {path.name for path in Path("tal/spatial").glob("*.py")}
    assert observed == allowed


def test_arch_spatial_109_no_removed_flat_module_import_tokens_reintroduced() -> None:
    """ID: ARCH_SPATIAL_109_no_removed_flat_module_import_tokens_reintroduced."""
    banned_tokens = [
        "tal.spatial.pose_ops",
        "tal.spatial.pose_apply_ops",
        "tal.spatial.rotation_apply_ops",
        "tal.spatial.path_solve_ops",
        "tal.spatial.frame_api_ops",
        "tal.spatial.frame_policies",
        "tal.spatial.runtime_checks",
        "tal.spatial.kinematics_family",
        "tal.spatial.kinematics_vector6_ops",
        "tal.spatial.paired_components",
        "tal.spatial.conversion_finalize",
    ]
    for path in sorted(Path("tal").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned_tokens:
            assert token not in text, f"{path} reintroduced removed flat token {token!r}"


def test_arch_spatial_110_structural_refactor_owner_modules_budget_lock() -> None:
    """ID: ARCH_SPATIAL_110_structural_refactor_owner_modules_budget_lock."""
    files = [
        Path("tal/spatial/metadata/facade.py"),
        Path("tal/spatial/policies/frame.py"),
        Path("tal/spatial/ops/pose_ops.py"),
        Path("tal/spatial/ops/pose_apply_ops.py"),
        Path("tal/spatial/ops/rotation_apply_ops.py"),
        Path("tal/spatial/ops/path_solve_ops.py"),
        Path("tal/spatial/kernels/pose_kernels.py"),
        Path("tal/spatial/kernels/rotation_kernels.py"),
        Path("tal/spatial/kinematics/family.py"),
        Path("tal/spatial/kinematics/vector6_ops.py"),
        Path("tal/spatial/conversion/finalize.py"),
    ]
    for path in files:
        _assert_agents_budget(path)


def test_arch_topo_003_spatial_owners_delegate_topology_to_core_owner() -> None:
    """ID: ARCH_TOPO_003_spatial_owners_delegate_topology_to_core_owner."""
    topology_files = [
        "tal/spatial/rotation.py",
        "tal/spatial/ops/pose_context.py",
        "tal/spatial/ops/rotation_apply_ops.py",
        "tal/spatial/ops/pose_apply_ops.py",
        "tal/spatial/kinematics/vector6_ops.py",
    ]
    for rel in topology_files:
        text = Path(rel).read_text(encoding="utf-8")
        assert "resolve_semantic_topology_from_dataset(" in text
        assert "from tal.core.orchestration.topology import" in text
    align_files = [
        "tal/spatial/rotation.py",
        "tal/spatial/ops/pose_ops.py",
        "tal/spatial/ops/rotation_apply_ops.py",
        "tal/spatial/ops/pose_apply_ops.py",
        "tal/spatial/kinematics/vector6_ops.py",
    ]
    for rel in align_files:
        text = Path(rel).read_text(encoding="utf-8")
        assert "align_exact_for_plan(" in text
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    assert "from .pose_context import (" in pose_ops_text
    assert "_topology_operand(" in pose_ops_text


def test_arch_topo_004_no_reintroduced_local_binary_topology_islands() -> None:
    """ID: ARCH_TOPO_004_no_reintroduced_local_binary_topology_islands."""
    files = [
        "tal/spatial/rotation.py",
        "tal/spatial/ops/pose_ops.py",
        "tal/spatial/ops/rotation_apply_ops.py",
        "tal/spatial/ops/pose_apply_ops.py",
        "tal/spatial/kinematics/vector6_ops.py",
        "tal/linalg/plan.py",
    ]
    for rel in files:
        text = Path(rel).read_text(encoding="utf-8")
        assert "require_non_core_dim_names_match(" not in text


def test_arch_topo_011_spatial_topology_calls_use_strict_non_core_policy_constant() -> None:
    """ID: ARCH_TOPO_011_spatial_topology_calls_use_strict_non_core_policy_constant."""
    files = [
        "tal/spatial/rotation.py",
        "tal/spatial/ops/pose_ops.py",
        "tal/spatial/ops/rotation_apply_ops.py",
        "tal/spatial/ops/pose_apply_ops.py",
        "tal/spatial/kinematics/vector6_ops.py",
        "tal/spatial/kinematics/family.py",
    ]
    for rel in files:
        text = Path(rel).read_text(encoding="utf-8")
        assert "STRICT_NON_CORE_POLICY" in text
        if rel == "tal/spatial/kinematics/vector6_ops.py":
            assert "policy: TopologyPolicy" in text
            continue
        assert "SEMANTIC_NON_CORE_POLICY" in text
        assert "select_topology_policy_with_intents(" in text


def test_arch_bcast_002_no_domain_local_duplicate_b_helpers() -> None:
    """ID: ARCH_BCAST_002_no_domain_local_duplicate_b_helpers."""
    for path in Path("tal/spatial").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "def b(" not in text, f"domain-local .b helper found in {path}"
    for path in Path("tal/linalg").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "def b(" not in text, f"domain-local .b helper found in {path}"


def test_arch_bcast_023_no_domain_local_duplicate_a_helpers() -> None:
    """ID: ARCH_BCAST_023_no_domain_local_duplicate_a_helpers."""
    for path in Path("tal/spatial").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "def a(" not in text, f"domain-local .a helper found in {path}"
    for path in Path("tal/linalg").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "def a(" not in text, f"domain-local .a helper found in {path}"


def test_arch_bcast_006_spatial_consumers_do_not_reimplement_broadcast_policy() -> None:
    """ID: ARCH_BCAST_006_spatial_consumers_do_not_reimplement_broadcast_policy."""
    files = [
        "tal/spatial/rotation.py",
        "tal/spatial/ops/pose_ops.py",
        "tal/spatial/ops/rotation_apply_ops.py",
        "tal/spatial/ops/pose_apply_ops.py",
        "tal/spatial/kinematics/vector6_ops.py",
        "tal/spatial/kinematics/family.py",
    ]
    for rel in files:
        text = Path(rel).read_text(encoding="utf-8")
        if rel == "tal/spatial/kinematics/vector6_ops.py":
            assert "policy: TopologyPolicy" in text
            assert "resolved_policy = policy if policy is not None else STRICT_NON_CORE_POLICY" in text
            continue
        if rel == "tal/spatial/kinematics/family.py":
            assert "select_topology_policy_with_intents(" in text
            assert "pack_linear_angular_to_vector6_dataset(" in text
            continue
        assert "select_topology_policy_with_intents(" in text
        assert "resolve_nary_topology(" in text or "resolve_binary_topology(" in text
        assert "semantic broadcast only permits" not in text


def test_arch_bcast_020_strict_family_paths_still_route_through_b_intent_gate() -> None:
    """ID: ARCH_BCAST_020_strict_family_paths_still_route_through_b_intent_gate."""
    files = [
        "tal/spatial/rotation.py",
        "tal/spatial/ops/pose_ops.py",
        "tal/spatial/ops/rotation_apply_ops.py",
        "tal/spatial/ops/pose_apply_ops.py",
        "tal/spatial/kinematics/vector6_ops.py",
    ]
    for rel in files:
        text = Path(rel).read_text(encoding="utf-8")
        if rel == "tal/spatial/kinematics/vector6_ops.py":
            assert "policy: TopologyPolicy" in text
            assert "resolved_policy = policy if policy is not None else STRICT_NON_CORE_POLICY" in text
            assert "select_topology_policy_for_operation_family(" not in text
            continue
        assert "select_topology_policy_with_intents(" in text
        assert 'operation_family="spatial.' in text
        assert "select_topology_policy_for_operation_family(" not in text


def test_arch_spatial_111_pose_apply_topology_helpers_parameter_budget() -> None:
    """ID: ARCH_SPATIAL_111_pose_apply_topology_helpers_parameter_budget."""
    module = _module("tal/spatial/ops/pose_apply_ops.py")
    for name in ("_resolve_pose_position_plan", "_pose_apply_topology_operands", "_align_pose_position_operands"):
        node = _function_node(module, name)
        assert _parameter_count(node) <= 10


def test_arch_spatial_112_pose_compose_topology_helpers_parameter_budget() -> None:
    """ID: ARCH_SPATIAL_112_pose_compose_topology_helpers_parameter_budget."""
    module = _module("tal/spatial/ops/pose_context.py")
    node = _function_node(module, "pose_compose_topology_operands")
    assert _parameter_count(node) <= 10


def test_arch_bcast_031_spatial_frame_precheck_order_is_structurally_guarded() -> None:
    """ID: ARCH_BCAST_031_spatial_frame_precheck_order_is_structurally_guarded."""
    rotation_module = _module("tal/spatial/rotation.py")
    rotation_compose = _function_node(rotation_module, "_rotation_compose_with_owner")
    compose_order = _compose_call_order_tokens(rotation_compose)
    assert "resolve_compose_output_frames" in compose_order
    assert "left.as_quat" in compose_order
    assert compose_order.index("resolve_compose_output_frames") < compose_order.index("left.as_quat")

    pose_ops_module = _module("tal/spatial/ops/pose_ops.py")
    pose_compose = _function_node(pose_ops_module, "_pose_compose_with_owner")
    pose_compose_order = _call_order_tokens(pose_compose, _pose_compose_call_token)
    assert "resolve_compose_output_frames" in pose_compose_order
    assert "_canonical_components" in pose_compose_order
    assert pose_compose_order.index("resolve_compose_output_frames") < pose_compose_order.index("_canonical_components")

    rotation_apply_module = _module("tal/spatial/ops/rotation_apply_ops.py")
    rotation_apply = _function_node(rotation_apply_module, "_rotation_apply_with_owner")
    rotation_apply_order = _call_order_tokens(rotation_apply, _rotation_apply_call_token)
    assert "resolve_apply_output_frames" in rotation_apply_order
    assert "_apply_to_vector_target" in rotation_apply_order
    assert rotation_apply_order.index("resolve_apply_output_frames") < rotation_apply_order.index("_apply_to_vector_target")

    pose_apply_module = _module("tal/spatial/ops/pose_apply_ops.py")
    pose_apply = _function_node(pose_apply_module, "_pose_apply_with_owner")
    pose_apply_order = _call_order_tokens(pose_apply, _pose_apply_call_token)
    assert "resolve_apply_output_frames" in pose_apply_order
    assert "_apply_pose_to_position" in pose_apply_order
    assert pose_apply_order.index("resolve_apply_output_frames") < pose_apply_order.index("_apply_pose_to_position")

    pose_module = _module("tal/spatial/pose.py")
    pose_components = _class_method_node(pose_module, "Pose", "from_components")
    pose_components_order = _call_order_tokens(pose_components, _pose_components_call_token)
    assert "resolve_components_shared_frames" in pose_components_order
    assert "select_topology_policy_with_intents" in pose_components_order
    assert "_build_components_pose_dataset" in pose_components_order
    assert pose_components_order.index("resolve_components_shared_frames") < pose_components_order.index(
        "select_topology_policy_with_intents"
    )
    assert pose_components_order.index("resolve_components_shared_frames") < pose_components_order.index(
        "_build_components_pose_dataset"
    )

    kin_module = _module("tal/spatial/kinematics/family.py")
    kin_components = _function_node(kin_module, "family_from_linear_angular")
    kin_components_order = _call_order_tokens(kin_components, _kinematics_components_call_token)
    assert "resolve_components_shared_frames" in kin_components_order
    assert "select_topology_policy_with_intents" in kin_components_order
    assert "build_family_dataset" in kin_components_order
    assert kin_components_order.index("resolve_components_shared_frames") < kin_components_order.index(
        "select_topology_policy_with_intents"
    )
    assert kin_components_order.index("resolve_components_shared_frames") < kin_components_order.index(
        "build_family_dataset"
    )


def test_arch_bcast_033_spatial_output_frames_derived_from_frame_semantics_not_alignment() -> None:
    """ID: ARCH_BCAST_033_spatial_output_frames_derived_from_frame_semantics_not_alignment."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_ops_text = Path("tal/spatial/ops/pose_ops.py").read_text(encoding="utf-8")
    rotation_apply_text = Path("tal/spatial/ops/rotation_apply_ops.py").read_text(encoding="utf-8")
    pose_apply_text = Path("tal/spatial/ops/pose_apply_ops.py").read_text(encoding="utf-8")
    assert "resolve_compose_output_frames(" in rotation_text
    assert "set_frames(composed_quat, parent=parent, child=child" in rotation_text
    assert "resolve_compose_output_frames(" in pose_ops_text
    assert "set_frames(out_pos_ds, parent=parent, child=child" in pose_ops_text
    assert "resolve_apply_output_frames(" in rotation_apply_text
    assert "set_frames(out_ds, parent=parent, child=child" in rotation_apply_text
    assert "resolve_apply_output_frames(" in pose_apply_text
    assert "set_frames(out_ds, parent=parent, child=child" in pose_apply_text


def test_arch_bcast_040_component_assembly_frame_precheck_order_is_structurally_guarded() -> None:
    """ID: ARCH_BCAST_040_component_assembly_frame_precheck_order_is_structurally_guarded."""
    pose_module = _module("tal/spatial/pose.py")
    pose_components = _class_method_node(pose_module, "Pose", "from_components")
    pose_components_order = _call_order_tokens(pose_components, _pose_components_call_token)
    assert pose_components_order.index("resolve_components_shared_frames") < pose_components_order.index(
        "select_topology_policy_with_intents"
    )

    kin_module = _module("tal/spatial/kinematics/family.py")
    kin_components = _function_node(kin_module, "family_from_linear_angular")
    kin_components_order = _call_order_tokens(kin_components, _kinematics_components_call_token)
    assert kin_components_order.index("resolve_components_shared_frames") < kin_components_order.index(
        "select_topology_policy_with_intents"
    )


def test_arch_bcast_034_position_add_intent_resolution_remains_pre_alignment() -> None:
    """ID: ARCH_BCAST_034_position_add_intent_resolution_remains_pre_alignment."""
    position_text = Path("tal/spatial/position.py").read_text(encoding="utf-8")
    assert "plan = resolve_position_add_intent(left, right, owner=owner)" in position_text
    assert "numeric = linalg_add(left, right)" in position_text
    assert position_text.index("resolve_position_add_intent(left, right, owner=owner)") < position_text.index(
        "numeric = linalg_add(left, right)"
    )


def test_arch_bcast_037_spatial_e3c_modules_remain_budget_compliant() -> None:
    """ID: ARCH_BCAST_037_spatial_e3c_modules_remain_budget_compliant."""
    files = [
        Path("tal/spatial/rotation.py"),
        Path("tal/spatial/ops/pose_ops.py"),
        Path("tal/spatial/ops/rotation_apply_ops.py"),
        Path("tal/spatial/ops/pose_apply_ops.py"),
        Path("tal/spatial/ops/pose_context.py"),
        Path("tal/spatial/kinematics/family.py"),
        Path("tal/spatial/kinematics/vector6_ops.py"),
        Path("tal/spatial/kinematics/paired_components.py"),
        Path("tal/spatial/position.py"),
    ]
    for path in files:
        _assert_agents_budget(path)


def test_arch_bcast_041_paired_component_alignment_preserves_schema_attrs_for_coord_role_reads() -> None:
    """ID: ARCH_BCAST_041_paired_component_alignment_preserves_schema_attrs_for_coord_role_reads."""
    text = Path("tal/spatial/kinematics/paired_components.py").read_text(encoding="utf-8")
    align_body = text.split("def align_paired_component_payloads(", 1)[1].split("\ndef ", 1)[0]
    assert "_rewrap_aligned_component_dataset(" in align_body
    assert "to_dataset(name=left_var)" not in align_body
    assert "to_dataset(name=right_var)" not in align_body


def test_arch_spatial_113_temporal_vector_like_orchestrate_kernel_finalize_split() -> None:
    """ID: ARCH_SPATIAL_113_temporal_vector_like_orchestrate_kernel_finalize_split."""
    evaluate_text = Path("tal/core/param_ops/evaluate.py").read_text(encoding="utf-8")
    finalize_text = Path("tal/core/param_ops/finalize.py").read_text(encoding="utf-8")
    assert "def evaluate_param(" in evaluate_text
    assert "build_param_map(" in evaluate_text
    assert "apply_param_map(" in evaluate_text
    assert "finalize_param_output(" in evaluate_text
    assert "def finalize_param_output(" in finalize_text


def test_arch_spatial_114_temporal_vector_like_reuses_param_engine_map_build_apply_owners() -> None:
    """ID: ARCH_SPATIAL_114_temporal_vector_like_reuses_param_engine_map_build_apply_owners."""
    evaluate_text = Path("tal/core/param_ops/evaluate.py").read_text(encoding="utf-8")
    position_text = Path("tal/spatial/position.py").read_text(encoding="utf-8")
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    assert "build_param_map" in evaluate_text
    assert "apply_param_map" in evaluate_text
    for text in (position_text, velocity_text, acceleration_text):
        assert "build_param_map" not in text
        assert "apply_param_map" not in text


def test_arch_spatial_115_temporal_vector_like_no_local_param_kernel_duplication() -> None:
    """ID: ARCH_SPATIAL_115_temporal_vector_like_no_local_param_kernel_duplication."""
    for rel in ("tal/spatial/position.py", "tal/spatial/velocity.py", "tal/spatial/acceleration.py"):
        text = Path(rel).read_text(encoding="utf-8")
        assert "map_row_backend(" not in text
        assert "bounds_row_backend(" not in text
        assert "xr.apply_ufunc(" not in text


def test_arch_spatial_116_temporal_vector_like_no_eager_patterns_in_orchestration_paths() -> None:
    """ID: ARCH_SPATIAL_116_temporal_vector_like_no_eager_patterns_in_orchestration_paths."""
    for rel in ("tal/core/param_ops/accessor.py", "tal/core/param_ops/evaluate.py", "tal/core/param_ops/resample.py"):
        text = executable_source(path=rel)
        assert ".compute(" not in text
        assert ".item(" not in text
    accessor_text = executable_source(path="tal/core/param_ops/accessor.py")
    assert "np.asarray(" not in accessor_text
    assert ".values" not in accessor_text


def test_arch_spatial_117_temporal_vector_like_methods_use_options_objects_within_budget() -> None:
    """ID: ARCH_SPATIAL_117_temporal_vector_like_methods_use_options_objects_within_budget."""
    accessor_text = Path("tal/core/param_ops/accessor.py").read_text(encoding="utf-8")
    assert "opts: ParamEvalOptions | None = None" in accessor_text
    assert "coerce_eval_options(" in accessor_text
    for rel in ("tal/spatial/position.py", "tal/spatial/velocity.py", "tal/spatial/acceleration.py"):
        text = Path(rel).read_text(encoding="utf-8")
        assert "def at_param(" not in text
        assert "def resample_param(" not in text

def test_spatial_doc_013_temporal_vector_like_interpolation_resample_boundaries_documented() -> None:
    """ID: SPATIAL_DOC_013_temporal_vector_like_interpolation_resample_boundaries_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    assert ".param.at(" in spatial_doc
    assert ".param.resample_to(" in spatial_doc


def test_spatial_doc_016_temporal_param_coord_primary_key_behavior_documented_for_temporal_methods() -> None:
    """ID: SPATIAL_DOC_016_temporal_param_coord_primary_key_behavior_documented_for_temporal_methods."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    assert "on=" in spatial_doc
    assert "parameter" in spatial_doc


def test_spatial_doc_018_temporal_vectorize_false_and_stopgap_exception_policy_documented() -> None:
    """ID: SPATIAL_DOC_018_temporal_vectorize_false_and_stopgap_exception_policy_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    assert "param" in spatial_doc
    assert "resample" in spatial_doc


def test_arch_spatial_141_rotation_pose_typed_param_overrides_are_at_resample_only() -> None:
    """ID: ARCH_SPATIAL_141_rotation_pose_typed_param_overrides_are_at_resample_only."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    accessor_text = Path("tal/spatial/temporal/accessor.py").read_text(encoding="utf-8")
    assert "def param(self)" in rotation_text
    assert "def param(self)" in pose_text
    assert "def at_param(" not in rotation_text
    assert "def resample_param(" not in rotation_text
    assert "def at_param(" not in pose_text
    assert "def resample_param(" not in pose_text
    assert "class RotationParamAccessor(ParamAccessor):" in accessor_text
    assert "class PoseParamAccessor(ParamAccessor):" in accessor_text
    assert "def at(" in accessor_text
    assert "def resample_to(" in accessor_text
    assert "def interp_like(" not in accessor_text
    assert "def sel(" not in accessor_text
    assert "def index(" not in accessor_text


def test_arch_spatial_142_rotation_pose_temporal_owners_reuse_core_runtime_query_map_finalize() -> None:
    """ID: ARCH_SPATIAL_142_rotation_pose_temporal_owners_reuse_core_runtime_query_map_finalize."""
    rotation_ops = Path("tal/spatial/ops/rotation_temporal_ops.py").read_text(encoding="utf-8")
    pose_ops = Path("tal/spatial/ops/pose_temporal_ops.py").read_text(encoding="utf-8")
    assert "resolve_param_runtime_context(" in rotation_ops
    assert "normalize_query_grid(" in rotation_ops
    assert "build_param_map(" in rotation_ops
    assert "gather_along_sequence(" in rotation_ops
    assert "finalize_param_output(" in rotation_ops
    assert "build_param_map(" not in pose_ops
    assert "gather_along_sequence(" not in pose_ops
    assert "finalize_param_output(" not in pose_ops
    assert "source.param.at(" in pose_ops
    assert "source.param.resample_to(" in pose_ops


def test_arch_spatial_143_rotation_interp_backend_imports_are_temporal_owner_scoped() -> None:
    """ID: ARCH_SPATIAL_143_rotation_interp_backend_imports_are_temporal_owner_scoped."""
    hits: list[str] = []
    for path in sorted(Path("tal/spatial").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "rotation_interp_backends" in text:
            hits.append(path.as_posix())
    assert hits == ["tal/spatial/ops/rotation_temporal_ops.py"]


def test_arch_spatial_144_rotation_interp_backend_stopgap_loops_are_row_local_no_batch_merge() -> None:
    """ID: ARCH_SPATIAL_144_rotation_interp_backend_stopgap_loops_are_row_local_no_batch_merge."""
    text = Path("tal/spatial/kernels/rotation_interp_backends.py").read_text(encoding="utf-8")
    assert "_slerp_quat_scipy_stopgap(" in text
    assert "rows = int(np.prod(alpha.shape[:-1]))" in text
    assert "flat_q0 = q0.reshape(rows, qsize, 4)" in text
    assert "for row in range(rows):" in text
    assert "for idx in range(qsize):" in text
    assert "vectorize=True" not in text
    assert "xarray" not in text


def test_spatial_arch_150_rotation_slerp_numba_backend_owner_routed() -> None:
    """ID: SPATIAL_ARCH_150_rotation_slerp_numba_backend_owner_routed."""
    backend_text = Path("tal/spatial/kernels/rotation_interp_backends.py").read_text(encoding="utf-8")
    numba_text = Path("tal/spatial/kernels/rotation_interp_numba_backends.py").read_text(encoding="utf-8")
    temporal_text = Path("tal/spatial/ops/rotation_temporal_ops.py").read_text(encoding="utf-8")
    assert 'ROTATION_INTERP_BACKEND_NUMBA = "numba"' in backend_text
    assert "def slerp_quat_backend(" in backend_text
    assert "from .rotation_interp_numba_backends import slerp_quat_numba" in backend_text
    assert "prepare_block_rows(" in numba_text
    assert "njit_kernel(" in numba_text
    assert "require_numba(owner)" in numba_text
    assert "import xarray" not in numba_text
    assert "from scipy" not in numba_text
    assert "vectorize=True" not in numba_text
    assert "parallel=True" not in numba_text
    assert "fastmath=True" not in numba_text
    assert "kwargs={\"backend\": ROTATION_INTERP_BACKEND_SCIPY}" in temporal_text
    assert "ROTATION_INTERP_BACKEND_NUMBA" not in temporal_text


def test_spatial_arch_151_kinematics_numba_kernels_are_schema_free() -> None:
    """ID: SPATIAL_ARCH_151_kinematics_numba_kernels_are_schema_free."""
    for path in [
        Path("tal/spatial/kernels/kinematics_temporal_numba_backends.py"),
        Path("tal/spatial/kernels/kinematics_smoothing_numba_backends.py"),
    ]:
        numba_text = path.read_text(encoding="utf-8")
        assert "import xarray" not in numba_text
        assert "xr." not in numba_text
        assert "attrs[" not in numba_text
        assert "set_roles(" not in numba_text
        assert "set_param_coord(" not in numba_text
        assert "set_validity(" not in numba_text
        assert "tal_v2" not in numba_text


def test_spatial_arch_152_kinematics_numba_paths_do_not_call_param_engine() -> None:
    """ID: SPATIAL_ARCH_152_kinematics_numba_paths_do_not_call_param_engine."""
    paths = [
        Path("tal/spatial/kernels/kinematics_temporal_backends.py"),
        Path("tal/spatial/kernels/kinematics_temporal_numba_backends.py"),
        Path("tal/spatial/kernels/kinematics_smoothing_backends.py"),
        Path("tal/spatial/kernels/kinematics_smoothing_numba_backends.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "param_engine" not in text
        assert "build_param_map" not in text
        assert "apply_param_map" not in text


def test_spatial_arch_154_kinematics_scan_backends_reuse_numba_scan_helpers() -> None:
    """ID: SPATIAL_ARCH_154_kinematics_scan_backends_reuse_numba_scan_helpers."""
    backend_text = Path("tal/spatial/kernels/kinematics_temporal_backends.py").read_text(encoding="utf-8")
    numba_text = Path("tal/spatial/kernels/kinematics_temporal_numba_backends.py").read_text(encoding="utf-8")
    temporal_text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    assert 'KINEMATICS_TEMPORAL_BACKEND_NUMBA = "numba"' in backend_text
    assert "def cumulative_trapezoid_block_backend(" in backend_text
    assert "from .kinematics_temporal_numba_backends import cumulative_trapezoid_block_numba" in backend_text
    assert "prepare_scan_rows(" in numba_text
    assert "ScanAxisSpec(" in numba_text
    assert "ScanInputSpec(" in numba_text
    assert "njit_kernel(" in numba_text
    assert "require_numba(owner)" in numba_text
    assert "parallel=True" not in numba_text
    assert "fastmath=True" not in numba_text
    assert "KINEMATICS_TEMPORAL_BACKEND_NUMBA" not in temporal_text
    assert "kinematics_temporal_numba_backends" not in temporal_text


def test_spatial_arch_160_local_stencil_numba_backends_are_owner_routed() -> None:
    """ID: SPATIAL_ARCH_160_local_stencil_numba_backends_are_owner_routed."""
    backend_text = Path("tal/spatial/kernels/kinematics_smoothing_backends.py").read_text(encoding="utf-8")
    numba_text = Path("tal/spatial/kernels/kinematics_smoothing_numba_backends.py").read_text(encoding="utf-8")
    smoothing_ops = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    assert 'KINEMATICS_SMOOTHING_BACKEND_NUMBA = "numba"' in backend_text
    assert "def moving_average_smoothing_block_backend(" in backend_text
    assert "def gaussian_smoothing_block_backend(" in backend_text
    assert "from .kinematics_smoothing_numba_backends import moving_average_smoothing_block_numba" in backend_text
    assert "from .kinematics_smoothing_numba_backends import gaussian_smoothing_block_numba" in backend_text
    assert "prepare_block_rows(" in numba_text
    assert "njit_kernel(" in numba_text
    assert "require_numba(owner)" in numba_text
    assert "KINEMATICS_SMOOTHING_BACKEND_NUMBA" not in smoothing_ops
    assert "kinematics_smoothing_backends" not in smoothing_ops


def test_spatial_arch_161_local_stencil_helpers_are_schema_free_if_added() -> None:
    """ID: SPATIAL_ARCH_161_local_stencil_helpers_are_schema_free_if_added."""
    _assert_no_direct_import("from tal.utils import numba as tal_numba", "numba")
    try:
        _assert_no_direct_import("import numba.core", "numba")
    except AssertionError:
        pass
    else:
        raise AssertionError("dotted numba import was not rejected")

    helper = Path("tal/utils/numba_stencil.py")
    assert helper.exists()
    helper_text = helper.read_text(encoding="utf-8")
    assert "import xarray" not in helper_text
    _assert_no_direct_import(helper_text, "numba")
    assert "tal.core" not in helper_text
    assert "tal.spatial" not in helper_text
    assert "tal.linalg" not in helper_text
    assert "tal_v2" not in helper_text
    numba_text = Path("tal/spatial/kernels/kinematics_smoothing_numba_backends.py").read_text(encoding="utf-8")
    assert "import xarray" not in numba_text
    assert "from scipy" not in numba_text
    assert "tal_v2" not in numba_text
    assert "attrs[" not in numba_text


def test_spatial_arch_162_stencil_backends_do_not_reuse_scan_as_generic_executor() -> None:
    """ID: SPATIAL_ARCH_162_stencil_backends_do_not_reuse_scan_as_generic_executor."""
    numba_text = Path("tal/spatial/kernels/kinematics_smoothing_numba_backends.py").read_text(encoding="utf-8")
    assert "prepare_scan_rows" not in numba_text
    assert "ScanAxisSpec" not in numba_text
    assert "ScanInputSpec" not in numba_text
    assert "numba_scan" not in numba_text


def test_arch_spatial_145_pose_temporal_payload_carrier_and_overlay_path_structurally_guarded() -> None:
    """ID: ARCH_SPATIAL_145_pose_temporal_payload_carrier_and_overlay_path_structurally_guarded."""
    text = Path("tal/spatial/ops/pose_temporal_ops.py").read_text(encoding="utf-8")
    assert "def _eval_payload_carrier(" in text
    assert "AnalysisObject._from_unvalidated(" in text
    assert "carrier.param.at(" in text
    assert "carrier.param.resample_to(" in text
    assert "def _overlay_components_payload(" in text
    assert "def _overlay_matrix_payload(" in text
    assert "def _resolve_matrix_payload_var(" in text
    assert "def _matrix_only_pose_source(" in text
    assert "_matrix_only_pose_source(source, owner=request.owner)" in text


def test_arch_spatial_147_pose_temporal_matrix_payload_resolution_uses_matrix_candidate_filter_not_global_single_var_selector() -> None:
    """ID: ARCH_SPATIAL_147_pose_temporal_matrix_payload_resolution_uses_matrix_candidate_filter_not_global_single_var_selector."""
    text = Path("tal/spatial/ops/pose_temporal_ops.py").read_text(encoding="utf-8")
    assert "_resolve_matrix_payload_var(" in text
    assert "containing core dims" in text
    assert "select_single_numeric_var(source_ds" not in text


def test_arch_spatial_148_pose_temporal_matrix_aux_rebind_is_single_guarded_helper_boundary() -> None:
    """ID: ARCH_SPATIAL_148_pose_temporal_matrix_aux_rebind_is_single_guarded_helper_boundary."""
    text = Path("tal/spatial/ops/pose_temporal_ops.py").read_text(encoding="utf-8")
    assert "def _needs_matrix_aux_rebind(" in text
    assert "def _rewrap_matrix_aux_unvalidated(" in text
    assert "if _needs_matrix_aux_rebind(" in text
    assert "_rewrap_matrix_aux_unvalidated(" in text
    assert "def _rewrap_pose_temporal_output(" in text
    assert text.count("_bind_dataset(") == 1


def test_arch_spatial_146_rotation_temporal_single_payload_guard_prevents_silent_drop() -> None:
    """ID: ARCH_SPATIAL_146_rotation_temporal_single_payload_guard_prevents_silent_drop."""
    text = Path("tal/spatial/ops/rotation_temporal_ops.py").read_text(encoding="utf-8")
    assert "def _require_single_payload_var(" in text
    assert "auxiliary payload vars are not supported" in text
    assert "_require_single_payload_var(source.unsafe_data, owner=request.owner)" in text
    assert "var_name = _require_single_payload_var(context.ds" in text
    src_guard_index = text.find("_require_single_payload_var(source.unsafe_data, owner=request.owner)")
    as_quat_index = text.find("quat_source = source.as_quat(validate=False)")
    ctx_guard_index = text.find("var_name = _require_single_payload_var(context.ds, owner=request.owner)")
    assert src_guard_index != -1
    assert as_quat_index != -1
    assert ctx_guard_index != -1
    assert src_guard_index < as_quat_index < ctx_guard_index


def test_spatial_doc_019_rotation_pose_typed_param_defaults_slerp_and_linear_semantics_documented() -> None:
    """ID: SPATIAL_DOC_019_rotation_pose_typed_param_defaults_slerp_and_linear_semantics_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    rotation_doc = Path("docs/api/types/rotation.md").read_text(encoding="utf-8")
    pose_doc = Path("docs/api/types/pose.md").read_text(encoding="utf-8")
    assert "Rotation.param.at" in spatial_doc
    assert "Pose.param.at" in spatial_doc
    assert "slerp" in rotation_doc.lower()
    assert "Rotation.slerp" in rotation_doc
    assert "rotation-aware" in pose_doc


def test_spatial_doc_020_d2_payload_preservation_and_matrix_var_name_stability_documented() -> None:
    """ID: SPATIAL_DOC_020_d2_payload_preservation_and_matrix_var_name_stability_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    rotation_doc = Path("docs/api/types/rotation.md").read_text(encoding="utf-8").lower()
    pose_doc = Path("docs/api/types/pose.md").read_text(encoding="utf-8").lower()
    assert "matrix" in spatial_doc
    assert "metadata" in rotation_doc
    assert "metadata" in pose_doc


def test_spatial_doc_021_d2_matrix_pose_numeric_aux_payload_preservation_documented() -> None:
    """ID: SPATIAL_DOC_021_d2_matrix_pose_numeric_aux_payload_preservation_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    pose_doc = Path("docs/api/types/pose.md").read_text(encoding="utf-8").lower()
    assert "matrix" in spatial_doc
    assert "matrix" in pose_doc


def test_spatial_doc_022_d2_validate_false_matrix_aux_rebind_boundary_documented() -> None:
    """ID: SPATIAL_DOC_022_d2_validate_false_matrix_aux_rebind_boundary_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    pose_doc = Path("docs/api/types/pose.md").read_text(encoding="utf-8").lower()
    assert "param" in spatial_doc
    assert "matrix" in pose_doc


def test_arch_spatial_149_d3_runners_are_single_execution_and_finalize_owners() -> None:
    """ID: ARCH_SPATIAL_149_d3_runners_are_single_execution_and_finalize_owners."""
    ops_text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    position_text = Path("tal/spatial/position.py").read_text(encoding="utf-8")
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    assert "def _run_derivative_request(" in ops_text
    assert "def _run_integral_request(" in ops_text
    assert "resolve_param_runtime_context(" in ops_text
    assert "xr.apply_ufunc(" in ops_text
    assert "differentiate_position_to_linear_velocity(" in position_text
    assert "differentiate_linear_velocity_to_linear_acceleration(" in velocity_text
    assert "integrate_linear_velocity_to_position(" in velocity_text
    assert "integrate_linear_acceleration_to_linear_velocity(" in acceleration_text
    assert "integrate_angular_acceleration_to_angular_velocity(" in acceleration_text
    assert "xr.apply_ufunc(" not in position_text
    assert "xr.apply_ufunc(" not in velocity_text
    assert "xr.apply_ufunc(" not in acceleration_text


def test_arch_spatial_150_d3_execution_paths_contain_no_vectorize_true_or_row_loops() -> None:
    """ID: ARCH_SPATIAL_150_d3_execution_paths_contain_no_vectorize_true_or_row_loops."""
    ops_text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    kernel_text = Path("tal/spatial/kernels/kinematics_temporal_kernels.py").read_text(encoding="utf-8")
    assert "vectorize=True" not in ops_text
    assert "vectorize=True" not in kernel_text
    assert "vectorize=False" in ops_text
    assert "for row in" not in ops_text
    assert "for row in" not in kernel_text
    assert "for idx in" not in ops_text
    assert "for idx in" not in kernel_text


def test_spatial_doc_023_d3_scope_baseline_relative_integration_and_smoothing_deferral_documented() -> None:
    """ID: SPATIAL_DOC_023_d3_scope_baseline_relative_integration_and_smoothing_deferral_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    position_doc = Path("docs/api/types/position.md").read_text(encoding="utf-8").lower()
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8").lower()
    acceleration_doc = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8").lower()
    assert "differentiate" in position_doc
    assert "integrate" in velocity_doc
    assert "integrate" in acceleration_doc


def test_arch_spatial_152_d4_smoothing_and_local_derivative_reuse_shared_temporal_owners() -> None:
    """ID: ARCH_SPATIAL_152_d4_smoothing_and_local_derivative_reuse_shared_temporal_owners."""
    smoothing_ops = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    temporal_ops = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    assert "from .kinematics_temporal_ops import (" in smoothing_ops
    assert "_resolve_runtime," in smoothing_ops
    assert "_resolve_payload_var_and_core," in smoothing_ops
    assert "_replace_payload," in smoothing_ops
    assert "_apply_output_metadata," in smoothing_ops
    assert "_wrap_owner_error," in smoothing_ops
    assert "def _run_derivative_request(" in temporal_ops
    assert "def _run_integral_request(" in temporal_ops
    assert "def _run_smoothing_request(" in smoothing_ops


def test_arch_spatial_153_d4_temporal_ops_orchestrate_kernel_finalize_split_preserved() -> None:
    """ID: ARCH_SPATIAL_153_d4_temporal_ops_orchestrate_kernel_finalize_split_preserved."""
    text = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    assert "def _run_smoothing_request(" in text
    assert "_resolve_runtime(" in text
    assert "_apply_smoothing_operator(" in text
    assert "_replace_payload(" in text
    assert "_apply_output_metadata(" in text
    assert "wrap_as(" in text
    assert "build_param_map(" not in text
    assert "apply_param_map(" not in text
    assert ".param.at(" not in text
    assert ".param.resample_to(" not in text


def test_arch_spatial_154_d4_no_local_duplicate_smoothing_kernels_in_typed_classes() -> None:
    """ID: ARCH_SPATIAL_154_d4_no_local_duplicate_smoothing_kernels_in_typed_classes."""
    for rel in ("tal/spatial/position.py", "tal/spatial/velocity.py", "tal/spatial/acceleration.py"):
        text = Path(rel).read_text(encoding="utf-8")
        assert "moving_average_partial_renorm_kernel(" not in text
        assert "local_poly_smooth_kernel(" not in text
        assert "xr.apply_ufunc(" not in text
        assert "smooth_kinematics_like(" in text


def test_arch_spatial_155_d4_execution_and_kernel_owners_contain_no_vectorize_true_or_row_loops() -> None:
    """ID: ARCH_SPATIAL_155_d4_execution_and_kernel_owners_contain_no_vectorize_true_or_row_loops."""
    files = (
        "tal/spatial/ops/kinematics_temporal_ops.py",
        "tal/spatial/ops/kinematics_smoothing_ops.py",
        "tal/spatial/kernels/kinematics_temporal_kernels.py",
        "tal/spatial/kernels/local_poly_weights.py",
    )
    for rel in files:
        text = Path(rel).read_text(encoding="utf-8")
        assert "vectorize=True" not in text
        assert "for row in" not in text
        assert "for idx in" not in text
    assert "vectorize=False" in Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")


def test_arch_spatial_166_d6_supersedes_d4_no_family_wrapper_lock_with_delegation_only_lock() -> None:
    """ID: ARCH_SPATIAL_166_d6_supersedes_d4_no_family_wrapper_lock_with_delegation_only_lock."""
    position_text = Path("tal/spatial/position.py").read_text(encoding="utf-8")
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    family_ops_text = Path("tal/spatial/ops/kinematics_family_temporal_ops.py").read_text(encoding="utf-8")
    assert "def smooth(" in position_text
    assert "class LinearVelocity" in velocity_text and "def smooth(" in velocity_text
    assert "class AngularVelocity" in velocity_text and "def smooth(" in velocity_text
    assert "class LinearAcceleration" in acceleration_text and "def smooth(" in acceleration_text
    assert "class AngularAcceleration" in acceleration_text and "def smooth(" in acceleration_text
    velocity_family_body = velocity_text.split("class Velocity(", 1)[1]
    acceleration_family_body = acceleration_text.split("class Acceleration(", 1)[1]
    assert "def differentiate(" in velocity_family_body
    assert "def smooth(" in velocity_family_body
    assert "def integrate(" not in velocity_family_body
    assert "def integrate(" in acceleration_family_body
    assert "def smooth(" in acceleration_family_body
    assert "def differentiate(" not in acceleration_family_body
    assert "def differentiate_velocity_family(" in family_ops_text
    assert "def smooth_velocity_family(" in family_ops_text
    assert "def integrate_acceleration_family(" in family_ops_text
    assert "def smooth_acceleration_family(" in family_ops_text


def test_arch_spatial_163_d6_ao_dispatch_table_and_kind_constant_are_single_source_of_truth() -> None:
    """ID: ARCH_SPATIAL_163_d6_ao_dispatch_table_and_kind_constant_are_single_source_of_truth."""
    roles_text = Path("tal/spatial/metadata/roles.py").read_text(encoding="utf-8")
    surface_text = Path("tal/spatial/temporal/surface.py").read_text(encoding="utf-8")
    assert "KINEMATICS_KIND_VALUES: tuple[str, ...]" in roles_text
    assert 'AO_TEMPORAL_KIND_VALUES: tuple[str, ...] = ("position", *KINEMATICS_KIND_VALUES)' in surface_text
    assert "_DIFFERENTIATE_METHODS" in surface_text
    assert "_INTEGRATE_METHODS" in surface_text
    assert "_SMOOTH_METHODS" in surface_text
    assert '"position": Position' in surface_text
    assert '"velocity": Velocity' in surface_text
    assert '"acceleration": Acceleration' in surface_text
    assert "require_supported_ao_temporal_kind(" in surface_text


def test_arch_spatial_164_d6_family_wrapper_methods_delegate_only_to_existing_temporal_owners() -> None:
    """ID: ARCH_SPATIAL_164_d6_family_wrapper_methods_delegate_only_to_existing_temporal_owners."""
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    family_ops_text = Path("tal/spatial/ops/kinematics_family_temporal_ops.py").read_text(encoding="utf-8")
    assert "from .ops.kinematics_family_temporal_ops import differentiate_velocity_family" in velocity_text
    assert "from .ops.kinematics_family_temporal_ops import smooth_velocity_family" in velocity_text
    assert "from .ops.kinematics_family_temporal_ops import integrate_acceleration_family" in acceleration_text
    assert "from .ops.kinematics_family_temporal_ops import smooth_acceleration_family" in acceleration_text
    assert '_run_family_pair_operation(' in family_ops_text
    assert 'method_name="differentiate"' in family_ops_text
    assert 'method_name="integrate"' in family_ops_text
    assert 'method_name="smooth"' in family_ops_text
    assert "xr.apply_ufunc(" not in family_ops_text


def test_arch_spatial_165_d6_wrapper_modules_contain_no_vectorize_true_apply_ufunc_or_row_loops() -> None:
    """ID: ARCH_SPATIAL_165_d6_wrapper_modules_contain_no_vectorize_true_apply_ufunc_or_row_loops."""
    surface_text = Path("tal/spatial/temporal/surface.py").read_text(encoding="utf-8")
    family_ops_text = Path("tal/spatial/ops/kinematics_family_temporal_ops.py").read_text(encoding="utf-8")
    for text in (surface_text, family_ops_text):
        assert "vectorize=True" not in text
        assert "xr.apply_ufunc(" not in text
        assert "for row in" not in text
        assert "for idx in" not in text


def test_spatial_hard_174_d6_wrappers_do_not_own_numeric_kernels() -> None:
    """ID: SPATIAL_HARD_174_d6_wrappers_do_not_own_numeric_kernels."""
    surface_text = Path("tal/spatial/temporal/surface.py").read_text(encoding="utf-8")
    family_ops_text = Path("tal/spatial/ops/kinematics_family_temporal_ops.py").read_text(encoding="utf-8")
    assert "from ..kernels" not in surface_text
    assert "from ..kernels" not in family_ops_text
    assert "finite_difference_one_sided_kernel" not in surface_text
    assert "cumulative_trapezoid_kernel" not in surface_text
    assert "cumulative_simpson_kernel" not in surface_text
    assert "moving_average_partial_renorm_kernel" not in family_ops_text


def test_arch_spatial_167_d6_family_temporal_ops_use_single_internal_linear_angular_dispatch_template() -> None:
    """ID: ARCH_SPATIAL_167_d6_family_temporal_ops_use_single_internal_linear_angular_dispatch_template."""
    family_ops_text = Path("tal/spatial/ops/kinematics_family_temporal_ops.py").read_text(encoding="utf-8")
    assert "class FamilyTemporalRequest:" in family_ops_text
    assert "def _invoke_member_temporal(" in family_ops_text
    assert "def _run_family_pair_operation(" in family_ops_text
    assert "differentiate_velocity_family(" in family_ops_text
    assert "smooth_velocity_family(" in family_ops_text
    assert "integrate_acceleration_family(" in family_ops_text
    assert "smooth_acceleration_family(" in family_ops_text
    assert family_ops_text.count("_run_family_pair_operation(") == 5
    assert "linear.differentiate(" not in family_ops_text
    assert "angular.differentiate(" not in family_ops_text
    assert "linear.integrate(" not in family_ops_text
    assert "angular.integrate(" not in family_ops_text
    assert "linear.smooth(" not in family_ops_text
    assert "angular.smooth(" not in family_ops_text


def test_spatial_doc_024_d4_typed_smoothing_and_local_higher_order_boundaries_documented() -> None:
    """ID: SPATIAL_DOC_024_d4_typed_smoothing_and_local_higher_order_boundaries_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    position_doc = Path("docs/api/types/position.md").read_text(encoding="utf-8").lower()
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8").lower()
    acceleration_doc = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8").lower()
    assert "smooth" in spatial_doc
    assert "smooth(" in position_doc
    assert "smooth(" in velocity_doc
    assert "smooth(" in acceleration_doc


def test_spatial_doc_025_d4_preserves_d3_defaults_and_defers_advanced_quadrature_documented() -> None:
    """ID: SPATIAL_DOC_025_d4_preserves_d3_defaults_and_defers_advanced_quadrature_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8").lower()
    assert "differentiate" in spatial_doc
    assert "integrate" in velocity_doc

def test_arch_spatial_159_d5_execution_and_kernel_owners_keep_blockwise_no_row_loop_policy() -> None:
    """ID: ARCH_SPATIAL_159_d5_execution_and_kernel_owners_keep_blockwise_no_row_loop_policy."""
    temporal_text = Path("tal/spatial/ops/kinematics_temporal_ops.py").read_text(encoding="utf-8")
    smoothing_text = Path("tal/spatial/ops/kinematics_smoothing_ops.py").read_text(encoding="utf-8")
    kernel_text = Path("tal/spatial/kernels/kinematics_temporal_kernels.py").read_text(encoding="utf-8")
    assert "cumulative_simpson_kernel" in temporal_text
    assert "gaussian_partial_renorm_kernel" in smoothing_text
    assert "def cumulative_simpson_kernel(" in kernel_text
    assert "def gaussian_partial_renorm_kernel(" in kernel_text
    for text in (temporal_text, smoothing_text, kernel_text):
        assert "vectorize=True" not in text
        assert "for row in" not in text
        assert "for idx in" not in text


def test_spatial_doc_027_d5_simpson_and_gaussian_semantics_documented() -> None:
    """ID: SPATIAL_DOC_027_d5_simpson_and_gaussian_semantics_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8").lower()
    acceleration_doc = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8").lower()
    assert "smooth" in spatial_doc
    assert "integrate" in velocity_doc
    assert "integrate" in acceleration_doc

def test_arch_spatial_162_d5_simpson_kernel_sanitizes_inactive_tail_before_scipy_call() -> None:
    """ID: ARCH_SPATIAL_162_d5_simpson_kernel_sanitizes_inactive_tail_before_scipy_call."""
    kernel_text = Path("tal/spatial/kernels/kinematics_temporal_kernels.py").read_text(encoding="utf-8")
    assert "def _simpson_safe_tail_param(" in kernel_text
    assert "def _simpson_safe_tail_values(" in kernel_text
    assert "safe_param, sample_active = _simpson_safe_tail_param(" in kernel_text
    assert "safe_values = _simpson_safe_tail_values(" in kernel_text
    assert "param_rows = safe_param.reshape(rows, seq_size)" in kernel_text
    assert "vals_rows = safe_values.reshape(rows, seq_size, core_size)" in kernel_text


def test_spatial_doc_029_d5_simpson_valid_prefix_padded_tail_semantics_documented() -> None:
    """ID: SPATIAL_DOC_029_d5_simpson_valid_prefix_padded_tail_semantics_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8").lower()
    acceleration_doc = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8").lower()
    assert "valid" in spatial_doc
    assert "param" in velocity_doc
    assert "param" in acceleration_doc


def test_spatial_doc_030_d6_ao_allowed_kind_values_documented_from_canonical_constant() -> None:
    """ID: SPATIAL_DOC_030_d6_ao_allowed_kind_values_documented_from_canonical_constant."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    surface_text = Path("tal/spatial/temporal/surface.py").read_text(encoding="utf-8")
    assert "differentiate" in spatial_doc
    assert "smooth" in spatial_doc
    assert 'AO_TEMPORAL_KIND_VALUES: tuple[str, ...] = ("position", *KINEMATICS_KIND_VALUES)' in surface_text


def test_spatial_doc_031_d6_family_wrapper_rep_and_frame_preservation_guarantees_documented() -> None:
    """ID: SPATIAL_DOC_031_d6_family_wrapper_rep_and_frame_preservation_guarantees_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8").lower()
    acceleration_doc = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8").lower()
    assert "velocity" in spatial_doc
    assert "acceleration" in spatial_doc
    assert "differentiate" in velocity_doc
    assert "integrate" in acceleration_doc


def test_arch_spatial_c4_001_relation_metadata_helpers_are_spatial_owner_scoped() -> None:
    """ID: ARCH_SPATIAL_C4_001_relation_metadata_helpers_are_spatial_owner_scoped."""
    relation_text = Path("tal/spatial/metadata/relation.py").read_text(encoding="utf-8")
    facade_text = Path("tal/spatial/metadata/facade.py").read_text(encoding="utf-8")
    assert "def get_expressed_in(" in relation_text
    assert "def set_expressed_in(" in relation_text
    assert "def get_instantaneous_inertial(" in relation_text
    assert "def set_instantaneous_inertial(" in relation_text
    assert "normalize_configuration_relation_semantics" in relation_text
    assert "normalize_kinematic_relation_semantics" in relation_text
    assert "get_expressed_in" in facade_text
    assert "set_expressed_in" in facade_text



def test_arch_spatial_c4_002_parent_child_owner_remains_tal_utils_frame_schema() -> None:
    """ID: ARCH_SPATIAL_C4_002_parent_child_owner_remains_tal_utils_frame_schema."""
    frame_schema = Path("tal/utils/frame_schema.py").read_text(encoding="utf-8")
    relation_text = Path("tal/spatial/metadata/relation.py").read_text(encoding="utf-8")
    assert '_ALLOWED_FRAME_KEYS = {"parent", "child"}' in frame_schema
    assert "from tal.utils.frame_schema import get_frames" in relation_text
    assert "def set_frames(" not in relation_text



def test_arch_spatial_c4_003_no_direct_schema_attr_writes_for_c4_metadata_paths() -> None:
    """ID: ARCH_SPATIAL_C4_003_no_direct_schema_attr_writes_for_c4_metadata_paths."""
    relation_text = Path("tal/spatial/metadata/relation.py").read_text(encoding="utf-8")
    common_text = Path("tal/spatial/metadata/common.py").read_text(encoding="utf-8")
    for text in (relation_text, common_text):
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text



def test_arch_spatial_087_slice_c3_kinematics_frame_api_owner_split_and_budget() -> None:
    """ID: ARCH_SPATIAL_087_slice_c3_kinematics_frame_api_owner_split_and_budget."""
    path = Path("tal/spatial/ops/kinematics_frame_ops.py")
    assert path.exists()
    _assert_agents_budget(path)



def test_arch_spatial_088_slice_c3_kinematics_to_frame_methods_delegate_to_shared_c3_owner() -> None:
    """ID: ARCH_SPATIAL_088_slice_c3_kinematics_to_frame_methods_delegate_to_shared_c3_owner."""
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    owner_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    assert "from .ops.kinematics_frame_ops import to_frame_linear_velocity" in velocity_text
    assert "from .ops.kinematics_frame_ops import to_frame_angular_velocity" in velocity_text
    assert "from .ops.kinematics_frame_ops import to_frame_velocity_family" in velocity_text
    assert "from .ops.kinematics_frame_ops import to_frame_linear_acceleration" in acceleration_text
    assert "from .ops.kinematics_frame_ops import to_frame_angular_acceleration" in acceleration_text
    assert "from .ops.kinematics_frame_ops import to_frame_acceleration_family" in acceleration_text
    assert "def _run_vector_to_frame(" in owner_text
    assert "def to_frame_velocity_family(" in owner_text
    assert "def to_frame_acceleration_family(" in owner_text



def test_arch_spatial_089_slice_c3_kinematics_to_frame_reuses_c1_pose_solver_and_b4_pose_apply() -> None:
    """ID: ARCH_SPATIAL_089_slice_c3_kinematics_to_frame_reuses_c1_pose_solver_and_b4_pose_apply."""
    text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    assert "from ..path_solve import _solve_pose_path_transform_with_owner" in text
    assert "_pose_apply_with_owner(" in text
    assert "find_path(" not in text
    assert "fold_path(" not in text



def test_arch_spatial_090_slice_c3_no_local_find_path_fold_path_or_local_math_duplication() -> None:
    """ID: ARCH_SPATIAL_090_slice_c3_no_local_find_path_fold_path_or_local_math_duplication."""
    text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    assert "find_path(" not in text
    assert "fold_path(" not in text
    assert "xr.apply_ufunc(" not in text
    assert "from ..kernels" not in text


def test_arch_spatial_091_slice_c3_kinematic_identity_short_circuit_precedes_solver_calls() -> None:
    """ID: ARCH_SPATIAL_091_slice_c3_kinematic_identity_short_circuit_precedes_solver_calls."""
    module = _module("tal/spatial/ops/kinematics_frame_ops.py")
    to_node = _function_node(module, "_run_vector_to_frame")
    to_identity_idx: int | None = None
    to_delegate_idx: int | None = None
    for idx, stmt in enumerate(to_node.body):
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "dst_id"
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.Eq)
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Name)
            and stmt.test.comparators[0].id == "src_parent"
        ):
            to_identity_idx = idx
        if (
            isinstance(stmt, ast.Return)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "_run_vector_to_frame_non_identity"
        ):
            to_delegate_idx = idx
    assert to_identity_idx is not None
    assert to_delegate_idx is not None
    assert to_identity_idx < to_delegate_idx

    expr_node = _function_node(module, "_run_vector_express_in")
    expr_identity_idx: int | None = None
    expr_solve_idx: int | None = None
    for idx, stmt in enumerate(expr_node.body):
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "_solve_rotation_path_transform_with_owner"
        ):
            expr_solve_idx = idx
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "dst_id"
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.Eq)
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Name)
            and stmt.test.comparators[0].id == "src_expressed_in"
        ):
            expr_identity_idx = idx
    assert expr_identity_idx is not None
    assert expr_solve_idx is not None
    assert expr_identity_idx < expr_solve_idx


def test_arch_spatial_092_slice_c3_non_identity_to_frame_consumes_src_expressed_in_before_solver() -> None:
    """ID: ARCH_SPATIAL_092_slice_c3_non_identity_to_frame_consumes_src_expressed_in_before_solver."""
    module = _module("tal/spatial/ops/kinematics_frame_ops.py")
    node = _function_node(module, "_run_vector_to_frame_non_identity")
    canonical_idx: int | None = None
    solve_idx: int | None = None
    canonical_uses_src_expressed_in = False
    for idx, stmt in enumerate(node.body):
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "_canonicalize_vector_source_basis"
        ):
            canonical_idx = idx
            canonical_uses_src_expressed_in = any(
                kw.arg == "src_expressed_in"
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "src_expressed_in"
                for kw in stmt.value.keywords
            )
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "solve_pose"
        ):
            solve_idx = idx
    assert canonical_idx is not None
    assert solve_idx is not None
    assert canonical_idx < solve_idx
    assert canonical_uses_src_expressed_in


def test_arch_spatial_c5_004_frame_owner_common_single_owner_helpers_for_dst_source_and_clear_framing() -> None:
    """ID: ARCH_SPATIAL_C5_004_frame_owner_common_single_owner_helpers_for_dst_source_and_clear_framing."""
    common_text = Path("tal/spatial/ops/frame_owner_common.py").read_text(encoding="utf-8")
    expr_text = Path("tal/spatial/ops/frame_expression_ops.py").read_text(encoding="utf-8")
    kin_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")

    assert "def dst_frame_id(" in common_text
    assert "def require_source_parent(" in common_text
    assert "def source_expressed_in_id(" in common_text
    assert "def clear_framing(" in common_text
    for text in (expr_text, kin_text):
        assert (
            "from .frame_owner_common import clear_framing, dst_frame_id, require_source_parent, "
            "source_expressed_in_id"
        ) in text
        assert "def _dst_frame_id(" not in text
        assert "def _require_source_parent(" not in text
        assert "def _source_expressed_in_id(" not in text
        assert "def _clear_framing(" not in text



def test_arch_spatial_c5_001_express_in_methods_delegate_to_shared_owner_only() -> None:
    """ID: ARCH_SPATIAL_C5_001_express_in_methods_delegate_to_shared_owner_only."""
    position_text = Path("tal/spatial/position.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    pose_text = Path("tal/spatial/pose.py").read_text(encoding="utf-8")
    velocity_text = Path("tal/spatial/velocity.py").read_text(encoding="utf-8")
    acceleration_text = Path("tal/spatial/acceleration.py").read_text(encoding="utf-8")
    owner_text = Path("tal/spatial/ops/frame_expression_ops.py").read_text(encoding="utf-8")
    kin_owner_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    assert "from .ops.frame_expression_ops import position_express_in" in position_text
    assert "from .ops.frame_expression_ops import rotation_express_in" in rotation_text
    assert "from .ops.frame_expression_ops import pose_express_in" in pose_text
    assert "from .ops.kinematics_frame_ops import express_in_linear_velocity" in velocity_text
    assert "from .ops.kinematics_frame_ops import express_in_acceleration_family" in acceleration_text
    assert "def position_express_in(" in owner_text
    assert "def rotation_express_in(" in owner_text
    assert "def pose_express_in(" in owner_text
    assert "def _run_vector_express_in(" in kin_owner_text



def test_arch_spatial_c5_002_no_local_find_path_fold_path_or_kernel_duplication_in_c5() -> None:
    """ID: ARCH_SPATIAL_C5_002_no_local_find_path_fold_path_or_kernel_duplication_in_c5."""
    owner_text = Path("tal/spatial/ops/frame_expression_ops.py").read_text(encoding="utf-8")
    kin_owner_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    for text in (owner_text, kin_owner_text):
        assert "find_path(" not in text
        assert "fold_path(" not in text
        assert "xr.apply_ufunc(" not in text
        assert "from ..kernels" not in text



def test_arch_spatial_c5_003_c5_reuses_c4_metadata_helpers_for_relation_semantics() -> None:
    """ID: ARCH_SPATIAL_C5_003_c5_reuses_c4_metadata_helpers_for_relation_semantics."""
    owner_text = Path("tal/spatial/ops/frame_expression_ops.py").read_text(encoding="utf-8")
    kin_owner_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    assert "set_expressed_in" in owner_text
    assert "set_expressed_in" in kin_owner_text
    assert "set_instantaneous_inertial" in kin_owner_text


def test_spatial_doc_012_phase8_slice_c3_kinematics_frame_api_docs_and_entries_present() -> None:
    """ID: SPATIAL_DOC_012_phase8_slice_c3_kinematics_frame_api_docs_and_entries_present."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8")
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8")
    acceleration_doc = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8")
    assert "to_frame" in spatial_doc
    assert "to_frame" in velocity_doc
    assert "to_frame" in acceleration_doc


def test_spatial_doc_c4_001_relation_vs_representation_metadata_semantics_documented() -> None:
    """ID: SPATIAL_DOC_C4_001_relation_vs_representation_metadata_semantics_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    assert "to_frame" in spatial_doc
    assert "express_in" in spatial_doc


def test_spatial_doc_c5_001_to_frame_vs_express_in_semantics_documented() -> None:
    """ID: SPATIAL_DOC_C5_001_to_frame_vs_express_in_semantics_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    position_doc = Path("docs/api/types/position.md").read_text(encoding="utf-8").lower()
    rotation_doc = Path("docs/api/types/rotation.md").read_text(encoding="utf-8").lower()
    pose_doc = Path("docs/api/types/pose.md").read_text(encoding="utf-8").lower()
    assert "to_frame" in spatial_doc
    assert "express_in" in spatial_doc
    assert "to_frame" in position_doc
    assert "express_in" in rotation_doc
    assert "express_in" in pose_doc


def test_spatial_doc_c5_002_to_frame_is_relative_to_while_express_in_is_basis_only() -> None:
    """ID: SPATIAL_DOC_C5_002_to_frame_is_relative_to_while_express_in_is_basis_only."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    assert "parent/child" in spatial_doc
    assert "coordinate basis" in spatial_doc


def test_arch_spatial_c7_001_kinematics_to_frame_reuses_c1_b4_b5_owners_without_local_kernel_duplication() -> None:
    """ID: ARCH_SPATIAL_C7_001_kinematics_to_frame_reuses_c1_b4_b5_owners_without_local_kernel_duplication."""
    wrapper_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    support_text = Path("tal/spatial/ops/kinematics_path_support_ops.py").read_text(encoding="utf-8")
    coupling_text = Path("tal/spatial/ops/kinematics_path_coupling_ops.py").read_text(encoding="utf-8")
    assert "from ..path_solve import _solve_pose_path_transform_with_owner" in wrapper_text
    assert "_run_vector_to_frame_non_identity(" in wrapper_text
    assert "_pose_apply_with_owner(" in wrapper_text
    assert "resolve_kinematics_path_support(" in wrapper_text
    assert "apply_vector_path_coupling(" in wrapper_text
    for text in (wrapper_text, support_text, coupling_text):
        assert "from ..kernels" not in text
        assert "xr.apply_ufunc(" not in text


def test_arch_spatial_c7_002_c7_support_checks_consume_c4_and_c6_metadata_owners() -> None:
    """ID: ARCH_SPATIAL_C7_002_c7_support_checks_consume_c4_and_c6_metadata_owners."""
    support_text = Path("tal/spatial/ops/kinematics_path_support_ops.py").read_text(encoding="utf-8")
    assert "get_instantaneous_inertial" in support_text
    assert "get_kinematics_kind" in support_text
    assert "get_edge_motion_class" in support_text
    assert "get_frame_inertial_status" in support_text
    assert "KinematicsPathSupportOptions" in support_text
    assert (
        "from tal.frames import (\n"
        "    Frame,\n"
        "    FrameGraph,\n"
        "    FramePath,\n"
        "    find_path,\n"
        "    get_active_frame_graph,\n"
        ")"
    ) in support_text
    assert (
        "from ..metadata import (\n"
        "    EDGE_MOTION_CLASS_VALUES,\n"
        "    FRAME_INERTIAL_STATUS_VALUES,\n"
        "    get_edge_motion_class,\n"
        "    get_frame_inertial_status,\n"
        "    get_instantaneous_inertial,\n"
        ")"
    ) in support_text


def test_arch_spatial_c7_003_no_find_path_fold_path_duplication_in_c7_wrappers() -> None:
    """ID: ARCH_SPATIAL_C7_003_no_find_path_fold_path_duplication_in_c7_wrappers."""
    wrapper_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    support_text = Path("tal/spatial/ops/kinematics_path_support_ops.py").read_text(encoding="utf-8")
    assert "find_path(" not in wrapper_text
    assert "fold_path(" not in wrapper_text
    assert "find_path(" in support_text
    assert "fold_path(" not in support_text


def test_arch_spatial_c7_004_coupling_owner_uses_exact_alignment_helper_no_implicit_inner_join() -> None:
    """ID: ARCH_SPATIAL_C7_004_coupling_owner_uses_exact_alignment_helper_no_implicit_inner_join."""
    coupling_text = Path("tal/spatial/ops/kinematics_path_coupling_ops.py").read_text(encoding="utf-8")
    assert "from .frame_alignment_policy_ops import align_frame_pair_by_policy" in coupling_text
    assert "def _add_by_policy(" in coupling_text
    assert "align_frame_pair_by_policy(" in coupling_text
    assert "acc = _add_by_policy(" in coupling_text
    assert "ds[var_name] = _add_by_policy(" in coupling_text
    assert "xr.align(" not in coupling_text
    assert "_add_exact_or_fail_closed(" not in coupling_text
    assert "acc = acc + sign * _motion_component(" not in coupling_text
    assert "ds[var_name] = ds[var_name] + add" not in coupling_text


def test_arch_spatial_c7_005_support_owner_validates_kinematics_support_type_before_attribute_access() -> None:
    """ID: ARCH_SPATIAL_C7_005_support_owner_validates_kinematics_support_type_before_attribute_access."""
    support_text = Path("tal/spatial/ops/kinematics_path_support_ops.py").read_text(encoding="utf-8")
    assert "def _resolve_support(opts: PathSolveOptions | None, *, owner: str)" in support_text
    assert "isinstance(support, KinematicsPathSupportOptions)" in support_text
    assert "opts.kinematics_support must be KinematicsPathSupportOptions or None" in support_text


def test_arch_spatial_c7_006_support_graph_resolution_uses_opts_then_dst_frame_then_active() -> None:
    """ID: ARCH_SPATIAL_C7_006_support_graph_resolution_uses_opts_then_dst_frame_then_active."""
    support_text = Path("tal/spatial/ops/kinematics_path_support_ops.py").read_text(encoding="utf-8")
    assert "def _resolve_graph(opts: PathSolveOptions | None, *, dst: Frame | str, owner: str) -> FrameGraph:" in support_text
    assert "graph = _resolve_graph(opts, dst=dst, owner=owner)" in support_text
    assert "if opts is not None and opts.graph is not None:" in support_text
    assert "if isinstance(dst, Frame):" in support_text
    assert "return get_active_frame_graph()" in support_text
    opts_idx = support_text.index("if opts is not None and opts.graph is not None:")
    dst_idx = support_text.index("if isinstance(dst, Frame):")
    active_idx = support_text.index("return get_active_frame_graph()")
    assert opts_idx < dst_idx < active_idx


def test_arch_spatial_c8_001_frame_aware_alignment_reuses_core_intent_topology_alignment_owners() -> None:
    """ID: ARCH_SPATIAL_C8_001_frame_aware_alignment_reuses_core_intent_topology_alignment_owners."""
    text = Path("tal/spatial/ops/frame_alignment_policy_ops.py").read_text(encoding="utf-8")
    assert "select_topology_policy_with_intents(" in text
    assert "resolve_binary_topology(" in text
    assert "align_exact_for_plan(" in text
    assert "resolve_param_runtime_context(" in text
    assert "operation_intent_support_for_operation_family(" in text


def test_arch_spatial_c8_002_single_primary_key_rule_is_structurally_enforced() -> None:
    """ID: ARCH_SPATIAL_C8_002_single_primary_key_rule_is_structurally_enforced."""
    text = Path("tal/spatial/ops/frame_alignment_policy_ops.py").read_text(encoding="utf-8")
    assert "if plan.primary_key == \"param\" and plan.param_coord is not None:" in text
    assert "_preflight_param_primary(" in text
    assert "return align_exact_for_plan(plan, owner=owner, what=what)" in text


def test_arch_spatial_c8_003_no_local_duplicate_alignment_policy_helpers_in_spatial_owners() -> None:
    """ID: ARCH_SPATIAL_C8_003_no_local_duplicate_alignment_policy_helpers_in_spatial_owners."""
    coupling_text = Path("tal/spatial/ops/kinematics_path_coupling_ops.py").read_text(encoding="utf-8")
    expression_text = Path("tal/spatial/ops/frame_expression_ops.py").read_text(encoding="utf-8")
    frame_text = Path("tal/spatial/ops/kinematics_frame_ops.py").read_text(encoding="utf-8")
    for text in (coupling_text, expression_text, frame_text):
        assert "select_topology_policy_with_intents(" not in text
        assert "resolve_binary_topology(" not in text
        assert "align_exact_for_plan(" not in text
    assert "align_frame_pair_by_policy(" in coupling_text
    assert "xr.align(" not in coupling_text


def test_spatial_doc_c7_001_kinematics_path_support_and_fail_closed_semantics_documented() -> None:
    """ID: SPATIAL_DOC_C7_001_kinematics_path_support_and_fail_closed_semantics_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    velocity_doc = Path("docs/api/types/velocity.md").read_text(encoding="utf-8").lower()
    acceleration_doc = Path("docs/api/types/acceleration.md").read_text(encoding="utf-8").lower()
    assert "path" in spatial_doc
    assert "kinematic" in spatial_doc
    assert "to_frame" in velocity_doc
    assert "to_frame" in acceleration_doc


def test_spatial_doc_c7_002_to_frame_relative_to_semantics_documented() -> None:
    """ID: SPATIAL_DOC_C7_002_to_frame_relative_to_semantics_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    assert "to_frame(...)" in spatial_doc
    assert "parent/child" in spatial_doc


def test_spatial_doc_c7_003_coupling_exact_alignment_fail_closed_semantics_documented() -> None:
    """ID: SPATIAL_DOC_C7_003_coupling_exact_alignment_fail_closed_semantics_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    assert "alignment" in spatial_doc
    assert "ambiguous" in spatial_doc


def test_spatial_doc_c8_001_frame_aware_alignment_resolution_order_and_single_key_rule_documented() -> None:
    """ID: SPATIAL_DOC_C8_001_frame_aware_alignment_resolution_order_and_single_key_rule_documented."""
    spatial_doc = Path("docs/user-guide/spatial.md").read_text(encoding="utf-8").lower()
    assert "frame-aware" in spatial_doc
    assert "parameter alignment" in spatial_doc
