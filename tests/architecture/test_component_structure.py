from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture._budget import executable_source, file_loc, function_loc


def _module(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def test_arch_component_001_component_ops_owner_split_and_budget() -> None:
    """ID: ARCH_COMPONENT_001_component_ops_owner_split_and_budget."""
    files = [
        Path("tal/core/component_ops/types.py"),
        Path("tal/core/component_ops/options.py"),
        Path("tal/core/component_ops/registry.py"),
        Path("tal/core/component_ops/runtime_checks.py"),
        Path("tal/core/component_ops/rewrite.py"),
        Path("tal/core/component_ops/compose.py"),
        Path("tal/core/component_ops/extract.py"),
        Path("tal/core/component_ops/patch.py"),
        Path("tal/core/component_ops/accessor.py"),
    ]
    for path in files:
        assert path.exists(), f"missing component owner file: {path}"
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        module = _module(path.as_posix())
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                length = function_loc(node, path=path)
                assert length <= 50, f"{path}:{node.name} exceeds function budget: {length} > 50"


def test_arch_component_003_component_registry_rewrite_single_owner() -> None:
    """ID: ARCH_COMPONENT_003_component_registry_rewrite_single_owner."""
    rewrite = Path("tal/core/component_ops/rewrite.py").read_text(encoding="utf-8")
    registry = Path("tal/core/component_ops/registry.py").read_text(encoding="utf-8")
    analysis_object = Path("tal/core/analysis_object.py").read_text(encoding="utf-8")
    assert "def rewrite_component_registry_after_structure(" in rewrite
    assert "def rewrite_component_registry_after_structure(" not in registry
    assert "def rewrite_component_registry_after_structure(" not in analysis_object


def test_arch_component_004_no_core_to_linalg_or_spatial_imports() -> None:
    """ID: ARCH_COMPONENT_004_no_core_to_linalg_or_spatial_imports."""
    for path in sorted(Path("tal/core/component_ops").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.linalg" not in text
        assert "tal.spatial" not in text
        assert "from ...linalg" not in text
        assert "from ...spatial" not in text


def test_arch_component_006_structural_finalize_invokes_component_registry_rewriter() -> None:
    """ID: ARCH_COMPONENT_006_structural_finalize_invokes_component_registry_rewriter."""
    analysis_object = Path("tal/core/analysis_object.py").read_text(encoding="utf-8")
    assert "from .component_ops.rewrite import rewrite_component_registry_after_structure" in analysis_object
    assert "rewritten = rewrite_component_registry_after_structure(" in analysis_object
    assert "_finalize_structural" in analysis_object


def test_arch_component_007_components_version_gate_single_owner_strict_type() -> None:
    """ID: ARCH_COMPONENT_007_components_version_gate_single_owner_strict_type."""
    options = Path("tal/core/component_ops/options.py").read_text(encoding="utf-8")
    registry = Path("tal/core/component_ops/registry.py").read_text(encoding="utf-8")
    rewrite = Path("tal/core/component_ops/rewrite.py").read_text(encoding="utf-8")
    assert "def require_supported_components_version(" in options
    assert "isinstance(version, bool)" in options
    assert "not isinstance(version, int)" in options
    assert "version != COMPONENTS_SCHEMA_VERSION" in options
    assert "require_supported_components_version(" in registry
    assert "require_supported_components_version(" in rewrite
    assert "def require_supported_components_version(" not in registry
    assert "def require_supported_components_version(" not in rewrite


def test_arch_component_008_rewrite_remaps_component_var_with_rename_map() -> None:
    """ID: ARCH_COMPONENT_008_rewrite_remaps_component_var_with_rename_map."""
    rewrite = Path("tal/core/component_ops/rewrite.py").read_text(encoding="utf-8")
    assert "var_name = spec.var" in rewrite
    assert "var_name = rename_map.get(var_name, var_name)" in rewrite
    assert "ComponentSpec(core_dim=dim, labels=spec.labels, var=var_name)" in rewrite


def test_arch_component_009_rewrite_rejects_non_mapping_components_payload() -> None:
    """ID: ARCH_COMPONENT_009_rewrite_rejects_non_mapping_components_payload."""
    rewrite = Path("tal/core/component_ops/rewrite.py").read_text(encoding="utf-8")
    assert "def _read_components_block(ds: xr.Dataset, *, owner: str)" in rewrite
    assert 'raise ValueError(f"{owner}: tal.ext.components must be a mapping.")' in rewrite
    assert "if _COMPONENTS_NAMESPACE not in ext:" in rewrite


def test_arch_component_010_extract_patch_owner_split_and_budget() -> None:
    """ID: ARCH_COMPONENT_010_extract_patch_owner_split_and_budget."""
    extract = Path("tal/core/component_ops/extract.py")
    patch = Path("tal/core/component_ops/patch.py")
    assert extract.exists()
    assert patch.exists()
    for path in [extract, patch]:
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        module = _module(path.as_posix())
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                length = function_loc(node, path=path)
                assert length <= 50, f"{path}:{node.name} exceeds function budget: {length} > 50"


def test_arch_component_011_extract_reuses_registry_read_owner_single_source() -> None:
    """ID: ARCH_COMPONENT_011_extract_reuses_registry_read_owner_single_source."""
    extract = Path("tal/core/component_ops/extract.py").read_text(encoding="utf-8")
    assert "from .registry import read_components" in extract
    assert "read_components(source)" in extract
    assert "_decode_registry_payload(" not in extract
    assert "_read_components_block(" not in extract


def test_arch_component_012_patch_reuses_overlay_core_owner_no_local_schema_mutation() -> None:
    """ID: ARCH_COMPONENT_012_patch_reuses_overlay_core_owner_no_local_schema_mutation."""
    patch = Path("tal/core/component_ops/patch.py").read_text(encoding="utf-8")
    assert "from ..combine_ops.overlay_core import overlay_core" in patch
    assert "overlay_core(" in patch
    assert "attrs[\"tal\"]" not in patch
    assert "set_roles(" not in patch
    assert "set_param_coord(" not in patch
    assert "set_validity(" not in patch


def test_arch_component_013_no_core_to_linalg_or_spatial_imports_component_runtime() -> None:
    """ID: ARCH_COMPONENT_013_no_core_to_linalg_or_spatial_imports_component_runtime."""
    for path in [Path("tal/core/component_ops/extract.py"), Path("tal/core/component_ops/patch.py")]:
        text = path.read_text(encoding="utf-8")
        assert "tal.linalg" not in text
        assert "tal.spatial" not in text
        assert "from ...linalg" not in text
        assert "from ...spatial" not in text


def test_arch_component_014_no_eager_materialization_in_extract_patch_paths() -> None:
    """ID: ARCH_COMPONENT_014_no_eager_materialization_in_extract_patch_paths."""
    for path in [Path("tal/core/component_ops/extract.py"), Path("tal/core/component_ops/patch.py")]:
        text = executable_source(path=path)
        assert ".values" not in text
        assert ".item(" not in text
        assert ".compute(" not in text
        assert "np.asarray(" not in text


def test_arch_component_015_extract_output_var_rename_after_finalize_boundary() -> None:
    """ID: ARCH_COMPONENT_015_extract_output_var_rename_after_finalize_boundary."""
    extract = Path("tal/core/component_ops/extract.py").read_text(encoding="utf-8")
    module = _module("tal/core/component_ops/extract.py")
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_extract_component_dataset"
    )
    kernel_args = [arg.arg for arg in kernel.args.args]
    assert "def _rename_output_var_after_finalize(" in extract
    assert "result.rename({source_var: output_var}, validate=validate)" in extract
    assert "finalize_like(source, ds_out, validate=validate, owner=owner)" in extract
    assert "output_var" not in kernel_args


def test_arch_component_016_compose_owner_split_and_budget() -> None:
    """ID: ARCH_COMPONENT_016_compose_owner_split_and_budget."""
    compose = Path("tal/core/component_ops/compose.py")
    assert compose.exists()
    assert file_loc(path=compose) <= 600, f"{compose} exceeds file budget."
    module = _module(compose.as_posix())
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            length = function_loc(node, path=compose)
            assert length <= 50, f"{compose}:{node.name} exceeds function budget: {length} > 50"


def test_arch_component_017_compose_reuses_concat_core_and_merge_owners() -> None:
    """ID: ARCH_COMPONENT_017_compose_reuses_concat_core_and_merge_owners."""
    compose = Path("tal/core/component_ops/compose.py").read_text(encoding="utf-8")
    assert "from ..combine_ops import CoreConcatOptions, MergeOptions, concat_core, merge" in compose
    assert "concat_core(" in compose
    assert "merge(" in compose


def test_arch_component_018_compose_reuses_define_components_owner_for_registry_finalize() -> None:
    """ID: ARCH_COMPONENT_018_compose_reuses_define_components_owner_for_registry_finalize."""
    compose = Path("tal/core/component_ops/compose.py").read_text(encoding="utf-8")
    assert "from .registry import define_components" in compose
    assert "define_components(" in compose
    assert "attrs[\"tal\"]" not in compose
    assert "set_roles(" not in compose
    assert "set_param_coord(" not in compose
    assert "set_validity(" not in compose


def test_arch_component_019_no_core_to_linalg_or_spatial_imports_component_compose_runtime() -> None:
    """ID: ARCH_COMPONENT_019_no_core_to_linalg_or_spatial_imports_component_compose_runtime."""
    compose = Path("tal/core/component_ops/compose.py").read_text(encoding="utf-8")
    assert "tal.linalg" not in compose
    assert "tal.spatial" not in compose
    assert "from ...linalg" not in compose
    assert "from ...spatial" not in compose


def test_arch_component_020_no_eager_materialization_in_compose_path() -> None:
    """ID: ARCH_COMPONENT_020_no_eager_materialization_in_compose_path."""
    compose = executable_source(path="tal/core/component_ops/compose.py")
    assert ".values" not in compose
    assert ".item(" not in compose
    assert ".compute(" not in compose
    assert "np.asarray(" not in compose


def test_arch_component_021_component_runtime_checks_single_owner_reused_across_extract_patch_compose() -> None:
    """ID: ARCH_COMPONENT_021_component_runtime_checks_single_owner_reused_across_extract_patch_compose."""
    runtime_checks = Path("tal/core/component_ops/runtime_checks.py").read_text(encoding="utf-8")
    compose = Path("tal/core/component_ops/compose.py").read_text(encoding="utf-8")
    extract = Path("tal/core/component_ops/extract.py").read_text(encoding="utf-8")
    patch = Path("tal/core/component_ops/patch.py").read_text(encoding="utf-8")
    assert "def require_declared_roles_with_sequence(" in runtime_checks
    assert "def select_component_var(" in runtime_checks
    assert "def require_explicit_unique_labels(" in runtime_checks
    assert "from .runtime_checks import" in compose
    assert "from .runtime_checks import" in extract
    assert "from .runtime_checks import" in patch
    assert "def _require_declared_roles_with_sequence(" not in compose
    assert "def _require_declared_roles_with_sequence(" not in patch
    assert "def _select_component_var(" not in compose
    assert "def _select_component_var(" not in extract
    assert "def _select_component_var(" not in patch
    assert "def _require_explicit_unique_labels(" not in compose
    assert "def _require_explicit_unique_labels(" not in patch


def test_arch_component_022_component_runtime_dim_guard_before_index_lookup() -> None:
    """ID: ARCH_COMPONENT_022_component_runtime_dim_guard_before_index_lookup."""
    runtime_checks = Path("tal/core/component_ops/runtime_checks.py").read_text(encoding="utf-8")
    compose = Path("tal/core/component_ops/compose.py").read_text(encoding="utf-8")
    patch = Path("tal/core/component_ops/patch.py").read_text(encoding="utf-8")
    assert "def require_explicit_unique_labels(" in runtime_checks
    assert "if dim not in data.dims:" in runtime_checks
    assert "data.get_index(dim)" in runtime_checks
    assert "except KeyError as exc:" in runtime_checks
    assert "get_index(" not in compose
    assert "get_index(" not in patch


def test_comp_backbone_doc_001_component_backbone_user_guide_and_api_entries_present() -> None:
    """ID: COMP_BACKBONE_DOC_001_component_backbone_user_guide_and_api_entries_present."""
    core_concepts = Path("docs/user-guide/core_concepts.md").read_text(encoding="utf-8")
    api_analysis = Path("docs/api/analysis-object.md").read_text(encoding="utf-8")
    api_components = Path("docs/api/components.md").read_text(encoding="utf-8")
    api_index = Path("docs/api/index.md").read_text(encoding="utf-8")
    assert "component registry" in core_concepts.lower()
    assert "ao.components" in api_analysis
    assert "tal.core.define_components" in api_components
    assert "tal.core.read_components" in api_components
    assert "components" in api_index


def test_comp_backbone_doc_002_component_extract_patch_user_guide_and_api_entries_present() -> None:
    """ID: COMP_BACKBONE_DOC_002_component_extract_patch_user_guide_and_api_entries_present."""
    api_analysis = Path("docs/api/analysis-object.md").read_text(encoding="utf-8")
    api_components = Path("docs/api/components.md").read_text(encoding="utf-8")
    assert "tal.core.extract_components" in api_components
    assert "tal.core.patch_components" in api_components
    assert "ao.components" in api_analysis


def test_comp_backbone_doc_003_component_compose_user_guide_and_api_entries_present() -> None:
    """ID: COMP_BACKBONE_DOC_003_component_compose_user_guide_and_api_entries_present."""
    api_analysis = Path("docs/api/analysis-object.md").read_text(encoding="utf-8")
    api_components = Path("docs/api/components.md").read_text(encoding="utf-8")
    assert "tal.core.compose_components" in api_components
    assert "ao.components" in api_analysis
