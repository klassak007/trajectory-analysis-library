from __future__ import annotations

import ast
from pathlib import Path

from ._budget import file_loc, function_lengths


def test_arch_paramops_001_sync_file_budget() -> None:
    """ID: ARCH_PARAMOPS_001_sync_file_budget."""
    path = Path("tal/core/param_ops/sync.py")
    assert file_loc(path=path) <= 600, f"{path} exceeds budget."


def test_arch_paramops_002_sync_function_budget() -> None:
    """ID: ARCH_PARAMOPS_002_sync_function_budget."""
    path = Path("tal/core/param_ops/sync.py")
    lengths = function_lengths(path)
    assert lengths.get("synchronize_param", 0) <= 50, lengths


def test_arch_paramops_003_single_batch_coord_helper_owner() -> None:
    """ID: ARCH_PARAMOPS_003_single_batch_coord_helper_owner."""
    root = Path("tal/core/param_ops")
    defs = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "def batch_coord(" in text:
            defs.append(path.as_posix())
        assert "def _batch_coord(" not in text, f"legacy _batch_coord helper remains in {path}"
    assert defs == ["tal/core/param_ops/axis_coords.py"], defs


def test_arch_paramops_004_no_interp_like_local_flatten_plan() -> None:
    """ID: ARCH_PARAMOPS_004_no_interp_like_local_flatten_plan."""
    text = Path("tal/core/param_ops/interp_like.py").read_text(encoding="utf-8")
    assert "_batch_flatten_plan(" not in text
    assert "flatten_param_contexts(" in text
    assert "from ..orchestration.topology import" in text


def test_arch_ao_001_private_finalize_calls_centralized() -> None:
    """ID: ARCH_AO_001_private_finalize_calls_centralized."""
    root = Path("tal/core")
    for path in sorted(root.rglob("*.py")):
        rel = path.as_posix()
        if rel in {"tal/core/analysis_object.py", "tal/core/ao_internal.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "._finalize_structural(" not in text, f"direct _finalize_structural call remains in {rel}"
        assert ".__class__._from_unvalidated(" not in text, f"direct _from_unvalidated call remains in {rel}"


def test_arch_combine_001_concat_sequence_file_and_function_budget() -> None:
    """ID: ARCH_COMBINE_001_concat_sequence_file_and_function_budget."""
    path = Path("tal/core/combine_ops/concat_sequence.py")
    assert file_loc(path=path) <= 600, f"{path} exceeds budget."
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{name} exceeds budget: {length} > 50"


def test_arch_combine_002_assemble_core_owner_split_and_budget() -> None:
    """ID: ARCH_COMBINE_002_assemble_core_owner_split_and_budget."""
    path = Path("tal/core/combine_ops/assemble_core.py")
    assert file_loc(path=path) <= 600, f"{path} exceeds budget."
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{name} exceeds budget: {length} > 50"


def test_arch_combine_003_assemble_core_finalize_owner_reuse() -> None:
    """ID: ARCH_COMBINE_003_assemble_core_finalize_owner_reuse."""
    text = Path("tal/core/combine_ops/assemble_core.py").read_text(encoding="utf-8")
    assert "finalize_combine_output(" in text
    assert "set_roles(" not in text
    assert "set_param_coord(" not in text
    assert "set_validity(" not in text


def test_arch_combine_004_assemble_core_uses_finalize_source_override_not_leaf_class() -> None:
    """ID: ARCH_COMBINE_004_assemble_core_uses_finalize_source_override_not_leaf_class."""
    assemble = Path("tal/core/combine_ops/assemble_core.py").read_text(encoding="utf-8")
    finalize = Path("tal/core/combine_ops/finalize.py").read_text(encoding="utf-8")
    assert "source_ao=source_ao" in assemble
    assert "source_ao:" in finalize
    assert "_resolve_finalize_source(" in finalize


def test_arch_combine_005_core_accessors_use_shared_self_inclusion_layout_helper() -> None:
    """ID: ARCH_COMBINE_005_core_accessors_use_shared_self_inclusion_layout_helper."""
    accessor = Path("tal/core/combine_ops/accessor.py").read_text(encoding="utf-8")
    assert "def _singleton_nested_layout(" in accessor
    assert "def _with_self_core_layout(" in accessor
    assert "merged = _with_self_core_layout(self._ao, values, depth=len(core_dims))" in accessor
    assert "merged = _with_self_core_layout(self._ao, values, depth=1)" in accessor
    assert "merged = _with_self_core_layout(self._ao, values, depth=2)" in accessor


def test_arch_combine_006_concat_core_owner_split_and_budget() -> None:
    """ID: ARCH_COMBINE_006_concat_core_owner_split_and_budget."""
    path = Path("tal/core/combine_ops/concat_core.py")
    assert file_loc(path=path) <= 600, f"{path} exceeds budget."
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{name} exceeds budget: {length} > 50"


def test_arch_combine_007_concat_core_reuses_shared_align_finalize_owners() -> None:
    """ID: ARCH_COMBINE_007_concat_core_reuses_shared_align_finalize_owners."""
    text = Path("tal/core/combine_ops/concat_core.py").read_text(encoding="utf-8")
    assert "align_contexts(" in text
    assert "finalize_combine_output(" in text
    assert "set_roles(" not in text
    assert "set_param_coord(" not in text
    assert "set_validity(" not in text


def test_arch_combine_008_decompose_core_owner_split_and_budget() -> None:
    """ID: ARCH_COMBINE_008_decompose_core_owner_split_and_budget."""
    path = Path("tal/core/combine_ops/decompose_core.py")
    assert file_loc(path=path) <= 600, f"{path} exceeds budget."
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{name} exceeds budget: {length} > 50"


def test_arch_combine_009_decompose_core_reuses_finalize_owner_no_local_schema_mutation() -> None:
    """ID: ARCH_COMBINE_009_decompose_core_reuses_finalize_owner_no_local_schema_mutation."""
    text = Path("tal/core/combine_ops/decompose_core.py").read_text(encoding="utf-8")
    assert "finalize_combine_output(" in text
    assert "set_roles(" not in text
    assert "set_param_coord(" not in text
    assert "set_validity(" not in text
    assert 'attrs["tal"]' not in text
    assert "attrs['tal']" not in text
    assert ".compute(" not in text
    assert ".item(" not in text
    assert ".values" not in text
    assert "np.asarray(" not in text


def test_arch_combine_010_decompose_core_label_uniqueness_nan_safe() -> None:
    """ID: ARCH_COMBINE_010_decompose_core_label_uniqueness_nan_safe."""
    text = Path("tal/core/combine_ops/decompose_core.py").read_text(encoding="utf-8")
    assert "index.has_duplicates" in text
    assert "len(set(labels))" not in text


def test_arch_combine_011_overlay_core_owner_split_and_budget() -> None:
    """ID: ARCH_COMBINE_011_overlay_core_owner_split_and_budget."""
    path = Path("tal/core/combine_ops/overlay_core.py")
    assert file_loc(path=path) <= 600, f"{path} exceeds budget."
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{name} exceeds budget: {length} > 50"


def test_arch_combine_012_overlay_core_reuses_align_finalize_owner_no_local_schema_mutation() -> None:
    """ID: ARCH_COMBINE_012_overlay_core_reuses_align_finalize_owner_no_local_schema_mutation."""
    text = Path("tal/core/combine_ops/overlay_core.py").read_text(encoding="utf-8")
    assert "align_contexts(" in text
    assert "finalize_combine_output(" in text
    assert "set_roles(" not in text
    assert "set_param_coord(" not in text
    assert "set_validity(" not in text
    assert 'attrs["tal"]' not in text
    assert "attrs['tal']" not in text
    assert ".compute(" not in text
    assert ".item(" not in text
    assert ".values" not in text
    assert "np.asarray(" not in text


def test_arch_combine_013_overlay_core_conflict_policy_single_owner() -> None:
    """ID: ARCH_COMBINE_013_overlay_core_conflict_policy_single_owner."""
    core_text = Path("tal/core/combine_ops/overlay_core.py").read_text(encoding="utf-8")
    linalg_text = Path("tal/linalg/ops/overlay_core.py").read_text(encoding="utf-8")
    assert "def _require_no_patch_overlap(" in core_text
    assert "opts.on_overlap" in core_text or "options.on_overlap" in core_text
    assert "_require_no_patch_overlap(" not in linalg_text
    assert "on_overlap" not in linalg_text


def test_arch_combine_014_overlay_core_nan_safe_label_key_canonicalization_single_owner() -> None:
    """ID: ARCH_COMBINE_014_overlay_core_nan_safe_label_key_canonicalization_single_owner."""
    text = Path("tal/core/combine_ops/overlay_core.py").read_text(encoding="utf-8")
    assert "def _canonical_label_key(" in text
    assert "def _is_missing_scalar(" in text
    assert "def _index_key_map(" in text
    assert "tuple(_canonical_label_key(item) for item in label)" in text
    assert "if _is_missing_scalar(label):" in text


def test_arch_combine_015_overlay_core_kernel_avoids_isin_reindex_where_nan_upcast_path() -> None:
    """ID: ARCH_COMBINE_015_overlay_core_kernel_avoids_isin_reindex_where_nan_upcast_path."""
    text = Path("tal/core/combine_ops/overlay_core.py").read_text(encoding="utf-8")
    assert "coords[core_dim].isin(patch_labels)" not in text
    assert "patch_data.reindex({core_dim: base_data.get_index(core_dim)})" not in text
    assert "xr.where(mask, aligned_patch, base_data)" not in text


def test_arch_combine_016_overlay_core_missing_scalar_canonicalization_type_agnostic() -> None:
    """ID: ARCH_COMBINE_016_overlay_core_missing_scalar_canonicalization_type_agnostic."""
    text = Path("tal/core/combine_ops/overlay_core.py").read_text(encoding="utf-8")
    assert "_MISSING_SCALAR_SENTINEL" in text
    assert "if _is_missing_scalar(label):" in text
    assert "return _MISSING_SCALAR_SENTINEL" in text
    assert 'type(label).__name__' not in text


def test_arch_doc_001_hotspot_modules_have_required_docstrings() -> None:
    """ID: ARCH_DOC_001_hotspot_modules_have_required_docstrings."""
    hotspots = [
        Path("tal/core/combine_ops/concat_sequence.py"),
        Path("tal/core/param_engine/schema_resolve.py"),
        Path("tal/core/param_ops/sync_runtime.py"),
    ]
    for path in hotspots:
        module = ast.parse(path.read_text(encoding="utf-8"))
        docs = sum(
            1
            for node in module.body
            if isinstance(node, ast.FunctionDef) and ast.get_docstring(node)
        )
        assert docs >= 1, f"{path.as_posix()} has no function-level docstrings"


def test_arch_param_engine_001_map_orchestrator_has_no_row_kernels() -> None:
    """ID: ARCH_PARAM_ENGINE_001_map_row_kernels_have_no_python_query_loops."""
    module = ast.parse(Path("tal/core/param_engine/map_build.py").read_text(encoding="utf-8"))
    definitions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert definitions.isdisjoint({"_nearest_row", "_linear_row", "_map_row", "_bounds_row"})
    exact_module = ast.parse(Path("tal/core/param_engine/numeric_rows.py").read_text(encoding="utf-8"))
    float_kernels = {
        node.name: node
        for node in exact_module.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_float_nearest_row", "_float_linear_row"}
    }
    assert set(float_kernels) == {"_float_nearest_row", "_float_linear_row"}
    for name, node in float_kernels.items():
        loops = [sub for sub in ast.walk(node) if isinstance(sub, (ast.For, ast.While, ast.AsyncFor))]
        assert not loops, f"{name} must retain vectorized float fallback execution"


def test_param_arch_048_ordered_numeric_dtype_has_shared_core_owner() -> None:
    """ID: PARAM_ARCH_048_ordered_numeric_dtype_has_shared_core_owner."""
    owner_name = "is_ordered_real_numeric_dtype"
    definitions = []
    for path in sorted(Path("tal/core").rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.FunctionDef) and node.name == owner_name for node in module.body):
            definitions.append(path.as_posix())
    assert definitions == ["tal/core/ordered_dtypes.py"]

    consumers = (
        Path("tal/core/schema_validate/phase_param.py"),
        Path("tal/core/orchestration/resolve.py"),
        Path("tal/core/param_engine/query_grid.py"),
        Path("tal/core/param_engine/map_build.py"),
    )
    for path in consumers:
        module = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom) and node.module == "ordered_dtypes"
            for alias in node.names
        }
        assert owner_name in imported, f"{path} must consume the neutral ordered dtype owner"


def test_param_arch_049_integral_param_maps_use_exact_block_owner() -> None:
    """ID: PARAM_ARCH_049_integral_param_maps_use_exact_block_owner."""
    exact_path = Path("tal/core/param_engine/numeric_rows.py")
    exact_module = ast.parse(exact_path.read_text(encoding="utf-8"))
    exact_defs = {node.name for node in exact_module.body if isinstance(node, ast.FunctionDef)}
    assert {"numeric_map_row", "numeric_bounds_row"} <= exact_defs

    map_path = Path("tal/core/param_engine/map_build.py")
    map_text = map_path.read_text(encoding="utf-8")
    map_module = ast.parse(map_text)
    map_defs = {node.name for node in map_module.body if isinstance(node, ast.FunctionDef)}
    assert {"numeric_map_row", "numeric_bounds_row", "_map_row", "_bounds_row"}.isdisjoint(map_defs)
    assert "def _float_numba_compatible(" in map_text
    assert 'np.dtype(value.dtype).kind == "f"' in map_text
    assert "backend = _select_map_normal_backend(param=param_da, query=query_da)" in map_text

    numpy_module = ast.parse(Path("tal/core/param_engine/numpy_backends.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(numpy_module)
        if isinstance(node, ast.ImportFrom) and node.module == "numeric_rows"
        for alias in node.names
    }
    assert imports == {"numeric_map_row", "numeric_bounds_row"}


def test_arch_paramops_005_no_ad_hoc_compute_in_select_or_sync_runtime() -> None:
    """ID: ARCH_PARAMOPS_005_no_ad_hoc_compute_in_select_or_sync_runtime."""
    select_text = Path("tal/core/param_ops/select.py").read_text(encoding="utf-8")
    sync_runtime_text = Path("tal/core/param_ops/sync_runtime.py").read_text(encoding="utf-8")
    assert ".compute(" not in select_text
    assert ".compute(" not in sync_runtime_text


def test_arch_paramops_006_sync_autogrid_owner_split_runtime_vs_backend_is_enforced() -> None:
    """ID: ARCH_PARAMOPS_006_sync_autogrid_owner_split_runtime_vs_backend_is_enforced."""
    runtime_text = Path("tal/core/param_ops/sync_runtime.py").read_text(encoding="utf-8")
    orchestrator_text = Path("tal/core/param_ops/sync_autogrid.py").read_text(encoding="utf-8")
    backend_text = Path("tal/core/param_ops/sync_autogrid_backend.py").read_text(encoding="utf-8")
    assert "from .sync_autogrid import build_auto_grid_from_join" in runtime_text
    assert "param_kind=contexts[0].param_kind" in runtime_text
    for needle in [
        "def _row_data(",
        "def _join_grid_unbatched(",
        "def _join_grid_batched(",
        "def _rows_join(",
        "def _merge_many(",
    ]:
        assert needle not in runtime_text
    assert "def _materialize_join_inputs(" in orchestrator_text
    assert "require_unchunked_dataarray(" not in orchestrator_text
    assert "def join_rows_batched(" in backend_text


def test_arch_paramops_007_sync_autogrid_row_loop_step_is_backend_owner_scoped_only() -> None:
    """ID: ARCH_PARAMOPS_007_sync_autogrid_row_loop_step_is_backend_owner_scoped_only."""
    runtime_text = Path("tal/core/param_ops/sync_runtime.py").read_text(encoding="utf-8")
    orchestrator_text = Path("tal/core/param_ops/sync_autogrid.py").read_text(encoding="utf-8")
    backend_text = Path("tal/core/param_ops/sync_autogrid_backend.py").read_text(encoding="utf-8")
    assert "for row_index in range(" not in runtime_text
    assert "for row_index in range(" not in orchestrator_text
    assert "for row_index in range(" in backend_text
    assert "for row in rows[1:]" in backend_text
