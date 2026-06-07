from __future__ import annotations

import ast
from pathlib import Path

from ._budget import function_loc


def _module(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _class_method_names(module: ast.Module, class_name: str) -> set[str]:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {child.name for child in node.body if isinstance(child, ast.FunctionDef)}
    raise AssertionError(f"missing class: {class_name}")


def test_linalg_arch_001_orchestrate_kernel_finalize_owner_split() -> None:
    """ID: LINALG_ARCH_001_orchestrate_kernel_finalize_owner_split."""
    plan = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    finalize = Path("tal/linalg/finalize.py").read_text(encoding="utf-8")
    linalg_init = Path("tal/linalg/__init__.py").read_text(encoding="utf-8")
    assert "def build_array_plan(" in plan
    assert "def compute_matmul(" in matmul_ops
    assert "def matmul(" in matmul_ops
    assert "def finalize_array_result(" in finalize
    assert "from .ops import" in linalg_init
    assert "MatmulOptions" in linalg_init
    assert "PInvOptions" in linalg_init
    assert "SolveOptions" in linalg_init
    assert "add" in linalg_init
    assert "assemble_core" in linalg_init
    assert "block_core" in linalg_init
    assert "dot" in linalg_init
    assert "inv" in linalg_init
    assert "matmul" in linalg_init
    assert "norm" in linalg_init
    assert "pinv" in linalg_init
    assert "solve" in linalg_init
    assert "stack_core" in linalg_init
    assert "sub" in linalg_init


def test_linalg_arch_002_no_spatial_hard_dependency() -> None:
    """ID: LINALG_ARCH_002_no_spatial_hard_dependency."""
    for path in Path("tal/linalg").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "tal.spatial" not in text
        assert "from ..spatial" not in text


def test_linalg_arch_003_no_local_schema_attr_mutation_in_kernels() -> None:
    """ID: LINALG_ARCH_003_no_local_schema_attr_mutation_in_kernels."""
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    assert 'attrs["tal"]' not in matmul_ops
    assert "attrs['tal']" not in matmul_ops


def test_linalg_arch_004_orchestrate_finalize_modules_thin_core_delegators() -> None:
    """ID: LINALG_ARCH_004_orchestrate_finalize_modules_thin_core_delegators."""
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    binops = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    assert "from ..plan import" in matmul_ops
    assert "from .binops import" in matmul_ops
    assert "build_strict_binary_plan(" in matmul_ops
    assert "finalize_binary_output(" in matmul_ops
    assert "from ..finalize import" in binops
    assert "finalize_array_result(" in binops
    assert not Path("tal/linalg/api.py").exists()
    assert not Path("tal/linalg/ops_kernel.py").exists()
    assert not Path("tal/linalg/ops_orchestrate.py").exists()
    assert not Path("tal/linalg/ops_finalize.py").exists()


def test_linalg_arch_005_public_linalg_ops_follow_validate_orchestrate_kernel_finalize() -> None:
    """ID: LINALG_ARCH_005_public_linalg_ops_follow_validate_orchestrate_kernel_finalize."""
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    binops = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    assert "coerce_matmul_options(" in matmul_ops
    assert "build_strict_binary_plan(" in matmul_ops
    assert "compute_matmul(" in matmul_ops
    assert "finalize_binary_output(" in matmul_ops
    assert "build_array_binary_plan(" in binops
    assert "finalize_array_result(" in binops
    assert "build_matmul_plan(" not in matmul_ops
    assert "finalize_matmul(" not in matmul_ops


def test_linalg_arch_006_optional_metadata_helpers_single_owner() -> None:
    """ID: LINALG_ARCH_006_optional_metadata_helpers_single_owner."""
    shared = Path("tal/core/metadata_optional.py").read_text(encoding="utf-8")
    combine = Path("tal/core/combine_ops/metadata.py").read_text(encoding="utf-8")
    assert "def shared_optional_name(" in shared
    assert "def canonicalize_optional_names(" in shared
    assert "def shared_optional_name(" not in combine
    assert "def canonicalize_optional_names(" not in combine


def test_linalg_arch_007_linalg_finalize_no_local_optional_name_merge_logic() -> None:
    """ID: LINALG_ARCH_007_linalg_finalize_no_local_optional_name_merge_logic."""
    finalize = Path("tal/linalg/finalize.py").read_text(encoding="utf-8")
    schema_finalize = Path("tal/core/orchestration/schema_finalize.py").read_text(encoding="utf-8")
    assert "shared_optional_name(" not in finalize
    assert "canonicalize_optional_names(" not in finalize
    assert "canonicalize_optional_names(" in schema_finalize
    assert "finalize_array_result(" in finalize


def test_linalg_arch_008_array_plan_owner_in_core_orchestration() -> None:
    """ID: LINALG_ARCH_008_array_plan_owner_in_core_orchestration."""
    linalg_plan = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    assert "class ArrayOperandContext" in linalg_plan
    assert "class ArrayPlan" in linalg_plan
    assert "def build_array_plan(" in linalg_plan
    assert not Path("tal/core/orchestration/array_plan.py").exists()


def test_linalg_arch_009_binary_plan_wraps_general_array_plan() -> None:
    """ID: LINALG_ARCH_009_binary_plan_wraps_general_array_plan."""
    linalg_plan = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    assert "def build_array_binary_plan(" in linalg_plan
    assert "return build_array_plan(" in linalg_plan


def test_linalg_arch_010_array_finalize_owner_in_core_orchestration() -> None:
    """ID: LINALG_ARCH_010_array_finalize_owner_in_core_orchestration."""
    linalg_finalize = Path("tal/linalg/finalize.py").read_text(encoding="utf-8")
    schema_finalize = Path("tal/core/orchestration/schema_finalize.py").read_text(encoding="utf-8")
    assert "class ArrayFinalizeSpec" in linalg_finalize
    assert "def finalize_array_result(" in linalg_finalize
    assert "def restore_optional_coord_from_sources(" not in linalg_finalize
    assert "def restore_optional_coord_from_sources(" in schema_finalize
    assert not Path("tal/core/orchestration/array_finalize.py").exists()


def test_linalg_arch_011_matmul_orchestrate_finalize_thin_wrappers() -> None:
    """ID: LINALG_ARCH_011_matmul_orchestrate_finalize_thin_wrappers."""
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    binops = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    assert not Path("tal/linalg/api.py").exists()
    assert not Path("tal/linalg/ops_kernel.py").exists()
    assert not Path("tal/linalg/ops_orchestrate.py").exists()
    assert not Path("tal/linalg/ops_finalize.py").exists()
    assert "def _single_numeric_data_var(" not in matmul_ops
    assert "def _align_exact(" not in matmul_ops
    assert "build_strict_binary_plan(" in matmul_ops
    assert "finalize_binary_output(" in matmul_ops
    assert "def _coerce_operand(" not in binops
    assert "def _build_finalize_spec(" in binops


def test_linalg_arch_012_array_is_ao_subclass_no_wrapper_storage() -> None:
    """ID: LINALG_ARCH_012_array_is_ao_subclass_no_wrapper_storage."""
    array_text = Path("tal/linalg/array.py").read_text(encoding="utf-8")
    assert "class Array(TypedAnalysisObject):" in array_text
    assert "from ..core.typed_lifecycle import TypedAnalysisObject" in array_text
    assert "self._ao" not in array_text
    assert "def _from_ao(" not in array_text
    assert "def analysis_object(" not in array_text
    assert not Path("tal/linalg/types.py").exists()


def test_linalg_arch_013_matmul_no_wrapper_rewrap_paths() -> None:
    """ID: LINALG_ARCH_013_matmul_no_wrapper_rewrap_paths."""
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    assert "value.analysis_object" not in matmul_ops
    assert "_from_ao(" not in matmul_ops
    assert "def _resolve_output_array_type(" not in matmul_ops
    assert "def _rewrap_output_array(" not in matmul_ops


def test_linalg_arch_014_binary_result_type_policy_single_owner() -> None:
    """ID: LINALG_ARCH_014_binary_result_type_policy_single_owner."""
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    binops = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    result_type = Path("tal/linalg/result_type.py").read_text(encoding="utf-8")
    assert "def resolve_binary_output_array_type(" in result_type
    assert "def resolve_binary_output_array_type_by_core_arity(" in result_type
    assert "def rewrap_binary_output_array(" in result_type
    assert "def resolve_matmul_output_array_type(" not in result_type
    assert "def resolve_solve_output_array_type(" not in result_type
    assert "def resolve_inv_output_array_type(" not in result_type
    assert "def resolve_pinv_output_array_type(" not in result_type
    assert "def resolve_dot_output_array_type(" not in result_type
    assert "def resolve_norm_output_array_type(" not in result_type
    assert "from ..result_type import" in binops
    assert "resolve_binary_output_array_type(" in binops
    assert "rewrap_binary_output_array(" in binops
    assert "resolve_binary_output_array_type(" not in matmul_ops
    assert "rewrap_binary_output_array(" not in matmul_ops


def test_linalg_arch_015_binops_shared_owner_module_single_owner() -> None:
    """ID: LINALG_ARCH_015_binops_shared_owner_module_single_owner."""
    binops = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    assert "def coerce_binary_operands(" in binops
    assert "def build_strict_binary_plan(" in binops
    assert "def finalize_binary_output(" in binops


def test_linalg_arch_016_add_sub_matmul_reuse_shared_binops_owner() -> None:
    """ID: LINALG_ARCH_016_add_sub_matmul_reuse_shared_binops_owner."""
    add_ops = Path("tal/linalg/ops/add.py").read_text(encoding="utf-8")
    sub_ops = Path("tal/linalg/ops/sub.py").read_text(encoding="utf-8")
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    assert "from .binops import" in add_ops
    assert "from .binops import" in sub_ops
    assert "from .binops import" in matmul_ops
    assert "run_elementwise_binary(" in add_ops
    assert "run_elementwise_binary(" in sub_ops
    assert "build_strict_binary_plan(" in matmul_ops
    assert "finalize_binary_output(" in matmul_ops


def test_linalg_arch_017_no_local_binop_operand_finalize_duplication() -> None:
    """ID: LINALG_ARCH_017_no_local_binop_operand_finalize_duplication."""
    add_ops = Path("tal/linalg/ops/add.py").read_text(encoding="utf-8")
    sub_ops = Path("tal/linalg/ops/sub.py").read_text(encoding="utf-8")
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    for text in (add_ops, sub_ops, matmul_ops):
        assert "def _coerce_operand(" not in text
        assert "def _build_finalize_spec(" not in text
        assert "def _finalize_matmul_output(" not in text


def test_linalg_arch_018_binops_kernel_no_schema_mutation() -> None:
    """ID: LINALG_ARCH_018_binops_kernel_no_schema_mutation."""
    add_ops = Path("tal/linalg/ops/add.py").read_text(encoding="utf-8")
    sub_ops = Path("tal/linalg/ops/sub.py").read_text(encoding="utf-8")
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    for text in (add_ops, sub_ops, matmul_ops):
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text


def test_linalg_arch_019_elementwise_binop_flow_single_owner() -> None:
    """ID: LINALG_ARCH_019_elementwise_binop_flow_single_owner."""
    binops = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    assert "def compute_elementwise_binary(" in binops
    assert "def run_elementwise_binary(" in binops
    assert "coerce_binary_operands(" in binops
    assert "build_strict_binary_plan(" in binops
    assert "resolve_elementwise_output_core_dims(" in binops
    assert "finalize_binary_output(" in binops


def test_linalg_arch_020_add_sub_no_local_elementwise_orchestration_duplication() -> None:
    """ID: LINALG_ARCH_020_add_sub_no_local_elementwise_orchestration_duplication."""
    add_ops = Path("tal/linalg/ops/add.py").read_text(encoding="utf-8")
    sub_ops = Path("tal/linalg/ops/sub.py").read_text(encoding="utf-8")
    for text in (add_ops, sub_ops):
        assert "run_elementwise_binary(" in text
        assert "build_strict_binary_plan(" not in text
        assert "finalize_binary_output(" not in text
        assert "resolve_elementwise_output_core_dims(" not in text
        assert "coerce_binary_operands(" not in text


def test_linalg_arch_021_vector_matrix_wrapper_owners_single_owner_modules() -> None:
    """ID: LINALG_ARCH_021_vector_matrix_wrapper_owners_single_owner_modules."""
    vector_text = Path("tal/linalg/vector.py").read_text(encoding="utf-8")
    matrix_text = Path("tal/linalg/matrix.py").read_text(encoding="utf-8")
    assert "class Vector(Array):" in vector_text
    assert "class Matrix(Array):" in matrix_text
    assert "def T(" in matrix_text


def test_linalg_arch_022_matmul_type_routing_single_owner_in_result_type() -> None:
    """ID: LINALG_ARCH_022_matmul_type_routing_single_owner_in_result_type."""
    result_type = Path("tal/linalg/result_type.py").read_text(encoding="utf-8")
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    assert "def resolve_binary_output_array_type_by_core_arity(" in result_type
    assert "resolve_binary_output_array_type_by_core_arity(" in matmul_ops
    assert "from ..result_type import resolve_binary_output_array_type_by_core_arity" in matmul_ops


def test_linalg_arch_023_matmul_owner_no_local_type_routing_duplication() -> None:
    """ID: LINALG_ARCH_023_matmul_owner_no_local_type_routing_duplication."""
    matmul_ops = Path("tal/linalg/ops/matmul.py").read_text(encoding="utf-8")
    assert "def _resolve_matmul_output_type(" not in matmul_ops
    assert "def _route_matmul_output_type(" not in matmul_ops
    assert "resolve_binary_output_array_type_by_core_arity(" in matmul_ops


def test_linalg_arch_024_typed_wrapper_invariants_enforced_on_rewrap_boundaries() -> None:
    """ID: LINALG_ARCH_024_typed_wrapper_invariants_enforced_on_rewrap_boundaries."""
    array_text = Path("tal/linalg/array.py").read_text(encoding="utf-8")
    vector_text = Path("tal/linalg/vector.py").read_text(encoding="utf-8")
    matrix_text = Path("tal/linalg/matrix.py").read_text(encoding="utf-8")
    lifecycle_text = Path("tal/linalg/lifecycle.py").read_text(encoding="utf-8")
    assert "LIFECYCLE = ARRAY_LIFECYCLE" in array_text
    assert "LIFECYCLE = VECTOR_LIFECYCLE" in vector_text
    assert "LIFECYCLE = MATRIX_LIFECYCLE" in matrix_text
    assert "def enforce_array_invariants(" in lifecycle_text
    assert "def enforce_vector_invariants(" in lifecycle_text
    assert "def enforce_matrix_invariants(" in lifecycle_text
    assert "def _from_validated(" not in array_text
    assert "def _from_unvalidated(" not in array_text
    assert "def _from_validated(" not in vector_text
    assert "def _from_unvalidated(" not in vector_text
    assert "def _from_validated(" not in matrix_text
    assert "def _from_unvalidated(" not in matrix_text


def test_linalg_arch_025_solve_owner_module_single_owner() -> None:
    """ID: LINALG_ARCH_025_solve_owner_module_single_owner."""
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    assert "class SolveOptions" in solve_ops
    assert "def coerce_solve_options(" in solve_ops
    assert "def compute_solve(" in solve_ops
    assert "def solve(" in solve_ops


def test_linalg_arch_026_solve_reuses_shared_array_plan_and_finalize_owners() -> None:
    """ID: LINALG_ARCH_026_solve_reuses_shared_array_plan_and_finalize_owners."""
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    assert "from .binops import" in solve_ops
    assert "coerce_binary_operands(" in solve_ops
    assert "build_strict_binary_plan(" in solve_ops
    assert "finalize_binary_output(" in solve_ops
    assert "resolve_binary_output_array_type_by_core_arity(" in solve_ops


def test_linalg_arch_027_solve_no_local_schema_mutation_in_kernels() -> None:
    """ID: LINALG_ARCH_027_solve_no_local_schema_mutation_in_kernels."""
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    assert 'attrs["tal"]' not in solve_ops
    assert "attrs['tal']" not in solve_ops


def test_linalg_arch_028_solve_no_duplicate_binop_helper_clusters() -> None:
    """ID: LINALG_ARCH_028_solve_no_duplicate_binop_helper_clusters."""
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    assert "def _coerce_operand(" not in solve_ops
    assert "def _build_finalize_spec(" not in solve_ops
    assert "def finalize_binary_output(" not in solve_ops


def test_linalg_arch_029_solve_no_result_type_output_dtype_forcing() -> None:
    """ID: LINALG_ARCH_029_solve_no_result_type_output_dtype_forcing."""
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    assert "output_dtypes=[np.result_type(" not in solve_ops
    assert "np.result_type(left.dtype, rhs.dtype)" not in solve_ops


def test_linalg_arch_030_inv_owner_module_single_owner() -> None:
    """ID: LINALG_ARCH_030_inv_owner_module_single_owner."""
    inv_ops = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    assert "def compute_inv_kernel(" in inv_ops
    assert "def inv(" in inv_ops


def test_linalg_arch_031_inv_reuses_shared_linear_system_helpers() -> None:
    """ID: LINALG_ARCH_031_inv_reuses_shared_linear_system_helpers."""
    inv_ops = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    linear_systems = Path("tal/linalg/ops/linear_systems.py").read_text(encoding="utf-8")
    assert "def enforce_unchunked_linear_system_inputs(" in linear_systems
    assert "def require_matrix_core_dims(" in linear_systems
    assert "def require_square_matrix(" in linear_systems
    assert "from .linear_systems import" in inv_ops
    assert "from .linear_systems import" in solve_ops
    assert "enforce_unchunked_linear_system_inputs(" in inv_ops
    assert "require_matrix_core_dims(" in inv_ops
    assert "require_square_matrix(" in inv_ops
    assert "enforce_unchunked_linear_system_inputs(" in solve_ops
    assert "require_matrix_core_dims(" in solve_ops
    assert "require_square_matrix(" in solve_ops


def test_linalg_arch_032_inv_no_local_schema_mutation_in_kernel() -> None:
    """ID: LINALG_ARCH_032_inv_no_local_schema_mutation_in_kernel."""
    inv_ops = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    assert 'attrs["tal"]' not in inv_ops
    assert "attrs['tal']" not in inv_ops


def test_linalg_arch_033_solve_inv_shared_linear_system_guards_single_owner() -> None:
    """ID: LINALG_ARCH_033_solve_inv_shared_linear_system_guards_single_owner."""
    inv_ops = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    assert "def _preflight_chunked(" not in solve_ops
    assert "from .linear_systems import" in solve_ops
    assert "from .linear_systems import" in inv_ops


def test_linalg_arch_034_unary_scaffold_single_owner_module() -> None:
    """ID: LINALG_ARCH_034_unary_scaffold_single_owner_module."""
    unary_ops = Path("tal/linalg/ops/unary.py").read_text(encoding="utf-8")
    assert "def coerce_unary_operand(" in unary_ops
    assert "def build_strict_unary_plan(" in unary_ops
    assert "def finalize_unary_output(" in unary_ops


def test_linalg_arch_035_inv_pinv_reuse_unary_scaffold() -> None:
    """ID: LINALG_ARCH_035_inv_pinv_reuse_unary_scaffold."""
    inv_ops = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    pinv_ops = Path("tal/linalg/ops/pinv.py").read_text(encoding="utf-8")
    assert "from .unary import" in inv_ops
    assert "from .unary import" in pinv_ops
    assert "build_strict_unary_plan(" in inv_ops
    assert "build_strict_unary_plan(" in pinv_ops
    assert "finalize_unary_output(" in inv_ops
    assert "finalize_unary_output(" in pinv_ops
    assert "def _coerce_operand(" not in inv_ops
    assert "def _build_inv_plan(" not in inv_ops
    assert "def _finalize_inv_output(" not in inv_ops


def test_linalg_arch_036_linalgerror_normalization_single_owner() -> None:
    """ID: LINALG_ARCH_036_linalgerror_normalization_single_owner."""
    linear_systems = Path("tal/linalg/ops/linear_systems.py").read_text(encoding="utf-8")
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    inv_ops = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    pinv_ops = Path("tal/linalg/ops/pinv.py").read_text(encoding="utf-8")
    assert "def run_with_linalgerror_normalization(" in linear_systems
    assert "run_with_linalgerror_normalization(" in solve_ops
    assert "run_with_linalgerror_normalization(" in inv_ops
    assert "run_with_linalgerror_normalization(" in pinv_ops


def test_linalg_arch_037_no_local_linalgerror_try_except_duplication() -> None:
    """ID: LINALG_ARCH_037_no_local_linalgerror_try_except_duplication."""
    solve_ops = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    inv_ops = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    pinv_ops = Path("tal/linalg/ops/pinv.py").read_text(encoding="utf-8")
    for text in (solve_ops, inv_ops, pinv_ops):
        assert "except np.linalg.LinAlgError" not in text


def test_linalg_arch_038_solve_entrypoint_within_function_budget() -> None:
    """ID: LINALG_ARCH_038_solve_entrypoint_within_function_budget."""
    path = Path("tal/linalg/ops/solve.py")
    module = ast.parse(path.read_text(encoding="utf-8"))
    solve_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    length = function_loc(solve_node, path=path)
    assert length <= 50


def test_linalg_doc_001_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_001_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_vector = Path("docs/api/types/vector.md").read_text(encoding="utf-8")
    api_matrix = Path("docs/api/types/matrix.md").read_text(encoding="utf-8")
    api_cov = Path("docs/api/types/covariance.md").read_text(encoding="utf-8")
    assert "set_core_dims" in user_guide
    assert "tal.linalg.Array" in api_array
    assert "tal.linalg.Vector" in api_vector
    assert "tal.linalg.Matrix" in api_matrix
    assert "Covariance" in api_cov


def test_linalg_doc_002_array_constructor_core_dims_preconditions_documented() -> None:
    """ID: LINALG_DOC_002_array_constructor_core_dims_preconditions_documented."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    assert "set_core_dims" in user_guide
    assert "core_dims" in api_array


def test_linalg_doc_003_add_sub_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_003_add_sub_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    assert "tal.linalg.add" in user_guide
    assert "sub" in user_guide
    assert "__add__" in api_array
    assert "__sub__" in api_array


def test_linalg_doc_004_vector_matrix_runtime_surface_documented() -> None:
    """ID: LINALG_DOC_004_vector_matrix_runtime_surface_documented."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_vector = Path("docs/api/types/vector.md").read_text(encoding="utf-8")
    api_matrix = Path("docs/api/types/matrix.md").read_text(encoding="utf-8")
    assert "Vector" in user_guide
    assert "Matrix" in user_guide
    assert "Matrix.T" in api_matrix
    assert "core dimension" in api_vector


def test_linalg_doc_005_solve_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_005_solve_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_matrix = Path("docs/api/types/matrix.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "tal.linalg.solve" in user_guide
    assert "Matrix.solve" in user_guide
    assert "tal.linalg.solve" in api_array
    assert "Matrix.solve" in api_matrix
    assert "matrix" in api_types_index


def test_linalg_doc_006_inv_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_006_inv_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_matrix = Path("docs/api/types/matrix.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "inv" in user_guide
    assert "Matrix.inv" in user_guide
    assert "tal.linalg.inv" in api_array
    assert "Matrix.inv" in api_matrix
    assert "matrix" in api_types_index


def test_linalg_doc_007_pinv_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_007_pinv_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_matrix = Path("docs/api/types/matrix.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "pinv" in user_guide
    assert "Matrix.pinv" in user_guide
    assert "tal.linalg.pinv" in api_array
    assert "Matrix.pinv" in api_matrix
    assert "pinv" in user_guide
    assert "matrix" in api_types_index


def test_linalg_doc_008_array_runtime_surface_slice_label_d1_consistent() -> None:
    """ID: LINALG_DOC_008_array_runtime_surface_slice_label_d1_consistent."""
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")


def test_linalg_arch_039_vector_ops_owner_module_single_owner() -> None:
    """ID: LINALG_ARCH_039_vector_ops_owner_module_single_owner."""
    vector_ops = Path("tal/linalg/ops/vector_ops.py").read_text(encoding="utf-8")
    assert "def require_vector_core_dim(" in vector_ops
    assert "def require_matching_vector_core_dim(" in vector_ops
    assert "def coerce_norm_ord(" in vector_ops


def test_linalg_arch_040_dot_norm_reuse_shared_plan_finalize_and_result_type_owners() -> None:
    """ID: LINALG_ARCH_040_dot_norm_reuse_shared_plan_finalize_and_result_type_owners."""
    dot_ops = Path("tal/linalg/ops/dot.py").read_text(encoding="utf-8")
    norm_ops = Path("tal/linalg/ops/norm.py").read_text(encoding="utf-8")
    assert "from .binops import" in dot_ops
    assert "build_strict_binary_plan(" in dot_ops
    assert "finalize_binary_output(" in dot_ops
    assert "resolve_binary_output_array_type_by_core_arity(" in dot_ops
    assert "from .unary import" in norm_ops
    assert "build_strict_unary_plan(" in norm_ops
    assert "finalize_unary_output(" in norm_ops
    assert "resolve_binary_output_array_type_by_core_arity(" in norm_ops


def test_linalg_arch_041_vector_dot_norm_no_local_schema_mutation_in_kernels() -> None:
    """ID: LINALG_ARCH_041_vector_dot_norm_no_local_schema_mutation_in_kernels."""
    dot_ops = Path("tal/linalg/ops/dot.py").read_text(encoding="utf-8")
    norm_ops = Path("tal/linalg/ops/norm.py").read_text(encoding="utf-8")
    for text in (dot_ops, norm_ops):
        assert 'attrs["tal"]' not in text
        assert "attrs['tal']" not in text
        assert ".compute(" not in text
        assert ".item(" not in text
        assert ".values" not in text
        assert "np.asarray(" not in text


def test_linalg_arch_042_dot_norm_no_duplicate_vector_guard_clusters() -> None:
    """ID: LINALG_ARCH_042_dot_norm_no_duplicate_vector_guard_clusters."""
    dot_ops = Path("tal/linalg/ops/dot.py").read_text(encoding="utf-8")
    norm_ops = Path("tal/linalg/ops/norm.py").read_text(encoding="utf-8")
    assert "def require_vector_core_dim(" not in dot_ops
    assert "def require_matching_vector_core_dim(" not in dot_ops
    assert "def coerce_norm_ord(" not in dot_ops
    assert "def require_vector_core_dim(" not in norm_ops
    assert "def require_matching_vector_core_dim(" not in norm_ops
    assert "def coerce_norm_ord(" not in norm_ops
    assert "from .vector_ops import" in dot_ops
    assert "from .vector_ops import" in norm_ops


def test_linalg_doc_009_dot_norm_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_009_dot_norm_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_vector = Path("docs/api/types/vector.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "tal.linalg.dot" in user_guide
    assert "norm" in user_guide
    assert "Vector.dot" in user_guide
    assert "Vector.norm" in user_guide
    assert "tal.linalg.dot" in api_array
    assert "tal.linalg.norm" in api_array
    assert "Vector.dot" in api_vector
    assert "Vector.norm" in api_vector
    assert "vector" in api_types_index


def test_linalg_arch_043_linalg_assemble_core_delegates_to_core_combine_owner() -> None:
    """ID: LINALG_ARCH_043_linalg_assemble_core_delegates_to_core_combine_owner."""
    assemble_ops = Path("tal/linalg/ops/assemble_core.py").read_text(encoding="utf-8")
    assert "from ...core import assemble_core as _core_assemble_core" in assemble_ops
    assert "from ...core import stack_core as _core_stack_core" in assemble_ops
    assert "from ...core import block_core as _core_block_core" in assemble_ops
    assert "def assemble_core(" in assemble_ops
    assert "def stack_core(" in assemble_ops
    assert "def block_core(" in assemble_ops
    assert "build_array_plan(" not in assemble_ops
    assert "align_contexts(" not in assemble_ops
    assert "finalize_array_result(" not in assemble_ops


def test_linalg_arch_044_array_typed_constructor_helpers_no_local_orchestration_duplication() -> None:
    """ID: LINALG_ARCH_044_array_typed_constructor_helpers_no_local_orchestration_duplication."""
    array_text = Path("tal/linalg/array.py").read_text(encoding="utf-8")
    assert "def assemble_core(" in array_text
    assert "def stack_core(" in array_text
    assert "def block_core(" in array_text
    assert "from .ops import assemble_core" in array_text
    assert "from .ops import stack_core" in array_text
    assert "from .ops import block_core" in array_text
    assert "build_array_plan(" not in array_text
    assert "align_contexts(" not in array_text
    assert "finalize_array_result(" not in array_text


def test_linalg_arch_045_linalg_concat_core_delegates_to_core_owner() -> None:
    """ID: LINALG_ARCH_045_linalg_concat_core_delegates_to_core_owner."""
    concat_ops = Path("tal/linalg/ops/concat_core.py").read_text(encoding="utf-8")
    assert "from ...core import concat_core as _core_concat_core" in concat_ops
    assert "def concat_core(" in concat_ops
    assert "align_contexts(" not in concat_ops
    assert "build_array_plan(" not in concat_ops
    assert "finalize_array_result(" not in concat_ops


def test_linalg_arch_046_array_concat_core_constructor_no_local_orchestration_duplication() -> None:
    """ID: LINALG_ARCH_046_array_concat_core_constructor_no_local_orchestration_duplication."""
    array_text = Path("tal/linalg/array.py").read_text(encoding="utf-8")
    assert "def concat_core(" in array_text
    assert "from .ops import concat_core" in array_text
    assert "build_array_plan(" not in array_text
    assert "align_contexts(" not in array_text
    assert "finalize_array_result(" not in array_text


def test_linalg_doc_010_assemble_core_stack_block_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_010_assemble_core_stack_block_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "tal.linalg.assemble_core" in user_guide
    assert "tal.linalg.stack_core" in user_guide
    assert "tal.linalg.block_core" in user_guide
    assert "Array.assemble_core" in api_array
    assert "Array.stack_core" in api_array
    assert "Array.block_core" in api_array
    assert "array" in api_types_index


def test_linalg_doc_011_concat_core_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_011_concat_core_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "tal.linalg.concat_core" in user_guide
    assert "concat_core" in api_array
    assert "Array.concat_core" in api_array
    assert "array" in api_types_index


def test_linalg_arch_047_vector3_owner_module_single_owner() -> None:
    """ID: LINALG_ARCH_047_vector3_owner_module_single_owner."""
    vector3_text = Path("tal/linalg/vector3.py").read_text(encoding="utf-8")
    assert "class Vector3(Vector):" in vector3_text
    assert "def from_xyz(" in vector3_text
    assert "def x(" in vector3_text
    assert "def y(" in vector3_text
    assert "def z(" in vector3_text


def test_linalg_arch_048_vector3_invariants_enforced_on_rewrap_boundaries() -> None:
    """ID: LINALG_ARCH_048_vector3_invariants_enforced_on_rewrap_boundaries."""
    vector3_text = Path("tal/linalg/vector3.py").read_text(encoding="utf-8")
    lifecycle_text = Path("tal/linalg/lifecycle.py").read_text(encoding="utf-8")
    assert "LIFECYCLE = VECTOR3_LIFECYCLE" in vector3_text
    assert "def enforce_vector3_invariants(" in lifecycle_text
    assert "def _enforce_array_invariants(" not in vector3_text
    assert "def _from_validated(" not in vector3_text
    assert "def _from_unvalidated(" not in vector3_text


def test_linalg_arch_049_result_type_treats_vector3_as_builtin_not_custom_left_wins() -> None:
    """ID: LINALG_ARCH_049_result_type_treats_vector3_as_builtin_not_custom_left_wins."""
    result_type = Path("tal/linalg/result_type.py").read_text(encoding="utf-8")
    assert "from .vector3 import Vector3" in result_type
    assert "builtin_types = (Array, Vector, Vector3, Matrix)" in result_type


def test_linalg_arch_050_result_type_arity_safe_typed_subclass_fallback_policy() -> None:
    """ID: LINALG_ARCH_050_result_type_arity_safe_typed_subclass_fallback_policy."""
    result_type = Path("tal/linalg/result_type.py").read_text(encoding="utf-8")
    assert "def _left_declared_core_arity(" in result_type
    assert "def _should_preserve_custom_left_for_arity_route(" in result_type
    assert "if issubclass(left_cls, Vector3):" in result_type
    assert "if not issubclass(left_cls, (Vector, Matrix)):" in result_type
    assert "left_arity == len(output_core_dims)" in result_type


def test_linalg_arch_051_vector3_from_xyz_semantic_dim_order_normalization_present() -> None:
    """ID: LINALG_ARCH_051_vector3_from_xyz_semantic_dim_order_normalization_present."""
    vector3_text = Path("tal/linalg/vector3.py").read_text(encoding="utf-8")
    assert "def _semantic_dim_order(" in vector3_text
    assert "semantic_order = _semantic_dim_order(" in vector3_text
    assert "assembled = assembled.transpose(*semantic_order, validate=validate)" in vector3_text


def test_linalg_arch_052_linalg_decompose_core_delegates_to_core_owner() -> None:
    """ID: LINALG_ARCH_052_linalg_decompose_core_delegates_to_core_owner."""
    decompose_ops = Path("tal/linalg/ops/decompose_core.py").read_text(encoding="utf-8")
    assert "from ...core import decompose_core as _core_decompose_core" in decompose_ops
    assert "def decompose_core(" in decompose_ops
    assert "align_contexts(" not in decompose_ops
    assert "build_array_plan(" not in decompose_ops
    assert "finalize_array_result(" not in decompose_ops


def test_linalg_arch_053_array_decompose_core_method_no_local_orchestration_duplication() -> None:
    """ID: LINALG_ARCH_053_array_decompose_core_method_no_local_orchestration_duplication."""
    array_text = Path("tal/linalg/array.py").read_text(encoding="utf-8")
    assert "def decompose_core(" in array_text
    assert "from .ops import decompose_core" in array_text
    assert "build_array_plan(" not in array_text
    assert "align_contexts(" not in array_text
    assert "finalize_array_result(" not in array_text


def test_linalg_arch_054_decompose_core_opts_required_signature_alignment() -> None:
    """ID: LINALG_ARCH_054_decompose_core_opts_required_signature_alignment."""
    core_ops = Path("tal/core/combine_ops/decompose_core.py").read_text(encoding="utf-8")
    core_accessor = Path("tal/core/combine_ops/accessor.py").read_text(encoding="utf-8")
    linalg_ops = Path("tal/linalg/ops/decompose_core.py").read_text(encoding="utf-8")
    linalg_array = Path("tal/linalg/array.py").read_text(encoding="utf-8")
    assert "opts: CoreDecomposeOptions | None = None" not in core_ops
    assert "opts: CoreDecomposeOptions | None = None" not in core_accessor
    assert "opts: CoreDecomposeOptions | None = None" not in linalg_ops
    assert "opts: \"CoreDecomposeOptions | None\" = None" not in linalg_array
    assert "opts: CoreDecomposeOptions," in core_ops
    assert "opts: CoreDecomposeOptions," in core_accessor
    assert "opts: CoreDecomposeOptions," in linalg_ops
    assert "opts: \"CoreDecomposeOptions\"," in linalg_array


def test_linalg_doc_012_vector3_runtime_surface_documented() -> None:
    """ID: LINALG_DOC_012_vector3_runtime_surface_documented."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_vector = Path("docs/api/types/vector.md").read_text(encoding="utf-8")
    api_vector3 = Path("docs/api/types/vector3.md").read_text(encoding="utf-8")
    assert "Vector3" in user_guide
    assert "Vector3.from_xyz" in user_guide
    assert "tal.linalg.Vector3" in api_vector3
    assert "Vector3.from_xyz" in api_vector3


def test_linalg_doc_013_vector3_dim_order_and_matrix_return_specificity_documented() -> None:
    """ID: LINALG_DOC_013_vector3_dim_order_and_matrix_return_specificity_documented."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_vector3 = Path("docs/api/types/vector3.md").read_text(encoding="utf-8")
    api_matrix = Path("docs/api/types/matrix.md").read_text(encoding="utf-8")
    assert "Vector3.from_xyz" in user_guide
    assert "Vector3.from_xyz" in api_vector3
    assert "Matrix.solve" in api_matrix
    assert "Matrix.inv" in api_matrix
    assert "Matrix.pinv" in api_matrix


def test_linalg_doc_014_decompose_core_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_014_decompose_core_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "tal.linalg.decompose_core" in user_guide
    assert "Array.decompose_core" in api_array
    assert "tal.linalg.decompose_core" in api_array
    assert "array" in api_types_index


def test_linalg_arch_055_linalg_overlay_core_delegates_to_core_owner() -> None:
    """ID: LINALG_ARCH_055_linalg_overlay_core_delegates_to_core_owner."""
    overlay_ops = Path("tal/linalg/ops/overlay_core.py").read_text(encoding="utf-8")
    assert "from ...core import overlay_core as _core_overlay_core" in overlay_ops
    assert "def overlay_core(" in overlay_ops
    assert "align_contexts(" not in overlay_ops
    assert "build_array_plan(" not in overlay_ops
    assert "finalize_array_result(" not in overlay_ops


def test_linalg_arch_056_array_overlay_core_method_no_local_orchestration_duplication() -> None:
    """ID: LINALG_ARCH_056_array_overlay_core_method_no_local_orchestration_duplication."""
    array_text = Path("tal/linalg/array.py").read_text(encoding="utf-8")
    assert "def overlay_core(" in array_text
    assert "from .ops import overlay_core" in array_text
    assert "build_array_plan(" not in array_text
    assert "align_contexts(" not in array_text
    assert "finalize_array_result(" not in array_text


def test_linalg_arch_057_overlay_core_base_type_rewrap_policy_single_owner() -> None:
    """ID: LINALG_ARCH_057_overlay_core_base_type_rewrap_policy_single_owner."""
    overlay_ops = Path("tal/linalg/ops/overlay_core.py").read_text(encoding="utf-8")
    array_text = Path("tal/linalg/array.py").read_text(encoding="utf-8")
    assert "def _base_array_type(" in overlay_ops
    assert "def _rewrap_overlay_output(" in overlay_ops
    assert "base_type._from_validated(" in overlay_ops
    assert "Array._from_validated(" in overlay_ops
    assert "def _rewrap_overlay_output(" not in array_text


def test_linalg_doc_015_overlay_core_user_guide_and_api_entries_present() -> None:
    """ID: LINALG_DOC_015_overlay_core_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/linalg.md").read_text(encoding="utf-8")
    api_array = Path("docs/api/types/array.md").read_text(encoding="utf-8")
    api_types_index = Path("docs/api/types/index.md").read_text(encoding="utf-8")
    assert "tal.linalg.overlay_core" in user_guide
    assert "Array.overlay_core" in api_array
    assert "tal.linalg.overlay_core" in api_array
    assert "array" in api_types_index


def test_linalg_arch_058_dataset_context_owner_reused_by_plan_and_component_context() -> None:
    """ID: LINALG_ARCH_058_dataset_context_owner_reused_by_plan_and_component_context."""
    plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    component_text = Path("tal/linalg/component_context.py").read_text(encoding="utf-8")
    vector3_text = Path("tal/linalg/vector3.py").read_text(encoding="utf-8")
    assert "resolve_dataset_context(" in plan_text
    assert "resolve_dataset_context(" in component_text
    assert "build_reference_component_context(" in vector3_text
    assert "def _select_single_numeric_var(" not in vector3_text


def test_linalg_arch_059_operand_coercion_reuses_orchestration_inputs_owner() -> None:
    """ID: LINALG_ARCH_059_operand_coercion_reuses_orchestration_inputs_owner."""
    inputs_text = Path("tal/core/orchestration/inputs.py").read_text(encoding="utf-8")
    binops_text = Path("tal/linalg/ops/binops.py").read_text(encoding="utf-8")
    unary_text = Path("tal/linalg/ops/unary.py").read_text(encoding="utf-8")
    component_text = Path("tal/linalg/component_context.py").read_text(encoding="utf-8")
    assert "def coerce_operand(" in inputs_text
    assert "coerce_operand(" in binops_text
    assert "coerce_operand(" in unary_text
    assert "coerce_operand(" in component_text
    assert "def _coerce_operand(" not in binops_text


def test_linalg_arch_060_array_finalize_reuses_core_schema_finalize_owner() -> None:
    """ID: LINALG_ARCH_060_array_finalize_reuses_core_schema_finalize_owner."""
    linalg_finalize = Path("tal/linalg/finalize.py").read_text(encoding="utf-8")
    schema_finalize = Path("tal/core/orchestration/schema_finalize.py").read_text(encoding="utf-8")
    assert "finalize_with_schema(" in linalg_finalize
    assert "class CoreSchemaFinalizeSpec" in schema_finalize
    assert "def clear_core_schema_blocks(" in schema_finalize
    assert "def stamp_core_schema(" in schema_finalize


def test_linalg_arch_061_build_array_plan_respects_agents_function_budget_with_helper_split() -> None:
    """ID: LINALG_ARCH_061_build_array_plan_respects_agents_function_budget_with_helper_split."""
    plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    module = ast.parse(plan_text)
    node = next(
        item for item in module.body if isinstance(item, ast.FunctionDef) and item.name == "build_array_plan"
    )
    length = function_loc(node, source=plan_text)
    assert length <= 50
    assert "def _build_context_options(" in plan_text
    assert "def _collect_operand_contexts(" in plan_text
    assert "def _assemble_array_plan(" in plan_text
    assert "_collect_operand_contexts(" in plan_text
    assert "_assemble_array_plan(" in plan_text


def test_arch_topo_002_linalg_plan_delegates_topology_to_core_owner() -> None:
    """ID: ARCH_TOPO_002_linalg_plan_delegates_topology_to_core_owner."""
    plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    assert "from ..core.orchestration.topology import" in plan_text
    assert "resolve_unary_topology(" in plan_text
    assert "resolve_binary_topology(" in plan_text
    assert "resolve_nary_topology(" in plan_text
    assert "resolve_semantic_topology_from_dataset(" in plan_text
    assert "_validate_context_compatibility" not in plan_text


def test_arch_topo_010_linalg_plan_uses_strict_exact_policy_constant() -> None:
    """ID: ARCH_TOPO_010_linalg_plan_uses_strict_exact_policy_constant."""
    plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    assert "STRICT_EXACT_POLICY" in plan_text
    assert "SEMANTIC_EXACT_POLICY" in plan_text
    assert "select_topology_policy_with_intents(" in plan_text
    assert 'operation_family="linalg.elementwise"' in plan_text
    assert "strict_policy=STRICT_EXACT_POLICY" in plan_text
    assert "semantic_policy=SEMANTIC_EXACT_POLICY" in plan_text


def test_arch_topo_012_linalg_relaxed_roles_path_not_dead() -> None:
    """ID: ARCH_TOPO_012_linalg_relaxed_roles_path_not_dead."""
    plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    assert "require_declared_roles: bool = True" in plan_text
    assert "def _semantic_topology_for_context(" in plan_text
    assert "if context.roles_declared:" in plan_text
    assert "if require_declared_roles:" in plan_text
    assert "SemanticTopology(" in plan_text


def test_arch_bcast_005_linalg_consumers_do_not_reimplement_broadcast_policy() -> None:
    """ID: ARCH_BCAST_005_linalg_consumers_do_not_reimplement_broadcast_policy."""
    plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    assert "select_topology_policy_with_intents" in plan_text
    assert "resolve_unary_topology(" in plan_text
    assert "resolve_binary_topology(" in plan_text
    assert "resolve_nary_topology(" in plan_text
    assert "semantic_broadcast only permits" not in plan_text


def test_arch_bcast_008_unary_binary_nary_paths_route_through_shared_topology_owner() -> None:
    """ID: ARCH_BCAST_008_unary_binary_nary_paths_route_through_shared_topology_owner."""
    topology_text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    plan_text = Path("tal/linalg/plan.py").read_text(encoding="utf-8")
    assert "def resolve_unary_topology(" in topology_text
    assert "def resolve_binary_topology(" in topology_text
    assert "def resolve_nary_topology(" in topology_text
    assert "resolve_unary_topology(" in plan_text
    assert "resolve_binary_topology(" in plan_text
    assert "resolve_nary_topology(" in plan_text


def test_linalg_arch_062_inv_kernel_contains_no_vectorize_true() -> None:
    """ID: LINALG_ARCH_062_inv_kernel_contains_no_vectorize_true."""
    text = Path("tal/linalg/ops/inv.py").read_text(encoding="utf-8")
    section = text.split("def compute_inv_kernel(", 1)[1].split("def inv(", 1)[0]
    assert "vectorize=True" not in section
    assert "vectorize=False" in section


def test_linalg_arch_063_pinv_kernel_contains_no_vectorize_true() -> None:
    """ID: LINALG_ARCH_063_pinv_kernel_contains_no_vectorize_true."""
    text = Path("tal/linalg/ops/pinv.py").read_text(encoding="utf-8")
    section = text.split("def compute_pinv_kernel(", 1)[1].split("def pinv(", 1)[0]
    assert "vectorize=True" not in section
    assert "vectorize=False" in section


def test_linalg_arch_064_solve_kernel_contains_no_vectorize_true() -> None:
    """ID: LINALG_ARCH_064_solve_kernel_contains_no_vectorize_true."""
    text = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    solve_section = text.split("def compute_solve_kernel(", 1)[1].split("def compute_lstsq_kernel(", 1)[0]
    assert "vectorize=True" not in solve_section
    assert "vectorize=False" in solve_section


def test_linalg_arch_065_lstsq_kernel_stopgap_routes_through_backend_owner() -> None:
    """ID: LINALG_ARCH_065_lstsq_kernel_stopgap_routes_through_backend_owner."""
    solve_text = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    backend_text = Path("tal/linalg/ops/solve_backends.py").read_text(encoding="utf-8")
    lstsq_section = solve_text.split("def compute_lstsq_kernel(", 1)[1].split("def compute_solve(", 1)[0]
    assert "vectorize=True" in lstsq_section
    assert "lstsq_solution_backend" in lstsq_section
    assert "LSTSQ_BACKEND_NUMPY_ROW" in lstsq_section
    assert "def lstsq_solution_backend(" in backend_text


def test_linalg_arch_069_lstsq_backend_selector_remains_owner_routed() -> None:
    """ID: LINALG_ARCH_069_lstsq_backend_selector_remains_owner_routed."""
    solve_text = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    backend_text = Path("tal/linalg/ops/solve_backends.py").read_text(encoding="utf-8")
    lstsq_section = solve_text.split("def compute_lstsq_kernel(", 1)[1].split("def compute_solve(", 1)[0]
    assert 'LSTSQ_BACKEND_NUMBA = "numba"' in backend_text
    assert "def lstsq_block_backend(" in backend_text
    assert "def _select_lstsq_backend(" in backend_text
    assert "from .numba_backends import lstsq_block_numba" in backend_text
    assert Path("tal/linalg/ops/numba_backends.py").exists()
    assert "LSTSQ_BACKEND_NUMPY_ROW" in lstsq_section
    assert "vectorize=True" in lstsq_section
    assert "LSTSQ_BACKEND_NUMBA" not in lstsq_section


def test_linalg_arch_070_lstsq_normal_path_status_is_explicit() -> None:
    """ID: LINALG_ARCH_070_lstsq_normal_path_status_is_explicit."""
    contract_083 = Path("contracts/083-compiled-kernel-backend-followon-phase-f2.md").read_text(encoding="utf-8")
    contract_114 = Path("contracts/114-numba-default-baseline-migration-slice-f2c.md").read_text(encoding="utf-8")
    benchmark = Path("benchmarks/bench_linalg_lstsq_numba_backends.py").read_text(encoding="utf-8")
    solve_text = Path("tal/linalg/ops/solve.py").read_text(encoding="utf-8")
    lstsq_section = solve_text.split("def compute_lstsq_kernel(", 1)[1].split("def compute_solve(", 1)[0]
    decision = contract_114.split("### linalg_lstsq", 1)[1].split("\n## ", 1)[0]
    assert "Status: Draft" in contract_083
    assert "event/linalg closeout open" in contract_083
    assert "Decision: promote" in decision
    assert "Gate result: PASS" in decision
    assert "LSTSQ_BACKEND_NUMPY_ROW" in lstsq_section
    assert "vectorize=True" in lstsq_section
    assert "LSTSQ_BACKEND_NUMBA" not in lstsq_section
    assert "linalg lstsq F2C decision input" in benchmark
    assert "fewer-large cases numpy_row-selected" in benchmark


def test_linalg_arch_067_linalg_classes_do_not_duplicate_lifecycle_methods() -> None:
    """ID: LINALG_ARCH_067_linalg_classes_do_not_duplicate_lifecycle_methods."""
    array_methods = _class_method_names(_module("tal/linalg/array.py"), "Array")
    assert "_from_validated" not in array_methods
    assert "_from_unvalidated" not in array_methods
    assert "_enforce_array_invariants" not in array_methods

    for rel, class_name in (
        ("tal/linalg/vector.py", "Vector"),
        ("tal/linalg/matrix.py", "Matrix"),
        ("tal/linalg/vector3.py", "Vector3"),
    ):
        methods = _class_method_names(_module(rel), class_name)
        assert "__init__" not in methods
        assert "_enforce_array_invariants" not in methods


def test_linalg_arch_068_linalg_typed_lifecycle_specs_stay_domain_owned() -> None:
    """ID: LINALG_ARCH_068_linalg_typed_lifecycle_specs_stay_domain_owned."""
    lifecycle_path = Path("tal/linalg/lifecycle.py")
    lifecycle_text = lifecycle_path.read_text(encoding="utf-8")
    core_text = Path("tal/core/typed_lifecycle.py").read_text(encoding="utf-8")

    assert lifecycle_path.exists()
    assert "ARRAY_LIFECYCLE" in lifecycle_text
    assert "VECTOR_LIFECYCLE" in lifecycle_text
    assert "MATRIX_LIFECYCLE" in lifecycle_text
    assert "VECTOR3_LIFECYCLE" in lifecycle_text
    assert "VECTOR3_LIFECYCLE" not in core_text
    assert "Vector3" not in core_text
