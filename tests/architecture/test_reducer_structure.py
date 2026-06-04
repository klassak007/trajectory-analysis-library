from __future__ import annotations

from pathlib import Path

from ._budget import file_loc, function_lengths


CORE_REDUCER_DIR = Path("tal/core/reducer_ops")


def test_arch_reduce_p9c_001_core_reducer_owners_do_not_embed_typed_domain_math() -> None:
    """ID: ARCH_REDUCE_P9C_001_core_reducer_owners_do_not_embed_typed_domain_math."""
    for path in sorted(CORE_REDUCER_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.spatial" not in text
        assert "np.linalg.eigh" not in text


def test_arch_reduce_p9c_002_typed_classes_delegate_and_remain_thin() -> None:
    """ID: ARCH_REDUCE_P9C_002_typed_classes_delegate_and_remain_thin."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    owner_text = Path("tal/spatial/ops/rotation_reduce_ops.py").read_text(encoding="utf-8")
    kernel_text = Path("tal/spatial/kernels/rotation_mean_kernels.py").read_text(encoding="utf-8")
    assert "install_rotation_reducer_methods" in rotation_text
    assert "from ..kernels.rotation_mean_kernels import quat_mean_kernel" in owner_text
    assert "quat_mean_kernel," in owner_text
    assert "np.linalg.eigh" not in owner_text
    assert "def quat_mean_kernel(" in kernel_text
    assert "np.linalg.eigh" in kernel_text


def test_arch_reduce_p9c_003_rotation_reduce_owner_split_and_budget_locked() -> None:
    """ID: ARCH_REDUCE_P9C_003_rotation_reduce_owner_split_and_budget_locked."""
    path = Path("tal/spatial/ops/rotation_reduce_ops.py")
    text = path.read_text(encoding="utf-8")
    assert "def _prepare_reduce_payload(" in text
    assert "def _prepare_weight_for_reduce(" in text
    assert "def _execute_quat_reduce(" in text
    assert "def _finalize_rotation_reduce(" in text
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{path}:{name} exceeds function budget ({length} > 50)."


def test_arch_reduce_p9c_004_rotation_mean_enforces_ndarray_single_dim_contract() -> None:
    """ID: ARCH_REDUCE_P9C_004_rotation_mean_enforces_ndarray_single_dim_contract."""
    text = Path("tal/spatial/ops/rotation_reduce_ops.py").read_text(encoding="utf-8")
    assert "isinstance(weights, np.ndarray) and len(reduce_dims) > 1" in text
    assert "ndarray weights are only valid for single-dim reduction" in text


def test_arch_reduce_p9c_005_rotation_multi_dim_uses_one_pass_reduce_path() -> None:
    """ID: ARCH_REDUCE_P9C_005_rotation_multi_dim_uses_one_pass_reduce_path."""
    text = Path("tal/spatial/ops/rotation_reduce_ops.py").read_text(encoding="utf-8")
    assert "def _reduce_multi_dim(" in text
    assert "return _reduce_multi_dim(" in text
    assert "if len(reduce_dims) > 1:" in text
    assert "def _stack_reduce_dims(" in text


def test_arch_reduce_p9c_006_rotation_finalize_uses_schema_finalize_owner_and_blocks_stale_assign_wrap() -> None:
    """ID: ARCH_REDUCE_P9C_006_rotation_finalize_uses_schema_finalize_owner_and_blocks_stale_assign_wrap."""
    text = Path("tal/spatial/ops/rotation_reduce_ops.py").read_text(encoding="utf-8")
    assert "CoreSchemaFinalizeSpec" in text
    assert "finalize_with_schema(" in text
    assert "assign({var_name: reduced})" not in text


def test_arch_reduce_p9c_007_rotation_quat_dim_resolution_is_shared_and_role_agnostic() -> None:
    """ID: ARCH_REDUCE_P9C_007_rotation_quat_dim_resolution_is_shared_and_role_agnostic."""
    helper = Path("tal/spatial/ops/quat_role_dim_ops.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    owner_text = Path("tal/spatial/ops/rotation_reduce_ops.py").read_text(encoding="utf-8")
    assert "def resolve_quat_dim_with_role_fallback(" in helper
    assert "resolve_quat_dim_with_role_fallback(" in rotation_text
    assert "resolve_quat_dim_with_role_fallback(" in owner_text
    assert "require_single_core_dim_with_length(source, expected_length=4" not in owner_text


def test_arch_reduce_p9c_008_rotation_component_dim_guard_persists_after_sequence_clear() -> None:
    """ID: ARCH_REDUCE_P9C_008_rotation_component_dim_guard_persists_after_sequence_clear."""
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    owner_text = Path("tal/spatial/ops/rotation_reduce_ops.py").read_text(encoding="utf-8")
    assert "def _required_component_dims_for_reduce(self)" in rotation_text
    assert "spatial.rotation._required_component_dims_for_reduce" in rotation_text
    assert "component_dims=rotation._required_component_dims_for_reduce()" in owner_text


def test_arch_reduce_p9c_009_rotation_component_dim_resolver_is_representation_aware_and_shared() -> None:
    """ID: ARCH_REDUCE_P9C_009_rotation_component_dim_resolver_is_representation_aware_and_shared."""
    helper = Path("tal/spatial/ops/quat_role_dim_ops.py").read_text(encoding="utf-8")
    rotation_text = Path("tal/spatial/rotation.py").read_text(encoding="utf-8")
    assert "def resolve_rotation_component_dims_for_reduce(" in helper
    assert "get_rotation_rep(" in helper
    assert "resolve_rotation_component_dims_for_reduce(" in rotation_text


def test_arch_reduce_p9c_010_reducer_finalize_target_policy_is_centralized_and_api_routed() -> None:
    """ID: ARCH_REDUCE_P9C_010_reducer_finalize_target_policy_is_centralized_and_api_routed."""
    policy_text = Path("tal/core/reducer_ops/finalize_policy.py").read_text(encoding="utf-8")
    api_text = Path("tal/core/reducer_ops/api.py").read_text(encoding="utf-8")
    assert "def resolve_reducer_finalize_source(" in policy_text
    assert "resolve_reducer_finalize_source" in api_text
    assert "finalize_source = resolve_reducer_finalize_source(" in api_text
    assert "return finalize_like(finalize_source, out, validate=validate, owner=owner)" in api_text


def test_reduce_hard_p9c_003_no_hidden_eager_materialization_in_reducer_owners() -> None:
    """ID: REDUCE_HARD_P9C_003_no_hidden_eager_materialization_in_reducer_owners."""
    for path in sorted(CORE_REDUCER_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert ".values" not in text
        assert ".compute(" not in text


def test_reducer_owner_budgets() -> None:
    """Reducer owners stay within AGENTS complexity budgets."""
    for path in sorted(CORE_REDUCER_DIR.glob("*.py")):
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget ({length} > 50)."
