from __future__ import annotations

from pathlib import Path


def test_orch_axis_arch_001_single_owner_namespace_and_safe_rename_helpers() -> None:
    """ID: ORCH_AXIS_ARCH_001_single_owner_namespace_and_safe_rename_helpers."""
    text = Path("tal/utils/xarray_namespace.py").read_text(encoding="utf-8")
    for needle in [
        "def dataarray_namespace_names(",
        "def dataset_namespace_names(",
        "def unique_temp_dim(",
        "def rename_dims_collision_safe(",
    ]:
        assert needle in text


def test_orch_axis_arch_002_no_local_temp_dim_loops_in_migrated_files() -> None:
    """ID: ORCH_AXIS_ARCH_002_no_local_temp_dim_loops_in_migrated_files."""
    query_grid = Path("tal/core/param_engine/query_grid.py").read_text(encoding="utf-8")
    evaluate = Path("tal/core/event_ops/evaluate.py").read_text(encoding="utf-8")
    assert "while temp_dim in" not in query_grid
    assert "while level_name in" not in query_grid
    assert "def _dataarray_namespace_names(" not in evaluate
    assert "def _rename_dims(" not in evaluate


def test_orch_axis_arch_003_event_semantic_axis_mapping_owned_by_orchestration() -> None:
    """ID: ORCH_AXIS_ARCH_003_event_semantic_axis_mapping_owned_by_orchestration."""
    evaluate = Path("tal/core/event_ops/evaluate.py").read_text(encoding="utf-8")
    assert "from ..orchestration.axis_map import resolve_role_axis_map" in evaluate
    for local in [
        "def _semantic_dim_map(",
        "def _resolve_batch_dim_map(",
        "def _lock_name_matched_batch_dims(",
        "def _select_unique_batch_mapping(",
    ]:
        assert local not in evaluate


def test_orch_axis_arch_004_no_duplicate_namespace_helper_algorithms_in_guards() -> None:
    """ID: ORCH_AXIS_ARCH_004_no_duplicate_namespace_helper_algorithms_in_guards."""
    guards = Path("tal/core/param_ops/guards.py").read_text(encoding="utf-8")
    assert "from tal.utils.xarray_namespace import dataset_namespace_names, unique_temp_dim" in guards
    assert "def unique_temp_dim(" not in guards
    assert "def dataset_namespace_names(" not in guards

