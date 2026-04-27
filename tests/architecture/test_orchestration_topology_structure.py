from __future__ import annotations

from pathlib import Path


def test_orch_topo_arch_001_single_owner_flatten_query_restore_wrappers() -> None:
    """ID: ORCH_TOPO_ARCH_001_single_owner_flatten_query_restore_wrappers."""
    text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    batch_text = Path("tal/core/orchestration/topology_batch.py").read_text(encoding="utf-8")
    assert "from .topology_batch import (" in text
    for needle in [
        "flatten_param_contexts,",
        "flatten_query_for_batch_plan,",
        "restore_dataset_batch_topology,",
        "join_batch_indices,",
        "batch_index_for_dataset,",
    ]:
        assert needle in text
    for needle in [
        "def flatten_param_contexts(",
        "def flatten_query_for_batch_plan(",
        "def restore_dataset_batch_topology(",
        "def join_batch_indices(",
        "def batch_index_for_dataset(",
    ]:
        assert needle in batch_text


def test_orch_topo_arch_002_no_paramops_direct_batch_topology_calls_in_migrated_files() -> None:
    """ID: ORCH_TOPO_ARCH_002_no_paramops_direct_batch_topology_calls_in_migrated_files."""
    for rel in [
        "tal/core/param_ops/sync.py",
        "tal/core/param_ops/interp_like.py",
    ]:
        text = Path(rel).read_text(encoding="utf-8")
        assert "from .batch_topology import" not in text
        assert "flatten_batch_contexts(" not in text
        assert "flatten_query_for_plan(" not in text
        assert "restore_dataset_batch_dims(" not in text


def test_orch_topo_arch_003_no_combine_local_batch_join_allocator_in_migrated_files() -> None:
    """ID: ORCH_TOPO_ARCH_003_no_combine_local_batch_join_allocator_in_migrated_files."""
    for rel in [
        "tal/core/combine_ops/align.py",
        "tal/core/combine_ops/concat_topology.py",
    ]:
        text = Path(rel).read_text(encoding="utf-8")
        assert "from ..param_ops.batch_labels import batch_index" not in text
        assert "from ..param_ops.batch_labels import join_batch_index" not in text
        assert "from ..param_ops.batch_topology import" not in text


def test_orch_topo_arch_004_orchestration_topology_reuses_batch_labels_and_batch_topology() -> None:
    """ID: ORCH_TOPO_ARCH_004_orchestration_topology_reuses_batch_labels_and_batch_topology."""
    text = Path("tal/core/orchestration/topology.py").read_text(encoding="utf-8")
    orch_batch_text = Path("tal/core/orchestration/topology_batch.py").read_text(encoding="utf-8")
    batch_text = Path("tal/core/param_ops/batch_topology.py").read_text(encoding="utf-8")
    assert "from .topology_batch import (" in text
    assert "from ..param_ops.batch_labels import" in orch_batch_text
    assert "from ..param_ops.batch_topology import" in orch_batch_text
    assert "stacked_batch_coords_with_labels" in orch_batch_text
    assert "def stacked_batch_coords_with_labels(" in batch_text
    helper_body = batch_text.split("def _stack_batch_coords(", 1)[1].split("\ndef ", 1)[0]
    assert "stacked_batch_coords_with_labels(" in helper_body


def test_orch_topo_arch_005_align_concat_batch_topology_single_owner() -> None:
    """ID: ORCH_TOPO_ARCH_005_align_concat_batch_topology_single_owner."""
    align_text = Path("tal/core/combine_ops/align.py").read_text(encoding="utf-8")
    concat_text = Path("tal/core/combine_ops/concat_topology.py").read_text(encoding="utf-8")
    for text in [align_text, concat_text]:
        assert "from ..orchestration.topology import" in text
        assert "stack_combine_batch_axis(" in text
        assert "restore_combine_batch_axis(" in text
        assert "join_combine_batch_labels(" in text


def test_arch_topo_016_batch_restore_helpers_contain_no_values_or_np_asarray_materialization_patterns() -> None:
    """ID: ARCH_TOPO_016_batch_restore_helpers_contain_no_values_or_np_asarray_materialization_patterns."""
    text = Path("tal/core/param_ops/batch_topology.py").read_text(encoding="utf-8")
    named_body = text.split("def _batch_cols_from_named_flat_coords(", 1)[1].split("\ndef ", 1)[0]
    restore_cols_body = text.split("def _batch_cols_for_restore(", 1)[1].split("\ndef ", 1)[0]
    restore_body = text.split("def restore_dataset_batch_dims(", 1)[1].split("\ndef ", 1)[0]
    for body in [named_body, restore_cols_body, restore_body]:
        assert ".values" not in body
        assert "np.asarray(" not in body
