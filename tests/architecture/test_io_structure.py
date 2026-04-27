from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.architecture._budget import file_loc, function_lengths


def _has_direct_tal_schema_write(text: str) -> bool:
    return bool(re.search(r"attrs\[['\"]tal['\"]\]\s*=", text))


def test_arch_io_p10a_001_io_parsing_logic_is_outside_tal_core() -> None:
    """ID: ARCH_IO_P10A_001_io_parsing_logic_is_outside_tal_core."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "from tal.io" not in text
        assert "import tal.io" not in text
        assert "from_csv(" not in text
        assert "from_zarr(" not in text


def test_arch_io_p10a_002_ao_direct_io_finalization_reuses_core_schema_owners() -> None:
    """ID: ARCH_IO_P10A_002_ao_direct_io_finalization_reuses_core_schema_owners."""
    finalize_text = Path("tal/io/finalize.py").read_text(encoding="utf-8")
    csv_text = Path("tal/io/csv_io.py").read_text(encoding="utf-8")
    zarr_text = Path("tal/io/zarr_io.py").read_text(encoding="utf-8")
    assert "resolve_dataset_context" in finalize_text
    assert "CoreSchemaFinalizeSpec" in finalize_text
    assert "finalize_with_schema" in finalize_text
    assert "finalize_loaded_dataset(" in csv_text
    assert "finalize_loaded_dataset(" in zarr_text


def test_arch_io_p10a_003_no_catalog_import_dependency_required_for_ao_direct_io() -> None:
    """ID: ARCH_IO_P10A_003_no_catalog_import_dependency_required_for_ao_direct_io."""
    ao_direct_paths = (
        Path("tal/io/csv_io.py"),
        Path("tal/io/zarr_io.py"),
        Path("tal/io/finalize.py"),
        Path("tal/io/surface.py"),
    )
    for path in ao_direct_paths:
        text = path.read_text(encoding="utf-8")
        assert "tal.catalog" not in text
        assert "Catalog" not in text


def test_arch_io_p10a_004_analysisobject_io_surface_installation_is_wired_from_top_level_package_init() -> None:
    """ID: ARCH_IO_P10A_004_analysisobject_io_surface_installation_is_wired_from_top_level_package_init."""
    top_text = Path("tal/__init__.py").read_text(encoding="utf-8")
    assert "from .io import install_analysis_object_io_surface" in top_text
    assert "install_analysis_object_io_surface()" in top_text


def test_arch_io_p10a_005_io_readers_use_requested_cls_as_finalize_source_and_do_not_ignore_cls() -> None:
    """ID: ARCH_IO_P10A_005_io_readers_use_requested_cls_as_finalize_source_and_do_not_ignore_cls."""
    finalize_text = Path("tal/io/finalize.py").read_text(encoding="utf-8")
    csv_text = Path("tal/io/csv_io.py").read_text(encoding="utf-8")
    zarr_text = Path("tal/io/zarr_io.py").read_text(encoding="utf-8")
    assert "def resolve_finalize_source_for_cls(" in finalize_text
    assert "def finalize_loaded_dataset(" in finalize_text
    assert "source_ao = resolve_finalize_source_for_cls(cls, ds, owner=owner)" in finalize_text
    assert "_ = cls" not in csv_text
    assert "_ = cls" not in zarr_text
    assert "finalize_loaded_dataset(cls, ds, validate=validate, owner=owner)" in csv_text
    assert "finalize_loaded_dataset(cls, ds, validate=validate, owner=owner)" in zarr_text


def test_arch_io_p10a_006_finalize_source_subclass_constructor_path_avoids_broad_exception_catch() -> None:
    """ID: ARCH_IO_P10A_006_finalize_source_subclass_constructor_path_avoids_broad_exception_catch."""
    finalize_text = Path("tal/io/finalize.py").read_text(encoding="utf-8")
    assert "def resolve_finalize_source_for_cls(" in finalize_text
    assert "return cls(base.unsafe_data)" in finalize_text
    assert "loaded payload is not valid for requested class" not in finalize_text
    assert "except Exception as exc:" in finalize_text
    assert "invalid persisted schema payload" in finalize_text


def test_io_hard_p10a_003_no_schema_bypass_writes_on_ao_direct_io_paths() -> None:
    """ID: IO_HARD_P10A_003_no_schema_bypass_writes_on_ao_direct_io_paths."""
    for path in sorted(Path("tal/io").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not _has_direct_tal_schema_write(text)


def test_io_doc_p10a_001_ao_direct_io_is_canonical_and_catalog_is_optional_documented() -> None:
    """ID: IO_DOC_P10A_001_ao_direct_io_is_canonical_and_catalog_is_optional_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "ao-direct" in text
    assert "ao.io.to_zarr" in text
    assert "ao.io.to_csv" in text


def test_io_doc_p10a_002_analysisobject_io_top_level_installation_guarantee_documented() -> None:
    """ID: IO_DOC_P10A_002_analysisobject_io_top_level_installation_guarantee_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "analysisobjectioaccessor" in text
    assert "analysisobject.from_zarr" in text


def test_io_doc_p10a_003_subclass_preserving_loader_and_sidecar_validation_semantics_documented() -> None:
    """ID: IO_DOC_P10A_003_subclass_preserving_loader_and_sidecar_validation_semantics_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "loaders" in text
    assert "csv sidecar metadata" in text


def test_io_doc_p10b_001_csv_ros_adapter_policies_and_bridge_behavior_documented() -> None:
    """ID: IO_DOC_P10B_001_csv_ros_adapter_policies_and_bridge_behavior_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "read_csv_logs" in text
    assert "read_ros_logs" in text
    assert "read_csv_logs_catalog" in text
    assert "timestamp" in text


def test_io_doc_p10b_002_ingest_and_export_label_ordering_policies_documented() -> None:
    """ID: IO_DOC_P10B_002_ingest_and_export_label_ordering_policies_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "caller order" in text
    assert "glob" in text


def test_io_doc_p10b_003_csv_export_validity_defaults_and_strict_explicit_value_column_parsing_documented() -> None:
    """ID: IO_DOC_P10B_003_csv_export_validity_defaults_and_strict_explicit_value_column_parsing_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "sequence-size metadata" in text
    assert "value-column" in text


def test_io_doc_p10b_004_reserved_adapter_metadata_key_policy_documented() -> None:
    """ID: IO_DOC_P10B_004_reserved_adapter_metadata_key_policy_documented."""
    text = Path("docs/api/io.md").read_text(encoding="utf-8").lower()
    assert "`tal` metadata key" in text
    assert "schema" in text


def test_arch_io_sidecar_validation_is_centralized_before_column_checks() -> None:
    """CSV sidecar validation normalizes metadata before downstream column reconstruction checks."""
    csv_text = Path("tal/io/csv_io.py").read_text(encoding="utf-8")
    assert "def _normalize_name_container(" in csv_text
    assert "def _normalize_scalar_coords(" in csv_text
    assert "metadata = _require_read_metadata(" in csv_text
    assert "_require_column_set(frame, metadata=metadata, owner=owner)" in csv_text


def test_arch_io_p10b_001_adapter_parsing_logic_does_not_enter_tal_core() -> None:
    """ID: ARCH_IO_P10B_001_adapter_parsing_logic_does_not_enter_tal_core."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "read_csv_logs(" not in text
        assert "read_ros_logs(" not in text
        assert "write_csv_logs(" not in text


def test_arch_io_p10b_002_adapters_reuse_core_schema_finalization_boundaries() -> None:
    """ID: ARCH_IO_P10B_002_adapters_reuse_core_schema_finalization_boundaries."""
    adapter_finalize_text = Path("tal/io/adapter_finalize.py").read_text(encoding="utf-8")
    csv_logs_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    ros_logs_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    assert "set_roles(" in adapter_finalize_text
    assert "set_param_coord(" in adapter_finalize_text
    assert "set_validity(" in adapter_finalize_text
    assert "finalize_loaded_dataset(" in adapter_finalize_text
    assert "finalize_adapter_dataset(" in csv_logs_text
    assert "finalize_adapter_dataset(" in ros_logs_text


def test_arch_io_p10b_003_catalog_bridge_is_optional_and_not_a_required_adapter_dependency() -> None:
    """ID: ARCH_IO_P10B_003_catalog_bridge_is_optional_and_not_a_required_adapter_dependency."""
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    assert "def read_csv_logs_catalog(" in csv_text
    assert "def read_ros_logs_catalog(" in ros_text
    assert "from tal.catalog import Catalog" in csv_text
    assert "from tal.catalog import Catalog" in ros_text
    assert "Catalog(" not in csv_text.split("def read_csv_logs_catalog(", maxsplit=1)[0]
    assert "Catalog(" not in ros_text.split("def read_ros_logs_catalog(", maxsplit=1)[0]


def test_arch_io_p10b_shared_path_and_label_policy_owners_are_centralized() -> None:
    """ID: ARCH_IO_P10B_004_adapter_path_and_label_policy_is_centralized."""
    paths_text = Path("tal/io/adapter_paths.py").read_text(encoding="utf-8")
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    assert "def resolve_ingest_inputs(" in paths_text
    assert "def normalize_export_label_path(" in paths_text
    assert "resolve_ingest_inputs" in csv_text
    assert "resolve_ingest_inputs" in ros_text
    assert "normalize_export_label_path" in csv_text


def test_arch_io_p10b_005_csv_export_size_and_value_parse_policies_are_centralized() -> None:
    """ID: ARCH_IO_P10B_005_csv_export_size_coord_and_value_parse_policies_are_centralized."""
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    assert "def _resolve_effective_export_size_name(" in csv_text
    assert "def _resolve_valid_length(" in csv_text
    assert "def _coerce_numeric_value_column(" in csv_text
    assert "def _append_csv_column(" in csv_text
    assert "_coerce_numeric_value_column(frame, name=name, owner=owner, path=path)" in csv_text
    assert "size_name = _resolve_effective_export_size_name(" in csv_text
    assert "np.isfinite(numeric)" in csv_text
    assert "is_integer()" in csv_text
    assert "_append_csv_column(" in csv_text


def test_arch_io_p10b_006_reserved_metadata_key_validation_is_centralized_in_adapter_metadata_owner() -> None:
    """ID: ARCH_IO_P10B_006_reserved_metadata_key_validation_is_centralized_in_adapter_metadata_owner."""
    metadata_text = Path("tal/io/adapter_metadata.py").read_text(encoding="utf-8")
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    ros_text = Path("tal/io/ros_logs.py").read_text(encoding="utf-8")
    assert "target == \"attrs\" and name == \"tal\"" in metadata_text
    assert "reserved for schema namespace" in metadata_text
    assert "name == \"tal\"" not in csv_text
    assert "name == \"tal\"" not in ros_text


def test_arch_io_p10b_007_csv_export_column_name_normalization_collision_checks_are_centralized() -> None:
    """ID: ARCH_IO_P10B_007_csv_export_column_name_normalization_collision_checks_are_centralized."""
    csv_text = Path("tal/io/csv_logs.py").read_text(encoding="utf-8")
    assert "def _append_csv_column(" in csv_text
    assert csv_text.count("_append_csv_column(") >= 3
    assert "columns[str(name)] =" not in csv_text
    assert "if name in columns" not in csv_text


def test_io_owner_budget_and_schema_write_boundary() -> None:
    """IO modules stay within AGENTS budgets and avoid direct tal-attrs writes."""
    for path in sorted(Path("tal/io").glob("*.py")):
        assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
        for name, length in function_lengths(path).items():
            assert length <= 50, f"{path}:{name} exceeds function budget ({length} > 50)."
        text = path.read_text(encoding="utf-8")
        assert not _has_direct_tal_schema_write(text)
