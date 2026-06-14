from __future__ import annotations

import ast
from pathlib import Path

from tal.astro import TopocentricDirection
from tal.core.typed_lifecycle import TypedAnalysisObject

from ._budget import file_loc, function_lengths


def _imports(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def _assert_agents_budget(path: Path) -> None:
    assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{path}:{name} exceeds function budget: {length} > 50"


def test_arch_astro_a1_001_core_and_frames_do_not_import_astro() -> None:
    """ID: ARCH_ASTRO_A1_001_core_and_frames_do_not_import_astro."""
    for root in (Path("tal/core"), Path("tal/frames")):
        for path in sorted(root.rglob("*.py")):
            imports = _imports(path)
            assert all(not name.startswith("tal.astro") for name in imports)


def test_arch_astro_a1_002_geo_does_not_import_astro() -> None:
    """ID: ARCH_ASTRO_A1_002_geo_does_not_import_astro."""
    for path in sorted(Path("tal/geo").glob("*.py")):
        imports = _imports(path)
        assert all(not name.startswith("tal.astro") for name in imports)


def test_arch_astro_a1_003_optional_backend_imports_are_local() -> None:
    """ID: ARCH_ASTRO_A1_003_optional_backend_imports_are_local."""
    offenders: list[str] = []
    for path in sorted(Path("tal/astro").rglob("*.py")):
        if path.as_posix() == "tal/astro/backends/astropy.py":
            continue
        imports = _imports(path)
        if any(name.split(".")[0] in {"astropy", "spiceypy"} for name in imports):
            offenders.append(path.as_posix())
        text = path.read_text(encoding="utf-8")
        if "import_module(\"astropy\")" in text or "import_module(\"spiceypy\")" in text:
            offenders.append(path.as_posix())
    assert offenders == []


def test_arch_astro_a1_004_astro_orchestration_reuses_core_context_owners() -> None:
    """ID: ARCH_ASTRO_A1_004_astro_orchestration_reuses_core_context_owners."""
    text = Path("tal/astro/orchestration.py").read_text(encoding="utf-8")
    assert "resolve_dataset_context" in text
    assert "resolve_param_runtime_context" in text
    assert "GeodeticPosition.from_lla" in text
    assert "_maybe_resolve_param_runtime" not in text


def test_arch_astro_a1_005_astro_finalize_reuses_core_finalize_owners() -> None:
    """ID: ARCH_ASTRO_A1_005_astro_finalize_reuses_core_finalize_owners."""
    text = Path("tal/astro/finalize.py").read_text(encoding="utf-8")
    assert "CoreSchemaFinalizeSpec" in text
    assert "finalize_with_schema" in text
    assert "observer.sequence_size_coord" in text
    assert "size_name=None" not in text


def test_arch_astro_a1_006_astro_has_no_local_topology_planner_clones() -> None:
    """ID: ARCH_ASTRO_A1_006_astro_has_no_local_topology_planner_clones."""
    banned = ["flatten_param_contexts", "restore_dataset_batch_topology", "BatchFlattenPlan"]
    for path in sorted(Path("tal/astro").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert [token for token in banned if token in text] == []


def test_arch_astro_a1_007_no_tal_v2_imports() -> None:
    """ID: ARCH_ASTRO_A1_007_no_tal_v2_imports."""
    for path in sorted(Path("tal/astro").rglob("*.py")):
        assert "tal_v2" not in path.read_text(encoding="utf-8")


def test_arch_astro_metadata_owner_is_the_only_schema_reader() -> None:
    for path in sorted(Path("tal/astro").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "metadata.py":
            continue
        assert "tal.ext.astro" not in text
        assert "attrs[\"tal\"]" not in text
        assert "attrs['tal']" not in text


def test_arch_astro_has_no_dead_owner_noop_assignments() -> None:
    for path in sorted(Path("tal/astro").rglob("*.py")):
        assert "_ = owner" not in path.read_text(encoding="utf-8")


def test_arch_astro_time_coercion_preflights_raw_lazy_before_numpy() -> None:
    text = Path("tal/astro/orchestration.py").read_text(encoding="utf-8")
    section = text.split("def _datetime64_array", 1)[1].split("def _time_from_array", 1)[0]
    assert "_fail_if_raw_lazy_time(value, owner=owner)" in section
    assert section.index("_fail_if_raw_lazy_time") < section.index("np.asarray(value)")


def test_arch_astro_direction_uses_typed_lifecycle_and_budget() -> None:
    assert issubclass(TopocentricDirection, TypedAnalysisObject)
    assert "TypedLifecycleSpec(" in Path("tal/astro/direction.py").read_text(encoding="utf-8")
    for path in sorted(Path("tal/astro").rglob("*.py")):
        _assert_agents_budget(path)
