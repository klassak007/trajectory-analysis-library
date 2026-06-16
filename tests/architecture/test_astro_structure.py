from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

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


def test_arch_astro_direction_to_vector3_keeps_linalg_import_local() -> None:
    module = ast.parse(Path("tal/astro/direction.py").read_text(encoding="utf-8"))
    top_level_imports: list[str] = []
    for node in module.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.append(node.module)
    assert "tal.linalg" not in top_level_imports
    assert "from tal.linalg import Vector3" in Path("tal/astro/direction.py").read_text(encoding="utf-8")


def test_arch_astro_direction_to_vector3_has_vector3_return_annotation() -> None:
    module = ast.parse(Path("tal/astro/direction.py").read_text(encoding="utf-8"))
    method = next(
        item
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TopocentricDirection"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "to_vector3"
    )
    assert method.returns is not None
    annotation = method.returns.value if isinstance(method.returns, ast.Constant) else ast.unparse(method.returns)
    assert "Vector3" in str(annotation)


def test_arch_astro_direction_to_vector3_has_no_finalize_owner_clone() -> None:
    text = Path("tal/astro/direction.py").read_text(encoding="utf-8")
    assert "CoreSchemaFinalizeSpec" not in text
    assert "finalize_with_schema" not in text


def test_arch_astro_direction_to_vector3_preserves_extension_metadata_deliberately() -> None:
    text = Path("tal/astro/direction.py").read_text(encoding="utf-8")
    section = text.split("def _vector3_dataset", 1)[1].split("class TopocentricDirection", 1)[0]
    assert "merge_schema(" in section
    assert '{"ext": {"astro": None}}' in section
    assert ".to_dataset(" not in section


def test_arch_astro_a2_001_sun_operation_uses_a1_runtime_context() -> None:
    """ID: ARCH_ASTRO_A2_001_sun_operation_uses_a1_runtime_context."""
    text = Path("tal/astro/sun.py").read_text(encoding="utf-8")
    assert "resolve_observer_context" in text
    assert "resolve_time_context" in text
    assert "AstroDirectionRuntimeContext" in text
    assert "resolve_direction_runtime_context(" not in text


def test_arch_astro_a2_002_sun_operation_uses_a1_finalize_owner() -> None:
    """ID: ARCH_ASTRO_A2_002_sun_operation_uses_a1_finalize_owner."""
    text = Path("tal/astro/sun.py").read_text(encoding="utf-8")
    assert "finalize_topocentric_direction" in text
    assert "_from_validated" not in text
    assert "_from_unvalidated" not in text


def test_arch_astro_a2_003_astropy_backend_is_schema_free() -> None:
    """ID: ARCH_ASTRO_A2_003_astropy_backend_is_schema_free."""
    text = Path("tal/astro/backends/astropy.py").read_text(encoding="utf-8")
    assert "xarray" not in text
    assert "attrs" not in text
    assert "tal.ext.astro" not in text


def test_arch_astro_a2_004_no_local_topology_planner_clones() -> None:
    """ID: ARCH_ASTRO_A2_004_no_local_topology_planner_clones."""
    text = Path("tal/astro/sun.py").read_text(encoding="utf-8")
    banned = ["flatten_param_contexts", "restore_dataset_batch_topology", "BatchFlattenPlan"]
    assert [token for token in banned if token in text] == []


def test_arch_astro_a2_005_no_tal_v2_imports() -> None:
    """ID: ARCH_ASTRO_A2_005_no_tal_v2_imports."""
    for path in sorted(Path("tal/astro").rglob("*.py")):
        assert "tal_v2" not in path.read_text(encoding="utf-8")


def test_arch_astro_a2_006_no_top_level_sun_alias_in_a2() -> None:
    """ID: ARCH_ASTRO_A2_006_no_top_level_sun_alias_in_a2."""
    text = Path("tal/astro/__init__.py").read_text(encoding="utf-8")
    assert "direction_to_sun" not in text
    assert ".sun" not in text


def test_arch_astro_a2_007_no_astropy_imports_outside_backend() -> None:
    """ID: ARCH_ASTRO_A2_007_no_astropy_imports_outside_backend."""
    offenders: list[str] = []
    for path in sorted(Path("tal/astro").rglob("*.py")):
        if path.as_posix() == "tal/astro/backends/astropy.py":
            continue
        imports = _imports(path)
        if any(name.split(".")[0] == "astropy" for name in imports):
            offenders.append(path.as_posix())
    assert offenders == []


def test_arch_astro_a2_sun_options_type_hints_are_evaluable() -> None:
    from tal.astro.sun import SpiceSunOptions, SunDirectionOptions

    assert get_type_hints(SunDirectionOptions)["spice"] == SpiceSunOptions | None


def test_arch_astro_a2_lightweight_import_boundary() -> None:
    code = (
        "import sys; import tal, tal.astro; "
        "assert 'tal.astro.sun' not in sys.modules; "
        "assert 'astropy' not in sys.modules; "
        "assert 'spiceypy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
