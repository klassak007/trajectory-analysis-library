from __future__ import annotations

import ast
from pathlib import Path

from tal.core.typed_lifecycle import TypedAnalysisObject
from tal.geo import GeodeticPosition

from ._budget import file_loc, function_lengths


def _assert_agents_budget(path: Path) -> None:
    assert file_loc(path=path) <= 600, f"{path} exceeds file budget."
    for name, length in function_lengths(path).items():
        assert length <= 50, f"{path}:{name} exceeds function budget: {length} > 50"


def _imports(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def _literal_dynamic_imports(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name not in {"import_module", "__import__"}:
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out.append(value.value)
    return out


def test_arch_geo_g1_001_geo_package_has_no_tal_v2_imports() -> None:
    """ID: ARCH_GEO_G1_001_geo_package_has_no_tal_v2_imports."""
    for path in sorted(Path("tal/geo").glob("*.py")):
        assert "tal_v2" not in path.read_text(encoding="utf-8")


def test_arch_geo_g1_002_core_and_frames_do_not_import_geo() -> None:
    """ID: ARCH_GEO_G1_002_core_and_frames_do_not_import_geo."""
    for root in (Path("tal/core"), Path("tal/frames")):
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            assert "tal.geo" not in text
            assert "from ..geo" not in text
            assert "from .geo" not in text


def test_arch_geo_g1_003_geo_backend_adapters_are_schema_free() -> None:
    """ID: ARCH_GEO_G1_003_geo_backend_adapters_are_schema_free."""
    text = Path("tal/geo/backends.py").read_text(encoding="utf-8")
    banned = ["xarray", "attrs[", "tal.ext", "merge_schema", "set_roles", "set_frames"]
    assert [token for token in banned if token in text] == []


def test_arch_geo_g1_004_geodetic_position_uses_typed_lifecycle() -> None:
    """ID: ARCH_GEO_G1_004_geodetic_position_uses_typed_lifecycle."""
    assert issubclass(GeodeticPosition, TypedAnalysisObject)
    assert "TypedLifecycleSpec(" in Path("tal/geo/geodetic.py").read_text(encoding="utf-8")


def test_arch_geo_g1_005_geo_modules_stay_within_budget() -> None:
    """ID: ARCH_GEO_G1_005_geo_modules_stay_within_budget."""
    for path in sorted(Path("tal/geo").glob("*.py")):
        _assert_agents_budget(path)


def test_arch_geo_g1_006_optional_pyproj_imports_are_backend_local() -> None:
    """ID: ARCH_GEO_G1_006_optional_pyproj_imports_are_backend_local."""
    offenders = []
    for path in sorted(Path("tal/geo").glob("*.py")):
        if path.name == "backends.py":
            continue
        imports = _imports(path)
        if any(name.split(".")[0] == "pyproj" for name in imports):
            offenders.append(path.as_posix())
        if any(name.split(".")[0] == "pyproj" for name in _literal_dynamic_imports(path)):
            offenders.append(path.as_posix())
    assert offenders == []


def test_arch_geo_g1_007_geo_does_not_import_astropy_or_spiceypy() -> None:
    """ID: ARCH_GEO_G1_007_geo_does_not_import_astropy_or_spiceypy."""
    for path in sorted(Path("tal/geo").glob("*.py")):
        imports = _imports(path)
        assert all(name.split(".")[0] not in {"astropy", "spiceypy"} for name in imports)
