from __future__ import annotations

import ast
from pathlib import Path

from tal.core.typed_lifecycle import TypedAnalysisObject
from tal.geo import GeodeticPosition, ProjectedPosition

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


def test_arch_geo_g2_001_position_geo_property_uses_lazy_import() -> None:
    """ID: ARCH_GEO_G2_001_position_geo_property_uses_lazy_import."""
    spatial_text = Path("tal/spatial/position.py").read_text(encoding="utf-8")
    accessor_text = Path("tal/geo/accessor.py").read_text(encoding="utf-8")
    assert "tal.geo" not in spatial_text
    assert "def geo(" in accessor_text
    assert "from .accessor import PositionGeoAccessor" in accessor_text


def test_arch_geo_g2_002_geo_accessors_are_thin_delegators() -> None:
    """ID: ARCH_GEO_G2_002_geo_accessors_are_thin_delegators."""
    text = Path("tal/geo/accessor.py").read_text(encoding="utf-8")
    banned = ["apply_ufunc", "merge_schema", "set_frames", "set_position_rep", "set_ecef_metadata", "set_enu_metadata"]
    assert [token for token in banned if token in text] == []
    assert "from .local import position_to_lla" in text
    assert "from .local import ecef_to_enu" in text
    assert "from .local import enu_to_ecef" in text


def test_arch_geo_g2_003_enu_kernels_are_schema_free() -> None:
    """ID: ARCH_GEO_G2_003_enu_kernels_are_schema_free."""
    text = Path("tal/geo/kernels.py").read_text(encoding="utf-8")
    banned = ["xarray", "attrs", "tal.", "merge_schema", "pyproj", "set_frames", "set_position_rep"]
    assert [token for token in banned if token in text] == []


def test_arch_geo_g2_004_no_core_or_frames_geo_imports() -> None:
    """ID: ARCH_GEO_G2_004_no_core_or_frames_geo_imports."""
    for root in (Path("tal/core"), Path("tal/frames")):
        for path in sorted(root.rglob("*.py")):
            imports = _imports(path)
            assert all(not name.startswith("tal.geo") for name in imports)


def test_arch_geo_g3_001_geo_interpolation_reuses_param_owners() -> None:
    """ID: ARCH_GEO_G3_001_geo_interpolation_reuses_param_owners."""
    text = Path("tal/geo/interpolation.py").read_text(encoding="utf-8")
    for token in (
        "resolve_param_runtime_context",
        "normalize_query_grid",
        "build_param_map",
        "gather_along_sequence",
        "finalize_param_output",
    ):
        assert token in text


def test_arch_geo_g3_002_geo_distance_backends_are_schema_free() -> None:
    """ID: ARCH_GEO_G3_002_geo_distance_backends_are_schema_free."""
    text = Path("tal/geo/backends.py").read_text(encoding="utf-8")
    assert "geod_inverse" in text
    assert "geod_interpolate" in text
    banned = ["xarray", "attrs[", "tal.ext", "merge_schema", "set_roles", "set_frames"]
    assert [token for token in banned if token in text] == []


def test_arch_geo_g3_003_geo_methods_are_thin_delegators() -> None:
    """ID: ARCH_GEO_G3_003_geo_methods_are_thin_delegators."""
    geodetic = Path("tal/geo/geodetic.py").read_text(encoding="utf-8")
    temporal = Path("tal/geo/temporal.py").read_text(encoding="utf-8")
    for banned in ("apply_ufunc", "merge_schema", "set_frames", "geod_inverse", "geod_interpolate"):
        assert banned not in geodetic
        assert banned not in temporal
    assert "from .distance import distance_to" in geodetic
    assert "from .interpolation import geodetic_param_at" in temporal
    assert "from .interpolation import geodetic_param_interp_like" in temporal


def test_arch_geo_g3_005_geo_does_not_import_astropy_or_spiceypy() -> None:
    """ID: ARCH_GEO_G3_005_geo_does_not_import_astropy_or_spiceypy."""
    for path in sorted(Path("tal/geo").glob("*.py")):
        imports = _imports(path)
        assert all(name.split(".")[0] not in {"astropy", "spiceypy"} for name in imports)


def test_arch_geo_g4_001_optional_backend_imports_are_local() -> None:
    """ID: ARCH_GEO_G4_001_optional_backend_imports_are_local."""
    offenders = []
    for path in sorted(Path("tal/geo").glob("*.py")):
        if path.name == "backends.py":
            continue
        imports = _imports(path)
        if any(name.split(".")[0] == "pyproj" for name in imports):
            offenders.append(path.as_posix())
    assert offenders == []


def test_arch_geo_g4_002_core_frames_spatial_do_not_depend_on_pyproj() -> None:
    """ID: ARCH_GEO_G4_002_core_frames_spatial_do_not_depend_on_pyproj."""
    for root in (Path("tal/core"), Path("tal/frames"), Path("tal/spatial")):
        for path in sorted(root.rglob("*.py")):
            imports = _imports(path)
            assert all(name.split(".")[0] != "pyproj" for name in imports)


def test_arch_geo_g4_003_crs_backend_isolation() -> None:
    """ID: ARCH_GEO_G4_003_crs_backend_isolation."""
    backend_text = Path("tal/geo/backends.py").read_text(encoding="utf-8")
    transform_path = Path("tal/geo/crs_transform.py")
    transform_text = transform_path.read_text(encoding="utf-8")
    assert "normalize_crs_for_class" in backend_text
    assert "normalize_crs_with_class" in backend_text
    assert "def normalize_crs(" not in backend_text
    assert "def crs_class(" not in backend_text
    assert "transform_crs_xyz" in backend_text
    assert all(name.split(".")[0] != "pyproj" for name in _imports(transform_path))
    assert "normalize_crs_with_class" in transform_text
    assert "crs_class(" not in transform_text
    assert "normalize_crs(" not in transform_text
    assert "xr.apply_ufunc" in transform_text


def test_arch_geo_g4_004_no_generic_crs_position_class() -> None:
    """ID: ARCH_GEO_G4_004_no_generic_crs_position_class."""
    for path in sorted(Path("tal/geo").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "class CRSPosition" not in text
        assert "Position(rep=\"projected\")" not in text
        assert "rep=\"projected\"" not in text
    assert issubclass(ProjectedPosition, TypedAnalysisObject)
    assert "TypedLifecycleSpec(" in Path("tal/geo/projected.py").read_text(encoding="utf-8")


def test_arch_geo_g4_005_geo_does_not_import_astropy_or_spiceypy() -> None:
    """ID: ARCH_GEO_G4_005_geo_does_not_import_astropy_or_spiceypy."""
    for path in sorted(Path("tal/geo").glob("*.py")):
        imports = _imports(path)
        assert all(name.split(".")[0] not in {"astropy", "spiceypy"} for name in imports)
