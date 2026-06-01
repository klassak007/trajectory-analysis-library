from __future__ import annotations

import ast
import inspect
from pathlib import Path
import tomllib

from tal.core.param_engine.map_build import build_param_bounds_map, build_param_map


def test_numba_opt_001_numba_extra_is_optional_only() -> None:
    """ID: NUMBA_OPT_001_numba_extra_is_optional_only."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]
    assert optional["numba"] == ["numba>=0.65.1"]
    assert all(not dep.startswith("numba") for dep in deps)
    assert all("tal[numba]" not in dep for dep in optional["test"])
    assert all("tal[numba]" not in dep for dep in optional["dev"])
    assert all("tal[numba]" not in dep for dep in optional["full"])


def test_numba_arch_001_no_unguarded_numba_imports_in_core_import_path() -> None:
    """ID: NUMBA_ARCH_001_no_unguarded_numba_imports_in_core_import_path."""
    offenders = []
    for path in sorted(Path("tal").rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                offenders.extend((path.as_posix(), alias.name) for alias in node.names if alias.name == "numba")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "numba":
                offenders.append((path.as_posix(), node.module))
    assert offenders == []


def test_numba_arch_002_optional_import_helper_has_no_domain_imports() -> None:
    """ID: NUMBA_ARCH_002_optional_import_helper_has_no_domain_imports."""
    module = ast.parse(Path("tal/utils/numba_support.py").read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert imports == ["__future__", "importlib"]
    text = Path("tal/utils/numba_support.py").read_text(encoding="utf-8")
    assert "tal.core" not in text
    assert "tal.spatial" not in text
    assert "tal.linalg" not in text
    assert "import numba" not in text


def test_numba_arch_003_numba_kernels_are_schema_free() -> None:
    """ID: NUMBA_ARCH_003_numba_kernels_are_schema_free."""
    banned = [
        "import xarray",
        "xr.",
        'attrs["tal"]',
        "attrs['tal']",
        "tal_v2",
        "set_roles(",
        "set_param_coord(",
        "set_validity(",
        "ParamMap",
        "ParamBoundsMap",
    ]
    for path in [
        Path("tal/core/param_engine/numba_backends.py"),
        Path("tal/core/event_ops/numba_backends.py"),
        Path("tal/linalg/ops/numba_backends.py"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert [token for token in banned if token in text] == []


def test_numba_arch_004_numba_backend_dispatch_keeps_public_signatures_stable() -> None:
    """ID: NUMBA_ARCH_004_numba_backend_dispatch_keeps_public_signatures_stable."""
    map_params = inspect.signature(build_param_map).parameters
    bounds_params = inspect.signature(build_param_bounds_map).parameters
    assert tuple(map_params) == ("param", "query", "sequence_dim", "query_dim", "valid_mask", "options")
    assert tuple(bounds_params) == ("param", "start", "stop", "sequence_dim", "valid_mask")
    assert all(param.kind is inspect.Parameter.KEYWORD_ONLY for param in map_params.values())
    assert all(param.kind is inspect.Parameter.KEYWORD_ONLY for param in bounds_params.values())


def test_numba_arch_005_numba_expected_failures_translate_through_wrappers() -> None:
    """ID: NUMBA_ARCH_005_numba_expected_failures_translate_through_wrappers."""
    helper = Path("tal/utils/numba_support.py").read_text(encoding="utf-8")
    param_text = Path("tal/core/param_engine/numba_backends.py").read_text(encoding="utf-8")
    event_text = Path("tal/core/event_ops/numba_backends.py").read_text(encoding="utf-8")
    linalg_text = Path("tal/linalg/ops/numba_backends.py").read_text(encoding="utf-8")
    assert "Install with 'tal[numba]'" in helper
    assert "def _raise_map_status(" in param_text
    assert "def _raise_bounds_status(" in param_text
    assert "_DUPLICATE_BRACKET_ERROR" in param_text
    assert "_MAP_MONOTONIC_ERROR" in param_text
    assert "_BOUNDS_MONOTONIC_ERROR" in param_text
    assert "extracted event boundaries include non-finite clock values" in event_text
    assert "require_numba(owner)" in linalg_text


def test_param_arch_041_param_map_numba_backend_owner_routed() -> None:
    """ID: PARAM_ARCH_041_param_map_numba_backend_owner_routed."""
    text = Path("tal/core/param_engine/backends.py").read_text(encoding="utf-8")
    assert 'PARAM_MAP_BACKEND_NUMBA = "numba"' in text
    assert "def map_block_backend(" in text
    assert "from .numba_backends import map_block_numba" in text


def test_param_arch_042_param_bounds_numba_backend_owner_routed() -> None:
    """ID: PARAM_ARCH_042_param_bounds_numba_backend_owner_routed."""
    text = Path("tal/core/param_engine/backends.py").read_text(encoding="utf-8")
    assert 'PARAM_BOUNDS_BACKEND_NUMBA = "numba"' in text
    assert "def bounds_block_backend(" in text
    assert "from .numba_backends import bounds_block_numba" in text


def test_param_arch_043_param_numba_paths_are_blockwise_vectorize_false() -> None:
    """ID: PARAM_ARCH_043_param_numba_paths_are_blockwise_vectorize_false."""
    map_build = Path("tal/core/param_engine/map_build.py").read_text(encoding="utf-8")
    numba_backends = Path("tal/core/param_engine/numba_backends.py").read_text(encoding="utf-8")
    assert "_build_param_map_numba" not in map_build
    assert "_build_param_bounds_map_numba" not in map_build
    assert "map_block_backend" not in map_build
    assert "bounds_block_backend" not in map_build
    assert "vectorize=True" not in numba_backends


def test_param_arch_044_baseline_param_stopgaps_remain_explicit_until_f2c() -> None:
    """ID: PARAM_ARCH_044_baseline_param_stopgaps_remain_explicit_until_f2c."""
    text = Path("tal/core/param_engine/map_build.py").read_text(encoding="utf-8")
    assert '"backend": PARAM_MAP_BACKEND_NUMPY_ROW' in text
    assert '"backend": PARAM_BOUNDS_BACKEND_NUMPY_ROW' in text
    assert "PARAM_MAP_BACKEND_NUMBA" not in text
    assert "PARAM_BOUNDS_BACKEND_NUMBA" not in text
