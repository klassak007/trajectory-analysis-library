from __future__ import annotations

import ast
import inspect
from pathlib import Path
import tomllib

from tal.core.param_engine.map_build import build_param_bounds_map, build_param_map


def _assert_no_direct_numba_import(text: str) -> None:
    module = ast.parse(text)
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "numba" for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "numba"


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
    _assert_no_direct_numba_import(text)


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
        Path("tal/spatial/kernels/rotation_interp_numba_backends.py"),
        Path("tal/spatial/kernels/kinematics_temporal_numba_backends.py"),
        Path("tal/spatial/kernels/kinematics_smoothing_numba_backends.py"),
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
    spatial_text = Path("tal/spatial/kernels/rotation_interp_numba_backends.py").read_text(encoding="utf-8")
    assert "Install with 'tal[numba]'" in helper
    assert "def _raise_map_status(" in param_text
    assert "def _raise_bounds_status(" in param_text
    assert "_DUPLICATE_BRACKET_ERROR" in param_text
    assert "_MAP_MONOTONIC_ERROR" in param_text
    assert "_BOUNDS_MONOTONIC_ERROR" in param_text
    assert "extracted event boundaries include non-finite clock values" in event_text
    assert "require_numba(owner)" in linalg_text
    assert "def _raise_slerp_status(" in spatial_text
    assert "finite alpha values must be within [0, 1]" in spatial_text


def test_numba_arch_006_shared_block_rows_helper_is_schema_free() -> None:
    """ID: NUMBA_ARCH_006_shared_block_rows_helper_is_schema_free."""
    path = Path("tal/utils/block_rows.py")
    module = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert imports == ["__future__", "dataclasses", "numpy"]
    text = path.read_text(encoding="utf-8")
    assert "xarray" not in text
    assert "tal.core" not in text
    assert "tal.linalg" not in text
    assert "tal.spatial" not in text
    assert "tal_v2" not in text
    _assert_no_direct_numba_import(text)


def test_numba_arch_007_shared_block_rows_helper_preserves_owner_boundaries() -> None:
    """ID: NUMBA_ARCH_007_shared_block_rows_helper_preserves_owner_boundaries."""
    text = Path("tal/utils/block_rows.py").read_text(encoding="utf-8")
    banned = ["param", "event", "lstsq", "quaternion", "PARAM_", "EVENT_", "LSTSQ_", "SPATIAL_"]
    assert [token for token in banned if token in text] == []


def test_numba_arch_008_numba_compile_policy_helper_is_minimal() -> None:
    """ID: NUMBA_ARCH_008_numba_compile_policy_helper_is_minimal."""
    text = Path("tal/utils/numba_support.py").read_text(encoding="utf-8")
    assert "def njit_kernel(" in text
    assert "cache=True" in text
    assert "fastmath=False" in text
    assert "parallel=True" not in text
    assert "**kwargs" not in text


def test_numba_arch_009_numba_benchmark_protocol_is_shared() -> None:
    """ID: NUMBA_ARCH_009_numba_benchmark_protocol_is_shared."""
    helper = Path("benchmarks/_numba_bench.py").read_text(encoding="utf-8")
    assert "NUMBA_CACHE_DIR" in helper
    assert "TemporaryDirectory" in helper
    for path in [
        Path("benchmarks/bench_param_numba_backends.py"),
        Path("benchmarks/bench_event_numba_backends.py"),
        Path("benchmarks/bench_linalg_lstsq_numba_backends.py"),
        Path("benchmarks/bench_spatial_slerp_numba_backends.py"),
        Path("benchmarks/bench_spatial_kinematics_scan_numba_backends.py"),
        Path("benchmarks/bench_spatial_kinematics_stencil_numba_backends.py"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "from _numba_bench import" in text
        assert "NUMBA_CACHE_DIR" not in text
        assert "TemporaryDirectory" not in text


def test_numba_arch_010_numba_scan_helper_is_schema_free() -> None:
    """ID: NUMBA_ARCH_010_numba_scan_helper_is_schema_free."""
    path = Path("tal/utils/numba_scan.py")
    module = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert imports == ["__future__", "dataclasses", "typing", "numpy", "tal.utils.block_rows"]
    text = path.read_text(encoding="utf-8")
    assert "xarray" not in text
    assert "tal.core" not in text
    assert "tal.linalg" not in text
    assert "tal.spatial" not in text
    assert "tal_v2" not in text
    _assert_no_direct_numba_import(text)


def test_numba_arch_011_numba_scan_helper_preserves_owner_boundaries() -> None:
    """ID: NUMBA_ARCH_011_numba_scan_helper_preserves_owner_boundaries."""
    text = Path("tal/utils/numba_scan.py").read_text(encoding="utf-8")
    banned = [
        "kalman",
        "quaternion",
        "event",
        "frame",
        "joint",
        "lstsq",
        "PARAM_",
        "EVENT_",
        "LSTSQ_",
        "SPATIAL_",
        "KINEMATICS_",
    ]
    assert [token for token in banned if token in text] == []


def test_numba_arch_012_numba_scan_substrate_has_no_generic_callback_executor() -> None:
    """ID: NUMBA_ARCH_012_numba_scan_substrate_has_no_generic_callback_executor."""
    text = Path("tal/utils/numba_scan.py").read_text(encoding="utf-8")
    assert "Callable" not in text
    assert "callback" not in text.lower()
    assert "def scan(" not in text
    assert "njit" not in text


def test_numba_arch_013_ordered_axes_are_not_inferred_from_batch_dims() -> None:
    """ID: NUMBA_ARCH_013_ordered_axes_are_not_inferred_from_batch_dims."""
    text = Path("tal/utils/numba_scan.py").read_text(encoding="utf-8")
    assert "ordered_axes:" in text
    assert "_validate_axes(axis_tuple" in text
    assert "spec.ordered_ndim != len(axes)" in text
    assert "axis.name ==" not in text
    assert "batch" not in text


def test_numba_arch_020_public_numba_utility_surface_import_boundaries() -> None:
    """ID: NUMBA_ARCH_020_public_numba_utility_surface_import_boundaries."""
    _assert_no_direct_numba_import("from tal.utils import numba as tal_numba")
    try:
        _assert_no_direct_numba_import("import numba.core")
    except AssertionError:
        pass
    else:
        raise AssertionError("dotted numba import was not rejected")

    facade = Path("tal/utils/numba/__init__.py")
    stencil = Path("tal/utils/numba_stencil.py")
    assert facade.exists()
    assert stencil.exists()

    facade_text = facade.read_text(encoding="utf-8")
    assert "from tal.utils.block_rows import" in facade_text
    assert "from tal.utils.numba_scan import" in facade_text
    assert "from tal.utils.numba_stencil import" in facade_text
    assert "from tal.utils.numba_support import" in facade_text
    assert "benchmarks" not in facade_text

    for path in [facade, stencil]:
        text = path.read_text(encoding="utf-8")
        _assert_no_direct_numba_import(text)
        assert "import xarray" not in text
        assert "tal.core" not in text
        assert "tal.spatial" not in text
        assert "tal.linalg" not in text
        assert "tal_v2" not in text


def test_numba_arch_021_public_numba_utility_surface_has_no_domain_policy() -> None:
    """ID: NUMBA_ARCH_021_public_numba_utility_surface_has_no_domain_policy."""
    stencil = Path("tal/utils/numba_stencil.py").read_text(encoding="utf-8")
    facade = Path("tal/utils/numba/__init__.py").read_text(encoding="utf-8")
    banned = [
        "valid_mask",
        "monotonic",
        "gaussian",
        "smooth",
        "interp",
        "quaternion",
        "event",
        "duplicate",
        "ParamMap",
        "ParamBoundsMap",
        "KINEMATICS_",
        "SPATIAL_",
        "EVENT_",
        "PARAM_",
        "LSTSQ_",
    ]
    assert [token for token in banned if token in stencil] == []
    assert "benchmarks" not in facade
    assert "_numba_bench" not in facade


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
