from __future__ import annotations

from pathlib import Path


def test_arch_time_t1_001_core_time_support_does_not_import_astro() -> None:
    """ID: ARCH_TIME_T1_001_core_time_support_does_not_import_astro."""
    for path in sorted(Path("tal/core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tal.astro" not in text, f"core time support must not import astro in {path}"


def test_arch_time_t1_002_datetime_maps_use_numpy_path_not_numeric_numba() -> None:
    """ID: ARCH_TIME_T1_002_datetime_maps_use_numpy_path_not_numeric_numba."""
    text = Path("tal/core/param_engine/map_build.py").read_text(encoding="utf-8")
    section = text.split("def _apply_param_map_block", 1)[1].split("def build_param_map", 1)[0]
    datetime_branch = section.split('if param_kind == "datetime64":', 1)[1].split("backend = _select_map_normal_backend()", 1)[0]
    assert "datetime_map_block_numpy" in datetime_branch
    assert "_select_map_normal_backend" not in datetime_branch
    assert "PARAM_MAP_BACKEND_NUMBA" not in datetime_branch


def test_arch_time_t1_003_query_and_map_owners_receive_param_kind() -> None:
    """ID: ARCH_TIME_T1_003_query_and_map_owners_receive_param_kind."""
    for path in (
        Path("tal/core/param_ops/index.py"),
        Path("tal/core/param_ops/evaluate.py"),
        Path("tal/core/param_ops/sync_runtime.py"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "param_kind=context.param_kind" in text, f"{path} must thread context.param_kind"
    select = Path("tal/core/param_ops/select.py").read_text(encoding="utf-8")
    assert "param_kind=context.param_kind" in select
    assert "coerce_float_scalar(query.start" not in select


def test_arch_time_t1_004_datetime_local_deltas_owned_by_param_map() -> None:
    """ID: ARCH_TIME_T1_004_datetime_local_deltas_owned_by_param_map."""
    text = Path("tal/core/param_engine/datetime_rows.py").read_text(encoding="utf-8")
    assert "def _local_ns(" in text
    assert "anchor" in text
    assert "float(q - t0) / float(t1 - t0)" in text


def test_arch_time_t1_005_datetime_autogrid_returns_datetime64_with_nat_padding() -> None:
    """ID: ARCH_TIME_T1_005_datetime_autogrid_returns_datetime64_with_nat_padding."""
    autogrid = Path("tal/core/param_ops/sync_autogrid.py").read_text(encoding="utf-8")
    backend = Path("tal/core/param_ops/sync_autogrid_backend.py").read_text(encoding="utf-8")
    assert "join_datetime_rows_batched" in autogrid
    assert '.view("datetime64[ns]")' in autogrid
    assert "_NAT_INT" in backend
    assert "np.full((len(rows), width), _NAT_INT" in backend


def test_arch_time_t1_006_outer_batch_reindex_is_datetime_fill_aware() -> None:
    """ID: ARCH_TIME_T1_006_outer_batch_reindex_is_datetime_fill_aware."""
    text = Path("tal/core/param_ops/sync_runtime.py").read_text(encoding="utf-8")
    assert "def _outer_batch_reindex_fill_values(" in text
    assert 'np.datetime64("NaT", "ns")' in text
    assert 'np.timedelta64("NaT", "ns")' in text
    assert "reindex({dim: labels}, fill_value=np.nan)" not in text
