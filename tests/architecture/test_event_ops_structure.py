from __future__ import annotations

import ast
from pathlib import Path


def _event_ops_files() -> list[Path]:
    return sorted(Path("tal/core/event_ops").glob("*.py"))


def test_event_arch_001_single_owner_condition_eval_modules() -> None:
    """ID: EVENT_ARCH_001_single_owner_condition_eval_modules."""
    text = Path("tal/core/event_ops/evaluate.py").read_text(encoding="utf-8")
    for needle in ["def _eval_compare(", "def _eval_not(", "def _eval_and(", "def _eval_or(", "def evaluate_mask("]:
        assert needle in text
    assert "class EventsAccessor" in Path("tal/core/event_ops/accessor.py").read_text(encoding="utf-8")
    boundary = Path("tal/core/event_ops/boundary.py").read_text(encoding="utf-8")
    pack = Path("tal/core/event_ops/pack.py").read_text(encoding="utf-8")
    assert "def extract_event_boundaries(" in boundary
    assert "def pack_event_table(" in pack


def test_event_arch_002_no_hardcoded_event_trigger_segment_output_core_dims() -> None:
    """ID: EVENT_ARCH_002_no_hardcoded_event_trigger_segment_output_core_dims."""
    forbidden = ['dims=("event"', "dims=('event'", 'dim="event"', "dim='event'", 'dims=("segment"', "dims=('segment'"]
    for path in _event_ops_files():
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"hardcoded event/segment dim allocation found in {path}: {needle}"


def test_event_arch_003_event_ops_use_orchestration_finalize_boundary() -> None:
    """ID: EVENT_ARCH_003_event_ops_use_orchestration_finalize_boundary."""
    for path in _event_ops_files():
        text = path.read_text(encoding="utf-8")
        assert "._finalize_structural(" not in text
        assert ".__class__._from_unvalidated(" not in text
        assert "from ..ao_internal import" not in text


def test_event_arch_004_event_ops_no_local_flatten_restore_plans() -> None:
    """ID: EVENT_ARCH_004_event_ops_no_local_flatten_restore_plans."""
    forbidden = ["flatten_batch_contexts(", "flatten_query_for_plan(", "restore_dataset_batch_dims("]
    for path in _event_ops_files():
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"local flatten/restore planner usage found in {path}: {needle}"


def test_event_arch_005_interval_owner_modules_single_owner() -> None:
    """ID: EVENT_ARCH_005_interval_owner_modules_single_owner."""
    interval_text = Path("tal/core/event_ops/intervals.py").read_text(encoding="utf-8")
    assert "def extract_intervals(" in interval_text
    pack_text = Path("tal/core/event_ops/pack.py").read_text(encoding="utf-8")
    assert "def pack_interval_table(" in pack_text
    accessor = Path("tal/core/event_ops/accessor.py").read_text(encoding="utf-8")
    assert "def intervals(" in accessor


def test_event_arch_006_interval_ops_no_local_boundary_duplication() -> None:
    """ID: EVENT_ARCH_006_interval_ops_no_local_boundary_duplication."""
    text = Path("tal/core/event_ops/intervals.py").read_text(encoding="utf-8")
    forbidden = ["def _transition_rows(", "def _trigger_rows(", "def _row_candidates("]
    for needle in forbidden:
        assert needle not in text, f"boundary helper duplication found in intervals.py: {needle}"


def test_event_arch_007_interval_pack_no_hardcoded_segment_edge_dims() -> None:
    """ID: EVENT_ARCH_007_interval_pack_no_hardcoded_segment_edge_dims."""
    forbidden = ['dims=("segment"', "dims=('segment'", 'dim="segment"', "dim='segment'", 'dims=("edge"', "dims=('edge'", 'dim="edge"', "dim='edge'"]
    for path in _event_ops_files():
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"hardcoded segment/edge dim allocation found in {path}: {needle}"


def test_event_arch_008_event_lane_shape_helpers_single_owner() -> None:
    """ID: EVENT_ARCH_008_event_lane_shape_helpers_single_owner."""
    primitives = Path("tal/core/event_ops/event_primitives.py").read_text(encoding="utf-8")
    assert "def batch_dims(" in primitives
    assert "def lane_count(" in primitives
    assert "def lane_data(" in primitives
    for path in ["tal/core/event_ops/boundary.py", "tal/core/event_ops/intervals.py"]:
        text = Path(path).read_text(encoding="utf-8")
        assert "def _batch_dims(" not in text
        assert "def _lane_count(" not in text
        assert "def _canonical_lane_data(" not in text
        assert "def _boundary_lanes(" not in text


def test_event_arch_009_intervals_no_private_boundary_internal_imports() -> None:
    """ID: EVENT_ARCH_009_intervals_no_private_boundary_internal_imports."""
    text = Path("tal/core/event_ops/intervals.py").read_text(encoding="utf-8")
    assert "_EDGE_ENTER" not in text
    assert "_EDGE_EXIT" not in text
    assert "_EDGE_INVALID" not in text
    assert "_EDGE_TRIGGER" not in text
    assert "_SAMPLE_SENTINEL" not in text


def test_event_arch_010_event_edge_sentinel_constants_single_owner() -> None:
    """ID: EVENT_ARCH_010_event_edge_sentinel_constants_single_owner."""
    constants_path = Path("tal/core/event_ops/_event_constants.py")
    constants = constants_path.read_text(encoding="utf-8")
    primitives = Path("tal/core/event_ops/event_primitives.py").read_text(encoding="utf-8")
    module = ast.parse(constants)
    imports = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert imports == ["__future__", "numpy"]
    assert "xarray" not in constants
    assert "tal." not in constants
    assert "attrs" not in constants
    assert "from ._event_constants import" in primitives
    for needle in [
        "EDGE_INVALID =",
        "EDGE_ENTER =",
        "EDGE_EXIT =",
        "EDGE_TRIGGER =",
        "SAMPLE_SENTINEL =",
    ]:
        assert needle in constants
    for path in _event_ops_files():
        if path == constants_path:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in [
            "EDGE_INVALID =",
            "EDGE_ENTER =",
            "EDGE_EXIT =",
            "EDGE_TRIGGER =",
            "SAMPLE_SENTINEL =",
        ]:
            assert needle not in text, f"duplicate edge/sentinel constant definition found in {path}: {needle}"


def test_event_arch_011_when_owner_module_single_owner() -> None:
    """ID: EVENT_ARCH_011_when_owner_module_single_owner."""
    text = Path("tal/core/event_ops/at_boundaries.py").read_text(encoding="utf-8")
    assert "def evaluate_at_boundaries_condition(" in text
    accessor = Path("tal/core/event_ops/accessor.py").read_text(encoding="utf-8")
    assert "def at_boundaries(" in accessor
    assert 'NotImplementedError("events.at_boundaries' not in accessor


def test_event_arch_012_when_no_local_boundary_crossing_duplication() -> None:
    """ID: EVENT_ARCH_012_when_no_local_boundary_crossing_duplication."""
    text = Path("tal/core/event_ops/at_boundaries.py").read_text(encoding="utf-8")
    forbidden = ["def _transition_rows(", "def _trigger_rows(", "def _row_candidates("]
    for needle in forbidden:
        assert needle not in text, f"boundary-crossing duplication found in when.py: {needle}"


def test_event_arch_013_when_reuses_param_eval_owner_path() -> None:
    """ID: EVENT_ARCH_013_when_reuses_param_eval_owner_path."""
    text = Path("tal/core/event_ops/at_boundaries.py").read_text(encoding="utf-8")
    assert ".param.at(" in text
    for needle in ["build_param_map(", "apply_param_map(", "normalize_query_grid("]:
        assert needle not in text, f"when.py should not own interpolation kernels: {needle}"


def test_event_arch_014_when_boundary_caps_are_owner_derived() -> None:
    """ID: EVENT_ARCH_014_when_boundary_caps_are_owner_derived."""
    text = Path("tal/core/event_ops/at_boundaries.py").read_text(encoding="utf-8")
    assert "def _boundary_extraction_cap(" in text
    assert "def _boundary_event_options(" in text
    assert "max_events=_boundary_extraction_cap(" in text


def test_event_arch_015_during_owner_module_single_owner() -> None:
    """ID: EVENT_ARCH_015_during_owner_module_single_owner."""
    during = Path("tal/core/event_ops/when.py").read_text(encoding="utf-8")
    assert "def evaluate_when_condition(" in during
    accessor = Path("tal/core/event_ops/accessor.py").read_text(encoding="utf-8")
    assert "def when(" in accessor
    assert "def subset(" not in accessor


def test_event_arch_016_during_reuses_mask_evaluate_and_structural_where_path() -> None:
    """ID: EVENT_ARCH_016_during_reuses_mask_evaluate_and_structural_where_path."""
    text = Path("tal/core/event_ops/when.py").read_text(encoding="utf-8")
    assert "resolve_event_eval_context(" in text
    assert "evaluate_mask(" in text
    assert ".where(" in text


def test_event_arch_017_during_no_local_boundary_or_interpolation_kernel_duplication() -> None:
    """ID: EVENT_ARCH_017_during_no_local_boundary_or_interpolation_kernel_duplication."""
    text = Path("tal/core/event_ops/when.py").read_text(encoding="utf-8")
    forbidden = [
        "def _transition_rows(",
        "def _trigger_rows(",
        "def _row_candidates(",
        "build_param_map(",
        "apply_param_map(",
        "normalize_query_grid(",
        ".param.at(",
    ]
    for needle in forbidden:
        assert needle not in text, f"when.py should not duplicate boundary/interpolation kernels: {needle}"


def test_event_arch_018_during_segments_owner_module_single_owner() -> None:
    """ID: EVENT_ARCH_018_during_segments_owner_module_single_owner."""
    text = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    assert "def evaluate_when_segments_layout(" in text
    during = Path("tal/core/event_ops/when.py").read_text(encoding="utf-8")
    assert "if opts.layout == \"segments\":" in during


def test_event_arch_019_during_segments_reuses_interval_and_gather_owners() -> None:
    """ID: EVENT_ARCH_019_during_segments_reuses_interval_and_gather_owners."""
    text = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    assert "extract_intervals(" in text
    assert "gather_dataset_along_sequence(" in text


def test_event_arch_020_during_segments_no_local_crossing_or_interp_kernels() -> None:
    """ID: EVENT_ARCH_020_during_segments_no_local_crossing_or_interp_kernels."""
    text = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    forbidden = [
        "def _transition_rows(",
        "def _trigger_rows(",
        "def _row_candidates(",
        "build_param_map(",
        "apply_param_map(",
        "normalize_query_grid(",
        ".param.at(",
    ]
    for needle in forbidden:
        assert needle not in text, f"when_segments.py should not duplicate kernels: {needle}"


def test_event_arch_021_during_segments_finalize_boundary_owner_only() -> None:
    """ID: EVENT_ARCH_021_during_segments_finalize_boundary_owner_only."""
    text = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    assert "from .finalize import finalize_event_output" in text
    assert "from ..ao_internal import" not in text
    assert "._finalize_structural(" not in text
    assert ".__class__._from_unvalidated(" not in text


def test_event_arch_022_during_selected_mask_single_owner() -> None:
    """ID: EVENT_ARCH_022_during_selected_mask_single_owner."""
    common = Path("tal/core/event_ops/when_common.py").read_text(encoding="utf-8")
    assert "def selected_when_mask(" in common
    during = Path("tal/core/event_ops/when.py").read_text(encoding="utf-8")
    segments = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    assert "def _selected_mask(" not in during
    assert "def _selected_mask(" not in segments
    assert "selected_when_mask(" in during
    assert "selected_when_mask(" in segments


def test_event_arch_023_around_owner_module_single_owner() -> None:
    """ID: EVENT_ARCH_023_around_owner_module_single_owner."""
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    assert "def evaluate_around_windows(" in around
    accessor = Path("tal/core/event_ops/accessor.py").read_text(encoding="utf-8")
    assert "def around(" in accessor
    assert 'NotImplementedError("events.around' not in accessor


def test_event_arch_024_around_reuses_boundary_and_param_eval_owners() -> None:
    """ID: EVENT_ARCH_024_around_reuses_boundary_and_param_eval_owners."""
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    assert "extract_event_boundaries(" in around
    assert "select_event_boundaries(" in around
    assert ".param.at(" in around
    select = Path("tal/core/event_ops/boundary_select.py").read_text(encoding="utf-8")
    assert "def select_event_boundaries(" in select
    for path in ("tal/core/event_ops/at_boundaries.py", "tal/core/event_ops/around.py"):
        text = Path(path).read_text(encoding="utf-8")
        assert "def _select_row_indices(" not in text
        assert "def _row_selected_arrays(" not in text


def test_event_arch_025_around_no_local_crossing_or_interp_kernels() -> None:
    """ID: EVENT_ARCH_025_around_no_local_crossing_or_interp_kernels."""
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    forbidden = [
        "def _transition_rows(",
        "def _trigger_rows(",
        "def _row_candidates(",
        "build_param_map(",
        "apply_param_map(",
        "normalize_query_grid(",
    ]
    for needle in forbidden:
        assert needle not in around, f"around.py should not duplicate boundary/interpolation kernels: {needle}"


def test_event_arch_026_around_finalize_boundary_owner_only() -> None:
    """ID: EVENT_ARCH_026_around_finalize_boundary_owner_only."""
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    assert "from .finalize import finalize_event_output" in around
    assert "from ..ao_internal import" not in around
    assert "._finalize_structural(" not in around
    assert ".__class__._from_unvalidated(" not in around


def test_event_arch_027_around_grid_validation_no_eager_np_asarray_patterns() -> None:
    """ID: EVENT_ARCH_027_around_grid_validation_no_eager_np_asarray_patterns."""
    options = Path("tal/core/event_ops/options.py").read_text(encoding="utf-8")
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    assert "def coerce_around_grid(" in options
    assert "np.asarray(out.data" not in options
    assert "np.asarray(tau.data" not in around
    assert "coerce_around_grid(" in around


def test_event_arch_028_around_public_docs_and_api_entries_present() -> None:
    """ID: EVENT_ARCH_028_around_public_docs_and_api_entries_present."""
    user_guide = Path("docs/user-guide/events.md").read_text(encoding="utf-8")
    api_index = Path("docs/api/index.md").read_text(encoding="utf-8")
    api_events = Path("docs/api/events.md").read_text(encoding="utf-8")
    analysis_api = Path("docs/api/analysis-object.md").read_text(encoding="utf-8")
    assert "ao.events.around(" in user_guide
    assert "events" in api_index
    assert "EventsAccessor.around" in api_events
    assert "tal.AnalysisObject.events" in analysis_api


def test_event_arch_029_around_stacked_owner_module_single_owner() -> None:
    """ID: EVENT_ARCH_029_around_stacked_owner_module_single_owner."""
    around_stacked = Path("tal/core/event_ops/around_stacked.py").read_text(encoding="utf-8")
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    assert "def evaluate_around_stacked_windows(" in around_stacked
    assert 'if opts.layout == "stacked":' in around
    assert "evaluate_around_stacked_windows(" in around


def test_event_arch_030_around_stacked_reuses_segments_owner_no_local_query_kernel() -> None:
    """ID: EVENT_ARCH_030_around_stacked_reuses_segments_owner_no_local_query_kernel."""
    around_stacked = Path("tal/core/event_ops/around_stacked.py").read_text(encoding="utf-8")
    assert "evaluate_around_segments_windows(" in around_stacked
    for needle in ["def _query_times(", ".param.at(", "extract_event_boundaries("]:
        assert needle not in around_stacked, f"around_stacked.py must not own query/crossing kernels: {needle}"


def test_event_arch_031_window_stack_single_owner_no_duplicate_flatten_helpers() -> None:
    """ID: EVENT_ARCH_031_window_stack_single_owner_no_duplicate_flatten_helpers."""
    owner = Path("tal/core/event_ops/window_stack.py").read_text(encoding="utf-8")
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    around_stacked = Path("tal/core/event_ops/around_stacked.py").read_text(encoding="utf-8")
    assert "def stack_event_windows(" in owner
    assert "stack_event_windows(" in around_stacked
    assert "def stack_event_windows(" not in around
    assert ".stack({" not in around_stacked
    assert "reset_index(" not in around_stacked


def test_event_arch_032_around_stacked_finalize_boundary_owner_only() -> None:
    """ID: EVENT_ARCH_032_around_stacked_finalize_boundary_owner_only."""
    text = Path("tal/core/event_ops/around_stacked.py").read_text(encoding="utf-8")
    assert "from .finalize import finalize_event_output" in text
    assert "from ..ao_internal import" not in text
    assert "._finalize_structural(" not in text
    assert ".__class__._from_unvalidated(" not in text


def test_event_arch_033_window_stack_zero_lane_restore_owns_coord_roundtrip() -> None:
    """ID: EVENT_ARCH_033_window_stack_zero_lane_restore_owns_coord_roundtrip."""
    window_stack = Path("tal/core/event_ops/window_stack.py").read_text(encoding="utf-8")
    around_stacked = Path("tal/core/event_ops/around_stacked.py").read_text(encoding="utf-8")
    assert "def _captured_zero_dim_coords(" in window_stack
    assert "captured_coords" in window_stack
    assert "assign_coords(" in window_stack
    assert "_captured_zero_dim_coords" not in around_stacked


def test_event_doc_002_around_stacked_user_guide_and_api_entries_present() -> None:
    """ID: EVENT_DOC_002_around_stacked_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/events.md").read_text(encoding="utf-8")
    api_events = Path("docs/api/events.md").read_text(encoding="utf-8")
    assert "AroundOptions" in user_guide
    assert "around" in api_events


def test_event_arch_039_event_finalize_owner_module_single_owner() -> None:
    """ID: EVENT_ARCH_039_event_finalize_owner_module_single_owner."""
    text = Path("tal/core/event_ops/finalize.py").read_text(encoding="utf-8")
    assert "def finalize_event_output(" in text
    assert "set_left_packed_validity_or_prune_from_size_coord(" in text


def test_event_arch_040_event_layout_owners_reuse_event_finalize_owner() -> None:
    """ID: EVENT_ARCH_040_event_layout_owners_reuse_event_finalize_owner."""
    for rel in [
        "tal/core/event_ops/around.py",
        "tal/core/event_ops/around_stacked.py",
        "tal/core/event_ops/when_stream.py",
        "tal/core/event_ops/when_segments.py",
    ]:
        text = Path(rel).read_text(encoding="utf-8")
        assert "finalize_event_output(" in text, rel


def test_event_arch_041_event_layout_owners_remove_local_finalize_clusters() -> None:
    """ID: EVENT_ARCH_041_event_layout_owners_remove_local_finalize_clusters."""
    around = Path("tal/core/event_ops/around.py").read_text(encoding="utf-8")
    around_stacked = Path("tal/core/event_ops/around_stacked.py").read_text(encoding="utf-8")
    stream = Path("tal/core/event_ops/when_stream.py").read_text(encoding="utf-8")
    segments = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    assert "def _finalize_around_output(" not in around
    assert "def _finalize_stacked_output(" not in around_stacked
    assert "def _finalize_stream_output(" not in stream
    assert "def _finalize_output_schema(" not in segments


def test_event_arch_042_event_validity_helper_owner_consolidated_into_core_validity_finalize() -> None:
    """ID: EVENT_ARCH_042_event_validity_helper_owner_consolidated_into_core_validity_finalize."""
    assert not Path("tal/core/event_ops/validity.py").exists()
    text = Path("tal/core/validity_finalize.py").read_text(encoding="utf-8")
    assert "def set_left_packed_validity_or_prune_from_size_coord(" in text


def test_event_arch_034_during_stream_owner_module_single_owner() -> None:
    """ID: EVENT_ARCH_034_during_stream_owner_module_single_owner."""
    owner = Path("tal/core/event_ops/when_stream.py").read_text(encoding="utf-8")
    during = Path("tal/core/event_ops/when.py").read_text(encoding="utf-8")
    assert "def evaluate_when_stream_layout(" in owner
    assert 'if opts.layout == "stream":' in during
    assert "evaluate_when_stream_layout(" in during


def test_event_arch_035_during_stream_reuses_segments_and_stack_owners() -> None:
    """ID: EVENT_ARCH_035_during_stream_reuses_segments_and_stack_owners."""
    owner = Path("tal/core/event_ops/when_stream.py").read_text(encoding="utf-8")
    assert "evaluate_when_segments_layout(" in owner
    assert "stack_segment_stream(" in owner
    assert "gather_dataset_along_sequence(" in owner
    assert ".stack({" not in owner
    assert "reset_index(" not in owner


def test_event_arch_036_during_stream_no_local_crossing_or_interp_kernels() -> None:
    """ID: EVENT_ARCH_036_during_stream_no_local_crossing_or_interp_kernels."""
    text = Path("tal/core/event_ops/when_stream.py").read_text(encoding="utf-8")
    forbidden = [
        "def _transition_rows(",
        "def _trigger_rows(",
        "def _row_candidates(",
        "build_param_map(",
        "apply_param_map(",
        "normalize_query_grid(",
        ".param.at(",
    ]
    for needle in forbidden:
        assert needle not in text, f"when_stream.py should not duplicate crossing/interpolation kernels: {needle}"


def test_event_arch_037_during_stream_finalize_boundary_owner_only() -> None:
    """ID: EVENT_ARCH_037_during_stream_finalize_boundary_owner_only."""
    text = Path("tal/core/event_ops/when_stream.py").read_text(encoding="utf-8")
    assert "from .finalize import finalize_event_output" in text
    assert "from ..ao_internal import" not in text
    assert "._finalize_structural(" not in text
    assert ".__class__._from_unvalidated(" not in text


def test_event_doc_001_conditions_events_user_guide_clock_validity_notes() -> None:
    """ID: EVENT_DOC_001_conditions_events_user_guide_clock_validity_notes."""
    user_guide = Path("docs/user-guide/events.md").read_text(encoding="utf-8")
    assert "context clock" in user_guide
    assert "Validity metadata" in user_guide or "validity metadata" in user_guide


def test_event_doc_003_during_stream_user_guide_and_api_entries_present() -> None:
    """ID: EVENT_DOC_003_during_stream_user_guide_and_api_entries_present."""
    user_guide = Path("docs/user-guide/events.md").read_text(encoding="utf-8")
    api_events = Path("docs/api/events.md").read_text(encoding="utf-8")
    assert "WhenOptions" in user_guide
    assert "EventsAccessor.when" in api_events
    assert "layout=\"stream\" remains planned" not in user_guide


def test_event_arch_038_during_on_empty_enforcement_single_owner() -> None:
    """ID: EVENT_ARCH_038_during_on_empty_enforcement_single_owner."""
    common = Path("tal/core/event_ops/when_common.py").read_text(encoding="utf-8")
    during = Path("tal/core/event_ops/when.py").read_text(encoding="utf-8")
    segments = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    stream = Path("tal/core/event_ops/when_stream.py").read_text(encoding="utf-8")
    assert "def enforce_when_on_empty(" in common
    assert "def _enforce_on_empty(" not in during
    assert "def _enforce_on_empty(" not in segments
    assert "enforce_when_on_empty(" in during
    assert "enforce_when_on_empty(" in segments
    assert "enforce_when_on_empty(" in stream


def test_event_arch_043_boundary_select_bounded_contains_no_vectorize_true() -> None:
    """ID: EVENT_ARCH_043_boundary_select_bounded_contains_no_vectorize_true."""
    text = Path("tal/core/event_ops/boundary_select.py").read_text(encoding="utf-8")
    section = text.split("def _select_boundaries_bounded(", 1)[1].split("def _preflight_selection(", 1)[0]
    assert "vectorize=True" not in section
    assert "vectorize=False" in section


def test_event_arch_044_during_stream_bounded_contains_no_vectorize_true() -> None:
    """ID: EVENT_ARCH_044_during_stream_bounded_contains_no_vectorize_true."""
    text = Path("tal/core/event_ops/when_stream.py").read_text(encoding="utf-8")
    section = text.split("def _bounded_stream_indexer(", 1)[1].split("def _stream_indexer(", 1)[0]
    assert "vectorize=True" not in section
    assert "vectorize=False" in section


def test_event_arch_045_during_segments_bounded_contains_no_vectorize_true() -> None:
    """ID: EVENT_ARCH_045_during_segments_bounded_contains_no_vectorize_true."""
    text = Path("tal/core/event_ops/when_segments.py").read_text(encoding="utf-8")
    section = text.split("def _build_orig_index_bounded(", 1)[1].split("def _build_orig_index(", 1)[0]
    assert "vectorize=True" not in section
    assert "vectorize=False" in section


def test_event_arch_046_bounded_event_stopgaps_route_through_backend_owner() -> None:
    """ID: EVENT_ARCH_046_bounded_event_stopgaps_route_through_backend_owner."""
    boundary = Path("tal/core/event_ops/boundary.py").read_text(encoding="utf-8")
    intervals = Path("tal/core/event_ops/intervals.py").read_text(encoding="utf-8")
    backends = Path("tal/core/event_ops/backends.py").read_text(encoding="utf-8")
    block_prep = Path("tal/core/event_ops/block_prep.py").read_text(encoding="utf-8")
    assert "boundary_bounded_block_backend" in boundary
    assert "intervals_bounded_block_backend" in intervals
    assert "EVENT_BOUNDARY_BACKEND_NUMPY_BLOCK" in backends
    assert "EVENT_INTERVALS_BACKEND_NUMPY_BLOCK" in backends
    assert "def boundary_bounded_block_backend(" in backends
    assert "def intervals_bounded_block_backend(" in backends
    assert "prepare_boundary_block_rows(" in block_prep
    assert "prepare_intervals_block_rows(" in block_prep
    assert "boundary_bounded_row_backend" not in backends
    assert "intervals_bounded_row_backend" not in backends


def test_event_arch_047_boundary_bounded_numba_backend_owner_routed() -> None:
    """ID: EVENT_ARCH_047_boundary_bounded_numba_backend_owner_routed."""
    backends = Path("tal/core/event_ops/backends.py").read_text(encoding="utf-8")
    assert 'EVENT_BOUNDARY_BACKEND_NUMBA = "numba"' in backends
    assert "def boundary_bounded_block_backend(" in backends
    assert "from .numba_backends import boundary_bounded_block_numba" in backends


def test_event_arch_048_intervals_bounded_numba_backend_owner_routed() -> None:
    """ID: EVENT_ARCH_048_intervals_bounded_numba_backend_owner_routed."""
    backends = Path("tal/core/event_ops/backends.py").read_text(encoding="utf-8")
    assert 'EVENT_INTERVALS_BACKEND_NUMBA = "numba"' in backends
    assert "def intervals_bounded_block_backend(" in backends
    assert "from .numba_backends import intervals_bounded_block_numba" in backends


def test_event_arch_049_bounded_event_numba_paths_are_blockwise_vectorize_false() -> None:
    """ID: EVENT_ARCH_049_bounded_event_numba_paths_are_blockwise_vectorize_false."""
    numba_backends = Path("tal/core/event_ops/numba_backends.py").read_text(encoding="utf-8")
    assert "prepare_boundary_block_rows" in numba_backends
    assert "prepare_intervals_block_rows" in numba_backends
    assert "xr.apply_ufunc" not in numba_backends
    assert "vectorize=True" not in numba_backends


def test_event_arch_050_baseline_event_stopgaps_remain_explicit_until_f2c() -> None:
    """ID: EVENT_ARCH_050_baseline_event_stopgaps_remain_explicit_until_f2c."""
    boundary = Path("tal/core/event_ops/boundary.py").read_text(encoding="utf-8")
    intervals = Path("tal/core/event_ops/intervals.py").read_text(encoding="utf-8")
    contract = Path("contracts/114-numba-default-baseline-migration-slice-f2c.md").read_text(encoding="utf-8")
    assert "Decision: migrated" in contract.split("### event_boundary", 1)[1].split("\n### ", 1)[0]
    assert "Decision: migrated" in contract.split("### event_intervals", 1)[1].split("\n### ", 1)[0]
    assert "EVENT_BOUNDARY_BACKEND_NUMPY_ROW" not in boundary
    assert "EVENT_INTERVALS_BACKEND_NUMPY_ROW" not in intervals
    assert "boundary_bounded_row_backend" not in boundary
    assert "intervals_bounded_row_backend" not in intervals


def test_event_arch_051_bounded_event_normal_paths_are_f2_stopgap_free_if_closed() -> None:
    """ID: EVENT_ARCH_051_bounded_event_normal_paths_are_f2_stopgap_free_if_closed."""
    contract_083 = Path("contracts/083-compiled-kernel-backend-followon-phase-f2.md").read_text(encoding="utf-8")
    contract_114 = Path("contracts/114-numba-default-baseline-migration-slice-f2c.md").read_text(encoding="utf-8")
    boundary = Path("tal/core/event_ops/boundary.py").read_text(encoding="utf-8")
    intervals = Path("tal/core/event_ops/intervals.py").read_text(encoding="utf-8")
    boundary_section = boundary.split("def _extract_bounded(", 1)[1]
    intervals_section = intervals.split("def _extract_bounded(", 1)[1]
    assert "Status: Draft" in contract_083
    assert "param/event targets closed; linalg remains open" in contract_083
    assert "### event_boundary" in contract_114
    assert "### event_intervals" in contract_114
    assert "Decision: migrated" in contract_114.split("### event_boundary", 1)[1].split("\n### ", 1)[0]
    assert "Decision: migrated" in contract_114.split("### event_intervals", 1)[1].split("\n### ", 1)[0]
    assert '"backend": _select_boundary_normal_backend()' in boundary_section
    assert '"backend": _select_intervals_normal_backend()' in intervals_section
    assert "vectorize=False" in boundary_section
    assert "vectorize=False" in intervals_section
    assert "vectorize=True" not in boundary_section
    assert "vectorize=True" not in intervals_section
    assert "boundary_bounded_row_backend" not in boundary
    assert "intervals_bounded_row_backend" not in intervals


def test_event_arch_053_boundary_normal_path_vectorize_true_removed() -> None:
    """ID: EVENT_ARCH_053_boundary_normal_path_vectorize_true_removed."""
    text = Path("tal/core/event_ops/boundary.py").read_text(encoding="utf-8")
    section = text.split("def _extract_bounded(", 1)[1]
    assert "boundary_bounded_block_backend" in section
    assert "vectorize=False" in section
    assert "vectorize=True" not in section


def test_event_arch_054_intervals_normal_path_vectorize_true_removed() -> None:
    """ID: EVENT_ARCH_054_intervals_normal_path_vectorize_true_removed."""
    text = Path("tal/core/event_ops/intervals.py").read_text(encoding="utf-8")
    section = text.split("def _extract_bounded(", 1)[1]
    assert "intervals_bounded_block_backend" in section
    assert "vectorize=False" in section
    assert "vectorize=True" not in section


def test_event_arch_052_bounded_event_numba_helper_parameter_budget() -> None:
    """ID: EVENT_ARCH_052_bounded_event_numba_helper_parameter_budget."""
    module = ast.parse(Path("tal/core/event_ops/numba_backends.py").read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef):
            continue
        params = len(node.args.args) + len(node.args.kwonlyargs)
        assert params <= 10, f"numba_backends.{node.name} exceeds parameter budget ({params} > 10)"
