from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tal.io import (
    AdapterMetadataPromotionOptions,
    CsvIngestOptions,
    read_csv_logs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_io_core_p10b_013_non_mapping_ingest_label_derivation_is_deterministic(tmp_path: Path) -> None:
    """ID: IO_CORE_P10B_013_non_mapping_ingest_label_derivation_is_deterministic."""
    path = tmp_path / "run_01.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0}, {"time": 1.0, "value": 2.0}])
    ao = read_csv_logs([str(path)], opts=CsvIngestOptions(time_col="time"))
    labels = ao.unsafe_data.coords["trial"].values.tolist()
    assert labels == ["run_01"]


def test_io_core_p10b_014_input_path_expansion_and_batch_ordering_are_deterministic(tmp_path: Path) -> None:
    """ID: IO_CORE_P10B_014_input_path_expansion_and_batch_ordering_are_deterministic."""
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    _write_csv(a_path, [{"time": 0.0, "value": 1.0}])
    _write_csv(b_path, [{"time": 0.0, "value": 2.0}])

    ao_seq = read_csv_logs(
        [str(b_path), str(a_path)],
        opts=CsvIngestOptions(time_col="time"),
    )
    assert ao_seq.unsafe_data.coords["trial"].values.tolist() == ["b", "a"]

    ao_glob = read_csv_logs(str(tmp_path / "*.csv"), opts=CsvIngestOptions(time_col="time"))
    assert ao_glob.unsafe_data.coords["trial"].values.tolist() == ["a", "b"]


def test_io_hard_p10b_005_duplicate_resolved_input_paths_fail_closed(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_005_duplicate_resolved_input_paths_fail_closed."""
    path = tmp_path / "dup.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0}])
    with pytest.raises(ValueError, match="tal.io.read_csv_logs"):
        read_csv_logs([str(path), str(path)], opts=CsvIngestOptions(time_col="time"))


def test_io_hard_p10b_004_derived_ingest_label_collisions_fail_closed_without_explicit_mapping(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_004_derived_ingest_label_collisions_fail_closed_without_explicit_mapping."""
    first = tmp_path / "d1" / "a.csv"
    second = tmp_path / "d2" / "a.csv"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(first, [{"time": 0.0, "value": 1.0}])
    _write_csv(second, [{"time": 0.0, "value": 2.0}])

    with pytest.raises(ValueError, match="derived ingest label collision"):
        read_csv_logs([str(first), str(second)], opts=CsvIngestOptions(time_col="time"))

    ao = read_csv_logs(
        {"left": str(first), "right": str(second)},
        opts=CsvIngestOptions(time_col="time"),
    )
    assert ao.unsafe_data.coords["trial"].values.tolist() == ["left", "right"]


def test_io_core_p10b_001_csv_multi_file_ingest_has_deterministic_time_policy_controls(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_001_csv_multi_file_ingest_has_deterministic_time_policy_controls."""
    path = tmp_path / "time_policy.csv"
    _write_csv(path, [{"time": 2.0, "value": 1.0}, {"time": np.nan, "value": 2.0}, {"time": 1.0, "value": 3.0}])

    with pytest.raises(ValueError, match="contains 1 non-finite"):
        read_csv_logs([str(path)], opts=CsvIngestOptions(time_col="time"))

    ao = read_csv_logs(
        [str(path)],
        opts=CsvIngestOptions(
            time_col="time",
            sort_time=False,
            invalid_time="drop",
            allow_nonmonotonic_normalize=True,
        ),
    )
    time_values = ao.unsafe_data.coords["time"].isel(trial=0).values.tolist()
    assert time_values[:2] == [1.0, 2.0]


def test_io_core_p10b_002_csv_ingest_rejects_name_collisions_and_invalid_specs_fail_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_002_csv_ingest_rejects_name_collisions_and_invalid_specs_fail_closed."""
    path = tmp_path / "collision.csv"
    _write_csv(path, [{"time": 0.0, "trial": 1.0}])
    with pytest.raises(ValueError, match="collide with reserved semantic names"):
        read_csv_logs([str(path)], opts=CsvIngestOptions(time_col="time"))


def test_io_hard_p10b_008_csv_ingest_explicit_value_columns_non_numeric_values_fail_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_008_csv_ingest_explicit_value_columns_non_numeric_values_fail_closed."""
    path = tmp_path / "non_numeric.csv"
    _write_csv(path, [{"time": 0.0, "value": "a"}, {"time": 1.0, "value": "b"}])
    with pytest.raises(ValueError, match="value column 'value'.*contains non-numeric values"):
        read_csv_logs(
            [str(path)],
            opts=CsvIngestOptions(time_col="time", value_columns=("value",)),
        )


def test_io_core_p10b_018_csv_ingest_explicit_value_columns_parse_numeric_text_deterministically(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_018_csv_ingest_explicit_value_columns_parse_numeric_text_deterministically."""
    path = tmp_path / "numeric_text.csv"
    _write_csv(path, [{"time": 0.0, "value": "1.5"}, {"time": 1.0, "value": "2"}])
    ao = read_csv_logs(
        [str(path)],
        opts=CsvIngestOptions(time_col="time", value_columns=("value",)),
    )
    values = ao.unsafe_data["value"].isel(trial=0).values.tolist()
    assert values[:2] == [1.5, 2.0]


def test_io_core_p10b_009_optional_csv_time_inference_requires_exactly_one_candidate(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_009_optional_csv_time_inference_requires_exactly_one_candidate."""
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    _write_csv(good, [{"timestamp": 0.0, "value": 1.0}])
    _write_csv(bad, [{"time": 0.0, "timestamp": 1.0, "value": 1.0}])

    ao = read_csv_logs([str(good)], opts=CsvIngestOptions(allow_time_infer=True))
    assert ao.unsafe_data.attrs["io_time_source"] == "timestamp"

    with pytest.raises(ValueError, match="requires exactly one candidate"):
        read_csv_logs([str(bad)], opts=CsvIngestOptions(allow_time_infer=True))


def test_io_core_p10b_006_csv_time_source_selection_order_is_deterministic(tmp_path: Path) -> None:
    """ID: IO_CORE_P10B_006_csv_time_source_selection_order_is_deterministic."""
    path = tmp_path / "selection_order.csv"
    _write_csv(path, [{"time": 0.0, "timestamp": 999.0, "value": 1.0}])
    ao = read_csv_logs(
        [str(path)],
        opts=CsvIngestOptions(time_col="time", allow_time_infer=True),
    )
    assert ao.unsafe_data.attrs["io_time_source"] == "time"
    values = ao.unsafe_data.coords["time"].isel(trial=0).values.tolist()
    assert values[:1] == [0.0]


def test_io_core_p10b_011_csv_time_inference_candidate_set_is_contract_locked(tmp_path: Path) -> None:
    """ID: IO_CORE_P10B_011_csv_time_inference_candidate_set_is_contract_locked."""
    allowed = tmp_path / "allowed.csv"
    blocked = tmp_path / "blocked.csv"
    _write_csv(allowed, [{"Time": 0.0, "value": 1.0}])
    _write_csv(blocked, [{"clock": 0.0, "value": 1.0}])

    ao = read_csv_logs([str(allowed)], opts=CsvIngestOptions(allow_time_infer=True))
    assert ao.unsafe_data.attrs["io_time_source"] == "Time"

    with pytest.raises(ValueError, match="requires exactly one candidate"):
        read_csv_logs([str(blocked)], opts=CsvIngestOptions(allow_time_infer=True))


def test_io_core_p10b_008_metadata_promotion_defaults_are_deterministic(tmp_path: Path) -> None:
    """ID: IO_CORE_P10B_008_metadata_promotion_defaults_are_deterministic."""
    first = tmp_path / "m1.csv"
    second = tmp_path / "m2.csv"
    _write_csv(first, [{"time": 0.0, "value": 1.0, "subject": "S1", "note": "A"}])
    _write_csv(second, [{"time": 0.0, "value": 2.0, "subject": "S1", "note": "B"}])

    ao = read_csv_logs(
        [str(first), str(second)],
        opts=CsvIngestOptions(time_col="time", metadata_columns=("subject", "note")),
    )
    assert ao.unsafe_data.attrs["subject"] == "S1"
    assert "note" not in ao.unsafe_data.attrs
    assert "note" not in ao.unsafe_data.coords


def test_io_core_p10b_010_metadata_promotion_default_targets_are_contract_locked(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_010_metadata_promotion_default_targets_are_contract_locked."""
    first = tmp_path / "p1.csv"
    second = tmp_path / "p2.csv"
    _write_csv(first, [{"time": 0.0, "value": 1.0, "subject": "S1", "path_note": "a"}])
    _write_csv(second, [{"time": 0.0, "value": 2.0, "subject": "S1", "path_note": "b"}])
    ao = read_csv_logs(
        [str(first), str(second)],
        opts=CsvIngestOptions(time_col="time", metadata_columns=("subject", "path_note")),
    )
    assert ao.unsafe_data.attrs["subject"] == "S1"
    assert "subject" not in ao.unsafe_data.coords
    assert "path_note" not in ao.unsafe_data.attrs
    assert "path_note" not in ao.unsafe_data.coords


def test_io_core_p10b_019_adapter_metadata_reserved_attr_key_tal_is_blocked_before_schema_finalize(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_019_adapter_metadata_reserved_attr_key_tal_is_blocked_before_schema_finalize."""
    path = tmp_path / "reserved_tal.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0, "tal": "x"}])
    with pytest.raises(ValueError, match="reserved for schema namespace"):
        read_csv_logs(
            [str(path)],
            opts=CsvIngestOptions(time_col="time", metadata_columns=("tal",)),
        )


def test_io_hard_p10b_009_adapter_metadata_key_tal_fails_closed_with_owner_prefixed_error(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_009_adapter_metadata_key_tal_fails_closed_with_owner_prefixed_error."""
    path = tmp_path / "reserved_tal_fail_closed.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0, "tal": "x"}])
    with pytest.raises(ValueError) as excinfo:
        read_csv_logs(
            [str(path)],
            opts=CsvIngestOptions(
                time_col="time",
                metadata_columns=("tal",),
                metadata_promotion=AdapterMetadataPromotionOptions(scalar_target="attrs"),
            ),
        )
    message = str(excinfo.value)
    assert message.startswith("tal.io.read_csv_logs:")
    assert "reserved for schema namespace" in message
    assert "schema.not_mapping" not in message
