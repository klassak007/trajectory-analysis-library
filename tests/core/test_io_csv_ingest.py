from __future__ import annotations

import gc
from pathlib import Path
import weakref

import numpy as np
import pandas as pd
import pytest

from tal.io import (
    AdapterMetadataPromotionOptions,
    CsvIngestOptions,
    read_csv_logs,
)
from tal.io import adapter_paths as adapter_paths_module
from tal.io import adapter_temp as adapter_temp_module
from tal.io import csv_logs as csv_logs_module
from tests._io_helpers import cleanup_failing_temporary_directory


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
    _write_csv(a_path, [{"time": 0.0, "x": 1.0, "y": 10.0}])
    _write_csv(b_path, [{"time": 0.0, "y": 20.0, "x": 2.0}])

    ao_seq = read_csv_logs(
        [str(b_path), str(a_path)],
        opts=CsvIngestOptions(time_col="time"),
    )
    assert ao_seq.unsafe_data.coords["trial"].values.tolist() == ["b", "a"]
    assert ao_seq.unsafe_data["x"].values.tolist() == [[2.0], [1.0]]
    assert ao_seq.unsafe_data["y"].values.tolist() == [[20.0], [10.0]]

    ao_glob = read_csv_logs(str(tmp_path / "*.csv"), opts=CsvIngestOptions(time_col="time"))
    assert ao_glob.unsafe_data.coords["trial"].values.tolist() == ["a", "b"]


def test_io_hard_p10b_005_duplicate_resolved_input_paths_fail_closed(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_005_duplicate_resolved_input_paths_fail_closed."""
    path = tmp_path / "dup.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0}])
    with pytest.raises(ValueError, match="tal.io.read_csv_logs"):
        read_csv_logs([str(path), str(path)], opts=CsvIngestOptions(time_col="time"))

    alias = tmp_path / "alias.csv"
    alias.hardlink_to(path)
    with pytest.raises(ValueError, match="duplicate resolved input path"):
        read_csv_logs([str(path), str(alias)], opts=CsvIngestOptions(time_col="time"))


def test_io_hard_p10b_030_zero_inode_files_use_path_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_030_zero_inode_files_use_path_identity."""
    stat = type("ZeroInodeStat", (), {"st_dev": 0, "st_ino": 0})()
    monkeypatch.setattr(Path, "stat", lambda _path: stat)

    adapter_paths_module._require_unique_resolved_paths(
        ["/logs/a.csv", "/logs/b.csv"],
        owner="tal.io.read_csv_logs",
    )
    with pytest.raises(ValueError, match="duplicate resolved input path"):
        adapter_paths_module._require_unique_resolved_paths(
            ["/logs/A.csv", "/logs/a.csv"],
            owner="tal.io.read_csv_logs",
        )


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


def test_io_hard_p10b_054_csv_integer_times_remain_exact_in_padded_grid(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_054_csv_integer_times_remain_exact_in_padded_grid."""
    base = 2**53
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        f"time,value\n{base + 1},1.0\n{base},2.0\n",
        encoding="utf-8",
    )
    second.write_text(f"time,value\n{base + 2},3.0\n", encoding="utf-8")

    ao = read_csv_logs(
        [str(first), str(second)],
        opts=CsvIngestOptions(time_col="time", monotonic_order="strict"),
    )
    time = ao.unsafe_data.coords["time"]

    assert time.dtype.kind in "iu"
    assert time.isel(trial=0).values.tolist() == [base, base + 1]
    assert int(time.isel(trial=1, sample=0)) == base + 2
    assert ao.unsafe_data.coords["sequence_size"].values.tolist() == [2, 1]


def test_io_hard_p10b_055_csv_mixed_time_kinds_fail_before_precision_loss(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_055_csv_mixed_time_kinds_fail_before_precision_loss."""
    integer = tmp_path / "integer.csv"
    floating = tmp_path / "floating.csv"
    integer.write_text(f"time,value\n{2**53 + 1},1.0\n", encoding="utf-8")
    floating.write_text("time,value\n0.5,2.0\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="tal.io.read_csv_logs: integer CSV time values cannot be combined .* precision loss",
    ):
        read_csv_logs(
            [str(integer), str(floating)],
            opts=CsvIngestOptions(time_col="time"),
        )


def test_io_hard_p10b_067_csv_same_file_mixed_time_kinds_fail_losslessly(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_067_csv_same_file_mixed_time_kinds_fail_losslessly."""
    path = tmp_path / "same_file_mixed_time.csv"
    path.write_text(
        f"time,value\n{2**53 + 1},1.0\n0.5,2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="tal.io.read_csv_logs: integer CSV time values cannot be combined .* precision loss",
    ):
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))


def test_io_hard_p10b_068_csv_dropped_invalid_time_preserves_exact_integers(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_068_csv_dropped_invalid_time_preserves_exact_integers."""
    base = 2**53
    path = tmp_path / "dropped_invalid_time.csv"
    path.write_text(
        f"time,value\n{base + 1},1.0\n,2.0\n{base + 2},3.0\n",
        encoding="utf-8",
    )

    ao = read_csv_logs(
        str(path),
        opts=CsvIngestOptions(time_col="time", invalid_time="drop"),
    )
    time = ao.unsafe_data.coords["time"]

    assert time.dtype.kind in "iu"
    assert time.isel(trial=0).values.tolist() == [base + 1, base + 2]
    assert ao.unsafe_data.coords["sequence_size"].values.tolist() == [2]


@pytest.mark.parametrize("token", ["oops", "NA", "NULL"])
def test_io_hard_p10b_072_csv_nonreal_time_fails_closed_under_drop_policy(
    tmp_path: Path,
    token: str,
) -> None:
    """ID: IO_HARD_P10B_072_csv_nonreal_time_fails_closed_under_drop_policy."""
    path = tmp_path / "nonreal_time.csv"
    path.write_text(f"time,value\n0,1.0\n{token},2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain ordered real numeric values"):
        read_csv_logs(
            str(path),
            opts=CsvIngestOptions(time_col="time", invalid_time="drop"),
        )


def test_io_hard_p10b_073_csv_raw_nonfinite_time_obeys_drop_policy(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_073_csv_raw_nonfinite_time_obeys_drop_policy."""
    path = tmp_path / "nonfinite_time.csv"
    path.write_text(
        "time,value\n0,1.0\nNaN,2.0\ninf,3.0\n-Infinity,4.0\n1,5.0\n",
        encoding="utf-8",
    )

    ao = read_csv_logs(
        str(path),
        opts=CsvIngestOptions(time_col="time", invalid_time="drop"),
    )

    assert ao.unsafe_data.coords["time"].isel(trial=0).values.tolist() == [0, 1]
    assert ao.unsafe_data["value"].isel(trial=0).values.tolist() == [1.0, 5.0]


@pytest.mark.parametrize(
    "token",
    ["1e309", "-1e309", "1e-4000", "-1e-4000", "١e-4000"],
)
def test_io_hard_p10b_082_finite_float64_range_failures_cannot_be_dropped(
    tmp_path: Path,
    token: str,
) -> None:
    """ID: IO_HARD_P10B_082_finite_float64_range_failures_cannot_be_dropped."""
    path = tmp_path / "finite_float64_range.csv"
    path.write_text(f"time,value\n0,1.0\n{token},2.0\n1,3.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finite numeric token outside the float64 range"):
        read_csv_logs(
            str(path),
            opts=CsvIngestOptions(time_col="time", invalid_time="drop"),
        )


@pytest.mark.parametrize("token", ["9" * 5_000, "٩" * 5_000])
def test_io_hard_p10b_083_long_integer_time_retains_public_owner_and_cause(
    tmp_path: Path,
    token: str,
) -> None:
    """ID: IO_HARD_P10B_083_long_integer_time_retains_public_owner_and_cause."""
    path = tmp_path / "long_integer_time.csv"
    path.write_text(f"time,value\n{token},1.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    assert str(error.value).startswith("tal.io.read_csv_logs: failed parsing an integer token")
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "token",
    ["٩٠٠٧١٩٩٢٥٤٧٤٠٩٩٣", "9_007_199_254_740_993"],
)
def test_io_hard_p10b_089_integral_time_spellings_remain_exact(
    tmp_path: Path,
    token: str,
) -> None:
    """ID: IO_HARD_P10B_089_integral_time_spellings_remain_exact."""
    path = tmp_path / "integral_time_spelling.csv"
    path.write_text(f"time,value\n{token},1.0\n", encoding="utf-8")

    ao = read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    time = ao.unsafe_data.coords["time"]
    assert time.dtype == np.dtype(np.int64)
    assert int(time.isel(trial=0, sample=0)) == 9_007_199_254_740_993


@pytest.mark.parametrize(
    "token",
    ["9__007", "_9007", "9007_", "+_9007", "٩__٠٠٧"],
)
def test_io_hard_p10b_090_malformed_integer_separators_fail_closed(
    tmp_path: Path,
    token: str,
) -> None:
    """ID: IO_HARD_P10B_090_malformed_integer_separators_fail_closed."""
    path = tmp_path / "malformed_integer_separator.csv"
    path.write_text(f"time,value\n0,1.0\n{token},2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain ordered real numeric values") as error:
        read_csv_logs(
            str(path),
            opts=CsvIngestOptions(time_col="time", invalid_time="drop"),
        )

    assert str(error.value).startswith("tal.io.read_csv_logs:")


def test_io_hard_p10b_056_csv_boolean_time_fails_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_056_csv_boolean_time_fails_closed."""
    path = tmp_path / "boolean_time.csv"
    path.write_text("time,value\nTrue,1.0\nFalse,2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain ordered real numeric values"):
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))


def test_io_core_p10b_002_csv_ingest_rejects_name_collisions_and_invalid_specs_fail_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_002_csv_ingest_rejects_name_collisions_and_invalid_specs_fail_closed."""
    path = tmp_path / "collision.csv"
    _write_csv(path, [{"time": 0.0, "trial": 1.0}])
    with pytest.raises(ValueError, match="collide with reserved semantic names"):
        read_csv_logs([str(path)], opts=CsvIngestOptions(time_col="time"))

    conflicting_layouts = (
        {"sequence_dim": "trial"},
        {"sequence_size_coord": "trial"},
        {"param_coord": "sample"},
    )
    for overrides in conflicting_layouts:
        with pytest.raises(ValueError, match="must be distinct"):
            read_csv_logs(
                [str(path)],
                opts=CsvIngestOptions(time_col="time", **overrides),  # type: ignore[arg-type]
            )


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


def test_io_core_p10b_025_generated_metadata_attrs_can_share_csv_data_names(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_025_generated_metadata_attrs_can_share_csv_data_names."""
    path = tmp_path / "metadata_namespace.csv"
    _write_csv(
        path,
        [
            {"time": 0.0, "io_time_source": 1.0},
            {"time": 1.0, "io_time_source": 2.0},
        ],
    )

    ao = read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    assert ao.unsafe_data["io_time_source"].values.tolist() == [[1.0, 2.0]]
    assert ao.unsafe_data.attrs["io_time_source"] == "time"


@pytest.mark.parametrize(
    "opts",
    [
        CsvIngestOptions(
            time_col="time",
            metadata_columns=("io_time_source",),
        ),
        CsvIngestOptions(
            time_col="time",
            metadata_columns=("io_source_paths",),
        ),
        CsvIngestOptions(
            time_col="time",
            value_columns=("io_time_source",),
            metadata_promotion=AdapterMetadataPromotionOptions(
                scalar_target="batch_coord"
            ),
        ),
    ],
)
def test_io_hard_p10b_065_csv_generated_metadata_collisions_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opts: CsvIngestOptions,
) -> None:
    """ID: IO_HARD_P10B_065_csv_generated_metadata_collisions_preflight."""

    def unexpected_path_resolution(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AssertionError("CSV metadata collision reached path resolution")

    monkeypatch.setattr(
        csv_logs_module,
        "resolve_ingest_inputs",
        unexpected_path_resolution,
    )

    with pytest.raises(ValueError, match="generated adapter metadata"):
        read_csv_logs(str(tmp_path / "missing.csv"), opts=opts)


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


@pytest.mark.parametrize(
    ("opts", "match"),
    [
        pytest.param(
            CsvIngestOptions(time_col="time", metadata_columns=("tal",)),
            "reserved for schema namespace",
            id="reserved-tal-attr",
        ),
        *[
            pytest.param(
                CsvIngestOptions(time_col="source_time", value_columns=(name,)),
                "value columns collide with reserved semantic names",
                id=f"value-{name}",
            )
            for name in ("trial", "sample", "sequence_size", "time")
        ],
    ],
)
def test_io_hard_p10b_084_static_csv_configuration_fails_before_input_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opts: CsvIngestOptions,
    match: str,
) -> None:
    """ID: IO_HARD_P10B_084_static_csv_configuration_fails_before_input_resolution."""

    def unexpected_path_resolution(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AssertionError("invalid CSV configuration reached input resolution")

    monkeypatch.setattr(csv_logs_module, "resolve_ingest_inputs", unexpected_path_resolution)
    with pytest.raises(ValueError, match=match) as error:
        read_csv_logs(str(tmp_path / "missing.csv"), opts=opts)

    assert str(error.value).startswith("tal.io.read_csv_logs:")


def test_io_hard_p10b_098_explicit_empty_value_columns_fail_before_input_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_098_explicit_empty_value_columns_fail_before_input_resolution."""

    def unexpected_path_resolution(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AssertionError("empty value-column selection reached input resolution")

    monkeypatch.setattr(csv_logs_module, "resolve_ingest_inputs", unexpected_path_resolution)
    opts = CsvIngestOptions(time_col="time", value_columns=())
    with pytest.raises(ValueError, match="value_columns must contain at least one name") as error:
        read_csv_logs(str(tmp_path / "missing.csv"), opts=opts)

    assert str(error.value).startswith("tal.io.read_csv_logs:")


def test_io_hard_p10b_014_csv_ingest_duplicate_headers_fail_closed(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_014_csv_ingest_duplicate_headers_fail_closed."""
    path = tmp_path / "duplicate_header.csv"
    path.write_text("time,value,value\n0.0,1.0,2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate CSV header 'value'"):
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))


@pytest.mark.parametrize(
    ("serialized_header", "field_name"),
    [
        pytest.param('"value,raw"', "value,raw", id="quoted-comma"),
        pytest.param('"value\nraw"', "value\nraw", id="quoted-newline"),
    ],
)
def test_io_core_p10b_030_csv_header_uses_standard_logical_record_grammar(
    tmp_path: Path,
    serialized_header: str,
    field_name: str,
) -> None:
    """ID: IO_CORE_P10B_030_csv_header_uses_standard_logical_record_grammar."""
    path = tmp_path / "quoted_header.csv"
    path.write_text(f"time,{serialized_header}\n0,1\n", encoding="utf-8")

    ao = read_csv_logs(
        str(path),
        opts=CsvIngestOptions(time_col="time", value_columns=(field_name,)),
    )

    assert ao.unsafe_data[field_name].values.tolist() == [[1.0]]


@pytest.mark.parametrize(
    ("contents", "cause_type"),
    [
        pytest.param(
            "time,value\n0,1,2\n",
            pd.errors.ParserWarning,
            id="first-row-parser-warning",
        ),
        pytest.param(
            "time,value\n0,1\n1,2,3\n",
            pd.errors.ParserError,
            id="later-row-parser-error",
        ),
    ],
)
def test_io_hard_p10b_080_overwide_csv_rows_fail_at_pandas_parser_boundary(
    tmp_path: Path,
    contents: str,
    cause_type: type[Exception],
) -> None:
    """ID: IO_HARD_P10B_080_overwide_csv_rows_fail_at_pandas_parser_boundary."""
    path = tmp_path / "overwide.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="failed reading CSV file") as error:
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    assert str(error.value).startswith("tal.io.read_csv_logs:")
    assert isinstance(error.value.__cause__, cause_type)


def test_io_hard_p10b_081_csv_reader_accepts_quoted_commas_and_multiline_fields(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_081_csv_reader_accepts_quoted_commas_and_multiline_fields."""
    path = tmp_path / "quoted_fields.csv"
    path.write_text(
        'time,note,value\n0,"alpha,beta",1.0\n1,"line one\nline two",2.0\n',
        encoding="utf-8",
    )

    ao = read_csv_logs(
        str(path),
        opts=CsvIngestOptions(time_col="time", value_columns=("value",)),
    )

    assert ao.unsafe_data["value"].values.tolist() == [[1.0, 2.0]]


def test_io_hard_p10b_051_csv_ingest_validates_header_after_blank_lines(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_051_csv_ingest_validates_header_after_blank_lines."""
    path = tmp_path / "leading_blank_duplicate_header.csv"
    path.write_text("   \ntime,time\n0.0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate CSV header 'time'"):
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))


@pytest.mark.parametrize(
    "record",
    [
        pytest.param("\v", id="vertical-tab"),
        pytest.param("\f", id="form-feed"),
        pytest.param("\u00a0", id="no-break-space"),
        pytest.param("\u2003", id="em-space"),
    ],
)
def test_io_hard_p10b_075_csv_header_blank_record_policy_matches_pandas(
    tmp_path: Path,
    record: str,
) -> None:
    """ID: IO_HARD_P10B_075_csv_header_blank_record_policy_matches_pandas."""
    path = tmp_path / "non_pandas_blank_record.csv"
    path.write_bytes(f"{record}\ntime,value\n0,1.0\n".encode())

    with pytest.raises(
        ValueError,
        match="CSV header at column 0 .* must contain a non-whitespace character",
    ) as error:
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    assert str(error.value).startswith("tal.io.read_csv_logs:")


def test_io_hard_p10b_026_csv_ingest_empty_headers_fail_closed(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_026_csv_ingest_empty_headers_fail_closed."""
    path = tmp_path / "empty_header.csv"
    path.write_text(",value\n0.0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty CSV header at column 0"):
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))


@pytest.mark.parametrize(
    ("header", "match"),
    [
        pytest.param(
            "time, ",
            "must contain a non-whitespace character",
            id="whitespace",
        ),
        pytest.param(
            "time,\ufeffvalue",
            "contains a BOM character",
            id="later-field-bom",
        ),
    ],
)
def test_io_hard_p10b_060_csv_ingest_parser_unsafe_headers_fail_closed(
    tmp_path: Path,
    header: str,
    match: str,
) -> None:
    """ID: IO_HARD_P10B_060_csv_ingest_parser_unsafe_headers_fail_closed."""
    path = tmp_path / "parser_unsafe_header.csv"
    path.write_text(f"{header}\n0.0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))


@pytest.mark.parametrize("header", ["time,value\x00hidden", "time,\x00"])
def test_io_hard_p10b_106_csv_ingest_nul_headers_fail_closed(
    tmp_path: Path,
    header: str,
) -> None:
    """ID: IO_HARD_P10B_106_csv_ingest_nul_headers_fail_closed."""
    path = tmp_path / f"nul_header_{len(header)}.csv"
    path.write_bytes(f"{header}\n0.0,1.0\n".encode())

    with pytest.raises(ValueError, match="CSV header at column .* contains a NUL byte") as error:
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    assert str(error.value).startswith("tal.io.read_csv_logs:")


def test_io_hard_p10b_102_csv_parser_backend_failure_retains_public_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_102_csv_parser_backend_failure_retains_public_owner."""
    path = tmp_path / "backend.csv"
    path.write_text("time,value\n0,1\n", encoding="utf-8")

    def fail_read(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise OSError("backend read failed")

    monkeypatch.setattr(csv_logs_module.pd, "read_csv", fail_read)
    with pytest.raises(ValueError, match="failed reading CSV file") as error:
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    assert str(error.value).startswith("tal.io.read_csv_logs:")
    assert isinstance(error.value.__cause__, OSError)


def test_io_hard_p10b_045_csv_ingest_cleanup_failure_retains_public_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_045_csv_ingest_cleanup_failure_retains_public_owner."""
    path = tmp_path / "cleanup.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0}])
    failing_directory = cleanup_failing_temporary_directory(
        adapter_temp_module.TemporaryDirectory,
        message="cleanup exploded",
    )
    monkeypatch.setattr(adapter_temp_module, "TemporaryDirectory", failing_directory)

    with pytest.raises(
        ValueError,
        match="tal.io.read_csv_logs: failed cleaning CSV ingest spool directory",
    ) as error:
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    assert isinstance(error.value.__cause__, OSError)


def test_io_hard_p10b_107_csv_cleanup_does_not_replace_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_107_csv_cleanup_does_not_replace_primary_failure."""
    path = tmp_path / "primary.csv"
    path.write_text("stub", encoding="utf-8")

    def _fail_record(*_args: object, **_kwargs: object) -> None:
        raise ValueError("tal.io.read_csv_logs: primary parse exploded")

    failing_directory = cleanup_failing_temporary_directory(
        adapter_temp_module.TemporaryDirectory,
        message="cleanup exploded",
    )
    monkeypatch.setattr(adapter_temp_module, "TemporaryDirectory", failing_directory)
    monkeypatch.setattr(csv_logs_module, "_read_csv_record", _fail_record)

    with pytest.raises(ValueError, match="primary parse exploded") as error:
        read_csv_logs(str(path), opts=CsvIngestOptions(time_col="time"))

    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        (field_name, bad_value)
        for field_name in ("allow_time_infer", "sort_time", "allow_nonmonotonic_normalize")
        for bad_value in ("false", 0, np.bool_(False))
    ],
)
def test_io_hard_p10b_015_csv_boolean_options_are_strict(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    """ID: IO_HARD_P10B_015_csv_boolean_options_are_strict."""
    path = tmp_path / "strict_bool.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0}])
    kwargs: dict[str, object] = {"time_col": "time", field_name: bad_value}
    opts = CsvIngestOptions(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=rf"{field_name} must be bool"):
        read_csv_logs(path, opts=opts)


def test_io_hard_p10b_016_csv_time_inference_errors_are_deterministic(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_016_csv_time_inference_errors_are_deterministic."""
    path = tmp_path / "ambiguous_time_candidates.csv"
    _write_csv(path, [{"time_sec": 0.0, "timestamp": 0.0, "value": 1.0}])

    with pytest.raises(ValueError) as excinfo:
        read_csv_logs(str(path), opts=CsvIngestOptions(allow_time_infer=True))

    assert "('time', 'timestamp', 't', 'time_s', 'time_sec')" in str(excinfo.value)
    assert "found ('timestamp', 'time_sec')" in str(excinfo.value)


def test_io_hard_p10b_021_csv_header_only_input_fails_closed(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_021_csv_header_only_input_fails_closed."""
    path = tmp_path / "header_only.csv"
    path.write_text("time,value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no data rows"):
        read_csv_logs(
            str(path),
            opts=CsvIngestOptions(time_col="time", value_columns=("value",)),
        )


def test_io_hard_p10b_022_csv_sequence_inputs_require_string_paths(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_022_csv_sequence_inputs_require_string_paths."""
    path = tmp_path / "path_object.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0}])

    with pytest.raises(TypeError, match="input path entries must be non-empty strings"):
        read_csv_logs([path], opts=CsvIngestOptions(time_col="time"))  # type: ignore[list-item]


def test_io_hard_p10b_037_csv_ingest_uses_resolved_paths_and_wraps_resolution_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_037_csv_ingest_uses_resolved_paths_and_wraps_resolution_errors."""
    resolved = tmp_path / "resolved.csv"
    _write_csv(resolved, [{"time": 0.0, "value": 1.0}])
    path_info = adapter_paths_module.ResolvedIngestInput(
        label="resolved",
        resolved_path=str(resolved.resolve()),
    )
    monkeypatch.setattr(
        csv_logs_module,
        "resolve_ingest_inputs",
        lambda *_args, **_kwargs: (path_info,),
    )

    ao = read_csv_logs("ignored", opts=CsvIngestOptions(time_col="time"))
    assert ao.unsafe_data["value"].values.tolist() == [[1.0]]

    monkeypatch.undo()
    with pytest.raises(ValueError, match="tal.io.read_csv_logs: failed resolving input path"):
        read_csv_logs(["invalid\x00path"], opts=CsvIngestOptions(time_col="time"))


@pytest.mark.parametrize(
    ("field_name", "opts"),
    [
        ("monotonic_order", CsvIngestOptions(time_col="time", monotonic_order=[])),  # type: ignore[arg-type]
        ("invalid_time", CsvIngestOptions(time_col="time", invalid_time={})),  # type: ignore[arg-type]
        (
            "scalar_target",
            CsvIngestOptions(
                time_col="time",
                metadata_promotion=AdapterMetadataPromotionOptions(scalar_target=[]),  # type: ignore[arg-type]
            ),
        ),
        (
            "nonscalar_target",
            CsvIngestOptions(
                time_col="time",
                metadata_promotion=AdapterMetadataPromotionOptions(nonscalar_target={}),  # type: ignore[arg-type]
            ),
        ),
    ],
)
def test_io_hard_p10b_027_csv_choice_options_reject_non_strings(
    tmp_path: Path,
    field_name: str,
    opts: CsvIngestOptions,
) -> None:
    """ID: IO_HARD_P10B_027_csv_choice_options_reject_non_strings."""
    path = tmp_path / "choice_option.csv"
    _write_csv(path, [{"time": 0.0, "value": 1.0}])

    with pytest.raises(TypeError, match=rf"tal.io.read_csv_logs: {field_name} must be a string"):
        read_csv_logs(str(path), opts=opts)


def test_io_perf_p10b_006_csv_ingest_releases_records_before_grid_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_006_csv_ingest_releases_records_before_grid_allocation."""
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    for path in paths:
        path.write_text("stub", encoding="utf-8")
    previous_refs: list[weakref.ReferenceType[np.ndarray]] = []
    spool_dirs: list[Path] = []
    call_count = 0
    real_temporary_directory = adapter_temp_module.TemporaryDirectory

    def _temporary_directory(*args: object, **kwargs: object):
        context = real_temporary_directory(*args, **kwargs)
        spool_dirs.append(Path(context.name))
        return context

    def _record(path_info: object, *, opts: CsvIngestOptions, owner: str):
        nonlocal call_count, previous_refs
        _ = opts, owner
        gc.collect()
        assert all(reference() is None for reference in previous_refs)
        call_count += 1
        times = np.asarray([float(call_count)], dtype=float)
        values = {"value": np.asarray([float(call_count)], dtype=float)}
        previous_refs = [weakref.ref(times), weakref.ref(values["value"])]
        return csv_logs_module._CsvRecord(
            label=path_info.label,
            resolved_path=path_info.resolved_path,
            time_col="time",
            time_values=times,
            value_columns=("value",),
            values=values,
            metadata={},
        )

    monkeypatch.setattr(adapter_temp_module, "TemporaryDirectory", _temporary_directory)
    monkeypatch.setattr(csv_logs_module, "_read_csv_record", _record)
    ao = read_csv_logs([str(path) for path in paths], opts=CsvIngestOptions(time_col="time"))
    gc.collect()

    assert all(reference() is None for reference in previous_refs)
    assert spool_dirs and all(not path.exists() for path in spool_dirs)
    assert ao.unsafe_data["value"].values.tolist() == [[1.0], [2.0]]


def test_io_perf_p10b_011_csv_scalar_metadata_fails_at_second_distinct_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_011_csv_scalar_metadata_fails_at_second_distinct_value."""
    from tal.io import csv_metadata as csv_metadata_module

    frame = pd.DataFrame({"subject": [None, np.int64(7), np.int64(8), "unvisited"]})
    real_isna = csv_metadata_module.pd.isna
    visited: list[object] = []

    def _record_isna(value: object) -> object:
        visited.append(value)
        if len(visited) > 3:
            raise AssertionError("metadata scalar validation scanned beyond the first mismatch")
        return real_isna(value)

    monkeypatch.setattr(csv_metadata_module.pd, "isna", _record_isna)
    with pytest.raises(ValueError, match="must be scalar per input file"):
        csv_metadata_module.collect_csv_scalar_metadata(
            frame,
            metadata_columns=("subject",),
            owner="tal.io.read_csv_logs",
            source_path="metadata.csv",
        )

    assert visited == [None, np.int64(7), np.int64(8)]


def test_io_core_p10b_027_csv_scalar_metadata_ignores_missing_and_normalizes_numpy_scalars() -> None:
    """ID: IO_CORE_P10B_027_csv_scalar_metadata_ignores_missing_and_normalizes_numpy_scalars."""
    from tal.io import csv_metadata as csv_metadata_module

    frame = pd.DataFrame({"subject": [pd.NA, np.int64(7), None, np.int64(7)]})

    metadata = csv_metadata_module.collect_csv_scalar_metadata(
        frame,
        metadata_columns=("subject",),
        owner="tal.io.read_csv_logs",
        source_path="metadata.csv",
    )

    assert metadata == {"subject": 7}
    assert isinstance(metadata["subject"], int)
