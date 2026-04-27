from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.io import CsvExportOptions, write_csv_logs


def _grouped_ao(
    *,
    labels: list[object],
    values: np.ndarray,
    sizes: np.ndarray,
    size_name: str = "sequence_size",
) -> AnalysisObject:
    trial_count, width = values.shape
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), values)},
        coords={
            "trial": np.asarray(labels, dtype=object),
            "sample": np.arange(width, dtype=np.int64),
            "time": (("trial", "sample"), np.tile(np.arange(width, dtype=float), (trial_count, 1))),
            size_name: ("trial", sizes.astype(np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["trial"],
        param_coord="time",
        sequence_size_coord=size_name,
    )


def test_io_core_p10b_015_csv_export_uses_batch_coord_labels_and_preserves_batch_order_in_outputs(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_015_csv_export_uses_batch_coord_labels_and_preserves_batch_order_in_outputs."""
    ao = _grouped_ao(
        labels=["b", "a"],
        values=np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=float),
        sizes=np.asarray([2, 2], dtype=np.int64),
    )
    paths = write_csv_logs(ao, str(tmp_path / "out"))
    assert [Path(path).name for path in paths] == ["b.csv", "a.csv"]

    first = pd.read_csv(paths[0])
    second = pd.read_csv(paths[1])
    assert first["value"].tolist() == [10.0, 11.0]
    assert second["value"].tolist() == [20.0, 21.0]


def test_io_core_p10b_016_csv_export_default_sequence_size_coord_uses_schema_declared_coord(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_016_csv_export_default_sequence_size_coord_uses_schema_declared_coord."""
    ao = _grouped_ao(
        labels=["trial_a"],
        values=np.asarray([[10.0, 11.0, 12.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
        size_name="n_valid",
    )
    paths = write_csv_logs(ao, str(tmp_path / "schema_default"))
    frame = pd.read_csv(paths[0])
    assert frame["value"].tolist() == [10.0, 11.0]


def test_io_core_p10b_017_csv_export_sequence_size_coord_override_validation_is_deterministic(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_017_csv_export_sequence_size_coord_override_validation_is_deterministic."""
    ao = _grouped_ao(
        labels=["trial_b"],
        values=np.asarray([[20.0, 21.0, 22.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
        size_name="n_valid",
    )
    paths = write_csv_logs(
        ao,
        str(tmp_path / "override_valid"),
        opts=CsvExportOptions(sequence_size_coord="n_valid"),
    )
    frame = pd.read_csv(paths[0])
    assert frame["value"].tolist() == [20.0, 21.0]


def test_io_hard_p10b_007_csv_export_invalid_explicit_sequence_size_coord_override_fails_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_007_csv_export_invalid_explicit_sequence_size_coord_override_fails_closed."""
    ao = _grouped_ao(
        labels=["trial_c"],
        values=np.asarray([[30.0, 31.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="sequence_size_coord 'n_valid' was not found"):
        write_csv_logs(
            ao,
            str(tmp_path / "override_invalid"),
            opts=CsvExportOptions(sequence_size_coord="n_valid"),
        )


def test_io_core_p10b_020_csv_export_integer_valued_float_sequence_size_override_is_accepted(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_020_csv_export_integer_valued_float_sequence_size_override_is_accepted."""
    trial_count = 1
    width = 3
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[5.0, 6.0, 7.0]], dtype=float))},
        coords={
            "trial": np.asarray(["trial_d"], dtype=object),
            "sample": np.arange(width, dtype=np.int64),
            "time": (("trial", "sample"), np.tile(np.arange(width, dtype=float), (trial_count, 1))),
            "n_valid": ("trial", np.asarray([2.0], dtype=float)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["trial"],
        param_coord="time",
        sequence_size_coord="n_valid",
        validate=False,
    )
    paths = write_csv_logs(
        ao,
        str(tmp_path / "override_integer_float"),
        opts=CsvExportOptions(sequence_size_coord="n_valid"),
    )
    frame = pd.read_csv(paths[0])
    assert frame["value"].tolist() == [5.0, 6.0]


def test_io_hard_p10b_010_csv_export_fractional_sequence_size_override_fails_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_010_csv_export_fractional_sequence_size_override_fails_closed."""
    trial_count = 1
    width = 3
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[8.0, 9.0, 10.0]], dtype=float))},
        coords={
            "trial": np.asarray(["trial_e"], dtype=object),
            "sample": np.arange(width, dtype=np.int64),
            "time": (("trial", "sample"), np.tile(np.arange(width, dtype=float), (trial_count, 1))),
            "sequence_size": ("trial", np.asarray([3], dtype=np.int64)),
            "n_valid": ("trial", np.asarray([1.9], dtype=float)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["trial"],
        param_coord="time",
        sequence_size_coord="sequence_size",
    )
    with pytest.raises(ValueError, match="must be integer-valued"):
        write_csv_logs(
            ao,
            str(tmp_path / "override_fractional"),
            opts=CsvExportOptions(sequence_size_coord="n_valid"),
        )


def test_io_hard_p10b_011_csv_export_non_finite_sequence_size_override_fails_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_011_csv_export_non_finite_sequence_size_override_fails_closed."""
    trial_count = 1
    width = 3
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[11.0, 12.0, 13.0]], dtype=float))},
        coords={
            "trial": np.asarray(["trial_f"], dtype=object),
            "sample": np.arange(width, dtype=np.int64),
            "time": (("trial", "sample"), np.tile(np.arange(width, dtype=float), (trial_count, 1))),
            "sequence_size": ("trial", np.asarray([3], dtype=np.int64)),
            "n_valid": ("trial", np.asarray([np.inf], dtype=float)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["trial"],
        param_coord="time",
        sequence_size_coord="sequence_size",
    )
    with pytest.raises(ValueError, match="must be finite"):
        write_csv_logs(
            ao,
            str(tmp_path / "override_nonfinite"),
            opts=CsvExportOptions(sequence_size_coord="n_valid"),
        )


def test_io_hard_p10b_012_csv_export_mixed_name_collisions_after_string_normalization_fail_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_012_csv_export_mixed_name_collisions_after_string_normalization_fail_closed."""
    trial_count = 1
    width = 2
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[1.0, 2.0]], dtype=float))},
        coords={
            "trial": np.asarray(["trial_g"], dtype=object),
            "sample": np.arange(width, dtype=np.int64),
            "time": (("trial", "sample"), np.tile(np.arange(width, dtype=float), (trial_count, 1))),
            "sequence_size": ("trial", np.asarray([2], dtype=np.int64)),
            1: ("sample", np.asarray([10.0, 11.0], dtype=float)),
            "1": ("sample", np.asarray([20.0, 21.0], dtype=float)),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=["trial"],
        param_coord="time",
        sequence_size_coord="sequence_size",
    )
    with pytest.raises(ValueError, match="after string normalization"):
        write_csv_logs(ao, str(tmp_path / "mixed_name_collision"))


def test_io_hard_p10b_006_duplicate_export_batch_labels_fail_closed_under_canonical_normalization(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_006_duplicate_export_batch_labels_fail_closed_under_canonical_normalization."""
    ao_dup = _grouped_ao(
        labels=["dup", "dup"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="duplicate batch label"):
        write_csv_logs(ao_dup, str(tmp_path / "raw"))

    ao_norm = _grouped_ao(
        labels=["a", "a.csv"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="normalized export path collision"):
        write_csv_logs(ao_norm, str(tmp_path / "normalized"))


def test_io_core_p10b_005_csv_export_rejects_duplicate_normalized_group_labels_that_would_overwrite(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_005_csv_export_rejects_duplicate_normalized_group_labels_that_would_overwrite."""
    ao = _grouped_ao(
        labels=["trial", "trial.csv"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="normalized export path collision"):
        write_csv_logs(ao, str(tmp_path / "out"))


def test_io_hard_p10b_001_path_traversal_and_absolute_label_paths_fail_closed(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_001_path_traversal_and_absolute_label_paths_fail_closed."""
    ao_traversal = _grouped_ao(
        labels=["../escape"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="traversal"):
        write_csv_logs(ao_traversal, str(tmp_path / "out"))

    ao_absolute = _grouped_ao(
        labels=[str(Path("/") / "tmp" / "abs")],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="absolute path"):
        write_csv_logs(ao_absolute, str(tmp_path / "out2"))


def test_io_core_p10b_012_export_label_normalization_rules_are_contract_locked(tmp_path: Path) -> None:
    """ID: IO_CORE_P10B_012_export_label_normalization_rules_are_contract_locked."""
    ao = _grouped_ao(
        labels=["nested\\\\trial_a"],
        values=np.asarray([[1.0, 2.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    )
    paths = write_csv_logs(ao, str(tmp_path / "out"))
    assert len(paths) == 1
    expected = (tmp_path / "out" / "nested" / "trial_a.csv").resolve()
    assert Path(paths[0]).resolve() == expected
