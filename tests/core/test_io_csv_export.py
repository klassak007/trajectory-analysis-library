from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import delayed
from dask.local import get_sync

from tal.core import AnalysisObject, SchemaError
from tal.core.schema import set_validity
from tal.io import CsvExportOptions, write_csv_logs
from tal.io import adapter_paths as adapter_paths_module
from tal.io import csv_commit as csv_commit_module
from tal.io import csv_export as csv_export_module
from tal.io import csv_sizes as csv_sizes_module


class _FailingStringIdentity:
    def __str__(self) -> str:
        raise RuntimeError("string conversion exploded")


class _FailingReprIdentity:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        raise RuntimeError("representation exploded")


class _FailingSizeFloat(float):
    def __float__(self) -> float:
        raise RuntimeError("float conversion exploded")

    def __repr__(self) -> str:
        raise RuntimeError("representation exploded")


class _FailingSizeComparison(float):
    def __eq__(self, other: object) -> bool:
        _ = other
        raise RuntimeError("comparison exploded")

    def __repr__(self) -> str:
        raise RuntimeError("representation exploded")


class _FailingSizeFraction(Fraction):
    @property
    def denominator(self) -> int:
        raise RuntimeError("denominator access exploded")

    def __repr__(self) -> str:
        raise RuntimeError("representation exploded")


def _ao_with_field_name(name: object) -> AnalysisObject:
    return AnalysisObject.from_data(
        xr.Dataset(
            {name: (("trial", "sample"), [[1.0]])},
            coords={
                "trial": ["run"],
                "sample": [0],
                "sequence_size": ("trial", [1]),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="sequence_size",
        validate=True,
    )


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


def _destination_preflight(path: Path) -> tuple[adapter_paths_module.ExportDestinationPreflight, ...]:
    return adapter_paths_module.snapshot_export_destinations(
        (str(path),),
        root=path.parent.resolve(),
        owner="tal.io.write_csv_logs",
    )


def _dimension_param_export_ao(*, include_value: bool) -> AnalysisObject:
    data_vars: dict[str, object] = {}
    if include_value:
        data_vars["value"] = (("trial", "time"), [[1.0, 2.0]])
    ds = xr.Dataset(
        data_vars,
        coords={
            "trial": ["run"],
            "time": [10, 20],
            "sequence_size": ("trial", [2]),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="time",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="sequence_size",
    )


@pytest.mark.parametrize("include_value", [False, True])
def test_io_hard_p10b_062_csv_export_preserves_dimension_parameter_coordinate(
    tmp_path: Path,
    include_value: bool,
) -> None:
    """ID: IO_HARD_P10B_062_csv_export_preserves_dimension_parameter_coordinate."""
    root = tmp_path / ("with_value" if include_value else "parameter_only")

    paths = write_csv_logs(_dimension_param_export_ao(include_value=include_value), str(root))
    frame = pd.read_csv(paths[0])

    assert frame["time"].tolist() == [10, 20]
    assert list(frame.columns) == (["time", "value"] if include_value else ["time"])


def _lazy_size_export_ao(
    size: object,
    *,
    override: np.ndarray | None = None,
) -> AnalysisObject:
    coords: dict[str, object] = {
        "trial": ["a"],
        "sample": [0, 1],
        "sequence_size": ("trial", size),
    }
    if override is not None:
        coords["effective_size"] = ("trial", override)
    ds = xr.Dataset(
        {"value": (("trial", "sample"), [[1.0, 2.0]])},
        coords=coords,
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        sequence_size_coord="sequence_size",
        validate=False,
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

    numeric_labels = _grouped_ao(
        labels=[1],
        values=np.asarray([[30.0, 31.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="cannot reuse batch coordinate"):
        write_csv_logs(
            numeric_labels,
            str(tmp_path / "override_batch_coord"),
            opts=CsvExportOptions(sequence_size_coord="trial"),
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

    ao_nan = _grouped_ao(
        labels=[np.nan, np.nan],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="duplicate batch label"):
        write_csv_logs(ao_nan, str(tmp_path / "missing"))

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


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(["A", "a"], id="case-folded"),
        pytest.param(["é", "e\N{COMBINING ACUTE ACCENT}"], id="unicode-normalized"),
    ],
)
def test_io_hard_p10b_018_csv_export_filesystem_equivalent_paths_fail_closed(
    tmp_path: Path,
    labels: list[str],
) -> None:
    """ID: IO_HARD_P10B_018_csv_export_filesystem_equivalent_paths_fail_closed."""
    ao = _grouped_ao(
        labels=labels,
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    root = tmp_path / "filesystem_collision"
    with pytest.raises(ValueError, match="normalized export path collision"):
        write_csv_logs(ao, str(root))
    assert not root.exists()


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(["A/one", "a/two"], id="case-folded-parent"),
        pytest.param(
            ["é/one", "e\N{COMBINING ACUTE ACCENT}/two"],
            id="unicode-normalized-parent",
        ),
    ],
)
def test_io_hard_p10b_103_csv_export_filesystem_equivalent_parents_fail_preflight(
    tmp_path: Path,
    labels: list[str],
) -> None:
    """ID: IO_HARD_P10B_103_csv_export_filesystem_equivalent_parents_fail_preflight."""
    ao = _grouped_ao(
        labels=labels,
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    root = tmp_path / "parent_collision"

    with pytest.raises(ValueError, match="failed preparing CSV export destinations") as error:
        write_csv_logs(ao, str(root))

    assert "planned export parent directory collision" in str(
        error.value.__cause__.__cause__
    )
    assert not root.exists()


def test_io_hard_p10b_034_csv_export_planned_parent_file_conflicts_fail_preflight(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_034_csv_export_planned_parent_file_conflicts_fail_preflight."""
    ao = _grouped_ao(
        labels=["run", "run.csv/child"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    root = tmp_path / "planned_topology"

    with pytest.raises(ValueError, match="another planned destination as parent path"):
        write_csv_logs(ao, str(root))

    assert not root.exists()


def test_io_hard_p10b_035_csv_export_existing_destination_aliases_fail_preflight(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_035_csv_export_existing_destination_aliases_fail_preflight."""
    root = tmp_path / "hardlink_alias"
    root.mkdir()
    first = root / "a.csv"
    second = root / "b.csv"
    first.write_text("sentinel\n", encoding="utf-8")
    try:
        second.hardlink_to(first)
    except OSError as exc:  # pragma: no cover - platform/filesystem capability.
        pytest.skip(f"hard links are unavailable: {exc}")
    ao = _grouped_ao(
        labels=["a", "b"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="alias one filesystem file"):
        write_csv_logs(ao, str(root))

    assert first.read_text(encoding="utf-8") == "sentinel\n"
    assert second.read_text(encoding="utf-8") == "sentinel\n"


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


def test_io_hard_p10b_033_invalid_filesystem_strings_retain_public_owner(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_033_invalid_filesystem_strings_retain_public_owner."""
    ordinary = _grouped_ao(
        labels=["trial"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: failed resolving export root",
    ):
        write_csv_logs(ordinary, f"{tmp_path}\x00invalid")

    malformed_label = _grouped_ao(
        labels=["trial\x00invalid"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: failed resolving export label",
    ):
        write_csv_logs(malformed_label, str(tmp_path / "out3"))


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


def test_io_hard_p10b_013_csv_export_preflight_precedes_output_mutation(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_013_csv_export_preflight_precedes_output_mutation."""
    collision = _grouped_ao(
        labels=["a", "a.csv"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    collision_root = tmp_path / "collision"
    with pytest.raises(ValueError, match="normalized export path collision"):
        write_csv_logs(collision, str(collision_root))
    assert not collision_root.exists()

    malformed_ds = _grouped_ao(
        labels=["a", "b"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    ).unsafe_data.assign_coords(sequence_size=("trial", [1, 2]))
    malformed = AnalysisObject.from_data(
        malformed_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="sequence_size",
        validate=False,
    )
    validity_root = tmp_path / "validity"
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        write_csv_logs(malformed, str(validity_root))
    assert not validity_root.exists()

    file_root = tmp_path / "file_root"
    file_root.write_text("sentinel", encoding="utf-8")
    with pytest.raises(ValueError, match="export parent path.*is not a directory"):
        write_csv_logs(collision.isel(trial=[0]), str(file_root))
    assert file_root.read_text(encoding="utf-8") == "sentinel"


def test_io_perf_p10b_001_csv_export_joint_realization_preserves_coherent_snapshot(
    tmp_path: Path,
) -> None:
    """ID: IO_PERF_P10B_001_csv_export_joint_realization_preserves_coherent_snapshot."""
    calls: list[int] = []

    def source() -> np.ndarray:
        calls.append(len(calls) + 1)
        return np.full((2, 2), calls[-1], dtype=float)

    shared = da.from_delayed(delayed(source)(), shape=(2, 2), dtype=float)
    ds = xr.Dataset(
        {"x": (("trial", "sample"), shared), "y": (("trial", "sample"), shared)},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1],
            "time": ("sample", [0.0, 1.0]),
            "sequence_size": ("trial", [2, 2]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="sequence_size",
        validate=True,
    )

    paths = write_csv_logs(ao, str(tmp_path / "snapshot"))

    assert calls == [1]
    for path in paths:
        frame = pd.read_csv(path)
        assert frame["x"].tolist() == [1.0, 1.0]
        assert frame["y"].tolist() == [1.0, 1.0]


def test_io_perf_p10b_018_eager_csv_export_does_not_invoke_dask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_018_eager_csv_export_does_not_invoke_dask."""

    def unexpected_compute(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AssertionError("eager CSV export invoked Dask")

    monkeypatch.setattr(csv_export_module, "compute", unexpected_compute)
    ao = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0, 2.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    )

    paths = write_csv_logs(ao, str(tmp_path / "eager"))

    assert pd.read_csv(paths[0])["value"].tolist() == [1.0, 2.0]


def test_io_perf_p10b_019_csv_export_honors_configured_dask_scheduler(
    tmp_path: Path,
) -> None:
    """ID: IO_PERF_P10B_019_csv_export_honors_configured_dask_scheduler."""
    scheduler_calls = 0

    def tracking_scheduler(dsk: object, keys: object, **kwargs: object) -> object:
        nonlocal scheduler_calls
        scheduler_calls += 1
        return get_sync(dsk, keys, **kwargs)  # type: ignore[arg-type]

    values = da.from_array(np.asarray([[1.0, 2.0]]), chunks=(1, 1))
    ao = _grouped_ao(
        labels=["run"],
        values=values,
        sizes=np.asarray([2], dtype=np.int64),
    )

    with dask.config.set(scheduler=tracking_scheduler):
        paths = write_csv_logs(ao, str(tmp_path / "configured-scheduler"))

    assert scheduler_calls == 1
    assert pd.read_csv(paths[0])["value"].tolist() == [1.0, 2.0]


def test_io_hard_p10b_116_lazy_export_realizes_before_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_116_lazy_export_realizes_before_serialization."""
    realization_complete = False
    real_compute = csv_export_module.compute
    real_serialize = csv_export_module._serialize_csv_row

    def tracking_compute(*args: object, **kwargs: object) -> tuple[object, ...]:
        nonlocal realization_complete
        result = tuple(real_compute(*args, **kwargs))
        realization_complete = True
        return result

    def checked_serialize(*args: object, **kwargs: object) -> str:
        assert realization_complete, "serialization entered the Dask graph"
        return real_serialize(*args, **kwargs)

    monkeypatch.setattr(csv_export_module, "compute", tracking_compute)
    monkeypatch.setattr(csv_export_module, "_serialize_csv_row", checked_serialize)
    values = da.from_array(np.asarray([[1.0], [2.0]]), chunks=(1, 1))
    ao = _grouped_ao(
        labels=["a", "b"],
        values=values,
        sizes=np.asarray([1, 1], dtype=np.int64),
    )

    paths = write_csv_logs(ao, str(tmp_path / "realize-before-serialize"))

    assert [pd.read_csv(path)["value"].tolist() for path in paths] == [
        [1.0],
        [2.0],
    ]


def test_io_perf_p10b_002_csv_export_does_not_execute_excluded_padded_tail_chunks(
    tmp_path: Path,
) -> None:
    """ID: IO_PERF_P10B_002_csv_export_does_not_execute_excluded_padded_tail_chunks."""
    calls: list[str] = []

    def load_tail() -> np.ndarray:
        calls.append("tail")
        raise AssertionError("invalid padded tail must not execute")

    head = da.from_array(np.asarray([[1.0, 2.0]], dtype=float), chunks=(1, 2))
    tail = da.from_delayed(delayed(load_tail)(), shape=(1, 2), dtype=float)
    values = da.concatenate((head, tail), axis=1)
    ds = xr.Dataset(
        {"value": (("trial", "sample"), values)},
        coords={
            "trial": ["a"],
            "sample": [0, 1, 2, 3],
            "time": ("sample", [0.0, 1.0, 2.0, 3.0]),
            "sequence_size": ("trial", [2]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="sequence_size",
        validate=True,
    )

    paths = write_csv_logs(ao, str(tmp_path / "valid_only"))

    assert calls == []
    assert pd.read_csv(paths[0])["value"].tolist() == [1.0, 2.0]


def test_io_perf_p10b_003_csv_export_does_not_execute_ignored_metadata_coordinates(
    tmp_path: Path,
) -> None:
    """ID: IO_PERF_P10B_003_csv_export_does_not_execute_ignored_metadata_coordinates."""
    calls: list[str] = []

    def load_ignored(name: str) -> np.ndarray:
        calls.append(name)
        raise AssertionError(f"ignored {name} coordinate must not execute")

    batch_meta = da.from_delayed(
        delayed(load_ignored)("batch"),
        shape=(1,),
        dtype=float,
    )
    scalar_meta = da.from_delayed(
        delayed(load_ignored)("scalar"),
        shape=(),
        dtype=float,
    )
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.asarray([[1.0, 2.0]], dtype=float))},
        coords={
            "trial": ["a"],
            "sample": [0, 1],
            "time": ("sample", [0.0, 1.0]),
            "sequence_size": ("trial", [2]),
            "batch_meta": ("trial", batch_meta),
            "scalar_meta": xr.Variable((), scalar_meta),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="sequence_size",
        validate=False,
    )

    paths = write_csv_logs(ao, str(tmp_path / "ignore_metadata"))

    assert calls == []
    assert pd.read_csv(paths[0])["value"].tolist() == [1.0, 2.0]


def test_io_hard_p10b_019_csv_export_validates_size_override_before_payload_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_019_csv_export_validates_size_override_before_payload_execution."""
    payload_calls: list[str] = []
    label_calls: list[str] = []

    def load_payload() -> np.ndarray:
        payload_calls.append("payload")
        return np.asarray([[1.0, 2.0]], dtype=float)

    payload = da.from_delayed(delayed(load_payload)(), shape=(1, 2), dtype=float)
    monkeypatch.setattr(
        csv_export_module,
        "_materialize_batch_labels",
        lambda *_args, **_kwargs: label_calls.append("labels") or ("a",),
    )
    ds = xr.Dataset(
        {"value": (("trial", "sample"), payload)},
        coords={
            "trial": ["a"],
            "sample": [0, 1],
            "time": ("sample", [0.0, 1.0]),
            "sequence_size": ("trial", [2]),
            "n_valid": ("trial", [1.5]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time",
        sequence_size_coord="sequence_size",
        validate=False,
    )
    root = tmp_path / "invalid_override"

    with pytest.raises(ValueError, match="must be integer-valued"):
        write_csv_logs(
            ao,
            str(root),
            opts=CsvExportOptions(sequence_size_coord="n_valid"),
        )

    assert payload_calls == []
    assert label_calls == []
    assert not root.exists()


def test_io_hard_p10b_069_csv_projection_preflight_precedes_lazy_size_read(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_069_csv_projection_preflight_precedes_lazy_size_read."""
    size_calls: list[str] = []

    def load_size() -> np.ndarray:
        size_calls.append("size")
        raise AssertionError("unrepresentable projection must fail before size realization")

    size = da.from_delayed(delayed(load_size)(), shape=(1,), dtype=np.int64)
    ds = xr.Dataset(
        {"value": (("trial", "sample", "axis"), np.ones((1, 2, 3)))},
        coords={
            "trial": ["run"],
            "sample": [0, 1],
            "axis": ["x", "y", "z"],
            "sequence_size": ("trial", size),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        sequence_size_coord="sequence_size",
        validate=False,
    )
    root = tmp_path / "unrepresentable"

    with pytest.raises(ValueError, match="is not representable"):
        write_csv_logs(ao, str(root))

    assert size_calls == []
    assert not root.exists()


def test_io_perf_p10b_004_csv_export_serializes_directly_and_commits_in_batch_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_004_csv_export_serializes_directly_and_commits_in_batch_order."""
    serialization_destinations: list[Path] = []
    replace_sources: list[Path] = []
    committed_destinations: list[Path] = []
    original_to_csv = pd.DataFrame.to_csv
    original_replace = csv_commit_module.os.replace
    export_root = tmp_path / "direct-staging"

    def tracked_to_csv(
        frame: pd.DataFrame,
        path_or_buf: object = None,
        *args: object,
        **kwargs: object,
    ) -> object:
        assert path_or_buf is not None
        serialization_destinations.append(Path(path_or_buf))
        return original_to_csv(frame, path_or_buf, *args, **kwargs)

    def tracked_replace(src: object, dst: object) -> None:
        replace_sources.append(Path(src))
        if committed_destinations:
            assert committed_destinations[-1].exists()
        committed_destinations.append(Path(dst))
        original_replace(src, dst)

    monkeypatch.setattr(pd.DataFrame, "to_csv", tracked_to_csv)
    monkeypatch.setattr(csv_commit_module.os, "replace", tracked_replace)
    ao = _grouped_ao(
        labels=["a", "b"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )

    paths = write_csv_logs(ao, str(export_root))

    assert committed_destinations == [Path(path) for path in paths]
    assert serialization_destinations == replace_sources
    assert all(path.parent == destination.parent for path, destination in zip(
        serialization_destinations, committed_destinations, strict=True
    ))
    assert all(path.name.startswith(".tal-csv-") for path in serialization_destinations)
    assert all(not path.exists() for path in serialization_destinations)


def test_io_hard_p10b_020_csv_export_without_exportable_fields_fails_closed(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_020_csv_export_without_exportable_fields_fails_closed."""
    ds = xr.Dataset(
        coords={
            "trial": ["a"],
            "sample": [0, 1],
            "sequence_size": ("trial", [2]),
        }
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="sequence_size",
        validate=True,
    )
    root = tmp_path / "no_fields"

    with pytest.raises(ValueError, match="at least one exportable sequence field"):
        write_csv_logs(ao, str(root))

    assert not root.exists()


def test_io_hard_p10b_023_csv_export_rejects_empty_normalized_column_names(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_023_csv_export_rejects_empty_normalized_column_names."""
    ds = xr.Dataset(
        {"": (("trial", "sample"), [[1.0, 2.0]])},
        coords={
            "trial": ["a"],
            "sample": [0, 1],
            "sequence_size": ("trial", [2]),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        sequence_size_coord="sequence_size",
    )
    root = tmp_path / "empty_column"

    with pytest.raises(ValueError, match="CSV column names must be non-empty"):
        write_csv_logs(ao, str(root))

    assert not root.exists()


def test_io_hard_p10b_024_csv_export_wraps_lazy_size_read_failures_before_payload_execution(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_024_csv_export_wraps_lazy_size_read_failures_before_payload_execution."""
    calls: list[str] = []

    def load_size() -> np.ndarray:
        calls.append("size")
        raise OSError("backend size failure")

    def load_payload() -> np.ndarray:
        calls.append("payload")
        raise AssertionError("payload must remain lazy")

    size = da.from_delayed(delayed(load_size)(), shape=(1,), dtype=np.int64)
    payload = da.from_delayed(delayed(load_payload)(), shape=(1, 2), dtype=float)
    ds = xr.Dataset(
        {"value": (("trial", "sample"), payload)},
        coords={
            "trial": ["a"],
            "sample": [0, 1],
            "sequence_size": ("trial", [2]),
            "n_valid": ("trial", size),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        sequence_size_coord="sequence_size",
        validate=False,
    )
    root = tmp_path / "size_failure"

    with pytest.raises(ValueError, match="failed reading sequence_size_coord 'n_valid'"):
        write_csv_logs(
            ao,
            str(root),
            opts=CsvExportOptions(sequence_size_coord="n_valid"),
        )

    assert calls == ["size"]
    assert not root.exists()


def test_io_hard_p10b_031_csv_export_materializes_only_effective_lazy_size(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_031_csv_export_materializes_only_effective_lazy_size."""
    calls: list[str] = []

    def load_size(label: str, value: int) -> np.ndarray:
        calls.append(label)
        if label == "ignored":
            raise AssertionError("overridden schema size must remain lazy")
        return np.asarray([value], dtype=np.int64)

    lazy_default = da.from_delayed(
        delayed(load_size)("effective", 1),
        shape=(1,),
        dtype=np.int64,
    )
    default_paths = write_csv_logs(_lazy_size_export_ao(lazy_default), str(tmp_path / "default"))
    assert pd.read_csv(default_paths[0])["value"].tolist() == [1.0]
    assert calls == ["effective"]

    lazy_ignored = da.from_delayed(
        delayed(load_size)("ignored", 2),
        shape=(1,),
        dtype=np.int64,
    )
    override_paths = write_csv_logs(
        _lazy_size_export_ao(lazy_ignored, override=np.asarray([1], dtype=np.int64)),
        str(tmp_path / "override"),
        opts=CsvExportOptions(sequence_size_coord="effective_size"),
    )
    assert pd.read_csv(override_paths[0])["value"].tolist() == [1.0]
    assert calls == ["effective"]


def test_io_hard_p10b_036_csv_export_size_override_requires_real_numeric_values(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_036_csv_export_size_override_requires_real_numeric_values."""
    ao = _lazy_size_export_ao(
        np.asarray([2], dtype=np.int64),
        override=np.asarray(["1"], dtype=object),
    )
    root = tmp_path / "string_size"

    with pytest.raises(ValueError, match="value at flat index 0 is not numeric"):
        write_csv_logs(
            ao,
            str(root),
            opts=CsvExportOptions(sequence_size_coord="effective_size"),
        )

    assert not root.exists()


def test_io_hard_p10b_038_csv_export_fractional_real_size_does_not_round_to_integer(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_038_csv_export_fractional_real_size_does_not_round_to_integer."""
    fractional = Fraction(9_007_199_254_740_993, 9_007_199_254_740_992)
    ao = _lazy_size_export_ao(
        np.asarray([2], dtype=np.int64),
        override=np.asarray([fractional], dtype=object),
    )
    root = tmp_path / "fractional_size"

    with pytest.raises(ValueError, match="must be integer-valued"):
        write_csv_logs(
            ao,
            str(root),
            opts=CsvExportOptions(sequence_size_coord="effective_size"),
        )

    assert not root.exists()


def test_io_hard_p10b_040_csv_export_execution_errors_retain_public_owner(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_040_csv_export_execution_errors_retain_public_owner."""
    ordinary = _grouped_ao(
        labels=["format"],
        values=np.asarray([[1.2]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    format_root = tmp_path / "invalid_format"
    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: failed serializing CSV export rows",
    ) as format_error:
        write_csv_logs(
            ordinary,
            str(format_root),
            opts=CsvExportOptions(float_format="%q"),
        )
    assert isinstance(format_error.value.__cause__, ValueError)
    assert format_root.is_dir()
    assert not tuple(format_root.iterdir())

    def fail_payload() -> np.ndarray:
        raise OSError("payload backend exploded")

    payload = da.from_delayed(delayed(fail_payload)(), shape=(1, 1), dtype=float)
    lazy = _grouped_ao(
        labels=["payload"],
        values=payload,
        sizes=np.asarray([1], dtype=np.int64),
    )
    payload_root = tmp_path / "payload_failure"
    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: failed serializing CSV export rows",
    ) as payload_error:
        write_csv_logs(lazy, str(payload_root))
    assert isinstance(payload_error.value.__cause__, OSError)
    assert payload_root.is_dir()
    assert not tuple(payload_root.iterdir())


def test_io_hard_p10b_041_csv_final_commit_failure_preserves_current_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_041_csv_final_commit_failure_preserves_current_destination."""
    ao = _grouped_ao(
        labels=["a", "b"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    root = tmp_path / "atomic_commit"
    root.mkdir()
    first = root / "a.csv"
    second = root / "b.csv"
    first.write_text("first sentinel\n", encoding="utf-8")
    second.write_text("second sentinel\n", encoding="utf-8")
    original_replace = csv_commit_module.os.replace
    replace_count = 0

    def fail_second_replace(src: object, dst: object) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("disk full")
        original_replace(src, dst)

    monkeypatch.setattr(csv_commit_module.os, "replace", fail_second_replace)

    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: failed writing CSV export destination",
    ) as error:
        write_csv_logs(ao, str(root))

    assert isinstance(error.value.__cause__, OSError)
    assert pd.read_csv(first)["value"].tolist() == [1.0]
    assert second.read_text(encoding="utf-8") == "second sentinel\n"
    assert sorted(path.name for path in root.iterdir()) == ["a.csv", "b.csv"]


def test_io_hard_p10b_053_csv_staged_cleanup_handles_non_oserror_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_053_csv_staged_cleanup_handles_non_oserror_failures."""
    ao = _grouped_ao(
        labels=["a", "b"],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    stage_root = tmp_path / "runtime_serialization_failure"
    original_to_csv = pd.DataFrame.to_csv
    write_count = 0

    def fail_second_write(frame: pd.DataFrame, *args: object, **kwargs: object) -> object:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("serialization rejected output")
        return original_to_csv(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_second_write)
    with pytest.raises(ValueError, match="failed serializing CSV export rows") as error:
        write_csv_logs(ao, str(stage_root))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert not (stage_root / "a.csv").exists()
    assert not (stage_root / "b.csv").exists()
    assert not tuple(stage_root.glob(".tal-csv-*"))


def test_io_hard_p10b_043_csv_commit_has_no_post_replace_directory_cleanup(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_043_csv_commit_has_no_post_replace_directory_cleanup."""
    destination = tmp_path / "out" / "result.csv"
    destination.parent.mkdir()
    destination.write_text("sentinel\n", encoding="utf-8")
    staging = csv_commit_module.prepare_csv_commit(
        (str(destination),),
        preflight=_destination_preflight(destination),
        owner="tal.io.write_csv_logs",
    )
    Path(staging.temporary_paths[0]).write_text("value\n1.0\n", encoding="utf-8")
    assert destination.read_text(encoding="utf-8") == "sentinel\n"

    csv_commit_module.commit_csv_export(
        staging,
        owner="tal.io.write_csv_logs",
    )

    assert destination.read_text(encoding="utf-8") == "value\n1.0\n"
    assert sorted(path.name for path in destination.parent.iterdir()) == ["result.csv"]


def test_io_hard_p10b_049_csv_export_schema_failure_retains_public_owner(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_049_csv_export_schema_failure_retains_public_owner."""
    source = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    malformed = source.unsafe_data.copy(deep=False)
    schema = deepcopy(malformed.attrs["tal"])
    schema["core"]["roles"]["sequence_dim"] = "missing"
    malformed.attrs = {**malformed.attrs, "tal": schema}
    ao = AnalysisObject._from_unvalidated(malformed)

    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: invalid AnalysisObject schema payload",
    ) as error:
        write_csv_logs(ao, str(tmp_path / "invalid_schema"))

    assert isinstance(error.value.__cause__, SchemaError)


def test_io_hard_p10b_076_csv_external_dataset_schema_failure_retains_public_owner(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_076_csv_external_dataset_schema_failure_retains_public_owner."""
    malformed = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    ).unsafe_data.copy(deep=False)
    schema = deepcopy(malformed.attrs["tal"])
    schema["core"]["roles"]["sequence_dim"] = "missing"
    malformed.attrs = {**malformed.attrs, "tal": schema}

    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: invalid AnalysisObject schema payload",
    ) as error:
        write_csv_logs(malformed, str(tmp_path / "invalid_external_schema"))

    assert isinstance(error.value.__cause__, SchemaError)


def test_io_hard_p10b_078_csv_empty_batch_validates_export_root(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_078_csv_empty_batch_validates_export_root."""
    ao = _grouped_ao(
        labels=[],
        values=np.empty((0, 1), dtype=float),
        sizes=np.empty((0,), dtype=np.int64),
    )
    with pytest.raises(TypeError, match="tal.io.write_csv_logs: out_dir must be"):
        write_csv_logs(ao, None)  # type: ignore[arg-type]

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="export parent path .* is not a directory"):
        write_csv_logs(ao, str(blocked / "child"))


def test_io_hard_p10b_111_empty_batch_skips_lazy_validity_values(tmp_path: Path) -> None:
    """ID: IO_HARD_P10B_111_empty_batch_skips_lazy_validity_values."""
    calls: list[str] = []

    def load_sizes() -> np.ndarray:
        calls.append("size")
        raise AssertionError("empty export must not realize validity values")

    sizes = da.from_delayed(
        delayed(load_sizes)(),
        shape=(0,),
        dtype=np.int64,
    )
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.empty((0, 1), dtype=float))},
        coords={
            "trial": np.empty((0,), dtype=object),
            "sample": [0],
            "sequence_size": ("trial", sizes),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="sequence_size",
        validate=False,
    )
    root = tmp_path / "empty"

    assert write_csv_logs(ao, str(root)) == ()
    assert calls == []
    assert not root.exists()


def test_io_core_p10b_118_empty_object_size_override_remains_lazy(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10B_118_empty_object_size_override_remains_lazy."""
    calls: list[str] = []

    def load_sizes() -> np.ndarray:
        calls.append("size")
        raise AssertionError("empty export must not realize object size values")

    effective_size = da.from_delayed(
        delayed(load_sizes)(),
        shape=(0,),
        dtype=object,
    )
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.empty((0, 1), dtype=float))},
        coords={
            "trial": np.empty((0,), dtype=object),
            "sample": [0],
            "sequence_size": ("trial", np.empty((0,), dtype=np.int64)),
            "effective_size": ("trial", effective_size),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="sequence_size",
        validate=False,
    )
    root = tmp_path / "object-empty"

    assert write_csv_logs(
        ao,
        str(root),
        opts=CsvExportOptions(sequence_size_coord="effective_size"),
    ) == ()
    assert calls == []
    assert not root.exists()


@pytest.mark.parametrize("dtype", [np.dtype("U1"), np.dtype(bool), np.dtype(complex)])
def test_io_hard_p10b_117_empty_batch_validates_effective_size_dtype_without_execution(
    tmp_path: Path,
    dtype: np.dtype[object],
) -> None:
    """ID: IO_HARD_P10B_117_empty_batch_validates_effective_size_dtype_without_execution."""
    calls: list[str] = []

    def load_sizes() -> np.ndarray:
        calls.append("size")
        raise AssertionError("structural dtype validation must not realize size values")

    effective_size = da.from_delayed(
        delayed(load_sizes)(),
        shape=(0,),
        dtype=dtype,
    )
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.empty((0, 1), dtype=float))},
        coords={
            "trial": np.empty((0,), dtype=object),
            "sample": [0],
            "sequence_size": ("trial", np.empty((0,), dtype=np.int64)),
            "effective_size": ("trial", effective_size),
        },
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="sequence_size",
        validate=False,
    )
    root = tmp_path / "invalid-empty"

    with pytest.raises(ValueError, match="values must contain real numeric values"):
        write_csv_logs(
            ao,
            str(root),
            opts=CsvExportOptions(sequence_size_coord="effective_size"),
        )

    assert calls == []
    assert not root.exists()


def test_io_hard_p10b_079_csv_external_dataset_layout_failure_retains_public_owner(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_079_csv_external_dataset_layout_failure_retains_public_owner."""
    index = pd.MultiIndex.from_product([["run"], [0]], names=["trial", "sample"])
    coords = xr.Coordinates.from_pandas_multiindex(index, "stacked")
    unsupported = xr.Dataset({"value": ("stacked", [1.0])}, coords=coords)

    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: invalid xarray AnalysisObject input",
    ) as error:
        write_csv_logs(unsupported, str(tmp_path / "unsupported_layout"))

    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__).startswith("AnalysisObject: PandasMultiIndex")


def test_io_hard_p10b_074_csv_export_preflight_does_not_reconstruct_subclass(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_074_csv_export_preflight_does_not_reconstruct_subclass."""

    class LifecycleProbeAO(AnalysisObject):
        bind_count = 0

        def _after_bind_dataset(self) -> None:
            type(self).bind_count += 1

    source = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    ao = LifecycleProbeAO(source.unsafe_data)
    LifecycleProbeAO.bind_count = 0

    paths = write_csv_logs(ao, str(tmp_path / "subclass"))

    assert LifecycleProbeAO.bind_count == 0
    assert pd.read_csv(paths[0])["value"].tolist() == [1.0]


def test_io_hard_p10b_042_csv_export_identity_conversion_failures_retain_owner(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_042_csv_export_identity_conversion_failures_retain_owner."""
    bad_label = _grouped_ao(
        labels=[_FailingStringIdentity()],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    bad_field = _ao_with_field_name(_FailingStringIdentity())

    for ao, detail in (
        (bad_label, "export label"),
        (bad_field, "CSV column name"),
    ):
        root = tmp_path / detail.replace(" ", "_").lower()
        with pytest.raises(
            ValueError,
            match=rf"tal.io.write_csv_logs: failed normalizing {detail} to a string",
        ) as error:
            write_csv_logs(ao, str(root))
        assert isinstance(error.value.__cause__, RuntimeError)
        assert not root.exists()


def test_io_hard_p10b_105_csv_export_diagnostics_avoid_raw_identity_repr(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_105_csv_export_diagnostics_avoid_raw_identity_repr."""
    duplicate = _FailingReprIdentity("duplicate")
    duplicate_labels = _grouped_ao(
        labels=[duplicate, duplicate],
        values=np.asarray([[1.0], [2.0]], dtype=float),
        sizes=np.asarray([1, 1], dtype=np.int64),
    )
    empty_field = _ao_with_field_name(_FailingReprIdentity(""))

    for ao, detail in (
        (duplicate_labels, "duplicate batch label 'duplicate'"),
        (empty_field, "CSV column names must be non-empty"),
    ):
        root = tmp_path / str(len(detail))
        with pytest.raises(ValueError, match=detail) as error:
            write_csv_logs(ao, str(root))
        assert str(error.value).startswith("tal.io.write_csv_logs:")
        assert not root.exists()


@pytest.mark.parametrize("field_name", ["\x00", "value\x00hidden"])
def test_io_hard_p10b_044_csv_export_nul_headers_fail_closed(
    tmp_path: Path,
    field_name: str,
) -> None:
    """ID: IO_HARD_P10B_044_csv_export_nul_headers_fail_closed."""
    root = tmp_path / f"nul_{len(field_name)}"

    with pytest.raises(ValueError, match="CSV column name .* must not contain NUL bytes"):
        write_csv_logs(_ao_with_field_name(field_name), str(root))

    assert not root.exists()


@pytest.mark.parametrize(
    ("field_name", "match"),
    [
        pytest.param(" ", "must contain a non-whitespace character", id="space"),
        pytest.param("\t", "must contain a non-whitespace character", id="tab"),
        pytest.param("\r\n", "must contain a non-whitespace character", id="newline"),
        pytest.param("\ufeffvalue", "must not contain BOM characters", id="leading-bom"),
        pytest.param("value\ufeffhidden", "must not contain BOM characters", id="embedded-bom"),
    ],
)
def test_io_hard_p10b_057_csv_export_parser_unsafe_headers_fail_closed(
    tmp_path: Path,
    field_name: str,
    match: str,
) -> None:
    """ID: IO_HARD_P10B_057_csv_export_parser_unsafe_headers_fail_closed."""
    root = tmp_path / f"blank_{len(field_name)}"

    with pytest.raises(ValueError, match=match):
        write_csv_logs(_ao_with_field_name(field_name), str(root))

    assert not root.exists()


@pytest.mark.parametrize(
    "size",
    [
        pytest.param(_FailingSizeFloat(1.0), id="float"),
        pytest.param(_FailingSizeComparison(1.0), id="comparison"),
        pytest.param(_FailingSizeFraction(1, 1), id="rational"),
    ],
)
def test_io_hard_p10b_058_csv_object_size_failures_retain_writer_owner(
    tmp_path: Path,
    size: object,
) -> None:
    """ID: IO_HARD_P10B_058_csv_object_size_failures_retain_writer_owner."""
    root = tmp_path / type(size).__name__
    ao = _lazy_size_export_ao(
        np.asarray([1], dtype=np.int64),
        override=np.asarray([size], dtype=object),
    )

    with pytest.raises(
        ValueError,
        match=(
            "tal.io.write_csv_logs: failed inspecting sequence_size_coord "
            "'effective_size' value at flat index 0"
        ),
    ) as error:
        write_csv_logs(
            ao,
            str(root),
            opts=CsvExportOptions(sequence_size_coord="effective_size"),
        )

    assert isinstance(error.value.__cause__, RuntimeError)
    assert not root.exists()


def test_io_hard_p10b_085_invalid_export_root_precedes_lazy_size_execution() -> None:
    """ID: IO_HARD_P10B_085_invalid_export_root_precedes_lazy_size_execution."""
    calls: list[str] = []

    def load_size() -> np.ndarray:
        calls.append("size")
        raise AssertionError("invalid out_dir must fail before size execution")

    size = da.from_delayed(delayed(load_size)(), shape=(1,), dtype=np.int64)

    with pytest.raises(TypeError, match="tal.io.write_csv_logs: out_dir must be"):
        write_csv_logs(_lazy_size_export_ao(size), None)  # type: ignore[arg-type]

    assert calls == []


def test_io_hard_p10b_086_destination_types_precede_lazy_size_execution(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_086_destination_types_precede_lazy_size_execution."""
    calls: list[str] = []

    def load_size() -> np.ndarray:
        calls.append("size")
        raise AssertionError("invalid destination must fail before size execution")

    size = da.from_delayed(delayed(load_size)(), shape=(1,), dtype=np.int64)
    root = tmp_path / "destination_type"
    (root / "a.csv").mkdir(parents=True)

    with pytest.raises(ValueError, match="export destination .* is an existing directory"):
        write_csv_logs(_lazy_size_export_ao(size), str(root))

    assert calls == []


def test_io_hard_p10b_087_field_lowering_failure_retains_writer_owner(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_087_field_lowering_failure_retains_writer_owner."""
    from xarray.backends import BackendArray
    from xarray.core import indexing

    class FailingFieldBackend(BackendArray):
        shape = (1, 1)
        dtype = np.dtype(float)

        def __getitem__(self, key: object) -> np.ndarray:
            return indexing.explicit_indexing_adapter(
                key,
                self.shape,
                indexing.IndexingSupport.BASIC,
                self._raw_indexing_method,
            )

        @staticmethod
        def _raw_indexing_method(key: object) -> np.ndarray:
            _ = key
            raise RuntimeError("field indexing exploded")

    base = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    ds = base.unsafe_data.copy(deep=False)
    ds["value"] = xr.Variable(
        ("trial", "sample"),
        indexing.LazilyIndexedArray(FailingFieldBackend()),
    )
    ao = AnalysisObject._from_unvalidated(ds)
    root = tmp_path / "field_failure"

    with pytest.raises(
        ValueError,
        match="tal.io.write_csv_logs: failed serializing CSV export rows",
    ) as error:
        write_csv_logs(ao, str(root))

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "field indexing exploded"
    assert root.is_dir()
    assert not tuple(root.iterdir())


@pytest.mark.parametrize("as_dataarray", [False, True], ids=["dataset", "dataarray"])
def test_io_hard_p10b_088_external_source_override_defers_declared_validity(
    tmp_path: Path,
    as_dataarray: bool,
) -> None:
    """ID: IO_HARD_P10B_088_external_source_override_defers_declared_validity."""
    calls: list[str] = []

    def load_declared_size() -> np.ndarray:
        calls.append("declared")
        raise AssertionError("overridden declared validity must remain lazy")

    declared = da.from_delayed(
        delayed(load_declared_size)(),
        shape=(1,),
        dtype=np.int64,
    )
    external = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0, 2.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    ).unsafe_data.assign_coords(
        sequence_size=("trial", declared),
        effective_size=("trial", np.asarray([1], dtype=np.int64)),
    )
    source: xr.Dataset | xr.DataArray = external
    if as_dataarray:
        source = external["value"].copy(deep=False)
        source.attrs = {**source.attrs, "tal": deepcopy(external.attrs["tal"])}

    paths = write_csv_logs(
        source,
        str(tmp_path / "external_override"),
        opts=CsvExportOptions(sequence_size_coord="effective_size"),
    )

    assert calls == []
    assert pd.read_csv(paths[0])["value"].tolist() == [1.0]


@pytest.mark.parametrize("declared_state", ["missing", "invalid-dtype"])
def test_io_hard_p10b_101_explicit_size_override_wins_before_schema_validity_validation(
    tmp_path: Path,
    declared_state: str,
) -> None:
    """ID: IO_HARD_P10B_101_explicit_size_override_wins_before_schema_validity_validation."""
    external = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0, 2.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    ).unsafe_data.assign_coords(
        effective_size=("trial", np.asarray([1], dtype=np.int64)),
    )
    if declared_state == "missing":
        external = external.drop_vars("sequence_size")
    else:
        external = external.assign_coords(sequence_size=("trial", ["invalid"]))

    paths = write_csv_logs(
        external,
        str(tmp_path / declared_state),
        opts=CsvExportOptions(sequence_size_coord="effective_size"),
    )

    assert pd.read_csv(paths[0])["value"].tolist() == [1.0]


@pytest.mark.parametrize(
    "validity_block",
    [
        "invalid",
        {"unknown": "field"},
        {"sequence_size_coord": "sequence_size", "layout": "invalid"},
        {"sequence_size_coord": 42, "layout": "left_packed"},
    ],
    ids=("not-mapping", "unknown-key", "invalid-layout", "invalid-name"),
)
def test_io_hard_p10b_113_explicit_size_override_supersedes_malformed_validity_block(
    tmp_path: Path,
    validity_block: object,
) -> None:
    """ID: IO_HARD_P10B_113_explicit_size_override_supersedes_malformed_validity_block."""
    source = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0, 2.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    ).unsafe_data.assign_coords(
        effective_size=("trial", np.asarray([1], dtype=np.int64)),
    )
    source.attrs = deepcopy(source.attrs)
    source.attrs["tal"]["core"]["validity"] = validity_block

    paths = write_csv_logs(
        source,
        str(tmp_path / "malformed-validity"),
        opts=CsvExportOptions(sequence_size_coord="effective_size"),
    )

    assert pd.read_csv(paths[0])["value"].tolist() == [1.0]


def test_io_hard_p10b_114_explicit_size_override_does_not_mask_other_schema_errors(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_114_explicit_size_override_does_not_mask_other_schema_errors."""
    source = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    ).unsafe_data.assign_coords(
        effective_size=("trial", np.asarray([1], dtype=np.int64)),
    )
    source.attrs = deepcopy(source.attrs)
    source.attrs["tal"]["core"]["roles"]["unknown"] = "field"

    with pytest.raises(ValueError, match="invalid AnalysisObject schema payload"):
        write_csv_logs(
            source,
            str(tmp_path / "malformed-roles"),
            opts=CsvExportOptions(sequence_size_coord="effective_size"),
        )


@pytest.mark.parametrize(
    "validity_block",
    [
        {"sequence_size_coord": "sequence_size", "layout": "invalid"},
        {
            "sequence_size_coord": "sequence_size",
            "layout": "left_packed",
            "unknown": "field",
        },
    ],
    ids=("invalid-layout", "unknown-key"),
)
def test_io_hard_p10b_115_malformed_override_omits_declared_validity_field(
    tmp_path: Path,
    validity_block: dict[str, str],
) -> None:
    """ID: IO_HARD_P10B_115_malformed_override_omits_declared_validity_field."""
    calls: list[str] = []

    def load_declared_size() -> np.ndarray:
        calls.append("declared")
        raise AssertionError("malformed overridden validity must remain lazy")

    declared = da.from_delayed(
        delayed(load_declared_size)(),
        shape=(2,),
        dtype=np.int64,
    )
    source = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0, 2.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    ).unsafe_data.assign_coords(
        sequence_size=("sample", declared),
        effective_size=("trial", np.asarray([1], dtype=np.int64)),
    )
    source.attrs = deepcopy(source.attrs)
    source.attrs["tal"]["core"]["validity"] = validity_block

    paths = write_csv_logs(
        source,
        str(tmp_path / "malformed-override-projection"),
        opts=CsvExportOptions(sequence_size_coord="effective_size"),
    )
    frame = pd.read_csv(paths[0])

    assert calls == []
    assert "sequence_size" not in frame.columns
    assert frame["value"].tolist() == [1.0]


def test_io_hard_p10b_104_explicit_size_override_omits_declared_validity_field(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_104_explicit_size_override_omits_declared_validity_field."""
    calls: list[str] = []

    def load_declared_size() -> np.ndarray:
        calls.append("declared")
        raise AssertionError("overridden validity must not enter the export projection")

    declared = da.from_delayed(
        delayed(load_declared_size)(),
        shape=(2,),
        dtype=np.int64,
    )
    external = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0, 2.0]], dtype=float),
        sizes=np.asarray([2], dtype=np.int64),
    ).unsafe_data.assign_coords(
        sequence_size=("sample", declared),
        effective_size=("trial", np.asarray([1], dtype=np.int64)),
    )

    paths = write_csv_logs(
        external,
        str(tmp_path / "override_projection"),
        opts=CsvExportOptions(sequence_size_coord="effective_size"),
    )
    frame = pd.read_csv(paths[0])

    assert calls == []
    assert frame["value"].tolist() == [1.0]
    assert "sequence_size" not in frame.columns


def test_io_hard_p10b_110_explicit_override_preserves_surviving_parameter(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_110_explicit_override_preserves_surviving_parameter."""
    source = _dimension_param_export_ao(include_value=True).unsafe_data.assign_coords(
        effective_size=("trial", np.asarray([2], dtype=np.int64)),
    )
    source = set_validity(
        source,
        sequence_size_coord="time",
        layout="left_packed",
        validate=False,
    )

    paths = write_csv_logs(
        AnalysisObject._from_unvalidated(source),
        str(tmp_path / "surviving_parameter"),
        opts=CsvExportOptions(sequence_size_coord="effective_size"),
    )
    frame = pd.read_csv(paths[0])

    assert list(frame.columns) == ["time", "value"]
    assert frame["time"].tolist() == [10, 20]
    assert frame["value"].tolist() == [1.0, 2.0]


def test_io_perf_p10b_014_external_source_coercion_shares_payload_buffers() -> None:
    """ID: IO_PERF_P10B_014_external_source_coercion_shares_payload_buffers."""
    payload = np.arange(256, dtype=float).reshape(1, 256)
    external = _grouped_ao(
        labels=["run"],
        values=payload,
        sizes=np.asarray([256], dtype=np.int64),
    ).unsafe_data

    coerced = csv_export_module.coerce_csv_export_source(
        external,
        owner="tal.io.write_csv_logs",
    )

    assert coerced.unsafe_data is not external
    assert np.shares_memory(
        coerced.unsafe_data["value"].data,
        external["value"].data,
    )


def test_io_perf_p10b_015_csv_backend_rows_spool_without_full_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_015_csv_backend_rows_spool_without_full_projection."""
    from xarray.backends import BackendArray
    from xarray.core import indexing

    events: list[tuple[str, str]] = []

    class TrackingBackend(BackendArray):
        def __init__(self, values: object, name: str) -> None:
            self.values = np.asarray(values, dtype=float)
            self.shape = self.values.shape
            self.dtype = self.values.dtype
            self.name = name

        def __getitem__(self, key: object) -> np.ndarray:
            return indexing.explicit_indexing_adapter(
                key,
                self.shape,
                indexing.IndexingSupport.BASIC,
                self._raw_indexing_method,
            )

        def _raw_indexing_method(self, key: object) -> np.ndarray:
            events.append(("read", self.name))
            return self.values[key]

    base = _grouped_ao(
        labels=["a", "b", "c"],
        values=np.ones((3, 3), dtype=float),
        sizes=np.asarray([1, 2, 3], dtype=np.int64),
    )
    ds = base.unsafe_data.copy(deep=False)
    ds["value"] = xr.Variable(
        ("trial", "sample"),
        indexing.LazilyIndexedArray(TrackingBackend(np.arange(9).reshape(3, 3), "value")),
    )
    ds = ds.assign_coords(
        clock=xr.Variable(
            ("sample",),
            indexing.LazilyIndexedArray(TrackingBackend([10, 20, 30], "clock")),
        )
    )
    original_to_csv = pd.DataFrame.to_csv

    def tracked_to_csv(frame: pd.DataFrame, *args: object, **kwargs: object) -> object:
        events.append(("write", str(len(frame))))
        return original_to_csv(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", tracked_to_csv)
    paths = write_csv_logs(AnalysisObject._from_unvalidated(ds), str(tmp_path / "backend"))

    assert [event for event in events if event == ("read", "clock")] == [("read", "clock")]
    assert sum(event == ("read", "value") for event in events) == 3
    assert sorted(event for event in events if event[0] == "write") == [
        ("write", "1"),
        ("write", "2"),
        ("write", "3"),
    ]
    assert [pd.read_csv(path)["clock"].tolist() for path in paths] == [
        [10.0],
        [10.0, 20.0],
        [10.0, 20.0, 30.0],
    ]


def test_io_hard_p10b_097_csv_zero_valid_length_skips_backend_payload(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_097_csv_zero_valid_length_skips_backend_payload."""
    from xarray.backends import BackendArray
    from xarray.core import indexing

    reads: list[object] = []

    class InvalidTailBackend(BackendArray):
        shape = (1, 1)
        dtype = np.dtype(float)

        def __getitem__(self, key: object) -> np.ndarray:
            return indexing.explicit_indexing_adapter(
                key,
                self.shape,
                indexing.IndexingSupport.BASIC,
                self._raw_indexing_method,
            )

        @staticmethod
        def _raw_indexing_method(key: object) -> np.ndarray:
            reads.append(key)
            raise AssertionError("zero-length payload must not execute")

    base = _grouped_ao(
        labels=["run"],
        values=np.asarray([[1.0]], dtype=float),
        sizes=np.asarray([0], dtype=np.int64),
    )
    ds = base.unsafe_data.copy(deep=False)
    ds["value"] = xr.Variable(
        ("trial", "sample"),
        indexing.LazilyIndexedArray(InvalidTailBackend()),
    )

    paths = write_csv_logs(AnalysisObject._from_unvalidated(ds), str(tmp_path / "zero"))

    assert reads == []
    assert pd.read_csv(paths[0]).empty
    assert pd.read_csv(paths[0]).columns.tolist() == ["time", "value"]


def test_io_hard_p10b_099_csv_commit_uses_short_staging_name(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10B_099_csv_commit_uses_short_staging_name."""
    try:
        name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    except (AttributeError, OSError, ValueError):
        pytest.skip("filesystem basename limit is unavailable")
    if name_max <= len(".csv"):
        pytest.skip("filesystem basename limit cannot represent a CSV filename")
    label = "a" * (name_max - len(".csv"))
    ao = _grouped_ao(
        labels=[label],
        values=np.asarray([[1.0]]),
        sizes=np.ones(1, dtype=np.int64),
    )

    paths = write_csv_logs(ao, str(tmp_path / "long-name"))

    assert Path(paths[0]).name == f"{label}.csv"
    assert pd.read_csv(paths[0])["value"].tolist() == [1.0]



def test_io_perf_p10b_012_csv_object_sizes_fill_one_final_int64_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10B_012_csv_object_sizes_fill_one_final_int64_buffer."""
    size = xr.DataArray(
        np.asarray([[1, Fraction(2, 1)], [np.int64(3), 4.0]], dtype=object),
        dims=("left", "right"),
    )

    normalized = csv_sizes_module.normalize_csv_export_sizes(
        size,
        width=4,
        size_name="n_valid",
        owner="tal.io.write_csv_logs",
    )

    assert normalized.shape == size.shape
    assert normalized.dtype == np.dtype("int64")
    np.testing.assert_array_equal(normalized, [[1, 2], [3, 4]])

    class TrackedInteger:
        alive = 0
        peak = 0

        def __init__(self, value: int) -> None:
            self.value = value
            type(self).alive += 1
            type(self).peak = max(type(self).peak, type(self).alive)

        def __int__(self) -> int:
            return self.value

        def __index__(self) -> int:
            return self.value

        def __del__(self) -> None:
            type(self).alive -= 1

    def tracked_size(value: object, **_kwargs: object) -> TrackedInteger:
        return TrackedInteger(int(value))

    monkeypatch.setattr(csv_sizes_module, "_coerce_object_size", tracked_size)
    probe = xr.DataArray(np.arange(100, dtype=object), dims=("trial",))
    csv_sizes_module._coerce_object_sizes(
        probe,
        width=100,
        size_name="n_valid",
        owner="tal.io.write_csv_logs",
    )

    assert TrackedInteger.alive == 0
    assert TrackedInteger.peak <= 2


@pytest.mark.parametrize("float_format", ["%d", "%.f"])
def test_io_core_p10b_021_csv_export_accepts_pandas_float_formats(
    tmp_path: Path,
    float_format: str,
) -> None:
    """ID: IO_CORE_P10B_021_csv_export_accepts_pandas_float_formats."""
    ao = _grouped_ao(
        labels=["format"],
        values=np.asarray([[1.2]], dtype=float),
        sizes=np.asarray([1], dtype=np.int64),
    )
    paths = write_csv_logs(
        ao,
        str(tmp_path / float_format.replace("%", "format_")),
        opts=CsvExportOptions(float_format=float_format),
    )
    assert pd.read_csv(paths[0])["value"].tolist() == [1]
