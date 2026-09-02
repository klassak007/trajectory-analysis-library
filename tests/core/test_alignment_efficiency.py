from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal import ufuncs as tal_ufuncs
from tal.core import AnalysisObject


def _ao(
    marker: str,
    *,
    sample_labels: Sequence[object],
    trial_labels: Sequence[object] = ("t0", "t1"),
    param_values: Sequence[float] | None = None,
    chunked: bool = False,
) -> AnalysisObject:
    shape = (len(sample_labels), len(trial_labels))
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    if marker == "right":
        values = values + 10.0
    payload = da.from_array(values, chunks=(max(1, shape[0] // 2), shape[1])) if chunked else values
    sample = np.asarray(sample_labels)
    trial = np.asarray(trial_labels)
    coords: dict[str, object] = {"sample": sample, "trial": trial}
    if param_values is not None:
        coords["time_s"] = ("sample", np.asarray(param_values, dtype=np.float64))
    data = xr.DataArray(
        payload,
        dims=("sample", "trial"),
        coords=coords,
        attrs={"test_operand": marker},
        name="value",
    )
    return AnalysisObject.from_data(
        data.to_dataset(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s" if param_values is not None else None,
        validate=True,
    )


def _canonical_param_ao(
    marker: str,
    *,
    sample_labels: Sequence[object],
    trial_labels: Sequence[object] = ("t0", "t1"),
) -> AnalysisObject:
    params = tuple(np.linspace(0.0, 1.0, len(sample_labels)))
    source = _ao(
        marker,
        sample_labels=sample_labels,
        trial_labels=trial_labels,
        param_values=params,
    )
    partner = _ao(
        f"{marker}_partner",
        sample_labels=sample_labels,
        trial_labels=trial_labels,
        param_values=params,
    )
    out = tal_ufuncs.add(source.a(on="param", sequence_join=None), partner)
    assert isinstance(out, AnalysisObject)
    return out


def _unindexed_param_ao(
    marker: str,
    *,
    coord_order: tuple[str, ...],
    sample_labels: Sequence[object] = (10, 11, 12),
    trial_labels: Sequence[object] = ("t0", "t1"),
) -> AnalysisObject:
    shape = (len(sample_labels), len(trial_labels))
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    if marker == "right":
        values = values + 10.0
    definitions: dict[str, object] = {
        "sample": ("sample", np.asarray(sample_labels)),
        "trial": ("trial", np.asarray(trial_labels)),
        "time_s": ("sample", np.linspace(0.0, 1.0, len(sample_labels))),
    }
    coords = xr.Coordinates(
        {name: definitions[name] for name in coord_order},
        indexes={},
    )
    data = xr.DataArray(
        values,
        dims=("sample", "trial"),
        coords=coords,
        attrs={"test_operand": marker},
        name="value",
    )
    return AnalysisObject.from_data(
        data.to_dataset(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
        validate=True,
    )


class _OffsetCoordinateTransform(xr.indexes.CoordinateTransform):
    def __init__(self, size: int, *, offset: float) -> None:
        self.offset = offset
        super().__init__(("sample",), {"sample": size})

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        return {"sample": dim_positions["sample"] + self.offset}

    def reverse(self, coord_labels: dict[str, Any]) -> dict[str, Any]:
        return {"sample": coord_labels["sample"] - self.offset}

    def equals(self, other: object, **kwargs: object) -> bool:
        return (
            isinstance(other, _OffsetCoordinateTransform)
            and self.dim_size == other.dim_size
            and self.offset == other.offset
        )


def _coordinate_transform_param_ao(marker: str, *, offset: float) -> AnalysisObject:
    index = xr.indexes.CoordinateTransformIndex(
        _OffsetCoordinateTransform(3, offset=offset)
    )
    coords = xr.Coordinates.from_xindex(index).assign(
        time_s=("sample", np.asarray([0.0, 0.5, 1.0])),
        trial=("trial", np.asarray(["t0", "t1"])),
    )
    values = np.arange(6.0).reshape(3, 2)
    if marker == "right":
        values = values + 10.0
    data = xr.DataArray(
        values,
        dims=("sample", "trial"),
        coords=coords,
        attrs={"test_operand": marker},
        name="value",
    )
    return AnalysisObject.from_data(
        data.to_dataset(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="time_s",
        validate=True,
    )


def _record_indexing(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    calls: list[tuple[str, str | None]] = []
    original_sel = xr.DataArray.sel
    original_reindex = xr.DataArray.reindex

    def tracked_sel(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("sel", self.attrs.get("test_operand")))
        return original_sel(self, *args, **kwargs)

    def tracked_reindex(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("reindex", self.attrs.get("test_operand")))
        return original_reindex(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "sel", tracked_sel)
    monkeypatch.setattr(xr.DataArray, "reindex", tracked_reindex)
    return calls


def _sum_with_sequence_join(left: AnalysisObject, right: AnalysisObject, *, join: str) -> AnalysisObject:
    hinted = left.a(on="sequence", sequence_join=join, batch_join="exact")
    out = tal_ufuncs.add(hinted, right)
    assert isinstance(out, AnalysisObject)
    return out


def _expected_sum(left: AnalysisObject, right: AnalysisObject, *, join: str) -> xr.DataArray:
    left_data = left.as_dataset(copy="none")["value"]
    right_data = right.as_dataset(copy="none")["value"]
    aligned = xr.align(left_data, right_data, join=join, copy=False)
    return aligned[0] + aligned[1]


def _expected_param_sum(
    left: AnalysisObject,
    right: AnalysisObject,
    *,
    batch_join: str,
) -> xr.DataArray:
    arrays = tuple(item.as_dataset(copy="none")["value"] for item in (left, right))
    target_param = arrays[0].coords["time_s"].to_index()
    try:
        canonical_sequence = arrays[0].get_index("sample")
    except (KeyError, TypeError, ValueError):
        canonical_sequence = pd.RangeIndex(arrays[0].sizes["sample"], name="sample")
    normalized: list[xr.DataArray] = []
    for array in arrays:
        selected = array.swap_dims({"sample": "time_s"}).sel(time_s=target_param)
        restored = selected.drop_vars("sample").rename({"time_s": "sample"})
        normalized.append(
            restored.assign_coords(
                sample=canonical_sequence,
                time_s=("sample", target_param),
            )
        )
    aligned = xr.align(*normalized, join=batch_join, copy=False)
    return aligned[0] + aligned[1]


def test_align_perf_001_exact_sequence_and_batch_skip_identity_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_PERF_001_exact_sequence_and_batch_skip_identity_indexing."""
    left = _ao("left", sample_labels=(0, 1, 2))
    right = _ao("right", sample_labels=(0, 1, 2))
    calls = _record_indexing(monkeypatch)

    out = _sum_with_sequence_join(left, right, join="exact")

    assert calls == []
    xr.testing.assert_equal(out.as_dataset(copy="none")["value"], _expected_sum(left, right, join="exact"))


@pytest.mark.parametrize(
    ("join", "left_labels", "right_labels", "expected_call"),
    [
        ("inner", (1, 2), (0, 1, 2), ("sel", "right")),
        ("outer", (0, 1, 2), (1, 2), ("reindex", "right")),
        ("left", (0, 1, 2), (1, 2), ("reindex", "right")),
        ("right", (1, 2), (0, 1, 2), ("reindex", "left")),
    ],
)
def test_align_perf_002_join_transforms_only_mismatched_operand(
    monkeypatch: pytest.MonkeyPatch,
    join: str,
    left_labels: tuple[int, ...],
    right_labels: tuple[int, ...],
    expected_call: tuple[str, str],
) -> None:
    """ID: ALIGN_PERF_002_join_transforms_only_mismatched_operand."""
    left = _ao("left", sample_labels=left_labels)
    right = _ao("right", sample_labels=right_labels)
    calls = _record_indexing(monkeypatch)

    out = _sum_with_sequence_join(left, right, join=join)

    assert calls == [expected_call]
    xr.testing.assert_equal(out.as_dataset(copy="none")["value"], _expected_sum(left, right, join=join))


def test_align_perf_003_batch_join_transforms_only_mismatched_operand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_PERF_003_batch_join_transforms_only_mismatched_operand."""
    left = _ao("left", sample_labels=(0, 1), trial_labels=("t0", "t1", "t2"))
    right = _ao("right", sample_labels=(0, 1), trial_labels=("t1", "t2"))
    calls = _record_indexing(monkeypatch)
    hinted = left.a(on="sequence", sequence_join="exact", batch_join="outer")

    out = tal_ufuncs.add(hinted, right)

    assert isinstance(out, AnalysisObject)
    assert calls == [("reindex", "right")]
    xr.testing.assert_equal(out.as_dataset(copy="none")["value"], _expected_sum(left, right, join="outer"))


def test_align_hard_001_equal_size_unequal_exact_labels_still_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_HARD_001_equal_size_unequal_exact_labels_still_fail."""
    left = _ao("left", sample_labels=(0, 1, 2))
    right = _ao("right", sample_labels=(1, 2, 3))
    calls = _record_indexing(monkeypatch)

    with pytest.raises(ValueError, match=r"^tal\.ufuncs\.add: .*requires exact 'sample' labels"):
        _sum_with_sequence_join(left, right, join="exact")

    assert calls == []


def test_align_perf_004_canonical_param_operands_skip_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_PERF_004_canonical_param_operands_skip_normalization."""
    left = _canonical_param_ao(
        "left",
        sample_labels=(0, 1, 2),
    )
    right = _canonical_param_ao(
        "right",
        sample_labels=(0, 1, 2),
    )
    expected = _expected_param_sum(left, right, batch_join="exact")
    calls = _record_indexing(monkeypatch)
    hinted = left.a(on="param", sequence_join=None, batch_join="exact")

    out = tal_ufuncs.add(hinted, right)

    assert isinstance(out, AnalysisObject)
    assert calls == []
    actual = out.as_dataset(copy="none")["value"]
    assert tuple(actual.coords) == ("trial", "sample", "time_s")
    assert tuple(actual.xindexes) == ("trial", "sample")
    xr.testing.assert_identical(actual, expected)


def test_align_perf_005_param_normalizes_only_noncanonical_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_PERF_005_param_normalizes_only_noncanonical_sequence."""
    left = _canonical_param_ao(
        "left",
        sample_labels=(0, 1, 2),
    )
    right = _canonical_param_ao(
        "right",
        sample_labels=("a", "b", "c"),
    )
    expected = _expected_param_sum(left, right, batch_join="exact")
    calls = _record_indexing(monkeypatch)
    hinted = left.a(on="param", sequence_join=None, batch_join="exact")

    out = tal_ufuncs.add(hinted, right)

    assert isinstance(out, AnalysisObject)
    assert [method for method, _ in calls] == ["sel"]
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["sample"], np.asarray([0, 1, 2]))
    xr.testing.assert_equal(out.as_dataset(copy="none")["value"], expected)


def test_align_perf_006_param_coord_order_and_batch_target_are_canonical_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_PERF_006_param_coord_order_and_batch_target_are_canonical_state."""
    left = _canonical_param_ao(
        "left",
        sample_labels=(0, 1, 2),
        trial_labels=("t0", "t1", "t2"),
    )
    right = _ao(
        "right",
        sample_labels=(0, 1, 2),
        trial_labels=("t1", "t2"),
        param_values=(0.0, 0.5, 1.0),
    )
    expected = _expected_param_sum(left, right, batch_join="outer")
    calls = _record_indexing(monkeypatch)
    hinted = left.a(on="param", sequence_join=None, batch_join="outer")

    out = tal_ufuncs.add(hinted, right)

    assert isinstance(out, AnalysisObject)
    assert [method for method, _ in calls] == ["sel", "reindex"]
    assert tuple(out.as_dataset(copy="none")["value"].coords) == ("trial", "sample", "time_s")
    xr.testing.assert_equal(out.as_dataset(copy="none")["value"], expected)


def test_align_hard_002_unindexed_sequence_is_not_param_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_HARD_002_unindexed_sequence_is_not_param_canonical."""
    left = _unindexed_param_ao(
        "left",
        coord_order=("trial", "sample", "time_s"),
    )
    right = _unindexed_param_ao(
        "right",
        coord_order=("trial", "sample", "time_s"),
    )
    expected = _expected_param_sum(left, right, batch_join="exact")
    calls = _record_indexing(monkeypatch)

    out = tal_ufuncs.add(left.a(on="param", sequence_join=None), right)

    assert isinstance(out, AnalysisObject)
    assert [method for method, _ in calls] == ["sel", "sel"]
    actual = out.as_dataset(copy="none")["value"]
    np.testing.assert_array_equal(actual.coords["sample"], np.asarray([0, 1, 2]))
    assert tuple(actual.xindexes) == ("trial", "sample")
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("mixed", (False, True), ids=("both-unindexed", "mixed-index-state"))
def test_align_hard_003_param_batch_targets_follow_index_normalization(
    monkeypatch: pytest.MonkeyPatch,
    mixed: bool,
) -> None:
    """ID: ALIGN_HARD_003_param_batch_targets_follow_index_normalization."""
    if mixed:
        left = _canonical_param_ao("left", sample_labels=(0, 1, 2))
    else:
        left = _unindexed_param_ao(
            "left",
            coord_order=("trial", "time_s", "sample"),
            sample_labels=(0, 1, 2),
        )
    right = _unindexed_param_ao(
        "right",
        coord_order=("trial", "time_s", "sample"),
        sample_labels=(0, 1, 2),
    )
    expected = _expected_param_sum(left, right, batch_join="exact")
    calls = _record_indexing(monkeypatch)

    out = tal_ufuncs.add(left.a(on="param", sequence_join=None), right)

    assert isinstance(out, AnalysisObject)
    expected_selections = 1 if mixed else 2
    assert [method for method, _ in calls] == ["sel"] * expected_selections
    actual = out.as_dataset(copy="none")["value"]
    assert tuple(actual.xindexes) == ("trial", "sample")
    xr.testing.assert_identical(actual, expected)


def test_align_hard_004_coordinate_transform_param_input_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_HARD_004_coordinate_transform_param_input_normalizes."""
    left = _coordinate_transform_param_ao("left", offset=0.5)
    right = _coordinate_transform_param_ao("right", offset=1.5)
    expected = _expected_param_sum(left, right, batch_join="exact")
    calls = _record_indexing(monkeypatch)

    out = tal_ufuncs.add(left.a(on="param", sequence_join=None), right)

    assert isinstance(out, AnalysisObject)
    assert [method for method, _ in calls] == ["sel", "sel"]
    actual = out.as_dataset(copy="none")["value"]
    assert isinstance(actual.xindexes["sample"], xr.indexes.PandasIndex)
    assert tuple(actual.xindexes) == ("trial", "sample")
    xr.testing.assert_identical(actual, expected)


def test_align_perf_007_dask_noop_graph_matches_direct_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ALIGN_PERF_007_dask_noop_graph_matches_direct_arithmetic."""
    left = _ao("left", sample_labels=range(8), chunked=True)
    right = _ao("right", sample_labels=range(8), chunked=True)
    left_data = left.as_dataset(copy="none")["value"]
    right_data = right.as_dataset(copy="none")["value"]
    expected = xr.ufuncs.add(left_data, right_data)
    calls = _record_indexing(monkeypatch)

    out = _sum_with_sequence_join(left, right, join="exact")
    actual = out.as_dataset(copy="none")["value"]

    assert calls == []
    assert actual.chunks is not None
    assert len(actual.data.__dask_graph__()) == len(expected.data.__dask_graph__())
    assert len(actual.data.dask.layers) == len(expected.data.dask.layers)
