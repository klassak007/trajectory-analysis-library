import numpy as np
import pytest
import xarray as xr
from pathlib import Path

from tal.core import AnalysisObject, BatchConcatOptions, ParamSelectOptions, SchemaError, SequenceConcatOptions
import tal.core.combine_ops.concat_sort as concat_sort
import tal.core.combine_ops.metadata as combine_metadata


def _ao_unbatched(
    *,
    tau: list[float],
    value: list[float],
    size: int | None = None,
    size_name: str = "group_size",
) -> AnalysisObject:
    coords: dict[str, object] = {
        "sample": np.arange(len(value), dtype="int64"),
        "tau": ("sample", np.asarray(tau, dtype="float64")),
    }
    kwargs: dict[str, object] = {
        "sequence_dim": "sample",
        "batch_dims": (),
        "core_dims": (),
        "param_coord": "tau",
    }
    if size is not None:
        coords[size_name] = xr.DataArray(np.asarray(size, dtype="int64"), dims=())
        kwargs["sequence_size_coord"] = size_name
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray(value, dtype="float64"))},
        coords=coords,
    )
    return AnalysisObject.from_data(ds, **kwargs)


def _ao_grouped(
    *,
    trial_labels: tuple[str, ...],
    tau_rows: list[list[float]],
    values: list[list[float]],
    sizes: list[int],
    size_name: str = "group_size",
) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray(values, dtype="float64"))},
        coords={
            "trial": list(trial_labels),
            "sample": np.arange(len(values[0]), dtype="int64"),
            "tau": (("trial", "sample"), np.asarray(tau_rows, dtype="float64")),
            size_name: ("trial", np.asarray(sizes, dtype="int64")),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord=size_name,
    )


def _ao_grouped_no_size(
    *,
    trial_labels: tuple[str, ...],
    tau_rows: list[list[float]],
    values: list[list[float]],
) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray(values, dtype="float64"))},
        coords={
            "trial": list(trial_labels),
            "sample": np.arange(len(values[0]), dtype="int64"),
            "tau": (("trial", "sample"), np.asarray(tau_rows, dtype="float64")),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
    )


def _ao_multi_batch(
    *,
    trial_labels: tuple[str, ...],
    sensor_labels: tuple[str, ...],
    base: list[float],
    offset: float = 0.0,
) -> AnalysisObject:
    seq = np.asarray(base, dtype="float64")
    values = np.zeros((len(trial_labels), len(sensor_labels), seq.size), dtype="float64")
    tau = np.zeros_like(values)
    for i in range(len(trial_labels)):
        for j in range(len(sensor_labels)):
            values[i, j] = seq + offset + i * 10.0 + j
            tau[i, j] = seq
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), values)},
        coords={
            "trial": list(trial_labels),
            "sensor": list(sensor_labels),
            "sample": np.arange(seq.size, dtype="int64"),
            "tau": (("trial", "sensor", "sample"), tau),
            "group_size": (("trial", "sensor"), np.full((len(trial_labels), len(sensor_labels)), seq.size)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )


def test_combine_concat_batch_001_new_outer_batch_dim_roles_updated() -> None:
    """ID: COMBINE_CONCAT_BATCH_001_new_outer_batch_dim_roles_updated."""
    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[3, 3],
    )
    right = _ao_grouped(
        trial_labels=("b", "c"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        sizes=[3, 3],
    )
    out = left.combine.concat_batch([right], opts=BatchConcatOptions(batch_dim="session", sequence_join="inner"))
    roles = out.data.attrs["tal"]["core"]["roles"]
    assert roles["sequence_dim"] == "sample"
    assert roles["batch_dims"] == ["session", "trial"]
    assert "session" in out.data.dims


def test_combine_concat_batch_002_batch_labels_validation() -> None:
    """ID: COMBINE_CONCAT_BATCH_002_batch_labels_validation."""
    left = _ao_unbatched(tau=[0.0, 1.0], value=[0.0, 1.0])
    right = _ao_unbatched(tau=[0.0, 1.0], value=[2.0, 3.0])
    with pytest.raises(ValueError) as err_dup:
        left.combine.concat_batch([right], opts=BatchConcatOptions(batch_labels=("x", "x")))
    assert "batch_labels must be unique" in str(err_dup.value)
    with pytest.raises(ValueError) as err_len:
        left.combine.concat_batch([right], opts=BatchConcatOptions(batch_labels=("x",)))
    assert "batch_labels length" in str(err_len.value)


def test_combine_concat_batch_003_param_coord_shared_vs_batched_shape() -> None:
    """ID: COMBINE_CONCAT_BATCH_003_param_coord_shared_vs_batched_shape."""
    shared_a = _ao_unbatched(tau=[0.0, 1.0, 2.0], value=[0.0, 1.0, 2.0])
    shared_b = _ao_unbatched(tau=[0.0, 1.0, 2.0], value=[10.0, 11.0, 12.0])
    shared = shared_a.combine.concat_batch([shared_b], opts=BatchConcatOptions(batch_dim="session"))
    assert tuple(shared.data.coords["tau"].dims) == ("sample",)
    varied = shared_a.combine.concat_batch(
        [_ao_unbatched(tau=[3.0, 4.0, 5.0], value=[10.0, 11.0, 12.0])],
        opts=BatchConcatOptions(batch_dim="session"),
    )
    assert tuple(varied.data.coords["tau"].dims) == ("session", "sample")


def test_combine_concat_batch_004_validity_truthful_after_outer_sequence_join() -> None:
    """ID: COMBINE_CONCAT_BATCH_004_validity_truthful_after_outer_sequence_join."""
    left = _ao_grouped(
        trial_labels=("a",),
        tau_rows=[[0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0]],
        sizes=[3],
    )
    right = _ao_grouped(
        trial_labels=("b",),
        tau_rows=[[5.0, 6.0, 7.0]],
        values=[[10.0, 11.0, 12.0]],
        sizes=[2],
    )
    out = left.combine.concat_batch([right], opts=BatchConcatOptions(batch_dim="session", sequence_join="outer"))
    core = out.data.attrs["tal"]["core"]
    assert core["validity"]["sequence_size_coord"] == "group_size"
    np.testing.assert_array_equal(out.data.coords["group_size"].sel(session=0).values, [3, 0])
    np.testing.assert_array_equal(out.data.coords["group_size"].sel(session=1).values, [0, 2])


def test_combine_concat_batch_005_preserve_declared_scalar_sequence_size_on_batch_concat() -> None:
    """ID: COMBINE_CONCAT_BATCH_005_preserve_declared_scalar_sequence_size_on_batch_concat."""
    left = _ao_unbatched(
        tau=[0.0, 1.0, 2.0, 3.0],
        value=[0.0, 1.0, 2.0, 3.0],
        size=2,
        size_name="n_valid",
    )
    right = _ao_unbatched(
        tau=[0.0, 1.0, 2.0, 3.0],
        value=[10.0, 11.0, 12.0, 13.0],
        size=3,
        size_name="n_valid",
    )
    out = left.combine.concat_batch([right], opts=BatchConcatOptions(batch_dim="session"))
    np.testing.assert_array_equal(out.data.coords["n_valid"].values, [2, 3])
    assert out.data.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "n_valid"


def test_combine_concat_seq_001_append_lengths_sum_to_sequence_size() -> None:
    """ID: COMBINE_CONCAT_SEQ_001_append_lengths_sum_to_sequence_size."""
    first = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[3, 2],
    )
    second = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        sizes=[2, 1],
    )
    out = first.combine.concat_sequence([second], opts=SequenceConcatOptions(batch_join="inner", overlap="error"))
    np.testing.assert_array_equal(out.data.coords["group_size"].values, [5, 3])


def test_combine_concat_seq_002_overlap_error_boundary_check() -> None:
    """ID: COMBINE_CONCAT_SEQ_002_overlap_error_boundary_check."""
    first = _ao_unbatched(tau=[0.0, 1.0, 2.0], value=[0.0, 1.0, 2.0], size=3)
    second = _ao_unbatched(tau=[1.5, 2.5, 3.5], value=[10.0, 11.0, 12.0], size=3)
    with pytest.raises(ValueError) as err:
        first.combine.concat_sequence([second], opts=SequenceConcatOptions(overlap="error"))
    assert "append would break monotonic order" in str(err.value)


def test_combine_concat_seq_003_overlap_sort_requires_param_coord() -> None:
    """ID: COMBINE_CONCAT_SEQ_003_overlap_sort_requires_param_coord."""
    ds_a = xr.Dataset(data_vars={"value": (("sample",), [0.0, 1.0])}, coords={"sample": [0, 1]})
    ds_b = xr.Dataset(data_vars={"value": (("sample",), [2.0, 3.0])}, coords={"sample": [0, 1]})
    left = AnalysisObject.from_data(ds_a, sequence_dim="sample", batch_dims=(), core_dims=(), validate=False)
    right = AnalysisObject.from_data(ds_b, sequence_dim="sample", batch_dims=(), core_dims=(), validate=False)
    with pytest.raises(ValueError) as err:
        left.combine.concat_sequence([right], opts=SequenceConcatOptions(overlap="sort"))
    assert "requires param_coord" in str(err.value)


def test_combine_concat_seq_004_overlap_sort_masks_invalid_padding() -> None:
    """ID: COMBINE_CONCAT_SEQ_004_overlap_sort_masks_invalid_padding."""
    first = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[3, 2],
    )
    second = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[1.0, 1.5, 3.0], [1.0, 1.5, 3.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        sizes=[2, 1],
    )
    out = first.combine.concat_sequence([second], opts=SequenceConcatOptions(batch_join="inner", overlap="sort"))
    assert bool(out.data.coords["valid"].sel(trial="b").isel(sample=3).item()) is False
    assert np.isnan(out.data["value"].sel(trial="b").isel(sample=3).item())


def test_combine_concat_seq_016_overlap_sort_keeps_invalid_tail_stable() -> None:
    """ID: COMBINE_CONCAT_SEQ_016_overlap_sort_keeps_invalid_tail_stable."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[30.0, 10.0, 90.0, 80.0, 70.0]])},
        coords={
            "trial": ["a"],
            "sample": [0, 1, 2, 3, 4],
            "tau": (("trial", "sample"), [[3.0, 1.0, np.nan, np.nan, np.nan]]),
            "valid": (("trial", "sample"), [[True, True, False, False, False]]),
        },
    )
    out = concat_sort.apply_overlap_sort(
        ds,
        overlap="sort",
        flat_dim="trial",
        sequence_dim="sample",
        param_name="tau",
    )
    values = out["value"].sel(trial="a").values
    np.testing.assert_allclose(values[:2], [10.0, 30.0])
    np.testing.assert_allclose(values[2:], [90.0, 80.0, 70.0])


def test_combine_concat_seq_005_sort_rejects_sample_only_payload_when_grouped() -> None:
    """ID: COMBINE_CONCAT_SEQ_005_sort_rejects_sample_only_payload_when_grouped."""
    ds_left = xr.Dataset(
        data_vars={
            "value": (("trial", "sample"), [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]),
            "shared": (("sample",), [100.0, 101.0, 102.0]),
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "tau": (("trial", "sample"), [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            "group_size": ("trial", [3, 3]),
        },
    )
    ds_right = xr.Dataset(
        data_vars={
            "value": (("trial", "sample"), [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]]),
            "shared": (("sample",), [200.0, 201.0, 202.0]),
        },
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "tau": (("trial", "sample"), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            "group_size": ("trial", [3, 3]),
        },
    )
    left = AnalysisObject.from_data(
        ds_left,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )
    right = AnalysisObject.from_data(
        ds_right,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )
    with pytest.raises(ValueError) as err:
        left.combine.concat_sequence([right], opts=SequenceConcatOptions(batch_join="inner", overlap="sort"))
    assert "sample-only" in str(err.value)


def test_combine_concat_seq_006_no_internal_invalid_gaps_after_packing() -> None:
    """ID: COMBINE_CONCAT_SEQ_006_no_internal_invalid_gaps_after_packing."""
    first = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[2, 1],
    )
    second = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        sizes=[1, 2],
    )
    out = first.combine.concat_sequence([second], opts=SequenceConcatOptions(batch_join="inner", overlap="error"))
    for trial in ["a", "b"]:
        valid = np.asarray(out.data.coords["valid"].sel(trial=trial).values, dtype=bool)
        if not valid.size:
            continue
        false_idx = np.flatnonzero(~valid)
        if false_idx.size == 0:
            continue
        start = int(false_idx[0])
        np.testing.assert_array_equal(valid[start:], np.zeros_like(valid[start:], dtype=bool))


def test_combine_concat_seq_007_invalid_tail_only_and_sequence_coords_masked() -> None:
    """ID: COMBINE_CONCAT_SEQ_007_invalid_tail_only_and_sequence_coords_masked."""
    first = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[3, 2],
    )
    second = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        sizes=[2, 1],
    )
    out = first.combine.concat_sequence([second], opts=SequenceConcatOptions(batch_join="inner", overlap="error"))
    assert bool(out.data.coords["valid"].sel(trial="b", sample=3).item()) is False
    assert np.isnan(out.data.coords["tau"].sel(trial="b", sample=3).item())


def test_combine_concat_batch_006_duplicate_sequence_labels_fail_fast_tal_error() -> None:
    """ID: COMBINE_CONCAT_BATCH_006_duplicate_sequence_labels_fail_fast_tal_error."""
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 0, 1], "tau": ("sample", [0.0, 0.0, 1.0])},
    )
    right = _ao_unbatched(tau=[0.0, 1.0, 2.0], value=[10.0, 11.0, 12.0])
    left = AnalysisObject.from_data(
        left_ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        left.combine.concat_batch([right], opts=BatchConcatOptions(sequence_join="outer"))
    assert "labels along 'sample' must be unique" in str(err.value)


def test_combine_concat_seq_008_temp_dim_namespace_collisions_avoided() -> None:
    """ID: COMBINE_CONCAT_SEQ_008_temp_dim_namespace_collisions_avoided."""
    ds_a = xr.Dataset(
        data_vars={
            "value": (("sample",), [0.0, 1.0]),
            "__segment__": (("sample",), [5.0, 6.0]),
        },
        coords={"sample": [0, 1], "tau": ("sample", [0.0, 1.0]), "__tal_out": xr.DataArray(1)},
    )
    ds_b = xr.Dataset(
        data_vars={
            "value": (("sample",), [2.0, 3.0]),
            "__segment__": (("sample",), [7.0, 8.0]),
        },
        coords={"sample": [0, 1], "tau": ("sample", [2.0, 3.0]), "__tal_out": xr.DataArray(2)},
    )
    left = AnalysisObject.from_data(ds_a, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    right = AnalysisObject.from_data(ds_b, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    out = left.combine.concat_sequence([right], opts=SequenceConcatOptions(overlap="error"))
    np.testing.assert_allclose(out.data["value"].values, [0.0, 1.0, 2.0, 3.0])


def test_combine_concat_seq_009_duplicate_batch_labels_fail_fast_tal_error() -> None:
    """ID: COMBINE_CONCAT_SEQ_009_duplicate_batch_labels_fail_fast_tal_error."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={
            "trial": ["a", "a"],
            "sample": [0, 1],
            "tau": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
            "group_size": ("trial", [2, 2]),
        },
    )
    dup = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )
    with pytest.raises(ValueError) as err:
        dup.combine.concat_sequence([dup], opts=SequenceConcatOptions(batch_join="outer"))
    assert "must be unique" in str(err.value)


def test_combine_concat_seq_010_multi_batch_join_mixed_topology_tal_owned() -> None:
    """ID: COMBINE_CONCAT_SEQ_010_multi_batch_join_mixed_topology_tal_owned."""
    grouped = _ao_multi_batch(
        trial_labels=("a",),
        sensor_labels=("s0", "s1"),
        base=[0.0, 1.0, 2.0],
    )
    unbatched = _ao_unbatched(
        tau=[3.0, 4.0],
        value=[30.0, 40.0],
        size=2,
    )
    out = grouped.combine.concat_sequence(
        [unbatched],
        opts=SequenceConcatOptions(batch_join="outer", overlap="error"),
    )
    assert set(out.data["value"].dims) == {"trial", "sensor", "sample"}
    assert out.data.sizes["sample"] == 5
    np.testing.assert_allclose(out.data["value"].sel(trial="a", sensor="s0").values, [0.0, 1.0, 2.0, 30.0, 40.0])


def test_orch_topo_parity_004_concat_topology_restore_behavior_parity() -> None:
    """ID: ORCH_TOPO_PARITY_004_concat_topology_restore_behavior_parity."""
    grouped = _ao_multi_batch(
        trial_labels=("a", "b"),
        sensor_labels=("s0", "s1"),
        base=[0.0, 1.0],
    )
    other = _ao_multi_batch(
        trial_labels=("b", "a"),
        sensor_labels=("s1", "s0"),
        base=[2.0, 3.0],
        offset=50.0,
    )
    out = grouped.combine.concat_sequence([other], opts=SequenceConcatOptions(batch_join="inner", overlap="error"))
    assert set(out.data["value"].dims) == {"trial", "sensor", "sample"}
    assert list(out.data.coords["trial"].values) == ["a", "b"]
    assert list(out.data.coords["sensor"].values) == ["s0", "s1"]


def test_combine_concat_batch_007_mixed_topology_order_independent() -> None:
    """ID: COMBINE_CONCAT_BATCH_007_mixed_topology_order_independent."""
    grouped = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[3, 3],
    )
    unbatched = _ao_unbatched(
        tau=[5.0, 6.0, 7.0],
        value=[100.0, 101.0, 102.0],
        size=3,
    )
    opts = BatchConcatOptions(batch_dim="run", sequence_join="inner")
    out_a = grouped.combine.concat_batch([unbatched], opts=opts)
    out_b = unbatched.combine.concat_batch([grouped], opts=opts)
    assert {"run", "trial", "sample"} <= set(out_a.data.dims)
    assert {"run", "trial", "sample"} <= set(out_b.data.dims)
    assert out_a.data.sizes["run"] == 2
    assert out_b.data.sizes["run"] == 2


def test_combine_meta_001_param_coord_dim_permutation_preserved() -> None:
    """ID: COMBINE_META_001_param_coord_dim_permutation_preserved."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), np.asarray([[[0.0, 1.0], [2.0, 3.0]]], dtype="float64"))},
        coords={
            "trial": ["a"],
            "sensor": ["s0", "s1"],
            "sample": [0, 1],
            "tau": (("sample", "trial", "sensor"), np.asarray([[[0.0, 0.0]], [[1.0, 1.0]]], dtype="float64")),
            "group_size": (("sensor", "trial"), np.asarray([[2], [2]], dtype="int64")),
        },
    )
    out_param = combine_metadata.align_param_coord_to_canonical(
        ds,
        name="tau",
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
    )
    assert out_param is not None
    assert tuple(out_param.coords["tau"].dims) == ("trial", "sensor", "sample")
    out_size = combine_metadata.align_size_coord_to_canonical(
        out_param,
        name="group_size",
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
    )
    assert out_size is not None
    assert tuple(out_size.coords["group_size"].dims) == ("trial", "sensor")


def test_combine_dry_001_shared_name_single_owner_parity() -> None:
    """ID: COMBINE_DRY_001_shared_name_single_owner_parity."""
    assert combine_metadata.shared_optional_name(["tau", "tau"]) == "tau"
    assert combine_metadata.shared_optional_name(["tau", None]) is None
    assert combine_metadata.shared_optional_name(["tau", "phi"]) is None
    for rel in [
        "tal/core/combine_ops/concat_batch.py",
        "tal/core/combine_ops/concat_plan.py",
        "tal/core/combine_ops/merge.py",
    ]:
        text = Path(rel).read_text(encoding="utf-8")
        assert "def _shared_name(" not in text


def test_combine_dry_002_optional_name_canonicalization_single_owner_parity() -> None:
    """ID: COMBINE_DRY_002_optional_name_canonicalization_single_owner_parity."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype="float64"))},
        coords={
            "trial": ["a"],
            "sensor": ["s0", "s1"],
            "sample": [0, 1],
            "tau": (("sample", "trial", "sensor"), np.asarray([[[0.0, 0.0]], [[1.0, 1.0]]], dtype="float64")),
            "group_size": (("sensor", "trial"), np.asarray([[2], [2]], dtype="int64")),
        },
    )
    out, param_name, size_name = combine_metadata.canonicalize_optional_names(
        ds,
        sequence_dim="sample",
        batch_dims=("trial", "sensor"),
        param_name="tau",
        size_name="group_size",
    )
    assert param_name == "tau"
    assert size_name == "group_size"
    assert tuple(out.coords["tau"].dims) == ("trial", "sensor", "sample")
    assert tuple(out.coords["group_size"].dims) == ("trial", "sensor")


def test_combine_concat_seq_011_outer_synthetic_rows_invalid_without_size_coord() -> None:
    """ID: COMBINE_CONCAT_SEQ_011_outer_synthetic_rows_invalid_without_size_coord."""
    left = _ao_grouped_no_size(
        trial_labels=("a",),
        tau_rows=[[0.0, 1.0]],
        values=[[0.0, 1.0]],
    )
    right = _ao_grouped_no_size(
        trial_labels=("b",),
        tau_rows=[[2.0, 3.0]],
        values=[[20.0, 21.0]],
    )
    out = left.combine.concat_sequence([right], opts=SequenceConcatOptions(batch_join="outer", overlap="error"))
    assert out.data.sizes["sample"] == 2
    np.testing.assert_array_equal(out.data.coords["valid"].sel(trial="a").values, [True, True])
    np.testing.assert_array_equal(out.data.coords["valid"].sel(trial="b").values, [True, True])
    np.testing.assert_allclose(out.data["value"].sel(trial="a").values, [0.0, 1.0])
    np.testing.assert_allclose(out.data["value"].sel(trial="b").values, [20.0, 21.0])


def test_combine_concat_seq_012_overlap_error_checks_across_empty_segments() -> None:
    """ID: COMBINE_CONCAT_SEQ_012_overlap_error_checks_across_empty_segments."""
    first = _ao_unbatched(tau=[1.0, 2.0], value=[10.0, 20.0], size=2)
    empty = _ao_unbatched(tau=[], value=[], size=0)
    third = _ao_unbatched(tau=[0.0, 1.0], value=[30.0, 40.0], size=2)
    with pytest.raises(ValueError) as err:
        first.combine.concat_sequence([empty, third], opts=SequenceConcatOptions(overlap="error"))
    assert "append would break monotonic order" in str(err.value)


def test_combine_concat_seq_013_sort_empty_batch_topology_no_crash() -> None:
    """ID: COMBINE_CONCAT_SEQ_013_sort_empty_batch_topology_no_crash."""
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([], dtype="float64").reshape(0, 2))},
        coords={
            "trial": np.asarray([], dtype=object),
            "sample": [0, 1],
            "tau": (("trial", "sample"), np.asarray([], dtype="float64").reshape(0, 2)),
            "group_size": ("trial", np.asarray([], dtype="int64")),
        },
    )
    empty = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )
    out = empty.combine.concat_sequence([empty], opts=SequenceConcatOptions(batch_join="inner", overlap="sort"))
    assert out.data.sizes["trial"] == 0
    assert out.data.sizes["sample"] == 0


def test_combine_concat_seq_014_unbatched_scalar_sequence_size_respected() -> None:
    """ID: COMBINE_CONCAT_SEQ_014_unbatched_scalar_sequence_size_respected."""
    first = _ao_unbatched(
        tau=[0.0, 1.0, 2.0, 3.0, 4.0],
        value=[0.0, 1.0, 2.0, 3.0, 4.0],
        size=2,
    )
    second = _ao_unbatched(
        tau=[10.0, 11.0],
        value=[10.0, 11.0],
        size=2,
    )
    out = first.combine.concat_sequence([second], opts=SequenceConcatOptions(overlap="error"))
    assert out.data.sizes["sample"] == 4
    np.testing.assert_allclose(out.data["value"].values, [0.0, 1.0, 10.0, 11.0])
    assert int(out.data.coords["group_size"].item()) == 4


def test_combine_concat_seq_015_overlap_error_rejects_non_finite_param() -> None:
    """ID: COMBINE_CONCAT_SEQ_015_overlap_error_rejects_non_finite_param."""
    first = _ao_unbatched(tau=[0.0, 1.0], value=[0.0, 1.0], size=2)
    second = _ao_unbatched(tau=[np.nan, 0.5], value=[2.0, 3.0], size=2)
    with pytest.raises(ValueError) as err:
        first.combine.concat_sequence([second], opts=SequenceConcatOptions(overlap="error"))
    assert "requires finite param_coord values" in str(err.value)


def test_combine_concat_seq_017_chunked_sequence_size_coord_fails_fast() -> None:
    """ID: COMBINE_CONCAT_SEQ_017_chunked_sequence_size_coord_fails_fast."""
    da = pytest.importorskip("dask.array")
    left_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[0.0, 1.0]], dtype="float64"))},
        coords={
            "trial": ["a"],
            "sample": [0, 1],
            "tau": (("trial", "sample"), np.asarray([[0.0, 1.0]], dtype="float64")),
            "group_size": ("trial", da.from_array(np.asarray([2], dtype="int64"), chunks=1)),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[2.0, 3.0]], dtype="float64"))},
        coords={
            "trial": ["a"],
            "sample": [0, 1],
            "tau": (("trial", "sample"), np.asarray([[2.0, 3.0]], dtype="float64")),
            "group_size": ("trial", np.asarray([2], dtype="int64")),
        },
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )
    with pytest.raises(SchemaError) as err:
        left = AnalysisObject.from_data(
            left_ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=(),
            param_coord="tau",
            sequence_size_coord="group_size",
        )
        left.combine.concat_sequence([right], opts=SequenceConcatOptions(batch_join="inner", overlap="error"))
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["coord"] == "group_size"
    assert err.value.actual["reason"] == "chunked coordinate not schema-value-validatable"


def test_combine_concat_seq_018_chunked_param_coord_overlap_checks_fail_fast() -> None:
    """ID: COMBINE_CONCAT_SEQ_018_chunked_param_coord_overlap_checks_fail_fast."""
    da = pytest.importorskip("dask.array")
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0])},
        coords={
            "sample": [0, 1],
            "tau": ("sample", da.from_array(np.asarray([0.0, 1.0], dtype="float64"), chunks=2)),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [2.0, 3.0])},
        coords={"sample": [0, 1], "tau": ("sample", [2.0, 3.0])},
    )
    left = AnalysisObject.from_data(left_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    right = AnalysisObject.from_data(right_ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")
    with pytest.raises(ValueError) as err:
        left.combine.concat_sequence([right], opts=SequenceConcatOptions(overlap="error"))
    assert "overlap checks do not support chunked param_coord values" in str(err.value)


def test_combine_concat_seq_019_outer_missing_rows_size_coord_does_not_false_fail() -> None:
    """ID: COMBINE_CONCAT_SEQ_019_outer_missing_rows_size_coord_does_not_false_fail."""
    first = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[2, 1],
    )
    second = _ao_grouped(
        trial_labels=("a",),
        tau_rows=[[3.0, 4.0, 5.0]],
        values=[[100.0, 101.0, 102.0]],
        sizes=[1],
    )
    out = first.combine.concat_sequence([second], opts=SequenceConcatOptions(batch_join="outer", overlap="error"))
    sizes = out.data.coords["group_size"]
    assert int(sizes.sel(trial="a").item()) == 3
    assert int(sizes.sel(trial="b").item()) == 1


def test_combine_concat_seq_020_outer_missing_rows_label_mapping_is_by_label_not_position() -> None:
    """ID: COMBINE_CONCAT_SEQ_020_outer_missing_rows_label_mapping_is_by_label_not_position."""
    first = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[2, 1],
    )
    second = _ao_grouped(
        trial_labels=("c", "a"),
        tau_rows=[[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
        values=[[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]],
        sizes=[1, 2],
    )
    out = first.combine.concat_sequence([second], opts=SequenceConcatOptions(batch_join="outer", overlap="error"))
    sizes = out.data.coords["group_size"]
    assert int(sizes.sel(trial="a").item()) == 4
    assert int(sizes.sel(trial="b").item()) == 1
    assert int(sizes.sel(trial="c").item()) == 1


def test_combine_concat_seq_021_invalid_present_source_size_still_rejected() -> None:
    """ID: COMBINE_CONCAT_SEQ_021_invalid_present_source_size_still_rejected."""
    first = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[2, 1],
    )
    second_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), np.asarray([[100.0, 101.0, 102.0]], dtype="float64"))},
        coords={
            "trial": ["a"],
            "sample": np.arange(3, dtype="int64"),
            "tau": (("trial", "sample"), np.asarray([[3.0, 4.0, 5.0]], dtype="float64")),
            "group_size": ("trial", np.asarray([1.9], dtype="float64")),
        },
    )
    second = AnalysisObject.from_data(
        second_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
        validate=False,
    )
    with pytest.raises(SchemaError) as err:
        first.combine.concat_sequence(
            [second],
            opts=SequenceConcatOptions(batch_join="outer", overlap="error"),
            validate=False,
        )
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"


def test_combine_concat_batch_008_chunked_scalar_sequence_size_coord_fails_fast() -> None:
    """ID: COMBINE_CONCAT_BATCH_008_chunked_scalar_sequence_size_coord_fails_fast."""
    da = pytest.importorskip("dask.array")
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "n_valid": xr.DataArray(da.from_array(np.asarray(2, dtype="int64"), chunks=1), dims=()),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [10.0, 11.0, 12.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "n_valid": xr.DataArray(np.asarray(3, dtype="int64"), dims=()),
        },
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="n_valid",
    )
    with pytest.raises(SchemaError) as err:
        left = AnalysisObject.from_data(
            left_ds,
            sequence_dim="sample",
            batch_dims=(),
            core_dims=(),
            param_coord="tau",
            sequence_size_coord="n_valid",
        )
        left.combine.concat_batch([right], opts=BatchConcatOptions(batch_dim="session"))
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["coord"] == "n_valid"
    assert err.value.actual["reason"] == "chunked coordinate not schema-value-validatable"


def test_combine_concat_batch_009_invalid_sequence_size_values_rejected() -> None:
    """ID: COMBINE_CONCAT_BATCH_009_invalid_sequence_size_values_rejected."""
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "n_valid": xr.DataArray(np.asarray(1.9, dtype="float64"), dims=()),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [10.0, 11.0, 12.0])},
        coords={
            "sample": [0, 1, 2],
            "tau": ("sample", [0.0, 1.0, 2.0]),
            "n_valid": xr.DataArray(np.asarray(2, dtype="int64"), dims=()),
        },
    )
    left = AnalysisObject.from_data(
        left_ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="n_valid",
        validate=False,
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="n_valid",
    )
    with pytest.raises(SchemaError) as err:
        left.combine.concat_batch([right], opts=BatchConcatOptions(batch_dim="session"), validate=False)
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"


def test_combine_concat_seq_022_valid_reserved_owned_after_concat_sequence() -> None:
    """ID: COMBINE_CONCAT_SEQ_022_valid_reserved_owned_after_concat_sequence."""
    left = _ao_unbatched(tau=[0.0, 2.0], value=[1.0, 2.0], size=2)
    right = _ao_unbatched(tau=[1.0, 3.0], value=[10.0, 20.0], size=2)
    out = left.combine.concat_sequence([right], opts=SequenceConcatOptions(overlap="sort"))
    attrs = out.unsafe_data.coords["valid"].attrs
    assert attrs.get("tal_reserved_owner") == "param_ops"
    assert attrs.get("tal_reserved_name") == "valid"
    chained = out.param.sel([0.5, 2.5])
    np.testing.assert_array_equal(chained.data.coords["valid"].values, [True, True])


def test_combine_concat_seq_023_unbatched_concat_sequence_drops_internal_flat_coord() -> None:
    """ID: COMBINE_CONCAT_SEQ_023_unbatched_concat_sequence_drops_internal_flat_coord."""
    left = _ao_unbatched(tau=[0.0, 1.0], value=[0.0, 1.0], size=2)
    right = _ao_unbatched(tau=[2.0, 3.0], value=[10.0, 11.0], size=2)
    out = left.combine.concat_sequence([right], opts=SequenceConcatOptions(overlap="error"))
    assert "__tal_batch__" not in out.unsafe_data.dims
    assert "__tal_batch__" not in out.unsafe_data.coords


def test_combine_concat_seq_024_overlap_sort_restores_sequence_xindex() -> None:
    """ID: COMBINE_CONCAT_SEQ_024_overlap_sort_restores_sequence_xindex."""
    left = _ao_unbatched(tau=[0.0, 2.0], value=[1.0, 2.0], size=2)
    right = _ao_unbatched(tau=[1.0, 3.0], value=[10.0, 20.0], size=2)
    out = left.combine.concat_sequence([right], opts=SequenceConcatOptions(overlap="sort"))
    assert "sample" in out.unsafe_data.indexes
    assert float(out.unsafe_data.sel(sample=1)["value"].item()) == 10.0


def test_combine_concat_seq_025_invalid_slots_sample_index_forced_to_minus_one() -> None:
    """ID: COMBINE_CONCAT_SEQ_025_invalid_slots_sample_index_forced_to_minus_one."""
    left = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        values=[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
        sizes=[3, 2],
    ).param.sel(slice(0.0, 2.0), opts=ParamSelectOptions(layout="packed"))
    right = _ao_grouped(
        trial_labels=("a", "b"),
        tau_rows=[[1.0, 1.5, 3.0], [1.0, 1.5, 3.0]],
        values=[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        sizes=[2, 1],
    ).param.sel(slice(0.5, 2.0), opts=ParamSelectOptions(layout="packed"))
    out = left.combine.concat_sequence([right], opts=SequenceConcatOptions(batch_join="inner", overlap="sort"))
    valid = out.data.coords["valid"].values.astype(bool)
    sample_index = out.data.coords["sample_index"].values
    invalid = np.logical_not(valid)
    assert bool(np.any(invalid))
    np.testing.assert_array_equal(sample_index[invalid], np.full(sample_index[invalid].shape, -1, dtype="int64"))


def test_orch_lazy_parity_004_concat_overlap_chunked_failfast_stable() -> None:
    """ID: ORCH_LAZY_PARITY_004_concat_overlap_chunked_failfast_stable."""
    test_combine_concat_seq_018_chunked_param_coord_overlap_checks_fail_fast()


def test_orch_concat_parity_001_overlap_error_behavior_stable() -> None:
    """ID: ORCH_CONCAT_PARITY_001_overlap_error_behavior_stable."""
    test_combine_concat_seq_002_overlap_error_boundary_check()


def test_orch_concat_parity_002_overlap_sort_behavior_stable() -> None:
    """ID: ORCH_CONCAT_PARITY_002_overlap_sort_behavior_stable."""
    test_combine_concat_seq_004_overlap_sort_masks_invalid_padding()


def test_orch_concat_parity_003_mixed_topology_behavior_stable() -> None:
    """ID: ORCH_CONCAT_PARITY_003_mixed_topology_behavior_stable."""
    test_combine_concat_seq_010_multi_batch_join_mixed_topology_tal_owned()


def test_orch_concat_parity_004_validity_and_sample_index_behavior_stable() -> None:
    """ID: ORCH_CONCAT_PARITY_004_validity_and_sample_index_behavior_stable."""
    test_combine_concat_seq_025_invalid_slots_sample_index_forced_to_minus_one()


def test_orch_concat_parity_005_pack_gather_behavior_stable() -> None:
    """ID: ORCH_CONCAT_PARITY_005_pack_gather_behavior_stable."""
    test_combine_concat_seq_006_no_internal_invalid_gaps_after_packing()


def test_orch_concat_parity_006_overlap_sort_and_mask_behavior_stable() -> None:
    """ID: ORCH_CONCAT_PARITY_006_overlap_sort_and_mask_behavior_stable."""
    test_combine_concat_seq_004_overlap_sort_masks_invalid_padding()


def test_orch_concat_parity_007_optional_names_and_total_size_behavior_stable() -> None:
    """ID: ORCH_CONCAT_PARITY_007_optional_names_and_total_size_behavior_stable."""
    test_combine_concat_seq_014_unbatched_scalar_sequence_size_respected()


def test_orch_concat_parity_008_concat_finalize_handoff_schema_stable() -> None:
    """ID: ORCH_CONCAT_PARITY_008_concat_finalize_handoff_schema_stable."""
    test_combine_concat_seq_001_append_lengths_sum_to_sequence_size()


def test_orch_parity_004_concat_sequence_behavior_parity_after_phase_split() -> None:
    """ID: ORCH_PARITY_004_concat_sequence_behavior_parity_after_phase_split."""
    test_orch_concat_parity_008_concat_finalize_handoff_schema_stable()


def test_orch_edge_004_group_query_reindexed_by_labels_not_position() -> None:
    """ID: ORCH_EDGE_004_group_query_reindexed_by_labels_not_position."""
    test_combine_concat_seq_020_outer_missing_rows_label_mapping_is_by_label_not_position()
