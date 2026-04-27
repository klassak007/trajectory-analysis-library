import importlib
import numpy as np
import pytest
import xarray as xr

from tal.core import AlignOptions, AnalysisObject, SchemaError, align_many, align_pair
from tal.core.param_ops.batch_topology import BatchFlattenPlan, restore_dataset_batch_dims


def _ao_grouped(
    *,
    trial_labels: tuple[str, ...],
    tau_rows: list[list[float]],
    value_offset: float = 0.0,
) -> AnalysisObject:
    base = np.asarray([[0.0, 1.0, 2.0] for _ in trial_labels], dtype="float64")
    values = base + value_offset + np.arange(len(trial_labels), dtype="float64")[:, None]
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), values)},
        coords={
            "trial": list(trial_labels),
            "sample": [0, 1, 2],
            "tau": (("trial", "sample"), np.asarray(tau_rows, dtype="float64")),
            "group_size": ("trial", np.full(len(trial_labels), 3, dtype="int64")),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="group_size",
    )


def _ao_multi_batch(
    *,
    trial_labels: tuple[str, ...],
    sensor_labels: tuple[str, ...],
    offset: float,
) -> AnalysisObject:
    base = np.asarray([0.0, 1.0, 2.0], dtype="float64")
    value = np.zeros((len(trial_labels), len(sensor_labels), base.size), dtype="float64")
    tau = np.zeros_like(value)
    for i in range(len(trial_labels)):
        for j in range(len(sensor_labels)):
            value[i, j] = base + offset + i * 10.0 + j
            tau[i, j] = base
    ds = xr.Dataset(
        data_vars={"value": (("trial", "sensor", "sample"), value)},
        coords={
            "trial": list(trial_labels),
            "sensor": list(sensor_labels),
            "sample": [0, 1, 2],
            "tau": (("trial", "sensor", "sample"), tau),
            "group_size": (("trial", "sensor"), np.full((len(trial_labels), len(sensor_labels)), 3, dtype="int64")),
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


def _ao_unbatched(
    *,
    sample_labels: list[int],
    values: list[float],
    size: int,
) -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), np.asarray(values, dtype="float64"))},
        coords={
            "sample": np.asarray(sample_labels, dtype="int64"),
            "tau": ("sample", np.asarray(sample_labels, dtype="float64")),
            "n_valid": xr.DataArray(np.asarray(size, dtype="int64"), dims=()),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="n_valid",
    )


def test_combine_align_001_pair_join_matrix_inner_outer_exact() -> None:
    """ID: COMBINE_ALIGN_001_pair_join_matrix_inner_outer_exact."""
    left = _ao_grouped(trial_labels=("a", "b"), tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    right = _ao_grouped(trial_labels=("b", "c"), tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], value_offset=100.0)
    inner_left, inner_right = align_pair(left, right, opts=AlignOptions(batch_join="inner", sequence_join="inner"))
    assert list(inner_left.data.coords["trial"].values) == ["b"]
    assert list(inner_right.data.coords["trial"].values) == ["b"]
    outer_left, outer_right = align_pair(left, right, opts=AlignOptions(batch_join="outer", sequence_join="inner"))
    assert list(outer_left.data.coords["trial"].values) == ["a", "b", "c"]
    assert list(outer_right.data.coords["trial"].values) == ["a", "b", "c"]
    with pytest.raises(ValueError) as err:
        align_pair(left, right, opts=AlignOptions(batch_join="exact", sequence_join="inner"))
    assert "batch_join='exact'" in str(err.value)


def test_combine_align_002_many_multi_batch_flatten_restore() -> None:
    """ID: COMBINE_ALIGN_002_many_multi_batch_flatten_restore."""
    first = _ao_multi_batch(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    second = _ao_multi_batch(trial_labels=("b", "c"), sensor_labels=("s1", "s2"), offset=100.0)
    out = align_many([first, second], opts=AlignOptions(batch_join="outer", sequence_join="inner"))
    assert len(out) == 2
    assert set(out[0].data.dims) == {"trial", "sensor", "sample"}
    assert list(out[0].data.coords["trial"].values) == ["a", "b", "c"]
    assert list(out[0].data.coords["sensor"].values) == ["s0", "s1", "s2"]
    assert out[0].data.attrs["tal"]["core"]["roles"]["batch_dims"] == ["trial", "sensor"]


def test_orch_topo_parity_003_combine_align_multi_batch_behavior_parity() -> None:
    """ID: ORCH_TOPO_PARITY_003_combine_align_multi_batch_behavior_parity."""
    first = _ao_multi_batch(trial_labels=("a", "b"), sensor_labels=("s0", "s1"), offset=0.0)
    second = _ao_multi_batch(trial_labels=("b", "a"), sensor_labels=("s1", "s0"), offset=25.0)
    out = align_many([first, second], opts=AlignOptions(batch_join="inner", sequence_join="inner"))
    assert len(out) == 2
    assert set(out[0].data.dims) == {"trial", "sensor", "sample"}
    assert list(out[0].data.coords["trial"].values) == ["a", "b"]
    assert list(out[0].data.coords["sensor"].values) == ["s0", "s1"]


def test_orch_finalize_parity_003_combine_align_finalize_path_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ORCH_FINALIZE_PARITY_003_combine_align_finalize_path_stable."""
    finalize_mod = importlib.import_module("tal.core.combine_ops.finalize")
    calls = {"finalize_with_schema": 0}
    orig_finalize = finalize_mod.finalize_with_schema

    def _count_finalize(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["finalize_with_schema"] += 1
        return orig_finalize(*args, **kwargs)

    monkeypatch.setattr(finalize_mod, "finalize_with_schema", _count_finalize)

    left = _ao_grouped(trial_labels=("a", "b"), tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    right = _ao_grouped(trial_labels=("b", "c"), tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], value_offset=10.0)
    out = align_many([left, right], opts=AlignOptions(batch_join="inner", sequence_join="inner"))
    assert len(out) == 2
    assert list(out[0].data.coords["trial"].values) == ["b"]
    assert calls["finalize_with_schema"] >= 1


def test_combine_align_003_duplicate_axis_labels_fail_fast() -> None:
    """ID: COMBINE_ALIGN_003_duplicate_axis_labels_fail_fast."""
    dup_ds = xr.Dataset(
        data_vars={"value": (("trial", "sample"), [[0.0, 1.0], [2.0, 3.0]])},
        coords={
            "trial": ["a", "a"],
            "sample": [0, 1],
            "tau": (("trial", "sample"), [[0.0, 1.0], [0.0, 1.0]]),
        },
    )
    dup = AnalysisObject.from_data(
        dup_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
    )
    ok = _ao_grouped(trial_labels=("a",), tau_rows=[[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError) as err:
        align_many([dup, ok], opts=AlignOptions(batch_join="inner", sequence_join="inner"))
    assert "labels along 'trial' must be unique" in str(err.value)


def test_combine_align_004_static_batch_or_scalar_alignment_supported() -> None:
    """ID: COMBINE_ALIGN_004_static_batch_or_scalar_alignment_supported."""
    dynamic = _ao_grouped(trial_labels=("a", "b"), tau_rows=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    scalar = AnalysisObject(xr.Dataset(data_vars={"offset": xr.DataArray(5.0)}))
    out = align_many([dynamic, scalar], opts=AlignOptions(batch_join="outer", sequence_join="inner"))
    assert "trial" in out[1].data.dims
    assert list(out[1].data.coords["trial"].values) == ["a", "b"]
    np.testing.assert_allclose(out[1].data["offset"].values, [5.0, 5.0])


def test_combine_align_005_pad_invalid_outer_true_rebuilds_validity_or_prunes_unrepresentable() -> None:
    """ID: COMBINE_ALIGN_005_pad_invalid_outer_true_rebuilds_validity_or_prunes_unrepresentable."""
    left = _ao_unbatched(sample_labels=[0, 2], values=[1.0, 2.0], size=2)
    right = _ao_unbatched(sample_labels=[0, 1], values=[10.0, 20.0], size=2)
    out_left, out_right = align_pair(
        left,
        right,
        opts=AlignOptions(batch_join="inner", sequence_join="outer", pad_invalid_outer=True),
    )
    assert out_left.data.attrs["tal"]["core"].get("validity") == {"sequence_size_coord": "n_valid", "layout": "left_packed"}
    assert out_right.data.attrs["tal"]["core"].get("validity") is None
    assert "n_valid" in out_left.data.coords
    assert "n_valid" not in out_right.data.coords


def test_combine_align_006_pad_invalid_outer_false_prunes_validity_metadata() -> None:
    """ID: COMBINE_ALIGN_006_pad_invalid_outer_false_prunes_validity_metadata."""
    left = _ao_unbatched(sample_labels=[0, 2], values=[1.0, 2.0], size=2)
    right = _ao_unbatched(sample_labels=[0, 1], values=[10.0, 20.0], size=2)
    out_left, out_right = align_pair(
        left,
        right,
        opts=AlignOptions(batch_join="inner", sequence_join="outer", pad_invalid_outer=False),
    )
    assert out_left.data.attrs["tal"]["core"].get("validity") is None
    assert out_right.data.attrs["tal"]["core"].get("validity") is None
    assert "n_valid" not in out_left.data.coords
    assert "n_valid" not in out_right.data.coords


def test_combine_align_007_multi_batch_outer_null_labels_restore_deterministic() -> None:
    """ID: COMBINE_ALIGN_007_multi_batch_outer_null_labels_restore_deterministic."""
    left = _ao_multi_batch(
        trial_labels=("a", None),  # type: ignore[arg-type]
        sensor_labels=("s0",),
        offset=0.0,
    )
    right = _ao_multi_batch(
        trial_labels=("a", np.nan),  # type: ignore[arg-type]
        sensor_labels=("s0",),
        offset=10.0,
    )
    out = align_many([left, right], opts=AlignOptions(batch_join="outer", sequence_join="exact"))
    assert len(out) == 2
    assert set(out[0].data.dims) == {"trial", "sensor", "sample"}


def test_combine_align_008_multi_batch_outer_nan_labels_restore() -> None:
    """ID: COMBINE_ALIGN_008_multi_batch_outer_nan_labels_restore."""
    left = _ao_multi_batch(
        trial_labels=(np.nan,),  # type: ignore[arg-type]
        sensor_labels=("s0",),
        offset=0.0,
    )
    right = _ao_multi_batch(
        trial_labels=(np.nan,),  # type: ignore[arg-type]
        sensor_labels=("s0",),
        offset=10.0,
    )
    out = align_many([left, right], opts=AlignOptions(batch_join="outer", sequence_join="exact"))
    assert len(out) == 2
    assert set(out[0].data.dims) == {"trial", "sensor", "sample"}
    assert bool(np.asarray(out[0].data.coords["trial"].isnull().values).item())
    assert out[0].data.coords["sensor"].values.tolist() == ["s0"]


def test_combine_align_009_outer_chunked_sequence_size_coord_fails_fast() -> None:
    """ID: COMBINE_ALIGN_009_outer_chunked_sequence_size_coord_fails_fast."""
    da = pytest.importorskip("dask.array")
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), [1.0, 2.0])},
        coords={
            "sample": [0, 2],
            "tau": ("sample", [0.0, 2.0]),
            "n_valid": xr.DataArray(da.from_array(np.asarray(2, dtype="int64"), chunks=1), dims=()),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [10.0, 20.0])},
        coords={
            "sample": [0, 1],
            "tau": ("sample", [0.0, 1.0]),
            "n_valid": xr.DataArray(np.asarray(2, dtype="int64"), dims=()),
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
        align_pair(left, right, opts=AlignOptions(sequence_join="outer"))
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["coord"] == "n_valid"
    assert err.value.actual["reason"] == "chunked coordinate not schema-value-validatable"


def test_combine_align_010_invalid_sequence_size_coord_values_rejected() -> None:
    """ID: COMBINE_ALIGN_010_invalid_sequence_size_coord_values_rejected."""
    left_ds = xr.Dataset(
        data_vars={"value": (("sample",), [1.0, 2.0])},
        coords={
            "sample": [0, 2],
            "tau": ("sample", [0.0, 2.0]),
            "n_valid": xr.DataArray(np.asarray(1.9, dtype="float64"), dims=()),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [10.0, 20.0])},
        coords={
            "sample": [0, 1],
            "tau": ("sample", [0.0, 1.0]),
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
        align_pair(left, right, opts=AlignOptions(sequence_join="outer"), validate=False)
    assert err.value.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.path == "tal.core.validity.sequence_size_coord"


def test_orch_lazy_parity_003_align_chunked_sequence_size_failfast_stable() -> None:
    """ID: ORCH_LAZY_PARITY_003_align_chunked_sequence_size_failfast_stable."""
    test_combine_align_009_outer_chunked_sequence_size_coord_fails_fast()


def test_orch_parity_003_combine_align_behavior_parity_after_migration() -> None:
    """ID: ORCH_PARITY_003_combine_align_behavior_parity_after_migration."""
    test_orch_topo_parity_003_combine_align_multi_batch_behavior_parity()


def test_combine_align_011_named_flat_coords_restore_not_tuple_position_dependent() -> None:
    """ID: COMBINE_ALIGN_011_named_flat_coords_restore_not_tuple_position_dependent."""
    plan = BatchFlattenPlan(enabled=True, flat_dim="__flat__", batch_dims=("trial", "sensor"))
    flat_labels = np.empty(2, dtype=object)
    flat_labels[:] = [("sensor_0", "trial_0"), ("sensor_1", "trial_1")]
    ds = xr.Dataset(
        data_vars={
            "value": (
                ("__flat__", "sample"),
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float64"),
            )
        },
        coords={
            "__flat__": ("__flat__", flat_labels),
            "trial": ("__flat__", ["trial_0", "trial_1"]),
            "sensor": ("__flat__", ["sensor_0", "sensor_1"]),
            "sample": [0, 1],
        },
    )
    restored = restore_dataset_batch_dims(ds, plan=plan, owner="combine align")
    assert list(restored.coords["trial"].values) == ["trial_0", "trial_1"]
    assert list(restored.coords["sensor"].values) == ["sensor_0", "sensor_1"]
    np.testing.assert_allclose(
        restored["value"].sel(trial="trial_0", sensor="sensor_0").values,
        np.asarray([1.0, 2.0], dtype="float64"),
    )


def test_orch_edge_001_align_excludes_core_dim_label_union() -> None:
    """ID: ORCH_EDGE_001_align_excludes_core_dim_label_union."""
    left_ds = xr.Dataset(
        data_vars={"value": (("trial", "axis", "sample"), np.zeros((1, 2, 2), dtype="float64"))},
        coords={
            "trial": ["t0"],
            "axis": ["x", "y"],
            "sample": [0, 1],
            "tau": (("trial", "sample"), [[0.0, 1.0]]),
            "group_size": ("trial", [2]),
        },
    )
    right_ds = xr.Dataset(
        data_vars={"value": (("trial", "axis", "sample"), np.ones((1, 3, 2), dtype="float64"))},
        coords={
            "trial": ["t0"],
            "axis": ["u", "v", "w"],
            "sample": [0, 1],
            "tau": (("trial", "sample"), [[0.0, 1.0]]),
            "group_size": ("trial", [2]),
        },
    )
    left = AnalysisObject.from_data(
        left_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="tau",
        sequence_size_coord="group_size",
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="tau",
        sequence_size_coord="group_size",
    )
    out_left, out_right = align_pair(left, right, opts=AlignOptions(batch_join="inner", sequence_join="inner"))
    assert out_left.data.sizes["axis"] == 2
    assert out_right.data.sizes["axis"] == 3
