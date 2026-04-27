import pytest
import xarray as xr
from pathlib import Path
import importlib
import numpy as np

from tal.core import AnalysisObject, BatchConcatOptions, align_many


def _ao_simple() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={"value": (("sample",), [0.0, 1.0, 2.0])},
        coords={"sample": [0, 1, 2], "tau": ("sample", [0.0, 1.0, 2.0])},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")


def test_combine_hard_001_invalid_input_type_rejected() -> None:
    """ID: COMBINE_HARD_001_invalid_input_type_rejected."""
    ao = _ao_simple()
    with pytest.raises(TypeError) as err:
        ao.combine.concat_batch([123])  # type: ignore[list-item]
    assert "invalid input at index" in str(err.value)


def test_combine_hard_002_invalid_options_type_rejected() -> None:
    """ID: COMBINE_HARD_002_invalid_options_type_rejected."""
    ao = _ao_simple()
    with pytest.raises(TypeError) as err_batch:
        ao.combine.concat_batch([_ao_simple()], opts={"batch_dim": "x"})  # type: ignore[arg-type]
    assert "BatchConcatOptions" in str(err_batch.value)
    with pytest.raises(TypeError) as err_seq:
        ao.combine.concat_sequence([_ao_simple()], opts={"overlap": "sort"})  # type: ignore[arg-type]
    assert "SequenceConcatOptions" in str(err_seq.value)
    with pytest.raises(TypeError) as err_merge:
        ao.combine.merge([_ao_simple()], opts={"compat": "override"})  # type: ignore[arg-type]
    assert "MergeOptions" in str(err_merge.value)


def test_combine_hard_003_namespace_collision_on_batch_dim_rejected() -> None:
    """ID: COMBINE_HARD_003_namespace_collision_on_batch_dim_rejected."""
    ao = _ao_simple()
    with pytest.raises(ValueError) as err:
        ao.combine.concat_batch([_ao_simple()], opts=BatchConcatOptions(batch_dim="sample"))
    assert "collides with existing dataset namespace" in str(err.value)


def test_combine_hard_004_backend_alignment_errors_are_tal_owned() -> None:
    """ID: COMBINE_HARD_004_backend_alignment_errors_are_tal_owned."""
    left = _ao_simple()
    right_ds = xr.Dataset(
        data_vars={"value": (("sample",), [5.0, 6.0])},
        coords={"sample": [0, 0], "tau": ("sample", [0.0, 0.0])},
    )
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(),
        param_coord="tau",
    )
    with pytest.raises(ValueError) as err:
        align_many([left, right])
    text = str(err.value)
    assert "labels along 'sample' must be unique" in text
    assert "xarray" not in text.lower()


def test_combine_hard_005_options_unhashable_batch_labels_raise_tal_valueerror() -> None:
    """ID: COMBINE_HARD_005_options_unhashable_batch_labels_raise_tal_valueerror."""
    ao = _ao_simple()
    with pytest.raises(ValueError) as err:
        ao.combine.concat_batch([_ao_simple()], opts=BatchConcatOptions(batch_labels=(["a"], ["b"])))  # type: ignore[list-item]
    assert "hashable" in str(err.value)


def test_combine_hard_006_schema_reader_boundary_no_direct_attrs_parsing_paths() -> None:
    """ID: COMBINE_HARD_006_schema_reader_boundary_no_direct_attrs_parsing_paths."""
    text = Path("tal/core/combine_ops/normalize.py").read_text(encoding="utf-8")
    assert 'attrs["tal"]' not in text


def test_combine_hard_007_left_packed_check_chunked_mask_no_eager_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: COMBINE_HARD_007_left_packed_check_chunked_mask_no_eager_compute."""
    da = pytest.importorskip("dask.array")
    validity_layout_mod = importlib.import_module("tal.core.validity_layout")
    valid = xr.DataArray(
        da.from_array([True, True, False], chunks=2),
        dims=("sample",),
    )

    def _boom_compute(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected eager compute")

    monkeypatch.setattr(da.Array, "compute", _boom_compute, raising=True)
    assert validity_layout_mod.is_left_packed_mask(valid, sequence_dim="sample") is False


def test_combine_hard_008_chunked_or_unverifiable_validity_prunes_size_coord(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: COMBINE_HARD_008_chunked_or_unverifiable_validity_prunes_size_coord."""
    da = pytest.importorskip("dask.array")
    validity_mod = importlib.import_module("tal.core.validity_finalize")
    ds = xr.Dataset(
        coords={
            "sample": [0, 1, 2],
            "n_valid": xr.DataArray(3, dims=()),
        }
    )
    valid = xr.DataArray(
        da.from_array([True, False, True], chunks=2),
        dims=("sample",),
    )

    def _boom_compute(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("unexpected eager compute")

    monkeypatch.setattr(da.Array, "compute", _boom_compute, raising=True)
    out, kept = validity_mod.assign_sequence_size_from_valid_mask(
        ds,
        valid=valid,
        sequence_dim="sample",
        batch_dims=(),
        sequence_size_coord="n_valid",
    )
    assert kept is None
    assert "n_valid" not in out.coords


def test_combine_hard_009_left_packed_validity_assigns_sequence_size_coord() -> None:
    """ID: COMBINE_HARD_009_left_packed_validity_assigns_sequence_size_coord."""
    validity_mod = importlib.import_module("tal.core.validity_finalize")
    ds = xr.Dataset(coords={"sample": [0, 1, 2]})
    valid = xr.DataArray(np.asarray([True, True, False], dtype=bool), dims=("sample",))
    out, kept = validity_mod.assign_sequence_size_from_valid_mask(
        ds,
        valid=valid,
        sequence_dim="sample",
        batch_dims=(),
        sequence_size_coord="n_valid",
    )
    assert kept == "n_valid"
    assert "n_valid" in out.coords
    assert int(out.coords["n_valid"].item()) == 2
