from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, CoreConcatOptions, concat_core
from tal.core.schema_read import read_roles, validate_schema_if_needed


def _vector_leaf(
    *,
    offset: float = 0.0,
    axis_start: int = 0,
    axis: str = "axis",
    var_name: str = "x",
) -> AnalysisObject:
    values = (np.arange(12, dtype=float).reshape(2, 2, 3) + offset) / 10.0
    ds = xr.Dataset(
        {var_name: (("sample", "trial", axis), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            axis: np.arange(axis_start, axis_start + 3, dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(axis,),
        validate=True,
    )


def _matrix_leaf(
    *,
    offset: float = 0.0,
    row_labels: tuple[object, object] = (0, 1),
    col_labels: tuple[object, object] = (0, 1),
    row_dim: str = "row",
    col_dim: str = "col",
) -> AnalysisObject:
    values = np.arange(16, dtype=float).reshape(2, 2, 2, 2) + offset
    ds = xr.Dataset(
        {"x": (("sample", "trial", row_dim, col_dim), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row_dim: np.asarray(row_labels, dtype=object),
            col_dim: np.asarray(col_labels, dtype=object),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row_dim, col_dim),
        validate=True,
    )


def _roles(ds: xr.Dataset) -> tuple[bool, str | None, tuple[str, ...], tuple[str, ...]]:
    return read_roles(validate_schema_if_needed(ds))


def test_combine_concat_core_001_extends_target_core_dim_length_sum() -> None:
    """ID: COMBINE_CONCAT_CORE_001_extends_target_core_dim_length_sum."""
    a = _vector_leaf(offset=0.0, axis_start=0)
    b = _vector_leaf(offset=1.0, axis_start=3)
    c = _vector_leaf(offset=2.0, axis_start=6)
    opts = CoreConcatOptions(core_dim="axis")
    out = concat_core([a, b, c], opts=opts, validate=True)
    assert _roles(out.unsafe_data)[3] == ("axis",)
    assert out.unsafe_data.sizes["axis"] == 9
    axis = xr.DataArray(np.arange(9, dtype=np.int64), dims=("axis",), coords={"axis": np.arange(9, dtype=np.int64)})
    expected = xr.concat([a.unsafe_data["x"], b.unsafe_data["x"], c.unsafe_data["x"]], dim=axis, join="exact")
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)


def test_combine_concat_core_002_requires_declared_roles_and_target_core_dim() -> None:
    """ID: COMBINE_CONCAT_CORE_002_requires_declared_roles_and_target_core_dim."""
    a = _vector_leaf(offset=0.0, axis_start=0)
    raw_ds = a.unsafe_data.copy(deep=True)
    raw_ds.attrs = {}
    raw = AnalysisObject(raw_ds)
    with pytest.raises(ValueError, match="declared roles"):
        _ = concat_core([raw, a], opts=CoreConcatOptions(core_dim="axis"), validate=True)
    with pytest.raises(ValueError, match="not in declared core_dims"):
        _ = concat_core([a], opts=CoreConcatOptions(core_dim="row"), validate=True)


def test_combine_concat_core_003_non_target_core_dims_exact_label_match() -> None:
    """ID: COMBINE_CONCAT_CORE_003_non_target_core_dims_exact_label_match."""
    left = _matrix_leaf(offset=0.0, row_labels=("r0", "r1"), col_labels=("c0", "c1"))
    right = _matrix_leaf(offset=1.0, row_labels=("r2", "r3"), col_labels=("c0", "cx"))
    with pytest.raises(ValueError, match="non-target core dim"):
        _ = concat_core([left, right], opts=CoreConcatOptions(core_dim="row"), validate=True)


def test_combine_concat_core_004_core_labels_override_length_and_uniqueness() -> None:
    """ID: COMBINE_CONCAT_CORE_004_core_labels_override_length_and_uniqueness."""
    a = _vector_leaf(offset=0.0, axis_start=0)
    b = _vector_leaf(offset=1.0, axis_start=3)
    with pytest.raises(ValueError, match="length"):
        _ = concat_core(
            [a, b],
            opts=CoreConcatOptions(core_dim="axis", core_labels=("a0", "a1")),
            validate=True,
        )
    with pytest.raises(ValueError, match="must be unique"):
        _ = concat_core(
            [a, b],
            opts=CoreConcatOptions(core_dim="axis", core_labels=("k0", "k1", "k2", "k0", "k4", "k5")),
            validate=True,
        )
    out = concat_core(
        [a, b],
        opts=CoreConcatOptions(core_dim="axis", core_labels=("k0", "k1", "k2", "k3", "k4", "k5")),
        validate=True,
    )
    assert out.unsafe_data.coords["axis"].values.tolist() == ["k0", "k1", "k2", "k3", "k4", "k5"]


def test_combine_concat_core_005_accessor_concat_core_includes_receiver() -> None:
    """ID: COMBINE_CONCAT_CORE_005_accessor_concat_core_includes_receiver."""
    a = _vector_leaf(offset=0.0, axis_start=0)
    b = _vector_leaf(offset=1.0, axis_start=3)
    opts = CoreConcatOptions(core_dim="axis")
    out = a.combine.concat_core([b], opts=opts, validate=True)
    expected = concat_core([a, b], opts=opts, validate=True)
    xr.testing.assert_identical(out.unsafe_data, expected.unsafe_data)


def test_combine_concat_core_006_single_numeric_var_required() -> None:
    """ID: COMBINE_CONCAT_CORE_006_single_numeric_var_required."""
    a = _vector_leaf(offset=0.0, axis_start=0)
    multi_ds = a.unsafe_data.copy(deep=True)
    multi_ds["y"] = multi_ds["x"] + 1.0
    multi = AnalysisObject.from_data(
        multi_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    with pytest.raises(ValueError, match="exactly one data variable"):
        _ = concat_core([a, multi], opts=CoreConcatOptions(core_dim="axis"), validate=True)


def test_combine_concat_core_007_duplicate_output_core_labels_fail_fast() -> None:
    """ID: COMBINE_CONCAT_CORE_007_duplicate_output_core_labels_fail_fast."""
    a = _vector_leaf(offset=0.0, axis_start=0)
    b = _vector_leaf(offset=1.0, axis_start=0)
    with pytest.raises(ValueError, match="must be unique"):
        _ = concat_core([a, b], opts=CoreConcatOptions(core_dim="axis"), validate=True)
