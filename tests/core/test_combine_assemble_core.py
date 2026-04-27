from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, assemble_core, block_core, stack_core
from tal.core.schema_read import read_roles, validate_schema_if_needed


def _vector_leaf(
    *,
    offset: float = 0.0,
    axis: str = "axis",
    var_name: str = "x",
) -> AnalysisObject:
    values = (np.arange(12, dtype=float).reshape(2, 2, 3) + offset) / 10.0
    ds = xr.Dataset(
        {var_name: (("sample", "trial", axis), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            axis: np.arange(3, dtype=np.int64),
            "tau": (("trial", "sample"), np.arange(4, dtype=float).reshape(2, 2)),
            "sample_size": ("trial", np.full(2, 2, dtype=np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(axis,),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )


def _matrix_leaf(*, offset: float, row: str = "r", col: str = "c") -> AnalysisObject:
    values = np.arange(16, dtype=float).reshape(2, 2, 2, 2) + offset
    ds = xr.Dataset(
        {"x": (("sample", "trial", row, col), values)},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row: np.arange(2, dtype=np.int64),
            col: np.arange(2, dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=True,
    )


def _roles(ds: xr.Dataset) -> tuple[bool, str | None, tuple[str, ...], tuple[str, ...]]:
    return read_roles(validate_schema_if_needed(ds))


def test_combine_assemble_001_dense_nd_layout_shape_and_core_labels() -> None:
    """ID: COMBINE_ASSEMBLE_001_dense_nd_layout_shape_and_core_labels."""
    a = _vector_leaf(offset=0.0)
    b = _vector_leaf(offset=1.0)
    c = _vector_leaf(offset=2.0)
    d = _vector_leaf(offset=3.0)
    out = assemble_core(
        [[a, b], [c, d]],
        core_dims=("row", "col"),
        core_labels=(("r0", "r1"), ("c0", "c1")),
        validate=True,
    )
    assert _roles(out.unsafe_data)[3] == ("row", "col", "axis")
    assert out.unsafe_data.coords["row"].values.tolist() == ["r0", "r1"]
    assert out.unsafe_data.coords["col"].values.tolist() == ["c0", "c1"]
    col_axis = xr.DataArray(["c0", "c1"], dims=("col",), coords={"col": ["c0", "c1"]})
    row_axis = xr.DataArray(["r0", "r1"], dims=("row",), coords={"row": ["r0", "r1"]})
    top = xr.concat([a.unsafe_data["x"], b.unsafe_data["x"]], dim=col_axis)
    bottom = xr.concat([c.unsafe_data["x"], d.unsafe_data["x"]], dim=col_axis)
    expected = xr.concat([top, bottom], dim=row_axis)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)
    assert out.unsafe_data.attrs["tal"]["core"]["param_coord"] == {"name": "tau"}
    assert out.unsafe_data.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "sample_size"


def test_combine_assemble_002_ragged_layout_rejected() -> None:
    """ID: COMBINE_ASSEMBLE_002_ragged_layout_rejected."""
    a = _vector_leaf(offset=0.0)
    b = _vector_leaf(offset=1.0)
    c = _vector_leaf(offset=2.0)
    with pytest.raises(ValueError, match="ragged"):
        _ = assemble_core([[a, b], [c]], core_dims=("row", "col"))


def test_combine_assemble_003_exact_alignment_label_safe_not_positional() -> None:
    """ID: COMBINE_ASSEMBLE_003_exact_alignment_label_safe_not_positional."""
    a = _vector_leaf(offset=0.0)
    b_ds = _vector_leaf(offset=5.0).unsafe_data.transpose("trial", "sample", "axis")
    b = AnalysisObject.from_data(
        b_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )
    out = stack_core([a, b], core_dim="row", core_labels=("top", "bottom"))
    row_axis = xr.DataArray(["top", "bottom"], dims=("row",), coords={"row": ["top", "bottom"]})
    expected = xr.concat([a.unsafe_data["x"], b.unsafe_data["x"]], dim=row_axis)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)


def test_combine_assemble_004_shared_leaf_core_dims_preserved() -> None:
    """ID: COMBINE_ASSEMBLE_004_shared_leaf_core_dims_preserved."""
    left = _matrix_leaf(offset=0.0, row="lr", col="lc")
    right = _matrix_leaf(offset=10.0, row="lr", col="lc")
    out = stack_core([left, right], core_dim="block", core_labels=("l", "r"))
    assert _roles(out.unsafe_data)[3] == ("block", "lr", "lc")


def test_combine_assemble_005_conflicting_leaf_core_dims_fail_closed() -> None:
    """ID: COMBINE_ASSEMBLE_005_conflicting_leaf_core_dims_fail_closed."""
    left = _vector_leaf(offset=0.0, axis="x_axis")
    right = _vector_leaf(offset=1.0, axis="y_axis")
    with pytest.raises(ValueError, match="leaf core_dims conflict"):
        _ = stack_core([left, right], core_dim="row")


def test_combine_assemble_006_single_numeric_leaf_var_required() -> None:
    """ID: COMBINE_ASSEMBLE_006_single_numeric_leaf_var_required."""
    base = _vector_leaf(offset=0.0)
    multi_ds = base.unsafe_data.copy(deep=True)
    multi_ds["y"] = multi_ds["x"] + 1.0
    multi = AnalysisObject.from_data(
        multi_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )
    with pytest.raises(ValueError, match="exactly one data variable"):
        _ = stack_core([base, multi], core_dim="row")


def test_combine_assemble_007_accessor_stack_core_includes_receiver() -> None:
    """ID: COMBINE_ASSEMBLE_007_accessor_stack_core_includes_receiver."""
    a = _vector_leaf(offset=0.0)
    b = _vector_leaf(offset=1.0)
    out = a.combine.stack_core([b], core_dim="row", core_labels=("self", "other"))
    expected = stack_core([a, b], core_dim="row", core_labels=("self", "other"))
    xr.testing.assert_identical(out.unsafe_data, expected.unsafe_data)


def test_combine_assemble_008_accessor_block_core_includes_receiver_outer_axis() -> None:
    """ID: COMBINE_ASSEMBLE_008_accessor_block_core_includes_receiver_outer_axis."""
    a = _vector_leaf(offset=0.0)
    b = _vector_leaf(offset=1.0)
    out = a.combine.block_core(
        [[b]],
        row_dim="row",
        col_dim="col",
        row_labels=("self", "other"),
        col_labels=("c0",),
    )
    expected = block_core(
        [[a], [b]],
        row_dim="row",
        col_dim="col",
        row_labels=("self", "other"),
        col_labels=("c0",),
    )
    xr.testing.assert_identical(out.unsafe_data, expected.unsafe_data)


def test_combine_assemble_009_accessor_assemble_core_includes_receiver_outer_axis() -> None:
    """ID: COMBINE_ASSEMBLE_009_accessor_assemble_core_includes_receiver_outer_axis."""
    a = _vector_leaf(offset=0.0)
    b = _vector_leaf(offset=1.0)
    out = a.combine.assemble_core(
        [[b]],
        core_dims=("row", "col"),
        core_labels=(("self", "other"), ("c0",)),
    )
    expected = assemble_core(
        [[a], [b]],
        core_dims=("row", "col"),
        core_labels=(("self", "other"), ("c0",)),
    )
    xr.testing.assert_identical(out.unsafe_data, expected.unsafe_data)
