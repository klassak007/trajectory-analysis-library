from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, CoreDecomposeOptions, assemble_core, decompose_core
from tal.core.schema_read import read_roles, validate_schema_if_needed


def _vector_leaf(*, offset: float = 0.0, with_axis_coord: bool = True) -> AnalysisObject:
    values = (np.arange(12, dtype=float).reshape(2, 2, 3) + offset) / 10.0
    coords: dict[str, object] = {
        "sample": np.arange(2, dtype=np.int64),
        "trial": np.asarray(["t0", "t1"], dtype=object),
        "tau": (("trial", "sample"), np.arange(4, dtype=float).reshape(2, 2)),
        "sample_size": ("trial", np.full(2, 2, dtype=np.int64)),
    }
    if with_axis_coord:
        coords["axis"] = np.arange(3, dtype=np.int64)
    ds = xr.Dataset(
        {"x": (("sample", "trial", "axis"), values)},
        coords=coords,
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )


def _assembled_grid(
    *,
    labels: tuple[tuple[object, ...], tuple[object, ...]] | None = None,
) -> tuple[AnalysisObject, list[AnalysisObject]]:
    leaves = [_vector_leaf(offset=0.0), _vector_leaf(offset=1.0), _vector_leaf(offset=2.0), _vector_leaf(offset=3.0)]
    assembled = assemble_core(
        [[leaves[0], leaves[1]], [leaves[2], leaves[3]]],
        core_dims=("row", "col"),
        core_labels=labels,
        validate=True,
    )
    return assembled, leaves


def _assert_mapping_identical(
    left: dict[tuple[object, ...], AnalysisObject],
    right: dict[tuple[object, ...], AnalysisObject],
) -> None:
    assert list(left.keys()) == list(right.keys())
    for key in left:
        xr.testing.assert_identical(left[key].as_dataset(copy="none"), right[key].as_dataset(copy="none"))


def test_combine_decompose_core_001_roundtrip_assemble_then_decompose_index_mode() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_001_roundtrip_assemble_then_decompose_index_mode."""
    assembled, leaves = _assembled_grid(labels=(("r0", "r1"), ("c0", "c1")))
    out = decompose_core(
        assembled,
        opts=CoreDecomposeOptions(core_dims=("row", "col"), key_mode="index"),
        validate=True,
    )
    expected_keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert list(out.keys()) == expected_keys
    for key, expected in zip(expected_keys, leaves):
        xr.testing.assert_allclose(out[key].as_dataset(copy="none")["x"], expected.as_dataset(copy="none")["x"])


def test_combine_decompose_core_002_roundtrip_assemble_then_decompose_label_mode() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_002_roundtrip_assemble_then_decompose_label_mode."""
    assembled, leaves = _assembled_grid(labels=(("top", "bottom"), ("left", "right")))
    out = decompose_core(
        assembled,
        opts=CoreDecomposeOptions(core_dims=("row", "col"), key_mode="label"),
        validate=True,
    )
    expected_keys = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    assert list(out.keys()) == expected_keys
    for key, expected in zip(expected_keys, leaves):
        xr.testing.assert_allclose(out[key].as_dataset(copy="none")["x"], expected.as_dataset(copy="none")["x"])


def test_combine_decompose_core_003_requires_declared_roles_and_prefix_core_dims() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_003_requires_declared_roles_and_prefix_core_dims."""
    base = _vector_leaf(offset=0.0)
    raw_ds = base.as_dataset(copy="none").copy(deep=True)
    raw_ds.attrs = {}
    raw = AnalysisObject(raw_ds)
    with pytest.raises(ValueError, match="declared roles"):
        _ = decompose_core(raw, opts=CoreDecomposeOptions(core_dims=("axis",)), validate=True)
    assembled, _ = _assembled_grid()
    with pytest.raises(ValueError, match="must be a prefix"):
        _ = decompose_core(assembled, opts=CoreDecomposeOptions(core_dims=("col",)), validate=True)


def test_combine_decompose_core_004_label_mode_requires_present_unique_labels() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_004_label_mode_requires_present_unique_labels."""
    missing = _vector_leaf(with_axis_coord=False)
    with pytest.raises(ValueError, match="requires explicit coord"):
        _ = decompose_core(
            missing,
            opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label"),
            validate=True,
        )
    dup_ds = _vector_leaf().as_dataset(copy="none").assign_coords({"axis": [0, 0, 1]})
    dup = AnalysisObject.from_data(
        dup_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )
    with pytest.raises(ValueError, match="requires unique labels"):
        _ = decompose_core(dup, opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label"), validate=True)


def test_combine_decompose_core_005_leaf_roles_preserve_sequence_batch_and_remaining_core() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_005_leaf_roles_preserve_sequence_batch_and_remaining_core."""
    assembled, _ = _assembled_grid()
    out = decompose_core(
        assembled,
        opts=CoreDecomposeOptions(core_dims=("row", "col"), key_mode="index"),
        validate=True,
    )
    leaf = out[(0, 0)]
    roles = read_roles(validate_schema_if_needed(leaf.as_dataset(copy="none")))
    assert roles == (True, "sample", ("trial",), ("axis",))
    assert leaf.as_dataset(copy="none").attrs["tal"]["core"]["param_coord"] == {"name": "tau"}
    assert leaf.as_dataset(copy="none").attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "sample_size"


def test_combine_decompose_core_006_accessor_decompose_core_self_semantics() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_006_accessor_decompose_core_self_semantics."""
    assembled, _ = _assembled_grid(labels=(("r0", "r1"), ("c0", "c1")))
    opts = CoreDecomposeOptions(core_dims=("row", "col"), key_mode="label")
    out_accessor = assembled.combine.decompose_core(opts=opts, validate=True)
    out_functional = decompose_core(assembled, opts=opts, validate=True)
    _assert_mapping_identical(out_accessor, out_functional)


def test_combine_decompose_core_007_single_numeric_var_required() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_007_single_numeric_var_required."""
    base = _vector_leaf(offset=0.0)
    multi_ds = base.as_dataset(copy="none").copy(deep=True)
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
        _ = decompose_core(multi, opts=CoreDecomposeOptions(core_dims=("axis",)), validate=True)


def test_combine_decompose_core_008_label_mode_duplicate_nan_labels_rejected() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_008_label_mode_duplicate_nan_labels_rejected."""
    dup_nan_ds = _vector_leaf().as_dataset(copy="none").assign_coords({"axis": np.asarray([float("nan"), float("nan"), 1.0], dtype=object)})
    dup_nan = AnalysisObject.from_data(
        dup_nan_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )
    with pytest.raises(ValueError, match="requires unique labels"):
        _ = decompose_core(
            dup_nan,
            opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label"),
            validate=True,
        )


def test_combine_decompose_core_009_opts_required_at_core_surfaces() -> None:
    """ID: COMBINE_DECOMPOSE_CORE_009_opts_required_at_core_surfaces."""
    base = _vector_leaf()
    with pytest.raises(TypeError, match="required keyword-only argument: 'opts'"):
        _ = decompose_core(base, validate=True)
    with pytest.raises(TypeError, match="required keyword-only argument: 'opts'"):
        _ = base.combine.decompose_core(validate=True)
