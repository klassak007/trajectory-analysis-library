from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, CoreOverlayOptions, overlay_core
from tal.core.schema_read import read_roles, validate_schema_if_needed


def _vector_leaf(
    *,
    offset: float = 0.0,
    axis_labels: tuple[object, ...] = ("a", "b", "c", "d"),
    sample_labels: tuple[object, ...] = ("s0", "s1"),
    trial_labels: tuple[object, ...] = ("t0", "t1"),
) -> AnalysisObject:
    axis_size = len(axis_labels)
    values = np.arange(2 * 2 * axis_size, dtype=float).reshape(2, 2, axis_size) + offset
    ds = xr.Dataset(
        {"x": (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.asarray(sample_labels, dtype=object),
            "trial": np.asarray(trial_labels, dtype=object),
            "axis": np.asarray(axis_labels, dtype=object),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )


def _matrix_leaf(
    *,
    offset: float = 0.0,
    row_labels: tuple[object, ...] = ("r0", "r1"),
    col_labels: tuple[object, ...] = ("c0", "c1"),
) -> AnalysisObject:
    row_size = len(row_labels)
    col_size = len(col_labels)
    values = np.arange(2 * 2 * row_size * col_size, dtype=float).reshape(2, 2, row_size, col_size) + offset
    ds = xr.Dataset(
        {"x": (("sample", "trial", "row", "col"), values)},
        coords={
            "sample": np.asarray(["s0", "s1"], dtype=object),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "row": np.asarray(row_labels, dtype=object),
            "col": np.asarray(col_labels, dtype=object),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        validate=True,
    )


def _roles(ao: AnalysisObject) -> tuple[bool, str | None, tuple[str, ...], tuple[str, ...]]:
    return read_roles(validate_schema_if_needed(ao.unsafe_data))


def test_combine_overlay_core_001_single_patch_label_overlay_semantics() -> None:
    """ID: COMBINE_OVERLAY_CORE_001_single_patch_label_overlay_semantics."""
    base = _vector_leaf(offset=0.0)
    patch = _vector_leaf(offset=100.0, axis_labels=("b", "d"))
    out = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    expected = base.unsafe_data["x"].copy(deep=True)
    expected.loc[{"axis": ["b", "d"]}] = patch.unsafe_data["x"]
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)
    assert out.unsafe_data.coords["axis"].values.tolist() == base.unsafe_data.coords["axis"].values.tolist()
    assert _roles(out)[3] == ("axis",)


def test_combine_overlay_core_002_exact_alignment_label_safe_not_positional() -> None:
    """ID: COMBINE_OVERLAY_CORE_002_exact_alignment_label_safe_not_positional."""
    base = _vector_leaf(offset=0.0, sample_labels=("s0", "s1"))
    patch = _vector_leaf(offset=100.0, axis_labels=("b",), sample_labels=("s1", "s0"))
    with pytest.raises(ValueError, match="exact"):
        _ = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)


def test_combine_overlay_core_003_requires_declared_roles_and_target_core_dim() -> None:
    """ID: COMBINE_OVERLAY_CORE_003_requires_declared_roles_and_target_core_dim."""
    base = _vector_leaf(offset=0.0)
    patch = _vector_leaf(offset=10.0, axis_labels=("b",))
    raw_ds = base.unsafe_data.copy(deep=True)
    raw_ds.attrs = {}
    raw = AnalysisObject(raw_ds)
    with pytest.raises(ValueError, match="declared roles"):
        _ = overlay_core(raw, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    with pytest.raises(ValueError, match="not in declared core_dims"):
        _ = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="row"), validate=True)


def test_combine_overlay_core_004_non_target_core_labels_must_match() -> None:
    """ID: COMBINE_OVERLAY_CORE_004_non_target_core_labels_must_match."""
    base = _matrix_leaf(offset=0.0, row_labels=("r0", "r1"), col_labels=("c0", "c1"))
    patch = _matrix_leaf(offset=10.0, row_labels=("r0",), col_labels=("c0", "cx"))
    with pytest.raises(ValueError, match="non-target core dim"):
        _ = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="row"), validate=True)


def test_combine_overlay_core_005_patch_labels_must_exist_in_base() -> None:
    """ID: COMBINE_OVERLAY_CORE_005_patch_labels_must_exist_in_base."""
    base = _vector_leaf(offset=0.0)
    patch = _vector_leaf(offset=10.0, axis_labels=("b", "z"))
    with pytest.raises(ValueError, match="not present in base target axis"):
        _ = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)


def test_combine_overlay_core_006_patch_overlap_error_policy_fail_closed() -> None:
    """ID: COMBINE_OVERLAY_CORE_006_patch_overlap_error_policy_fail_closed."""
    base = _vector_leaf(offset=0.0)
    patch_a = _vector_leaf(offset=10.0, axis_labels=("b", "c"))
    patch_b = _vector_leaf(offset=20.0, axis_labels=("c",))
    with pytest.raises(ValueError, match="overlapping patch labels"):
        _ = overlay_core(
            base,
            [patch_a, patch_b],
            opts=CoreOverlayOptions(core_dim="axis", on_overlap="error"),
            validate=True,
        )


def test_combine_overlay_core_007_patch_overlap_replace_policy_last_wins() -> None:
    """ID: COMBINE_OVERLAY_CORE_007_patch_overlap_replace_policy_last_wins."""
    base = _vector_leaf(offset=0.0)
    patch_a = _vector_leaf(offset=10.0, axis_labels=("b", "c"))
    patch_b = _vector_leaf(offset=20.0, axis_labels=("c",))
    out = overlay_core(
        base,
        [patch_a, patch_b],
        opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace"),
        validate=True,
    )
    expected = base.unsafe_data["x"].copy(deep=True)
    expected.loc[{"axis": ["b", "c"]}] = patch_a.unsafe_data["x"]
    expected.loc[{"axis": ["c"]}] = patch_b.unsafe_data["x"]
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)


def test_combine_overlay_core_008_base_and_patch_target_labels_unique_required() -> None:
    """ID: COMBINE_OVERLAY_CORE_008_base_and_patch_target_labels_unique_required."""
    base_dup = _vector_leaf(offset=0.0, axis_labels=("a", "a", "c", "d"))
    patch = _vector_leaf(offset=10.0, axis_labels=("a",))
    with pytest.raises(ValueError, match="base labels"):
        _ = overlay_core(base_dup, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    patch_dup = _vector_leaf(offset=10.0, axis_labels=("b", "b"))
    with pytest.raises(ValueError, match="patch index 0 labels"):
        _ = overlay_core(_vector_leaf(offset=0.0), [patch_dup], opts=CoreOverlayOptions(core_dim="axis"), validate=True)


def test_combine_overlay_core_009_accessor_overlay_core_self_semantics() -> None:
    """ID: COMBINE_OVERLAY_CORE_009_accessor_overlay_core_self_semantics."""
    base = _vector_leaf(offset=0.0)
    patch_a = _vector_leaf(offset=10.0, axis_labels=("a",))
    patch_b = _vector_leaf(offset=20.0, axis_labels=("d",))
    opts = CoreOverlayOptions(core_dim="axis")
    out_accessor = base.combine.overlay_core([patch_a, patch_b], opts=opts, validate=True)
    out_functional = overlay_core(base, [patch_a, patch_b], opts=opts, validate=True)
    xr.testing.assert_identical(out_accessor.unsafe_data, out_functional.unsafe_data)


def test_combine_overlay_core_010_single_numeric_var_required() -> None:
    """ID: COMBINE_OVERLAY_CORE_010_single_numeric_var_required."""
    base = _vector_leaf(offset=0.0)
    patch_ds = _vector_leaf(offset=10.0, axis_labels=("b",)).unsafe_data.copy(deep=True)
    patch_ds["y"] = patch_ds["x"] + 1.0
    patch = AnalysisObject.from_data(
        patch_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    with pytest.raises(ValueError, match="exactly one data variable"):
        _ = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)


def test_combine_overlay_core_011_nan_target_labels_patch_applied() -> None:
    """ID: COMBINE_OVERLAY_CORE_011_nan_target_labels_patch_applied."""
    nan_label = float("nan")
    base = _vector_leaf(offset=0.0, axis_labels=("a", nan_label, "c"))
    patch = _vector_leaf(offset=100.0, axis_labels=(float("nan"),))
    out = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    expected = base.unsafe_data["x"].copy(deep=True)
    expected.loc[{"axis": [patch.unsafe_data.get_index("axis")[0]]}] = patch.unsafe_data["x"]
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)


def test_combine_overlay_core_012_overlap_error_composite_nan_labels_fail_closed() -> None:
    """ID: COMBINE_OVERLAY_CORE_012_overlap_error_composite_nan_labels_fail_closed."""
    label_nan_a = frozenset({("k", float("nan"))})
    label_nan_b = frozenset({("k", float("nan"))})
    label_other = frozenset({("k", 1.0)})
    base = _vector_leaf(offset=0.0, axis_labels=(label_nan_a, label_other))
    patch_a = _vector_leaf(offset=10.0, axis_labels=(label_nan_a,))
    patch_b = _vector_leaf(offset=20.0, axis_labels=(label_nan_b,))
    with pytest.raises(ValueError, match="overlapping patch labels"):
        _ = overlay_core(
            base,
            [patch_a, patch_b],
            opts=CoreOverlayOptions(core_dim="axis", on_overlap="error"),
            validate=True,
        )


def test_combine_overlay_core_013_replace_mode_composite_nan_last_wins_deterministic() -> None:
    """ID: COMBINE_OVERLAY_CORE_013_replace_mode_composite_nan_last_wins_deterministic."""
    label_nan_a = frozenset({("k", float("nan"))})
    label_nan_b = frozenset({("k", float("nan"))})
    label_other = frozenset({("k", 1.0)})
    base = _vector_leaf(offset=0.0, axis_labels=(label_nan_a, label_other))
    patch_a = _vector_leaf(offset=10.0, axis_labels=(label_nan_a,))
    patch_b = _vector_leaf(offset=20.0, axis_labels=(label_nan_b,))
    out = overlay_core(
        base,
        [patch_a, patch_b],
        opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace"),
        validate=True,
    )
    expected = base.unsafe_data["x"].copy(deep=True)
    expected[{"axis": 0}] = patch_a.unsafe_data["x"].isel(axis=0)
    expected[{"axis": 0}] = patch_b.unsafe_data["x"].isel(axis=0)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)


def test_combine_overlay_core_014_integer_overlay_preserves_integer_dtype() -> None:
    """ID: COMBINE_OVERLAY_CORE_014_integer_overlay_preserves_integer_dtype."""
    base = AnalysisObject.from_data(
        xr.Dataset(
            {"x": (("sample", "trial", "axis"), np.arange(16, dtype=np.int64).reshape(2, 2, 4))},
            coords={
                "sample": np.asarray(["s0", "s1"], dtype=object),
                "trial": np.asarray(["t0", "t1"], dtype=object),
                "axis": np.asarray(["a", "b", "c", "d"], dtype=object),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    patch = AnalysisObject.from_data(
        xr.Dataset(
            {"x": (("sample", "trial", "axis"), np.full((2, 2, 1), 99, dtype=np.int64))},
            coords={
                "sample": np.asarray(["s0", "s1"], dtype=object),
                "trial": np.asarray(["t0", "t1"], dtype=object),
                "axis": np.asarray(["b"], dtype=object),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    out = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    assert out.unsafe_data["x"].dtype.kind in {"i", "u"}
    assert np.all(out.unsafe_data["x"].sel(axis="b").data == 99)


def test_combine_overlay_core_015_mixed_nan_scalar_types_subset_equivalent() -> None:
    """ID: COMBINE_OVERLAY_CORE_015_mixed_nan_scalar_types_subset_equivalent."""
    base_nan = np.float64(np.nan)
    patch_nan = float("nan")
    base = _vector_leaf(offset=0.0, axis_labels=(base_nan, "a", "c"))
    patch = _vector_leaf(offset=100.0, axis_labels=(patch_nan,))
    out = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    expected = base.unsafe_data["x"].copy(deep=True)
    expected[{"axis": 0}] = patch.unsafe_data["x"].isel(axis=0)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)


def test_combine_overlay_core_016_mixed_nan_scalar_types_overlap_equivalent_under_canonical_keys() -> None:
    """ID: COMBINE_OVERLAY_CORE_016_mixed_nan_scalar_types_overlap_equivalent_under_canonical_keys."""
    label_nan_float = frozenset({("k", float("nan"))})
    label_nan_np = frozenset({("k", np.float64(np.nan))})
    label_other = frozenset({("k", 1.0)})
    base = _vector_leaf(offset=0.0, axis_labels=(label_nan_float, label_other))
    patch_a = _vector_leaf(offset=10.0, axis_labels=(label_nan_float,))
    patch_b = _vector_leaf(offset=20.0, axis_labels=(label_nan_np,))
    with pytest.raises(ValueError, match="overlapping patch labels"):
        _ = overlay_core(
            base,
            [patch_a, patch_b],
            opts=CoreOverlayOptions(core_dim="axis", on_overlap="error"),
            validate=True,
        )
    out = overlay_core(
        base,
        [patch_a, patch_b],
        opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace"),
        validate=True,
    )
    expected = base.unsafe_data["x"].copy(deep=True)
    expected[{"axis": 0}] = patch_a.unsafe_data["x"].isel(axis=0)
    expected[{"axis": 0}] = patch_b.unsafe_data["x"].isel(axis=0)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)
