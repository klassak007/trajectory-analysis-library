from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema_read import read_roles
from tal.linalg import Array, MatmulOptions, add, inv, matmul, sub
from tal.linalg.plan import build_array_plan


def _matrix_ao(
    values: np.ndarray,
    *,
    trial_labels: tuple[str, ...] = ("t0", "t1"),
    row: str = "row",
    col: str = "col",
) -> AnalysisObject:
    sample = np.arange(values.shape[0], dtype=np.int64)
    coords: dict[str, object] = {
        "sample": sample,
        "trial": np.asarray(trial_labels, dtype=object),
        row: np.arange(values.shape[2], dtype=np.int64),
        col: np.arange(values.shape[3], dtype=np.int64),
    }
    ds = xr.Dataset(
        {"x": (("sample", "trial", row, col), values)},
        coords=coords,
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=True,
    )


def _matrix_ao_missing_sequence_declared(
    values: np.ndarray,
    *,
    row: str = "row",
    col: str = "col",
) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("trial", row, col), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row: np.arange(values.shape[1], dtype=np.int64),
            col: np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=False,
    )


def _matrix_ao_missing_batch_declared(
    values: np.ndarray,
    *,
    row: str = "row",
    col: str = "col",
) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", row, col), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row: np.arange(values.shape[1], dtype=np.int64),
            col: np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=False,
    )


def _vector_ao(values: np.ndarray, *, axis: str = "axis") -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", "trial", axis), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            axis: np.arange(values.shape[2], dtype=np.int64),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(axis,),
        validate=True,
    )


def _scalar_ao(values: np.ndarray) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", "trial"), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        validate=True,
    )


def _param_vector_ao(
    values: np.ndarray,
    *,
    sample_labels: np.ndarray,
    param_values: np.ndarray,
    axis: str = "axis",
) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", axis), values)},
        coords={
            "sample": sample_labels,
            axis: np.arange(values.shape[1], dtype=np.int64),
            "time_s": ("sample", param_values),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=(axis,),
        param_coord="time_s",
        validate=True,
    )


def _undeclared_matrix_ao(
    values: np.ndarray,
    *,
    row: str = "row",
    col: str = "col",
    row_labels: tuple[int, ...] = (0, 1, 2),
    col_labels: tuple[int, ...] = (0, 1, 2),
) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": ((row, col), values)},
        coords={row: np.asarray(row_labels, dtype=np.int64), col: np.asarray(col_labels, dtype=np.int64)},
    )
    return AnalysisObject(ds)


def _matrix_ao_with_optional(
    values: np.ndarray,
    *,
    row: str,
    col: str,
    tau_offset: float = 0.0,
) -> AnalysisObject:
    tau = np.arange(values.shape[0] * values.shape[1], dtype=float).reshape(values.shape[1], values.shape[0])
    ds = xr.Dataset(
        {"x": (("sample", "trial", row, col), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row: np.arange(values.shape[2], dtype=np.int64),
            col: np.arange(values.shape[3], dtype=np.int64),
            "tau": (("trial", "sample"), tau + tau_offset),
            "sample_size": ("trial", np.full(values.shape[1], values.shape[0], dtype=np.int64)),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )


def test_linalg_core_001_array_core_role_declaration_roundtrip() -> None:
    """ID: LINALG_CORE_001_array_core_role_declaration_roundtrip."""
    base = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="r", col="c")
    arr = Array(base).set_core_dims("r", "c")
    declared, _, _, core_dims = xr_roles(arr.unsafe_data)
    assert declared
    assert core_dims == ("r", "c")


def test_linalg_hard_096_array_typed_lifecycle_parity() -> None:
    """ID: LINALG_HARD_096_array_typed_lifecycle_parity."""
    base = _vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis")
    from_ao = Array(base)
    from_ds = Array(base.unsafe_data)

    assert isinstance(from_ao, Array)
    assert isinstance(from_ds, Array)
    xr.testing.assert_identical(from_ao.unsafe_data, from_ds.unsafe_data)
    assert xr_roles(from_ao.unsafe_data) == xr_roles(from_ds.unsafe_data)


def test_linalg_hard_100_array_core_dims_init_options_parity() -> None:
    """ID: LINALG_HARD_100_array_core_dims_init_options_parity."""
    base = _vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis")
    direct = Array(base, core_dims=("axis",))
    via_setter = Array(base).set_core_dims("axis")
    xr.testing.assert_identical(direct.unsafe_data, via_setter.unsafe_data)

    raw = xr.Dataset(
        {"x": (("sample", "axis"), np.arange(6, dtype=float).reshape(2, 3))},
        coords={"sample": [0, 1], "axis": [0, 1, 2]},
    )
    with pytest.raises(ValueError, match="Array requires declared roles before setting core dims"):
        _ = Array(raw, core_dims=("axis",))


def test_linalg_core_002_vector_matrix_core_shape_validation() -> None:
    """ID: LINALG_CORE_002_vector_matrix_core_shape_validation."""
    base = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="r", col="c")
    arr = Array(base)
    assert xr_roles(arr.set_vector_axis("c").unsafe_data)[3] == ("c",)
    assert xr_roles(arr.set_matrix_axes("r", "c").unsafe_data)[3] == ("r", "c")
    with pytest.raises(ValueError):
        arr.set_matrix_axes("r", "r")
    with pytest.raises(ValueError):
        arr.set_core_dims("r", "r")
    with pytest.raises(TypeError):
        arr.set_core_dims("r", 1)  # type: ignore[arg-type]


def test_linalg_core_003_matmul_strict_role_contraction() -> None:
    """ID: LINALG_CORE_003_matmul_strict_role_contraction."""
    left_vals = np.arange(36, dtype=float).reshape(2, 2, 3, 3)
    right_vals = (np.arange(24, dtype=float).reshape(2, 2, 3, 2) + 1.0) / 5.0
    left = Array(_matrix_ao(left_vals, row="row", col="mid"))
    right = Array(_matrix_ao(right_vals, row="mid", col="out"))
    out = matmul(left, right)
    expected = xr.dot(left.unsafe_data["x"], right.unsafe_data["x"], dim=["mid"]).rename("datavar")
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)
    assert list(out.unsafe_data.data_vars) == ["datavar"]
    assert not any(name.endswith("_matmul_x") for name in out.unsafe_data.data_vars)
    assert xr_roles(out.unsafe_data)[3] == ("row", "out")


def test_linalg_core_004_matmul_label_alignment_not_positional() -> None:
    """ID: LINALG_CORE_004_matmul_label_alignment_not_positional."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")
    right_ds = right.unsafe_data.transpose("trial", "sample", "mid", "out")
    right_ao = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("mid", "out"),
        validate=True,
    )
    out = matmul(Array(left), Array(right_ao))
    expected = xr.dot(left.unsafe_data["x"], right_ao.unsafe_data["x"], dim=["mid"]).rename("datavar")
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)


def test_linalg_core_005_matmul_ambiguous_roles_fail_closed() -> None:
    """ID: LINALG_CORE_005_matmul_ambiguous_roles_fail_closed."""
    raw = xr.Dataset({"x": (("a", "b"), np.arange(6, dtype=float).reshape(2, 3))})
    left = Array(raw)
    right = Array(raw)
    with pytest.raises(ValueError):
        _ = left @ right

    left2 = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid"))
    bad = Array(_matrix_ao(np.arange(16, dtype=float).reshape(2, 2, 2, 2), row="other", col="out"))
    with pytest.raises(ValueError):
        _ = left2 @ bad

    multi = xr.Dataset(
        {
            "a": (("sample", "trial", "row", "col"), np.ones((2, 2, 2, 2))),
            "b": (("sample", "trial", "row", "col"), np.ones((2, 2, 2, 2))),
        },
        coords={
            "sample": [0, 1],
            "trial": ["t0", "t1"],
            "row": [0, 1],
            "col": [0, 1],
        },
    )
    multi_ao = AnalysisObject.from_data(
        multi,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        validate=True,
    )
    with pytest.raises(ValueError):
        _ = Array(multi_ao) @ left2
    with pytest.raises(ValueError):
        _ = matmul(left2, left2, opts=MatmulOptions(strict_core=False))


def test_linalg_core_006_core_role_aliases_parity() -> None:
    """ID: LINALG_CORE_006_core_role_aliases_parity."""
    base = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="r", col="c")
    arr = Array(base)
    xr.testing.assert_identical(arr.set_core_dims("r", "c").unsafe_data, arr.as_core("r", "c").unsafe_data)
    xr.testing.assert_identical(arr.set_vector_axis("c").unsafe_data, arr.axis("c").unsafe_data)
    xr.testing.assert_identical(arr.set_matrix_axes("r", "c").unsafe_data, arr.rc("r", "c").unsafe_data)


def test_linalg_hard_001_operand_var_must_contain_declared_semantic_dims() -> None:
    """ID: LINALG_HARD_001_operand_var_must_contain_declared_semantic_dims."""
    left_ds = xr.Dataset(
        {"x": (("sample", "trial", "row"), np.arange(12, dtype=float).reshape(2, 2, 3))},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "row": np.arange(3, dtype=np.int64),
            "mid": np.arange(3, dtype=np.int64),
        },
    )
    left = AnalysisObject.from_data(
        left_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "mid"),
        validate=True,
    )
    right = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")
    with pytest.raises(ValueError, match="missing required semantic dims"):
        _ = matmul(Array(left), Array(right))


def test_linalg_hard_002_param_coord_and_validity_canonicalized_after_matmul() -> None:
    """ID: LINALG_HARD_002_param_coord_and_validity_canonicalized_after_matmul."""
    from tal.core.schema_read import read_param_coord_name, read_sequence_size_coord_name

    left = _matrix_ao_with_optional(
        np.arange(36, dtype=float).reshape(2, 2, 3, 3),
        row="row",
        col="mid",
    )
    right = _matrix_ao_with_optional(
        np.arange(24, dtype=float).reshape(2, 2, 3, 2),
        row="mid",
        col="out",
    )
    out = matmul(Array(left), Array(right))
    assert read_param_coord_name(out.unsafe_data) == "tau"
    assert read_sequence_size_coord_name(out.unsafe_data) == "sample_size"
    assert out.unsafe_data.coords["tau"].dims == ("trial", "sample")
    assert out.unsafe_data.coords["sample_size"].dims == ("trial",)
    np.testing.assert_array_equal(
        out.unsafe_data.coords["tau"].values,
        left.unsafe_data.coords["tau"].values,
    )


def test_linalg_hard_003_allow_vector_row_convention_option_enforced() -> None:
    """ID: LINALG_HARD_003_allow_vector_row_convention_option_enforced."""
    left = Array(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="mid"))
    right = Array(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out"))
    with pytest.raises(ValueError, match="allow_vector_row_convention=False"):
        _ = matmul(left, right, opts=MatmulOptions(allow_vector_row_convention=False))

    out = matmul(left, right, opts=MatmulOptions(allow_vector_row_convention=True))
    expected = xr.dot(left.unsafe_data["x"], right.unsafe_data["x"], dim=["mid"]).rename("datavar")
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)
    assert xr_roles(out.unsafe_data)[3] == ("out",)


def test_linalg_hard_004_set_matrix_axes_non_string_rejected_at_typed_boundary() -> None:
    """ID: LINALG_HARD_004_set_matrix_axes_non_string_rejected_at_typed_boundary."""
    base = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="r", col="c")
    arr = Array(base)
    with pytest.raises(TypeError, match="row_dim must be str"):
        arr.set_matrix_axes(1, "c")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="col_dim must be str"):
        arr.set_matrix_axes("r", 2)  # type: ignore[arg-type]


def test_linalg_hard_005_array_is_analysis_object_subclass() -> None:
    """ID: LINALG_HARD_005_array_is_analysis_object_subclass."""
    assert issubclass(Array, AnalysisObject)
    base = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="r", col="c")
    arr = Array(base)
    out = arr.set_roles(sequence_dim="sample", batch_dims=("trial",), core_dims=("r", "c"), validate=True)
    assert isinstance(out, Array)


def test_linalg_hard_006_matmul_left_subclass_wins_result_type() -> None:
    """ID: LINALG_HARD_006_matmul_left_subclass_wins_result_type."""

    class MyArray(Array):
        pass

    left = MyArray(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid"))
    right = Array(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out"))
    out = matmul(left, right)
    assert isinstance(out, MyArray)
    out_op = left @ right
    assert isinstance(out_op, MyArray)


def test_linalg_hard_007_matmul_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_007_matmul_plain_ao_inputs_fallback_to_array."""
    left_ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    right_ao = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="mid", col="out")
    out = matmul(left_ao, right_ao)
    assert type(out) is Array

    out_ds = matmul(left_ao.unsafe_data, right_ao.unsafe_data)
    assert type(out_ds) is Array


def test_linalg_hard_008_set_core_dims_single_role_read_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: LINALG_HARD_008_set_core_dims_single_role_read_path."""
    import tal.linalg.lifecycle as lifecycle_module

    calls = {"count": 0}
    original = lifecycle_module.read_roles

    def _counted_read_roles(ds: xr.Dataset):
        calls["count"] += 1
        return original(ds)

    monkeypatch.setattr(lifecycle_module, "read_roles", _counted_read_roles)
    base = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="r", col="c")
    _ = Array(base).set_core_dims("r", "c")
    assert calls["count"] == 1


def test_linalg_hard_009_array_constructor_core_dims_rejects_raw_xarray_without_declared_roles() -> None:
    """ID: LINALG_HARD_009_array_constructor_core_dims_rejects_raw_xarray_without_declared_roles."""
    raw = xr.Dataset(
        {"x": (("sample", "axis"), np.arange(6, dtype=float).reshape(3, 2))},
        coords={"sample": [0, 1, 2], "axis": [0, 1]},
    )
    with pytest.raises(ValueError, match="requires declared roles before setting core dims"):
        _ = Array(raw, core_dims=("axis",))


def test_linalg_hard_010_array_constructor_core_dims_accepts_declared_roles_input() -> None:
    """ID: LINALG_HARD_010_array_constructor_core_dims_accepts_declared_roles_input."""
    base = AnalysisObject.from_data(
        xr.Dataset(
            {"x": (("sample", "axis"), np.arange(6, dtype=float).reshape(3, 2))},
            coords={"sample": [0, 1, 2], "axis": [0, 1]},
        ),
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
        validate=True,
    )
    arr = Array(base, core_dims=("axis",))
    assert xr_roles(arr.unsafe_data) == (True, "sample", (), ("axis",))


def test_linalg_core_007_add_strict_core_dims_match() -> None:
    """ID: LINALG_CORE_007_add_strict_core_dims_match."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(_matrix_ao((np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 2.0) / 10.0, row="row", col="col"))
    out = add(left, right)
    expected = (left.unsafe_data["x"] + right.unsafe_data["x"]).rename("datavar")
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)
    assert list(out.unsafe_data.data_vars) == ["datavar"]
    assert not any("_add_" in name for name in out.unsafe_data.data_vars)
    assert xr_roles(out.unsafe_data)[3] == ("row", "col")


def test_linalg_core_008_sub_strict_core_dims_match() -> None:
    """ID: LINALG_CORE_008_sub_strict_core_dims_match."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(_matrix_ao((np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 5.0) / 7.0, row="row", col="col"))
    out = sub(left, right)
    expected = (left.unsafe_data["x"] - right.unsafe_data["x"]).rename("datavar")
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected)
    assert list(out.unsafe_data.data_vars) == ["datavar"]
    assert not any("_sub_" in name for name in out.unsafe_data.data_vars)
    assert xr_roles(out.unsafe_data)[3] == ("row", "col")


def test_linalg_core_009_add_sub_label_alignment_not_positional() -> None:
    """ID: LINALG_CORE_009_add_sub_label_alignment_not_positional."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    right = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 1.0, row="row", col="col")
    right_ds = right.unsafe_data.transpose("trial", "sample", "row", "col")
    right_ao = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        validate=True,
    )
    add_out = add(Array(left), Array(right_ao))
    sub_out = sub(Array(left), Array(right_ao))
    expected_add = (left.unsafe_data["x"] + right_ao.unsafe_data["x"]).rename("datavar")
    expected_sub = (left.unsafe_data["x"] - right_ao.unsafe_data["x"]).rename("datavar")
    xr.testing.assert_allclose(add_out.unsafe_data["datavar"], expected_add)
    xr.testing.assert_allclose(sub_out.unsafe_data["datavar"], expected_sub)


def test_linalg_core_010_add_sub_alias_operator_parity() -> None:
    """ID: LINALG_CORE_010_add_sub_alias_operator_parity."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 3.0, row="row", col="col"))
    xr.testing.assert_identical(add(left, right).unsafe_data, (left + right).unsafe_data)
    xr.testing.assert_identical(sub(left, right).unsafe_data, (left - right).unsafe_data)

    left_ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    xr.testing.assert_identical(add(left_ao, right).unsafe_data, (left_ao + right).unsafe_data)
    xr.testing.assert_identical(sub(left_ao, right).unsafe_data, (left_ao - right).unsafe_data)


def test_linalg_hard_011_add_sub_mismatched_core_dims_fail_closed() -> None:
    """ID: LINALG_HARD_011_add_sub_mismatched_core_dims_fail_closed."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="row", col="out"))
    with pytest.raises(ValueError, match="matching core_dims under core_policy='strict'"):
        _ = add(left, right)
    with pytest.raises(ValueError, match="require matching core_dims"):
        _ = sub(left, right)


def test_linalg_hard_012_add_sub_plain_ao_inputs_fallback_to_array() -> None:
    """ID: LINALG_HARD_012_add_sub_plain_ao_inputs_fallback_to_array."""
    left_ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    right_ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 2.0, row="row", col="col")
    out_add = add(left_ao, right_ao)
    out_sub = sub(left_ao.unsafe_data, right_ao.unsafe_data)
    assert type(out_add) is Array
    assert type(out_sub) is Array


def test_linalg_hard_013_add_sub_left_subclass_wins_result_type() -> None:
    """ID: LINALG_HARD_013_add_sub_left_subclass_wins_result_type."""

    class MyArray(Array):
        pass

    left = MyArray(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 1.0, row="row", col="col"))
    assert isinstance(add(left, right), MyArray)
    assert isinstance(sub(left, right), MyArray)
    assert isinstance(left + right, MyArray)
    assert isinstance(left - right, MyArray)


def test_linalg_hard_014_add_sub_multi_var_rejected() -> None:
    """ID: LINALG_HARD_014_add_sub_multi_var_rejected."""
    multi = xr.Dataset(
        {
            "a": (("sample", "trial", "row", "col"), np.ones((2, 2, 2, 2))),
            "b": (("sample", "trial", "row", "col"), np.ones((2, 2, 2, 2))),
        },
        coords={
            "sample": [0, 1],
            "trial": ["t0", "t1"],
            "row": [0, 1],
            "col": [0, 1],
        },
    )
    multi_ao = AnalysisObject.from_data(
        multi,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        validate=True,
    )
    with pytest.raises(ValueError, match="exactly one data variable"):
        _ = add(Array(multi_ao), Array(multi_ao))
    with pytest.raises(ValueError, match="exactly one data variable"):
        _ = sub(Array(multi_ao), Array(multi_ao))


def test_linalg_hard_015_add_sub_undeclared_operands_default_to_core_only_semantics() -> None:
    """ID: LINALG_HARD_015_add_sub_undeclared_operands_default_to_core_only_semantics."""
    raw = xr.Dataset(
        {"x": (("sample", "row"), np.ones((2, 2), dtype=float))},
        coords={"sample": [0, 1], "row": [0, 1]},
    )
    ao = AnalysisObject(raw)
    out_add = add(ao, ao)
    out_sub = sub(ao, ao)
    xr.testing.assert_allclose(out_add.unsafe_data["datavar"], xr.ufuncs.add(raw["x"], raw["x"]))
    xr.testing.assert_allclose(out_sub.unsafe_data["datavar"], xr.ufuncs.subtract(raw["x"], raw["x"]))
    declared, sequence_dim, batch_dims, core_dims = read_roles(out_add.unsafe_data)
    assert declared is True
    assert sequence_dim is None
    assert batch_dims == ()
    assert core_dims == ("sample", "row")


def test_topo_core_004_linalg_plan_uses_core_topology_touchpoint() -> None:
    """ID: TOPO_CORE_004_linalg_plan_uses_core_topology_touchpoint."""
    a = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="r0", col="r1")
    b = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="r1", col="r2")
    c = _matrix_ao(np.arange(96, dtype=float).reshape(2, 2, 2, 12), row="r2", col="r3")
    plan = build_array_plan([a, b, c], owner="linalg.topology")
    assert plan.sequence_dim == "sample"
    assert plan.batch_dims == ("trial",)
    assert len(plan.operands) == 3


def test_topo_core_009_unary_topology_path_uses_shared_touchpoint() -> None:
    """ID: TOPO_CORE_009_unary_topology_path_uses_shared_touchpoint."""
    ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="mid")
    plan = build_array_plan([ao], owner="linalg.topology")
    assert len(plan.operands) == 1
    assert plan.sequence_dim == "sample"
    assert plan.batch_dims == ("trial",)


def test_topo_core_012_unary_operation_class_topology_preservation_parity() -> None:
    """ID: TOPO_CORE_012_unary_operation_class_topology_preservation_parity."""
    values = np.asarray(
        [
            [[[2.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 1.0]]],
            [[[4.0, 0.0], [0.0, 1.0]], [[5.0, 0.0], [0.0, 1.0]]],
        ],
        dtype=float,
    )
    matrix = Array(_matrix_ao(values, row="row", col="col"))
    out = inv(matrix)
    _, sequence_dim, batch_dims, _ = read_roles(out.unsafe_data)
    assert sequence_dim == "sample"
    assert batch_dims == ("trial",)


def test_topo_core_013_linalg_require_declared_roles_false_unary_undeclared_supported() -> None:
    """ID: TOPO_CORE_013_linalg_require_declared_roles_false_unary_undeclared_supported."""
    ao = _undeclared_matrix_ao(np.arange(9, dtype=float).reshape(3, 3))
    plan = build_array_plan(
        [ao],
        owner="linalg.topology",
        require_declared_roles=False,
    )
    assert plan.sequence_dim is None
    assert plan.batch_dims == ()
    assert len(plan.operands) == 1
    assert plan.operands[0].data.dims == ("row", "col")


def test_topo_core_014_linalg_require_declared_roles_false_binary_undeclared_supported() -> None:
    """ID: TOPO_CORE_014_linalg_require_declared_roles_false_binary_undeclared_supported."""
    left = _undeclared_matrix_ao(np.arange(9, dtype=float).reshape(3, 3))
    right = _undeclared_matrix_ao((np.arange(9, dtype=float).reshape(3, 3) + 1.0) / 10.0)
    plan = build_array_plan(
        [left, right],
        owner="linalg.topology",
        require_declared_roles=False,
    )
    assert plan.sequence_dim is None
    assert plan.batch_dims == ()
    assert len(plan.operands) == 2
    xr.testing.assert_identical(plan.operands[0].data, left.unsafe_data["x"])
    xr.testing.assert_identical(plan.operands[1].data, right.unsafe_data["x"])


def test_topo_hard_009_linalg_strict_exact_rejects_core_label_mismatch_before_kernel() -> None:
    """ID: TOPO_HARD_009_linalg_strict_exact_rejects_core_label_mismatch_before_kernel."""
    left = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    right_ds = _matrix_ao(
        (np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 5.0) / 11.0,
        row="row",
        col="col",
    ).unsafe_data.assign_coords(row=np.asarray([10, 11, 12], dtype=np.int64))
    right = AnalysisObject.from_data(
        right_ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("row", "col"),
        validate=True,
    )
    with pytest.raises(ValueError, match="strict exact policy"):
        _ = build_array_plan([left, right], owner="linalg.topology")


def test_topo_hard_010_linalg_elementwise_no_silent_inner_core_alignment_under_strict_exact() -> None:
    """ID: TOPO_HARD_010_linalg_elementwise_no_silent_inner_core_alignment_under_strict_exact."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right_ds = _matrix_ao(
        (np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 3.0) / 7.0,
        row="row",
        col="col",
    ).unsafe_data.assign_coords(row=np.asarray([1, 2, 3], dtype=np.int64))
    right = Array(
        AnalysisObject.from_data(
            right_ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("row", "col"),
            validate=True,
        )
    )
    with pytest.raises(ValueError, match="strict exact policy"):
        _ = add(left, right)


def test_topo_core_011_linalg_require_declared_roles_false_mixed_declared_undeclared_uses_core_only_inference() -> None:
    """ID: TOPO_CORE_900_linalg_require_declared_roles_false_mixed_declared_undeclared_uses_core_only_inference."""
    declared = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    undeclared = _undeclared_matrix_ao(np.arange(9, dtype=float).reshape(3, 3), row="row", col="col")
    plan = build_array_plan(
        [declared, undeclared],
        owner="linalg.topology",
        require_declared_roles=False,
    )
    assert plan.sequence_dim == "sample"
    assert plan.batch_dims == ("trial",)
    assert plan.output_core_dims == ("row", "col")
    assert plan.operands[0].roles_declared is True
    assert plan.operands[1].roles_declared is False


def test_bcast_core_007_linalg_optin_broadcast_path_uses_shared_core_policy() -> None:
    """ID: BCAST_CORE_007_linalg_optin_broadcast_path_uses_shared_core_policy."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(
        _matrix_ao_missing_sequence_declared(
            (np.arange(18, dtype=float).reshape(2, 3, 3) + 1.0) / 10.0,
            row="row",
            col="col",
        )
    )
    out = add(left.b(), right)
    expected = left.unsafe_data["x"] + right.unsafe_data["x"]
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected.rename("datavar"))


def test_bcast_core_028_semantic_default_enabled_for_approved_ordinary_elementwise_families() -> None:
    """ID: BCAST_CORE_028_semantic_default_enabled_for_approved_ordinary_elementwise_families."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(
        _matrix_ao_missing_sequence_declared(
            (np.arange(18, dtype=float).reshape(2, 3, 3) + 2.0) / 11.0,
            row="row",
            col="col",
        )
    )
    out = add(left, right)
    expected = left.unsafe_data["x"] + right.unsafe_data["x"]
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected.rename("datavar"))


def test_bcast_core_032_b_helper_remains_valid_on_semantic_default_families() -> None:
    """ID: BCAST_CORE_032_b_helper_remains_valid_on_semantic_default_families."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(
        _matrix_ao_missing_sequence_declared(
            (np.arange(18, dtype=float).reshape(2, 3, 3) + 3.0) / 13.0,
            row="row",
            col="col",
        )
    )
    out_default = add(left, right)
    out_optin = add(left.b(), right)
    xr.testing.assert_identical(out_default.unsafe_data, out_optin.unsafe_data)


def test_bcast_core_033_unary_default_path_preserves_topology_without_synthetic_materialization() -> None:
    """ID: BCAST_CORE_033_unary_default_path_preserves_topology_without_synthetic_materialization."""
    ao = _matrix_ao_missing_sequence_declared(
        np.arange(18, dtype=float).reshape(2, 3, 3),
        row="row",
        col="mid",
    )
    plan = build_array_plan([ao], owner="linalg.topology")
    assert plan.sequence_dim == "sample"
    assert plan.batch_dims == ("trial",)
    assert tuple(plan.operands[0].data.dims) == ("trial", "row", "mid")
    semantic = build_array_plan([ao.b()], owner="linalg.topology")
    assert tuple(semantic.operands[0].data.dims) == ("trial", "row", "mid")


def test_bcast_core_013_nary_mixed_intent_merge_is_deterministic() -> None:
    """ID: BCAST_CORE_013_nary_mixed_intent_merge_is_deterministic."""
    a = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="r0", col="r1")
    b = _matrix_ao_missing_sequence_declared(
        (np.arange(18, dtype=float).reshape(2, 3, 3) + 1.0) / 9.0,
        row="r0",
        col="r1",
    )
    c = _matrix_ao_missing_batch_declared(
        (np.arange(18, dtype=float).reshape(2, 3, 3) + 2.0) / 7.0,
        row="r0",
        col="r1",
    )
    plan_left = build_array_plan([a.b(), b, c], owner="linalg.topology")
    plan_right = build_array_plan([a, b.b(), c], owner="linalg.topology")
    assert plan_left.sequence_dim == plan_right.sequence_dim == "sample"
    assert plan_left.batch_dims == plan_right.batch_dims == ("trial",)
    assert tuple(plan_left.operands[2].data.dims) == tuple(plan_right.operands[2].data.dims)


def test_bcast_hard_002_unexpected_intent_carrier_type_fail_closed() -> None:
    """ID: BCAST_HARD_002_unexpected_intent_carrier_type_fail_closed."""
    ao = _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col")
    object.__setattr__(ao, "_broadcast_intent", "bad-carrier")
    with pytest.raises(TypeError, match="invalid broadcast intent carrier type"):
        _ = build_array_plan([ao], owner="linalg.topology")


def test_bcast_hard_004_owner_prefixed_errors_preserved_on_optin_paths() -> None:
    """ID: BCAST_HARD_004_owner_prefixed_errors_preserved_on_optin_paths."""
    left = Array(
        _matrix_ao(
            np.arange(36, dtype=float).reshape(2, 2, 3, 3),
            row="row",
            col="col",
            trial_labels=("t0", "t1"),
        )
    )
    right = Array(
        _matrix_ao(
            np.arange(36, dtype=float).reshape(2, 2, 3, 3),
            row="row",
            col="col",
            trial_labels=("t0", "t2"),
        )
    )
    with pytest.raises(ValueError, match=r"^linalg\.add:"):
        _ = add(left.b(), right)


def test_bcast_core_034_sequence_exact_and_batch_exact_defaults_remain_deterministic() -> None:
    """ID: BCAST_CORE_034_sequence_exact_and_batch_exact_defaults_remain_deterministic."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right_ds = _matrix_ao(
        (np.arange(36, dtype=float).reshape(2, 2, 3, 3) + 1.0) / 5.0,
        row="row",
        col="col",
    ).unsafe_data.assign_coords(sample=np.asarray([10, 11], dtype=np.int64))
    right = Array(
        AnalysisObject.from_data(
            right_ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("row", "col"),
            validate=True,
        )
    )
    with pytest.raises(ValueError, match="requires exact 'sample' labels under sequence/batch policy"):
        _ = add(left, right)


def test_bcast_hard_020_missing_shared_semantic_dim_labels_fail_closed_when_unmaterializable() -> None:
    """ID: BCAST_HARD_020_missing_shared_semantic_dim_labels_fail_closed_when_unmaterializable."""
    left = Array(
        _matrix_ao_missing_sequence_declared(
            np.arange(18, dtype=float).reshape(2, 3, 3),
            row="row",
            col="col",
        )
    )
    right = Array(
        _matrix_ao_missing_sequence_declared(
            (np.arange(18, dtype=float).reshape(2, 3, 3) + 1.0) / 7.0,
            row="row",
            col="col",
        )
    )
    with pytest.raises(
        ValueError,
        match=r"^linalg\.add: array plan semantic broadcast requires at least one operand carrying dim 'sample'",
    ):
        _ = add(left, right)


def test_bcast_hard_021_extra_undeclared_nonsemantic_dims_fail_closed_under_default_policy() -> None:
    """ID: BCAST_HARD_021_extra_undeclared_nonsemantic_dims_fail_closed_under_default_policy."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right_ds = xr.Dataset(
        {"x": (("sample", "trial", "camera", "row", "col"), np.arange(72, dtype=float).reshape(2, 2, 2, 3, 3))},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "camera": np.asarray(["c0", "c1"], dtype=object),
            "row": np.arange(3, dtype=np.int64),
            "col": np.arange(3, dtype=np.int64),
        },
    )
    right = Array(
        AnalysisObject.from_data(
            right_ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("row", "col"),
            validate=False,
        )
    )
    with pytest.raises(ValueError, match="semantic broadcast only permits missing sequence/batch dims"):
        _ = add(left, right)


def test_bcast_hard_022_topology_mismatch_still_fails_fast_under_semantic_default() -> None:
    """ID: BCAST_HARD_022_topology_mismatch_still_fails_fast_under_semantic_default."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(
        AnalysisObject.from_data(
            _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col").unsafe_data.rename(
                {"sample": "time"}
            ),
            sequence_dim="time",
            batch_dims=("trial",),
            core_dims=("row", "col"),
            validate=True,
        )
    )
    with pytest.raises(ValueError, match="requires matching sequence_dim"):
        _ = add(left, right)


def test_bcast_hard_023_core_dim_ambiguity_rejected_under_e3a_default_policy() -> None:
    """ID: BCAST_HARD_023_core_dim_ambiguity_rejected_under_e3a_default_policy."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(_matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 3, 2), row="row", col="out"))
    with pytest.raises(ValueError, match="matching core_dims under core_policy='strict'"):
        _ = add(left, right)


def test_bcast_hard_025_owner_prefixed_boundaries_preserved_after_default_flip() -> None:
    """ID: BCAST_HARD_025_owner_prefixed_boundaries_preserved_after_default_flip."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right = Array(
        AnalysisObject.from_data(
            _matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col").unsafe_data.rename(
                {"sample": "time"}
            ),
            sequence_dim="time",
            batch_dims=("trial",),
            core_dims=("row", "col"),
            validate=True,
        )
    )
    with pytest.raises(ValueError, match=r"^linalg\.add:"):
        _ = add(left, right)


def test_bcast_core_042_numpy_named_core_policy_supports_scalar_and_matching_named_core_dims() -> None:
    """ID: BCAST_CORE_042_numpy_named_core_policy_supports_scalar_and_matching_named_core_dims."""
    left = Array(_scalar_ao(np.arange(4, dtype=float).reshape(2, 2)))
    right = Array(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    with pytest.raises(ValueError, match="core_dims cardinality"):
        _ = add(left, right)
    out = add(left.a(core_policy="numpy_named"), right)
    expected = left.unsafe_data["x"] + right.unsafe_data["x"]
    xr.testing.assert_allclose(out.unsafe_data["datavar"], expected.rename("datavar"))
    assert xr_roles(out.unsafe_data)[3] == ("axis",)


def test_bcast_core_044_conflicting_intent_merge_fails_closed() -> None:
    """ID: BCAST_CORE_044_conflicting_intent_merge_fails_closed."""
    left = Array(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    right = Array(_vector_ao((np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 3.0, axis="axis"))
    with pytest.raises(ValueError, match="conflicting alignment intents"):
        _ = add(left.a(on="sequence"), right.a(on="param", sequence_join=None))


def test_bcast_hard_031_extra_undeclared_nonsemantic_dims_rejected_under_alignment_intent() -> None:
    """ID: BCAST_HARD_031_extra_undeclared_nonsemantic_dims_rejected_under_alignment_intent."""
    left = Array(_matrix_ao(np.arange(36, dtype=float).reshape(2, 2, 3, 3), row="row", col="col"))
    right_ds = xr.Dataset(
        {"x": (("sample", "trial", "camera", "row", "col"), np.arange(72, dtype=float).reshape(2, 2, 2, 3, 3))},
        coords={
            "sample": np.arange(2, dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            "camera": np.asarray(["c0", "c1"], dtype=object),
            "row": np.arange(3, dtype=np.int64),
            "col": np.arange(3, dtype=np.int64),
        },
    )
    right = Array(
        AnalysisObject.from_data(
            right_ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("row", "col"),
            validate=False,
        )
    )
    with pytest.raises(ValueError, match="semantic broadcast only permits missing sequence/batch dims"):
        _ = add(left.a(), right)


def test_bcast_hard_033_alignment_intent_paths_preserve_owner_prefixed_error_boundaries() -> None:
    """ID: BCAST_HARD_033_alignment_intent_paths_preserve_owner_prefixed_error_boundaries."""
    left = Array(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    right = Array(_vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis"))
    with pytest.raises(ValueError, match=r"^linalg\.add:"):
        _ = add(left.a(on="sequence"), right.a(on="param", sequence_join=None))


def test_bcast_hard_034_no_hidden_interpolation_in_a_intent_paths() -> None:
    """ID: BCAST_HARD_034_no_hidden_interpolation_in_a_intent_paths."""
    left = Array(
        _param_vector_ao(
            np.arange(6, dtype=float).reshape(3, 2),
            sample_labels=np.asarray([0, 1, 2], dtype=np.int64),
            param_values=np.asarray([0.0, 0.5, 1.0], dtype=float),
        )
    )
    right = Array(
        _param_vector_ao(
            np.arange(6, dtype=float).reshape(3, 2) + 1.0,
            sample_labels=np.asarray(["a", "b", "c"], dtype=object),
            param_values=np.asarray([0.0, 0.75, 1.0], dtype=float),
        )
    )
    with pytest.raises(ValueError, match="requires exact 'time_s' labels"):
        _ = add(left.a(on="param", sequence_join=None), right)


def xr_roles(ds: xr.Dataset) -> tuple[bool, str | None, tuple[str, ...], tuple[str, ...]]:
    from tal.core.schema_read import read_roles

    return read_roles(ds)
