from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.linalg import Array, Matrix, Vector, Vector3, dot, matmul, norm


def _core_dims(ds: xr.Dataset) -> tuple[str, ...]:
    from tal.core.schema_read import read_roles

    return read_roles(ds)[3]


def _vector_ao(values: np.ndarray, *, axis: str, labels: tuple[object, ...]) -> AnalysisObject:
    ds = xr.Dataset(
        {"x": (("sample", "trial", axis), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            axis: np.asarray(labels, dtype=object),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(axis,),
        validate=True,
    )


def _matrix_ao(
    values: np.ndarray,
    *,
    row: str,
    col: str,
    row_labels: tuple[object, ...] | None = None,
    col_labels: tuple[object, ...] | None = None,
) -> AnalysisObject:
    resolved_row = row_labels if row_labels is not None else tuple(np.arange(values.shape[2], dtype=np.int64))
    resolved_col = col_labels if col_labels is not None else tuple(np.arange(values.shape[3], dtype=np.int64))
    ds = xr.Dataset(
        {"x": (("sample", "trial", row, col), values)},
        coords={
            "sample": np.arange(values.shape[0], dtype=np.int64),
            "trial": np.asarray(["t0", "t1"], dtype=object),
            row: np.asarray(resolved_row, dtype=object),
            col: np.asarray(resolved_col, dtype=object),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(row, col),
        validate=True,
    )


def _scalar_ao(values: np.ndarray, *, var: str) -> AnalysisObject:
    ds = xr.Dataset(
        {var: (("sample", "trial"), values)},
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


def test_linalg_core_022_vector3_constructor_enforces_xyz_len3_invariants() -> None:
    """ID: LINALG_CORE_022_vector3_constructor_enforces_xyz_len3_invariants."""
    ao = _vector_ao(
        np.arange(12, dtype=float).reshape(2, 2, 3),
        axis="axis",
        labels=("x", "y", "z"),
    )
    out = Vector3(ao)
    assert isinstance(out, Vector3)
    assert _core_dims(out.unsafe_data) == ("axis",)
    assert tuple(out.unsafe_data.coords["axis"].to_index().tolist()) == ("x", "y", "z")


def test_linalg_core_023_vector3_from_xyz_assembly_truthful() -> None:
    """ID: LINALG_CORE_023_vector3_from_xyz_assembly_truthful."""
    x_vals = np.arange(4, dtype=float).reshape(2, 2)
    z_vals = (np.arange(4, dtype=float).reshape(2, 2) + 10.0) / 2.0
    x_ao = _scalar_ao(x_vals, var="x")
    z_ao = _scalar_ao(z_vals, var="z")
    out = Vector3.from_xyz(x_ao, 2.0, z_ao, axis="axis", output_var="vec", validate=True)
    assert isinstance(out, Vector3)
    assert _core_dims(out.unsafe_data) == ("axis",)
    assert tuple(out.unsafe_data.coords["axis"].to_index().tolist()) == ("x", "y", "z")
    xr.testing.assert_allclose(out.unsafe_data["vec"].sel(axis="x", drop=True), x_ao.unsafe_data["x"])
    xr.testing.assert_allclose(
        out.unsafe_data["vec"].sel(axis="y", drop=True),
        xr.full_like(x_ao.unsafe_data["x"], 2.0),
    )
    xr.testing.assert_allclose(out.unsafe_data["vec"].sel(axis="z", drop=True), z_ao.unsafe_data["z"])


def test_linalg_core_025_vector3_from_xyz_canonical_semantic_dim_order() -> None:
    """ID: LINALG_CORE_025_vector3_from_xyz_canonical_semantic_dim_order."""
    x_vals = np.arange(4, dtype=float).reshape(2, 2)
    y_vals = x_vals + 1.0
    z_vals = x_vals + 2.0
    x_ao = _scalar_ao(x_vals, var="x")
    y_ao = _scalar_ao(y_vals, var="y")
    z_ao = _scalar_ao(z_vals, var="z")
    out = Vector3.from_xyz(x_ao, y_ao, z_ao, axis="axis", output_var="vec", validate=True)
    assert tuple(out.unsafe_data["vec"].dims) == ("sample", "trial", "axis")
    xr.testing.assert_allclose(out.unsafe_data["vec"].sel(axis="x", drop=True), x_ao.unsafe_data["x"])
    xr.testing.assert_allclose(out.unsafe_data["vec"].sel(axis="y", drop=True), y_ao.unsafe_data["y"])
    xr.testing.assert_allclose(out.unsafe_data["vec"].sel(axis="z", drop=True), z_ao.unsafe_data["z"])


def test_linalg_core_024_vector3_component_accessors_scalar_core_truthful() -> None:
    """ID: LINALG_CORE_024_vector3_component_accessors_scalar_core_truthful."""
    vector3 = Vector3(
        _vector_ao(
            np.arange(12, dtype=float).reshape(2, 2, 3),
            axis="axis",
            labels=("x", "y", "z"),
        )
    )
    out_x = vector3.x
    out_y = vector3.y
    out_z = vector3.z
    assert type(out_x) is Array
    assert type(out_y) is Array
    assert type(out_z) is Array
    assert _core_dims(out_x.unsafe_data) == ()
    assert _core_dims(out_y.unsafe_data) == ()
    assert _core_dims(out_z.unsafe_data) == ()
    xr.testing.assert_allclose(out_x.unsafe_data["x"], vector3.unsafe_data["x"].sel(axis="x", drop=True))
    xr.testing.assert_allclose(out_y.unsafe_data["x"], vector3.unsafe_data["x"].sel(axis="y", drop=True))
    xr.testing.assert_allclose(out_z.unsafe_data["x"], vector3.unsafe_data["x"].sel(axis="z", drop=True))


def test_linalg_hard_066_vector3_requires_single_core_dim() -> None:
    """ID: LINALG_HARD_066_vector3_requires_single_core_dim."""
    matrix_ao = _matrix_ao(np.arange(24, dtype=float).reshape(2, 2, 2, 3), row="row", col="col")
    with pytest.raises(ValueError, match="requires exactly one core dim"):
        _ = Vector3(matrix_ao)


def test_linalg_hard_067_vector3_requires_exact_xyz_labels() -> None:
    """ID: LINALG_HARD_067_vector3_requires_exact_xyz_labels."""
    wrong_labels = _vector_ao(np.arange(12, dtype=float).reshape(2, 2, 3), axis="axis", labels=("a", "b", "c"))
    wrong_size = _vector_ao(
        np.arange(16, dtype=float).reshape(2, 2, 4),
        axis="axis",
        labels=("x", "y", "z", "w"),
    )
    with pytest.raises(ValueError, match="labels must equal"):
        _ = Vector3(wrong_labels)
    with pytest.raises(ValueError, match="must have length 3"):
        _ = Vector3(wrong_size)


def test_linalg_hard_068_vector3_rewrap_invariants_enforced_on_structural_paths() -> None:
    """ID: LINALG_HARD_068_vector3_rewrap_invariants_enforced_on_structural_paths."""
    vector3 = Vector3(
        _vector_ao(
            np.arange(12, dtype=float).reshape(2, 2, 3),
            axis="axis",
            labels=("x", "y", "z"),
        )
    )
    with pytest.raises(ValueError, match="must have length 3"):
        _ = vector3.isel({"axis": slice(0, 2)}, validate=True)
    with pytest.raises(ValueError, match="must have length 3"):
        _ = vector3.isel({"axis": slice(0, 2)}, validate=False)


def test_linalg_hard_069_vector3_binary_ops_do_not_invalid_rewrap_scalar_or_matrix_outputs() -> None:
    """ID: LINALG_HARD_069_vector3_binary_ops_do_not_invalid_rewrap_scalar_or_matrix_outputs."""
    vector3 = Vector3(
        _vector_ao(
            np.arange(12, dtype=float).reshape(2, 2, 3),
            axis="axis",
            labels=("x", "y", "z"),
        )
    )
    matrix_left = Matrix(
        _matrix_ao(
            np.arange(36, dtype=float).reshape(2, 2, 3, 3),
            row="row",
            col="axis",
            col_labels=("x", "y", "z"),
        )
    )
    matrix_right = Matrix(
        _matrix_ao(
            np.arange(24, dtype=float).reshape(2, 2, 3, 2),
            row="axis",
            col="col",
            row_labels=("x", "y", "z"),
        )
    )
    out_dot = dot(vector3, vector3)
    out_norm = norm(vector3)
    out_mv = matmul(matrix_left, vector3)
    out_vm = matmul(vector3, matrix_right)
    out_vv = matmul(vector3, vector3)
    assert type(out_dot) is Array
    assert type(out_norm) is Array
    assert isinstance(out_mv, Vector)
    assert isinstance(out_vm, Vector)
    assert type(out_vv) is Array
    assert not isinstance(out_mv, Vector3)
    assert not isinstance(out_vm, Vector3)


def test_linalg_hard_073_custom_vector3_subclass_uses_builtin_arity_routing() -> None:
    """ID: LINALG_HARD_073_custom_vector3_subclass_uses_builtin_arity_routing."""

    class MyVector3(Vector3):
        pass

    left = MyVector3(
        _vector_ao(
            np.arange(12, dtype=float).reshape(2, 2, 3),
            axis="axis",
            labels=("x", "y", "z"),
        )
    )
    right = MyVector3(
        _vector_ao(
            (np.arange(12, dtype=float).reshape(2, 2, 3) + 1.0) / 3.0,
            axis="axis",
            labels=("x", "y", "z"),
        )
    )
    matrix = Matrix(
        _matrix_ao(
            np.arange(24, dtype=float).reshape(2, 2, 3, 2),
            row="axis",
            col="col",
            row_labels=("x", "y", "z"),
        )
    )
    out_dot = dot(left, right)
    out_norm = norm(left)
    out_matmul = matmul(left, matrix)
    assert type(out_dot) is Array
    assert type(out_norm) is Array
    assert isinstance(out_matmul, Vector)
    assert type(out_matmul) is Vector
    assert not isinstance(out_matmul, MyVector3)


def test_linalg_hard_088_vector3_from_xyz_rejects_all_scalar_components() -> None:
    """ID: LINALG_HARD_088_vector3_from_xyz_rejects_all_scalar_components."""
    with pytest.raises(ValueError, match="at least one AO-like component is required"):
        _ = Vector3.from_xyz(1.0, 2.0, 3.0)


def test_linalg_hard_089_vector3_from_xyz_scalar_promotion_optional_metadata_pruned_deterministically() -> None:
    """ID: LINALG_HARD_089_vector3_from_xyz_scalar_promotion_optional_metadata_pruned_deterministically."""
    x_vals = np.arange(4, dtype=float).reshape(2, 2)
    z_vals = x_vals + 10.0
    x_ao = AnalysisObject.from_data(
        xr.Dataset(
            {"x": (("sample", "trial"), x_vals)},
            coords={
                "sample": np.arange(2, dtype=np.int64),
                "trial": np.asarray(["t0", "t1"], dtype=object),
                "tau": (("trial", "sample"), np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)),
                "sample_size": ("trial", np.asarray([2, 2], dtype=np.int64)),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )
    z_ao = AnalysisObject.from_data(
        xr.Dataset(
            {"z": (("sample", "trial"), z_vals)},
            coords={
                "sample": np.arange(2, dtype=np.int64),
                "trial": np.asarray(["t0", "t1"], dtype=object),
                "tau": (("trial", "sample"), np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float)),
                "sample_size": ("trial", np.asarray([2, 2], dtype=np.int64)),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        param_coord="tau",
        sequence_size_coord="sample_size",
        validate=True,
    )
    out = Vector3.from_xyz(x_ao, 3.0, z_ao, axis="axis", output_var="vec", validate=True)
    tal_core = out.unsafe_data.attrs["tal"]["core"]
    assert "param_coord" not in tal_core
    assert tal_core["validity"]["sequence_size_coord"] == "sample_size"
