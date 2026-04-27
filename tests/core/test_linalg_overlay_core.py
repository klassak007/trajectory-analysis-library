from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema_read import read_roles, validate_schema_if_needed
from tal.linalg import Array, CoreOverlayOptions, Matrix, Vector, Vector3, overlay_core


def _vector_leaf(
    *,
    offset: float = 0.0,
    axis_labels: tuple[object, ...] = ("a", "b", "c", "d"),
) -> AnalysisObject:
    axis_size = len(axis_labels)
    values = np.arange(2 * 2 * axis_size, dtype=float).reshape(2, 2, axis_size) + offset
    ds = xr.Dataset(
        {"x": (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.asarray(["s0", "s1"], dtype=object),
            "trial": np.asarray(["t0", "t1"], dtype=object),
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


def _vector3_leaf(*, offset: float = 0.0, axis_labels: tuple[object, ...] = ("x", "y", "z")) -> AnalysisObject:
    axis_size = len(axis_labels)
    values = np.arange(2 * 2 * axis_size, dtype=float).reshape(2, 2, axis_size) + offset
    ds = xr.Dataset(
        {"x": (("sample", "trial", "axis"), values)},
        coords={
            "sample": np.asarray(["s0", "s1"], dtype=object),
            "trial": np.asarray(["t0", "t1"], dtype=object),
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


def _core_dims(ao: AnalysisObject) -> tuple[str, ...]:
    return read_roles(validate_schema_if_needed(ao.unsafe_data))[3]


def test_linalg_core_028_overlay_core_functional_and_method_parity() -> None:
    """ID: LINALG_CORE_028_overlay_core_functional_and_method_parity."""
    base = Array(_vector_leaf(offset=0.0))
    patch = _vector_leaf(offset=100.0, axis_labels=("b", "d"))
    opts = CoreOverlayOptions(core_dim="axis")
    out_fn = overlay_core(base, [patch], opts=opts, validate=True)
    out_method = base.overlay_core([patch], opts=opts, validate=True)
    assert type(out_fn) is Array
    assert type(out_method) is Array
    xr.testing.assert_identical(out_fn.unsafe_data, out_method.unsafe_data)


def test_linalg_hard_082_overlay_core_plain_ao_base_fallback_to_array() -> None:
    """ID: LINALG_HARD_082_overlay_core_plain_ao_base_fallback_to_array."""
    base = _vector_leaf(offset=0.0)
    patch = _vector_leaf(offset=100.0, axis_labels=("b",))
    out = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    assert type(out) is Array


def test_linalg_hard_083_overlay_core_base_custom_subclass_preserved_when_representable() -> None:
    """ID: LINALG_HARD_083_overlay_core_base_custom_subclass_preserved_when_representable."""

    class MyArray(Array):
        pass

    base = MyArray(_vector_leaf(offset=0.0))
    patch = _vector_leaf(offset=100.0, axis_labels=("a",))
    out = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    assert type(out) is MyArray


def test_linalg_hard_084_overlay_core_vector_matrix_vector3_base_types_remain_valid() -> None:
    """ID: LINALG_HARD_084_overlay_core_vector_matrix_vector3_base_types_remain_valid."""
    vector_base = Vector(_vector_leaf(offset=0.0))
    vector_out = overlay_core(
        vector_base,
        [_vector_leaf(offset=100.0, axis_labels=("b",))],
        opts=CoreOverlayOptions(core_dim="axis"),
        validate=True,
    )
    assert type(vector_out) is Vector
    assert _core_dims(vector_out) == ("axis",)

    matrix_base = Matrix(_matrix_leaf(offset=0.0))
    matrix_out = overlay_core(
        matrix_base,
        [_matrix_leaf(offset=100.0, row_labels=("r1",), col_labels=("c0", "c1"))],
        opts=CoreOverlayOptions(core_dim="row"),
        validate=True,
    )
    assert type(matrix_out) is Matrix
    assert _core_dims(matrix_out) == ("row", "col")

    vector3_base = Vector3(_vector3_leaf(offset=0.0))
    vector3_out = overlay_core(
        vector3_base,
        [_vector3_leaf(offset=100.0, axis_labels=("y",))],
        opts=CoreOverlayOptions(core_dim="axis"),
        validate=True,
    )
    assert type(vector3_out) is Vector3
    assert vector3_out.unsafe_data.coords["axis"].values.tolist() == ["x", "y", "z"]


def test_linalg_hard_085_overlay_core_opts_required_signature_alignment() -> None:
    """ID: LINALG_HARD_085_overlay_core_opts_required_signature_alignment."""
    base = Array(_vector_leaf(offset=0.0))
    patch = _vector_leaf(offset=100.0, axis_labels=("b",))
    with pytest.raises(TypeError, match="required keyword-only argument: 'opts'"):
        _ = overlay_core(base, [patch], validate=True)
    with pytest.raises(TypeError, match="required keyword-only argument: 'opts'"):
        _ = base.overlay_core([patch], validate=True)


def test_linalg_hard_086_overlay_core_nan_and_integer_semantics_preserved_through_linalg_boundary() -> None:
    """ID: LINALG_HARD_086_overlay_core_nan_and_integer_semantics_preserved_through_linalg_boundary."""
    base_nan = Array(_vector_leaf(offset=0.0, axis_labels=("a", float("nan"), "c")))
    patch_nan = _vector_leaf(offset=100.0, axis_labels=(float("nan"),))
    out_nan = overlay_core(base_nan, [patch_nan], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    expected_nan = base_nan.unsafe_data["x"].copy(deep=True)
    expected_nan.loc[{"axis": [patch_nan.unsafe_data.get_index("axis")[0]]}] = patch_nan.unsafe_data["x"]
    xr.testing.assert_allclose(out_nan.unsafe_data["x"], expected_nan)

    base_int = Array(
        AnalysisObject.from_data(
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
    )
    patch_int = AnalysisObject.from_data(
        xr.Dataset(
            {"x": (("sample", "trial", "axis"), np.full((2, 2, 1), 77, dtype=np.int64))},
            coords={
                "sample": np.asarray(["s0", "s1"], dtype=object),
                "trial": np.asarray(["t0", "t1"], dtype=object),
                "axis": np.asarray(["c"], dtype=object),
            },
        ),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    out_int = overlay_core(base_int, [patch_int], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    assert out_int.unsafe_data["x"].dtype.kind in {"i", "u"}
    assert np.all(out_int.unsafe_data["x"].sel(axis="c").data == 77)


def test_linalg_hard_087_overlay_core_mixed_nan_scalar_types_canonicalized_through_linalg_boundary() -> None:
    """ID: LINALG_HARD_087_overlay_core_mixed_nan_scalar_types_canonicalized_through_linalg_boundary."""
    base = Array(_vector_leaf(offset=0.0, axis_labels=(np.float64(np.nan), "a", "c")))
    patch = _vector_leaf(offset=100.0, axis_labels=(float("nan"),))
    out = overlay_core(base, [patch], opts=CoreOverlayOptions(core_dim="axis"), validate=True)
    assert type(out) is Array
    expected = base.unsafe_data["x"].copy(deep=True)
    expected[{"axis": 0}] = patch.unsafe_data["x"].isel(axis=0)
    xr.testing.assert_allclose(out.unsafe_data["x"], expected)
