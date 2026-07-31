from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject, SchemaError
from tal.core import merge_schema, set_param_coord, set_roles, set_validity


class _HostileKey:
    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return False

    def __repr__(self) -> str:
        raise RuntimeError("repr exploded")

    def __str__(self) -> str:
        raise RuntimeError("str exploded")


def _ds_sample_axis() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("sample", "axis"), np.ones((4, 3)))},
        coords={"sample": [0, 1, 2, 3], "axis": ["x", "y", "z"]},
    )


def _ds_trial_sample_axis() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("trial", "sample", "axis"), np.ones((2, 4, 3)))},
        coords={
            "trial": [0, 1],
            "sample": [0, 1, 2, 3],
            "axis": ["x", "y", "z"],
        },
    )


def _assert_schema_error(
    err: pytest.ExceptionInfo[SchemaError],
    *,
    code: str,
    path: str,
) -> None:
    assert err.value.code == code
    assert err.value.path == path


def _tal_snapshot(ds: xr.Dataset) -> dict:
    return deepcopy(ds.attrs["tal"])


def test_schema_write_roles_001_minimal_success() -> None:
    """ID: SCHEMA_WRITE_ROLES_001_minimal_success."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=())
    roles = ds.attrs["tal"]["core"]["roles"]
    assert roles == {"sequence_dim": "sample", "batch_dims": [], "core_dims": []}


def test_schema_write_roles_002_batch_and_core_success() -> None:
    """ID: SCHEMA_WRITE_ROLES_002_batch_and_core_success."""
    ds = set_roles(
        _ds_trial_sample_axis(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
    )
    roles = ds.attrs["tal"]["core"]["roles"]
    assert roles == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }


def test_schema_write_roles_002b_core_only_without_sequence_success() -> None:
    """ID: SCHEMA_WRITE_ROLES_007_core_only_without_sequence_success."""
    ds = set_roles(_ds_sample_axis(), core_dims=("axis",))
    roles = ds.attrs["tal"]["core"]["roles"]
    assert roles == {
        "batch_dims": [],
        "core_dims": ["axis"],
    }


def test_schema_write_roles_002c_batch_and_core_without_sequence_success() -> None:
    """ID: SCHEMA_WRITE_ROLES_008_batch_and_core_without_sequence_success."""
    ds = set_roles(_ds_trial_sample_axis(), batch_dims=("trial",), core_dims=("axis",))
    roles = ds.attrs["tal"]["core"]["roles"]
    assert roles == {
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }


def test_schema_write_roles_003_returns_new_ao() -> None:
    """ID: SCHEMA_WRITE_ROLES_003_returns_new_ao."""
    ao = AnalysisObject(_ds_sample_axis())
    updated = ao.set_roles(sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    assert updated is not ao
    assert "roles" not in ao.data.attrs["tal"]["core"]
    assert updated.data.attrs["tal"]["core"]["roles"]["core_dims"] == ["axis"]


def test_schema_write_roles_004_atomic_on_failure() -> None:
    """ID: SCHEMA_WRITE_ROLES_004_atomic_on_failure."""
    ao = AnalysisObject(_ds_sample_axis())
    before = _tal_snapshot(ao.data)
    with pytest.raises(SchemaError) as err:
        ao.set_roles(sequence_dim="missing", batch_dims=(), core_dims=())
    _assert_schema_error(
        err,
        code="schema.roles.sequence_dim.not_in_dataset",
        path="tal.core.roles.sequence_dim",
    )
    assert ao.data.attrs["tal"] == before


def test_schema_write_roles_005_partial_preserve_omitted() -> None:
    """ID: SCHEMA_WRITE_ROLES_005_partial_preserve_omitted."""
    ds = set_roles(
        _ds_trial_sample_axis(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
    )
    out = set_roles(ds, sequence_dim="sample")
    roles = out.attrs["tal"]["core"]["roles"]
    assert roles["sequence_dim"] == "sample"
    assert roles["batch_dims"] == ["trial"]
    assert roles["core_dims"] == ["axis"]


def test_schema_write_roles_006_explicit_empty_clears() -> None:
    """ID: SCHEMA_WRITE_ROLES_006_explicit_empty_clears."""
    ds = set_roles(
        _ds_trial_sample_axis(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
    )
    out = set_roles(ds, batch_dims=(), core_dims=())
    roles = out.attrs["tal"]["core"]["roles"]
    assert roles["sequence_dim"] == "sample"
    assert roles["batch_dims"] == []
    assert roles["core_dims"] == []


def test_schema_write_param_001_set_1d_param_coord() -> None:
    """ID: SCHEMA_WRITE_PARAM_001_set_1d_param_coord."""
    ds = _ds_sample_axis().assign_coords(time=("sample", [0.0, 0.1, 0.2, 0.3]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    out = set_param_coord(ds, name="time")
    assert out.attrs["tal"]["core"]["param_coord"]["name"] == "time"


def test_schema_write_param_002_set_nd_param_coord() -> None:
    """ID: SCHEMA_WRITE_PARAM_002_set_nd_param_coord."""
    ds = _ds_trial_sample_axis().assign_coords(
        phase=(("trial", "sample"), [[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]])
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    out = set_param_coord(ds, name="phase")
    assert out.attrs["tal"]["core"]["param_coord"]["name"] == "phase"


def test_schema_write_param_003_clear_param_coord() -> None:
    """ID: SCHEMA_WRITE_PARAM_003_clear_param_coord."""
    ds = _ds_sample_axis().assign_coords(time=("sample", [0.0, 0.1, 0.2, 0.3]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    with_param = set_param_coord(ds, name="time")
    cleared = set_param_coord(with_param, name=None)
    assert "param_coord" not in cleared.attrs["tal"]["core"]


def test_schema_write_param_004_invalid_shape_rejected() -> None:
    """ID: SCHEMA_WRITE_PARAM_004_invalid_shape_rejected."""
    ds = _ds_trial_sample_axis().assign_coords(
        bad=(("sample", "trial"), np.zeros((4, 2), dtype="float64"))
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        set_param_coord(ds, name="bad")
    _assert_schema_error(
        err,
        code="schema.param_coord.dims.invalid",
        path="tal.core.param_coord.name",
    )


def test_schema_write_param_005_atomic_on_failure() -> None:
    """ID: SCHEMA_WRITE_PARAM_005_atomic_on_failure."""
    ds = _ds_trial_sample_axis().assign_coords(
        phase=(("trial", "sample"), np.zeros((2, 4), dtype="float64")),
        bad=(("sample", "trial"), np.zeros((4, 2), dtype="float64")),
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    ao = AnalysisObject(set_param_coord(ds, name="phase"))
    before = _tal_snapshot(ao.data)
    with pytest.raises(SchemaError) as err:
        ao.set_param_coord(name="bad")
    _assert_schema_error(
        err,
        code="schema.param_coord.dims.invalid",
        path="tal.core.param_coord.name",
    )
    assert ao.data.attrs["tal"] == before


def test_schema_write_param_006_non_numeric_param_coord_rejected() -> None:
    """ID: SCHEMA_WRITE_PARAM_006_non_numeric_param_coord_rejected."""
    ds = _ds_trial_sample_axis().assign_coords(
        phase=(("trial", "sample"), np.asarray([["a", "b", "c", "d"], ["e", "f", "g", "h"]], dtype=object))
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        set_param_coord(ds, name="phase")
    _assert_schema_error(
        err,
        code="schema.param_coord.dtype.invalid",
        path="tal.core.param_coord.name",
    )


def test_schema_write_validity_001_set_validity_batched() -> None:
    """ID: SCHEMA_WRITE_VALIDITY_001_set_validity_batched."""
    ds = _ds_trial_sample_axis().assign_coords(group_size=("trial", [4, 3]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    out = set_validity(ds, sequence_size_coord="group_size")
    validity = out.attrs["tal"]["core"]["validity"]
    assert validity == {"sequence_size_coord": "group_size", "layout": "left_packed"}


def test_schema_write_validity_002_set_validity_unbatched_scalar() -> None:
    """ID: SCHEMA_WRITE_VALIDITY_002_set_validity_unbatched_scalar."""
    ds = _ds_sample_axis().assign_coords(group_size=np.int64(4))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    out = set_validity(ds, sequence_size_coord="group_size")
    assert out.attrs["tal"]["core"]["validity"]["sequence_size_coord"] == "group_size"


def test_schema_write_validity_003_clear_validity() -> None:
    """ID: SCHEMA_WRITE_VALIDITY_003_clear_validity."""
    ds = _ds_trial_sample_axis().assign_coords(group_size=("trial", [4, 3]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with_validity = set_validity(ds, sequence_size_coord="group_size")
    cleared = set_validity(with_validity, sequence_size_coord=None)
    assert "validity" not in cleared.attrs["tal"]["core"]


def test_schema_write_validity_004_layout_invalid_rejected() -> None:
    """ID: SCHEMA_WRITE_VALIDITY_004_layout_invalid_rejected."""
    ds = _ds_trial_sample_axis().assign_coords(group_size=("trial", [4, 3]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size", layout="bad")  # type: ignore[arg-type]
    _assert_schema_error(
        err,
        code="schema.validity.layout.invalid",
        path="tal.core.validity.layout",
    )


def test_schema_write_validity_005_dims_invalid_rejected() -> None:
    """ID: SCHEMA_WRITE_VALIDITY_005_dims_invalid_rejected."""
    ds = _ds_trial_sample_axis().assign_coords(group_size=("sample", [0, 1, 2, 3]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size")
    _assert_schema_error(
        err,
        code="schema.validity.sequence_size_coord.dims.invalid",
        path="tal.core.validity.sequence_size_coord",
    )


def test_schema_write_validity_006_values_out_of_bounds_rejected() -> None:
    """ID: SCHEMA_WRITE_VALIDITY_006_values_out_of_bounds_rejected."""
    ds = _ds_trial_sample_axis().assign_coords(group_size=("trial", [4, 5]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size")
    _assert_schema_error(
        err,
        code="schema.validity.sequence_size_coord.values.invalid",
        path="tal.core.validity.sequence_size_coord",
    )


def test_schema_write_validity_007_values_non_integer_rejected() -> None:
    """ID: SCHEMA_WRITE_VALIDITY_007_values_non_integer_rejected."""
    ds = _ds_trial_sample_axis().assign_coords(group_size=("trial", [3.0, 2.5]))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size")
    _assert_schema_error(
        err,
        code="schema.validity.sequence_size_coord.values.invalid",
        path="tal.core.validity.sequence_size_coord",
    )


def test_schema_write_validity_008_chunked_values_rejected() -> None:
    """ID: SCHEMA_WRITE_VALIDITY_008_chunked_values_rejected."""
    da = pytest.importorskip("dask.array")
    ds = _ds_trial_sample_axis().assign_coords(
        group_size=("trial", da.from_array(np.asarray([3, 2], dtype="int64"), chunks=1))
    )
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size")
    _assert_schema_error(
        err,
        code="schema.validity.sequence_size_coord.values.invalid",
        path="tal.core.validity.sequence_size_coord",
    )
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["coord"] == "group_size"
    assert err.value.actual["reason"] == "chunked coordinate not schema-value-validatable"


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.asarray([3 + 0j, 2 + 1j], dtype=np.complex128), id="complex"),
        pytest.param(np.asarray([True, False], dtype=bool), id="bool"),
        pytest.param(np.asarray([3, 2], dtype="timedelta64[s]"), id="timedelta"),
        pytest.param(
            np.asarray(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
            id="datetime",
        ),
    ],
)
def test_schema_write_validity_009_non_count_dtypes_rejected(values: np.ndarray) -> None:
    """ID: SCHEMA_WRITE_VALIDITY_009_non_count_dtypes_rejected."""
    ds = _ds_trial_sample_axis().assign_coords(group_size=("trial", values))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        set_validity(ds, sequence_size_coord="group_size")
    _assert_schema_error(
        err,
        code="schema.validity.sequence_size_coord.values.invalid",
        path="tal.core.validity.sequence_size_coord",
    )


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint16, np.float32, np.float64])
def test_schema_write_validity_010_real_integer_valued_dtypes_accepted(dtype: type[np.generic]) -> None:
    """ID: SCHEMA_WRITE_VALIDITY_010_real_integer_valued_dtypes_accepted."""
    values = np.asarray([4, 3], dtype=dtype)
    ds = _ds_trial_sample_axis().assign_coords(group_size=("trial", values))
    ds = set_roles(ds, sequence_dim="sample", batch_dims=("trial",), core_dims=("axis",))
    out = set_validity(ds, sequence_size_coord="group_size")
    np.testing.assert_array_equal(out.coords["group_size"].to_numpy(), values)


def test_schema_write_merge_001_patch_root_valid() -> None:
    """ID: SCHEMA_WRITE_MERGE_001_patch_root_valid."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    out = merge_schema(ds, {"ext": {"spatial": {"frames": {"parent": "world"}}}})
    assert out.attrs["tal"]["ext"]["spatial"]["frames"]["parent"] == "world"


def test_schema_write_merge_002_patch_root_invalid() -> None:
    """ID: SCHEMA_WRITE_MERGE_002_patch_root_invalid."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        merge_schema(ds, {"tal": {"core": {}}})
    _assert_schema_error(err, code="schema.patch.root.invalid", path="tal")


def test_schema_write_merge_003_recursive_merge_mapping() -> None:
    """ID: SCHEMA_WRITE_MERGE_003_recursive_merge_mapping."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    ds = merge_schema(ds, {"ext": {"spatial": {"frames": {"parent": "world", "child": "imu"}}}})
    out = merge_schema(ds, {"ext": {"spatial": {"frames": {"child": "cam"}, "rep": "quat"}}})
    frames = out.attrs["tal"]["ext"]["spatial"]["frames"]
    assert frames == {"parent": "world", "child": "cam"}
    assert out.attrs["tal"]["ext"]["spatial"]["rep"] == "quat"


def test_schema_write_merge_004_none_deletes_key() -> None:
    """ID: SCHEMA_WRITE_MERGE_004_none_deletes_key."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    ds = merge_schema(ds, {"ext": {"spatial": {"frames": {"parent": "world"}, "rep": "quat"}}})
    out = merge_schema(ds, {"ext": {"spatial": {"rep": None}}})
    assert "rep" not in out.attrs["tal"]["ext"]["spatial"]


def test_schema_write_merge_005_list_replace_semantics() -> None:
    """ID: SCHEMA_WRITE_MERGE_005_list_replace_semantics."""
    ds = set_roles(
        _ds_trial_sample_axis(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
    )
    out = merge_schema(ds, {"core": {"roles": {"batch_dims": []}}})
    roles = out.attrs["tal"]["core"]["roles"]
    assert roles["batch_dims"] == []
    assert roles["core_dims"] == ["axis"]


def test_schema_write_merge_006_atomic_on_failure() -> None:
    """ID: SCHEMA_WRITE_MERGE_006_atomic_on_failure."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    ao = AnalysisObject(ds)
    before = _tal_snapshot(ao.data)
    with pytest.raises(SchemaError) as err:
        ao.merge_schema({"core": {"roles": {"sequence_dim": "missing"}}})
    _assert_schema_error(
        err,
        code="schema.roles.sequence_dim.not_in_dataset",
        path="tal.core.roles.sequence_dim",
    )
    assert ao.data.attrs["tal"] == before


def test_schema_write_merge_007_non_string_patch_key_raises_schema_error() -> None:
    """ID: SCHEMA_WRITE_MERGE_007_non_string_patch_key_raises_schema_error."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        merge_schema(ds, {"core": {1: "bad"}}, validate=True)
    _assert_schema_error(err, code="schema.core.unknown_key", path="tal.core.1")
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "int"


def test_schema_write_merge_008_hostile_patch_key_validate_true_raises_schema_error() -> None:
    """ID: SCHEMA_WRITE_MERGE_008_hostile_patch_key_validate_true_raises_schema_error."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        merge_schema(ds, {"core": {_HostileKey(): "bad"}}, validate=True)
    _assert_schema_error(err, code="schema.core.unknown_key", path="tal.core.<_HostileKey>")
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "_HostileKey"


def test_schema_write_009_non_dataset_input_rejected_consistently() -> None:
    """ID: SCHEMA_WRITE_009_non_dataset_input_rejected_consistently."""
    da = xr.DataArray(np.asarray([1.0, 2.0]), dims=("sample",), coords={"sample": [0, 1]})
    writers = (
        lambda validate: set_roles(
            da,
            sequence_dim="sample",
            batch_dims=(),
            core_dims=(),
            validate=validate,
        ),
        lambda validate: set_param_coord(da, name="sample", validate=validate),
        lambda validate: set_validity(da, sequence_size_coord="sample", validate=validate),
        lambda validate: merge_schema(da, {"core": {}}, validate=validate),
    )
    for validate in (False, True):
        for write in writers:
            with pytest.raises(TypeError) as err:
                write(validate)
            assert "expects xr.Dataset" in str(err.value)
