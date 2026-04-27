import numpy as np
import pytest
import xarray as xr

from tal.core import SchemaError, merge_schema, set_roles, validate_schema

CANONICAL_CODES = {
    "schema.not_mapping",
    "schema.core.not_mapping",
    "schema.unknown_key",
    "schema.ext.namespace.invalid",
    "schema.roles.not_mapping",
    "schema.param_coord.not_mapping",
    "schema.validity.not_mapping",
    "schema.version.invalid",
    "schema.core.unknown_key",
    "schema.roles.unknown_key",
    "schema.param_coord.unknown_key",
    "schema.validity.unknown_key",
    "schema.roles.sequence_dim.missing",
    "schema.roles.sequence_dim.invalid",
    "schema.roles.sequence_dim.not_in_dataset",
    "schema.roles.batch_dims.invalid",
    "schema.roles.batch_dims.duplicate",
    "schema.roles.batch_dims.not_in_dataset",
    "schema.roles.core_dims.invalid",
    "schema.roles.core_dims.duplicate",
    "schema.roles.core_dims.not_in_dataset",
    "schema.roles.overlap",
    "schema.param_coord.sequence_dim.missing",
    "schema.param_coord.name.missing",
    "schema.param_coord.name.invalid",
    "schema.param_coord.not_found",
    "schema.param_coord.dims.invalid",
    "schema.param_coord.dtype.invalid",
    "schema.validity.sequence_dim.missing",
    "schema.validity.sequence_size_coord.missing",
    "schema.validity.sequence_size_coord.invalid",
    "schema.validity.sequence_size_coord.not_found",
    "schema.validity.sequence_size_coord.dims.invalid",
    "schema.validity.sequence_size_coord.values.invalid",
    "schema.validity.layout.invalid",
    "schema.patch.root.invalid",
    "schema.patch.type.invalid",
}


class _HostileRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr exploded")


def _ds_sample_axis() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("sample", "axis"), np.ones((3, 3)))},
        coords={"sample": [0, 1, 2], "axis": ["x", "y", "z"]},
    )


def _trigger_error() -> SchemaError:
    ds = _ds_sample_axis()
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    return err.value


def test_schema_error_001_required_fields_present() -> None:
    """ID: SCHEMA_ERROR_001_required_fields_present."""
    err = _trigger_error()
    assert isinstance(err.code, str)
    assert isinstance(err.path, str)
    assert hasattr(err, "expected")
    assert hasattr(err, "actual")
    assert isinstance(err.hint, str)


def test_schema_error_002_str_contains_key_context() -> None:
    """ID: SCHEMA_ERROR_002_str_contains_key_context."""
    err = _trigger_error()
    text = str(err)
    assert err.code in text
    assert err.path in text
    assert err.hint in text


def test_schema_error_003_code_set_membership() -> None:
    """ID: SCHEMA_ERROR_003_code_set_membership."""
    codes: list[str] = []

    with pytest.raises(SchemaError) as err1:
        validate_schema(_ds_sample_axis())
    codes.append(err1.value.code)

    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    with pytest.raises(SchemaError) as err2:
        merge_schema(ds, {"tal": {"core": {}}})
    codes.append(err2.value.code)

    ds = _ds_sample_axis()
    ds.attrs["tal"] = {"version": 1, "core": {"roles": {"sequence_dim": "sample", "batch_dims": [], "core_dims": []}, "validity": {"layout": "left_packed"}}}
    with pytest.raises(SchemaError) as err3:
        validate_schema(ds)
    codes.append(err3.value.code)

    assert set(codes).issubset(CANONICAL_CODES)


def test_schema_error_004_path_format_dotted() -> None:
    """ID: SCHEMA_ERROR_004_path_format_dotted."""
    err = _trigger_error()
    assert "." in err.path
    assert " " not in err.path
    assert err.path.startswith("tal.")


def test_schema_error_005_hostile_repr_in_validation_is_wrapped() -> None:
    """ID: SCHEMA_ERROR_005_hostile_repr_in_validation_is_wrapped."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {"version": _HostileRepr(), "core": {}}
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    assert err.value.code == "schema.version.invalid"
    assert "<unrepr:_HostileRepr>" in str(err.value)


def test_schema_error_006_hostile_repr_in_writer_is_wrapped() -> None:
    """ID: SCHEMA_ERROR_006_hostile_repr_in_writer_is_wrapped."""
    with pytest.raises(SchemaError) as err:
        set_roles(_ds_sample_axis(), sequence_dim=_HostileRepr(), batch_dims=(), core_dims=())  # type: ignore[arg-type]
    assert err.value.code == "schema.roles.sequence_dim.invalid"
    assert "<unrepr:_HostileRepr>" in str(err.value)
