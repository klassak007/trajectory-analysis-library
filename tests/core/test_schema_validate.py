from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tal.core import SchemaError, merge_schema, set_roles, validate_schema
from tal.core.schema_validate import validate_schema_structure


def _ds_sample_axis() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("sample", "axis"), np.ones((4, 3)))},
        coords={"sample": [0, 1, 2, 3], "axis": ["x", "y", "z"]},
    )


def _valid_schema_payload() -> dict:
    return {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            }
        },
    }


def _batched_validity_ds(size: object) -> xr.Dataset:
    ds = xr.Dataset(
        {"value": (("trial", "sample"), np.arange(6.0).reshape(2, 3))},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "n_valid": ("trial", size),
        },
    )
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": ["trial"],
                "core_dims": [],
            },
            "validity": {
                "sequence_size_coord": "n_valid",
                "layout": "left_packed",
            },
        },
    }
    return ds


class _HostileKey:
    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return False

    def __repr__(self) -> str:
        raise RuntimeError("repr exploded")

    def __str__(self) -> str:
        raise RuntimeError("str exploded")


class _MutableName(str):
    notes: list[str]

    def __new__(cls, value: str) -> "_MutableName":
        name = super().__new__(cls, value)
        name.notes = []
        return name

    def __str__(self) -> "_MutableName":
        return self


def _assert_schema_error(
    err: pytest.ExceptionInfo[SchemaError],
    *,
    code: str,
    path: str,
) -> None:
    assert err.value.code == code
    assert err.value.path == path


def test_schema_validate_flow_001_phase_order_root_before_roles() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_001_phase_order_root_before_roles."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = "not-a-mapping"
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.not_mapping", path="tal")


def test_schema_validate_flow_002_phase_order_roles_before_param() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_002_phase_order_roles_before_param."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {"batch_dims": [], "core_dims": ["axis"]},
            "param_coord": {"name": "time"},
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(
        err,
        code="schema.param_coord.sequence_dim.missing",
        path="tal.core.roles.sequence_dim",
    )


def test_schema_validate_flow_003_phase_order_param_before_validity() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_003_phase_order_param_before_validity."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            },
            "param_coord": {"name": "missing_coord"},
            "validity": {"sequence_size_coord": "missing_size", "layout": "bad"},
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(
        err,
        code="schema.param_coord.not_found",
        path="tal.core.param_coord.name",
    )


def test_schema_validate_flow_004_first_failure_deterministic() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_004_first_failure_deterministic."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": ["sample"],
                "core_dims": [],
            }
        },
    }
    with pytest.raises(SchemaError) as err1:
        validate_schema(ds)
    with pytest.raises(SchemaError) as err2:
        validate_schema(ds)
    assert err1.value.code == err2.value.code
    assert err1.value.path == err2.value.path


def test_schema_validate_flow_005_patch_failures_precede_validation() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_005_patch_failures_precede_validation."""
    ds = set_roles(_ds_sample_axis(), sequence_dim="sample", batch_dims=(), core_dims=("axis",))
    with pytest.raises(SchemaError) as err:
        merge_schema(ds, {"tal": {"core": {}}})
    _assert_schema_error(err, code="schema.patch.root.invalid", path="tal")


def test_schema_validate_flow_006_core_mixed_key_types_raise_schema_error() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_006_core_mixed_key_types_raise_schema_error."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            },
            1: "bad",
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.core.unknown_key", path="tal.core.1")
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "int"


def test_schema_validate_flow_007_roles_mixed_key_types_raise_schema_error() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_007_roles_mixed_key_types_raise_schema_error."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
                1: "bad",
            }
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.roles.unknown_key", path="tal.core.roles.1")
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "int"


def test_schema_validate_flow_008_param_mixed_key_types_raise_schema_error() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_008_param_mixed_key_types_raise_schema_error."""
    ds = _ds_sample_axis().assign_coords(time=("sample", [0.0, 0.1, 0.2, 0.3]))
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            },
            "param_coord": {"name": "time", 1: "bad"},
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.param_coord.unknown_key", path="tal.core.param_coord.1")
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "int"


def test_schema_validate_flow_009_validity_mixed_key_types_raise_schema_error() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_009_validity_mixed_key_types_raise_schema_error."""
    ds = _ds_sample_axis().assign_coords(group_size=np.int64(4))
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            },
            "validity": {"sequence_size_coord": "group_size", "layout": "left_packed", 1: "bad"},
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.validity.unknown_key", path="tal.core.validity.1")
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "int"


def test_schema_validate_flow_010_hostile_core_key_raises_schema_error_not_runtime() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_010_hostile_core_key_raises_schema_error_not_runtime."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            },
            _HostileKey(): "bad",
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(
        err,
        code="schema.core.unknown_key",
        path="tal.core.<_HostileKey>",
    )
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "_HostileKey"


def test_schema_validate_flow_011_hostile_roles_key_raises_schema_error_not_runtime() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_011_hostile_roles_key_raises_schema_error_not_runtime."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
                _HostileKey(): "bad",
            }
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(
        err,
        code="schema.roles.unknown_key",
        path="tal.core.roles.<_HostileKey>",
    )
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "_HostileKey"


def test_schema_validate_flow_012_hostile_param_key_raises_schema_error_not_runtime() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_012_hostile_param_key_raises_schema_error_not_runtime."""
    ds = _ds_sample_axis().assign_coords(time=("sample", [0.0, 0.1, 0.2, 0.3]))
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            },
            "param_coord": {"name": "time", _HostileKey(): "bad"},
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(
        err,
        code="schema.param_coord.unknown_key",
        path="tal.core.param_coord.<_HostileKey>",
    )
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "_HostileKey"


def test_schema_validate_flow_013_hostile_validity_key_raises_schema_error_not_runtime() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_013_hostile_validity_key_raises_schema_error_not_runtime."""
    ds = _ds_sample_axis().assign_coords(group_size=np.int64(4))
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            },
            "validity": {
                "sequence_size_coord": "group_size",
                "layout": "left_packed",
                _HostileKey(): "bad",
            },
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(
        err,
        code="schema.validity.unknown_key",
        path="tal.core.validity.<_HostileKey>",
    )
    assert isinstance(err.value.actual, dict)
    assert err.value.actual["key_type"] == "_HostileKey"


def test_schema_validate_flow_014_unknown_tal_key_rejected() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_014_unknown_tal_key_rejected."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            }
        },
        "unknown": 1,
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.unknown_key", path="tal.unknown")


def test_schema_validate_flow_015_ext_namespace_key_must_be_non_empty_string() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_015_ext_namespace_key_must_be_non_empty_string."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            }
        },
        "ext": {1: {"value": 1}},
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.ext.namespace.invalid", path="tal.ext.1")


def test_schema_validate_flow_016_hostile_ext_namespace_key_raises_schema_error() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_016_hostile_ext_namespace_key_raises_schema_error."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            }
        },
        "ext": {_HostileKey(): {"value": 1}},
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.ext.namespace.invalid", path="tal.ext.<_HostileKey>")


def test_schema_validate_flow_017_validity_values_precede_extension_envelope() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_017_validity_values_precede_extension_envelope."""
    ds = _batched_validity_ds([2, 4])
    ds.attrs["tal"]["ext"] = {1: {}}

    with pytest.raises(SchemaError) as err:
        validate_schema(ds)

    _assert_schema_error(
        err,
        code="schema.validity.sequence_size_coord.values.invalid",
        path="tal.core.validity.sequence_size_coord",
    )


@pytest.mark.parametrize("validator", [validate_schema_structure, validate_schema])
def test_schema_validate_flow_018_categorical_validity_dtype_raises_schema_error(
    validator: Callable[[xr.Dataset], object],
) -> None:
    """ID: SCHEMA_VALIDATE_FLOW_018_categorical_validity_dtype_raises_schema_error."""
    ds = _batched_validity_ds(pd.Categorical([2, 3]))

    with pytest.raises(SchemaError) as err:
        validator(ds)

    _assert_schema_error(
        err,
        code="schema.validity.sequence_size_coord.values.invalid",
        path="tal.core.validity.sequence_size_coord",
    )


@pytest.mark.parametrize("validator", [validate_schema_structure, validate_schema])
def test_schema_validate_flow_019_categorical_param_dtype_raises_schema_error(
    validator: Callable[[xr.Dataset], object],
) -> None:
    """ID: SCHEMA_VALIDATE_FLOW_019_categorical_param_dtype_raises_schema_error."""
    ds = _ds_sample_axis().assign_coords(
        category_time=("sample", pd.Categorical([1, 2, 3, 4]))
    )
    schema = _valid_schema_payload()
    schema["core"]["param_coord"] = {"name": "category_time"}
    ds.attrs["tal"] = schema

    with pytest.raises(SchemaError) as err:
        validator(ds)

    _assert_schema_error(
        err,
        code="schema.param_coord.dtype.invalid",
        path="tal.core.param_coord.name",
    )


def test_schema_validate_flow_020_structure_projection_does_not_expose_schema_alias() -> None:
    """ID: SCHEMA_VALIDATE_FLOW_020_structure_projection_does_not_expose_schema_alias."""
    ds = _batched_validity_ds([2, 3])
    original_name = _MutableName("n_valid")
    ds.attrs["tal"]["core"]["validity"]["sequence_size_coord"] = original_name
    before = deepcopy(ds.attrs["tal"])

    size_name = validate_schema_structure(ds)

    assert size_name == "n_valid"
    assert type(size_name) is str
    assert size_name is not original_name
    assert not hasattr(size_name, "notes")
    assert original_name.notes == []
    assert ds.attrs["tal"] == before


def test_nolegacy_001_legacy_role_keys_rejected() -> None:
    """ID: NOLEGACY_001_legacy_role_keys_rejected."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sample_dim": "sample",
                "batch_dims": [],
                "core_dims": [],
            }
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.roles.unknown_key", path="tal.core.roles.sample_dim")


def test_nolegacy_002_legacy_param_keys_rejected() -> None:
    """ID: NOLEGACY_002_legacy_param_keys_rejected."""
    ds = _ds_sample_axis().assign_coords(time=("sample", [0.0, 0.1, 0.2, 0.3]))
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            },
            "param_coord": {"coord": "time"},
        },
    }
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.param_coord.unknown_key", path="tal.core.param_coord.coord")


def test_nolegacy_003_no_auto_upgrade() -> None:
    """ID: NOLEGACY_003_no_auto_upgrade."""
    ds = _ds_sample_axis()
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sample_dim": "sample",
                "group_dims": [],
                "core_dims": [],
            }
        },
    }
    before = deepcopy(ds.attrs["tal"])
    with pytest.raises(SchemaError) as err:
        validate_schema(ds)
    _assert_schema_error(err, code="schema.roles.unknown_key", path="tal.core.roles.group_dims")
    assert ds.attrs["tal"] == before
