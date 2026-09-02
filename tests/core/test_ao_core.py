from collections import UserDict
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core import SchemaError
from tal.core.orchestration.alignment_intent import read_alignment_intent
from tal.core.orchestration.broadcast_intent import read_broadcast_intent
from tal.core.schema_read import read_roles
from tal.core.typed_lifecycle import (
    TypedAnalysisObject,
    TypedLifecycleContext,
    TypedLifecycleSpec,
    default_coerce_source,
    identity_init_options,
    identity_normalize,
    no_op_enforce,
)
from tal.linalg import Array
from tal.spatial import Acceleration, Pose, Position, Rotation, Velocity
import tal.core.analysis_object as ao_mod
import tal.core.schema as schema_mod
import tal.core.schema_validate as schema_validate_mod


def _ds_single() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("sample", "axis"), np.ones((3, 2)))},
        coords={"sample": [0, 1, 2], "axis": ["x", "y"]},
    )


def _ds_trial() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"value": (("trial", "sample", "axis"), np.ones((2, 3, 2)))},
        coords={"trial": [0, 1], "sample": [0, 1, 2], "axis": ["x", "y"]},
    )


def _ds_multiindex() -> xr.Dataset:
    return _ds_single().stack(sample_axis=("sample", "axis"))


def _mapping_schema(
    kind: str,
    *,
    bootstrap: bool,
) -> tuple[Mapping[str, Any], list[str]]:
    wrap = MappingProxyType if kind == "mapping-proxy" else UserDict
    items = ["value"]
    record = wrap({"name": "entry"})
    demo = wrap({"enabled": True, "items": items, "records": [record]})
    ext = wrap({"demo": demo})
    if bootstrap:
        core = wrap({})
    else:
        roles = wrap(
            {
                "sequence_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            }
        )
        core = wrap({"roles": roles})
    return wrap({"version": 1, "core": core, "ext": ext}), items


class _CopyBomb:
    def __deepcopy__(self, memo: dict[int, object]) -> object:
        del memo
        raise RuntimeError("copy exploded")


class _MutableSchemaKey:
    def __init__(self, label: str) -> None:
        self.label = label
        self.notes: list[str] = []

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MutableSchemaKey) and self.label == other.label


class _SchemaCopyCounter:
    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    def __deepcopy__(self, memo: dict[int, object]) -> "_SchemaCopyCounter":
        self.calls[0] += 1
        out = type(self)(self.calls)
        memo[id(self)] = out
        return out


def _aliasing_schema() -> tuple[Mapping[str, Any], _MutableSchemaKey, list[str]]:
    key = _MutableSchemaKey("entry")
    items = ["value"]
    shared = MappingProxyType({key: MappingProxyType({"items": items})})
    demo = MappingProxyType({"first": shared, "again": [shared, (shared,)]})
    roles = MappingProxyType(
        {
            "sequence_dim": "sample",
            "batch_dims": [],
            "core_dims": ["axis"],
        }
    )
    return (
        MappingProxyType(
            {
                "version": 1,
                "core": MappingProxyType({"roles": roles}),
                "ext": MappingProxyType({"demo": demo}),
            }
        ),
        key,
        items,
    )


def _assert_aliasing_schema_owned(
    tal: Mapping[str, Any],
    source_key: _MutableSchemaKey,
    source_items: list[str],
) -> None:
    demo = tal["ext"]["demo"]
    first = demo["first"]
    assert first is demo["again"][0]
    assert first is demo["again"][1][0]
    copied_key = next(iter(first))
    assert copied_key is not source_key
    source_key.notes.append("caller-key-mutation")
    source_items.append("caller-value-mutation")
    assert copied_key.notes == []
    assert first[copied_key]["items"] == ["value"]


def test_ao_init_001_dataset_accept() -> None:
    """ID: AO_INIT_001_dataset_accept."""
    ao = AnalysisObject(_ds_single())
    assert isinstance(ao.data, xr.Dataset)
    assert "value" in ao.data.data_vars


def test_ao_init_002_dataarray_promote() -> None:
    """ID: AO_INIT_002_dataarray_promote."""
    da = xr.DataArray(np.arange(3), dims=("sample",), name="signal")
    ao = AnalysisObject(da)
    assert isinstance(ao.data, xr.Dataset)
    assert list(ao.data.data_vars) == ["signal"]


def test_ao_init_012_dataarray_promote_unnamed_uses_datavar() -> None:
    """ID: AO_INIT_012_dataarray_promote_unnamed_uses_datavar."""
    da = xr.DataArray(np.arange(3), dims=("sample",))
    ao = AnalysisObject(da)
    assert isinstance(ao.data, xr.Dataset)
    assert list(ao.data.data_vars) == ["datavar"]


def test_ao_init_003_invalid_type_reject() -> None:
    """ID: AO_INIT_003_invalid_type_reject."""
    try:
        AnalysisObject(123)  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("Expected TypeError for non-xarray input.")


def test_ao_convert_001_to_dataarray_single_var() -> None:
    """ID: AO_CONVERT_001_to_dataarray_single_var."""
    ao = AnalysisObject(_ds_single())
    out = ao.to_dataarray(name="renamed")
    assert isinstance(out, xr.DataArray)
    assert out.name == "renamed"
    assert tuple(out.dims) == ("sample", "axis")


def test_ao_convert_002_to_dataarray_multi_var_error() -> None:
    """ID: AO_CONVERT_002_to_dataarray_multi_var_error."""
    ds = _ds_single().assign(other=("sample", np.arange(3)))
    ao = AnalysisObject(ds)
    try:
        ao.to_dataarray()
    except ValueError as err:
        assert "exactly one data variable" in str(err)
        return
    raise AssertionError("Expected ValueError for multi-variable dataset.")


def test_ao_convert_003_to_dataarray_mutation_not_reflective() -> None:
    """ID: AO_CONVERT_003_to_dataarray_mutation_not_reflective."""
    ao = AnalysisObject(_ds_single())
    out = ao.to_dataarray()
    out.values[0, 0] = 321.0
    assert float(ao.unsafe_data["value"].values[0, 0]) == 1.0


def test_ao_init_004_constructor_minimal_no_roles_required() -> None:
    """ID: AO_INIT_004_constructor_minimal_no_roles_required."""
    ao = AnalysisObject(_ds_single())
    assert ao.data.attrs["tal"] == {"version": 1, "core": {}}


def test_ao_init_005_constructor_validates_existing_tal_payload() -> None:
    """ID: AO_INIT_005_constructor_validates_existing_tal_payload."""
    ds = _ds_single().copy(deep=False)
    ds.attrs["tal"] = {
        "version": 1,
        "core": {
            "roles": {
                "sample_dim": "sample",
                "batch_dims": [],
                "core_dims": ["axis"],
            }
        },
    }
    with pytest.raises(SchemaError) as err:
        AnalysisObject(ds)
    assert err.value.code == "schema.roles.unknown_key"
    assert err.value.path == "tal.core.roles.sample_dim"


def test_ao_init_006_constructor_skips_full_validation_without_tal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_INIT_006_constructor_skips_full_validation_without_tal."""
    calls: list[str] = []

    def _count_validate(ds: xr.Dataset) -> xr.Dataset:
        calls.append("validate")
        return schema_validate_mod.validate_schema(ds)

    monkeypatch.setattr(ao_mod, "_validate_schema", _count_validate)
    ao = AnalysisObject(_ds_single())
    assert calls == []
    assert ao.data.attrs["tal"] == {"version": 1, "core": {}}


def test_ao_init_007_constructor_roundtrip_bootstrap_schema_idempotent() -> None:
    """ID: AO_INIT_007_constructor_roundtrip_bootstrap_schema_idempotent."""
    ao = AnalysisObject(_ds_single())
    roundtrip = AnalysisObject(ao.data)
    assert roundtrip.data.attrs["tal"] == {"version": 1, "core": {}}


def test_ao_init_008_constructor_invalid_tal_type_dataset_dataarray_parity() -> None:
    """ID: AO_INIT_008_constructor_invalid_tal_type_dataset_dataarray_parity."""
    ds = _ds_single().copy(deep=False)
    ds.attrs["tal"] = "bad"
    da = xr.DataArray(np.arange(3), dims=("sample",), name="signal")
    da.attrs["tal"] = "bad"
    with pytest.raises(SchemaError) as err_ds:
        AnalysisObject(ds)
    with pytest.raises(SchemaError) as err_da:
        AnalysisObject(da)
    assert err_ds.value.code == err_da.value.code == "schema.not_mapping"
    assert err_ds.value.path == err_da.value.path == "tal"


def test_ao_init_009_constructor_bootstrap_with_ext_roundtrip_idempotent() -> None:
    """ID: AO_INIT_009_constructor_bootstrap_with_ext_roundtrip_idempotent."""
    base = AnalysisObject(_ds_single())
    with_ext = base.merge_schema({"ext": {"demo": {"enabled": True}}}, validate=False)
    roundtrip = AnalysisObject(with_ext.data)
    tal = roundtrip.data.attrs["tal"]
    assert tal["version"] == 1
    assert tal["core"] == {}
    assert tal["ext"] == {"demo": {"enabled": True}}


@pytest.mark.parametrize(
    "construct",
    (AnalysisObject, AnalysisObject.from_data),
    ids=("constructor", "from-data"),
)
def test_ao_init_013_dataarray_schema_is_relocated_once(construct: Any) -> None:
    """ID: AO_INIT_013_dataarray_schema_is_relocated_once."""
    tal = {"version": 1, "core": {}, "ext": {"demo": None}}
    da = xr.DataArray(
        np.arange(3),
        dims=("sample",),
        name="signal",
        attrs={"tal": tal, "ordinary": {"items": ["value"]}},
    )

    ao = construct(da)

    assert ao.unsafe_data.attrs["tal"] == tal
    assert "tal" not in ao.unsafe_data["signal"].attrs
    assert ao.unsafe_data["signal"].attrs["ordinary"] == {"items": ["value"]}


@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
@pytest.mark.parametrize(
    "tal",
    (
        {"version": 999, "core": {}},
        {"core": {}},
        {"version": "1", "core": {}},
        {"version": True, "core": {}},
        {"version": 1.0, "core": {}},
    ),
    ids=(
        "wrong-version",
        "missing-version",
        "version-string",
        "version-bool",
        "version-float",
    ),
)
def test_ao_init_014_constructor_validates_schema_before_bootstrap(
    as_dataarray: bool,
    tal: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_INIT_014_constructor_validates_schema_before_bootstrap."""
    data: xr.Dataset | xr.DataArray
    if as_dataarray:
        data = xr.DataArray([1], dims=("sample",), name="signal", attrs={"tal": tal})
    else:
        data = xr.Dataset({"signal": ("sample", [1])}, attrs={"tal": tal})

    def fail_ownership_copy(_data: object) -> xr.Dataset:
        raise AssertionError("deterministic schema failure reached ownership copy")

    monkeypatch.setattr(
        ao_mod._dataset_ownership,
        "isolate_external_dataset",
        fail_ownership_copy,
    )
    with pytest.raises(SchemaError) as error:
        AnalysisObject(data)

    assert error.value.code == "schema.version.invalid"
    assert error.value.path == "tal.version"


@pytest.mark.parametrize("kind", ("mapping-proxy", "user-dict"))
@pytest.mark.parametrize("bootstrap", (False, True), ids=("roles", "bootstrap"))
@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
@pytest.mark.parametrize(
    "ingress",
    ("constructor", "from-data-unvalidated", "from-data-validated"),
)
def test_ao_ingress_016_mapping_schema_is_canonical_and_isolated(
    kind: str,
    bootstrap: bool,
    as_dataarray: bool,
    ingress: str,
) -> None:
    """ID: AO_INGRESS_016_mapping_schema_is_canonical_and_isolated."""
    tal, source_items = _mapping_schema(kind, bootstrap=bootstrap)
    source: xr.Dataset | xr.DataArray
    source = _ds_single()["value"] if as_dataarray else _ds_single()
    source.attrs["tal"] = tal

    if ingress == "constructor":
        ao = AnalysisObject(source)
    else:
        validate = ingress == "from-data-validated"
        ao = AnalysisObject.from_data(source, validate=validate)

    actual = ao.unsafe_data.attrs["tal"]
    assert type(actual) is dict
    assert type(actual["core"]) is dict
    assert type(actual["ext"]) is dict
    assert type(actual["ext"]["demo"]) is dict
    assert type(actual["ext"]["demo"]["records"][0]) is dict
    if not bootstrap:
        assert type(actual["core"]["roles"]) is dict
    source_items.append("caller-mutation")
    assert actual["ext"]["demo"]["items"] == ["value"]


@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
@pytest.mark.parametrize(
    "ingress",
    ("constructor", "from-data-unvalidated", "from-data-validated"),
)
def test_ao_ingress_018_extension_graph_keys_and_aliases_owned(
    as_dataarray: bool,
    ingress: str,
) -> None:
    """ID: AO_INGRESS_018_extension_graph_keys_and_aliases_owned."""
    tal, key, items = _aliasing_schema()
    source: xr.Dataset | xr.DataArray
    source = _ds_single()["value"] if as_dataarray else _ds_single()
    source.attrs["tal"] = tal

    if ingress == "constructor":
        ao = AnalysisObject(source)
    else:
        ao = AnalysisObject.from_data(
            source,
            validate=ingress == "from-data-validated",
        )

    _assert_aliasing_schema_owned(ao.unsafe_data.attrs["tal"], key, items)


@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
@pytest.mark.parametrize(
    "ingress",
    ("constructor", "from-data-unvalidated", "from-data-validated"),
)
def test_ao_ingress_019_core_extension_alias_is_value_only(
    as_dataarray: bool,
    ingress: str,
) -> None:
    """ID: AO_INGRESS_019_core_extension_alias_is_value_only."""
    shared_dims = ["axis"]
    shared_core = {
        "roles": {
            "sequence_dim": "sample",
            "batch_dims": [],
            "core_dims": shared_dims,
        }
    }
    source: xr.Dataset | xr.DataArray
    source = _ds_single()["value"] if as_dataarray else _ds_single()
    source.attrs["tal"] = {
        "version": 1,
        "core": shared_core,
        "ext": {"demo": shared_core},
    }

    if ingress == "constructor":
        ao = AnalysisObject(source).set_roles(core_dims=(), validate=True)
    else:
        ao = AnalysisObject.from_data(
            source,
            sequence_dim="sample",
            batch_dims=(),
            core_dims=(),
            validate=ingress == "from-data-validated",
        )
    tal = ao.unsafe_data.attrs["tal"]

    assert type(tal["core"]["roles"]["core_dims"]) is list
    assert tal["core"]["roles"]["core_dims"] == []
    assert tal["ext"]["demo"]["roles"]["core_dims"] == ["axis"]
    shared_dims.append("caller-mutation")
    assert tal["core"]["roles"]["core_dims"] == []
    assert tal["ext"]["demo"]["roles"]["core_dims"] == ["axis"]


@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
@pytest.mark.parametrize(
    "ingress",
    ("constructor", "from-data-unvalidated", "from-data-validated"),
)
def test_ao_ingress_017_invalid_schema_precedes_extension_copy(
    as_dataarray: bool,
    ingress: str,
) -> None:
    """ID: AO_INGRESS_017_invalid_schema_precedes_extension_copy."""
    source: xr.Dataset | xr.DataArray
    source = _ds_single()["value"] if as_dataarray else _ds_single()
    source.attrs["tal"] = {
        "version": 999,
        "core": {},
        "ext": {"demo": _CopyBomb()},
    }

    with pytest.raises(SchemaError) as error:
        if ingress == "constructor":
            AnalysisObject(source)
        else:
            AnalysisObject.from_data(
                source,
                validate=ingress == "from-data-validated",
            )

    assert error.value.code == "schema.version.invalid"
    assert error.value.path == "tal.version"


def test_ao_init_010_constructor_rejects_multiindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_INIT_010_constructor_rejects_multiindex."""

    def fail_ownership_copy(_data: object) -> xr.Dataset:
        raise AssertionError("MultiIndex failure reached ownership copy")

    monkeypatch.setattr(
        ao_mod._dataset_ownership,
        "isolate_external_dataset",
        fail_ownership_copy,
    )
    with pytest.raises(ValueError) as err:
        AnalysisObject(_ds_multiindex())
    assert "PandasMultiIndex dimensions are not supported" in str(err.value)


def test_ao_init_011_constructor_ingress_mutation_not_reflective() -> None:
    """ID: AO_INIT_011_constructor_ingress_mutation_not_reflective."""
    ds = _ds_single()
    ao = AnalysisObject(ds)
    ds["value"].values[0, 0] = 999.0
    ds = ds.assign_coords(sample=[999, 1, 2])
    assert float(ao.unsafe_data["value"].values[0, 0]) == 1.0
    assert int(ao.unsafe_data.coords["sample"].values[0]) == 0


def test_ao_mutability_001_data_returns_safe_copy() -> None:
    """ID: AO_MUTABILITY_001_data_returns_shallow_copy."""
    ao = AnalysisObject(_ds_single())
    out = ao.data
    assert out is not ao.unsafe_data
    assert out["value"].data is not ao.unsafe_data["value"].data


def test_ao_mutability_002_as_dataset_returns_safe_copy() -> None:
    """ID: AO_MUTABILITY_002_as_dataset_returns_shallow_copy."""
    ao = AnalysisObject(_ds_single())
    out = ao.as_dataset()
    assert out is not ao.unsafe_data
    assert out["value"].data is not ao.unsafe_data["value"].data


def test_ao_mutability_003_mutating_accessor_copy_does_not_change_internal_schema() -> None:
    """ID: AO_MUTABILITY_003_mutating_accessor_copy_does_not_change_internal_schema."""
    ao = AnalysisObject(_ds_single())
    out = ao.data
    out.attrs["tal"]["core"]["roles"] = {
        "sequence_dim": "sample",
        "batch_dims": [],
        "core_dims": ["axis"],
    }
    assert "roles" not in ao.unsafe_data.attrs["tal"]["core"]


def test_ao_mutability_004_unsafe_data_mutation_is_reflective() -> None:
    """ID: AO_MUTABILITY_004_unsafe_data_mutation_is_reflective."""
    ao = AnalysisObject(_ds_single())
    ao.unsafe_data.attrs["tal"]["core"]["roles"] = {
        "sequence_dim": "sample",
        "batch_dims": [],
        "core_dims": ["axis"],
    }
    assert ao.unsafe_data.attrs["tal"]["core"]["roles"]["sequence_dim"] == "sample"


def test_ao_mutability_005_data_value_mutation_not_reflective() -> None:
    """ID: AO_MUTABILITY_005_data_value_mutation_not_reflective."""
    ao = AnalysisObject(_ds_single())
    out = ao.data
    out["value"].values[0, 0] = 999.0
    assert float(ao.unsafe_data["value"].values[0, 0]) == 1.0


def test_ao_mutability_006_data_coord_mutation_not_reflective() -> None:
    """ID: AO_MUTABILITY_006_data_coord_mutation_not_reflective."""
    ao = AnalysisObject(_ds_single())
    out = ao.as_dataset()
    out = out.assign_coords(sample=[999, 1, 2])
    assert int(ao.unsafe_data.coords["sample"].values[0]) == 0


def test_ao_mutability_007_data_rejects_multiindex_internal_state() -> None:
    """ID: AO_MUTABILITY_007_data_rejects_multiindex_internal_state."""
    ao = AnalysisObject(_ds_single())
    object.__setattr__(ao, "_data", _ds_multiindex())
    with pytest.raises(ValueError) as err_data:
        _ = ao.data
    assert "PandasMultiIndex dimensions are not supported" in str(err_data.value)
    with pytest.raises(ValueError) as err_ds:
        _ = ao.as_dataset()
    assert "PandasMultiIndex dimensions are not supported" in str(err_ds.value)


def test_ao_events_001_accessor_available() -> None:
    """ID: AO_EVENTS_001_accessor_available."""
    ao = AnalysisObject(_ds_single().assign_coords(time=("sample", [0.0, 1.0, 2.0])))
    accessor = ao.events
    assert accessor.__class__.__name__ == "EventsAccessor"


def test_ao_fromdata_001_set_roles_success() -> None:
    """ID: AO_FROMDATA_001_set_roles_success."""
    ao = AnalysisObject.from_data(
        _ds_trial(),
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
    )
    roles = ao.data.attrs["tal"]["core"]["roles"]
    assert roles == {
        "sequence_dim": "sample",
        "batch_dims": ["trial"],
        "core_dims": ["axis"],
    }


def test_ao_fromdata_002_set_param_coord_1d_success() -> None:
    """ID: AO_FROMDATA_002_set_param_coord_1d_success."""
    ds = _ds_single().assign_coords(time=("sample", [0.0, 0.1, 0.2]))
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
        param_coord="time",
    )
    assert ao.data.attrs["tal"]["core"]["param_coord"]["name"] == "time"


def test_ao_fromdata_003_set_param_coord_nd_success() -> None:
    """ID: AO_FROMDATA_003_set_param_coord_nd_success."""
    ds = _ds_trial().assign_coords(
        phase=(("trial", "sample"), np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]))
    )
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="phase",
    )
    assert ao.data.attrs["tal"]["core"]["param_coord"]["name"] == "phase"


def test_ao_fromdata_004_set_validity_success() -> None:
    """ID: AO_FROMDATA_004_set_validity_success."""
    ds = _ds_trial().assign_coords(group_size=("trial", [3, 2]))
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        sequence_size_coord="group_size",
    )
    validity = ao.data.attrs["tal"]["core"]["validity"]
    assert validity == {"sequence_size_coord": "group_size", "layout": "left_packed"}


def test_ao_fromdata_005_batch_and_core_without_sequence_dim_allowed() -> None:
    """ID: AO_FROMDATA_012_batch_and_core_without_sequence_dim_allowed."""
    ao = AnalysisObject.from_data(_ds_trial(), batch_dims=("trial",), core_dims=("axis",))
    declared, sequence_dim, batch_dims, core_dims = read_roles(ao.unsafe_data)
    assert declared is True
    assert sequence_dim is None
    assert batch_dims == ("trial",)
    assert core_dims == ("axis",)


def test_ao_fromdata_006_no_roles_assigned_validates_as_core_only() -> None:
    """ID: AO_FROMDATA_013_no_roles_assigned_validates_as_core_only."""
    ao = AnalysisObject.from_data(_ds_trial())
    declared, sequence_dim, batch_dims, core_dims = read_roles(ao.unsafe_data)
    assert declared is False
    assert sequence_dim is None
    assert batch_dims == ()
    assert core_dims == ()


def test_ao_fromdata_008_param_coord_without_sequence_dim_rejected() -> None:
    """ID: AO_FROMDATA_008_param_coord_without_sequence_dim_rejected."""
    ds = _ds_single().assign_coords(time=("sample", [0.0, 0.1, 0.2]))
    with pytest.raises(ValueError) as err:
        AnalysisObject.from_data(ds, param_coord="time")
    assert "requires sequence_dim" in str(err.value)
    assert "param_coord" in str(err.value)


def test_ao_fromdata_009_validity_without_sequence_dim_rejected() -> None:
    """ID: AO_FROMDATA_009_validity_without_sequence_dim_rejected."""
    ds = _ds_single().assign_coords(group_size=np.int64(3))
    with pytest.raises(ValueError) as err:
        AnalysisObject.from_data(ds, sequence_size_coord="group_size")
    assert "requires sequence_dim" in str(err.value)
    assert "sequence_size_coord" in str(err.value)


def test_ao_fromdata_010_rejects_multiindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_FROMDATA_010_rejects_multiindex."""

    def fail_ownership_copy(_data: object) -> xr.Dataset:
        raise AssertionError("MultiIndex failure reached ownership copy")

    monkeypatch.setattr(
        ao_mod._dataset_ownership,
        "isolate_external_dataset",
        fail_ownership_copy,
    )
    with pytest.raises(ValueError) as err:
        AnalysisObject.from_data(
            _ds_multiindex(),
            sequence_dim="sample_axis",
            batch_dims=(),
            core_dims=(),
        )
    assert "PandasMultiIndex dimensions are not supported" in str(err.value)


def test_ao_fromdata_011_from_data_ingress_mutation_not_reflective() -> None:
    """ID: AO_FROMDATA_011_from_data_ingress_mutation_not_reflective."""
    ds = _ds_trial()
    ao = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
    )
    ds["value"].values[0, 0, 0] = 777.0
    ds = ds.assign_coords(trial=[99, 1])
    assert float(ao.unsafe_data["value"].values[0, 0, 0]) == 1.0
    assert int(ao.unsafe_data.coords["trial"].values[0]) == 0


@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
def test_ao_fromdata_014_invalid_tal_type_has_input_parity(
    as_dataarray: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_FROMDATA_014_invalid_tal_type_has_input_parity."""
    data: xr.Dataset | xr.DataArray
    if as_dataarray:
        data = xr.DataArray([1], dims=("sample",), name="signal", attrs={"tal": "bad"})
    else:
        data = xr.Dataset({"signal": ("sample", [1])}, attrs={"tal": "bad"})

    def fail_ownership_copy(_data: object) -> xr.Dataset:
        raise AssertionError("invalid tal type reached ownership copy")

    monkeypatch.setattr(
        ao_mod._dataset_ownership,
        "isolate_external_dataset",
        fail_ownership_copy,
    )
    with pytest.raises(SchemaError) as error:
        AnalysisObject.from_data(data)

    assert error.value.code == "schema.not_mapping"
    assert error.value.path == "tal"


@pytest.mark.parametrize("validate", (False, True), ids=("unvalidated", "validated"))
@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
@pytest.mark.parametrize(
    ("tal", "code", "path"),
    (
        ({"version": 999, "core": {}}, "schema.version.invalid", "tal.version"),
        ({"core": {}}, "schema.version.invalid", "tal.version"),
        ({"version": "1", "core": {}}, "schema.version.invalid", "tal.version"),
        ({"version": True, "core": {}}, "schema.version.invalid", "tal.version"),
        ({"version": 1.0, "core": {}}, "schema.version.invalid", "tal.version"),
        ({"version": 1, "core": "bad"}, "schema.core.not_mapping", "tal.core"),
        (
            {"version": 1, "core": {}, "unknown": True},
            "schema.unknown_key",
            "tal.unknown",
        ),
        (
            {"version": 1, "core": {"unknown": True}},
            "schema.core.unknown_key",
            "tal.core.unknown",
        ),
        (
            {"version": 1, "core": {}, "ext": "bad"},
            "schema.not_mapping",
            "tal.ext",
        ),
        (
            {"version": 1, "core": {}, "ext": {"": {}}},
            "schema.ext.namespace.invalid",
            "tal.ext.",
        ),
    ),
    ids=(
        "wrong-version",
        "missing-version",
        "version-string",
        "version-bool",
        "version-float",
        "core-type",
        "root-key",
        "core-key",
        "ext-type",
        "ext-namespace",
    ),
)
def test_ao_fromdata_015_existing_schema_envelope_fails_before_writers_and_copy(
    validate: bool,
    as_dataarray: bool,
    tal: dict[str, Any],
    code: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_FROMDATA_015_existing_schema_envelope_fails_before_writers_and_copy."""

    data: xr.Dataset | xr.DataArray
    if as_dataarray:
        data = xr.DataArray([1.0], dims=("x",), name="value", attrs={"tal": tal})
    else:
        data = xr.Dataset({"value": ("x", [1.0])}, attrs={"tal": tal})

    def fail_ownership_copy(_data: object) -> xr.Dataset:
        raise AssertionError("schema envelope failure reached ownership copy")

    monkeypatch.setattr(
        ao_mod._dataset_ownership,
        "isolate_external_dataset",
        fail_ownership_copy,
    )
    with pytest.raises(SchemaError) as error:
        AnalysisObject.from_data(data, core_dims=("x",), validate=validate)

    assert error.value.code == code
    assert error.value.path == path


def test_ao_fromdata_006_atomic_on_failure() -> None:
    """ID: AO_FROMDATA_006_atomic_on_failure."""
    ds = _ds_trial().assign_coords(
        bad=(("sample", "trial"), np.zeros((3, 2), dtype="float64"))
    )
    before = dict(ds.attrs)
    with pytest.raises(SchemaError):
        AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            batch_dims=("trial",),
            core_dims=("axis",),
            param_coord="bad",
        )
    assert ds.attrs == before
    assert "tal" not in ds.attrs


def test_ao_fromdata_007_single_final_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: AO_FROMDATA_007_single_final_validation."""
    calls: list[str] = []

    def _count_validate(ds: xr.Dataset) -> xr.Dataset:
        calls.append("validate")
        return schema_validate_mod.validate_schema(ds)

    monkeypatch.setattr(ao_mod, "_validate_schema", _count_validate)
    monkeypatch.setattr(schema_mod, "_validate_schema", _count_validate)
    ds = _ds_trial().assign_coords(
        phase=(("trial", "sample"), np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]))
    )
    AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="phase",
        validate=True,
    )
    assert calls == ["validate"]


@pytest.mark.parametrize(
    ("validate", "expected"),
    ((False, 2), (True, 3)),
    ids=("unvalidated", "validated"),
)
def test_ao_fromdata_016_extension_copy_count_is_update_independent(
    validate: bool,
    expected: int,
) -> None:
    """ID: AO_FROMDATA_016_extension_copy_count_is_update_independent."""

    def copy_count(*, complete_update: bool) -> int:
        calls = [0]
        ds = _ds_trial().assign_coords(
            phase=(
                ("trial", "sample"),
                np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]),
            ),
            group_size=("trial", np.asarray([3, 2], dtype=np.int64)),
        )
        ds.attrs["tal"] = {
            "version": 1,
            "core": {},
            "ext": {"demo": _SchemaCopyCounter(calls)},
        }
        kwargs: dict[str, Any] = {}
        if complete_update:
            kwargs = {
                "sequence_dim": "sample",
                "batch_dims": ("trial",),
                "core_dims": ("axis",),
                "param_coord": "phase",
                "sequence_size_coord": "group_size",
            }
        AnalysisObject.from_data(ds, validate=validate, **kwargs)
        return calls[0]

    assert copy_count(complete_update=False) == expected
    assert copy_count(complete_update=True) == expected


def test_ao_fastpath_001_from_validated_binds_without_validate_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_FASTPATH_001_from_validated_binds_without_validate_call."""
    calls: list[str] = []

    def _count_validate(ds: xr.Dataset) -> xr.Dataset:
        calls.append("validate")
        return schema_validate_mod.validate_schema(ds)

    monkeypatch.setattr(ao_mod, "_validate_schema", _count_validate)
    candidate = schema_mod.merge_schema(_ds_single(), {"version": 1, "core": {}}, validate=False)
    candidate = schema_mod.set_roles(
        candidate,
        sequence_dim="sample",
        batch_dims=(),
        core_dims=("axis",),
        validate=False,
    )
    candidate = schema_validate_mod.validate_schema(candidate)
    out = AnalysisObject._from_validated(candidate)
    assert calls == []
    assert out.unsafe_data is candidate


def test_ao_fastpath_002_set_roles_validate_true_single_validation_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_FASTPATH_002_set_roles_validate_true_single_validation_pass."""
    calls: list[str] = []

    def _count_validate(ds: xr.Dataset) -> xr.Dataset:
        calls.append("validate")
        return schema_validate_mod.validate_schema(ds)

    monkeypatch.setattr(ao_mod, "_validate_schema", _count_validate)
    monkeypatch.setattr(schema_mod, "_validate_schema", _count_validate)
    ao = AnalysisObject(_ds_trial())
    ao.set_roles(
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=True,
    )
    assert calls == ["validate"]


def test_ao_fastpath_003_set_roles_validate_false_no_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: AO_FASTPATH_003_set_roles_validate_false_no_validation."""
    calls: list[str] = []

    def _count_validate(ds: xr.Dataset) -> xr.Dataset:
        calls.append("validate")
        return schema_validate_mod.validate_schema(ds)

    monkeypatch.setattr(ao_mod, "_validate_schema", _count_validate)
    monkeypatch.setattr(schema_mod, "_validate_schema", _count_validate)
    ao = AnalysisObject(_ds_trial())
    ao.set_roles(
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        validate=False,
    )
    assert calls == []


def test_bcast_core_001_analysis_object_b_helper_available() -> None:
    """ID: BCAST_CORE_001_analysis_object_b_helper_available."""
    ao = AnalysisObject(_ds_single())
    wrapped = ao.b()
    assert isinstance(wrapped, AnalysisObject)


def test_bcast_core_002_b_helper_inherited_by_spatial_subclasses() -> None:
    """ID: BCAST_CORE_002_b_helper_inherited_by_spatial_subclasses."""
    for cls in (Array, Position, Rotation, Pose, Velocity, Acceleration):
        assert hasattr(cls, "b")
        assert "b" not in cls.__dict__


def test_bcast_core_003_b_helper_does_not_mutate_source_ao() -> None:
    """ID: BCAST_CORE_003_b_helper_does_not_mutate_source_ao."""
    ao = AnalysisObject(_ds_single())
    wrapped = ao.b()
    assert wrapped is not ao
    assert read_broadcast_intent(ao, owner="test.ao.b") is None
    intent = read_broadcast_intent(wrapped, owner="test.ao.b")
    assert intent is not None
    assert intent.mode == "semantic_broadcast"
    xr.testing.assert_identical(wrapped.unsafe_data, ao.unsafe_data)


def test_bcast_hard_001_b_helper_invalid_options_fail_closed() -> None:
    """ID: BCAST_HARD_001_b_helper_invalid_options_fail_closed."""
    ao = AnalysisObject(_ds_single())
    with pytest.raises(ValueError, match="unsupported broadcast mode"):
        _ = ao.b(mode="strict")  # type: ignore[arg-type]


def test_bcast_core_035_analysis_object_a_helper_available() -> None:
    """ID: BCAST_CORE_035_analysis_object_a_helper_available."""
    ao = AnalysisObject(_ds_single())
    wrapped = ao.a()
    assert isinstance(wrapped, AnalysisObject)


def test_bcast_core_036_a_helper_inherited_by_ao_subclasses() -> None:
    """ID: BCAST_CORE_036_a_helper_inherited_by_ao_subclasses."""
    for cls in (Array, Position, Rotation, Pose, Velocity, Acceleration):
        assert hasattr(cls, "a")
        assert "a" not in cls.__dict__


def test_bcast_core_037_a_helper_is_operand_local_and_non_mutating() -> None:
    """ID: BCAST_CORE_037_a_helper_is_operand_local_and_non_mutating."""
    ao = AnalysisObject(_ds_single())
    wrapped = ao.a(on="sequence", sequence_join="exact", batch_join="exact", core_policy="strict")
    assert wrapped is not ao
    assert read_alignment_intent(ao, owner="test.ao.a") is None
    intent = read_alignment_intent(wrapped, owner="test.ao.a")
    assert intent is not None
    assert intent.on == "sequence"
    assert intent.sequence_join == "exact"
    xr.testing.assert_identical(wrapped.unsafe_data, ao.unsafe_data)


def test_bcast_core_038_a_and_b_chain_nonconflicting_merge_is_deterministic() -> None:
    """ID: BCAST_CORE_038_a_and_b_chain_nonconflicting_merge_is_deterministic."""
    ao = AnalysisObject(_ds_single())
    left = ao.a(on="sequence", sequence_join="exact", batch_join="exact").b()
    right = ao.b().a(on="sequence", sequence_join="exact", batch_join="exact")
    left_a = read_alignment_intent(left, owner="test.ao.chain.left")
    right_a = read_alignment_intent(right, owner="test.ao.chain.right")
    left_b = read_broadcast_intent(left, owner="test.ao.chain.left")
    right_b = read_broadcast_intent(right, owner="test.ao.chain.right")
    assert left_a == right_a
    assert left_b == right_b
    xr.testing.assert_identical(left.unsafe_data, right.unsafe_data)


def test_bcast_hard_032_b_and_a_chain_conflicts_fail_closed_with_owner_context() -> None:
    """ID: BCAST_HARD_032_b_and_a_chain_conflicts_fail_closed_with_owner_context."""
    ao = AnalysisObject(_ds_single())
    with pytest.raises(ValueError, match="^AnalysisObject\\.a: conflicting chained alignment intents"):
        _ = ao.a(on="sequence").a(on="param", sequence_join=None)


def test_typed_lifecycle_001_context_and_spec_validation() -> None:
    """ID: TYPED_LIFECYCLE_001_context_and_spec_validation."""
    ctx = TypedLifecycleContext(owner="typed.test", phase="init", options={"x": 1})
    assert ctx.owner == "typed.test"
    assert ctx.phase == "init"

    with pytest.raises(ValueError, match="type_name"):
        TypedLifecycleSpec(type_name="", owner_prefix="typed.test")
    with pytest.raises(ValueError, match="owner_prefix"):
        TypedLifecycleSpec(type_name="Probe", owner_prefix="")
    with pytest.raises(TypeError, match="normalize"):
        invalid_hook = cast(Any, object())
        TypedLifecycleSpec(type_name="Probe", owner_prefix="typed.test", normalize=invalid_hook)


def test_typed_lifecycle_002_default_source_coercer_uses_canonical_ao_input() -> None:
    """ID: TYPED_LIFECYCLE_002_default_source_coercer_uses_canonical_ao_input."""
    ctx = TypedLifecycleContext(owner="typed.default", phase="init")
    ao = AnalysisObject(_ds_single())
    da = xr.DataArray(np.arange(3.0), dims=("sample",), name="value")

    assert default_coerce_source(ao, ctx) is ao
    assert isinstance(default_coerce_source(_ds_single(), ctx), AnalysisObject)
    assert isinstance(default_coerce_source(da, ctx), AnalysisObject)
    with pytest.raises(TypeError, match="^typed\\.default: expected AnalysisObject"):
        default_coerce_source(1.0, ctx)


def test_typed_lifecycle_003_default_hooks_are_safe_noops() -> None:
    """ID: TYPED_LIFECYCLE_003_default_hooks_are_safe_noops."""
    ds = _ds_single()
    ctx = TypedLifecycleContext(owner="typed.noop", phase="init")
    before = ds.copy(deep=True)

    assert identity_init_options(ds, ctx) is ds
    assert identity_normalize(ds, ctx) is ds
    assert no_op_enforce(ds, ctx) is None
    xr.testing.assert_identical(ds, before)


def test_typed_lifecycle_004_init_from_validated_from_unvalidated_ordering() -> None:
    """ID: TYPED_LIFECYCLE_004_init_from_validated_from_unvalidated_ordering."""
    calls: list[str] = []

    def coerce(value: object, ctx: TypedLifecycleContext) -> AnalysisObject:
        calls.append(f"{ctx.phase}:coerce:{ctx.owner}")
        return default_coerce_source(value, ctx)

    def apply_options(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
        calls.append(f"{ctx.phase}:options:{ctx.owner}")
        return ds

    def normalize(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
        calls.append(f"{ctx.phase}:normalize:{ctx.owner}")
        return ds

    def enforce(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
        calls.append(f"{ctx.phase}:enforce:{ctx.owner}")

    class Probe(TypedAnalysisObject):
        LIFECYCLE = TypedLifecycleSpec(
            type_name="Probe",
            owner_prefix="typed.probe",
            coerce_source=coerce,
            apply_init_options=apply_options,
            normalize=normalize,
            enforce=enforce,
        )

    _ = Probe(_ds_single())
    assert calls == [
        "init:coerce:typed.probe.__init__",
        "init:options:typed.probe.__init__",
        "init:normalize:typed.probe.__init__",
        "init:enforce:typed.probe.__init__",
    ]

    calls.clear()
    _ = Probe._from_validated(_ds_single())
    assert calls == [
        "from_validated:normalize:Probe._from_validated",
        "from_validated:enforce:Probe._from_validated",
    ]

    calls.clear()
    _ = Probe._from_unvalidated(_ds_single())
    assert calls == [
        "from_unvalidated:normalize:Probe._from_unvalidated",
        "from_unvalidated:enforce:Probe._from_unvalidated",
    ]


def test_typed_lifecycle_005_self_typed_rewrap_contract() -> None:
    """ID: TYPED_LIFECYCLE_005_self_typed_rewrap_contract."""

    class Probe(TypedAnalysisObject):
        LIFECYCLE = TypedLifecycleSpec(type_name="Probe", owner_prefix="typed.probe")

    assert isinstance(Probe._from_validated(_ds_single()), Probe)
    assert isinstance(Probe._from_unvalidated(_ds_single()), Probe)
    assert TypedAnalysisObject._from_validated.__annotations__["return"] == "Self"
    assert TypedAnalysisObject._from_unvalidated.__annotations__["return"] == "Self"
