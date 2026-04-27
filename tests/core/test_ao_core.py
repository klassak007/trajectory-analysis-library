import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core import SchemaError
from tal.core.orchestration.alignment_intent import read_alignment_intent
from tal.core.orchestration.broadcast_intent import read_broadcast_intent
from tal.core.schema_read import read_roles
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


def test_ao_init_010_constructor_rejects_multiindex() -> None:
    """ID: AO_INIT_010_constructor_rejects_multiindex."""
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


def test_ao_fromdata_010_rejects_multiindex() -> None:
    """ID: AO_FROMDATA_010_rejects_multiindex."""
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
