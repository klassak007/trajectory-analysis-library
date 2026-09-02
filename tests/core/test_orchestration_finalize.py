from __future__ import annotations

from copy import deepcopy

import pytest
import xarray as xr

from tal.core import AnalysisObject, SchemaError, merge_schema, set_roles
from tal.core.orchestration.finalize import transfer_dataset_attrs
from tal.core.orchestration.schema_finalize import (
    CoreSchemaFinalizeSpec,
    finalize_with_schema,
)


def _source_dataset() -> xr.Dataset:
    source = xr.Dataset(
        {"value": ("sample", [1.0])},
        attrs={"note": "source"},
    )
    return set_roles(source, sequence_dim="sample", core_dims=())


def test_orch_final_001_transfer_replaces_attrs_and_isolates_schema() -> None:
    """ID: ORCH_FINAL_001_transfer_replaces_attrs_and_isolates_schema."""
    source = _source_dataset()
    target = xr.Dataset(
        {"result": ("sample", [2.0])},
        attrs={"stale": True},
    )

    out = transfer_dataset_attrs(source, target, validate=True)

    assert out.attrs["note"] == "source"
    assert "stale" not in out.attrs
    assert out.attrs["tal"] == source.attrs["tal"]
    assert out.attrs["tal"] is not source.attrs["tal"]
    assert target.attrs == {"stale": True}


def test_orch_final_002_transfer_is_atomic_on_validation_failure() -> None:
    """ID: ORCH_FINAL_002_transfer_is_atomic_on_validation_failure."""
    source = _source_dataset()
    target = xr.Dataset(
        {"result": ("other", [2.0])},
        attrs={"stale": True},
    )
    source_before = deepcopy(source.attrs)
    target_before = deepcopy(target.attrs)

    with pytest.raises(SchemaError, match="sequence_dim.not_in_dataset"):
        transfer_dataset_attrs(source, target, validate=True)

    assert source.attrs == source_before
    assert target.attrs == target_before


def test_orch_final_003_unvalidated_transfer_keeps_temporary_schema() -> None:
    """ID: ORCH_FINAL_003_unvalidated_transfer_keeps_temporary_schema."""
    source = _source_dataset()
    target = xr.Dataset({"result": ("other", [2.0])})

    out = transfer_dataset_attrs(source, target, validate=False)

    roles = out.attrs["tal"]["core"]["roles"]
    assert roles["sequence_dim"] == "sample"
    assert out.attrs["tal"] is not source.attrs["tal"]
    assert target.attrs == {}


def test_orch_final_004_missing_schema_behavior_is_explicit() -> None:
    """ID: ORCH_FINAL_004_missing_schema_behavior_is_explicit."""
    source = xr.Dataset(attrs={"note": "source"})
    target = xr.Dataset(attrs={"stale": True})

    out = transfer_dataset_attrs(source, target, validate=False)

    assert out.attrs == {"note": "source"}
    with pytest.raises(SchemaError, match="schema.version.invalid"):
        transfer_dataset_attrs(source, target, validate=True)
    assert source.attrs == {"note": "source"}
    assert target.attrs == {"stale": True}


def test_orch_final_005_unknown_extensions_are_independently_preserved() -> None:
    """ID: ORCH_FINAL_005_unknown_extensions_are_independently_preserved."""
    source = merge_schema(
        _source_dataset(),
        {"ext": {"custom": {"values": [1, 2]}}},
        validate=True,
    )

    out = transfer_dataset_attrs(source, xr.Dataset(), validate=False)
    out.attrs["tal"]["ext"]["custom"]["values"].append(3)

    assert source.attrs["tal"]["ext"]["custom"]["values"] == [1, 2]


class _CopyProbe:
    calls = 0

    def __deepcopy__(self, memo: dict[int, object]) -> "_CopyProbe":
        type(self).calls += 1
        return type(self)()


def test_orch_final_006_unvalidated_transfer_copies_schema_once() -> None:
    """ID: ORCH_FINAL_006_unvalidated_transfer_copies_schema_once."""
    _CopyProbe.calls = 0
    source = merge_schema(
        _source_dataset(),
        {"ext": {"probe": _CopyProbe()}},
        validate=False,
    )
    _CopyProbe.calls = 0

    transfer_dataset_attrs(source, xr.Dataset(), validate=False)

    assert _CopyProbe.calls == 1


def test_orch_final_007_malformed_source_schema_fails_before_transfer() -> None:
    """ID: ORCH_FINAL_007_malformed_source_schema_fails_before_transfer."""
    source = xr.Dataset(attrs={"tal": []})
    target = xr.Dataset(attrs={"stale": True})

    with pytest.raises(SchemaError, match="schema.not_mapping"):
        transfer_dataset_attrs(source, target, validate=False)

    assert source.attrs == {"tal": []}
    assert target.attrs == {"stale": True}


def test_orch_final_008_sequence_free_plan_clears_inherited_sequence_schema() -> None:
    """ID: ORCH_FINAL_008_sequence_free_plan_clears_inherited_sequence_schema."""
    source_ds = _source_dataset()
    source = AnalysisObject._from_validated(source_ds)
    candidate = transfer_dataset_attrs(
        source_ds,
        xr.Dataset({"result": ("other", [2.0])}),
        validate=False,
    )
    spec = CoreSchemaFinalizeSpec(
        sequence_dim=None,
        batch_dims=(),
        core_dims=(),
        param_name=None,
        size_name=None,
    )

    out = finalize_with_schema(
        source,
        candidate,
        spec=spec,
        validate=True,
        owner="test.sequence_free",
    )

    core = out.as_dataset(copy="none").attrs["tal"]["core"]
    assert core["roles"] == {"batch_dims": [], "core_dims": []}
    assert "param_coord" not in core
    assert "validity" not in core


def test_orch_final_009_public_dataset_deep_copies_schema_once() -> None:
    """ID: ORCH_FINAL_009_public_dataset_deep_copies_schema_once."""
    source = merge_schema(
        _source_dataset(),
        {"ext": {"probe": _CopyProbe(), "custom": {"values": [1, 2]}}},
        validate=False,
    )
    ao = AnalysisObject._from_validated(source)
    _CopyProbe.calls = 0

    out = ao.as_dataset()
    out.attrs["tal"]["ext"]["custom"]["values"].append(3)

    assert _CopyProbe.calls == 1
    assert source.attrs["tal"]["ext"]["custom"]["values"] == [1, 2]
