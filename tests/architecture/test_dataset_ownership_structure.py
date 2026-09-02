from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tal
import tal.core
from tal.core import AnalysisObject
from tal.core import dataset_ownership as dataset_owner
from tal.core.dataset_ownership import metadata_isolated_dataset as _direct_owner_alias
from tal.core.typed_lifecycle import TypedAnalysisObject, TypedLifecycleSpec
from tal.io.finalize import finalize_loaded_dataset

from ._budget import file_loc, function_lengths
from ._dataset_ownership_guard import (
    OwnerCopyMode,
    OwnerResultKind,
    has_owner_lineage,
    observe_owner_results,
)


ANALYSIS_OBJECT_PATH = Path("tal/core/analysis_object.py")
OWNER_MODULE = "tal.core.dataset_ownership"


class _TypedOwnershipProbe(TypedAnalysisObject):
    LIFECYCLE = TypedLifecycleSpec(
        type_name="OwnershipProbe",
        owner_prefix="architecture.ownership_probe",
    )


@dataclass(frozen=True)
class _EquivalentOwnerOptions:
    payload: xr.Dataset
    policy: OwnerCopyMode


def _equivalent_options_owner(configuration: _EquivalentOwnerOptions):  # type: ignore[no-untyped-def]
    return dataset_owner.dataset_view(
        configuration.payload,
        copy=configuration.policy,
        owner="architecture.equivalent_options_owner",
    )


def _runtime_public_owner_objects(module: object) -> set[str]:
    owner_names = set(dataset_owner.__all__)
    owner_names.update(
        name
        for name, value in vars(dataset_owner).items()
        if getattr(value, "__module__", None) == OWNER_MODULE
    )
    owner_objects = (dataset_owner,) + tuple(
        vars(dataset_owner)[name] for name in owner_names
    )
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = tuple(name for name in vars(module) if not name.startswith("_"))
    return {
        name
        for name in public_names
        if (value := getattr(module, name, None)) is not None
        and any(value is owner_value for owner_value in owner_objects)
    }


def _source(
    *,
    as_dataarray: bool = False,
    with_schema: bool = False,
) -> xr.Dataset | xr.DataArray:
    attrs = {"tal": {"version": 1, "core": {}}} if with_schema else None
    if as_dataarray:
        return xr.DataArray(
            np.arange(3.0),
            dims="sample",
            coords={"sample": np.arange(3)},
            name="value",
            attrs=attrs,
        )
    return xr.Dataset(
        {"value": ("sample", np.arange(3.0))},
        coords={"sample": np.arange(3)},
        attrs=attrs,
    )


def _assert_public_origin(
    invoke: Callable[[], object],
    select: Callable[[object], object],
    *,
    result_kind: OwnerResultKind,
    copy_mode: OwnerCopyMode,
) -> None:
    result, observed = observe_owner_results(invoke)
    value = select(result)
    assert has_owner_lineage(
        value,
        observed,
        result_kind=result_kind,
        copy_mode=copy_mode,
    )


def test_arch_dataset_ownership_001_owner_is_not_public_core_api() -> None:
    """ID: ARCH_DATASET_OWNERSHIP_001_owner_is_not_public_core_api."""
    for module in (tal, tal.core):
        assert _runtime_public_owner_objects(module) == set()


@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
@pytest.mark.parametrize("with_schema", (False, True), ids=("plain", "schema"))
def test_arch_dataset_ownership_002_constructor_ingress_reaches_deep_owner(
    as_dataarray: bool,
    with_schema: bool,
) -> None:
    """ID: ARCH_DATASET_OWNERSHIP_002_constructor_ingress_reaches_deep_owner."""
    _assert_public_origin(
        lambda: AnalysisObject(
            _source(as_dataarray=as_dataarray, with_schema=with_schema)
        ),
        lambda result: result.unsafe_data,
        result_kind="dataset",
        copy_mode="deep",
    )


@pytest.mark.parametrize("as_dataarray", (False, True), ids=("dataset", "dataarray"))
@pytest.mark.parametrize("with_schema", (False, True), ids=("plain", "schema"))
@pytest.mark.parametrize("validate", (False, True), ids=("unvalidated", "validated"))
def test_arch_dataset_ownership_003_from_data_ingress_reaches_deep_owner(
    as_dataarray: bool,
    with_schema: bool,
    validate: bool,
) -> None:
    """ID: ARCH_DATASET_OWNERSHIP_003_from_data_ingress_reaches_deep_owner."""
    _assert_public_origin(
        lambda: AnalysisObject.from_data(
            _source(as_dataarray=as_dataarray, with_schema=with_schema),
            validate=validate,
        ),
        lambda result: result.unsafe_data,
        result_kind="dataset",
        copy_mode="deep",
    )


def test_arch_dataset_ownership_004_public_exposure_reaches_owner() -> None:
    """ID: ARCH_DATASET_OWNERSHIP_004_public_exposure_reaches_owner."""
    ao = AnalysisObject(_source())
    roots: tuple[
        tuple[Callable[[], object], OwnerResultKind, OwnerCopyMode], ...
    ] = (
        (lambda: ao.data, "dataset", "deep"),
        (lambda: ao.as_dataset(), "dataset", "deep"),
        (lambda: ao.unsafe_data, "dataset", "none"),
        (lambda: ao.to_dataarray(), "dataarray", "deep"),
    )
    for invoke, result_kind, copy_mode in roots:
        _assert_public_origin(
            invoke,
            lambda result: result,
            result_kind=result_kind,
            copy_mode=copy_mode,
        )


@pytest.mark.parametrize(
    ("operation", "copy_mode", "sibling_deep"),
    (
        (dataset_owner.deep_public_dataset, "deep", True),
        (dataset_owner.metadata_isolated_dataset, "shallow", False),
    ),
    ids=("deep", "shallow"),
)
def test_arch_dataset_ownership_005_lineage_rejects_irrelevant_owner_result(
    operation: Callable[..., xr.Dataset],
    copy_mode: OwnerCopyMode,
    sibling_deep: bool,
) -> None:
    """ID: ARCH_DATASET_OWNERSHIP_005_lineage_rejects_irrelevant_owner_result."""
    source = _source()
    owner_result, observed = observe_owner_results(
        lambda: operation(source, owner="architecture.test")
    )
    derived = owner_result.copy(deep=False)
    unrelated = source.copy(deep=sibling_deep)

    assert has_owner_lineage(
        derived,
        observed,
        result_kind="dataset",
        copy_mode=copy_mode,
    )
    assert not has_owner_lineage(
        unrelated,
        observed,
        result_kind="dataset",
        copy_mode=copy_mode,
    )


@pytest.mark.parametrize("module", (tal, tal.core), ids=("tal", "tal-core"))
def test_arch_dataset_ownership_006_public_roots_have_no_dynamic_getattr(
    module: object,
) -> None:
    """ID: ARCH_DATASET_OWNERSHIP_006_public_roots_have_no_dynamic_getattr."""
    assert "__getattr__" not in vars(module)


def test_arch_dataset_ownership_008_changed_production_stays_within_budgets() -> None:
    """ID: ARCH_DATASET_OWNERSHIP_008_changed_production_stays_within_budgets."""
    owner_path = Path("tal/core/dataset_ownership.py")
    assert file_loc(path=owner_path) <= 600
    assert file_loc(path=ANALYSIS_OBJECT_PATH) <= 600
    for name, length in function_lengths(owner_path).items():
        assert length <= 50, f"{owner_path}:{name} exceeds function budget"
    assert function_lengths(ANALYSIS_OBJECT_PATH)["AnalysisObject.from_data"] <= 50


def test_arch_dataset_ownership_009_domain_consumers_reach_owner() -> None:
    """ID: ARCH_DATASET_OWNERSHIP_009_domain_consumers_reach_owner."""
    _assert_public_origin(
        lambda: _TypedOwnershipProbe(_source()),
        lambda result: result.unsafe_data,
        result_kind="dataset",
        copy_mode="deep",
    )
    persisted = AnalysisObject.from_data(
        _source(),
        sequence_dim="sample",
        core_dims=(),
        validate=True,
    ).unsafe_data
    _assert_public_origin(
        lambda: finalize_loaded_dataset(
            AnalysisObject,
            persisted,
            validate=True,
            owner="architecture.io_finalize",
        ),
        lambda result: result.unsafe_data,
        result_kind="dataset",
        copy_mode="deep",
    )


def test_arch_dataset_ownership_010_frozen_options_and_renamed_args_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: ARCH_DATASET_OWNERSHIP_010_frozen_options_and_renamed_args_supported."""
    export_name = "equivalent_options_owner"
    monkeypatch.setattr(
        dataset_owner,
        export_name,
        _equivalent_options_owner,
        raising=False,
    )
    monkeypatch.setattr(dataset_owner, "__all__", (*dataset_owner.__all__, export_name))
    source = _source()

    result, observed = observe_owner_results(
        lambda: _equivalent_options_owner(
            _EquivalentOwnerOptions(source, "shallow")
        )
    )

    assert any(
        item.operation == export_name and item.copy_mode == "shallow"
        for item in observed
    )
    assert has_owner_lineage(
        result,
        observed,
        result_kind="dataset",
        copy_mode="shallow",
    )


def test_arch_dataset_ownership_011_direct_import_alias_supported() -> None:
    """ID: ARCH_DATASET_OWNERSHIP_011_direct_import_alias_supported."""
    source = _source()
    result, observed = observe_owner_results(
        lambda: _direct_owner_alias(source, owner="architecture.direct_alias")
    )

    assert has_owner_lineage(
        result,
        observed,
        result_kind="dataset",
        copy_mode="shallow",
    )
