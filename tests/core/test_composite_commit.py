from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from tal.core import AnalysisObject, ComponentSpec, SchemaError
from tal.core.component_ops import read_components
from tal.core.dataset_ownership import analysis_object_dataset, couple_dataset_resource
from tal.core.orchestration.composite_commit import (
    _commit_composite_result,
    _ComponentRegistryCommit,
    _CompositeCommitSpec,
)
from tal.core.orchestration.schema_finalize import CoreSchemaFinalizeSpec
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)


class _CopyProbe:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __deepcopy__(self, memo: dict[int, object]) -> _CopyProbe:
        _ = memo
        self.calls.append("copy")
        return self


class _RaisingCopyProbe:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def __deepcopy__(self, memo: dict[int, object]) -> _RaisingCopyProbe:
        _ = memo
        raise self.error


class _ContextAnalysisObject(AnalysisObject):
    wrappers = 0

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> _ContextAnalysisObject:
        cls.wrappers += 1
        return super()._from_validated(ds)

    @classmethod
    def _from_unvalidated(
        cls,
        ds: xr.Dataset | xr.DataArray,
        *,
        schema_prepared: bool = False,
    ) -> _ContextAnalysisObject:
        cls.wrappers += 1
        return super()._from_unvalidated(ds, schema_prepared=schema_prepared)

    def _apply_result_rewrap_context(
        self,
        result: AnalysisObject,
        *,
        context: object | None,
    ) -> AnalysisObject:
        result.applied_context = context  # type: ignore[attr-defined]
        return result


def _source_dataset(*, data: object | None = None) -> xr.Dataset:
    values = np.arange(12.0).reshape(2, 2, 3) if data is None else data
    ds = xr.Dataset(
        {"vector": (("trial", "sample", "axis"), values)},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1],
            "axis": ["x", "y", "z"],
            "time": ("sample", [0.0, 1.0]),
            "group_size": ("trial", [2, 1]),
        },
        attrs={"ordinary": {"nested": True}},
    )
    source = AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=("axis",),
        param_coord="time",
        sequence_size_coord="group_size",
        validate=True,
    )
    return analysis_object_dataset(source)


def _schema_spec(*, core_dims: tuple[str, ...] = ("axis",)) -> CoreSchemaFinalizeSpec:
    return CoreSchemaFinalizeSpec(
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=core_dims,
        param_name="time",
        size_name="group_size",
    )


def _registry_commit() -> _ComponentRegistryCommit:
    return _ComponentRegistryCommit(
        action="replace",
        entries=(("xy", ComponentSpec("axis", ("x", "y"), var="vector")),),
    )


def _spec(
    prototype: AnalysisObject,
    *,
    validate: bool,
    components: _ComponentRegistryCommit | None = None,
    context: object | None = None,
    resource_action: Callable[[AnalysisObject], None] | None = None,
    schema: CoreSchemaFinalizeSpec | None = None,
) -> _CompositeCommitSpec:
    return _CompositeCommitSpec(
        owner="composite.example",
        validate=validate,
        schema=_schema_spec() if schema is None else schema,
        components=_registry_commit() if components is None else components,
        prototype=prototype,
        result_context=context,
        resource_action=resource_action,
    )


@pytest.mark.parametrize("validate", [False, True])
def test_composite_commit_stamps_schema_registry_context_once(validate: bool) -> None:
    candidate = _source_dataset()
    copies: list[str] = []
    tal = deepcopy(candidate.attrs["tal"])
    tal.setdefault("ext", {})["custom"] = {"probe": _CopyProbe(copies)}
    candidate = candidate.assign_attrs({**candidate.attrs, "tal": tal})
    prototype = _ContextAnalysisObject._from_validated(candidate)
    _ContextAnalysisObject.wrappers = 0
    context = object()

    result = _commit_composite_result(
        candidate,
        spec=_spec(prototype, validate=validate, context=context),
    )

    assert isinstance(result, _ContextAnalysisObject)
    assert result.applied_context is context  # type: ignore[attr-defined]
    assert _ContextAnalysisObject.wrappers == 1
    assert copies == ["copy"]
    assert read_roles(analysis_object_dataset(result)) == (
        True,
        "sample",
        ("trial",),
        ("axis",),
    )
    assert read_param_coord_name(analysis_object_dataset(result)) == "time"
    assert read_sequence_size_coord_name(analysis_object_dataset(result)) == "group_size"
    assert read_components(result) == {
        "xy": ComponentSpec("axis", ("x", "y"), var="vector")
    }
    assert analysis_object_dataset(result)["vector"].data is candidate["vector"].data


def _delayed_payload(
    calls: list[str],
    *,
    error: BaseException | None = None,
) -> da.Array:
    @delayed
    def load() -> np.ndarray:
        calls.append("payload")
        if error is not None:
            raise error
        return np.arange(12.0).reshape(2, 2, 3)

    return da.from_delayed(load(), shape=(2, 2, 3), dtype=np.float64)


def _coupling_action(source: xr.Dataset) -> Callable[[AnalysisObject], None]:
    def couple(result: AnalysisObject) -> None:
        couple_dataset_resource(source, analysis_object_dataset(result))

    return couple


def test_composite_commit_lazy_result_couples_resource_without_execution() -> None:
    payload_calls: list[str] = []
    close_calls: list[str] = []
    candidate = _source_dataset(data=_delayed_payload(payload_calls))
    candidate.set_close(lambda: close_calls.append("close"))
    prototype = AnalysisObject._from_validated(candidate)

    result = _commit_composite_result(
        candidate,
        spec=_spec(
            prototype,
            validate=False,
            resource_action=_coupling_action(candidate),
        ),
    )

    assert payload_calls == []
    assert isinstance(analysis_object_dataset(result)["vector"].data, da.Array)
    result.close()
    result.close()
    assert payload_calls == []
    assert close_calls == ["close"]


def test_composite_commit_deferred_failure_retains_coupled_lifetime() -> None:
    payload_calls: list[str] = []
    close_calls: list[str] = []
    candidate = _source_dataset(
        data=_delayed_payload(payload_calls, error=RuntimeError("deferred boom"))
    )
    candidate.set_close(lambda: close_calls.append("close"))
    result = _commit_composite_result(
        candidate,
        spec=_spec(
            AnalysisObject._from_validated(candidate),
            validate=False,
            resource_action=_coupling_action(candidate),
        ),
    )

    with pytest.raises(RuntimeError, match="deferred boom"):
        analysis_object_dataset(result).compute(scheduler="synchronous")
    result.close()
    assert payload_calls == ["payload"]
    assert close_calls == ["close"]


def test_composite_commit_construction_failure_precedes_copy_payload_and_resource() -> None:
    payload_calls: list[str] = []
    copy_calls: list[str] = []
    resource_calls: list[str] = []
    candidate = _source_dataset(data=_delayed_payload(payload_calls))
    tal = deepcopy(candidate.attrs["tal"])
    tal.setdefault("ext", {})["custom"] = _CopyProbe(copy_calls)
    candidate = candidate.assign_attrs({**candidate.attrs, "tal": tal})
    invalid = _spec(
        AnalysisObject._from_validated(candidate),
        validate=False,
        schema=_schema_spec(core_dims=("missing",)),
        resource_action=lambda result: resource_calls.append(type(result).__name__),
    )

    with pytest.raises(ValueError, match="composite.example"):
        _commit_composite_result(candidate, spec=invalid)

    assert payload_calls == []
    assert copy_calls == []
    assert resource_calls == []


def test_composite_commit_preserves_schema_error_details_and_cause() -> None:
    candidate = _source_dataset()
    invalid = _spec(
        AnalysisObject._from_validated(candidate),
        validate=False,
        schema=_schema_spec(core_dims=("missing",)),
    )

    with pytest.raises(SchemaError) as captured:
        _commit_composite_result(candidate, spec=invalid)

    assert captured.value.code == "schema.roles.core_dims.not_in_dataset"
    assert captured.value.path == "tal.core.roles.core_dims"
    assert captured.value.hint.count("composite.example") == 1
    assert isinstance(captured.value.__cause__, SchemaError)


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("copy value failure"),
        TypeError("copy type failure"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
    ],
)
def test_composite_commit_propagates_copy_failure_unchanged(
    failure: Exception,
) -> None:
    candidate = _source_dataset()
    tal = deepcopy(candidate.attrs["tal"])
    tal.setdefault("ext", {})["custom"] = _RaisingCopyProbe(failure)
    candidate = candidate.assign_attrs({**candidate.attrs, "tal": tal})

    with pytest.raises(type(failure)) as captured:
        _commit_composite_result(
            candidate,
            spec=_spec(AnalysisObject._from_validated(candidate), validate=False),
        )

    assert captured.value is failure
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("failure", [ValueError("context value"), TypeError("context type")])
def test_composite_commit_propagates_context_failure_unchanged(
    failure: Exception,
) -> None:
    class _RaisingContext(AnalysisObject):
        def _apply_result_rewrap_context(
            self,
            result: AnalysisObject,
            *,
            context: object | None,
        ) -> AnalysisObject:
            _ = result, context
            raise failure

    candidate = _source_dataset()
    prototype = _RaisingContext._from_validated(candidate)

    with pytest.raises(type(failure)) as captured:
        _commit_composite_result(candidate, spec=_spec(prototype, validate=False))

    assert captured.value is failure
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("failure", [ValueError("resource value"), TypeError("resource type")])
def test_composite_commit_propagates_resource_failure_unchanged(
    failure: Exception,
) -> None:
    candidate = _source_dataset()

    def fail_resource(result: AnalysisObject) -> None:
        _ = result
        raise failure

    with pytest.raises(type(failure)) as captured:
        _commit_composite_result(
            candidate,
            spec=_spec(
                AnalysisObject._from_validated(candidate),
                validate=False,
                resource_action=fail_resource,
            ),
        )

    assert captured.value is failure
    assert captured.value.__cause__ is None


def test_composite_commit_prunes_registry_without_mutating_source() -> None:
    candidate = _source_dataset()
    with_registry = _commit_composite_result(
        candidate,
        spec=_spec(AnalysisObject._from_validated(candidate), validate=True),
    )
    registered_ds = analysis_object_dataset(with_registry)
    pruned = _commit_composite_result(
        registered_ds,
        spec=_spec(
            AnalysisObject._from_validated(registered_ds),
            validate=False,
            components=_ComponentRegistryCommit(action="prune"),
        ),
    )

    assert read_components(pruned) == {}
    assert read_components(with_registry)


def test_composite_commit_rejects_rewrapping_result_context_before_resource() -> None:
    class _ReplacingContext(AnalysisObject):
        def _apply_result_rewrap_context(
            self,
            result: AnalysisObject,
            *,
            context: object | None,
        ) -> AnalysisObject:
            _ = context
            return AnalysisObject._from_validated(analysis_object_dataset(result))

    candidate = _source_dataset()
    resource_calls: list[str] = []
    spec = _spec(
        _ReplacingContext._from_validated(candidate),
        validate=False,
        resource_action=lambda result: resource_calls.append(type(result).__name__),
    )

    with pytest.raises(
        ValueError,
        match="composite.example: result context must not create another wrapper",
    ):
        _commit_composite_result(candidate, spec=spec)
    assert resource_calls == []
