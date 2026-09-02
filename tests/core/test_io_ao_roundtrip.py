from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import delayed
from dask.callbacks import Callback
from xarray.backends import BackendArray
from xarray.core import indexing

from tal.core import AnalysisObject, SchemaError
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.io import AOZarrReadOptions


class _TrackingValidityBackend(BackendArray):
    def __init__(self, values: object, *, failure: Exception | None = None) -> None:
        self._values = np.asarray(values, dtype=np.int64)
        self.shape = self._values.shape
        self.dtype = self._values.dtype
        self.failure = failure
        self.calls = 0

    def __getitem__(self, key: object) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: object) -> np.ndarray:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self._values[key]


def _with_backend_validity(backend: BackendArray) -> xr.Dataset:
    ds = _batched_roundtrip_source().as_dataset(copy="none").copy(deep=False)
    variable = xr.Variable(
        ("trial",),
        indexing.LazilyIndexedArray(backend),
    )
    return ds.assign_coords(n_valid=variable)


def _roundtrip_source() -> AnalysisObject:
    ds = xr.Dataset(
        {
            "x": ("sample", np.array([1.0, 2.0, 3.0], dtype=float)),
            "y": ("sample", np.array([4.0, 5.0, 6.0], dtype=float)),
        },
        coords={
            "sample": np.array([10, 20, 30], dtype=int),
            "param": ("sample", np.array([0.1, 0.2, 0.3], dtype=float)),
            "n_valid": np.array(3, dtype=int),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        param_coord="param",
        sequence_size_coord="n_valid",
    )


def _batched_roundtrip_source(*, cls: type[AnalysisObject] = AnalysisObject) -> AnalysisObject:
    ds = xr.Dataset(
        {"payload_probe": (("trial", "sample"), np.arange(6.0).reshape(2, 3))},
        coords={
            "trial": ["a", "b"],
            "sample": [0, 1, 2],
            "n_valid": ("trial", [2, 3]),
        },
    )
    return cls.from_data(
        ds,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="n_valid",
        validate=True,
    )


@pytest.mark.parametrize("chunks", [None, {"sample": 2}])
def test_io_core_p10a_002_ao_zarr_round_trip_preserves_schema_roles_and_validity(
    tmp_path: Path,
    chunks: object | None,
) -> None:
    """ID: IO_CORE_P10A_002_ao_zarr_round_trip_preserves_schema_roles_and_validity."""
    src = _roundtrip_source()
    store = tmp_path / "ao.zarr"
    src.io.to_zarr(str(store))
    out = AnalysisObject.from_zarr(str(store), opts=AOZarrReadOptions(chunks=chunks))
    assert read_roles(out.as_dataset(copy="none")) == read_roles(src.as_dataset(copy="none"))
    assert read_param_coord_name(out.as_dataset(copy="none")) == read_param_coord_name(src.as_dataset(copy="none"))
    assert read_sequence_size_coord_name(out.as_dataset(copy="none")) == read_sequence_size_coord_name(
        src.as_dataset(copy="none")
    )
    xr.testing.assert_allclose(out.as_dataset(copy="none")["x"], src.as_dataset(copy="none")["x"])
    xr.testing.assert_allclose(out.as_dataset(copy="none")["y"], src.as_dataset(copy="none")["y"])


@pytest.mark.parametrize("chunks", [None, {"trial": 1, "sample": 2}])
def test_io_core_p10a_008_zarr_batched_validity_roundtrip_materializes_size_only(
    tmp_path: Path,
    chunks: object | None,
) -> None:
    """ID: IO_CORE_P10A_008_zarr_batched_validity_roundtrip_materializes_size_only."""
    src = _batched_roundtrip_source()
    store = tmp_path / "batched.zarr"
    src.io.to_zarr(str(store))

    task_keys: list[object] = []
    with Callback(pretask=lambda key, _dsk, _state: task_keys.append(key)):
        out = AnalysisObject.from_zarr(str(store), opts=AOZarrReadOptions(chunks=chunks))

    assert task_keys
    assert getattr(out.as_dataset(copy="none")["payload_probe"].data, "chunks", None) is not None
    assert getattr(out.as_dataset(copy="none").coords["n_valid"].data, "chunks", None) is None
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["n_valid"], [2, 3])
    np.testing.assert_allclose(out.as_dataset(copy="none")["payload_probe"].compute(), [[0, 1, 2], [3, 4, 5]])


def test_io_perf_p10a_001_zarr_payload_nonexecution_uses_execution_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10A_001_zarr_payload_nonexecution_uses_execution_sentinel."""

    def fail_if_payload_executes() -> np.ndarray:
        raise AssertionError("Zarr ingress executed a payload task")

    src = _batched_roundtrip_source().as_dataset(copy="none")
    payload = da.from_delayed(
        delayed(fail_if_payload_executes)(),
        shape=(2, 3),
        dtype=np.dtype("float64"),
    )
    size = da.from_array(np.array([2, 3], dtype=np.int64), chunks=(1,))
    lazy = src.assign(payload_probe=(("trial", "sample"), payload))
    lazy = lazy.assign_coords(n_valid=("trial", size))
    monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: lazy)

    out = AnalysisObject.from_zarr("sentinel.zarr")

    assert getattr(out.as_dataset(copy="none")["payload_probe"].data, "chunks", None) is not None
    assert getattr(out.as_dataset(copy="none").coords["n_valid"].data, "chunks", None) is None
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["n_valid"], [2, 3])


def test_io_core_p10a_009_zarr_batched_validity_roundtrip_preserves_subclass(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10A_009_zarr_batched_validity_roundtrip_preserves_subclass."""

    class SubAO(AnalysisObject):
        pass

    src = _batched_roundtrip_source(cls=SubAO)
    store = tmp_path / "batched_subclass.zarr"
    src.io.to_zarr(str(store))
    out = SubAO.from_zarr(str(store))
    assert isinstance(out, SubAO)
    np.testing.assert_array_equal(out.as_dataset(copy="none").coords["n_valid"], [2, 3])


@pytest.mark.parametrize("validate", [True, False])
def test_io_hard_p10a_006_zarr_malformed_batched_validity_fails_after_materialization(
    tmp_path: Path,
    validate: bool,
) -> None:
    """ID: IO_HARD_P10A_006_zarr_malformed_batched_validity_fails_after_materialization."""
    malformed = _batched_roundtrip_source().as_dataset(copy="none").assign_coords(n_valid=("trial", [2, 4]))
    store = tmp_path / "malformed_batched.zarr"
    malformed.to_zarr(str(store), mode="w")
    with pytest.raises(ValueError, match="AnalysisObject.from_zarr: invalid persisted schema payload"):
        AnalysisObject.from_zarr(str(store), validate=validate)


def test_io_hard_p10a_007_zarr_schema_structure_fails_before_validity_materialization(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_007_zarr_schema_structure_fails_before_validity_materialization."""
    malformed = _batched_roundtrip_source().as_dataset(copy="none").copy(deep=True)
    tal = deepcopy(malformed.attrs["tal"])
    tal["core"]["roles"]["sequence_dim"] = "missing"
    malformed.attrs["tal"] = tal
    store = tmp_path / "malformed_structure.zarr"
    malformed.to_zarr(str(store), mode="w")

    task_keys: list[object] = []
    with pytest.raises(ValueError, match="AnalysisObject.from_zarr: invalid persisted schema payload") as err:
        with Callback(pretask=lambda key, _dsk, _state: task_keys.append(key)):
            AnalysisObject.from_zarr(str(store))

    assert task_keys == []
    assert isinstance(err.value.__cause__, SchemaError)
    assert err.value.__cause__.code == "schema.roles.sequence_dim.not_in_dataset"
    assert err.value.__cause__.path == "tal.core.roles.sequence_dim"


def test_io_hard_p10a_008_zarr_validity_dtype_fails_before_materialization(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_008_zarr_validity_dtype_fails_before_materialization."""
    malformed = _batched_roundtrip_source().as_dataset(copy="none").assign_coords(n_valid=("trial", ["2", "3"]))
    store = tmp_path / "malformed_validity_dtype.zarr"
    malformed.to_zarr(str(store), mode="w")

    task_keys: list[object] = []
    with pytest.raises(ValueError, match="AnalysisObject.from_zarr: invalid persisted schema payload") as err:
        with Callback(pretask=lambda key, _dsk, _state: task_keys.append(key)):
            AnalysisObject.from_zarr(str(store))

    assert task_keys == []
    assert isinstance(err.value.__cause__, SchemaError)
    assert err.value.__cause__.code == "schema.validity.sequence_size_coord.values.invalid"
    assert err.value.__cause__.path == "tal.core.validity.sequence_size_coord"


def test_io_hard_p10a_009_zarr_categorical_param_preserves_owner_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10A_009_zarr_categorical_param_preserves_owner_before_execution."""
    src = _roundtrip_source().as_dataset(copy="none")
    size = da.from_array(np.array(3, dtype=np.int64), chunks=())
    payload = da.from_array(np.array([1.0, 2.0, 3.0]), chunks=(2,))
    malformed = src.assign(x=("sample", payload)).assign_coords(
        param=("sample", pd.Categorical([1, 2, 3])),
        n_valid=xr.Variable((), size),
    )
    monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: malformed)

    task_keys: list[object] = []
    with pytest.raises(ValueError, match="AnalysisObject.from_zarr: invalid persisted schema payload") as err:
        with Callback(pretask=lambda key, _dsk, _state: task_keys.append(key)):
            AnalysisObject.from_zarr("categorical_param.zarr")

    assert task_keys == []
    assert isinstance(err.value.__cause__, SchemaError)
    assert err.value.__cause__.code == "schema.param_coord.dtype.invalid"
    assert err.value.__cause__.path == "tal.core.param_coord.name"


@pytest.mark.parametrize("direction", ["write", "read"])
def test_io_hard_p10a_018_zarr_nonresident_validity_failure_retains_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
) -> None:
    """ID: IO_HARD_P10A_018_zarr_nonresident_validity_failure_retains_owner."""
    backend = _TrackingValidityBackend(
        [2, 3],
        failure=RuntimeError("validity backend exploded"),
    )
    ds = _with_backend_validity(backend)
    closed: list[bool] = []

    if direction == "write":
        operation = lambda: AnalysisObject._from_unvalidated(ds).io.to_zarr(
            str(tmp_path / "failed.zarr")
        )
        pattern = "AnalysisObject.io.to_zarr: failed reading AnalysisObject sequence_size_coord"
    else:
        ds.set_close(lambda: closed.append(True))
        monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: ds)
        operation = lambda: AnalysisObject.from_zarr("failed.zarr")
        pattern = "AnalysisObject.from_zarr: failed reading persisted sequence_size_coord"

    with pytest.raises(ValueError, match=pattern) as error:
        operation()

    assert isinstance(error.value.__cause__, RuntimeError)
    assert backend.calls == 1
    assert closed == ([True] if direction == "read" else [])


def test_io_perf_p10a_002_zarr_nonresident_validity_is_materialized_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10A_002_zarr_nonresident_validity_is_materialized_once."""
    backend = _TrackingValidityBackend([2, 3])
    ao = AnalysisObject._from_unvalidated(_with_backend_validity(backend))
    written: list[xr.Dataset] = []

    def capture_write(ds: xr.Dataset, *_args: object, **_kwargs: object) -> str:
        written.append(ds)
        return "written"

    monkeypatch.setattr(xr.Dataset, "to_zarr", capture_write)

    assert ao.io.to_zarr("captured.zarr") == "written"
    assert backend.calls == 1
    assert written[0].coords["n_valid"].variable._in_memory
    np.testing.assert_array_equal(written[0].coords["n_valid"], [2, 3])


def test_io_hard_p10a_014_zarr_read_failure_closes_open_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10A_014_zarr_read_failure_closes_open_dataset."""
    malformed = _batched_roundtrip_source().as_dataset(copy="none").copy(deep=False)
    tal = deepcopy(malformed.attrs["tal"])
    tal["core"]["roles"]["sequence_dim"] = "missing"
    malformed.attrs = {**malformed.attrs, "tal": tal}
    closed: list[bool] = []
    malformed.set_close(lambda: closed.append(True))
    monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: malformed)

    with pytest.raises(ValueError, match="AnalysisObject.from_zarr: invalid persisted schema payload"):
        AnalysisObject.from_zarr("malformed.zarr")

    assert closed == [True]


def test_io_hard_p10a_017_zarr_cleanup_interrupt_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10A_017_zarr_cleanup_interrupt_preserves_primary."""
    malformed = _batched_roundtrip_source().as_dataset(copy="none").copy(deep=False)
    tal = deepcopy(malformed.attrs["tal"])
    tal["core"]["roles"]["sequence_dim"] = "missing"
    malformed.attrs = {**malformed.attrs, "tal": tal}

    def interrupt_cleanup() -> None:
        raise KeyboardInterrupt("cleanup interrupted")

    malformed.set_close(interrupt_cleanup)
    monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: malformed)

    with pytest.raises(
        ValueError,
        match="AnalysisObject.from_zarr: invalid persisted schema payload",
    ) as error:
        AnalysisObject.from_zarr("malformed-cleanup.zarr")

    assert isinstance(error.value.__cause__, SchemaError)


def test_io_hard_p10a_015_zarr_success_transfers_close_ownership_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10A_015_zarr_success_transfers_close_ownership_once."""

    class SubAO(AnalysisObject):
        pass

    source = _batched_roundtrip_source().as_dataset(copy="none")
    payload = da.from_array(np.arange(6.0).reshape(2, 3), chunks=(1, 2))
    size = da.from_array(np.array([2, 3], dtype=np.int64), chunks=(1,))
    opened = source.assign(payload_probe=(("trial", "sample"), payload))
    opened = opened.assign_coords(n_valid=("trial", size))
    closed: list[bool] = []
    opened.set_close(lambda: closed.append(True))
    monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: opened)

    out = SubAO.from_zarr("owned.zarr")

    assert isinstance(out, SubAO)
    assert getattr(out.as_dataset(copy="none")["payload_probe"].data, "chunks", None) is not None
    assert closed == []
    out.close()
    out.close()
    assert closed == [True]


@pytest.mark.parametrize("fail_subclass_close", [False, True])
def test_io_hard_p10a_016_zarr_close_composes_subclass_and_backend_ownership(
    monkeypatch: pytest.MonkeyPatch,
    fail_subclass_close: bool,
) -> None:
    """ID: IO_HARD_P10A_016_zarr_close_composes_subclass_and_backend_ownership."""
    events: list[str] = []

    class ResourceSubAO(AnalysisObject):
        def _after_bind_dataset(self) -> None:
            events.append("bind")

            def close_subclass_resource() -> None:
                events.append("subclass-close")
                if fail_subclass_close:
                    raise RuntimeError("subclass close failed")

            self._data.set_close(close_subclass_resource)

    opened = _batched_roundtrip_source().as_dataset(copy="none")
    opened.set_close(lambda: events.append("backend-close"))
    monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: opened)

    out = ResourceSubAO.from_zarr("composed-ownership.zarr")

    assert events == ["bind"]
    if fail_subclass_close:
        with pytest.raises(RuntimeError, match="subclass close failed"):
            out.close()
    else:
        out.close()
    out.close()
    assert events == ["bind", "subclass-close", "backend-close"]


def test_io_hard_p10a_010_zarr_writer_rejects_invalid_schema_before_store_mutation(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_010_zarr_writer_rejects_invalid_schema_before_store_mutation."""
    malformed = _batched_roundtrip_source().as_dataset(copy="none").assign_coords(n_valid=("trial", [2, 4]))
    ao = AnalysisObject.from_data(
        malformed,
        sequence_dim="sample",
        batch_dims=("trial",),
        core_dims=(),
        sequence_size_coord="n_valid",
        validate=False,
    )
    store = tmp_path / "invalid_write.zarr"

    with pytest.raises(
        ValueError,
        match="AnalysisObject.io.to_zarr: invalid AnalysisObject schema payload",
    ):
        ao.io.to_zarr(str(store))

    assert not store.exists()


def test_io_hard_p10a_013_zarr_writer_backend_failure_retains_public_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10A_013_zarr_writer_backend_failure_retains_public_owner."""
    source = _roundtrip_source()

    def _fail_write(*_args: object, **_kwargs: object) -> None:
        raise FileExistsError("store already exists")

    monkeypatch.setattr(xr.Dataset, "to_zarr", _fail_write)

    with pytest.raises(
        ValueError,
        match="AnalysisObject.io.to_zarr: failed writing zarr store 'existing.zarr'",
    ) as error:
        source.io.to_zarr("existing.zarr")

    assert isinstance(error.value.__cause__, FileExistsError)


def test_io_hard_p10a_001_invalid_ingest_schema_payload_fails_closed_with_owner_prefixed_error(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_001_invalid_ingest_schema_payload_fails_closed_with_owner_prefixed_error."""
    ds = xr.Dataset({"x": ("sample", [1.0, 2.0, 3.0])}, coords={"sample": [0, 1, 2]})
    store = tmp_path / "invalid.zarr"
    ds.to_zarr(str(store), mode="w")
    with pytest.raises(ValueError, match="AnalysisObject.from_zarr"):
        AnalysisObject.from_zarr(str(store))


def test_io_core_p10a_007_ao_class_loaders_preserve_requested_subclass_on_success(
    tmp_path: Path,
) -> None:
    """ID: IO_CORE_P10A_007_ao_class_loaders_preserve_requested_subclass_on_success."""

    class SubAO(AnalysisObject):
        pass

    src = SubAO.from_data(
        xr.Dataset({"x": ("sample", [1.0, 2.0, 3.0])}, coords={"sample": [0, 1, 2]}),
        sequence_dim="sample",
    )
    store = tmp_path / "sub.zarr"
    src.io.to_zarr(str(store))
    assert isinstance(SubAO.from_zarr(str(store)), SubAO)


def test_io_hard_p10a_005_subclass_constructor_unexpected_exceptions_are_not_masked_by_loader_wraps(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_005_subclass_constructor_unexpected_exceptions_are_not_masked_by_loader_wraps."""

    class CrashAO(AnalysisObject):
        def __init__(self, data: xr.Dataset | xr.DataArray) -> None:
            super().__init__(data)
            raise RuntimeError("boom constructor")

    store = tmp_path / "boom.zarr"
    _roundtrip_source().io.to_zarr(str(store))
    with pytest.raises(RuntimeError, match="boom constructor"):
        CrashAO.from_zarr(str(store))
