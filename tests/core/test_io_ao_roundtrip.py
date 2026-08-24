from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import delayed
from dask.callbacks import Callback

from tal.core import AnalysisObject, SchemaError
from tal.core.schema_read import read_param_coord_name, read_roles, read_sequence_size_coord_name
from tal.io import AOZarrReadOptions


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
    assert read_roles(out.unsafe_data) == read_roles(src.unsafe_data)
    assert read_param_coord_name(out.unsafe_data) == read_param_coord_name(src.unsafe_data)
    assert read_sequence_size_coord_name(out.unsafe_data) == read_sequence_size_coord_name(src.unsafe_data)
    xr.testing.assert_allclose(out.unsafe_data["x"], src.unsafe_data["x"])
    xr.testing.assert_allclose(out.unsafe_data["y"], src.unsafe_data["y"])


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
    assert getattr(out.unsafe_data["payload_probe"].data, "chunks", None) is not None
    assert getattr(out.unsafe_data.coords["n_valid"].data, "chunks", None) is None
    np.testing.assert_array_equal(out.unsafe_data.coords["n_valid"], [2, 3])
    np.testing.assert_allclose(
        out.unsafe_data["payload_probe"].compute(),
        [[0, 1, 2], [3, 4, 5]],
    )


def test_io_perf_p10a_001_zarr_payload_nonexecution_uses_execution_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_PERF_P10A_001_zarr_payload_nonexecution_uses_execution_sentinel."""

    def fail_if_payload_executes() -> np.ndarray:
        raise AssertionError("Zarr ingress executed a payload task")

    src = _batched_roundtrip_source().unsafe_data
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

    assert getattr(out.unsafe_data["payload_probe"].data, "chunks", None) is not None
    assert getattr(out.unsafe_data.coords["n_valid"].data, "chunks", None) is None
    np.testing.assert_array_equal(out.unsafe_data.coords["n_valid"], [2, 3])


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
    np.testing.assert_array_equal(out.unsafe_data.coords["n_valid"], [2, 3])


def test_io_hard_p10a_006_zarr_malformed_batched_validity_fails_after_materialization(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_006_zarr_malformed_batched_validity_fails_after_materialization."""
    src = _batched_roundtrip_source()
    malformed = src.unsafe_data.assign_coords(n_valid=("trial", [2, 4]))
    store = tmp_path / "malformed_batched.zarr"
    malformed.to_zarr(str(store), mode="w")
    with pytest.raises(ValueError, match="AnalysisObject.from_zarr: invalid persisted schema payload"):
        AnalysisObject.from_zarr(str(store))


def test_io_hard_p10a_007_zarr_schema_structure_fails_before_validity_materialization(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_007_zarr_schema_structure_fails_before_validity_materialization."""
    src = _batched_roundtrip_source()
    malformed = src.unsafe_data.copy(deep=True)
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
    src = _batched_roundtrip_source()
    malformed = src.unsafe_data.assign_coords(n_valid=("trial", ["2", "3"]))
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
    src = _roundtrip_source().unsafe_data
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


def test_io_core_p10a_003_ao_single_object_csv_round_trip_is_deterministic(tmp_path: Path) -> None:
    """ID: IO_CORE_P10A_003_ao_single_object_csv_round_trip_is_deterministic."""
    src = _roundtrip_source()
    csv_path = tmp_path / "ao.csv"
    src.io.to_csv(str(csv_path))
    out = AnalysisObject.from_csv(str(csv_path))
    xr.testing.assert_allclose(out.unsafe_data["x"], src.unsafe_data["x"])
    xr.testing.assert_allclose(out.unsafe_data["y"], src.unsafe_data["y"])
    xr.testing.assert_allclose(out.unsafe_data.coords["sample"], src.unsafe_data.coords["sample"])
    assert read_roles(out.unsafe_data) == read_roles(src.unsafe_data)


def test_io_core_p10a_005_single_object_csv_rejects_nonrepresentable_higher_rank_payloads_fail_closed(tmp_path: Path) -> None:
    """ID: IO_CORE_P10A_005_single_object_csv_rejects_nonrepresentable_higher_rank_payloads_fail_closed."""
    ds = xr.Dataset(
        {"x": (("sample", "axis"), np.arange(6, dtype=float).reshape(3, 2))},
        coords={"sample": [0, 1, 2], "axis": [0, 1]},
    )
    ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=["axis"])
    with pytest.raises(ValueError, match="AnalysisObject.io.to_csv"):
        ao.io.to_csv(str(tmp_path / "bad.csv"))


def test_io_core_p10a_006_csv_metadata_sidecar_default_is_deterministic(tmp_path: Path) -> None:
    """ID: IO_CORE_P10A_006_csv_metadata_sidecar_default_is_deterministic."""
    src = _roundtrip_source()
    csv_path = tmp_path / "deterministic.csv"
    sidecar = csv_path.with_suffix(".tal.json")
    src.io.to_csv(str(csv_path))
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["sequence_dim"] == "sample"


def test_io_hard_p10a_001_invalid_ingest_schema_payload_fails_closed_with_owner_prefixed_error(tmp_path: Path) -> None:
    """ID: IO_HARD_P10A_001_invalid_ingest_schema_payload_fails_closed_with_owner_prefixed_error."""
    ds = xr.Dataset({"x": ("sample", [1.0, 2.0, 3.0])}, coords={"sample": [0, 1, 2]})
    store = tmp_path / "invalid.zarr"
    ds.to_zarr(str(store), mode="w")
    with pytest.raises(ValueError, match="AnalysisObject.from_zarr"):
        AnalysisObject.from_zarr(str(store))


def test_io_core_p10a_007_ao_class_loaders_preserve_requested_subclass_on_success(tmp_path: Path) -> None:
    """ID: IO_CORE_P10A_007_ao_class_loaders_preserve_requested_subclass_on_success."""

    class SubAO(AnalysisObject):
        pass

    src = SubAO.from_data(
        xr.Dataset({"x": ("sample", [1.0, 2.0, 3.0])}, coords={"sample": [0, 1, 2]}),
        sequence_dim="sample",
    )
    store = tmp_path / "sub.zarr"
    csv_path = tmp_path / "sub.csv"
    src.io.to_zarr(str(store))
    src.io.to_csv(str(csv_path))
    rt_zarr = SubAO.from_zarr(str(store))
    rt_csv = SubAO.from_csv(str(csv_path))
    assert isinstance(rt_zarr, SubAO)
    assert isinstance(rt_csv, SubAO)


def test_io_hard_p10a_004_csv_sidecar_malformed_container_types_fail_closed_with_owner_prefix(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_004_csv_sidecar_malformed_container_types_fail_closed_with_owner_prefix."""
    src = _roundtrip_source()
    csv_path = tmp_path / "broken.csv"
    sidecar = csv_path.with_suffix(".tal.json")
    src.io.to_csv(str(csv_path))
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    cases = (
        {"data_vars": 1},
        {"coords": 1},
        {"data_vars": ["x", 1]},
        {"coords": ["param", ""]},
        {"scalar_coords": 1},
        {"scalar_coords": {"": 1}},
    )
    for patch in cases:
        broken = dict(payload)
        broken.update(patch)
        sidecar.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="AnalysisObject.from_csv"):
            AnalysisObject.from_csv(str(csv_path))


def test_io_hard_p10a_005_subclass_constructor_unexpected_exceptions_are_not_masked_by_loader_wraps(
    tmp_path: Path,
) -> None:
    """ID: IO_HARD_P10A_005_subclass_constructor_unexpected_exceptions_are_not_masked_by_loader_wraps."""

    class CrashAO(AnalysisObject):
        def __init__(self, data: xr.Dataset | xr.DataArray) -> None:
            super().__init__(data)
            raise RuntimeError("boom constructor")

    src = _roundtrip_source()
    store = tmp_path / "boom.zarr"
    csv_path = tmp_path / "boom.csv"
    src.io.to_zarr(str(store))
    src.io.to_csv(str(csv_path))

    with pytest.raises(RuntimeError, match="boom constructor"):
        CrashAO.from_zarr(str(store))
    with pytest.raises(RuntimeError, match="boom constructor"):
        CrashAO.from_csv(str(csv_path))
