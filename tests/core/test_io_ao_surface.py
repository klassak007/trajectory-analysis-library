from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import xarray as xr

import tal
from tal.core import AnalysisObject
from tal.io import AOZarrReadOptions, AOZarrWriteOptions, read_csv_logs, write_csv_logs


def _ao() -> AnalysisObject:
    ds = xr.Dataset(
        {"value": ("sample", [1.0, 2.0, 3.0])},
        coords={"sample": [0, 1, 2]},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample")


def test_io_core_p10a_001_analysisobject_has_direct_io_surface() -> None:
    """ID: IO_CORE_P10A_001_analysisobject_has_direct_io_surface."""
    assert hasattr(tal.AnalysisObject, "io")
    assert hasattr(tal.AnalysisObject, "from_zarr")
    assert hasattr(_ao().io, "to_zarr")
    assert not hasattr(tal.AnalysisObject, "from_csv")
    assert not hasattr(_ao().io, "to_csv")
    assert callable(read_csv_logs)
    assert callable(write_csv_logs)


def test_io_core_p10a_004_ao_direct_io_uses_options_objects_for_multi_policy_controls() -> None:
    """ID: IO_CORE_P10A_004_ao_direct_io_uses_options_objects_for_multi_policy_controls."""
    signature_io = inspect.signature(AnalysisObject.io.fget)  # type: ignore[attr-defined]
    assert "self" in signature_io.parameters
    assert isinstance(AOZarrWriteOptions(), AOZarrWriteOptions)
    assert isinstance(AOZarrReadOptions(), AOZarrReadOptions)


def test_io_hard_p10a_002_unsupported_ao_direct_io_input_type_fails_closed() -> None:
    """ID: IO_HARD_P10A_002_unsupported_ao_direct_io_input_type_fails_closed."""
    with pytest.raises(TypeError, match="AnalysisObject.from_zarr"):
        AnalysisObject.from_zarr(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "opts",
    [
        AOZarrWriteOptions(mode="replace"),  # type: ignore[arg-type]
        AOZarrWriteOptions(mode="a"),  # type: ignore[arg-type]
        AOZarrWriteOptions(mode="a-"),  # type: ignore[arg-type]
        AOZarrWriteOptions(mode="r+"),  # type: ignore[arg-type]
        AOZarrWriteOptions(consolidated="yes"),  # type: ignore[arg-type]
    ],
)
def test_io_hard_p10a_011_zarr_write_options_fail_before_backend_access(
    tmp_path: Path,
    opts: AOZarrWriteOptions,
) -> None:
    """ID: IO_HARD_P10A_011_zarr_write_options_fail_before_backend_access."""
    store = tmp_path / "invalid_options.zarr"
    with pytest.raises((TypeError, ValueError), match="AnalysisObject.io.to_zarr"):
        _ao().io.to_zarr(str(store), opts=opts)
    assert not store.exists()


def test_io_hard_p10a_012_zarr_read_options_fail_before_backend_access() -> None:
    """ID: IO_HARD_P10A_012_zarr_read_options_fail_before_backend_access."""
    opts = AOZarrReadOptions(consolidated="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AnalysisObject.from_zarr: consolidated must be bool"):
        AnalysisObject.from_zarr("missing.zarr", opts=opts)
