from __future__ import annotations

import inspect

import pytest
import xarray as xr

import tal
from tal.core import AnalysisObject
from tal.io import AOCsvReadOptions, AOCsvWriteOptions, AOZarrReadOptions, AOZarrWriteOptions


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
    assert hasattr(tal.AnalysisObject, "from_csv")
    assert hasattr(_ao().io, "to_zarr")
    assert hasattr(_ao().io, "to_csv")


def test_io_core_p10a_004_ao_direct_io_uses_options_objects_for_multi_policy_controls() -> None:
    """ID: IO_CORE_P10A_004_ao_direct_io_uses_options_objects_for_multi_policy_controls."""
    signature_to_csv = inspect.signature(AnalysisObject.io.fget)  # type: ignore[attr-defined]
    assert "self" in signature_to_csv.parameters
    assert isinstance(AOCsvWriteOptions(), AOCsvWriteOptions)
    assert isinstance(AOCsvReadOptions(), AOCsvReadOptions)
    assert isinstance(AOZarrWriteOptions(), AOZarrWriteOptions)
    assert isinstance(AOZarrReadOptions(), AOZarrReadOptions)


def test_io_hard_p10a_002_unsupported_ao_direct_io_input_type_fails_closed() -> None:
    """ID: IO_HARD_P10A_002_unsupported_ao_direct_io_input_type_fails_closed."""
    with pytest.raises(TypeError, match="AnalysisObject.from_csv"):
        AnalysisObject.from_csv(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AnalysisObject.from_zarr"):
        AnalysisObject.from_zarr(123)  # type: ignore[arg-type]
