from __future__ import annotations

import importlib

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.geo import ProjectedPosition, transform_crs


def _projected_dataset() -> xr.Dataset:
    ds = xr.Dataset(
        {"position": (("sample", "projected"), np.asarray([[500000.0, 4100000.0]], dtype=float))},
        coords={"sample": [0], "projected": ["easting", "northing"]},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=("projected",), validate=True).as_dataset(copy="none")


def test_geo_core_g4_001_tal_geo_imports_without_pyproj() -> None:
    """Importing tal.geo exposes CRS symbols without importing pyproj."""
    geo = importlib.import_module("tal.geo")
    assert hasattr(geo, "ProjectedPosition")
    assert hasattr(geo, "transform_crs")


def test_geo_hard_g4_004_backend_object_leakage_rejected() -> None:
    """ID: GEO_HARD_G4_004_backend_object_leakage_rejected."""
    with pytest.raises(TypeError, match="dst"):
        transform_crs(object(), dst={"type": "GeographicCRS"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="crs"):
        ProjectedPosition.from_projected(_projected_dataset(), crs={"type": "ProjectedCRS"})  # type: ignore[arg-type]


def test_geo_hard_g4_transform_crs_invalid_value_fails_before_pyproj(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid string dst should not trigger pyproj before unsupported value fails."""
    import tal.geo.backends as backends

    def missing(name: str):
        if name == "pyproj":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(backends, "import_module", missing)
    with pytest.raises(TypeError, match="value must be GeodeticPosition"):
        transform_crs(object(), dst="EPSG:4979")


def test_geo_hard_g4_006_missing_pyproj_api_raises_guided_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: GEO_HARD_G4_006_missing_pyproj_api_raises_guided_importerror."""
    import tal.geo.backends as backends

    def missing(name: str):
        if name == "pyproj":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(backends, "import_module", missing)
    with pytest.raises(ImportError, match=r"tal\[geo\]"):
        ProjectedPosition.from_projected(_projected_dataset(), crs="EPSG:32611")
