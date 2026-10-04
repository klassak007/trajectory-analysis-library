"""Contract 135B: registration is explicit and independent of domain execution."""

import doctest
from inspect import getattr_static
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from tal.core import AnalysisObject
from tal.spatial import Position
from tal_extensions.geo import (
    GeodeticPosition,
    GeodeticInterpolationOptions,
    LocalOrigin,
    register_position_accessor,
)
from tal_extensions.geo import accessor


@pytest.fixture
def unregistered():
    missing = object()
    original = getattr_static(Position, "geo", missing)
    if original is not missing:
        delattr(Position, "geo")
    yield
    if "geo" in vars(Position):
        delattr(Position, "geo")
    if original is not missing:
        setattr(Position, "geo", original)


def _location(*, lazy, empty):
    values = np.empty((0, 3)) if empty else np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ds = xr.Dataset(
        {"position": (("sample", "lla"), values)},
        coords={
            "lla": ["lat", "lon", "alt"],
            "time": ("sample", np.arange(len(values), dtype=float)),
        },
    )
    if lazy:
        ds["position"] = ds.position.chunk(sample=2)
    return GeodeticPosition.from_lla(
        AnalysisObject.from_data(
            ds, sequence_dim="sample", core_dims=("lla",), param_coord="time"
        )
    )


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize(
    "method", ("nearest", "geodesic_linear", "ecef_linear", "local_enu_linear")
)
@pytest.mark.parametrize("operation", ("at", "resample_to", "interp_like"))
def test_interpolation_without_accessor(unregistered, lazy, empty, method, operation):
    source = _location(lazy=lazy, empty=empty)
    before = source.as_dataset(copy="deep")
    opts = GeodeticInterpolationOptions(
        method=method, local_origin=LocalOrigin(lat=0.0, lon=0.0, alt=0.0)
    )
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        argument = [0.5]
        if operation == "interp_like":
            argument = AnalysisObject.from_data(
                xr.Dataset(
                    {"other": ("sample", [0.0])}, coords={"time": ("sample", [0.5])}
                ),
                sequence_dim="sample",
                param_coord="time",
            )
        actual = getattr(source.param, operation)(argument, opts=opts, validate=False)
    assert not tasks and not hasattr(Position, "geo")
    ds = actual.as_dataset()
    assert set(ds.data_vars) == {"position"}
    expected_dims = (
        ("sample", "lla") if method == "geodesic_linear" else ("lla", "sample")
    )
    assert ds.position.dims == expected_dims and dict(ds.sizes) == {
        "sample": 1,
        "lla": 3,
    }
    assert ds.attrs["tal"]["ext"]["geo"]["kind"] == "geodetic_position"
    if lazy:
        assert ds.position.chunks is not None
    computed = ds.compute(scheduler="synchronous")
    if empty:
        assert np.isnan(computed.position).all()
    else:
        expected_lon = 0.0 if method == "nearest" else 0.5
        np.testing.assert_allclose(
            computed.position.sel(lla="lon"), [expected_lon], atol=1e-7
        )
    xr.testing.assert_identical(source.as_dataset(), before)


def test_typed_conversions_without_accessor(unregistered):
    source = _location(lazy=False, empty=False)
    before = source.as_dataset(copy="deep")
    ecef = source.to_ecef()
    restored = GeodeticPosition.from_ecef(ecef)
    projected = source.to_crs("EPSG:3857")
    assert not hasattr(Position, "geo")
    np.testing.assert_allclose(
        restored.as_dataset().position, before.position, atol=1e-6
    )
    assert (
        projected.as_dataset().attrs["tal"]["ext"]["geo"]["kind"]
        == "projected_position"
    )
    xr.testing.assert_identical(source.as_dataset(), before)


def test_registration_is_idempotent_and_documented(unregistered):
    register_position_accessor()
    descriptor = Position.geo
    register_position_accessor()
    assert Position.geo is descriptor
    test = doctest.DocTestParser().get_doctest(
        register_position_accessor.__doc__, vars(accessor), "registration", None, 0
    )
    result = doctest.DocTestRunner().run(test)
    assert result.failed == 0 and result.attempted > 0


@pytest.mark.parametrize("occupied", (None, property(lambda self: "other"), object()))
def test_registration_rejects_conflicting_attribute(unregistered, occupied):
    Position.geo = occupied
    with pytest.raises(
        RuntimeError,
        match="^geo.register_position_accessor: Position.geo is already occupied",
    ):
        register_position_accessor()
    assert Position.geo is occupied


def test_registration_rejects_descriptor_without_invoking_it(unregistered):
    class OccupiedDescriptor:
        def __get__(self, instance, owner):
            raise AssertionError("registration must not invoke a competing descriptor")

    occupied = OccupiedDescriptor()
    Position.geo = occupied
    with pytest.raises(RuntimeError, match="^geo.register_position_accessor:"):
        register_position_accessor()
    assert vars(Position)["geo"] is occupied


def test_imports_do_not_register_or_load_backends(tmp_path):
    repo = Path(__file__).resolve().parents[4]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(repo), str(repo / "extensions/src"))),
    }
    script = """
import importlib.abc
import sys
class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyproj', 'astropy', 'spiceypy'}:
            raise AssertionError('eager backend: ' + fullname)
sys.meta_path.insert(0, BlockBackends())
import tal
from tal.spatial import Position
assert not hasattr(Position, 'geo')
import tal_extensions
assert not any(name.startswith('tal_extensions.') for name in sys.modules)
import tal_extensions.geo
import tal_extensions.astro
import tal_extensions.astro.sun
assert not hasattr(Position, 'geo')
assert not {'pyproj', 'astropy', 'spiceypy'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
