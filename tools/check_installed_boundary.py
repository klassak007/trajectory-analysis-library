"""Check Contract 135 installed behavior from a directory outside the checkout.

Run with an isolated interpreter whose dependencies match the named profile.
This tool uses public operations and installed distribution metadata only.
"""

import argparse
import json
import sys
import tempfile
from copy import deepcopy
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import xarray as xr


def _metadata_boundary():
    from tal.core import AnalysisObject, merge_schema

    ds = xr.Dataset(
        {"value": ("sample", [0.0, 10.0])}, coords={"time": ("sample", [0.0, 1.0])}
    )
    source = AnalysisObject.from_data(ds, sequence_dim="sample", param_coord="time")
    namespaces = {
        "geo": {"kind": "unknown_geo", "nested": {"labels": [1, 2]}},
        "astro": {"kind": "unknown_astro", "backend": "future"},
    }
    source = AnalysisObject(merge_schema(source.as_dataset(), {"ext": namespaces}))
    before = source.as_dataset(copy="deep")
    interpolated = source.param.at([0.5])
    assert interpolated.as_dataset().attrs["tal"]["ext"] == namespaces
    assert interpolated.as_dataset().value.item() == 5.0
    with tempfile.TemporaryDirectory() as directory:
        store = str(Path(directory) / "metadata.zarr")
        source.io.to_zarr(store)
        restored = AnalysisObject.from_zarr(store)
        assert restored.as_dataset().attrs["tal"]["ext"] == namespaces
    xr.testing.assert_identical(source.as_dataset(), before)
    assert namespaces == deepcopy(before.attrs["tal"]["ext"])


def _base_boundary(profile):
    import tal
    from tal.spatial import Position

    assert metadata.version("tal") == "0.2.0"
    assert not {"geo", "astro"}.intersection(tal.__all__)
    assert find_spec("tal.geo") is None and find_spec("tal.astro") is None
    assert not hasattr(Position, "geo")
    distribution = metadata.distribution("tal")
    assert not any(str(p).startswith("tal_extensions/") for p in distribution.files)
    requirements = distribution.requires or []
    assert not any(
        r.lower().startswith(("pyproj", "astropy", "spiceypy", "tal-extensions"))
        for r in requirements
    )
    assert not {"geo", "astro", "spice"}.intersection(
        distribution.metadata.get_all("Provides-Extra", [])
    )
    if profile == "tal":
        assert find_spec("tal_extensions") is None
    _metadata_boundary()


def _location():
    from tal_extensions.geo import GeodeticPosition

    from tal.core import AnalysisObject

    ds = xr.Dataset(
        {"position": (("sample", "lla"), [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])},
        coords={"lla": ["lat", "lon", "alt"], "time": ("sample", [0.0, 1.0])},
    )
    return GeodeticPosition.from_lla(
        AnalysisObject.from_data(
            ds, sequence_dim="sample", core_dims=("lla",), param_coord="time"
        )
    )


def _backend_boundary(profile, source):
    if profile == "extensions":
        assert find_spec("pyproj") is None and find_spec("astropy") is None
        try:
            source.to_ecef()
        except ImportError as exc:
            assert "tal-extensions[geo]" in str(exc)
        else:
            raise AssertionError("conversion should require the geo extra")
        return
    from tal_extensions.geo import GeodeticInterpolationOptions, LocalOrigin

    from tal.core.schema_read import read_roles

    ecef = source.to_ecef()
    ds = ecef.as_dataset()
    core_dim = read_roles(ds)[3][0]
    np.testing.assert_allclose(ds.position.sel({core_dim: "x"})[0], 6378137.0)
    for method in ("nearest", "geodesic_linear", "ecef_linear", "local_enu_linear"):
        result = source.param.at(
            [0.5],
            opts=GeodeticInterpolationOptions(
                method=method, local_origin=LocalOrigin(lat=0.0, lon=0.0, alt=0.0)
            ),
        )
        assert result.as_dataset().sizes["sample"] == 1
    if profile == "geo":
        assert find_spec("astropy") is None
        return
    from tal_extensions.astro import AstroIERSOptions
    from tal_extensions.astro.sun import SunDirectionOptions, direction_to_sun

    result = direction_to_sun(
        source,
        time="2024-06-01T12:00:00",
        opts=SunDirectionOptions(
            iers=AstroIERSOptions(auto_download=False, degraded_accuracy="ignore")
        ),
    )
    vector = result.as_dataset().direction
    np.testing.assert_allclose((vector**2).sum("enu"), 1.0, atol=1e-12)


def _extension_boundary(profile):
    import tal_extensions

    assert not any(name.startswith("tal_extensions.") for name in sys.modules)
    import tal_extensions.astro
    import tal_extensions.astro.sun
    import tal_extensions.geo
    from tal_extensions.geo import register_position_accessor

    from tal.spatial import Position

    assert tal_extensions.__name__ == "tal_extensions"
    assert metadata.version("tal-extensions") == "0.1.0"
    assert not hasattr(Position, "geo")
    assert not {"pyproj", "astropy", "spiceypy"}.intersection(sys.modules)
    distribution = metadata.distribution("tal-extensions")
    assert not any(str(p).startswith("tal/") for p in distribution.files)
    assert "spice" not in distribution.metadata.get_all("Provides-Extra", [])
    assert any(
        r.replace(" ", "").startswith("tal==0.2.0") for r in distribution.requires
    )
    source = _location()
    _backend_boundary(profile, source)
    assert not hasattr(Position, "geo")
    register_position_accessor()
    descriptor = Position.geo
    register_position_accessor()
    assert Position.geo is descriptor
    Position.geo = None
    try:
        register_position_accessor()
    except RuntimeError as exc:
        assert str(exc).startswith("geo.register_position_accessor:")
    else:
        raise AssertionError("registration must reject occupied attributes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("tal", "extensions", "geo", "astro"))
    args = parser.parse_args()
    _base_boundary(args.profile)
    if args.profile != "tal":
        _extension_boundary(args.profile)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "python": sys.version.split()[0],
                "result": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
