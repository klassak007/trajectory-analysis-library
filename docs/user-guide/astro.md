(astro-foundation)=
# Astro

TAL's astro layer provides topocentric direction results and an Astropy-backed
Sun direction operation. The base `tal.astro` import stays lightweight; Sun
calculation lives in `tal.astro.sun`.

## Foundation Options

<!-- example-id: UG-ASTRO-OPTIONS -->
```python
from tal.astro import AstroOptions, AstroTimeOptions

opts = AstroOptions(time=AstroTimeOptions(scale="utc", source="utc_time"))
backend = opts.backend
time_source = opts.time.source
```

## Sun Direction

<!-- example-id: UG-ASTRO-DIRECTION -->
```python
import xarray as xr
from tal.astro import AstroIERSOptions
from tal.astro.sun import SunDirectionOptions, direction_to_sun
from tal.core import AnalysisObject
from tal.geo import GeodeticPosition

ao = AnalysisObject.from_data(
    xr.Dataset(
        {"lla": (("sample", "lla_axis"), [[35.0, -106.0, 1600.0]])},
        coords={"sample": [0], "lla_axis": ["lat", "lon", "alt"]},
    ),
    sequence_dim="sample",
    core_dims=("lla_axis",),
    validate=True,
)

opts = SunDirectionOptions(iers=AstroIERSOptions(auto_download=False, degraded_accuracy="ignore"))
sun = direction_to_sun(
    GeodeticPosition.from_lla(ao),
    time="2024-06-01T12:00:00",
    opts=opts,
)
direction = sun.unsafe_data["direction"]
altitude = sun.unsafe_data["altitude_deg"]
azimuth = sun.unsafe_data["azimuth_deg"]
```

The payload is a unit direction, not a `tal.spatial.Position`. Astro metadata
lives under `tal.ext.astro`; geo observer metadata remains under `tal.ext.geo`.
SPICE options are present as a reserved public shape, but SPICE execution is a
later backend phase.

## See Also

- API: {doc}`../api/astro`
- Geo observers: {doc}`geo`
