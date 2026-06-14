(astro-foundation)=
# Astro

TAL's astro layer starts with a small foundation for topocentric direction
results. The first slice defines options, metadata, and the
`TopocentricDirection` output type. Sun direction calculation is added by later
backend slices.

## Foundation Options

<!-- example-id: UG-ASTRO-OPTIONS -->
```python
from tal.astro import AstroOptions, AstroTimeOptions

opts = AstroOptions(time=AstroTimeOptions(scale="utc", source="utc_time"))
backend = opts.backend
time_source = opts.time.source
```

## Topocentric Direction Payloads

<!-- example-id: UG-ASTRO-DIRECTION -->
```python
import numpy as np
import xarray as xr
from tal.astro import TopocentricDirection
from tal.core import AnalysisObject

ao = AnalysisObject.from_data(
    xr.Dataset(
        {"direction": (("sample", "enu"), np.array([[1.0, 0.0, 0.0]]))},
        coords={"sample": [0], "enu": ["east", "north", "up"]},
    ),
    sequence_dim="sample",
    core_dims=("enu",),
    validate=True,
)

direction = TopocentricDirection(ao)
altitude = direction.unsafe_data["altitude_deg"]
azimuth = direction.unsafe_data["azimuth_deg"]
```

The payload is a unit direction, not a `tal.spatial.Position`. Astro metadata
lives under `tal.ext.astro`; geo observer metadata remains under `tal.ext.geo`.

## See Also

- API: {doc}`../api/astro`
- Geo observers: {doc}`geo`
