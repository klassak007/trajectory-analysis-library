(geo-geodetic)=
# Geo

TAL's geo layer represents latitude, longitude, and altitude with
`GeodeticPosition`. LLA payloads are not cartesian vectors, so conversion to
`tal.spatial.Position` is always explicit.

## Minimal Example

<!-- example-id: UG-GEO-LLA -->
```python
import numpy as np
import xarray as xr
from tal.core import AnalysisObject
from tal.geo import GeodeticPosition

ao = AnalysisObject.from_data(
    xr.Dataset(
        {"position": (("sample", "lla"), np.array([[45.0, -75.0, 100.0]]))},
        coords={"sample": [0], "lla": ["lat", "lon", "alt"]},
    ),
    sequence_dim="sample",
    core_dims=("lla",),
    validate=True,
)

lla = GeodeticPosition.from_lla(ao)
geo_block = lla.unsafe_data.attrs["tal"]["ext"]["geo"]
```

The payload uses public LLA label order: `lat`, `lon`, `alt`. ECEF conversion
uses pyproj from the optional `tal[geo]` dependency group:

<!-- example-id: UG-GEO-CONVERSION -->
```python
ecef = lla.to_ecef()
roundtrip = GeodeticPosition.from_ecef(ecef)
```

## Metadata

Geo semantics live under `tal.ext.geo`. Units stored on xarray variables or
coordinates are inert metadata and do not drive conversion.

## See Also

- API: {doc}`../api/types/geodetic`
- Spatial cartesian positions: {doc}`../api/types/position`
