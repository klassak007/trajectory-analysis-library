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

Local ENU conversion is explicit and requires an origin. ENU output remains a
Cartesian `Position` with `x`, `y`, `z` labels for east, north, and up:

<!-- example-id: UG-GEO-ENU -->
```python
from tal.geo import ENUOptions, LocalOrigin

origin = LocalOrigin(45.0, -75.0, 100.0)
enu = lla.to_enu(opts=ENUOptions(origin=origin, output_frame="site_enu"))
ecef_again = enu.geo.to_ecef()
```

Geodesic distance and bearing are explicit operations on LLA payloads. They
return scalar-core arrays rather than positions:

<!-- example-id: UG-GEO-DISTANCE -->
```python
from tal.geo import GeodesicOptions

distance = lla.distance_to(lla)
bearing = lla.initial_bearing_to(lla, opts=GeodesicOptions())
```

Geodetic param interpolation uses the typed `param` accessor. The default is
geodesic interpolation for latitude/longitude and linear interpolation for
altitude:

<!-- example-id: UG-GEO-INTERPOLATION -->
```python
from tal.geo import GeodeticInterpolationOptions

interpolated = lla.param.at([0.0], on="sample", opts=GeodeticInterpolationOptions())
nearest = lla.param.resample_to([0.0], on="sample", opts=GeodeticInterpolationOptions(method="nearest"))
matched = lla.param.interp_like(nearest, on="sample", opts=GeodeticInterpolationOptions(method="nearest"))
```

CRS transforms are explicit and select the output type from the destination
CRS. Projected destinations use `ProjectedPosition` instead of
`tal.spatial.Position`:

<!-- example-id: UG-GEO-CRS -->
```python
from tal.geo import ProjectedPosition, transform_crs

projected = lla.to_crs("EPSG:32611")
ecef = transform_crs(lla, dst="EPSG:4978")
roundtrip = projected.to_crs("EPSG:4979")
assert isinstance(projected, ProjectedPosition)
```

## Metadata

Geo semantics live under `tal.ext.geo`. Units stored on xarray variables or
coordinates are inert metadata and do not drive conversion.

## See Also

- API: {doc}`../api/types/geodetic`
- Spatial cartesian positions: {doc}`../api/types/position`
