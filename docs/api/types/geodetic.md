(api-geodetic)=
# `GeodeticPosition`

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`tal.geo.GeodeticPosition` is an `AnalysisObject` subtype for geodetic
latitude, longitude, altitude payloads. It keeps LLA separate from
`tal.spatial.Position` so cartesian vector operations only run on xyz data.

```{contents}
:local:
:depth: 2
```

## Constructor Contract

- Input may be AO-like: `AnalysisObject`, `xarray.Dataset`, or
  `xarray.DataArray`.
- Exactly one numeric data variable is required.
- Declared TAL roles are required.
- Exactly one core dimension of length `3` is required.
- Core labels must be exactly `("lat", "lon", "alt")`.
- Geo metadata is normalized under `tal.ext.geo`.

## Geodetic Methods

```python
GeodeticPosition.from_lla(value, *, opts=None, validate=True)
geodetic.to_ecef(*, opts=None, validate=True)
geodetic.to_enu(origin=None, *, opts=None, validate=True)
geodetic.to_crs(dst, *, validate=True)
geodetic.distance_to(other, *, opts=None, validate=True)
geodetic.initial_bearing_to(other, *, opts=None, validate=True)
geodetic.final_bearing_to(other, *, opts=None, validate=True)
geodetic.param.at(query, *, on=None, opts=None, validate=True)
geodetic.param.resample_to(grid, *, on=None, opts=None, validate=True)
geodetic.param.interp_like(other, *, on=None, opts=None, validate=True)
GeodeticPosition.from_ecef(value, *, opts=None, validate=True)
tal.geo.from_ecef(value, *, opts=None, validate=True)
tal.geo.transform_crs(value, *, dst, validate=True)
ProjectedPosition.from_projected(value, *, crs, validate=True)
projected.to_crs(dst, *, validate=True)
ecef.geo.to_lla(*, opts=None, validate=True)
ecef.geo.to_enu(origin=None, *, opts=None, validate=True)
enu.geo.to_ecef(origin=None, *, opts=None, validate=True)
```

`from_lla(...)` stamps and validates metadata on an already-shaped LLA payload.
`to_ecef(...)` and `from_ecef(...)` require the optional `tal[geo]` dependency
group and use pyproj-backed WGS84 CRS transforms.
`to_enu(...)` and `Position.geo.to_enu(...)` require an explicit local origin.
ENU payloads remain Cartesian `Position` objects with `x`, `y`, `z` labels
where x=east, y=north, and z=up.
`distance_to(...)` and bearing methods return scalar-core `tal.linalg.Array`
objects. Geodetic `param.at(...)`, `param.resample_to(...)`, and
`param.interp_like(...)` preserve `GeodeticPosition` identity and use geodesic
interpolation defaults.
`to_crs(...)` and `tal.geo.transform_crs(...)` are explicit G4 CRS transforms.
Projected destinations return `ProjectedPosition` with `easting`, `northing`,
and optional `height` labels.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/geodetic
   :nosignatures:

   tal.geo.GeodeticOptions
   tal.geo.LocalOrigin
   tal.geo.ENUOptions
   tal.geo.GeodesicOptions
   tal.geo.GeodeticInterpolationOptions
   tal.geo.GeodeticPosition
   tal.geo.ProjectedPosition
   tal.geo.GeodeticPosition.from_lla
   tal.geo.GeodeticPosition.to_ecef
   tal.geo.GeodeticPosition.to_enu
   tal.geo.GeodeticPosition.to_crs
   tal.geo.GeodeticPosition.distance_to
   tal.geo.GeodeticPosition.initial_bearing_to
   tal.geo.GeodeticPosition.final_bearing_to
   tal.geo.temporal.GeodeticParamAccessor.at
   tal.geo.temporal.GeodeticParamAccessor.resample_to
   tal.geo.temporal.GeodeticParamAccessor.interp_like
   tal.geo.GeodeticPosition.from_ecef
   tal.geo.accessor.PositionGeoAccessor.to_lla
   tal.geo.accessor.PositionGeoAccessor.to_enu
   tal.geo.accessor.PositionGeoAccessor.to_ecef
   tal.geo.ProjectedPosition.from_projected
   tal.geo.ProjectedPosition.to_crs
   tal.geo.from_ecef
   tal.geo.transform_crs
```

## See Also

- User guide: {doc}`../../user-guide/geo`
- {doc}`position`
- Frame metadata bridge: {doc}`../frames`
