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
GeodeticPosition.from_ecef(value, *, opts=None, validate=True)
tal.geo.from_ecef(value, *, opts=None, validate=True)
```

`from_lla(...)` stamps and validates metadata on an already-shaped LLA payload.
`to_ecef(...)` and `from_ecef(...)` require the optional `tal[geo]` dependency
group and use pyproj-backed WGS84 CRS transforms.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/geodetic
   :nosignatures:

   tal.geo.GeodeticOptions
   tal.geo.GeodeticPosition
   tal.geo.GeodeticPosition.from_lla
   tal.geo.GeodeticPosition.to_ecef
   tal.geo.GeodeticPosition.from_ecef
   tal.geo.from_ecef
```

## See Also

- User guide: {doc}`../../user-guide/geo`
- {doc}`position`
- Frame metadata bridge: {doc}`../frames`
