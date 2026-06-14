(api-astro)=
# Astro

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`tal.astro` provides the foundation types for topocentric astronomy
calculations. Astro A1 does not compute Sun direction yet; it defines the
options, metadata, and output type that later backend slices use.

```{contents}
:local:
:depth: 2
```

## Foundation Types

```python
AstroBackend = Literal["astropy", "spice"]
AstroOptions(backend="astropy", time=None, iers=None)
AstroTimeOptions(scale="utc", source=None)
AstroIERSOptions(auto_download=False, degraded_accuracy="error")
TopocentricDirection(data)
```

`AstroBackend` is the public backend literal accepted by
`AstroOptions.backend`. Astro A1 records `"spice"` as a valid option value, but
SPICE behavior is contracted in a later slice.

`TopocentricDirection` stores a `direction` variable with ENU labels
`east`, `north`, and `up`. It also stores `altitude_deg` and `azimuth_deg`
payload variables derived from the ENU vector when they are not supplied.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/astro
   :nosignatures:

   tal.astro.AstroBackend
   tal.astro.AstroOptions
   tal.astro.AstroTimeOptions
   tal.astro.AstroIERSOptions
   tal.astro.TopocentricDirection
```

## See Also

- User guide: {doc}`../user-guide/astro`
- Geodetic observers: {doc}`types/geodetic`
