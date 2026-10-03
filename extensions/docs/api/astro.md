(api-astro)=
# Astro

> **Audience:** User API. This page is for application code and normal
> analysis workflows.

`tal_extensions.astro` provides foundation types for topocentric astronomy calculations.
`tal_extensions.astro.sun` adds the Astropy-backed Sun direction operation without making
the base `tal_extensions.astro` import heavy.

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
TopocentricDirection.to_vector3(axis="axis", output_var="direction")
```

`AstroBackend` is the public backend literal accepted by
`AstroOptions.backend`. `"spice"` is accepted as an option value, but
Sun-direction calculation supports only the Astropy backend.

`TopocentricDirection` stores a `direction` variable with ENU labels
`east`, `north`, and `up`. It also stores `altitude_deg` and `azimuth_deg`
payload variables derived from the ENU vector when they are not supplied.
Public construction preserves vector magnitude. Use
`TopocentricDirection.to_vector3()` when a label-safe xyz bridge is needed for
ENU vector math.

## Sun Direction

```python
from tal_extensions.astro.sun import SpiceSunOptions, SunDirectionOptions, direction_to_sun
```

`direction_to_sun(...)` computes a topocentric ENU direction to the Sun from a
geodetic observer and absolute datetime-like observation time. Only the Astropy
backend executes this calculation. Passing `SpiceSunOptions` or selecting
`backend="spice"` raises `ValueError` because SPICE execution is unsupported.

Import the operation from `tal_extensions.astro.sun`.

## Autosummary

```{eval-rst}
.. autosummary::
   :toctree: _generated/astro
   :nosignatures:

   tal_extensions.astro.AstroBackend
   tal_extensions.astro.AstroOptions
   tal_extensions.astro.AstroTimeOptions
   tal_extensions.astro.AstroIERSOptions
   tal_extensions.astro.TopocentricDirection
   tal_extensions.astro.TopocentricDirection.to_vector3
   tal_extensions.astro.sun.SpiceSunOptions
   tal_extensions.astro.sun.SunDirectionOptions
   tal_extensions.astro.sun.direction_to_sun
```

## See Also

- User guide: {doc}`../user-guide/astro`
- Geodetic observers: {doc}`types/geodetic`
